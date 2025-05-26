# Parameter tuning using Optuna automated hyperparameter optimization framework
import os
import torch
import optuna
import yaml
import csv
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from monai.networks.nets import DenseNet121
from monai.losses import FocalLoss
from monai.metrics import ROCAUCMetric
from monai.data import Dataset, DataLoader, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityd, NormalizeIntensityd, RandGaussianSmoothd,
    RandFlipd, RandAffined, Resized, LambdaD, Activations, AsDiscrete
)

torch.manual_seed(42)

# Load static config data
with open("config/classifier/classifier_tune_covid.yaml", "r") as f:
    base_config = yaml.safe_load(f)
paths = base_config["paths"]
columns = base_config["columns"]
param_grid = base_config["param_grid"] 

# preparing logging
csv_log_path = paths["csv_log_output"]
os.makedirs(os.path.dirname(csv_log_path), exist_ok=True)
csv_fields = list(param_grid.keys()) + ["f1", "accuracy", "recall", "precision", "auc"]
with open(csv_log_path, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
    writer.writeheader()

# Read data
print("Reading metadata ...")
metadata = pd.read_csv(paths["metadata"])
metadata = metadata[metadata["use"] == True]
train_df = metadata[metadata["split"] == "train"]
val_df = metadata[metadata["split"] == "val"]

def get_transforms(flip_prob):
    train_transforms = Compose([
        LoadImaged(keys=["img"], ensure_channel_first=True),
        ScaleIntensityd(keys=["img"]),
        LambdaD(keys=["img"], func=lambda x: x.repeat(3, 1, 1)),
        NormalizeIntensityd(keys=["img"],
            subtrahend=torch.tensor([0.485, 0.456, 0.406]),
            divisor=torch.tensor([0.229, 0.224, 0.225]),
            channel_wise=True
        ),
        RandGaussianSmoothd(keys=["img"], prob=0.5),
        RandFlipd(keys=["img"], prob=flip_prob, spatial_axis=-1),
        RandAffined(keys=["img"], prob=0.5,
            rotate_range=(0.0, 0.0, 0.1),
            padding_mode="border"
        ),
        Resized(keys=["img"], spatial_size=(256, 256)),
    ])
    val_transforms = Compose([
        LoadImaged(keys=["img"], ensure_channel_first=True),
        ScaleIntensityd(keys=["img"]),
        LambdaD(keys=["img"], func=lambda x: x.repeat(3, 1, 1)),
        NormalizeIntensityd(keys=["img"],
            subtrahend=torch.tensor([0.485, 0.456, 0.406]),
            divisor=torch.tensor([0.229, 0.224, 0.225]),
            channel_wise=True
        ),
        Resized(keys=["img"], spatial_size=(256, 256)),
    ])
    return train_transforms, val_transforms

# GLOBAL best model tracker across trials
best_f1_overall = -1  

def objective(trial):
    global best_f1_overall

    batch_size = trial.suggest_categorical("batch_size", param_grid["batch_size"])
    lr_range = [float(x) for x in param_grid["learning_rate"]]
    lr = trial.suggest_float("learning_rate", min(lr_range), max(lr_range), log=True)
    num_epochs = trial.suggest_categorical("num_epochs", param_grid["num_epochs"])
    gamma = trial.suggest_categorical("loss_gamma", param_grid["loss_gamma"])
    alpha = trial.suggest_categorical("loss_alpha", param_grid["loss_alpha"])
    factor = trial.suggest_categorical("LRS_factor", param_grid["LRS_factor"])
    patience = trial.suggest_categorical("LRS_patience", param_grid["LRS_patience"])
    flip_prob = trial.suggest_categorical("flip_prob", param_grid["flip_prob"])

    print("Trial params:")
    print(f"batch_size:{batch_size}, lr:{lr}, num_epochs:{num_epochs}, gamma:{gamma}, alpha:{alpha}, factor:{factor}, patience:{patience}, flip_prob:{flip_prob}")

    train_data = [{"img": row[columns["image_path"]], "label": int(row["label"])} for _, row in train_df.iterrows()]
    val_data = [{"img": row[columns["image_path"]], "label": int(row["label"])} for _, row in val_df.iterrows()]
    train_tf, val_tf = get_transforms(flip_prob)
    train_ds = Dataset(data=train_data, transform=train_tf)
    val_ds = Dataset(data=val_data, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=2, pretrained=True).to(device)
    loss_fn = FocalLoss(to_onehot_y=True, use_softmax=True, gamma=gamma, alpha=alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
    auc_metric = ROCAUCMetric()

    post_pred = Activations(softmax=True)
    post_label = AsDiscrete(to_onehot=2)
    post_discrete = AsDiscrete(argmax=True, to_onehot=2)

    best_f1 = -1
    best_model_state = None
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = batch["img"].to(device), batch["label"].to(device).long()
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Train Loss={total_loss:.4f}")

        model.eval()
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)
        with torch.no_grad():
            for batch in val_loader:
                x = batch["img"].to(device)
                labels = batch["label"].to(device)
                outputs = model(x)
                y_pred = torch.cat([y_pred, outputs], dim=0)
                y = torch.cat([y, labels], dim=0)

        y_pred_probs = post_pred(y_pred)
        y_true = [post_label(i) for i in decollate_batch(y, detach=False)]
        y_pred_bin = [post_discrete(i) for i in decollate_batch(y_pred_probs, detach=False)]
        y_pred_classes = torch.stack([p.argmax(dim=0) for p in y_pred_bin]).cpu().numpy()
        y_true_classes = torch.stack([t.argmax(dim=0) for t in y_true]).cpu().numpy()

        auc_metric.reset()
        auc_metric(y_pred_probs, y_true)
        auc = auc_metric.aggregate().item()
        acc = accuracy_score(y_true_classes, y_pred_classes)
        f1 = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
        recall = recall_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
        precision = precision_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)

        print(f"Epoch {epoch+1} VAL: ACC={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, REC={recall:.4f}, PREC={precision:.4f}")

        scheduler.step(1 - f1)

        if f1 > best_f1:
            best_f1 = f1
            best_model_state = model.state_dict()

        # Check if the trial should be pruned
        trial.report(f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Save best epoch of trial only if it's the best across all trials (or first trial)
    if best_f1 > best_f1_overall:
        best_f1_overall = best_f1
        model_path = paths["best_model_output"]
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(best_model_state, model_path)
        trial.set_user_attr("best_model_path", model_path)
        print("Saving new best model ...")

    # logging info of trial
    trial.set_user_attr("metrics", {"f1": best_f1, "accuracy": acc, "recall": recall, "precision": precision, "auc": auc})

    return best_f1

# Start OPTUNA
pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
study = optuna.create_study(direction="maximize", pruner=pruner)
study.optimize(objective, n_trials=20) # try 20 configurations

# write trials info to csv
with open(csv_log_path, "a", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
    for trial in study.trials:
        row = {**trial.params}
        metrics = trial.user_attrs.get("metrics", {})
        row.update(metrics)
        writer.writerow(row)

print("Best trial:")
print(study.best_trial.params)
print(f"F1: {study.best_trial.value:.4f}")
print("Model path:", study.best_trial.user_attrs.get("best_model_path", "Not saved"))