import os
import yaml
import itertools
import torch
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
from torch.utils.data import DataLoader
from monai.metrics import ROCAUCMetric
from monai.data import Dataset, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, Resized, ScaleIntensityd,
    RandRotate90d, Activations, AsDiscrete, LambdaD
)
from monai.networks.nets import DenseNet121
from sklearn.metrics import f1_score, roc_auc_score

# Load config
with open("config/classifier/classifier_tune.yaml", "r") as f:
    base_config = yaml.safe_load(f)

paths = base_config["paths"]
columns = base_config["columns"]
param_grid = base_config["param_grid"]
out_model_path = paths["best_model_output"]
out_config_path = paths["best_config_output"]

# Prepare metadata
metadata = pd.read_csv(paths["metadata"])
labels_dict = dict(zip(metadata[columns["patient_id"]], metadata[columns["diagnosis"]]))
train_df = metadata[metadata["split"] == "train"]
val_df = metadata[metadata["split"] == "val"]
print("Metadata loaded")

train_data_base = [
    {"img": os.path.join(paths["nifti_root"], "train", f"{pid}.nii.gz"), "label": int(labels_dict[pid])}
    for pid in train_df[columns["patient_id"]]
]
val_data_base = [
    {"img": os.path.join(paths["nifti_root"], "val", f"{pid}.nii.gz"), "label": int(labels_dict[pid])}
    for pid in val_df[columns["patient_id"]]
]

# Predefine CSV columns
max_epochs = max(param_grid["num_epochs"])
static_fields = ["run_id", "resize", "batch_size", "learning_rate", "rotation_prob", "num_workers", "num_epochs"]
epoch_metrics = ["train_loss", "train_acc", "train_f1", "train_auc", "val_acc", "val_f1", "val_auc"]
csv_fieldnames = static_fields + [f"epoch_{e}_{m}" for e in range(max_epochs) for m in epoch_metrics]
log_csv = paths["csv_log_output"]

# Best overall result
best_result = {"f1": -1, "auc": -1, "acc": -1, "config": None, "state_dict": None, "epoch": -1}

def get_transforms(resize_size, rotate_prob):
    train_tf = Compose([
        LoadImaged(keys=["img"], ensure_channel_first=True),
        LambdaD(keys=["img"], func=lambda x: x.permute(0, 3, 1, 2)),
        ScaleIntensityd(keys=["img"]),
        Resized(keys=["img"], spatial_size=tuple(resize_size)),
        RandRotate90d(keys=["img"], prob=rotate_prob, spatial_axes=[0, 2]),
    ])
    val_tf = Compose([
        LoadImaged(keys=["img"], ensure_channel_first=True),
        LambdaD(keys=["img"], func=lambda x: x.permute(0, 3, 1, 2)),
        ScaleIntensityd(keys=["img"]),
        Resized(keys=["img"], spatial_size=tuple(resize_size)),
    ])
    return train_tf, val_tf

# Tuning loop
run_id = 0
for resize, rotate, bs, lr, nw, epochs in itertools.product(
    param_grid["resize"], param_grid["rotation_prob"], param_grid["batch_size"],
    param_grid["learning_rate"], param_grid["num_workers"], param_grid["num_epochs"]
):
    run_id += 1
    print(f"\n===== Run {run_id}: resize={resize}, rotate={rotate}, batch_size={bs}, lr={lr}, num_workers={nw}, epochs={epochs} =====")

    run_log = {k: None for k in csv_fieldnames}
    run_log.update({
        "run_id": run_id,
        "resize": resize,
        "batch_size": bs,
        "learning_rate": lr,
        "rotation_prob": rotate,
        "num_workers": nw,
        "num_epochs": epochs,
    })

    train_tf, val_tf = get_transforms(resize, rotate)

    train_ds = Dataset(data=train_data_base, transform=train_tf)
    val_ds = Dataset(data=val_data_base, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=bs, num_workers=nw, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = torch.nn.CrossEntropyLoss()
    auc_metric = ROCAUCMetric()

    best_f1 = -1
    best_auc = -1
    best_acc = -1
    best_epoch = -1
    best_state_dict = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        all_train_preds = []
        all_train_labels = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs = batch["img"].to(device)
            labels = batch["label"].to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        train_acc = correct / total
        train_loss = epoch_loss / len(train_loader)
        train_f1 = f1_score(all_train_labels, all_train_preds)
        train_auc = roc_auc_score(all_train_labels, all_train_preds) if len(set(all_train_labels)) == 2 else 0.0

        # Validation
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

        acc = (y_pred.argmax(dim=1) == y).sum().item() / len(y)
        y_np = y.cpu().numpy()
        y_pred_np = y_pred.argmax(dim=1).cpu().numpy()
        f1 = f1_score(y_np, y_pred_np)
        y_pred_act = Activations(softmax=True)(y_pred)
        y_onehot = [AsDiscrete(to_onehot=2)(i) for i in decollate_batch(y, detach=False)]
        auc_metric(y_pred_act, y_onehot)
        auc = auc_metric.aggregate().item()
        auc_metric.reset()

        # Save results of epoch for csv log
        run_log[f"epoch_{epoch}_train_loss"] = train_loss
        run_log[f"epoch_{epoch}_train_acc"] = train_acc
        run_log[f"epoch_{epoch}_train_f1"] = train_f1
        run_log[f"epoch_{epoch}_train_auc"] = train_auc
        run_log[f"epoch_{epoch}_val_acc"] = acc
        run_log[f"epoch_{epoch}_val_f1"] = f1
        run_log[f"epoch_{epoch}_val_auc"] = auc

        if f1 > best_f1 or (f1 == best_f1 and auc > best_auc):
            best_f1 = f1
            best_auc = auc
            best_acc = acc
            best_epoch = epoch
            best_state_dict = model.state_dict()

    if best_f1 > best_result["f1"] or (best_f1 == best_result["f1"] and best_auc > best_result["auc"]):
        best_result.update({
            "f1": best_f1,
            "acc": best_acc,
            "auc": best_auc,
            "epoch": best_epoch,
            "config": {
                "resize": resize,
                "rotation_prob": rotate,
                "batch_size": bs,
                "learning_rate": lr,
                "num_workers": nw,
                "num_epochs": epochs,
                "best_epoch": best_epoch,
                "val_f1": best_f1,
                "val_accuracy": best_acc,
                "val_auc": best_auc
            },
            "state_dict": best_state_dict
        })

        # Save best model and config incrementally
        os.makedirs(os.path.dirname(out_model_path), exist_ok=True)
        torch.save(best_state_dict, out_model_path)

        os.makedirs(os.path.dirname(out_config_path), exist_ok=True)
        with open(out_config_path, "w") as f:
            yaml.dump(best_result["config"], f)

        print(f"New best model saved: Epoch {best_epoch} | F1: {best_f1:.4f} | AUC: {best_auc:.4f} | ACC: {best_acc:.4f}")

    # Save run results to csv
    os.makedirs(os.path.dirname(log_csv), exist_ok=True)
    file_exists = os.path.isfile(log_csv)
    with open(log_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(run_log)

print("Tuning complete.")
print(f"Best model found: Epoch {best_result['epoch']} | F1: {best_result['f1']:.4f} | AUC: {best_result['auc']:.4f} | ACC: {best_result['acc']:.4f}")
print(f"Model saved to: {out_model_path}")
print(f"Config saved to: {out_config_path}")