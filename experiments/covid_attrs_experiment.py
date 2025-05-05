# Attribute-based classification experiment script using MONAI (2D CT slices, ImageNet pretrained)
import os
import torch
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from monai.data import Dataset, decollate_batch
from monai.transforms import Compose, LoadImaged, ScaleIntensityd, NormalizeIntensityd, RandGaussianSmoothd, RandFlipd, RandAffined, Resized, LambdaD, Activations, AsDiscrete
from monai.losses import FocalLoss
from monai.networks.nets import DenseNet121
from monai.metrics import ROCAUCMetric
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
import random
import csv

# ========== Load Config ==========
with open("config/experiments/covid_exps.yaml", "r") as f:
    config = yaml.safe_load(f)
    
DEBUG = config["DEBUG"]
metadata_path = config["paths"]["metadata_csv"]
image_col = config["paths"]["image_column"]
label_cols = config["labeling"]["attribute_columns"]
normalize_labels = config["labeling"].get("normalize_labels", True)
training = config["training"]
split = config["split"]
output = config["output"]

# ============== SETUP LOGGING ==============
os.makedirs(output["results_dir"], exist_ok=True)
csv_columns = ["seed", "train_loss", "f1", "acc", "recall", "precision", "auc", "test_f1", "test_acc", "test_auc", "test_recall", "test_precision"]

label_name = label_cols[0] if len(label_cols) == 1 else "_".join(label_cols)
log_path = os.path.join(output["results_dir"], f"experiment_{label_name}.csv")
log_file_exists = os.path.exists(log_path)

log_f = open(log_path, "a", newline="")
writer = csv.DictWriter(log_f, fieldnames=csv_columns)
if not log_file_exists:
    writer.writeheader()

# ========== Load and preprocess data ==========
print("Reading metadata ...")
df = pd.read_csv(metadata_path)
print(f"Running experiment on: {label_cols}")

# Combine label columns if more than one attribute
if len(label_cols) == 1:
    df["label"] = df[label_cols[0]]
else:
    df["label"] = df[label_cols].astype(str).agg("_".join, axis=1)

# Encode labels to integers
if normalize_labels:
    df["label"] = df["label"].astype("category").cat.codes
label_map = dict(enumerate(df["label"].astype("category").cat.categories))
num_classes = len(label_map)

# ========== Helper Functions ==========
def build_monai_data(df_subset):
    return [{"img": row[image_col], "label": int(row["label"])} for _, row in df_subset.iterrows()]

def get_train_transform():
    return Compose([
        # Load the image from file and make sure the channel is first: [C, H, W]
        LoadImaged(keys=["img"], ensure_channel_first=True),
        # Scale pixel intensities to [0, 1] range (from raw intensity range)
        ScaleIntensityd(keys=["img"]),
        LambdaD(keys=["img"], func=lambda x: x.repeat(3, 1, 1)),
        NormalizeIntensityd(keys=["img"],
            subtrahend=torch.tensor([0.485, 0.456, 0.406]),  # ImageNet means (R, G, B)
            divisor=torch.tensor([0.229, 0.224, 0.225]),      # ImageNet stds (R, G, B)
            channel_wise=True
        ),
        # Apply Gaussian smoothing with 50% probability to slightly blur the image
        RandGaussianSmoothd(keys=["img"], prob=0.5),
        # Random horizontal flip (axis=2) with 50% probability to augment left-right symmetry
        RandFlipd(keys=["img"], prob=0.5, spatial_axis=-1),
        # Random mild affine rotation on the 2D plane (max ~5.7°), preserves shape with border padding
        RandAffined(keys=["img"], prob=0.5,
            rotate_range=(0.0, 0.0, 0.1),
            padding_mode="border"
        ),
        # Resize the final image to 256x256 as expected by ImageNet-pretrained models
        Resized(keys=["img"], spatial_size=(256, 256)),
    ])

def get_eval_transform():
    return Compose([
        LoadImaged(keys=["img"], ensure_channel_first=True),
        ScaleIntensityd(keys=["img"]),
        LambdaD(keys=["img"], func=lambda x: x.repeat(3, 1, 1)),
        NormalizeIntensityd(keys=["img"], subtrahend=torch.tensor([0.485, 0.456, 0.406]), divisor=torch.tensor([0.229, 0.224, 0.225]), channel_wise=True),
        Resized(keys=["img"], spatial_size=(256, 256)),
    ])

def train_model(train_df, val_df, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using CUDA: {torch.cuda.is_available()}")
    # use pretrained ImageNet weights
    model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=num_classes, pretrained=True).to(device)

    loss_fn = FocalLoss(to_onehot_y=True, use_softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    auc_metric = ROCAUCMetric(average="macro")
    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=num_classes)])
    post_discrete = AsDiscrete(argmax=True, to_onehot=num_classes)

    train_ds = Dataset(data=build_monai_data(train_df), transform=get_train_transform())
    val_ds = Dataset(data=build_monai_data(val_df), transform=get_eval_transform())

    train_loader = DataLoader(train_ds, batch_size=training["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=training["batch_size"], shuffle=False, num_workers=2)

    best_model_state = None
    best_f1 = -1
    best_auc = -1
    best_metrics = {}

    for epoch in range(training["num_epochs"]):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"[Seed {seed}] Epoch {epoch+1}"):
            x, y = batch["img"].to(device), batch["label"].to(device).long()
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Train Loss={total_loss:.4f}")

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

        # Post-processing directly on the full batch
        y_pred_probs = post_pred(y_pred) # no need to decollate here
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

        if f1 > best_f1 or (f1 == best_f1 and auc > best_auc):
            best_f1 = f1
            best_auc = auc
            best_metrics = {
                "train_loss": total_loss,
                "f1": f1, "acc": acc, "recall": recall,
                "precision": precision, "auc": auc
            }
            if output["save_models"]:
                best_model_state = model.state_dict()

    print(f"Best F1: {best_f1:.4f}, AUC: {best_auc:.4f}")
    return best_metrics, best_model_state

def evaluate_on_test(best_model_state, test_df):
    print("Evaluating on TEST ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=num_classes, pretrained=True).to(device)
    model.load_state_dict(best_model_state)
    model.eval()

    test_ds = Dataset(data=build_monai_data(test_df), transform=get_eval_transform())
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=2)

    post_pred = Activations(softmax=True)
    post_label = AsDiscrete(to_onehot=num_classes)
    post_discrete = AsDiscrete(argmax=True, to_onehot=num_classes)
    auc_metric = ROCAUCMetric(average="macro")

    y_pred = torch.tensor([], dtype=torch.float32, device=device)
    y = torch.tensor([], dtype=torch.long, device=device)
    with torch.no_grad():
        for batch in test_loader:
            x = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = model(x)
            y_pred = torch.cat([y_pred, outputs], dim=0)
            y = torch.cat([y, labels], dim=0)

    # Post-processing directly on the full batch
    y_pred_probs = post_pred(y_pred) # no need to decollate here
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

    print(f"TEST: ACC={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, REC={recall:.4f}, PREC={precision:.4f}")

    return {"test_f1": f1, "test_acc": acc, "test_auc": auc, "test_recall": recall, "test_precision": precision}

# ========== Run Experiments ==========
# DEBUG check (should be placed outside the loop)
if DEBUG:
    print("[DEBUG MODE] Sampling small subset of the data for quick run...")
    df, _ = train_test_split(df, train_size=200, stratify=df["label"], random_state=100)
    df = df.reset_index(drop=True)

for seed in range(training["num_runs"]):
    print(f"\n=== Experiment {seed} ===")
    train_df, temp_df = train_test_split(df, train_size=split["train_size"], stratify=df["label"], random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=split["test_size"] / (split["test_size"] + split["val_size"]), stratify=temp_df["label"], random_state=seed)

    train_metrics, best_model_state = train_model(train_df, val_df, seed)

    if best_model_state is not None:
        test_metrics = evaluate_on_test(best_model_state, test_df)
        # save results
        writer.writerow({"seed": seed, **train_metrics, **test_metrics})
    else:
        # just save train results
        writer.writerow({"seed": seed, **train_metrics})
    log_f.flush()

log_f.close()
print("\nAll runs completed and logged.")
