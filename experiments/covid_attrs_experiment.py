# Attribute-based classification experiment script using MONAI (2D CT slices, ImageNet pretrained)
import os
import torch
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score, precision_score
from monai.data import Dataset, decollate_batch
from monai.transforms import Compose, LoadImaged, ScaleIntensityd, NormalizeIntensityd, RandGaussianSmoothd, RandFlipd, RandAffined, Resized, LambdaD
from monai.losses import FocalLoss
from monai.networks.nets import DenseNet121
from torch.utils.data import DataLoader
import random
import csv

# ========== Load Config ==========
with open("config/experiments/covid_exps.yaml", "r") as f:
    config = yaml.safe_load(f)

metadata_path = config["paths"]["metadata_csv"]
image_col = config["paths"]["image_column"]
label_cols = config["labeling"]["attribute_columns"]
normalize_labels = config["labeling"].get("normalize_labels", True)
training = config["training"]
split = config["split"]
output = config["output"]

os.makedirs(output["results_dir"], exist_ok=True)
label_name = label_cols[0] if len(label_cols) == 1 else "_".join(label_cols)
log_path = os.path.join(output["results_dir"], f"experiment_{label_name}.csv")

# ========== Load and preprocess data ==========
print("Reading metadata ...")
df = pd.read_csv(metadata_path)

# Combine label columns if more than one attribute
if len(label_cols) == 1:
    df["label"] = df[label_cols[0]]
else:
    df["label"] = df[label_cols].astype(str).agg("_".join, axis=1)

# Encode labels to integers
if normalize_labels:
    df["label"] = df["label"].astype("category").cat.codes
label_map = dict(enumerate(df["label"].astype("category").cat.categories))

# ========== Helper Functions ==========
def build_monai_data(df_subset):
    return [{"img": row[image_col], "label": int(row["label"])} for _, row in df_subset.iterrows()]

def get_train_transform():
    return Compose([
        # Load the image from file and make sure the channel is first: [C, H, W]
        LoadImaged(keys=["img"], ensure_channel_first=True),
        # Scale pixel intensities to [0, 1] range (from raw intensity range)
        ScaleIntensityd(keys=["img"]),
        # Convert single-channel grayscale image [1, H, W] to 3-channel by duplicating it
        LambdaD(keys=["img"], func=lambda x: np.repeat(x, 3, axis=0)),
        # Normalize using ImageNet mean and std for each channel
        NormalizeIntensityd(keys=["img"],
            subtrahend=torch.tensor([0.485, 0.456, 0.406]),  # ImageNet means (R, G, B)
            divisor=torch.tensor([0.229, 0.224, 0.225])      # ImageNet stds (R, G, B)
        ),
        # Apply Gaussian smoothing with 50% probability to slightly blur the image
        RandGaussianSmoothd(keys=["img"], prob=0.5),
        # Random horizontal flip (axis=2) with 50% probability to augment left-right symmetry
        RandFlipd(keys=["img"], prob=0.5, spatial_axis=2),
        # Random mild affine rotation on the 2D plane (max ~5.7°), preserves shape with border padding
        RandAffined(keys=["img"], prob=0.5,
            rotate_range=(0.0, 0.0, 0.1),  # Only rotate around z-axis (axial plane)
            padding_mode="border"         # Fill empty pixels after rotation with border values
        ),
        # Resize the final image to 256x256 as expected by ImageNet-pretrained models
        Resized(keys=["img"], spatial_size=(256, 256)),
    ])

def get_eval_transform():
    return Compose([
        LoadImaged(keys=["img"], ensure_channel_first=True),
        ScaleIntensityd(keys=["img"]),
        LambdaD(keys=["img"], func=lambda x: np.repeat(x, 3, axis=0)),
        NormalizeIntensityd(keys=["img"], subtrahend=torch.tensor([0.485, 0.456, 0.406]), divisor=torch.tensor([0.229, 0.224, 0.225])),
        Resized(keys=["img"], spatial_size=(256, 256)),
    ])

def train_model(train_df, val_df, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using CUDA: {torch.cuda.is_available()}")
    # use pretrained ImageNet weights
    model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=len(label_map), pretrained=True).to(device)

    loss_fn = FocalLoss(to_onehot_y=True, use_softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

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

        # Evaluate (validation)
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch["img"].to(device), batch["label"].to(device).long()
                outputs = model(x)
                y_pred.extend(outputs.argmax(dim=1).cpu().numpy())
                y_true.extend(y.cpu().numpy())

        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        auc = roc_auc_score(pd.get_dummies(y_true), pd.get_dummies(y_pred), average='macro') if len(label_map) == 2 else 0.0
        print(f"Epoch {epoch+1}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, REC={recall:.4f}, PREC={precision:.4f}")

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

def evaluate_on_test(model, test_df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_ds = Dataset(data=build_monai_data(test_df), transform=get_eval_transform())
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=2)

    y_pred, y_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["img"].to(device)
            labels = batch["label"].to(device).long()
            outputs = model(inputs)
            y_pred.extend(outputs.argmax(dim=1).cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = np.mean(np.array(y_pred) == np.array(y_true))
    auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) == 2 else 0.0
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    print(f"TEST Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    return {"test_f1": f1, "test_acc": acc, "test_auc": auc, "test_recall": recall, "test_precision": precision}

# ========== Run Experiments ==========
with open(log_path, "w", newline="") as f:
    csv_col_names = ["seed", "train_loss", "f1", "acc", "recall", "precision", "auc", "test_f1", "test_acc", "test_auc", "test_recall", "test_precision"]
    writer = writer = csv.DictWriter(f, fieldnames=csv_col_names)
    writer.writeheader()

    for seed in range(training["num_runs"]):
        print(f"\n=== Experiment {seed} ===")
        train_df, temp_df = train_test_split(df, train_size=split["train_size"], stratify=df["label"], random_state=seed)
        val_df, test_df = train_test_split(temp_df, test_size=split["test_size"] / (split["test_size"] + split["val_size"]), stratify=temp_df["label"], random_state=seed)

        train_metrics, model = train_model(train_df, val_df, seed)

        if model is not None:
            test_metrics = evaluate_on_test(model, test_df)
            print(f"Test Metrics: {test_metrics}")
            # save results
            writer.writerow({"seed": seed, **train_metrics, **test_metrics})
        
        else:
            # just save train results
            writer.writerow({"seed": seed, **train_metrics})


print("\nAll runs completed and logged.")
