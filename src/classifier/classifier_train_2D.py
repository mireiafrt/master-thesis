# MONAI training script for 2D classification with pre-trained ImageNet weights using defined splits
import os
import yaml
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from monai.metrics import ROCAUCMetric
from monai.data import Dataset, decollate_batch
from monai.transforms import Compose, LoadImaged, ScaleIntensityd, NormalizeIntensityd, RandGaussianSmoothd, RandFlipd, RandAffined, Resized, LambdaD, Activations, AsDiscrete
from monai.networks.nets import DenseNet121
from monai.losses import FocalLoss
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import numpy as np

# Set global seed
torch.manual_seed(42)

# Load config
with open("config/classifier/classifier_train_covid.yaml", "r") as f:
    config = yaml.safe_load(f)

DEBUG = config['DEBUG']
paths = config["paths"]
columns = config["columns"]
training = config["training"]

# Load metadata CSV with pre-defined splits
print("Reading metadata ...")
metadata = pd.read_csv(paths["metadata_csv"])
metadata = metadata[metadata["use"] == True]

# Split according to 'split' column
train_df = metadata[metadata["split"] == "train"]
val_df = metadata[metadata["split"] == "val"]

if DEBUG:
    print("[DEBUG MODE] Sampling small subset of the data for quick run...")
    train_df, _ = train_test_split(train_df, train_size=200, stratify=train_df["label"], random_state=100)
    train_df = train_df.reset_index(drop=True)
    val_df, _ = train_test_split(val_df, train_size=50, stratify=val_df["label"], random_state=100)
    val_df = val_df.reset_index(drop=True)

# Data dicts
train_data = [{"img": row[columns["image_path"]], "label": int(row["label"])} for _, row in train_df.iterrows()]
val_data = [{"img": row[columns["image_path"]], "label": int(row["label"])} for _, row in val_df.iterrows()]

# Transforms
train_transforms = Compose([
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
post_pred = Compose([Activations(softmax=True)])
post_label = Compose([AsDiscrete(to_onehot=2)])
post_discrete = Compose([AsDiscrete(argmax=True, to_onehot=2)])

# Datasets and loaders
train_ds = Dataset(data=train_data, transform=train_transforms)
val_ds = Dataset(data=val_data, transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=training["batch_size"], shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=training["batch_size"], shuffle=False, num_workers=2)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Model, loss, ROCAUC
model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=2, pretrained=True).to(device)
loss_fn = FocalLoss(to_onehot_y=True, use_softmax=True, gamma=2.0)
optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
auc_metric = ROCAUCMetric()

# Training
best_f1 = -1
best_auc = -1
best_epoch = -1
val_interval = 1

# early stopping criteria
early_stop_patience = training.get("early_stopping_patience", 5)
epochs_since_improvement = 0

for epoch in range(training["num_epochs"]):
    print("-" * 30)
    print(f"Epoch {epoch + 1}/{training['num_epochs']}")
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

    # Validation
    if (epoch + 1) % val_interval == 0:
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

        # Save best model based on F1 and break ties with AUC
        if f1 > best_f1 or (f1 == best_f1 and auc > best_auc):
            best_f1 = f1
            best_auc = auc
            best_epoch = epoch + 1
            os.makedirs(os.path.dirname(paths["model_output"]), exist_ok=True)
            torch.save(model.state_dict(), paths["model_output"])
            print("Saved new best model")
            epochs_since_improvement = 0  # reset
        else:
            epochs_since_improvement += 1
        
    # check if should stop training due to early stopping criteria
    if epochs_since_improvement >= early_stop_patience:
        print(f"Early stopping at epoch {epoch+1} due to no improvement.")
        break

print(f"Training complete. Best F1: {best_f1:.4f} and AUC: {best_auc:.4f} at Epoch {best_epoch}")