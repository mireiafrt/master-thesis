# MONAI training script for 2D classification using CSV-defined splits
import os
import yaml
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from monai.metrics import ROCAUCMetric
from monai.data import Dataset, decollate_batch
from monai.transforms import Compose, LoadImaged, ScaleIntensityd, RandRotateD, RandFlipd, RandGaussianSmoothd, Resized, Activations, AsDiscrete, LambdaD
from monai.networks.nets import DenseNet121
from monai.losses import FocalLoss
import numpy as np

# Set global seed
torch.manual_seed(42)

# Load config
with open("config/classifier/classifier_train_covid.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
columns = config["columns"]
training = config["training"]

# TensorBoard writer
writer = SummaryWriter(log_dir="runs/classifier_2d")

# Load metadata CSV with pre-defined splits
metadata = pd.read_csv(paths["metadata"])
metadata = metadata[metadata["use"] == True]

# Split according to 'split' column
train_df = metadata[metadata["split"] == "train"]
val_df = metadata[metadata["split"] == "val"]

# Data dicts
train_data = [{"img": row[columns["image_path"]], "label": int(row["label"])} for _, row in train_df.iterrows()]
val_data = [{"img": row[columns["image_path"]], "label": int(row["label"])} for _, row in val_df.iterrows()]

# Transforms
train_transforms = Compose([
    LoadImaged(keys=["img"], ensure_channel_first=True),
    ScaleIntensityd(keys=["img"]),
    Resized(keys=["img"], spatial_size=(224, 224)),
    RandGaussianSmoothd(keys=["img"], prob=0.2),
    RandFlipd(keys=["img"], spatial_axis=2, prob=0.5),
    RandRotateD(keys=["img"], range_z=0.2, prob=0.3),
])
val_transforms = Compose([
    LoadImaged(keys=["img"], ensure_channel_first=True),
    ScaleIntensityd(keys=["img"]),
    Resized(keys=["img"], spatial_size=(224, 224)),
])
# Post transforms
post_pred = Compose([Activations(softmax=True)])
post_label = Compose([AsDiscrete(to_onehot=2)])

# Datasets
train_ds = Dataset(data=train_data, transform=train_transforms)
val_ds = Dataset(data=val_data, transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=training["batch_size"], shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=training["batch_size"], shuffle=False, num_workers=2)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Model and loss
model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=2).to(device)
loss_fn = FocalLoss(to_onehot_y=True, gamma=2.0)
optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
auc_metric = ROCAUCMetric()

# Training
best_metric = -1
best_metric_epoch = -1
val_interval = 1

for epoch in range(training["num_epochs"]):
    print("-" * 30)
    print(f"Epoch {epoch + 1}/{training['num_epochs']}")
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        inputs = batch["img"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        writer.add_scalar("Loss/train_step", loss.item(), epoch * len(train_loader) + step)

    avg_loss = epoch_loss / len(train_loader)
    train_acc = correct / total
    writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")

    # Validation
    if (epoch + 1) % val_interval == 0:
        model.eval()
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["img"].to(device)
                labels = batch["label"].to(device)
                outputs = model(inputs)
                y_pred = torch.cat([y_pred, outputs], dim=0)
                y = torch.cat([y, labels], dim=0)

        acc = (y_pred.argmax(dim=1) == y).sum().item() / len(y)
        y_pred_act = post_pred(y_pred)
        y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
        auc_metric(y_pred_act, y_onehot)
        auc = auc_metric.aggregate().item()
        auc_metric.reset()

        writer.add_scalar("Accuracy/val", acc, epoch)
        writer.add_scalar("AUC/val", auc, epoch)
        scheduler.step(avg_loss)

        print(f"[Epoch {epoch+1}] Val Acc: {acc:.4f} | AUC: {auc:.4f}")

        if acc > best_metric:
            best_metric = acc
            best_metric_epoch = epoch + 1
            os.makedirs(os.path.dirname(paths["model_output"]), exist_ok=True)
            torch.save(model.state_dict(), paths["model_output"])
            print("Saved new best model")

print(f"Training complete. Best Accuracy: {best_metric:.4f} at Epoch {best_metric_epoch}")
writer.close()