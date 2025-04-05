import os
import yaml
import json
import itertools
import torch
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from monai.metrics import ROCAUCMetric
from monai.data import Dataset, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, Resized, ScaleIntensityd,
    RandRotate90d, ToTensord, Activations, AsDiscrete, LambdaD
)
from monai.networks.nets import DenseNet121

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

# Define tuning grid
resize_opts = param_grid["resize"]
rotate_probs = param_grid["rotation_prob"]
batch_sizes = param_grid["batch_size"]
learning_rates = param_grid["learning_rate"]
num_workers_opts = param_grid["num_workers"]
num_epochs_opts = param_grid["num_epochs"]

# Track best result
best_result = {"acc": -1, "auc": -1, "config": None, "state_dict": None, "epoch": -1}

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
    resize_opts, rotate_probs, batch_sizes, learning_rates, num_workers_opts, num_epochs_opts
):
    run_id += 1
    print(f"\n===== Run {run_id}: resize={resize}, rotate={rotate}, batch_size={bs}, lr={lr}, num_workers={nw}, epochs={epochs} =====")

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

    best_acc = -1
    best_auc = -1
    best_epoch = -1
    best_state_dict = None

    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs = batch["img"].to(device)
            labels = batch["label"].to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

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
        y_pred_act = Activations(softmax=True)(y_pred)
        y_onehot = [AsDiscrete(to_onehot=2)(i) for i in decollate_batch(y, detach=False)]
        auc_metric(y_pred_act, y_onehot)
        auc = auc_metric.aggregate().item()
        auc_metric.reset()

        if acc > best_acc or (acc == best_acc and auc > best_auc):
            best_acc = acc
            best_auc = auc
            best_epoch = epoch + 1
            best_state_dict = model.state_dict()

    # Save if best overall
    if best_acc > best_result["acc"] or (best_acc == best_result["acc"] and best_auc > best_result["auc"]):
        best_result.update({
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
                "val_accuracy": best_acc,
                "val_auc": best_auc
            },
            "state_dict": best_state_dict
        })
        print(f"New best model found! Accuracy: {best_acc:.4f}, AUC: {best_auc:.4f}, Epoch: {best_epoch}")

# Final save of best model and config
os.makedirs(os.path.dirname(out_model_path), exist_ok=True)
torch.save(best_result["state_dict"], out_model_path)

os.makedirs(os.path.dirname(out_config_path), exist_ok=True)
with open(out_config_path, "w") as f:
    yaml.dump(best_result["config"], f)
