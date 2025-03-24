import os
import yaml
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from monai.transforms import Compose, Resize, NormalizeIntensity, ToTensor, RandRotate90
from monai.networks.nets import DenseNet121

from datasets.dicom_dataset import DICOM3DDataset

# Set seeds for reproducibility
torch.manual_seed(42)

# Load config
with open("classifier/config.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
columns = config["columns"]
training = config["training"]

# Load metadata and get train and val sets
metadata = pd.read_csv(paths["metadata"])
train_df = metadata[metadata["split"] == "train"]
val_df = metadata[metadata["split"] == "val"]

train_ids = train_df[columns["patient_id"]].tolist()
val_ids = val_df[columns["patient_id"]].tolist()

# dictionary with patient id and diagnosis together
labels_dict = dict(zip(metadata[columns["patient_id"]], metadata[columns["diagnosis"]]))

# Define transforms
transforms = Compose([
    Resize(spatial_size=(64, 128, 128)),
    NormalizeIntensity(nonzero=True, channel_wise=True),
    RandRotate90(prob=0.5, spatial_axes=[0, 2]),
    ToTensor()
])

# Datasets and loaders
train_ds = DICOM3DDataset(train_ids, labels_dict, os.path.join(paths["data_split"], "train"), transform=transforms)
val_ds = DICOM3DDataset(val_ids, labels_dict, os.path.join(paths["data_split"], "val"), transform=transforms)

train_loader = DataLoader(train_ds, batch_size=training["batch_size"], shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=training["batch_size"], shuffle=False, num_workers=2)

# Defining device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (NVIDIA GPU)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Model, loss, optimizer
model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device) # 3 bc 3D, 1 in bc greyscale, 2 out bc output is 0 or 1
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])

# Training loop
for epoch in range(training["num_epochs"]):
    model.train()
    total_loss = 0
    correct, total = 0, 0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{training['num_epochs']}"):
        x, y = x.to(device), y.to(device).long()
        preds = model(x)

        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        predicted = preds.argmax(dim=1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / len(train_loader)
    train_acc = correct / total
    print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f} | Train Accuracy: {train_acc:.4f}")

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device).long()
            preds = model(x)
            predicted = preds.argmax(dim=1)
            val_correct += (predicted == y).sum().item()
            val_total += y.size(0)
    val_acc = val_correct / val_total
    print(f"[Epoch {epoch+1}] Val Accuracy: {val_acc:.4f}")

# Print model summary
print(model)

# Save model
torch.save(model.state_dict(), paths["model_output"])
print(f"Model saved to: {paths['model_output']}")

