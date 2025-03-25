import os
import yaml
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from monai.metrics import ROCAUCMetric
from monai.data import Dataset, decollate_batch
from monai.transforms import Compose, LoadImaged, Resized, ScaleIntensityd, RandRotate90d, ToTensord, Activations, AsDiscrete, LambdaD
from monai.networks.nets import DenseNet121


# Set seeds for reproducibility
torch.manual_seed(42)

# Load config
with open("config/classifier/classifier_train.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
columns = config["columns"]
training = config["training"]

# Init writer
writer = SummaryWriter(log_dir="runs/classifier_nifti")

# Load metadata and get train and val sets
metadata = pd.read_csv(paths["metadata"])
train_df = metadata[metadata["split"] == "train"]
val_df = metadata[metadata["split"] == "val"]

labels_dict = dict(zip(metadata[columns["patient_id"]], metadata[columns["diagnosis"]]))

# Build MONAI-friendly data dicts
train_data = [
    {"img": os.path.join(paths["nifti_root"], "train", f"{pid}.nii.gz"), "label": int(labels_dict[pid])}
    for pid in train_df[columns["patient_id"]]
]
val_data = [
    {"img": os.path.join(paths["nifti_root"], "val", f"{pid}.nii.gz"), "label": int(labels_dict[pid])}
    for pid in val_df[columns["patient_id"]]
]

# Define transformations for the training dataset.
train_transforms = Compose([
    LoadImaged(keys=["img"], ensure_channel_first=True),  # Load images and ensure channel dimension is first.
    LambdaD(keys=["img"], func=lambda x: x.permute(0, 3, 1, 2)),  # → [1, D, H, W]
    ScaleIntensityd(keys=["img"]),  # Normalize the intensity of the images.
    Resized(keys=["img"], spatial_size=(96, 96, 96)),  # Resize images to 96x96x96 for consistent input size.
    RandRotate90d(keys=["img"], prob=0.8, spatial_axes=[0, 2]),  # Randomly rotate images on specified axes with 80% probability.
])

# Define transformations for the validation dataset (no random rotations to maintain original orientation).
val_transforms = Compose([
    LoadImaged(keys=["img"], ensure_channel_first=True),  # Load images ensuring the channel dimension is first.
    LambdaD(keys=["img"], func=lambda x: x.permute(0, 3, 1, 2)),  # → [1, D, H, W]
    ScaleIntensityd(keys=["img"]),  # Normalize image intensities.
    Resized(keys=["img"], spatial_size=(96, 96, 96)),  # Resize images to 96x96x96.
])

# Post-processing for predictions using softmax activation to convert logits to probabilities.
post_pred = Compose([Activations(softmax=True)])
# Post-processing for labels to convert them to one-hot encoded format.
post_label = Compose([AsDiscrete(to_onehot=2)])

# Check device availability for CUDA-based operations
pin_memory = torch.cuda.is_available()

# Create MONAI Datasets and loaders
train_ds = Dataset(data=train_data, transform=train_transforms)
val_ds = Dataset(data=val_data, transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=training["batch_size"], shuffle=True, num_workers=2, pin_memory=pin_memory)
val_loader = DataLoader(val_ds, batch_size=training["batch_size"], num_workers=2, pin_memory=pin_memory)

# Device setup
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
model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])

# Metrics
auc_metric = ROCAUCMetric()

val_interval = 1
best_metric = -1
best_metric_epoch = -1

# Training loop
for epoch in range(training["num_epochs"]):
    print("-" * 30)
    print(f"Epoch {epoch + 1}/{training['num_epochs']}")
    model.train()
    epoch_loss = 0
    step = 0
    correct, total = 0, 0

    for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        step += 1
        inputs = batch_data["img"].to(device)
        labels = batch_data["label"].to(device).long()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        writer.add_scalar("Loss/train_step", loss.item(), epoch * len(train_loader) + step)

    avg_loss = epoch_loss / step
    train_acc = correct / total if total > 0 else 0
    print(f"[Epoch {epoch+1}] Avg Train Loss: {avg_loss:.4f} | Train Accuracy: {train_acc:.4f}")

    writer.add_scalar("Loss/train_epoch", avg_loss, epoch + 1)
    writer.add_scalar("Accuracy/train", train_acc, epoch + 1)

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

        acc = (y_pred.argmax(dim=1) == y).sum().item() / len(y)

        # Post-processing directly on the full batch
        y_pred_act = post_pred(y_pred) # no need to decollate here
        y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]

        # AUC Metric
        auc_metric(y_pred_act, y_onehot)
        auc = auc_metric.aggregate().item()
        auc_metric.reset()

        print(f"[Epoch {epoch+1}] Val Accuracy: {acc:.4f} | AUC: {auc:.4f}")

        writer.add_scalar("Accuracy/val", acc, epoch+1)
        writer.add_scalar("AUC/val", auc, epoch+1)

        if acc > best_metric:
            best_metric = acc
            best_metric_epoch = epoch + 1
            # Ensure output directory exists and save the model
            os.makedirs(os.path.dirname(paths["model_output"]), exist_ok=True)
            torch.save(model.state_dict(), paths["model_output"])
            print("Saved new best model")

print(f"Training completed. Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")

writer.close()
