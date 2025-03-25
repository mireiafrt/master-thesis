import os
import torch
import yaml
import pandas as pd
from tqdm import tqdm
from monai.transforms import Compose, LoadImaged, ScaleIntensityd, Resized, LambdaD, Activations, AsDiscrete
from monai.data import Dataset, DataLoader, decollate_batch
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121

# Set seeds for reproducibility
torch.manual_seed(42)

# Load config
with open("config/classifier/classifier_eval.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config["paths"]

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


# Load metadata + test data
metadata = pd.read_csv(paths['metadata'])
test_df = metadata[metadata["split"] == "test"].copy()

test_data = [
    {"img": os.path.join(paths['nifti_test'], f"{pid}.nii.gz"), "label": int(label), "Patient ID": pid}
    for pid, label in zip(test_df["Patient ID"], test_df["label"])
]

# Transforms (no augmentation)
transforms = Compose([
    LoadImaged(keys=["img"], ensure_channel_first=True),
    LambdaD(keys=["img"], func=lambda x: x.permute(0, 3, 1, 2)),  # [1, D, H, W]
    ScaleIntensityd(keys=["img"]),
    Resized(keys=["img"], spatial_size=(96, 96, 96)),
])
post_pred = Compose([Activations(softmax=True)])
post_label = Compose([AsDiscrete(to_onehot=2)])

# Dataset and DataLoader
test_ds = Dataset(data=test_data, transform=transforms)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

# Load trained model and set to evaluation mode
model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
model.load_state_dict(torch.load(paths['model'], map_location=device))
model.eval()

# Evaluation
auc_metric = ROCAUCMetric()
all_preds = []
all_true = []
all_probs = []
all_pids = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        inputs = batch["img"].to(device)
        labels = batch["label"].to(device)
        outputs = model(inputs)

        preds = torch.argmax(outputs, dim=1).cpu().item()
        prob_class_1 = torch.softmax(outputs, dim=1)[0, 1].cpu().item()

        all_preds.append(preds)
        all_true.append(labels.cpu().item())
        all_probs.append(prob_class_1)
        all_pids.append(batch["Patient ID"][0])

# Save predictions merged with metadata
test_df["predicted_label"] = all_preds
test_df["predicted_prob_class1"] = all_probs

# Save to CSV
test_df.to_csv(paths['pred_output'], index=False)
print(f"Predictions saved to {paths['pred_output']}")

# Compute overall accuracy and AUC
correct = sum([p == t for p, t in zip(all_preds, all_true)])
acc = correct / len(all_true)

y_pred_tensor = torch.tensor([[1 - p, p] for p in all_probs])  # shape [N, 2]
y_true_tensor = torch.tensor(all_true)

y_onehot = [post_label(i) for i in decollate_batch(y_true_tensor)]
y_pred_act = [post_pred(i) for i in decollate_batch(y_pred_tensor)]

auc_metric(y_pred_act, y_onehot)
auc = auc_metric.aggregate().item()

print(f"Test Accuracy: {acc:.4f}")
print(f"Test AUC: {auc:.4f}")
