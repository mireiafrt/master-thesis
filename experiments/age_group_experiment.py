import os
import yaml
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from monai.metrics import ROCAUCMetric
from monai.data import Dataset, decollate_batch
from monai.transforms import Compose, LoadImaged, Resized, ScaleIntensityd, RandRotate90d, ToTensord, Activations, AsDiscrete, LambdaD
from monai.networks.nets import DenseNet121
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import random

# ============== SETUP ==============
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Load config
yaml_path = "config/classifier/classifier_train.yaml"
with open(yaml_path, "r") as f:
    config = yaml.safe_load(f)

# Config
paths = {"metadata":"data/metadata.csv", "nifti_files":"data/preprocessed_nifti"}
columns = {"patient_id": "Patient ID", "diagnosis": "binary_diagnosis_patient", "gender": "Patient Sex", "age_group":"age_group"}
training = {"batch_size": 4, "num_epochs": 40, "learning_rate": 0.0001, "resize": [128, 128, 128], "rotation_prob": 0.5,
            "N_train":15, "N_val":5, "N_test":20, "age_group": "40-60"}

# for 40-60, 60-80 i can use: "N_train":15, "N_val":5, "N_test":20
# if want to evaluate on equal ground also 20-40, idk what to do to include all groups

age_group_to_evaluate = training["age_group"]  # example: "40–60"

# ============== CREATE SPLITS IN MEMORY ==============
metadata = pd.read_csv(paths["metadata"])
metadata["filepath"] = metadata.apply(
    lambda row: os.path.join(paths["nifti_files"], row["split"], f"{row[columns['patient_id']]}.nii.gz"),
    axis=1
)
age_groups = metadata["age_group"].unique()
current_group = metadata[metadata["age_group"] == age_group_to_evaluate].copy()
other_groups = metadata[metadata["age_group"] != age_group_to_evaluate].copy()

N_TRAIN = training["N_train"]
N_VAL = training["N_val"]
N_TEST = training["N_test"]

# --- Stratified test split from current age group only
remaining_current_group, test_set = train_test_split(current_group,
    test_size=N_TEST,
    stratify=current_group[columns["diagnosis"]],
    random_state=42
)

# --- Stratified VAL_A and train_A from current age group only
train_A, val_A = train_test_split(remaining_current_group,
    train_size=N_TRAIN,
    test_size=N_VAL,
    stratify=remaining_current_group[columns["diagnosis"]],
    random_state=1
)

# --- Stratified VAL_B and train_B from each group age all together
train_B_parts, val_B_parts = [], []
# hardcoded sizes so that it matches total n_train and n_val
train_parts = {'Under 20': 2, '20-40': 4, '40-60': 3, '60-80': 3, 'Over 80': 3}
val_parts = {'Under 20': 1, '20-40': 1, '40-60': 1, '60-80': 1, 'Over 80': 1}

for grp in age_groups:
    group_data = metadata[
        (metadata[columns["age_group"]] == grp) &
        (~metadata[columns["patient_id"]].isin(test_set[columns["patient_id"]]))
    ]

    # using hard coded sizes pre-calculated to re-distribute when one group lacks amount of data
    n_train_part = train_parts[grp]
    n_val_part = val_parts[grp]

    # try to do stratified sampling, if error dont do stratified (not enough samples of class to stratify)
    try:
        train_grp, val_grp = train_test_split(group_data,
            train_size=n_train_part,
            test_size=n_val_part,
            stratify=group_data[columns["diagnosis"]],
            random_state=7
        )
    except ValueError:
        train_grp, val_grp = train_test_split(group_data,
            train_size=n_train_part,
            test_size=n_val_part,
            random_state=7
        )

    train_B_parts.append(train_grp)
    val_B_parts.append(val_grp)

train_B = pd.concat(train_B_parts)
val_B = pd.concat(val_B_parts)

print("Set splits completed")

# ============== FUNCTION TO BUILD DATASET ==============
def build_monai_data(df):
    return [{"img": row["filepath"], "label": int(row[columns["diagnosis"]])} for _, row in df.iterrows()]

# ============== MONAI TRANSFORMS ==============
train_transforms = Compose([
    LoadImaged(keys=["img"], ensure_channel_first=True),
    LambdaD(keys=["img"], func=lambda x: x.permute(0, 3, 1, 2)),
    ScaleIntensityd(keys=["img"]),
    Resized(keys=["img"], spatial_size=tuple(training['resize'])),
    RandRotate90d(keys=["img"], prob=training['rotation_prob'], spatial_axes=[0, 2]),
])

val_transforms = Compose([
    LoadImaged(keys=["img"], ensure_channel_first=True),
    LambdaD(keys=["img"], func=lambda x: x.permute(0, 3, 1, 2)),
    ScaleIntensityd(keys=["img"]),
    Resized(keys=["img"], spatial_size=tuple(training['resize'])),
])
post_pred = Compose([Activations(softmax=True)])
post_label = Compose([AsDiscrete(to_onehot=2)])

# ============== EVALUATION FUNCTION ==============
def evaluate_on_test(model_path, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_ds = Dataset(data=build_monai_data(test_set), transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=2, pin_memory=True)

    y_pred, y_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = model(x)
            y_pred.extend(outputs.argmax(dim=1).cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    acc = np.mean(np.array(y_pred) == np.array(y_true))
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) == 2 else 0.0

    print(f"[TEST {model_name}] Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

# ============== TRAINING FUNCTION ==============
def train_and_eval(model_name, train_data, val_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    auc_metric = ROCAUCMetric()

    train_ds = Dataset(data=build_monai_data(train_data), transform=train_transforms)
    val_ds = Dataset(data=build_monai_data(val_data), transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=training["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=training["batch_size"], num_workers=2, pin_memory=True)

    best_f1 = -1
    best_auc = -1
    best_epoch = -1
    best_model_path = f"model_{model_name}_{age_group_to_evaluate}.pt"

    for epoch in range(training["num_epochs"]):
        model.train()
        epoch_loss, correct, total = 0, 0, 0

        for batch_data in tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1} ({age_group_to_evaluate})"):
            inputs = batch_data["img"].to(device)
            labels = batch_data["label"].to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        print(f"[{model_name}] Epoch {epoch+1}: Train Acc={train_acc:.4f}")

        # Validation
        model.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["img"].to(device)
                labels = batch["label"].to(device)
                outputs = model(x)
                y_pred.extend(outputs.argmax(dim=1).cpu().numpy())
                y_true.extend(labels.cpu().numpy())

        acc = np.mean(np.array(y_pred) == np.array(y_true))
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) == 2 else 0.0

        if f1 > best_f1 and auc > best_auc:
            best_f1, best_auc, best_epoch = f1, auc, epoch + 1
            torch.save(model.state_dict(), best_model_path)

        print(f"[{model_name}] Epoch {epoch+1}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    print(f"[{model_name}] Best F1: {best_f1:.4f}, AUC: {best_auc:.4f} at epoch {best_epoch}")
    return best_model_path

# ============== RUN EXPERIMENTS ==============
print("Evaluating for age_group: ", age_group_to_evaluate)
model_A_path = train_and_eval("A_age_group_only", train_A, val_A)
model_B_path = train_and_eval("B_balanced_age_groups", train_B, val_B)

evaluate_on_test(model_A_path, f"Model A (Group {age_group_to_evaluate})")
evaluate_on_test(model_B_path, f"Model B (Balanced)")
