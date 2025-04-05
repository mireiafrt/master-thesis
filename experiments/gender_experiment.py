import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from monai.metrics import ROCAUCMetric
from monai.data import Dataset, decollate_batch
from monai.transforms import Compose, LoadImaged, Resized, ScaleIntensityd, RandRotate90d, ToTensord, Activations, AsDiscrete, LambdaD
from monai.networks.nets import DenseNet121
from sklearn.metrics import f1_score, roc_auc_score
import random

# ============== SETUP ==============
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Config
paths = {"metadata":"data/metadata.csv", "nifti_files":"data/preprocessed_nifti"}
columns = {"patient_id": "Patient ID", "diagnosis": "binary_diagnosis_patient", "gender": "Patient Sex"}
training = {"batch_size": 4, "num_epochs": 40, "learning_rate": 0.0001, "resize": [128, 128, 128], "rotation_prob": 0.5,
            "N_train":30, "N_val":6, "N_test":30}

# ============== CREATE SPLITS IN MEMORY ==============
metadata = pd.read_csv(paths["metadata"])
metadata["filepath"] = metadata.apply(
    lambda row: os.path.join(paths["nifti_files"], row["split"], f"{row[columns['patient_id']]}.nii.gz"),
    axis=1
)
males = metadata[metadata[columns["gender"]] == "M"]
females = metadata[metadata[columns["gender"]] == "F"]

N_TRAIN = training["N_train"]
N_VAL = training["N_val"]
N_TEST = training["N_test"]

# Sample TEST set
test_set = males.sample(n=N_TEST, random_state=7)
test_ids = set(test_set[columns["patient_id"]])

# Remove test IDs from males and females that have been used in test
males_wo_test = males[~males[columns["patient_id"]].isin(test_ids)]
females_wo_test = females.copy()  # females not in test set by design

# Sample VAL sets (A is all male, B is 50-50 males-females)
val_A = males_wo_test.sample(n=N_VAL, random_state=1)
val_A_ids = set(val_A[columns["patient_id"]])

val_B = pd.concat([
    males_wo_test.sample(n=N_VAL // 2, random_state=2),
    females_wo_test.sample(n=N_VAL // 2, random_state=3)
])
val_B_ids = set(val_B[columns["patient_id"]])

# Final exclusion set for training (test + val)
exclude_ids_A = test_ids.union(val_A_ids)
exclude_ids_B = test_ids.union(val_B_ids)

# Train sets can overlap with each other but not with val/test
train_males_A = males[~males[columns["patient_id"]].isin(exclude_ids_A)]
train_males_B = males[~males[columns["patient_id"]].isin(exclude_ids_B)]
train_females_B = females[~females[columns["patient_id"]].isin(exclude_ids_B)]

train_A = train_males_A.sample(n=N_TRAIN, random_state=4)
train_B = pd.concat([
    train_males_B.sample(n=N_TRAIN // 2, random_state=5),
    train_females_B.sample(n=N_TRAIN // 2, random_state=6)
])

print("Set splits completed")

# ============== FUNCTION TO BUILD DATASET ==============
def build_monai_data(df):
    return [{"img": row["filepath"], "label": int(row[columns["diagnosis"]])} for _, row in df.iterrows()]

# ============== DEFINE MONAI TRANSFORMS ==============
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

    print("Train and val data loaded")

    best_f1 = -1
    best_auc = -1
    best_epoch = -1
    best_model_path = f"{model_name}_best_model.pt"

    for epoch in range(training["num_epochs"]):
        model.train()
        epoch_loss, correct, total = 0, 0, 0

        for batch_data in tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1}"):
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

        if f1 >= best_f1 and auc >= best_auc:
            best_f1, best_auc, best_epoch = f1, auc, epoch + 1
            torch.save(model.state_dict(), best_model_path)

        print(f"[{model_name}] Epoch {epoch+1}: Val Acc={acc:.4f}, Val F1={f1:.4f}, Val AUC={auc:.4f}")

    print(f"[{model_name}] Best F1: {best_f1:.4f}, AUC: {best_auc:.4f} at epoch {best_epoch}")
    return best_model_path

# ============== RUN EXPERIMENTS ==============
model_A_path = train_and_eval("model_A_male_only", train_A, val_A)
model_B_path = train_and_eval("model_B_balanced", train_B, val_B)

evaluate_on_test(model_A_path, "Model A (Male Only)")
evaluate_on_test(model_B_path, "Model B (Balanced)")
