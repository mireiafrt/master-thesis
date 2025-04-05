import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from monai.metrics import ROCAUCMetric
from monai.data import Dataset, decollate_batch
from monai.transforms import Compose, LoadImaged, Resized, ScaleIntensityd, RandRotate90d, Activations, AsDiscrete, LambdaD
from monai.networks.nets import DenseNet121
from torch.utils.data import DataLoader
import csv
import random

# ============== SETUP CONFIG ==============
paths = {"metadata":"data/metadata.csv", "nifti_files":"data/preprocessed_nifti", "log_csv":"logs/gender_experiment_results.csv"}
columns = {"patient_id": "Patient ID", "diagnosis": "binary_diagnosis_patient", "gender": "Patient Sex", "age_group":"age_group"}
training = {"batch_size": 4, "num_epochs": 20, "learning_rate": 0.0001, "resize": [128, 128, 128], "rotation_prob": 0.5,
            "N_train":15, "N_val":5, "N_test":20, "age_group": "40-60", "n_experiments": 10}

# for 40-60, 60-80 i can use: "N_train":15, "N_val":5, "N_test":20
# if want to evaluate on equal ground also 20-40, idk what to do to include all groups

N_TRAIN = training["N_train"]
N_VAL = training["N_val"]
N_TEST = training["N_test"]
age_group_to_evaluate = training["age_group"]  # example: "40–60"

# ============== SETUP LOGGING ==============
os.makedirs("logs", exist_ok=True)
csv_columns = ["seed", "model", "train_loss", "val_f1", "val_acc", "val_auc", "val_recall", "val_precision",
    "test_f1", "test_acc", "test_auc", "test_recall", "test_precision"
]
log_file_exists = os.path.exists(paths["log_csv"])
log_f = open(paths["log_csv"], "a", newline="")
writer = csv.DictWriter(log_f, fieldnames=csv_columns)
if not log_file_exists:
    writer.writeheader()

# ============== HELPER FUNCTIONS ==============
def build_monai_data(df):
    return [{"img": row["filepath"], "label": int(row[columns["diagnosis"]])} for _, row in df.iterrows()]

def get_transform_train():
    return Compose([
        LoadImaged(keys=["img"], ensure_channel_first=True),
        LambdaD(keys=["img"], func=lambda x: x.permute(0, 3, 1, 2)),
        ScaleIntensityd(keys=["img"]),
        Resized(keys=["img"], spatial_size=tuple(training['resize'])),
        RandRotate90d(keys=["img"], prob=training['rotation_prob'], spatial_axes=[0, 2])
    ])
def get_transform_eval():
    return Compose([
        LoadImaged(keys=["img"], ensure_channel_first=True),
        LambdaD(keys=["img"], func=lambda x: x.permute(0, 3, 1, 2)),
        ScaleIntensityd(keys=["img"]),
        Resized(keys=["img"], spatial_size=tuple(training["resize"]))
    ])
def train_and_val(train_df, val_df, seed, model_name):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])

    train_ds = Dataset(data=build_monai_data(train_df), transform=get_transform_train())
    val_ds = Dataset(data=build_monai_data(val_df), transform=get_transform_eval())
    train_loader = DataLoader(train_ds, batch_size=training["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=training["batch_size"], shuffle=False, num_workers=2)

    best_f1 = -1
    best_auc = -1
    best_metrics = {}
    best_state_dict = None

    for epoch in range(training["num_epochs"]):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1}"):
            inputs = batch["img"].to(device)
            labels = batch["label"].to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"[{model_name}] Epoch {epoch+1}: Train Loss={epoch_loss:.4f}")

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["img"].to(device)
                labels = batch["label"].to(device).long()
                outputs = model(x)
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        f1 = f1_score(val_labels, val_preds, zero_division=0)
        acc = accuracy_score(val_labels, val_preds)
        auc = roc_auc_score(val_labels, val_preds) if len(set(val_labels)) == 2 else 0.0
        recall = recall_score(val_labels, val_preds, zero_division=0)
        precision = precision_score(val_labels, val_preds, zero_division=0)
        print(f"[{model_name}] Epoch {epoch+1}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

        if f1 > best_f1 or (f1 == best_f1 and auc > best_auc):
            best_f1 = f1
            best_auc = auc
            best_state_dict = model.state_dict()
            best_metrics = {
                "train_loss": epoch_loss,
                "val_f1": f1,
                "val_acc": acc,
                "val_auc": auc,
                "val_recall": recall,
                "val_precision": precision
            }

    print(f"[{model_name}] Best F1: {best_f1:.4f}, AUC: {best_auc:.4f}")
    # Restore best weights before returning
    model.load_state_dict(best_state_dict)
    return best_metrics, model

def evaluate_on_test(model, test_df, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_ds = Dataset(data=build_monai_data(test_df), transform=get_transform_eval())
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
    print(f"[TEST {model_name}] Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    return {
        "test_f1": f1,
        "test_acc": acc,
        "test_auc": auc,
        "test_recall": recall,
        "test_precision": precision
    }

# ============== EXPERIMENT LOOP ==============
metadata = pd.read_csv(paths["metadata"])
metadata["filepath"] = metadata.apply(
    lambda row: os.path.join(paths["nifti_files"], row["split"], f"{row[columns['patient_id']]}.nii.gz"),
    axis=1
)
age_groups = metadata["age_group"].unique()
current_group = metadata[metadata["age_group"] == age_group_to_evaluate].copy()
other_groups = metadata[metadata["age_group"] != age_group_to_evaluate].copy()

for seed in range(training["n_experiments"]):
    print(f"\n========= Seed {seed} =========")

    # --- Stratified test split from current age group only
    remaining_current_group, test_set = train_test_split(current_group, test_size=N_TEST,
        stratify=current_group[columns["diagnosis"]],
        random_state=seed
    )

    # --- Stratified VAL_A and train_A from current age group only
    train_A, val_A = train_test_split(remaining_current_group, train_size=N_TRAIN, test_size=N_VAL,
        stratify=remaining_current_group[columns["diagnosis"]],
        random_state=seed + 1
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
        # try to do stratified sampling, if error dont do stratified (not enough samples of class to stratify)
        try:
            train_grp, val_grp = train_test_split(group_data,
                train_size=train_parts[grp],
                test_size=val_parts[grp],
                stratify=group_data[columns["diagnosis"]],
                random_state=seed + 2
            )
        except ValueError:
            train_grp, val_grp = train_test_split(group_data,
                train_size=train_parts[grp],
                test_size=val_parts[grp],
                random_state=seed + 2
            )

        train_B_parts.append(train_grp)
        val_B_parts.append(val_grp)
    train_B = pd.concat(train_B_parts)
    val_B = pd.concat(val_B_parts)

    # Train and Evaluate Model A
    metrics_A_train, model_A = train_and_val(train_A, val_A, seed, "model A")
    metrics_A_test = evaluate_on_test(model_A, test_set, "model A")

    # Train and Evaluate Model B
    metrics_B_train, model_B = train_and_val(train_B, val_B, seed, "model B")
    metrics_B_test = evaluate_on_test(model_B, test_set, "model B")

    # Log results
    writer.writerow({"seed": seed, "model": "A", **metrics_A_train, **metrics_A_test})
    writer.writerow({"seed": seed, "model": "B", **metrics_B_train, **metrics_B_test})
    log_f.flush()

log_f.close()
print("All experiments completed and logged.")
