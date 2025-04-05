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
columns = {"patient_id": "Patient ID", "diagnosis": "binary_diagnosis_patient", "gender": "Patient Sex"}
training = {"batch_size": 4, "num_epochs": 40, "learning_rate": 0.0001, "resize": [128, 128, 128], "rotation_prob": 0.5,
            "N_train":30, "N_val":6, "N_test":30, "n_experiments": 10}

N_TRAIN = training["N_train"]
N_VAL = training["N_val"]
N_TEST = training["N_test"]

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

        f1 = f1_score(val_labels, val_preds)
        acc = accuracy_score(val_labels, val_preds)
        auc = roc_auc_score(val_labels, val_preds) if len(set(val_labels)) == 2 else 0.0
        recall = recall_score(val_labels, val_preds)
        precision = precision_score(val_labels, val_preds)
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

    f1 = f1_score(y_true, y_pred)
    acc = np.mean(np.array(y_pred) == np.array(y_true))
    auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) == 2 else 0.0
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
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
males = metadata[metadata[columns["gender"]] == "M"]
females = metadata[metadata[columns["gender"]] == "F"]

for seed in range(training["n_experiments"]):
    print(f"\n========= Seed {seed} =========")

    # Test set (males only)
    remaining_males, test_set = train_test_split(males, test_size=N_TEST, stratify=males[columns["diagnosis"]], random_state=seed)

    # Model A
    train_A, val_A = train_test_split(
        remaining_males, train_size=N_TRAIN, test_size=N_VAL,
        stratify=remaining_males[columns["diagnosis"]], random_state=seed + 1
    )

    # Model B
    train_females_B, val_females_B = train_test_split(females, train_size=N_TRAIN // 2, test_size=N_VAL // 2,
                                                      stratify=females[columns["diagnosis"]], random_state=seed + 2)
    train_males_B, val_males_B = train_test_split(remaining_males, train_size=N_TRAIN // 2, test_size=N_VAL // 2,
                                                  stratify=remaining_males[columns["diagnosis"]], random_state=seed + 3)

    train_B = pd.concat([train_males_B, train_females_B])
    val_B = pd.concat([val_males_B, val_females_B])

    # Train and Evaluate Model A
    metrics_A_train, model_A = train_and_val(train_A, val_A, seed, "model A")
    metrics_A_test = evaluate_on_test(model_A, test_set, "model A")

    # Train and Evaluate Model B
    metrics_B_train, model_B = train_and_val(train_B, val_B, seed, "model B")
    metrics_B_test = evaluate_on_test(model_B, test_set, "model A")

    # Log results
    writer.writerow({"seed": seed, "model": "A", **metrics_A_train, **metrics_A_test})
    writer.writerow({"seed": seed, "model": "B", **metrics_B_train, **metrics_B_test})
    log_f.flush()

log_f.close()
print("All experiments completed and logged.")