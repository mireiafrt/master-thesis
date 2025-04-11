import os
import torch
import yaml
import pandas as pd
from tqdm import tqdm
from monai.transforms import Compose, LoadImaged, ScaleIntensityd, Resized, LambdaD, Activations, AsDiscrete
from monai.data import Dataset, DataLoader, decollate_batch
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

def compute_fpr_fnr_by_subgroup(df, subgroup_col, true_label, pred_label):
    """
    Compute FPR and FNR for each category in the selected subgroup column.

    Parameters:
        df (pd.DataFrame): DataFrame with true labels and predictions
        subgroup_col (str): Column name for subgroup (e.g., 'Patient Sex', 'age_group')
        label_col (str): Column name for true labels (default is 'binary_diagnosis_patient')
        pred_col (str): Column name for predicted labels (default is 'predicted_label')

    Returns:
        pd.DataFrame: FPR and FNR per subgroup value
    """
    results = []

    for group in df[subgroup_col].unique():
        group_df = df[df[subgroup_col] == group]

        y_true = group_df[true_label]
        y_pred = group_df[pred_label]

        # False Positive Rate = FP / (FP + TN)
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else None

        # False Negative Rate = FN / (FN + TP)
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else None

        results.append({
            subgroup_col: group,
            "FPR": round(fpr, 4) if fpr is not None else None,
            "FNR": round(fnr, 4) if fnr is not None else None,
            "N": len(group_df)
        })

    return pd.DataFrame(results)

# Set seeds for reproducibility
torch.manual_seed(42)

# Load config
with open("config/classifier/classifier_eval.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config["paths"]
columns = config["columns"]

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


# Load metadata + eval data
metadata = pd.read_csv(paths['metadata'])
eval_df = metadata[metadata["split"] == columns['split']].copy()
eval_data = [
    {"img": os.path.join(paths['nifti_files'], f"{pid}.nii.gz"), "label": int(label), "Patient ID": pid}
    for pid, label in zip(eval_df[columns['patient_id']], eval_df[columns['diagnosis']])
]

# Transforms (no augmentation)
transforms = Compose([
    LoadImaged(keys=["img"], ensure_channel_first=True),
    LambdaD(keys=["img"], func=lambda x: x.permute(0, 3, 1, 2)),  # [1, D, H, W]
    ScaleIntensityd(keys=["img"]),
    Resized(keys=["img"], spatial_size=(128, 128, 128)),
])
post_pred = Compose([Activations(softmax=True)])
post_label = Compose([AsDiscrete(to_onehot=2)])

# Dataset and DataLoader
eval_ds = Dataset(data=eval_data, transform=transforms)
eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=2)

# Load trained model and set to evaluation mode
model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
model.load_state_dict(torch.load(paths['model'], map_location=device))
model.eval()

# Evaluation
auc_metric = ROCAUCMetric()
all_preds = []
all_true = []
all_probs = []
with torch.no_grad():
    for batch in tqdm(eval_loader, desc="Evaluating"):
        inputs = batch["img"].to(device)
        labels = batch["label"].to(device)
        outputs = model(inputs)

        preds = torch.argmax(outputs, dim=1).cpu().item()
        prob_class_1 = torch.softmax(outputs, dim=1)[0, 1].cpu().item()

        all_preds.append(preds)
        all_true.append(labels.cpu().item())
        all_probs.append(prob_class_1)

# Save predictions merged with metadata and save csv with predictions
eval_df["predicted_label"] = all_preds
eval_df["predicted_prob_class1"] = all_probs
eval_df.to_csv(paths['pred_output'], index=False)
print(f"Predictions saved to {paths['pred_output']}")

# Compute metrics
f1 = f1_score(all_true, all_preds, zero_division=0)
acc = accuracy_score(all_true, all_preds)
auc = roc_auc_score(all_true, all_probs)
recall = recall_score(all_true, all_preds, zero_division=0)
precision = precision_score(all_true, all_preds, zero_division=0)
conf_matrix = confusion_matrix(all_true, all_preds)

# Print results
print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Analyze FPR/FNR by gender
fpr_fnr_by_gender = compute_fpr_fnr_by_subgroup(eval_df, subgroup_col="Patient Sex", true_label=columns['diagnosis'], pred_label='predicted_label')
print("FPR/FNR by Gender:")
print(fpr_fnr_by_gender)

# Analyze FPR/FNR by age group
fpr_fnr_by_age = compute_fpr_fnr_by_subgroup(eval_df, subgroup_col="age_group", true_label=columns['diagnosis'], pred_label='predicted_label')
print("\nFPR/FNR by Age Group:")
print(fpr_fnr_by_age)
