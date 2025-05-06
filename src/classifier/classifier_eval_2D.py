import os
import torch
import yaml
import pandas as pd
from tqdm import tqdm
from monai.transforms import Compose, LoadImaged, ScaleIntensityd, Resized, LambdaD, Activations, AsDiscrete
from monai.data import Dataset, DataLoader, decollate_batch
from monai.networks.nets import DenseNet121
from monai.metrics import ROCAUCMetric
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

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
with open("config/classifier/classifier_eval_covid.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config["paths"]
columns = config["columns"]

# Define transformations
eval_transform = Compose([
    LoadImaged(keys=["img"], ensure_channel_first=True),
    ScaleIntensityd(keys=["img"]),
    LambdaD(keys=["img"], func=lambda x: x.repeat(3, 1, 1)),
    NormalizeIntensityd(keys=["img"], subtrahend=torch.tensor([0.485, 0.456, 0.406]), divisor=torch.tensor([0.229, 0.224, 0.225]), channel_wise=True),
    Resized(keys=["img"], spatial_size=(256, 256)),
])
post_pred = Activations(softmax=True)
post_label = AsDiscrete(to_onehot=2)
post_discrete = AsDiscrete(argmax=True, to_onehot=2)

# Read and prepare data
metadata = pd.read_csv(paths['metadata'])
eval_df = metadata[metadata["split"] == columns['split']].copy()
eval_data = [{"img": row[columns["image_path"]], "label": int(row["label"])} for _, row in eval_df.iterrows()]
# Dataset and DataLoader
eval_ds = Dataset(data=eval_data, transform=eval_transform)
eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=2)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# Load trained model and set to evaluation mode
model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=2, pretrained=True).to(device)
model.load_state_dict(torch.load(paths['model'], map_location=device))
model.eval()

# EVALUATION
auc_metric = ROCAUCMetric(average="macro")
y_pred = torch.tensor([], dtype=torch.float32, device=device)
y = torch.tensor([], dtype=torch.long, device=device)
with torch.no_grad():
    for batch in eval_loader:
        x = batch["img"].to(device)
        labels = batch["label"].to(device)
        outputs = model(x)
        y_pred = torch.cat([y_pred, outputs], dim=0)
        y = torch.cat([y, labels], dim=0)

# Post-processing directly on the full batch
y_pred_probs = post_pred(y_pred) # no need to decollate here
y_true = [post_label(i) for i in decollate_batch(y, detach=False)]
y_pred_bin = [post_discrete(i) for i in decollate_batch(y_pred_probs, detach=False)]
y_pred_classes = torch.stack([p.argmax(dim=0) for p in y_pred_bin]).cpu().numpy()
y_true_classes = torch.stack([t.argmax(dim=0) for t in y_true]).cpu().numpy()

auc_metric.reset()
auc_metric(y_pred_probs, y_true)
auc = auc_metric.aggregate().item()
acc = accuracy_score(y_true_classes, y_pred_classes)
f1 = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
recall = recall_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
precision = precision_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)

print(f"TEST: ACC={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, REC={recall:.4f}, PREC={precision:.4f}")

# Save predictions merged with metadata and save csv with predictions
eval_df["predicted_label"] = y_pred_classes
eval_df["predicted_prob_class1"] = y_pred_probs[:, 1].detach().cpu().numpy()
eval_df.to_csv(paths['pred_output'], index=False)
print(f"Predictions saved to {paths['pred_output']}")

################# ANALYZE PER GROUP METRICS #################
# Analyze FPR/FNR by gender
fpr_fnr_by_gender = compute_fpr_fnr_by_subgroup(eval_df, subgroup_col="sex", true_label=columns['label'], pred_label='predicted_label')
print("FPR/FNR by Gender:")
print(fpr_fnr_by_gender)

# Analyze FPR/FNR by age group
fpr_fnr_by_age = compute_fpr_fnr_by_subgroup(eval_df, subgroup_col="age_group", true_label=columns['label'], pred_label='predicted_label')
print("\nFPR/FNR by Age Group:")
print(fpr_fnr_by_age)

# Analyze FPR/FNR by sex & age group
eval_df["sex_age_group"] = eval_df[['sex', 'age_group']].astype(str).agg("_".join, axis=1)
fpr_fnr_by_sex_age = compute_fpr_fnr_by_subgroup(eval_df, subgroup_col="sex_age_group", true_label=columns['label'], pred_label='predicted_label')
print("\nFPR/FNR by Sex & Age Group:")
print(fpr_fnr_by_sex_age)