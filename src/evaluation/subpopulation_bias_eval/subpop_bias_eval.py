# First, train a (5) single model on the hold out real data (train+val) (can be the same as the one from utility_eval?)

# Before next step, we need the 5 synthetic GT sets (on each set, each subgroup has to match the size of the real GT)

# For each subgroup:
    # Evaluate the classifier performance on the subgroup on the REAL TEST set
    # Evaluate the classifier performance on the subgroup on the REAL GT set
    # Evaluate the classifier performance on the subgroup on EACH of the 5 SYN GT sets

# Compare difference between each TRIO of metrics (5 times, 1 for each syn GT set variation)

import os
import torch
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from monai.data import Dataset, decollate_batch
from monai.transforms import Compose, LoadImaged, ScaleIntensityd, NormalizeIntensityd, RandGaussianSmoothd, RandFlipd, RandAffined, Resized, LambdaD, Activations, AsDiscrete
from monai.losses import FocalLoss
from monai.utils import set_determinism
from monai.networks.nets import DenseNet121
from monai.metrics import ROCAUCMetric
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, average_precision_score, matthews_corrcoef
from torch.utils.data import DataLoader
import random
from scipy.stats import t
import csv

# ========== Load Config ==========
with open("config/evaluation/subpop_eval.yaml", "r") as f:
    config = yaml.safe_load(f)
    
paths = config["paths"]
models = config["models"]
columns = config["columns"]
num_trains = config["num_trains"]
training = config["training"]
split = config["split"]
output = config["output"]

syn_paths = [paths["syn_1_path"], paths["syn_2_path"], paths["syn_3_path"], paths["syn_4_path"], paths["syn_5_path"]]
models = [models["model_1"], models["model_2"], models["model_3"], models["model_4"], models["model_5"]]

# ============== SETUP LOGGING ==============
os.makedirs(output["results_dir"], exist_ok=True)
csv_columns = ["model", "set", "subgroup", "test_f1", "test_acc", "test_auc", "test_recall", "test_precision", "test_auc_pr", "test_mcc"]
log_path = os.path.join(output["results_dir"], "results_5runs.csv")
log_file_exists = os.path.exists(log_path)

log_f = open(log_path, "a", newline="")
writer = csv.DictWriter(log_f, fieldnames=csv_columns)
if not log_file_exists:
    writer.writeheader()

# ========== Load and preprocess REAL data ========== (we only need to do this once, to create train, val and test df)
print("Reading metadata ...")
df = pd.read_csv(paths["real_imgs_csv"])
df = df[(df["use"] == True)]

# Split according to 'split' column
test_df = df[df[columns["split_col"]] == "test"]
gt_df = df[df[columns["split_col"]] == "ground_truth"]

# Get subgroups of interest from gt set
reference_combinations = gt_df[["sex", "age_group"]].drop_duplicates().values.tolist()

# ========== Helper Functions ==========
def build_monai_data(df_subset, image_col, target_col):
    return [{"img": row[image_col], "target": int(row[target_col])} for _, row in df_subset.iterrows()]

def get_eval_transform():
    return Compose([
        LoadImaged(keys=["img"], ensure_channel_first=True),
        ScaleIntensityd(keys=["img"]),
        LambdaD(keys=["img"], func=lambda x: x.repeat(3, 1, 1)),
        NormalizeIntensityd(keys=["img"], subtrahend=torch.tensor([0.485, 0.456, 0.406]), divisor=torch.tensor([0.229, 0.224, 0.225]), channel_wise=True),
        Resized(keys=["img"], spatial_size=(256, 256)),
    ])

def evaluate_on_test(best_model_path, test_df, image_path_col):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=2, pretrained=True).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    test_ds = Dataset(data=build_monai_data(test_df, image_path_col, columns["target"]), transform=get_eval_transform())
    test_loader = DataLoader(test_ds, batch_size=training["batch_size"], num_workers=2) # changed batch size from 1 to param batch size

    post_pred = Activations(softmax=True)
    post_target = AsDiscrete(to_onehot=2)
    post_discrete = AsDiscrete(argmax=True, to_onehot=2)
    auc_metric = ROCAUCMetric(average="macro")

    y_pred = torch.tensor([], dtype=torch.float32, device=device)
    y = torch.tensor([], dtype=torch.long, device=device)
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            x = batch["img"].to(device)
            targets = batch["target"].to(device)
            outputs = model(x)
            y_pred = torch.cat([y_pred, outputs], dim=0)
            y = torch.cat([y, targets], dim=0)

    # Post-processing directly on the full batch
    y_pred_probs = post_pred(y_pred) # no need to decollate here
    y_true = [post_target(i) for i in decollate_batch(y, detach=False)]
    y_pred_bin = [post_discrete(i) for i in decollate_batch(y_pred_probs, detach=False)]
    y_pred_classes = torch.stack([p.argmax(dim=0) for p in y_pred_bin]).cpu().numpy()
    y_true_classes = torch.stack([t.argmax(dim=0) for t in y_true]).cpu().numpy()

    # see how many classes are actually present
    labels_present = np.unique(y_true_classes)

    # for auc, skip if less than 2 classes (would give nan)
    if len(labels_present) < 2:
        auc = np.nan
        auc_pr = np.nan
        mcc = np.nan
    else:
        # AUC (ROC)
        auc_metric.reset()
        auc_metric(y_pred_probs, y_true)
        auc = auc_metric.aggregate().item()

        # AUC-PR (macro) using existing one-hot encoded y_true
        y_true_onehot = torch.stack(y_true).cpu().numpy()
        y_probs = y_pred_probs.cpu().numpy()
        auc_pr = average_precision_score(y_true_onehot, y_probs, average="macro")

        # MATTHEW CORR COEFF
        mcc = matthews_corrcoef(y_true_classes, y_pred_classes)

    acc = accuracy_score(y_true_classes, y_pred_classes)

    # for f1, recall, and precision, send labels so to only compute metrics for the present classes
    f1 = f1_score(y_true_classes, y_pred_classes, labels=labels_present, average='macro', zero_division=0)
    recall = recall_score(y_true_classes, y_pred_classes, labels=labels_present, average='macro', zero_division=0)
    precision = precision_score(y_true_classes, y_pred_classes, labels=labels_present, average='macro', zero_division=0)

    print(f"TEST: ACC={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, REC={recall:.4f}, PREC={precision:.4f}, AUC-PR={auc_pr:.4f}, MCC={mcc:.4f}")
    print(classification_report(y_true_classes, y_pred_classes,digits=4, zero_division=0))
    
    return {"test_f1": f1, "test_acc": acc, "test_auc": auc, "test_recall": recall, "test_precision": precision, "test_auc_pr": auc_pr, "test_mcc": mcc}

def eval_per_groups(model_num, set_name, writer, reference_combinations, best_model_state, test_df, image_path_col):
    # loop through the different subgroups
    for sex, age_group in reference_combinations:
        sub_name = f"{sex}-{age_group}"
        sub_df = test_df[(test_df["sex"] == sex) & (test_df["age_group"] == age_group)]
        # get evaluation results of the subgroup
        print(f"Evaluating on subgroup: {sex} and {age_group}")
        test_metrics = evaluate_on_test(best_model_state, sub_df, image_path_col)
        writer.writerow({"model":model_num, "set":set_name, "subgroup":sub_name, **test_metrics})
        log_f.flush()

# ========== Run Experiments ==========
for j in range(0, num_trains):
    print(f"Using Trained model {j+1}")
    model_state = models[j]

    # FIRST EVAL: REAL GT
    print("Evaluating on REAL Ground Truth ...")
    eval_per_groups(model_num=j+1, set_name="GT", writer=writer, reference_combinations=reference_combinations,
                    best_model_state=model_state, test_df=gt_df, image_path_col=columns["real_img_path"])
    
    # SECOND EVAL: REAL TEST
    print("Evaluating on REAL Test ...")
    eval_per_groups(model_num=j+1, set_name="Test", writer=writer, reference_combinations=reference_combinations,
                    best_model_state=model_state, test_df=test_df, image_path_col=columns["real_img_path"])
    
    # THIRD EVAL: SYN GT sets
    for i in range(0, len(syn_paths)):
        print(f"Evaluating on SYN GT set {i+1}")
        syn_df = pd.read_csv(syn_paths[i])
        eval_per_groups(model_num=j+1, set_name=f"SYN_{i+1}", writer=writer, reference_combinations=reference_combinations,
                        best_model_state=model_state, test_df=syn_df, image_path_col=columns["syn_img_path"])
        
log_f.close()
print("\nAll runs completed and logged.")

# ========== COMPUTE CI INTERVALS ==========

# analysis better in notebook