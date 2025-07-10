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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
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

# I only have 1 trained model here, but get the 5 from the other computer
models = [models["model_1"], models["model_1"], models["model_1"], models["model_1"], models["model_1"]]

# ============== SETUP LOGGING ==============
os.makedirs(output["results_dir"], exist_ok=True)
csv_columns = ["run", "set", "subgroup", "test_f1", "test_acc", "test_auc", "test_recall", "test_precision"]
log_path = os.path.join(output["results_dir"], "general_results_5runs.csv")
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
    else:
        auc_metric.reset()
        auc_metric(y_pred_probs, y_true)
        auc = auc_metric.aggregate().item()

    acc = accuracy_score(y_true_classes, y_pred_classes)

    # for f1, recall, and precision, send labels so to only compute metrics for the present classes
    f1 = f1_score(y_true_classes, y_pred_classes, labels=labels_present, average='macro', zero_division=0)
    recall = recall_score(y_true_classes, y_pred_classes, labels=labels_present, average='macro', zero_division=0)
    precision = precision_score(y_true_classes, y_pred_classes, labels=labels_present, average='macro', zero_division=0)

    print(f"TEST: ACC={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, REC={recall:.4f}, PREC={precision:.4f}")
    print(classification_report(y_true_classes, y_pred_classes,digits=4, zero_division=0))
    
    return {"test_f1": f1, "test_acc": acc, "test_auc": auc, "test_recall": recall, "test_precision": precision}

def eval_per_groups(model_num, set_name, writer, reference_combinations, best_model_state, test_df, image_path_col):
    # loop through the different subgroups
    for sex, age_group in reference_combinations:
        sub_name = f"{sex}-{age_group}"
        sub_df = test_df[(test_df["sex"] == sex) & (test_df["age_group"] == age_group)]
        # get evaluation results of the subgroup
        print(f"Evaluating on subgroup: {sex} and {age_group}")
        test_metrics = evaluate_on_test(best_model_state, sub_df, image_path_col)
        writer.writerow({"run":model_num, "set":set_name, "subgroup":sub_name, **test_metrics})
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
    
        
log_f.close()
print("\nAll runs completed and logged.")

# ========== COMPUTE CI INTERVALS ==========
def mean_ci(results_array):
    """
    Return mean and 95 % CI across sets.
    """
    mean = results_array.mean()
    sem = results_array.std(ddof=0) / np.sqrt(len(results_array))
    # 95 % two-sided t-interval with df = K-1
    df = len(results_array) - 1
    t95 = t.ppf(0.975, df)  
    ci = t95 * sem
    return mean.item(), ci.item()

# read the log results file, and compute 95% CI intervals of all test metrics between the 5 synthetic set runs
test_metrics = ["test_f1", "test_acc", "test_auc", "test_recall", "test_precision"]
results_df = pd.read_csv(log_path)

for metric in test_metrics:
    for sex, age_group in reference_combinations:
        filtered_df = results_df[(results_df["sex"] == sex) & (results_df["age_group"] == age_group)]

        # get result of REAL GT on the metric
        gt_metric_result = filtered_df[(filtered_df['model_num']==j+1)&(filtered_df['set_name']=="GT")][metric].values
        # get results of REAL Test on the metric
        test_metric_result = filtered_df[(filtered_df['model_num']==j+1)&(filtered_df['set_name']=="Test")][metric].values
            
        # print GT CI
        metric_mean, metric_ci   = mean_ci(gt_metric_result)
        print(f"GT 95%CI {metric} for {sex}-{age_group}: {metric_mean:.6f} ± {metric_ci:.6f}  (95 % CI, n={len(gt_metric_result)})")

        # print Test CI
        metric_mean, metric_ci   = mean_ci(test_metric_result)
        print(f"Test 95% CI {metric} for {sex}-{age_group}: {metric_mean:.6f} ± {metric_ci:.6f}  (95 % CI, n={len(test_metric_result)})")
