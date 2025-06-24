# TRTR --> Train one model with real test set and evaluate on hold out real (initial train+val)
# TSTR ---> Train 5 models, each with 1 of the 5 synthetic test sets, and evaluate on the hold out real

# Compute the difference in metrics between each TRTR and TSTR pairs

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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
import random
from scipy.stats import t
import csv

# ========== Load Config ==========
with open("config/evaluation/utility.yaml", "r") as f:
    config = yaml.safe_load(f)
    
paths = config["paths"]
columns = config["columns"]
training = config["training"]
split = config["split"]
output = config["output"]

syn_paths = [paths["syn_1_path"], paths["syn_2_path"], paths["syn_3_path"], paths["syn_4_path"], paths["syn_5_path"]]

# ============== SETUP LOGGING ==============
os.makedirs(output["results_dir"], exist_ok=True)
csv_columns = ["syn_set", "train_loss", "val_f1", "val_acc", "val_recall", "val_precision", "val_auc", "test_f1", "test_acc", "test_auc", "test_recall", "test_precision"]
log_path = os.path.join(output["results_dir"], "general.csv")
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
hold_out_df = df[(df[columns["split_col"]] == "train") or (df[columns["split_col"]] == "val")]
test_df = df[df[columns["split_col"]] == "test"]

# ========== Helper Functions ==========
def build_monai_data(df_subset, image_col, target_col):
    return [{"img": row[image_col], "target": int(row[target_col])} for _, row in df_subset.iterrows()]

def get_train_transform():
    return Compose([
        # Load the image from file and make sure the channel is first: [C, H, W]
        LoadImaged(keys=["img"], ensure_channel_first=True),
        # Scale pixel intensities to [0, 1] range (from raw intensity range)
        ScaleIntensityd(keys=["img"]),
        LambdaD(keys=["img"], func=lambda x: x.repeat(3, 1, 1)),
        NormalizeIntensityd(keys=["img"],
            subtrahend=torch.tensor([0.485, 0.456, 0.406]),  # ImageNet means (R, G, B)
            divisor=torch.tensor([0.229, 0.224, 0.225]),      # ImageNet stds (R, G, B)
            channel_wise=True
        ),
        # Apply Gaussian smoothing with 50% probability to slightly blur the image
        RandGaussianSmoothd(keys=["img"], prob=0.5),
        # Random horizontal flip (axis=2) with 50% probability to augment left-right symmetry
        RandFlipd(keys=["img"], prob=0.5, spatial_axis=-1),
        # Random mild affine rotation on the 2D plane (max ~5.7°), preserves shape with border padding
        RandAffined(keys=["img"], prob=0.5,
            rotate_range=(0.0, 0.0, 0.1),
            padding_mode="border"
        ),
        # Resize the final image to 256x256 as expected by ImageNet-pretrained models
        Resized(keys=["img"], spatial_size=(256, 256)),
    ])

def get_eval_transform():
    return Compose([
        LoadImaged(keys=["img"], ensure_channel_first=True),
        ScaleIntensityd(keys=["img"]),
        LambdaD(keys=["img"], func=lambda x: x.repeat(3, 1, 1)),
        NormalizeIntensityd(keys=["img"], subtrahend=torch.tensor([0.485, 0.456, 0.406]), divisor=torch.tensor([0.229, 0.224, 0.225]), channel_wise=True),
        Resized(keys=["img"], spatial_size=(256, 256)),
    ])

def train_model(train_df, val_df, syn_set_num, image_path_col):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using CUDA: {torch.cuda.is_available()}")
    # use pretrained ImageNet weights
    model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=2, pretrained=True).to(device)

    loss_fn = FocalLoss(to_onehot_y=True, use_softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    auc_metric = ROCAUCMetric(average="macro")
    post_pred = Compose([Activations(softmax=True)])
    post_target = Compose([AsDiscrete(to_onehot=2)])
    post_discrete = AsDiscrete(argmax=True, to_onehot=2)

    train_ds = Dataset(data=build_monai_data(train_df, image_path_col, columns["target"]), transform=get_train_transform())
    val_ds = Dataset(data=build_monai_data(val_df, image_path_col, columns["target"]), transform=get_eval_transform())

    train_loader = DataLoader(train_ds, batch_size=training["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=training["batch_size"], shuffle=False, num_workers=2)

    best_model_state = None
    best_f1 = -1
    best_auc = -1
    best_metrics = {}

    for epoch in range(training["num_epochs"]):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"[Syn set {syn_set_num}] Epoch {epoch+1}"):
            x, y = batch["img"].to(device), batch["target"].to(device).long()
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Train Loss={total_loss:.4f}")

        # Validation
        model.eval()
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)
        with torch.no_grad():
            for batch in val_loader:
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

        auc_metric.reset()
        auc_metric(y_pred_probs, y_true)
        auc = auc_metric.aggregate().item()
        acc = accuracy_score(y_true_classes, y_pred_classes)
        f1 = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
        recall = recall_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
        precision = precision_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)

        print(f"Epoch {epoch+1} VAL: ACC={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, REC={recall:.4f}, PREC={precision:.4f}")

        scheduler.step(1 - f1)

        if f1 > best_f1 or (f1 == best_f1 and auc > best_auc):
            best_f1 = f1
            best_auc = auc
            best_metrics = {
                "train_loss": total_loss,
                "f1": f1, "acc": acc, "recall": recall,
                "precision": precision, "auc": auc
            }
            # save state of model as the best state
            best_model_state = model.state_dict()

    print(f"Best F1: {best_f1:.4f}, AUC: {best_auc:.4f}")
    return best_metrics, best_model_state

def evaluate_on_test(best_model_state, test_df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=2, pretrained=True).to(device)
    model.load_state_dict(best_model_state)
    model.eval()

    test_ds = Dataset(data=build_monai_data(test_df, columns["real_img_path"], columns["target"]), transform=get_eval_transform())
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=2)

    post_pred = Activations(softmax=True)
    post_target = AsDiscrete(to_onehot=2)
    post_discrete = AsDiscrete(argmax=True, to_onehot=2)
    auc_metric = ROCAUCMetric(average="macro")

    y_pred = torch.tensor([], dtype=torch.float32, device=device)
    y = torch.tensor([], dtype=torch.long, device=device)
    with torch.no_grad():
        for batch in test_loader:
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

    auc_metric.reset()
    auc_metric(y_pred_probs, y_true)
    auc = auc_metric.aggregate().item()
    acc = accuracy_score(y_true_classes, y_pred_classes)
    f1 = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
    recall = recall_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
    precision = precision_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)

    print(f"TEST: ACC={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, REC={recall:.4f}, PREC={precision:.4f}")
    return {"test_f1": f1, "test_acc": acc, "test_auc": auc, "test_recall": recall, "test_precision": precision}

# ========== Run Experiments ==========

# FIRST EXPERIMENT: TRAIN ON REAL and EVAL ON REAL (hold-out)
print("Training model on REAL test set")
# split train_df into train-val
train_df, val_df = train_test_split(test_df, train_size=split["train_size"], stratify=test_df[columns["target"]], random_state=42)
# train model
train_metrics, best_model_state = train_model(train_df, val_df)
# save the model for possible future uses
os.makedirs(os.path.dirname(output["real_model_output"]), exist_ok=True)
torch.save(best_model_state, output["real_model_output"])
print("Saved best model")
# Evaluate on real holdout
print("Evaluating on REAL HOLD-OUT")
test_metrics = evaluate_on_test(best_model_state, hold_out_df)
writer.writerow({"syn_set": 0, **train_metrics, **test_metrics})
log_f.flush()

# SECOND EXPERIMENT: Train on synthetic sets and EVAL on REAL (hold-out)
model_output_paths = [output["syn_model_output_1"], output["syn_model_output_2"], output["syn_model_output_3"], output["syn_model_output_4"], output["syn_model_output_5"]]
for i in range(0, len(syn_paths)):
    print(f"Training model on SYN set {i+1}")
    syn_df = pd.read_csv(syn_paths[i])
    # split train_df into train-val
    train_df, val_df = train_test_split(syn_df, train_size=split["train_size"], stratify=syn_df[columns["target"]], random_state=42)
    # train model
    train_metrics, best_model_state = train_model(train_df, val_df)
    # save the model for possible future uses
    os.makedirs(os.path.dirname(model_output_paths[i]), exist_ok=True)
    torch.save(best_model_state, model_output_paths[i])
    print("Saved best model")
    # Evaluate on real holdout
    print("Evaluating on REAL HOLD-OUT")
    test_metrics = evaluate_on_test(best_model_state, hold_out_df)
    writer.writerow({"syn_set": i+1, **train_metrics, **test_metrics})
    log_f.flush()

log_f.close()
print("\nAll runs completed and logged.")

# ========== COMPUTE CI INTERVALS ==========
def mean_ci(results_array):
    """
    Return mean and 95 % CI across sets.
    """
    mean = results_array.mean()
    sem  = results_array.std(unbiased=False) / (len(results_array) ** 0.5)
    # 95 % two-sided t-interval with df = K-1
    df = len(results_array) - 1
    t95 = t.ppf(0.975, df)  
    ci = t95 * sem
    return mean.item(), ci.item()

# read the log results file, and compute 95% CI intervals of all test metrics between the 5 synthetic set runs
test_metrics = ["test_f1", "test_acc", "test_auc", "test_recall", "test_precision"]
results_df = pd.read_csv(log_path)
for metric in test_metrics:
    # get result of REAL test on the metric
    real_metric_result = results_df[results_df['syn_set']==0][metric].values
    # get results of SYN sets on the metric
    syn_metric_results = results_df[results_df['syn_set']!=0][metric].values
    # subtract real result from each synthetic result to create the difference
    metric_difference = syn_metric_results - real_metric_result[0]
    # calculate CI for the difference in the metric
    metric_mean, metric_ci   = mean_ci(metric_difference)
    print(f"Difference in {metric} : {metric_mean:.6f} ± {metric_ci:.6f}  (95 % CI, n={len(syn_paths)})")