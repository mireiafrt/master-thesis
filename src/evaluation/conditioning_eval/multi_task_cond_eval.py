# train the same classifier, but with multi-task (2 prediction heads, one for sex one for age_group)
# Train with synthetic test sets (train 5 different models), evaluate on real test set
# compute 95% CI between the 5 results

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
from monai.networks.nets import DenseNet121
from monai.metrics import ROCAUCMetric
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from torch.utils.data import DataLoader
import torch
from torch import nn
import random
from scipy.stats import t
import csv

# ========== Load Config ==========
with open("config/evaluation/conditioning_eval.yaml", "r") as f:
    config = yaml.safe_load(f)
    
paths = config["paths"]
columns = config["columns"]
target_cols = config["target"]["attribute_columns"]
normalize_target = config["target"].get("normalize_target", True)
training = config["training"]
split = config["split"]
output = config["output"]

syn_paths = [paths["syn_1_path"], paths["syn_2_path"], paths["syn_3_path"], paths["syn_4_path"], paths["syn_5_path"]]

# ============== SETUP LOGGING ==============
os.makedirs(output["results_dir"], exist_ok=True)
csv_columns = ["syn_set", "train_loss",
               "val_f1_sex", "val_acc_sex", "val_recall_sex", "val_precision_sex", "val_auc_sex",
               "val_f1_age", "val_acc_age", "val_recall_age", "val_precision_age", "val_auc_age", 
               "test_f1_sex", "test_acc_sex", "test_auc_sex", "test_recall_sex", "test_precision_sex",
               "test_f1_age", "test_acc_age", "test_auc_age", "test_recall_age", "test_precision_age"]

target_name = target_cols[0] if len(target_cols) == 1 else "_".join(target_cols) #concatenate target col(s) into single string for name of experiment
log_path = os.path.join(output["results_dir"], f"cond_multi_task.csv")
log_file_exists = os.path.exists(log_path)

log_f = open(log_path, "a", newline="")
writer = csv.DictWriter(log_f, fieldnames=csv_columns)
if not log_file_exists:
    writer.writeheader()

# ========== Load and preprocess REAL TEST data ========== (we only need to do this once)
print("Reading metadata ...")
df = pd.read_csv(paths["real_imgs_csv"])
test_df = df[(df["use"] == True)&(df["split"]=="test")]
print(f"Size train set: {len(test_df)}")

print(f"Running experiment on: {target_cols}")
# prepare each target column as category codes encoded into integers
for col in target_cols:
    if normalize_target:
        test_df[col] = test_df[col].astype("category").cat.codes
# Keep track of mapping and number of classes per target
target_maps = {
    col: dict(enumerate(test_df[col].astype("category").cat.categories))
    for col in target_cols
}
num_classes_dict = {col: len(mapping) for col, mapping in target_maps.items()}

# ========== Helper Functions and Classes ==========
class MultiTaskDenseNet(nn.Module):
    def __init__(self, num_classes_sex, num_classes_age):
        super().__init__()
        self.backbone = DenseNet121(spatial_dims=2, in_channels=3, out_channels=None, pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head_sex = nn.Linear(1024, num_classes_sex)
        self.head_age = nn.Linear(1024, num_classes_age)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x).flatten(1)
        return self.head_sex(x), self.head_age(x)

def build_monai_data(df_subset, image_col):
    return [{"img": row[image_col], "sex_target": int(row["sex"]), "age_target": int(row["age_group"])}
            for _, row in df_subset.iterrows()
    ]

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

def train_model(train_df, val_df, syn_set_num, num_classes_dict):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using CUDA: {torch.cuda.is_available()}")
    # use pretrained ImageNet weights (encoded in the backbone of this model class)
    model = MultiTaskDenseNet(num_classes_dict["sex"], num_classes_dict["age_group"]).to(device)

    loss_fn_sex = FocalLoss(to_onehot_y=True, use_softmax=True)
    loss_fn_age = FocalLoss(to_onehot_y=True, use_softmax=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    auc_metric_sex = ROCAUCMetric(average="macro")
    auc_metric_age = ROCAUCMetric(average="macro")
    post_pred = Compose([Activations(softmax=True)])
    post_target_sex = Compose([AsDiscrete(to_onehot=num_classes_dict["sex"])])
    post_discrete_sex = AsDiscrete(argmax=True, to_onehot=num_classes_dict["sex"])
    post_target_age = Compose([AsDiscrete(to_onehot=num_classes_dict["age_group"])])
    post_discrete_age = AsDiscrete(argmax=True, to_onehot=num_classes_dict["age_group"])

    train_ds = Dataset(data=build_monai_data(train_df, columns["syn_img_path"]), transform=get_train_transform())
    val_ds = Dataset(data=build_monai_data(val_df, columns["syn_img_path"]), transform=get_eval_transform())

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
            x = batch["img"].to(device)
            y_sex = batch["sex_target"].to(device).long()
            y_age = batch["age_target"].to(device).long()

            optimizer.zero_grad()

            pred_sex, pred_age = model(x)
            loss = loss_fn_sex(pred_sex, y_sex) + loss_fn_age(pred_age, y_age)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Train Loss={total_loss:.4f}")

        # Validation
        model.eval()
        pred_sex = torch.tensor([], dtype=torch.float32, device=device)
        y_sex = torch.tensor([], dtype=torch.long, device=device)
        pred_age = torch.tensor([], dtype=torch.float32, device=device)
        y_age = torch.tensor([], dtype=torch.long, device=device)
        with torch.no_grad():
            for batch in val_loader:
                x = batch["img"].to(device)
                target_sex = batch["sex_target"].to(device).long()
                target_age = batch["age_target"].to(device).long()
                output_sex, output_age = model(x)
                pred_sex = torch.cat([pred_sex, output_sex], dim=0)
                y_sex = torch.cat([y_sex, target_sex], dim=0)
                pred_age = torch.cat([pred_age, output_age], dim=0)
                y_age = torch.cat([y_age, target_age], dim=0)

        # Post-processing directly on the full batch
        # SEX
        pred_sex_probs = post_pred(pred_sex)
        y_true_sex = [post_target_sex(i) for i in decollate_batch(y_sex, detach=False)]
        y_pred_sex_bin = [post_discrete_sex(i) for i in decollate_batch(pred_sex_probs, detach=False)]
        y_pred_sex_classes = torch.stack([p.argmax(dim=0) for p in y_pred_sex_bin]).cpu().numpy()
        y_true_sex_classes = torch.stack([t.argmax(dim=0) for t in y_true_sex]).cpu().numpy()
        # AGE GROUPS
        pred_age_probs = post_pred(pred_age)
        y_true_age = [post_target_age(i) for i in decollate_batch(y_age, detach=False)]
        y_pred_age_bin = [post_discrete_age(i) for i in decollate_batch(pred_age_probs, detach=False)]
        y_pred_age_classes = torch.stack([p.argmax(dim=0) for p in y_pred_age_bin]).cpu().numpy()
        y_true_age_classes = torch.stack([t.argmax(dim=0) for t in y_true_age]).cpu().numpy()

        # Compute metrics separately
        # SEX
        auc_metric_sex.reset()
        auc_metric_sex(pred_sex_probs, y_true_sex)
        auc_sex = auc_metric_sex.aggregate().item()
        acc_sex = accuracy_score(y_true_sex_classes, y_pred_sex_classes)
        f1_sex = f1_score(y_true_sex_classes, y_pred_sex_classes, average='macro', zero_division=0)
        recall_sex = recall_score(y_true_sex_classes, y_pred_sex_classes, average='macro', zero_division=0)
        precision_sex = precision_score(y_true_sex_classes, y_pred_sex_classes, average='macro', zero_division=0)
        print(f"Epoch {epoch+1} VAL: SEX_ACC={acc_sex:.4f}, SEX_F1={f1_sex:.4f}, SEX_AUC={auc_sex:.4f}, SEX_REC={recall_sex:.4f}, SEX_PREC={precision_sex:.4f}")
        # AGE 
        auc_metric_age.reset()
        auc_metric_age(pred_age_probs, y_true_age)
        auc_age = auc_metric_age.aggregate().item()
        acc_age = accuracy_score(y_true_age_classes, y_pred_age_classes)
        f1_age = f1_score(y_true_age_classes, y_pred_age_classes, average='macro', zero_division=0)
        recall_age = recall_score(y_true_age_classes, y_pred_age_classes, average='macro', zero_division=0)
        precision_age = precision_score(y_true_age_classes, y_pred_age_classes, average='macro', zero_division=0)
        print(f"Epoch {epoch+1} VAL: AGE_ACC={acc_age:.4f}, AGE_F1={f1_age:.4f}, AGE_AUC={auc_age:.4f}, AGE_REC={recall_age:.4f}, AGE_PREC={precision_age:.4f}")

        # AVERAGE IMPORTANT METRICS 
        avg_f1 = (f1_sex + f1_age) / 2
        avg_auc = (auc_sex + auc_age) / 2
        # Scheduler step
        scheduler.step(1 - avg_f1)

        if avg_f1 > best_f1 or (avg_f1 == best_f1 and avg_auc > best_auc):
            best_f1 = avg_f1
            best_auc = avg_auc
            best_metrics = {
                "train_loss": total_loss,
                "val_f1_sex": f1_sex, "val_acc_sex": acc_sex, "val_recall_sex": recall_sex, "val_precision_sex": precision_sex, "val_auc_sex": auc_sex,
                "val_f1_age": f1_age, "val_acc_age": acc_age, "val_recall_age": recall_age, "val_precision_age": precision_age, "val_auc_age": auc_age
            }
            # save state of model as the best state. Also works or should i do model.backbone.state_dict()?
            best_model_state = model.state_dict()

    return best_metrics, best_model_state

def evaluate_on_test(best_model_state, test_df, num_classes_dict, target_maps):
    print("Evaluating on TEST ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskDenseNet(num_classes_dict["sex"], num_classes_dict["age_group"]).to(device)
    model.load_state_dict(best_model_state) # does this work or needs to be loaded into the backbone?
    model.eval()

    test_ds = Dataset(data=build_monai_data(test_df, columns["real_img_path"]), transform=get_eval_transform())
    test_loader = DataLoader(test_ds, batch_size=training["batch_size"], num_workers=2) # changed batch size from 1 to param batch size
    
    auc_metric_sex = ROCAUCMetric(average="macro")
    auc_metric_age = ROCAUCMetric(average="macro")
    post_pred = Compose([Activations(softmax=True)])
    post_target_sex = Compose([AsDiscrete(to_onehot=num_classes_dict["sex"])])
    post_discrete_sex = AsDiscrete(argmax=True, to_onehot=num_classes_dict["sex"])
    post_target_age = Compose([AsDiscrete(to_onehot=num_classes_dict["age_group"])])
    post_discrete_age = AsDiscrete(argmax=True, to_onehot=num_classes_dict["age_group"])

    pred_sex = torch.tensor([], dtype=torch.float32, device=device)
    y_sex = torch.tensor([], dtype=torch.long, device=device)
    pred_age = torch.tensor([], dtype=torch.float32, device=device)
    y_age = torch.tensor([], dtype=torch.long, device=device)
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            x = batch["img"].to(device)
            target_sex = batch["sex_target"].to(device).long()
            target_age = batch["age_target"].to(device).long()
            output_sex, output_age = model(x)
            pred_sex = torch.cat([pred_sex, output_sex], dim=0)
            y_sex = torch.cat([y_sex, target_sex], dim=0)
            pred_age = torch.cat([pred_age, output_age], dim=0)
            y_age = torch.cat([y_age, target_age], dim=0)

    # Post-processing directly on the full batch
    # SEX
    pred_sex_probs = post_pred(pred_sex)
    y_true_sex = [post_target_sex(i) for i in decollate_batch(y_sex, detach=False)]
    y_pred_sex_bin = [post_discrete_sex(i) for i in decollate_batch(pred_sex_probs, detach=False)]
    y_pred_sex_classes = torch.stack([p.argmax(dim=0) for p in y_pred_sex_bin]).cpu().numpy()
    y_true_sex_classes = torch.stack([t.argmax(dim=0) for t in y_true_sex]).cpu().numpy()
    # AGE GROUPS
    pred_age_probs = post_pred(pred_age)
    y_true_age = [post_target_age(i) for i in decollate_batch(y_age, detach=False)]
    y_pred_age_bin = [post_discrete_age(i) for i in decollate_batch(pred_age_probs, detach=False)]
    y_pred_age_classes = torch.stack([p.argmax(dim=0) for p in y_pred_age_bin]).cpu().numpy()
    y_true_age_classes = torch.stack([t.argmax(dim=0) for t in y_true_age]).cpu().numpy()

    # Compute metrics separately
    # SEX
    auc_metric_sex.reset()
    auc_metric_sex(pred_sex_probs, y_true_sex)
    auc_sex = auc_metric_sex.aggregate().item()
    acc_sex = accuracy_score(y_true_sex_classes, y_pred_sex_classes)
    f1_sex = f1_score(y_true_sex_classes, y_pred_sex_classes, average='macro', zero_division=0)
    recall_sex = recall_score(y_true_sex_classes, y_pred_sex_classes, average='macro', zero_division=0)
    precision_sex = precision_score(y_true_sex_classes, y_pred_sex_classes, average='macro', zero_division=0)
    print(f"TEST: SEX_ACC={acc_sex:.4f}, SEX_F1={f1_sex:.4f}, SEX_AUC={auc_sex:.4f}, SEX_REC={recall_sex:.4f}, SEX_PREC={precision_sex:.4f}")
    print("=== SEX CLASSIFICATION REPORT ===")
    print(classification_report(y_true_sex_classes, y_pred_sex_classes,
        target_names=list(target_maps["sex"].values()),  # Will show 'F' and 'M'
        digits=4, zero_division=0
    ))

    # AGE 
    auc_metric_age.reset()
    auc_metric_age(pred_age_probs, y_true_age)
    auc_age = auc_metric_age.aggregate().item()
    acc_age = accuracy_score(y_true_age_classes, y_pred_age_classes)
    f1_age = f1_score(y_true_age_classes, y_pred_age_classes, average='macro', zero_division=0)
    recall_age = recall_score(y_true_age_classes, y_pred_age_classes, average='macro', zero_division=0)
    precision_age = precision_score(y_true_age_classes, y_pred_age_classes, average='macro', zero_division=0)
    print(f"TEST: AGE_ACC={acc_age:.4f}, AGE_F1={f1_age:.4f}, AGE_AUC={auc_age:.4f}, AGE_REC={recall_age:.4f}, AGE_PREC={precision_age:.4f}")
    print("=== AGE GROUP CLASSIFICATION REPORT ===")
    print(classification_report(y_true_age_classes, y_pred_age_classes,
        target_names=list(target_maps["age_group"].values()),  # Will show real age bin labels
        digits=4, zero_division=0
    ))

    return {"test_f1_sex": f1_sex, "test_acc_sex": acc_sex, "test_auc_sex": auc_sex, "test_recall_sex": recall_sex, "test_precision_sex": precision_sex,
            "test_f1_age": f1_age, "test_acc_age": acc_age, "test_auc_age": auc_age, "test_recall_age": recall_age, "test_precision_age": precision_age}


# ========== Run Experiments ==========
for i in range(0, len(syn_paths)):
    print(f"\n=== Syn set {i+1} ===")

    # load and process synthetic set
    print("Reading synthetic set metadata ...")
    syn_df = pd.read_csv(syn_paths[i])
    print(f"Size syn set {i+1}: {len(syn_df)}")
    
    # prepare each target column as category codes encoded into integers
    for col in target_cols:
        if normalize_target:
            syn_df[col] = syn_df[col].astype("category").cat.codes

    # split syn_df into train-val
    syn_df["stratify_key"] = syn_df[target_cols].astype(str).agg("_".join, axis=1)
    train_df, val_df = train_test_split(syn_df, train_size=split["train_size"], stratify=syn_df["stratify_key"], random_state=42)

    # train the model on the syntehtic set
    train_metrics, best_model_state = train_model(train_df, val_df, syn_set_num=i+1, num_classes_dict=num_classes_dict)

    # evaluate trained model on the real test data
    test_metrics = evaluate_on_test(best_model_state, test_df, num_classes_dict, target_maps)
    # save results
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
    sem = results_array.std(ddof=0) / np.sqrt(len(results_array))
    # 95 % two-sided t-interval with df = K-1
    df = len(results_array) - 1
    t95 = t.ppf(0.975, df)  
    ci = t95 * sem
    return mean.item(), ci.item()

# read the log results file, and compute 95% CI intervals of all test metrics between the 5 synthetic set runs
test_metrics = ["test_f1_sex", "test_acc_sex", "test_auc_sex", "test_recall_sex", "test_precision_sex",
               "test_f1_age", "test_acc_age", "test_auc_age", "test_recall_age", "test_precision_age"]
results_df = pd.read_csv(log_path)
for metric in test_metrics:
    metric_results = results_df[metric].values
    metric_mean, metric_ci   = mean_ci(metric_results)
    print(f"{metric} : {metric_mean:.6f} ± {metric_ci:.6f}  (95 % CI, n={len(syn_paths)})")