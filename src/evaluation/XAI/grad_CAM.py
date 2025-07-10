import os
import torch
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from monai.data import Dataset, decollate_batch
from monai.transforms import Compose, LoadImaged, ScaleIntensityd, NormalizeIntensityd, LambdaD, Resized, Activations, AsDiscrete
from monai.losses import FocalLoss
from monai.utils import set_determinism
from monai.networks.nets import DenseNet121
from monai.metrics import ROCAUCMetric
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from torch.utils.data import DataLoader
from monai.visualize import GradCAM  #model for the XAI
import matplotlib.pyplot as plt
import random
from scipy.stats import t
import csv

# ========== Load Config ==========
with open("config/evaluation/xai.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
model_dir = paths["trained_models"]
columns = config["columns"]
batch_size = config["batch_size"]
num_models = config["num_models"]
output = config["output"]

models = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".pth")]
models.sort() # get the models sorted by number

# make sure output exists
os.makedirs(os.path.join(output["results_dir"], "images"), exist_ok=True)

os.makedirs(output["results_dir"], exist_ok=True)
csv_columns = ["run", "set", "subgroup", "test_f1", "test_acc", "test_auc", "test_recall", "test_precision"]
log_path = os.path.join(output["results_dir"], "results.csv")
log_file_exists = os.path.exists(log_path)
log_f = open(log_path, "a", newline="")
writer = csv.DictWriter(log_f, fieldnames=csv_columns)
if not log_file_exists:
    writer.writeheader()

print("Reading metadata ...")
df = pd.read_csv(paths["imgs_csv"])
df = df[(df["use"] == True)]
df = df[df[columns["split_col"]] == columns["split_val"]]

reference_combinations = df[["sex", "age_group"]].drop_duplicates().values.tolist()

def build_monai_data(df_subset, image_col, target_col):
    return [{"img": row[image_col], "target": int(row[target_col])} for _, row in df_subset.iterrows()]

def get_eval_transform():
    return Compose([
        LoadImaged(keys=["img"], ensure_channel_first=True),
        ScaleIntensityd(keys=["img"]),
        LambdaD(keys=["img"], func=lambda x: x.repeat(3, 1, 1)),
        NormalizeIntensityd(keys=["img"], subtrahend=torch.tensor([0.485, 0.456, 0.406]),
                            divisor=torch.tensor([0.229, 0.224, 0.225]), channel_wise=True),
        Resized(keys=["img"], spatial_size=(256, 256)),
    ])

def get_gradcam_overlay(model, image_path, label, class_idx, layer="features.denseblock4.denselayer16"):
    device = next(model.parameters()).device
    sample = {"img": image_path, "target": label}
    transform = get_eval_transform()
    data = transform(sample)
    image_tensor = data["img"].unsqueeze(0).to(device)

    cam = GradCAM(nn_module=model, target_layers=layer)
    cam_map = cam(image_tensor, class_idx=class_idx).squeeze().cpu().numpy()
    cam_map -= cam_map.min()
    cam_map /= cam_map.max()

    img_np = data["img"].cpu().numpy().squeeze()
    img_gray = img_np.mean(axis=0)

    return img_gray, cam_map  # return both grayscale base and cam overlay

def evaluate_on_test(model, test_df, image_path_col):
    device = next(model.parameters()).device
    test_ds = Dataset(data=build_monai_data(test_df, image_path_col, columns["target"]), transform=get_eval_transform())
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=2)

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

    y_pred_probs = post_pred(y_pred)
    y_true = [post_target(i) for i in decollate_batch(y, detach=False)]
    y_pred_bin = [post_discrete(i) for i in decollate_batch(y_pred_probs, detach=False)]
    y_pred_classes = torch.stack([p.argmax(dim=0) for p in y_pred_bin]).cpu().numpy()
    y_true_classes = torch.stack([t.argmax(dim=0) for t in y_true]).cpu().numpy()

    labels_present = np.unique(y_true_classes)

    if len(labels_present) < 2:
        auc = np.nan
    else:
        auc_metric.reset()
        auc_metric(y_pred_probs, y_true)
        auc = auc_metric.aggregate().item()

    acc = accuracy_score(y_true_classes, y_pred_classes)
    f1 = f1_score(y_true_classes, y_pred_classes, labels=labels_present, average='macro', zero_division=0)
    recall = recall_score(y_true_classes, y_pred_classes, labels=labels_present, average='macro', zero_division=0)
    precision = precision_score(y_true_classes, y_pred_classes, labels=labels_present, average='macro', zero_division=0)

    print(f"TEST: ACC={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, REC={recall:.4f}, PREC={precision:.4f}")
    print(classification_report(y_true_classes, y_pred_classes,digits=4, zero_division=0))

    metrics = {"test_f1": f1, "test_acc": acc, "test_auc": auc, "test_recall": recall, "test_precision": precision}

    result_df = test_df.copy()
    result_df["y_true"] = y_true_classes
    result_df["y_pred"] = y_pred_classes
    return metrics, result_df

def eval_per_groups(model_num, set_name, writer, reference_combinations, best_model_path, test_df, image_path_col):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=2, pretrained=True).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    for sex, age_group in reference_combinations:
        safe_age = age_group.replace(" ", "_")
        sub_name = f"{sex}-{age_group}"
        sub_df = test_df[(test_df["sex"] == sex) & (test_df["age_group"] == age_group)]
        print(f"Evaluating on subgroup: {sex} and {age_group}")
        test_metrics, result_df = evaluate_on_test(model, sub_df, image_path_col)
        writer.writerow({"run": model_num, "set": set_name, "subgroup": sub_name, **test_metrics})
        log_f.flush()

        overlay_dict = {}
        for category, condition in {
            "TP": lambda df: (df.y_true == 1) & (df.y_pred == 1),
            "TN": lambda df: (df.y_true == 0) & (df.y_pred == 0),
            "FP": lambda df: (df.y_true == 0) & (df.y_pred == 1),
            "FN": lambda df: (df.y_true == 1) & (df.y_pred == 0),
        }.items():
            sample_row = result_df[condition(result_df)]
            print(f"{category}: found {len(sample_row)} sample(s)")
            if len(sample_row) > 0:
                sample = sample_row.sample(1).iloc[0]
                image_path = sample[image_path_col]
                true_label = sample["y_true"]
                pred_label = sample["y_pred"]

                base_img, cam = get_gradcam_overlay(model, image_path, true_label, pred_label)
                overlay_dict[category] = (base_img, cam)

        # Plot 2x2 grid if we have overlays
        if overlay_dict:
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            fig.suptitle(f"{sex} - {age_group}", fontsize=16)
            categories = ["TP", "TN", "FP", "FN"]
            for i, category in enumerate(categories):
                row, col = divmod(i, 2)
                ax = axs[row][col]
                # if category was found in the subgroup plot it
                if category in overlay_dict:
                    base_img, cam = overlay_dict[category]
                    ax.imshow(base_img, cmap="gray")
                    ax.imshow(cam, cmap="jet", alpha=0.5)
                    ax.set_title(category)
                # else empty plot
                else:
                    ax.axis("off")
                    ax.set_title(f"{category} (N/A)")
            for ax_row in axs:
                for ax in ax_row:
                    ax.axis("off")

            save_path = os.path.join(
                output["results_dir"],
                f"images/{set_name}_GRID_{sex}_{safe_age}_run{model_num}.png"
            )
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()


# ========== Run Experiments ==========
for j in range(0, num_models):
    set_name = columns['split_val']

    model_state = models[j]
    print(f"Using Trained model {model_state}")

    print(f"Evaluating on SET: {set_name} ...")
    eval_per_groups(j+1, set_name, writer, reference_combinations, model_state, df, columns["img_path"])

log_f.close()
print("\nAll runs completed and logged.")