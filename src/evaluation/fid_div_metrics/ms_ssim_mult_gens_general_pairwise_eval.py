import os
import numpy as np
import pandas as pd
import yaml

from itertools import combinations
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import t

import torch
from monai import transforms
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from generative.metrics import MultiScaleSSIMMetric, SSIMMetric


# set global seed
set_determinism(42)

# Load config
with open("config/evaluation/mult_gens_eval.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
columns = config["columns"]
sample_size = config["sample_size"]
filters = config["conditioning"]
prompt_column = "report"

# create syn_paths_array
syn_paths = [paths["syn_1_path"], paths["syn_2_path"], paths["syn_3_path"], paths["syn_4_path"], paths["syn_5_path"]]

###### TRANSFORMS FOR DATA ######
common_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image"]),
    transforms.EnsureChannelFirstd(keys=["image"]),
    transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    transforms.Resized(keys=["image"], spatial_size=(256, 256)),
])

# prepare model to impute image features
device = torch.device("cuda")

# metrics init and arrays to store for each syntehtic set
ms_ssim_sets  = []   # list of 1-D tensors, one per syn set (each 1D tensor has size N images)
ssim_sets     = []

for csv_path in syn_paths:
    print(f"\n=== Processing {Path(csv_path).name} ===")
    df_syn = pd.read_csv(csv_path)

    # Do stratified sampling if dataset has more than 125 images
    if len(df_syn) > 125:
        print("Stratified sampling on prompt to 125 images...")
        df_syn, _ = train_test_split(df_syn, train_size=125, stratify=df_syn["report"], random_state=42)

    # ---------- load every image in the set ----------
    data = [{"image": row[columns["syn_img_path"]]} for _, row in df_syn.iterrows()]
    imgs = []

    for b in DataLoader(Dataset(data, common_transforms), batch_size=1, num_workers=0, pin_memory=False):
        imgs.append(b["image"][0])         # each is C×H×W  (on CPU)

    n = len(imgs)
    print(f"Total images  : {n}")
    print(f"Total pairs   : {n*(n-1)//2}")

    # ---------- pairwise on-GPU, collect scalar scores ----------
    ms_vals, ssim_vals = [], []
    # set new metric compute wrappers for every set
    ms_ssim = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=11)
    ssim    = SSIMMetric        (spatial_dims=2, data_range=1.0, kernel_size=11)

    for i, j in tqdm(combinations(range(n), 2), total=n*(n-1)//2, desc="Pairwise"):
        x = imgs[i].unsqueeze(0).to(device)
        y = imgs[j].unsqueeze(0).to(device)

        ms_vals.append(ms_ssim(x, y))
        ssim_vals.append(ssim(x, y))

    ms_ssim_sets.append(torch.stack(ms_vals))   # shape (num_pairs,)
    ssim_sets   .append(torch.stack(ssim_vals))


# ------------- CI helper unchanged -------------
# Confidence-interval function
def mean_ci(tensor_list):
    per_set_mean = torch.stack([t.mean() for t in tensor_list])
    print("Means per set:", per_set_mean)
    mean = per_set_mean.mean()
    sem  = per_set_mean.std(unbiased=False) / (len(tensor_list) ** 0.5)
    df   = len(tensor_list) - 1
    ci   = t.ppf(0.975, df) * sem
    return mean.item(), ci.item()

# compute CIs
ms_mean,  ms_ci   = mean_ci(ms_ssim_sets)
ssim_mean, ssim_ci = mean_ci(ssim_sets)

print(f"\nGeneral pairwise MS-SSIM : {ms_mean:.6f} ± {ms_ci:.6f}  (95 % CI, n={len(syn_paths)})")
print(f"General pairwise SSIM    : {ssim_mean:.6f} ± {ssim_ci:.6f}  (95 % CI, n={len(syn_paths)})")