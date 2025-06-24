import os
import numpy as np
import pandas as pd
import yaml

from itertools import combinations
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
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

# create syn_paths_array
syn_paths = [paths["syn_1_path"], paths["syn_2_path"], paths["syn_3_path"], paths["syn_4_path"], paths["syn_5_path"]]

###### TRANSFORMS FOR DATA ######
common_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image"]),
    transforms.EnsureChannelFirstd(keys=["image"]),
    transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    transforms.Resized(keys=["image"], spatial_size=(256, 256)),
])

# create syn data loaderS
syn_loaders = []
for i in range(0, len(syn_paths)):
    df_syn   = pd.read_csv(syn_paths[i])                  
    syn_data = [{"image": row[columns["syn_img_path"]]} for _, row in df_syn.iterrows()]
    print(f"Size of syn data {i+1}: {len(syn_data)}")
    syn_ds   = Dataset(syn_data, transform=common_transforms)
    syn_loaders.append(DataLoader(syn_ds, batch_size=16, shuffle=False, num_workers=4))

# prepare model to impute image features
device = torch.device("cuda")

# metrics init and arrays to store for each syntehtic set
ms_ssim_sets  = []   # list of 1-D tensors, one per syn set (each 1D tensor has size N images)
ssim_sets     = []
ms_ssim = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=11)
ssim    = SSIMMetric        (spatial_dims=2, data_range=1.0, kernel_size=11)

# help function per syntehtic set
def compute_pair_metrics(syn_loader_1, syn_loader_2, ms_ssim, ssim, device):
    ms_vals, ssim_vals = [], []

    for step, (syn_batch_1, syn_batch_2) in tqdm(enumerate(zip(syn_loader_1, syn_loader_2)), total=len(syn_loader_1)):
        syn_1 = syn_batch_1["image"].to(device)
        syn_2  = syn_batch_2["image"].to(device)

        ms_vals.append(ms_ssim(syn_1, syn_2))   # (B,)
        ssim_vals.append(ssim(syn_1, syn_2))   # (B,)

    ms_vals   = torch.cat(ms_vals)    # (N_images,)
    ssim_vals = torch.cat(ssim_vals)  # (N_images,)
    return ms_vals, ssim_vals

# compute the metrics for each synthetic set (loader) pair
for loader1, loader2 in combinations(syn_loaders, 2):   # (n=2) pairs, no repeats
    ms_vals, ssim_vals = compute_pair_metrics(loader1, loader2, ms_ssim, ssim, device)
    ms_ssim_sets.append(ms_vals)
    ssim_sets.append(ssim_vals)

# function to calculate confidence intervals 
def mean_ci(tensor_list):
    """
    Return mean and 95 % CI across *sets* (not images).
    Each tensor in the list has shape (N_images,).
    """
    # stack: (K, N_images) → per-set means: (K,)
    per_set_mean = torch.stack([t.mean() for t in tensor_list])
    print(f"Means per set: {per_set_mean}")
    mean = per_set_mean.mean()
    sem  = per_set_mean.std(unbiased=False) / (len(tensor_list) ** 0.5)
    # 95 % two-sided t-interval with df = K-1
    df = len(tensor_list) - 1
    t95 = t.ppf(0.975, df)  
    ci = t95 * sem
    return mean.item(), ci.item()

# calculate intervsals 
ms_mean, ms_ci   = mean_ci(ms_ssim_sets)
ssim_mean, ssim_ci = mean_ci(ssim_sets)
print(f"MS-SSIM  : {ms_mean:.6f} ± {ms_ci:.6f}  (95 % CI, n={len(ms_ssim_sets)})")
print(f"SSIM     : {ssim_mean:.6f} ± {ssim_ci:.6f}  (95 % CI, n={len(ms_ssim_sets)})")