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

# Loop over the 5 synthetic sets
for csv_path in syn_paths:
    print(f"\n=== Processing {Path(csv_path).name} ===")
    df = pd.read_csv(csv_path)
    grouped = df.groupby(prompt_column)

    prompt_ms_means  = []
    prompt_ssim_means = []
    # set new metric compute wrappers for every set
    ms_ssim = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=11)
    ssim    = SSIMMetric        (spatial_dims=2, data_range=1.0, kernel_size=11)

    for prompt, gdf in tqdm(grouped, desc="By Prompts...", total=len(grouped)):
        if len(gdf) < 2:
            continue                                     # need >1 img per prompt

        # SAMPLE 30 if available
        if len(gdf) >= 30:
            print("Sampling 30 images per prompt for intra-prompt pairwise comparison")
            gdf = gdf.sample(n=30, random_state=42)  # fixed seed for reproducibility

        # load images
        data = [{"image": row[columns["syn_img_path"]]} for _, row in gdf.iterrows()]
        ds   = Dataset(data=data, transform=common_transforms)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

        imgs = [b["image"][0] for b in loader]           # list of C×H×W
        n = len(imgs)
        if n < 2:
            continue

        # pairwise comparisons within group of prompt
        ms_vals, ssim_vals = [], []
        total_pairs = n * (n - 1) // 2
        for i, j in tqdm(combinations(range(n), 2), total=total_pairs, desc="Pairwise prompt"):
            x = imgs[i].unsqueeze(0).to(device)
            y = imgs[j].unsqueeze(0).to(device)
            ms_vals.append(ms_ssim(x, y))
            ssim_vals.append(ssim(x, y))

        prompt_ms_means.append(torch.stack(ms_vals).mean())
        prompt_ssim_means.append(torch.stack(ssim_vals).mean())

    # one tensor per set  →  shape: (num_prompts,)
    ms_ssim_sets.append(torch.stack(prompt_ms_means))
    ssim_sets.append(torch.stack(prompt_ssim_means))

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

print(f"\nPrompt-aware MS-SSIM : {ms_mean:.6f} ± {ms_ci:.6f}  (95 % CI, n={len(syn_paths)})")
print(f"Prompt-aware SSIM    : {ssim_mean:.6f} ± {ssim_ci:.6f}  (95 % CI, n={len(syn_paths)})")