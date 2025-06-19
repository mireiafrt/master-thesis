import os
import numpy as np
import pandas as pd
import yaml

from itertools import combinations
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

import torch
from monai import transforms
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from generative.metrics import MultiScaleSSIMMetric, SSIMMetric

# Set global seed
set_determinism(42)

# Load config
with open("config/evaluation/ms_ssim_config.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
columns = config["columns"]
prompt_column = "report"

# Load and group data by prompt
print("Reading and grouping data...")
df_syn = pd.read_csv(paths["syn_imgs_1_csv"])
grouped = df_syn.groupby(prompt_column)

# Define common image transform
common_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image"]),
    transforms.EnsureChannelFirstd(keys=["image"]),
    transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    transforms.Resized(keys=["image"], spatial_size=(256, 256)),
])

# Metric setup
device = torch.device("cuda")
ms_ssim_metric = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=11)
ssim_metric = SSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=11)

# Store per-prompt MS-SSIM
prompt_ms_ssim_means = []
prompt_ssim_means = []

print("Computing per-prompt pairwise MS-SSIM and SSIM...")
for prompt, group_df in tqdm(grouped, desc="By Prompt", total=len(grouped)):
    if len(group_df) < 2:
        print(f"Can't compute pairwise scores for <2 images. PROMPT: {prompt}")
        continue

    # Build data dict for this group
    syn_data = [{"image": row[columns["syn_img_path"]]} for _, row in group_df.iterrows()]

    # Build dataset and loader
    syn_ds = Dataset(data=syn_data, transform=common_transforms)
    syn_loader = DataLoader(syn_ds, batch_size=1, shuffle=False, num_workers=4)

    # Cache images
    image_list = []
    for batch in syn_loader:
        image = batch["image"][0]  # shape: [1, H, W]
        image_list.append(image)

    # Pairwise comparisons
    ms_ssim_scores = []
    ssim_scores = []
    n = len(image_list)
    total_pairs = n * (n - 1) // 2
    for i, j in tqdm(combinations(range(n), 2), total=total_pairs, desc="Pairwise"):
        img1 = image_list[i].unsqueeze(0).to(device)
        img2 = image_list[j].unsqueeze(0).to(device)

        ms_ssim_scores.append(ms_ssim_metric(img1, img2).item())
        ssim_scores.append(ssim_metric(img1, img2).item())

    # Aggregate for this prompt
    prompt_ms_ssim_means.append(np.mean(ms_ssim_scores))
    prompt_ssim_means.append(np.mean(ssim_scores))

# Final results
prompt_ms_ssim_means = np.array(prompt_ms_ssim_means)
prompt_ssim_means = np.array(prompt_ssim_means)
print(f"MS-SSIM per-prompt: {prompt_ms_ssim_means.mean():.7f} ± {prompt_ms_ssim_means.std():.7f}")
print(f"SSIM per-prompt:    {prompt_ssim_means.mean():.7f} ± {prompt_ssim_means.std():.7f}")