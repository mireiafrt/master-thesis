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


# set global seed
set_determinism(42)

# Load config
with open("config/evaluation/ms_ssim_config.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
columns = config["columns"]

########## PREPARE DATA ##########
print("Reading data...")
df_syn = pd.read_csv(paths["syn_imgs_1_csv"])
syn_data = [{"image": row[columns["syn_img_path"]]} for _, row in df_syn.iterrows()]
print(f"Size of syn data: {len(syn_data)}")

###### TRANSFORMS ######
common_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image"]),
    transforms.EnsureChannelFirstd(keys=["image"]),
    transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    transforms.Resized(keys=["image"], spatial_size=(256, 256)),
])

syn_ds = Dataset(data=syn_data, transform=common_transforms)
syn_loader = DataLoader(syn_ds, batch_size=1, shuffle=False, num_workers=4)

###### LOAD IMAGES TO MEMORY ######
print("Caching images to memory...")
image_list = []
for batch in tqdm(syn_loader, desc="Caching"):
    image = batch["image"][0]  # shape: [1, H, W]
    image_list.append(image)

###### METRIC SETUP ######
device = torch.device("cuda")
ms_ssim_metric = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=11)
ssim_metric = SSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=11)

###### PAIRWISE COMPARISON ######
print("Computing pairwise MS-SSIM and SSIM...")
ms_ssim_scores = []
ssim_scores = []

n = len(image_list)
total_pairs = n * (n - 1) // 2
for i, j in tqdm(combinations(range(n), 2), total=total_pairs, desc="Pairwise"):
    img1 = image_list[i].unsqueeze(0).to(device)
    img2 = image_list[j].unsqueeze(0).to(device)

    ms_ssim_scores.append(ms_ssim_metric(img1, img2).item())
    ssim_scores.append(ssim_metric(img1, img2).item())

###### RESULTS ######
ms_ssim_scores = np.array(ms_ssim_scores)
ssim_scores = np.array(ssim_scores)
print(f"MS-SSIM Metric: {ms_ssim_scores.mean():.7f} +- {ms_ssim_scores.std():.7f}")
print(f"SSIM Metric: {ssim_scores.mean():.7f} +- {ssim_scores.std():.7f}")