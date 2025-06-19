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

########## PREPARE PROMPTS and RESULT CSV ##########
print("Reading data...")
# Read synthetic data 1
df_syn_1 = pd.read_csv(paths["syn_imgs_1_csv"])
# Read synthetic data 2
df_syn_2 = pd.read_csv(paths["syn_imgs_2_csv"])

# Create data dictionaries
syn_data_1 = [{"image": row[columns["syn_img_path"]]} for _, row in df_syn_1.iterrows()]
syn_data_2 = [{"image": row[columns["syn_img_path"]]} for _, row in df_syn_2.iterrows()]
print(f"Size of syn data 1: {len(syn_data_1)}")
print(f"Size of syn data 2: {len(syn_data_2)}")
assert len(df_syn_1) == len(df_syn_2), "Synthetic datasets must have the same length for pairwise comparison"

###### TRANSFORMS FOR DATA ######
common_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image"]),
    transforms.EnsureChannelFirstd(keys=["image"]),
    transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    transforms.Resized(keys=["image"], spatial_size=(256, 256)),
])

syn_ds_1 = Dataset(data=syn_data_1, transform=common_transforms)
syn_loader_1 = DataLoader(syn_ds_1, batch_size=16, shuffle=False, num_workers=4)
syn_ds_2 = Dataset(data=syn_data_2, transform=common_transforms)
syn_loader_2 = DataLoader(syn_ds_2, batch_size=16, shuffle=False, num_workers=4)

# prepare model to impute image features
device = torch.device("cuda")

# compute metrics
ms_ssim_recon_scores = []
ssim_recon_scores = []

ms_ssim = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=11) # changed kernel size to 11 (standard), MONAI was using 4 (for small images)
ssim = SSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=11) # changed kernel size to 11 (standard), MONAI was using 4 (for small images)

for step, (syn_batch_1, syn_batch_2) in tqdm(enumerate(zip(syn_loader_1, syn_loader_2)), total=len(syn_loader_1)):
    # Get the real images
    syn_imgs_1 = syn_batch_1["image"].to(device)
    # Get the syn images
    syn_imgs_2 = syn_batch_2["image"].to(device)
    
    # compute metric
    ms_ssim_recon_scores.append(ms_ssim(syn_imgs_1, syn_imgs_2))
    ssim_recon_scores.append(ssim(syn_imgs_1, syn_imgs_2))

ms_ssim_recon_scores = torch.cat(ms_ssim_recon_scores, dim=0)
ssim_recon_scores = torch.cat(ssim_recon_scores, dim=0)
print(f"MS-SSIM Metric: {ms_ssim_recon_scores.mean():.7f} +- {ms_ssim_recon_scores.std():.7f}")
print(f"SSIM Metric: {ssim_recon_scores.mean():.7f} +- {ssim_recon_scores.std():.7f}")