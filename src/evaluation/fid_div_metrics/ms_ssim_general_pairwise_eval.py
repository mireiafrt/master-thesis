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
df = pd.read_csv(paths["imgs_csv"])
# For test
df = df[(df["use"] == True)&(df["split"]=="test")]
# generate report
def build_clip_prompt(row):
    sex_term = "female" if row["sex"] == "F" else "male"
    diagnosis_text = "healthy" if row["label"] == 0 else "with COVID-19"
    return f"A {sex_term} patient in the {row['age_group']} age group, {diagnosis_text}."
df["report"] = df.apply(build_clip_prompt, axis=1)
print("Reports created ...")

# Do stratified sampling if dataset has more than 125 images
if len(df) > 125:
    print("Stratified sampling on prompt to 125 images...")
    df, _ = train_test_split(df, train_size=125, stratify=df["report"], random_state=42)

data = [{"image": row[columns["img_path"]]} for _, row in df.iterrows()]
print(f"Size of syn data: {len(data)}")

###### TRANSFORMS ######
common_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image"]),
    transforms.EnsureChannelFirstd(keys=["image"]),
    transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    transforms.Resized(keys=["image"], spatial_size=(256, 256)),
])

ds = Dataset(data=data, transform=common_transforms)
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

###### LOAD IMAGES TO MEMORY ######
print("Caching images to memory...")
image_list = []
for batch in tqdm(loader, desc="Caching"):
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