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
with open("config/evaluation/mult_gens_eval.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
syn_paths = config["exp_names"]
columns = config["columns"]
sample_size = config["sample_size"]
filters = config["conditioning"]

# create syn_paths_array
syn_paths = [paths["syn_1_path"], paths["syn_2_path"], paths["syn_3_path"], paths["syn_4_path"], paths["syn_5_path"]]

########## PREPARE PROMPTS and RESULT CSV ##########
print("Reading data...")
# Read real data
df_real = pd.read_csv(paths["real_imgs_csv"])
df_real = df_real[df_real["use"] == True]

# filter metadata based on the conditioning (has to match the conditioning of the synthetic data)
print("Filters:", filters)
df_real = df_real.copy()
for key, value in filters.items():
    if value is not None:
        df_real = df_real[df_real[key] == value]
df_real = df_real.reset_index(drop=True)
print("Filtered data...")

# Sample a subset instead of full data if it is none (random, to match the sample size that was used for the synthetic data)
if sample_size is not None:
    df_real = df_real.sample(n=sample_size, random_state=42).reset_index(drop=True)
    print(f"Subsampled to {sample_size} random rows with seed 42.")

###### TRANSFORMS FOR DATA ######
common_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image"]),
    transforms.EnsureChannelFirstd(keys=["image"]),
    transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    transforms.Resized(keys=["image"], spatial_size=(256, 256)),
])

# Create real data loader
real_data = [{"image": row[columns["real_img_path"]]} for _, row in df_real.iterrows()]
rea_ds = Dataset(data=real_data, transform=common_transforms)
real_loader = DataLoader(rea_ds, batch_size=16, shuffle=False, num_workers=4)
print(f"Size of real data: {len(real_data)}")

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

# metric init and array to store for each syntehtic set
mmd_sets = []
mmd = MMDMetric()

# help function per syntehtic set
def compute_pair_metrics(real_loader, syn_loader, mmd, device):
    mmd_vals = []

    for step, (real_batch, syn_batch) in tqdm(enumerate(zip(real_loader, syn_loader)), total=len(real_loader)):
        real = real_batch["image"].to(device)
        syn  = syn_batch["image"].to(device)

        mmd_vals.append(mmd(real, syn).cpu()) # (B,)

    mmd_vals   = torch.cat(mmd_vals)    # (N_images,)
    return mmd_vals

# compute the metrics for each synthetic set (loader)
for syn_loader in syn_loaders:
    mmd_vals = compute_pair_metrics(real_loader, syn_loader,mmd, device)
    mmd_sets.append(mmd_vals)

# function to calculate confidence intervals 
def mean_ci(tensor_list):
    """
    Return mean and 95 % CI across *sets* (not images).
    Each tensor in the list has shape (N_images,).
    """
    # stack: (K, N_images) → per-set means: (K,)
    per_set_mean = torch.stack([t.mean() for t in tensor_list])
    mean = per_set_mean.mean()
    sem  = per_set_mean.std(unbiased=False) / (len(tensor_list) ** 0.5)
    # 95 % two-sided t-interval with df = K-1
    t95 = torch.distributions.studentT.StudentT(df=len(tensor_list)-1).icdf(torch.tensor(0.975))
    ci  = t95 * sem
    return mean.item(), ci.item()

# calculate intervsals 
mmd_mean, mmd_ci   = mean_ci(mmd_sets)
print(f"MMD : {mmd_mean:.6f} ± {mmd_ci:.6f}  (95 % CI, n={len(syn_paths)})")