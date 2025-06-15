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

from generative.metrics import FIDMetric


# set global seed
set_determinism(42)

############ HELPER FUNCTIONS ############
def subtract_mean(x: torch.Tensor) -> torch.Tensor:
    mean = [0.406, 0.456, 0.485]
    x[:, 0, :, :] -= mean[0]
    x[:, 1, :, :] -= mean[1]
    x[:, 2, :, :] -= mean[2]
    return x


def spatial_average(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3], keepdim=keepdim)


def get_features(image):
    # If input has just 1 channel, repeat channel to have 3 channels
    if image.shape[1]:
        image = image.repeat(1, 3, 1, 1)

    # Change order from 'RGB' to 'BGR'
    image = image[:, [2, 1, 0], ...]

    # Subtract mean used during training
    image = subtract_mean(image)

    # Get model outputs
    with torch.no_grad():
        feature_image = radnet.forward(image)
        # flattens the image spatially
        feature_image = spatial_average(feature_image, keepdim=False)

    return feature_image

# Load config
with open("config/evaluation/fid_config.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
columns = config["columns"]
sample_size = config["sample_size"]
filters = config["conditioning"]

########## PREPARE PROMPTS and RESULT CSV ##########
print("Reading data...")
# Read real data
df_real = pd.read_csv(paths["real_imgs_csv"])
df_real = df_real[df_real["use"] == True]
# Read synthetic data
df_syn = pd.read_csv(paths["syn_imgs_csv"])

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

# Create data dictionaries
real_data = [{"image": row[columns["real_img_path"]]} for _, row in df_real.iterrows()]
syn_data = [{"image": row[columns["syn_img_path"]]} for _, row in df_syn.iterrows()]
print(f"Size of real data: {len(real_data)}")
print(f"Size of syn data: {len(syn_data)}")

###### TRANSFORMS FOR DATA ######
common_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image"]),
    transforms.EnsureChannelFirstd(keys=["image"]),
    transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    transforms.Resized(keys=["image"], spatial_size=(256, 256)),
])
rea_ds = Dataset(data=real_data, transform=common_transforms)
real_loader = DataLoader(rea_ds, batch_size=16, shuffle=True, num_workers=4)
syn_ds = Dataset(data=syn_data, transform=common_transforms)
syn_loader = DataLoader(syn_ds, batch_size=16, shuffle=True, num_workers=4)

# prepare model to impute image features
device = torch.device("cuda")
radnet = torch.hub.load("Warvito/radimagenet-models", model="radimagenet_resnet50", trust_repo=True, verbose=True)
radnet.to(device)
radnet.eval()

####### COMPUTE FEATURES #######
synth_features = []
real_features = []

# loop through batches of both loaders together
for step, (real_batch, syn_batch) in enumerate(zip(real_loader, syn_loader)):
    # Get the real images
    real_images = real_batch["image"].to(device)
    # Get the syn images
    syn_images = syn_batch["image"].to(device)

    # Get the features for the real data
    real_eval_feats = get_features(real_images)
    real_features.append(real_eval_feats)

    # Get the features for the synthetic data
    synth_eval_feats = get_features(syn_images)
    synth_features.append(synth_eval_feats)

####### COMPUTE FID #######
synth_features = torch.vstack(synth_features)
real_features = torch.vstack(real_features)

fid = FIDMetric()
fid_res = fid(synth_features, real_features)
print(f"FID Score: {fid_res.item():.4f}")