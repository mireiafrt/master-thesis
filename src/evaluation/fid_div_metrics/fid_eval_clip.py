import os
import numpy as np
import pandas as pd
import yaml

from tqdm import tqdm
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

from monai import transforms
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from generative.metrics import FIDMetric


# set global seed
set_determinism(42)

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

# Sample a subset instead of full data if it is none (to match the sample size that was used for the synthetic data)
if sample_size is not None:
    df_real = df_real.iloc[:sample_size].reset_index(drop=True)
    print(f"Subsampled to {sample_size} rows.")

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
    transforms.Resized(keys=["image"], spatial_size=(224, 224)), # size needed for CLIP
    transforms.Lambdad(keys=["image"], func=lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    # no normalization for CLIP?
])

# Create Datasets and Loaders
real_ds = Dataset(data=real_data, transform=common_transforms)
real_loader = DataLoader(real_ds, batch_size=16, shuffle=False, num_workers=4)
syn_ds = Dataset(data=syn_data, transform=common_transforms)
syn_loader = DataLoader(syn_ds, batch_size=16, shuffle=False, num_workers=4)

# prepare model to impute image features
device = torch.device("cuda")
# load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

####### COMPUTE FEATURES #######
synth_features = []
real_features = []

# loop through batches of both loaders together
for real_batch, syn_batch in tqdm(zip(real_loader, syn_loader), total=len(real_loader)):
    # Convert to PIL
    real_imgs = [transforms.ToPIL()(img.cpu()) for img in real_batch["image"]]
    syn_imgs = [transforms.ToPIL()(img.cpu()) for img in syn_batch["image"]]

    # Process with CLIP
    real_inputs = clip_processor(images=real_imgs, return_tensors="pt", padding=True).to(device)
    syn_inputs = clip_processor(images=syn_imgs, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        real_feats = clip_model.get_image_features(**real_inputs)
        syn_feats = clip_model.get_image_features(**syn_inputs)

    # Normalize embeddings (recommended)
    real_feats = F.normalize(real_feats, dim=-1)
    syn_feats = F.normalize(syn_feats, dim=-1)

    real_features.append(real_feats)
    synth_features.append(syn_feats)

####### COMPUTE FID #######
synth_features = torch.vstack(synth_features)
real_features = torch.vstack(real_features)

fid = FIDMetric()
fid_res = fid(synth_features, real_features)
print(f"FID Score: {fid_res.item():.4f}")