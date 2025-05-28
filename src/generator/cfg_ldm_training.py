import os
import shutil
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from monai import transforms
from monai.data import CacheDataset, DataLoader, Dataset
from monai.utils import set_determinism
from generative.inferers import DiffusionInferer, LatentDiffusionInferer
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler


# set global seed
set_determinism(42)

# Load config
with open("config/generator/autoencoder_train.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
columns = config["columns"]
training = config["training"]

os.makedirs(paths["model_output"], exist_ok=True)

############ PREPARE TRAIN AND VAL DATA #############
print("Reading metadata ...")
metadata = pd.read_csv(paths["metadata_csv"])
metadata = metadata[metadata["use"] == True]
# use the test data to create a train-val split (same as autoencoder)
test_df = metadata[metadata["split"] == "test"]

# create new class column by combining the attributes and label into a new target class
test_df["combined"] = test_df[[columns["label"]] + columns["attribute_cols"]].astype(str).agg("_".join, axis=1)
# Factorize: map each unique string to an integer from 0 to 19
test_df["class"] = pd.factorize(test_df["combined"])[0]
print(f"Number of classes: {test_df['class'].nunique()}")

# keep track of the mapping for later in inference
_, class_map = pd.factorize(test_df["combined"])
# Create forward and reverse mappings
forward_mapping = {i: val for i, val in enumerate(class_map)}
reverse_mapping = {val: i for i, val in forward_mapping.items()}
# Combine into one dictionary
full_mapping = {"forward": forward_mapping, "reverse": reverse_mapping}
# Save to model output path
mapping_path = os.path.join(paths["model_output"], "class_mapping.json")
with open(mapping_path, "w") as f:
    json.dump(full_mapping, f, indent=2)
print(f"Saved mapping to: {mapping_path}")

# split set into 80-20 train-val
train, val = train_test_split(test_df, train_size=0.8, stratify=test_df[columns["label"]], random_state=42)
# prepare the data for the autoencoder model
train_data = [{"image": row[columns["image_path"]], "class": row["class"]} for _, row in train.iterrows()]
val_data = [{"image": row[columns["image_path"]], "class": row["class"]} for _, row in val.iterrows()]

# Transforms for the data 
train_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image"], ensure_channel_first=True),
    transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    transforms.RandAffined(
        keys=["image"],                               # Apply to the "image" key in the input dictionary
        rotate_range=[(-np.pi/36, np.pi/36), (-np.pi/36, np.pi/36)],  # Rotation angle range (in radians) for each 2D axis
        translate_range=[(-1, 1), (-1, 1)],           # Max translation in pixels along x and y
        scale_range=[(-0.05, 0.05), (-0.05, 0.05)],   # Scale factor range — ±5% random zoom
        spatial_size=[64, 64],                        # Final output size (crop or pad to this size)
        padding_mode="zeros",                         # Fill value for areas outside original image
        prob=0.5                                      # Apply this transform 50% of the time
    ),
    transforms.RandLambdad(keys=["class"], prob=0.15, func=lambda x: -1 * torch.ones_like(x)), # with 15% add unconditional class
    transforms.Lambdad(
        keys=["class"], func=lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    ), # conditioning variable need to have the format (batch_size, 1, cross_attention_dim)
])
val_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"], ensure_channel_first=True),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        transforms.Lambdad(
            keys=["class"], func=lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        ), # conditioning variable need to have the format (batch_size, 1, cross_attention_dim)
    ]
)

train_ds = Dataset(data=train_data, transform=train_transforms)
val_ds = Dataset(data=val_data, transform=val_transforms)
train_loader = DataLoader(train_ds, batch_size=training["batch_size"], shuffle=True, num_workers=4, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=training["batch_size"], shuffle=True, num_workers=4, persistent_workers=True)
