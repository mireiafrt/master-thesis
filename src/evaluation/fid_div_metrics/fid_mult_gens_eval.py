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
from torchvision.models import densenet121, inception_v3, resnet50
from torchvision import transforms as T
import torch.nn.functional as F
import torch.nn as nn

from monai import transforms
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from generative.metrics import FIDMetric


############ HELPER FUNCTIONS FOR RADNET IMPLEMENTATION ############
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
resnet_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image"]),
    transforms.EnsureChannelFirstd(keys=["image"]),
    transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    # transforms for resnet
    transforms.Resized(keys=["image"], spatial_size=(224, 224)),
    transforms.Lambdad(keys=["image"], func=lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.NormalizeIntensityd(
        keys=["image"],
        subtrahend=torch.tensor([0.485, 0.456, 0.406]),
        divisor=torch.tensor([0.229, 0.224, 0.225]),
        channel_wise=True,
    ),
])
radnet_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image"]),
    transforms.EnsureChannelFirstd(keys=["image"]),
    transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    transforms.Resized(keys=["image"], spatial_size=(256, 256)),
])
densenet_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image"]),
    transforms.EnsureChannelFirstd(keys=["image"]),
    transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    # transforms for densenet
    transforms.Resized(keys=["image"], spatial_size=(224, 224)),
    transforms.Lambdad(keys=["image"], func=lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.NormalizeIntensityd(
        keys=["image"],
        subtrahend=torch.tensor([0.485, 0.456, 0.406]),
        divisor=torch.tensor([0.229, 0.224, 0.225]),
        channel_wise=True,
    ),
])
inception_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image"]),
    transforms.EnsureChannelFirstd(keys=["image"]),
    transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    # transforms for inception model
    transforms.Resized(keys=["image"], spatial_size=(299, 299)),
    transforms.Lambdad(keys=["image"], func=lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.NormalizeIntensityd(keys=["image"], subtrahend=0.5, divisor=0.5),
])

# Create real data loader
real_data = [{"image": row[columns["real_img_path"]]} for _, row in df_real.iterrows()]
all_real_loaders = {
    "resnet": DataLoader(Dataset(real_data, transform=resnet_transforms), batch_size=16, shuffle=False, num_workers=4),
    "densenet": DataLoader(Dataset(real_data, transform=densenet_transforms), batch_size=16, shuffle=False, num_workers=4),
    "inception": DataLoader(Dataset(real_data, transform=inception_transforms), batch_size=16, shuffle=False, num_workers=4),
    "radnet": DataLoader(Dataset(real_data, transform=radnet_transforms), batch_size=16, shuffle=False, num_workers=4),
}
print(f"Size of real data: {len(real_data)}")

# create syn data loaderS
all_syn_loaders = {"resnet": [], "densenet": [], "inception": [], "radnet": []}
for i in range(len(syn_paths)):
    df_syn = pd.read_csv(syn_paths[i])
    syn_data = [{"image": row[columns["syn_img_path"]]} for _, row in df_syn.iterrows()]
    print(f"Size of syn data {i+1}: {len(syn_data)}")

    all_syn_loaders["resnet"].append(
        DataLoader(Dataset(syn_data, transform=resnet_transforms), batch_size=16, shuffle=False, num_workers=4)
    )
    all_syn_loaders["densenet"].append(
        DataLoader(Dataset(syn_data, transform=densenet_transforms), batch_size=16, shuffle=False, num_workers=4)
    )
    all_syn_loaders["inception"].append(
        DataLoader(Dataset(syn_data, transform=inception_transforms), batch_size=16, shuffle=False, num_workers=4)
    )
    all_syn_loaders["radnet"].append(
        DataLoader(Dataset(syn_data, transform=radnet_transforms), batch_size=16, shuffle=False, num_workers=4)
    )

###### LOAD MODELS ######
device = torch.device("cuda")

# Load ResNet50 and remove classification head
resnet = resnet50(pretrained=True)
resnet.fc = nn.Identity()  # Remove classification layer to get features
resnet.eval().to(device)

# Load radnet pretrained from wavWarvitoito repo
radnet = torch.hub.load("Warvito/radimagenet-models", model="radimagenet_resnet50", trust_repo=True, verbose=True)
radnet.to(device).eval()

# Load DenseNet121 and remove classification head
densenet = densenet121(pretrained=True)
densenet.classifier = nn.Identity()  # Remove final classifier
densenet.eval().to(device)

# Load InceptionV3
inception = inception_v3(pretrained=True, transform_input=False)  # aux_logits will default to True
inception.fc = torch.nn.Identity()  # remove final classification head
inception.eval().to(device)

###### SET UP SCORES & ARRAYS TO STORE ######
fid_scores = {"resnet": [], "densenet": [], "inception": [], "radnet": []}

# help function per syntehtic set
def compute_pair_metrics(real_loader, syn_loader, device, model_name):
    real_feats, syn_feats = [], []

    for step, (real_batch, syn_batch) in tqdm(enumerate(zip(real_loader, syn_loader)), total=len(real_loader)):
        real = real_batch["image"].to(device)
        syn  = syn_batch["image"].to(device)

        # Get the features and store them in the appropiate arrays
        with torch.no_grad():
            if model_name == "resnet":
                real_feats.append(resnet(real))
                syn_feats.append(resnet(syn))
            if model_name == "densenet":
                real_feats.append(densenet(real))
                syn_feats.append(densenet(syn))
            if model_name == "inception":
                real_feats.append(inception(real))
                syn_feats.append(inception(syn))
            if model_name == "radnet":
                real_feats.append(get_features(real))
                syn_feats.append(get_features(syn))

    # stack featues
    real_feats = torch.vstack(real_feats)
    syn_feats = torch.vstack(syn_feats)

    # calculate metrics
    fid = FIDMetric()
    fid = fid(syn_feats, real_feats)
    return fid

# compute the metrics for each synthetic set (loader)
for model_name in fid_scores:
    real_loader = all_real_loaders[model_name]
    for syn_loader in all_syn_loaders[model_name]:
        fid_value = compute_pair_metrics(real_loader, syn_loader, device, model_name)
        fid_scores[model_name].append(fid_value)

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

# calculate CIs for each metric
for model_name in fid_scores:
    fids_tensors = fid_scores[model_name]
    fid_mean, fid_ci = mean_ci(fids_tensors)
    print(f"{model_name} FID: {fid_mean:.6f} ± {fid_ci:.6f}  (95 % CI, n={len(syn_paths)})")
