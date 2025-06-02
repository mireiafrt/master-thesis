import os
import shutil
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from monai import transforms
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler


# set global seed
set_determinism(42)

# Load config
with open("config/generator/generator_train.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
columns = config["columns"]
training = config["training"]

os.makedirs(paths["model_output"], exist_ok=True)
device = torch.device("cuda")

############ PREPARE TRAIN AND VAL DATA #############
print("Reading metadata ...")
metadata = pd.read_csv(paths["metadata_csv"])
metadata = metadata[metadata["use"] == True]
# use the test data to create a train-val split (same as autoencoder)
test_df = metadata[metadata["split"] == "test"]
# split test set into train-val (same split as autoencoder, seed 42 and 80-20)
train, val = train_test_split(test_df, train_size=0.8, stratify=test_df[columns["label"]], random_state=42)

# prepare data dicts for loaders
train_data = [{"image": row[columns["image_path"]]} for _, row in train.iterrows()]
val_data = [{"image": row[columns["image_path"]]} for _, row in val.iterrows()]

# TRANSFORMS
train_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image"], ensure_channel_first=True),
    transforms.ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0, b_max=1, clip=True),
    transforms.RandAffined(
        keys=["image"],                               # Apply to the "image" key in the input dictionary
        rotate_range=[(-np.pi/36, np.pi/36), (-np.pi/36, np.pi/36)],  # Rotation angle range (in radians) for each 2D axis
        translate_range=[(-1, 1), (-1, 1)],           # Max translation in pixels along x and y
        scale_range=[(-0.05, 0.05), (-0.05, 0.05)],   # Scale factor range — ±5% random zoom
        spatial_size=[256, 256],                      # Final output size (crop or pad to this size) --> has to match transform of autoencoder
        padding_mode="zeros",                         # Fill value for areas outside original image
        prob=0.5                                      # Apply this transform 50% of the time
    ),
])
val_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image"], ensure_channel_first=True),
    transforms.ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0, b_max=1, clip=True),
    transforms.Resized(keys=["image"], spatial_size=[256, 256]),  # crop to fixed size to match train transform
])

train_ds = Dataset(data=train_data, transform=train_transforms)
val_ds = Dataset(data=val_data, transform=val_transforms)
train_loader = DataLoader(train_ds, batch_size=training["batch_size"], shuffle=True, num_workers=4, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=training["batch_size"], shuffle=False, num_workers=4, persistent_workers=True)

############## DEFINE MODEL AND ARCHITECTURE ##############
# define the UNET to take for a Latent Diffusion Model with condtional information
unet = DiffusionModelUNet(
    spatial_dims=2,              # 2D CT slices
    in_channels=1,               # 1 greyscale
    out_channels=1,              # same as in_channels
    num_channels=(128, 256, 256),   # Number of channels at each level of the UNet
    attention_levels=(False, True, True), # Whether to apply self-attention at each UNet level
    num_res_blocks=1,             # Number of residual blocks per level in the UNet
    num_head_channels=256,        # Number of channels per attention head at each level (if attention is enabled)
)
unet = unet.to(device)

# set scheduler
scheduler = DDPMScheduler(num_train_timesteps=1000)
# set inferer
inferer = DiffusionInferer(scheduler)
# set optimizer
optimizer = torch.optim.Adam(unet.parameters(), lr=2.5e-5)

# set GradScaler
scaler = GradScaler()

############ TRAIN MODEL ############
# === Training settings ===
n_epochs = training['num_epochs']       # Total number of training epochs
val_interval = training['val_interval'] # Run validation every N epochs
# === Logging ===
epoch_losses = []
val_losses = []

# === Training loop ===
for epoch in range(n_epochs):
    unet.train() # set unet to train
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["image"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            # Add noise and timestep
            noise = torch.randn_like(images).to(device)
            timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device).long()

            # Get model prediction
            noise_pred = inferer(inputs=images, diffusion_model=unet, noise=noise, timesteps=timesteps)

            # Loss
            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
    epoch_losses.append(epoch_loss / (step + 1))

    if (epoch + 1) % val_interval == 0:
        unet.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(val_loader, start=1):
                images = batch["image"].to(device)
                with autocast(enabled=True):
                    noise = torch.randn_like(images).to(device)
                    timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device).long()

                    noise_pred = inferer(inputs=images, diffusion_model=unet, noise=noise, timesteps=timesteps)
                    loss = F.mse_loss(noise_pred.float(), noise.float())

                val_loss += loss.item()
                progress_bar.set_postfix({"val_loss": val_loss / (val_step + 1)})
        val_losses.append(val_loss / (val_step + 1))

        # Sampling a conditional image during training to have visual inspection
        noise = torch.randn((1, 1, 256, 256), device=device) # 1 random noise with 1 channel, 256x256
        scheduler.set_timesteps(num_inference_steps=1000)
        with autocast(enabled=True):
                image = inferer.sample(input_noise=noise, diffusion_model=unet, scheduler=scheduler)
        # save output
        plt.imsave(f"{paths['model_output']}/sample_epoch{epoch}.png", image[0, 0].cpu(), cmap="gray", vmin=0, vmax=1)

progress_bar.close()
print("Finished training")

# save model at the end to model_output + "generator.pth"
output_model_path = os.path.join(paths["model_output"], "generator.pth")
torch.save(unet.state_dict(), output_model_path)
print(f"Saved trained unet model to: {output_model_path}")

# Create a DataFrame to store the losses
loss_df = pd.DataFrame({
    "epoch": list(range(0, n_epochs)),
    "train_loss": epoch_losses,
    "val_recon_loss": [val_losses[i // val_interval] if (i + 1) % val_interval == 0 else None for i in range(n_epochs)]
})
loss_csv_path = os.path.join(paths["model_output"], "training_losses.csv")
loss_df.to_csv(loss_csv_path, index=False)
print(f"Saved training loss log to: {loss_csv_path}")

# Cleanup after training
torch.cuda.empty_cache()
