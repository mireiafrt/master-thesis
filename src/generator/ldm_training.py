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
from generative.inferers import DiffusionInferer, LatentDiffusionInferer
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
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

# prepare the data for the autoencoder model
train_data = [{"image": row[columns["image_path"]]} for _, row in train.iterrows()]
val_data = [{"image": row[columns["image_path"]]} for _, row in val.iterrows()]

# TRANSFORMS 
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
])
val_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"], ensure_channel_first=True),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        transforms.Resized(keys=["image"], spatial_size=[64, 64]),  # crop to fixed size to match train transform
    ]
)

train_ds = Dataset(data=train_data, transform=train_transforms)
val_ds = Dataset(data=val_data, transform=val_transforms)
train_loader = DataLoader(train_ds, batch_size=training["batch_size"], shuffle=True, num_workers=4, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=training["batch_size"], shuffle=True, num_workers=4, persistent_workers=True)

############## DEFINE MODEL AND ARCHITECTURE ##############
autoencoderkl = AutoencoderKL(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=(128, 128, 256),
    latent_channels=3,
    num_res_blocks=2,
    attention_levels=(False, False, False),
    with_encoder_nonlocal_attn=False,
    with_decoder_nonlocal_attn=False,
)
autoencoderkl = autoencoderkl.to(device)
autoencoderkl.load_state_dict(torch.load(paths["autoencoder_model"])) # load the trained autoencoder model

# define the UNET to take for a Latent Diffusion Model with condtional information
unet = DiffusionModelUNet(
    spatial_dims=2,              # 2D CT slices
    in_channels=3,               # match AutoencoderKL latent_channels (latent space channels)
    out_channels=3,              # same as in_channels
    num_channels=(128, 256, 512), # Number of channels at each level of the UNet encoder/decoder
    attention_levels=(False, True, True), # Whether to apply self-attention at each UNet level
    num_res_blocks=2,             # Number of residual blocks per level in the UNet
    num_head_channels=(0, 256, 512), # Number of channels per attention head at each level (if attention is enabled)
    with_conditioning=False,       # Enables support for passing external conditioning information (e.g. embeddings)
)
unet = unet.to(device)

# set scheduler
scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta", beta_start=0.0015, beta_end=0.0195)

# setting up scaling_factor from autoencoder z std
with torch.no_grad():
    with autocast(enabled=True):
        z = autoencoderkl.encode_stage_2_inputs(next(iter(train_loader))["image"].to(device))
scale_factor = 1 / torch.std(z)
print(f"Scale factor: {scale_factor}")

# set inferer
inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

# set optimizer
optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)

# set GradScaler
scaler = GradScaler()

############ TRAIN MODEL ############
# === Training settings ===
n_epochs = training['num_epochs']       # Total number of training epochs
val_interval = training['val_interval'] # Run validation every N epochs
guidance_scale = training['guidance_scale']
# === Logging ===
epoch_losses = []
val_losses = []

# === Training loop ===
for epoch in range(n_epochs):
    unet.train() # set unet to train
    autoencoderkl.eval() # set autoencoder to eval so it does not get updated?
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["image"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            # Encode image to latent
            z_mu, z_sigma = autoencoderkl.encode(images)
            z = autoencoderkl.sampling(z_mu, z_sigma)

            # Add noise and timestep
            noise = torch.randn_like(z).to(device)
            timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
            
            # Get model prediction
            noise_pred = inferer(
                inputs=images, diffusion_model=unet, noise=noise, timesteps=timesteps, autoencoder_model=autoencoderkl
            )

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
                    z_mu, z_sigma = autoencoderkl.encode(images)
                    z = autoencoderkl.sampling(z_mu, z_sigma)

                    noise = torch.randn_like(z).to(device)
                    timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()

                    noise_pred = inferer(
                        inputs=images, diffusion_model=unet, noise=noise, timesteps=timesteps, autoencoder_model=autoencoderkl
                    )

                    loss = F.mse_loss(noise_pred.float(), noise.float())

                val_loss += loss.item()
        val_loss /= val_step
        val_losses.append(val_loss)
        print(f"Epoch {epoch} val loss: {val_loss:.4f}")

        # Sampling image during training to have visual inspection
        z = torch.randn((1, 3, 16, 16), device=device)
        scheduler.set_timesteps(num_inference_steps=1000)
        with autocast(enabled=True):
            decoded = inferer.sample(
                input_noise=z, diffusion_model=unet, scheduler=scheduler, autoencoder_model=autoencoderkl
            )
        # save output
        plt.imsave(f"{paths['model_output']}/examples_uncond/sample_epoch{epoch}.png", decoded[0, 0].detach().cpu(), cmap="gray", vmin=0, vmax=1)

progress_bar.close()
print("Finished training")

# save model at the end to model_output + "generator.pth"
output_model_path = os.path.join(paths["model_output"], "uncond_generator.pth")
torch.save(unet.state_dict(), output_model_path)
print(f"Saved trained unet model to: {output_model_path}")

# scave scale_factor to use for generater sampling as well
torch.save({"scale_factor": scale_factor.item()}, os.path.join(paths["model_output"], "scale.pt"))

# Create a DataFrame to store the losses
loss_df = pd.DataFrame({
    "epoch": list(range(0, n_epochs)),
    "train_loss": epoch_losses,
    "val_recon_loss": [val_losses[i // val_interval] if (i + 1) % val_interval == 0 else None for i in range(n_epochs)]
})
loss_csv_path = os.path.join(paths["model_output"], "uncond_training_losses.csv")
loss_df.to_csv(loss_csv_path, index=False)
print(f"Saved training loss log to: {loss_csv_path}")

# Cleanup after training
torch.cuda.empty_cache()
