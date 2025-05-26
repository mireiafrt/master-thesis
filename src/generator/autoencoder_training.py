import os
import shutil
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
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator


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

# use the test data to train-val the autoencoder, since it will be fed into the LDM
test_df = metadata[metadata["split"] == "test"]
# split set into 80-20 train-val
train, val = train_test_split(test_df, train_size=0.8, stratify=test_df[columns["label"]], random_state=42)

# prepare the data for the autoencoder model
train_data = [{"image": row[columns["image_path"]]} for _, row in train.iterrows()]
val_data = [{"image": row[columns["image_path"]]} for _, row in val.iterrows()]

train_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image"], ensure_channel_first=True),
    #transforms.EnsureChannelFirstd(keys=["image"]),
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
        #transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    ]
)

train_ds = Dataset(data=train_data, transform=train_transforms)
val_ds = Dataset(data=val_data, transform=val_transforms)
train_loader = DataLoader(train_ds, batch_size=training["batch_size"], shuffle=True, num_workers=4, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=training["batch_size"], shuffle=True, num_workers=4, persistent_workers=True)

######## DEFINE AUTOENCODER NET, LOSS, and OPT ########
device = torch.device("cuda")

# autoencoder architecture
autoencoderkl = AutoencoderKL(
    spatial_dims=2,                         # 2D input (e.g., grayscale images, CT slices)
    in_channels=1,                          # Input has 1 channel (e.g., grayscale medical image)
    out_channels=1,                         # Reconstructed output has 1 channel (same as input)
    num_channels=(128, 128, 256),           # Channels for encoder/decoder at each resolution level
    latent_channels=3,                      # Number of channels in the latent representation (z)
    num_res_blocks=2,                       # Number of residual blocks per level
    attention_levels=(False, False, False), # No self-attention at any level
    with_encoder_nonlocal_attn=False,       # No non-local attention in encoder
    with_decoder_nonlocal_attn=False        # No non-local attention in decoder
)
autoencoderkl = autoencoderkl.to(device)

# Initialize perceptual loss using a pretrained AlexNet model. Compares feature activations between reconstructed and real images.
perceptual_loss = PerceptualLoss(
    spatial_dims=2,            # 2D images (e.g., CT slices)
    network_type="alex"        # Use AlexNet as the pretrained feature extractor
)
perceptual_loss.to(device)     # Move to GPU
perceptual_weight = 0.001 # How much perceptual loss contributes to the total generator loss

# Initialize a PatchGAN-style discriminator that works on small patches of the image
discriminator = PatchDiscriminator(
    spatial_dims=2,            # 2D input images
    num_layers_d=3,            # Depth of the discriminator network
    num_channels=64,           # Starting number of filters
    in_channels=1,             # Grayscale input (1 channel)
    out_channels=1             # Outputs a prediction per patch
)
discriminator = discriminator.to(device)  # Move to GPU
adv_loss = PatchAdversarialLoss(criterion="least_squares") # Use least-squares GAN (LSGAN) for stable training
adv_weight = 0.01 # How much the adversarial loss contributes to the total generator loss

# Optimizer for the autoencoder (generator)
optimizer_g = torch.optim.Adam(autoencoderkl.parameters(),lr=1e-4)
# Optimizer for the discriminator
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=5e-4)

# Mixed-precision training: Gradient scalers prevent underflow when using float16
scaler_g = torch.cuda.amp.GradScaler()   # Scaler for generator
scaler_d = torch.cuda.amp.GradScaler()   # Scaler for discriminator

############ TRAIN MODEL ############
# === Training settings ===
kl_weight = 1e-6                        # Weight for KL divergence in the VAE loss
n_epochs = training['num_epochs']       # Total number of training epochs
val_interval = training['val_interval'] # Run validation every N epochs
autoencoder_warm_up_n_epochs = training['warm_up'] # First N epochs: no adversarial training (VAE-only)
# === Logging ===
epoch_recon_losses = []               # Track per-epoch reconstruction loss
epoch_gen_losses = []                 # Track per-epoch generator adversarial loss
epoch_disc_losses = []                # Track per-epoch discriminator loss
val_recon_losses = []                 # Track per-epoch validation reconstruction loss

# === Training loop ===
for epoch in range(n_epochs):
    autoencoderkl.train()
    discriminator.train()
    epoch_loss = 0
    gen_epoch_loss = 0
    disc_epoch_loss = 0

    # Set up progress bar for the epoch
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")

    for step, batch in progress_bar:
        images = batch["image"].to(device)

        # === Train Generator ===
        optimizer_g.zero_grad(set_to_none=True)
        with autocast(enabled=True):  # Mixed precision context
            # Forward pass through the autoencoder
            reconstruction, z_mu, z_sigma = autoencoderkl(images)

            # 1. Reconstruction loss (L1 between input and output)
            recons_loss = F.l1_loss(reconstruction.float(), images.float())

            # 2. Perceptual loss using pretrained AlexNet features
            p_loss = perceptual_loss(reconstruction.float(), images.float())

            # 3. KL divergence loss (VAE regularizer)
            kl_loss = 0.5 * torch.sum(
                z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
                dim=[1, 2, 3]
            )
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]  # Mean over batch

            # Combine losses into generator loss (reconstruction + perceptual + KL)
            loss_g = recons_loss + (kl_weight * kl_loss) + (perceptual_weight * p_loss)

            # Add adversarial generator loss only after warm-up period
            if epoch > autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g += adv_weight * generator_loss

        # Backprop for generator
        scaler_g.scale(loss_g).backward()
        scaler_g.step(optimizer_g)
        scaler_g.update()

        # === Train Discriminator ===
        if epoch > autoencoder_warm_up_n_epochs:
            with autocast(enabled=True):
                optimizer_d.zero_grad(set_to_none=True)

                # Discriminator loss on fake (reconstructed) images
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)

                # Discriminator loss on real images
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)

                # Average the two
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                loss_d = adv_weight * discriminator_loss

            # Backprop for discriminator
            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

        # === Logging batch losses ===
        epoch_loss += recons_loss.item()
        if epoch > autoencoder_warm_up_n_epochs:
            gen_epoch_loss += generator_loss.item()
            disc_epoch_loss += discriminator_loss.item()

        # Update tqdm progress bar display
        progress_bar.set_postfix({
            "recons_loss": epoch_loss / (step + 1),
            "gen_loss": gen_epoch_loss / (step + 1),
            "disc_loss": disc_epoch_loss / (step + 1),
        })

    # === Logging epoch-level losses ===
    epoch_recon_losses.append(epoch_loss / (step + 1))
    epoch_gen_losses.append(gen_epoch_loss / (step + 1))
    epoch_disc_losses.append(disc_epoch_loss / (step + 1))

    # === Validation ===
    if (epoch + 1) % val_interval == 0:
        autoencoderkl.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(val_loader, start=1):
                images = batch["image"].to(device)

                with autocast(enabled=True):
                    reconstruction, z_mu, z_sigma = autoencoderkl(images)
                    
                    # Get the first 2 reconstruction from the first validation batch for visualisation purposes
                    if val_step == 1:
                        recon_sample = reconstruction[:2].detach().cpu()  # first 2 reconstructed images
                        for i, img in enumerate(recon_sample):
                            # img shape: [1, H, W] → [H, W]
                            img_array = img[0] if img.shape[0] == 1 else img  # in case channel dim is present
                            out_path = os.path.join(paths["model_output"], f"recon_epoch{epoch}_img{i}.png")
                            plt.imsave(out_path, img_array, cmap="gray", vmin=0, vmax=1)
                    
                    # L1 validation reconstruction loss
                    recons_loss = F.l1_loss(images.float(), reconstruction.float())

                val_loss += recons_loss.item()

        val_loss /= val_step
        val_recon_losses.append(val_loss)
        print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")

progress_bar.close()
print("Finished training")

# === Save trained autoencoder ===
output_model_path = os.path.join(paths["model_output"], "autoencoder.pth")
torch.save(autoencoderkl.state_dict(), output_model_path)
print(f"Saved trained autoencoder model to: {output_model_path}")

# Create a DataFrame to store the losses
loss_df = pd.DataFrame({
    "epoch": list(range(0, n_epochs)),
    "recon_loss": epoch_recon_losses,
    "gen_loss": [None] * autoencoder_warm_up_n_epochs + epoch_gen_losses, # pad first warmup epochs with nulls
    "disc_loss": [None] * autoencoder_warm_up_n_epochs + epoch_disc_losses,
    "val_recon_loss": [val_recon_losses[i // val_interval] if (i + 1) % val_interval == 0 else None for i in range(n_epochs)]
})
loss_csv_path = os.path.join(paths["model_output"], "training_losses.csv")
loss_df.to_csv(loss_csv_path, index=False)
print(f"Saved training loss log to: {loss_csv_path}")

# Cleanup after training
del discriminator
del perceptual_loss
torch.cuda.empty_cache()

