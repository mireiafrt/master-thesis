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

# prepare the columns of interest for the embeddings
sex_map = {"M": 0, "F": 1, "uncond": 2}
test_df["sex_id"] = test_df["sex"].map(sex_map)
age_map = {"Under 20": 0, "20-40": 1, "40-60": 2, "60-80": 3, "Over 80": 4, "uncond": 5}
test_df["age_id"] = test_df["age_group"].map(age_map)
label_map = {0: 0, 1: 1, "uncond": 2}
test_df["label_id"] = test_df["label"].map(label_map)

# set up embeddings for later
label_embed = torch.nn.Embedding(num_embeddings=len(label_map), embedding_dim=training["emb_size"]).to(device)
sex_embed = torch.nn.Embedding(num_embeddings=len(sex_map), embedding_dim=training["emb_size"]).to(device)
age_embed = torch.nn.Embedding(num_embeddings=len(age_map), embedding_dim=training["emb_size"]).to(device)
# Combine embeddings by summing (similar to other embedding set ups like CLIP)
def get_conditioning_vector(label_id, sex_id, age_id):
    return label_embed(label_id) + sex_embed(sex_id) + age_embed(age_id)  # shape: [B, 128]

# split test set into train-val (same split as autoencoder, seed 42 and 80-20)
train, val = train_test_split(test_df, train_size=0.8, stratify=test_df[columns["label"]], random_state=42)

# prepare data dicts for loaders
train_data = [{"image": row[columns["image_path"]], "label_id": row["label_id"], "sex_id": row["sex_id"], "age_id": row["age_id"]} for _, row in train.iterrows()]
val_data = [{"image": row[columns["image_path"]], "label_id": row["label_id"], "sex_id": row["sex_id"], "age_id": row["age_id"]} for _, row in val.iterrows()]

# TRANSFORMS
train_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image"], ensure_channel_first=True),
    transforms.ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0, b_max=1, clip=True),
    transforms.RandAffined(
        keys=["image"],
        rotate_range=[(-np.pi/36, np.pi/36)] * 2,
        translate_range=[(-1, 1)] * 2,
        scale_range=[(-0.05, 0.05)] * 2,
        spatial_size=[64, 64],
        padding_mode="zeros",
        prob=0.5,
    ),
    transforms.RandLambdad(keys=["label_id"], prob=0.15, func=lambda x: label_map['uncond']), # with prob 15%, unconditional label
    transforms.RandLambdad(keys=["sex_id"], prob=0.15, func=lambda x: sex_map['uncond']), # with prob 15%, unconditional sex
    transforms.RandLambdad(keys=["age_id"], prob=0.15, func=lambda x: age_map['uncond']), # with prob 15%, unconditional age group
    transforms.Lambdad(keys=["label_id"], func=lambda x: torch.tensor(x, dtype=torch.long)), # shape needed for nn.Embedding
    transforms.Lambdad(keys=["sex_id"], func=lambda x: torch.tensor(x, dtype=torch.long)),
    transforms.Lambdad(keys=["age_id"], func=lambda x: torch.tensor(x, dtype=torch.long)),
])
val_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image"], ensure_channel_first=True),
    transforms.ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0, b_max=1, clip=True),
    transforms.RandAffined(
        keys=["image"],
        rotate_range=[(-np.pi/36, np.pi/36)] * 2,
        translate_range=[(-1, 1)] * 2,
        scale_range=[(-0.05, 0.05)] * 2,
        spatial_size=[64, 64],
        padding_mode="zeros",
        prob=0.5,
    ),
    transforms.Lambdad(keys=["label_id"], func=lambda x: torch.tensor(x, dtype=torch.long)), # shape needed for nn.Embedding
    transforms.Lambdad(keys=["sex_id"], func=lambda x: torch.tensor(x, dtype=torch.long)),
    transforms.Lambdad(keys=["age_id"], func=lambda x: torch.tensor(x, dtype=torch.long)),
])

train_ds = Dataset(data=train_data, transform=train_transforms)
val_ds = Dataset(data=val_data, transform=val_transforms)
train_loader = DataLoader(train_ds, batch_size=training["batch_size"], shuffle=True, num_workers=4, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=training["batch_size"], shuffle=False, num_workers=4, persistent_workers=True)

############## DEFINE MODEL AND ARCHITECTURE ##############
device = torch.device("cuda")
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
    with_conditioning=True,       # Enables support for passing external conditioning information (e.g. embeddings)
    cross_attention_dim=training["emb_size"], # The size of the conditioning vector, must match the combined embedding vector dim
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
        label_id = batch["label_id"].to(device)
        sex_id = batch["sex_id"].to(device)
        age_id = batch["age_id"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            # Encode image to latent
            z_mu, z_sigma = autoencoderkl.encode(images)
            z = autoencoderkl.sampling(z_mu, z_sigma)

            # Add noise and timestep
            noise = torch.randn_like(z).to(device)
            timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()

            # Get conditioning vector
            cond = get_conditioning_vector(label_id, sex_id, age_id)  # shape: [B, 128]
            cond = cond.unsqueeze(1)  # shape: [B, 1, 128] for cross-attn

            # Get model prediction
            noise_pred = inferer(
                inputs=images, diffusion_model=unet, noise=noise, timesteps=timesteps, autoencoder_model=autoencoderkl, condition=cond
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
                label_id = batch["label_id"].to(device)
                sex_id = batch["sex_id"].to(device)
                age_id = batch["age_id"].to(device)

                with autocast(enabled=True):
                    z_mu, z_sigma = autoencoderkl.encode(images)
                    z = autoencoderkl.sampling(z_mu, z_sigma)

                    noise = torch.randn_like(z).to(device)
                    timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
                    
                    cond = get_conditioning_vector(label_id, sex_id, age_id) 
                    cond = cond.unsqueeze(1)

                    noise_pred = inferer(
                        inputs=images, diffusion_model=unet, noise=noise, timesteps=timesteps, autoencoder_model=autoencoderkl, condition=cond
                    )

                    loss = F.mse_loss(noise_pred.float(), noise.float())

                val_loss += loss.item()
        val_loss /= val_step
        val_losses.append(val_loss)
        print(f"Epoch {epoch} val loss: {val_loss:.4f}")

        # Sampling a conditional image during training to have visual inspection
        label_c = torch.tensor([1, 2], dtype=torch.long).to(device)  # label_id: [cond(covid), uncond]
        sex_c = torch.tensor([0, 2], dtype=torch.long).to(device)    # sex_id: [cond(M), uncond]
        age_c = torch.tensor([1, 5], dtype=torch.long).to(device)    # age_id: [cond(20-40), uncond]
        cond = get_conditioning_vector(label_c, sex_c, age_c)  # shape: [2, 128]
        cond = cond.unsqueeze(1)  # shape: [2, 1, 128]

        z = torch.randn((1, 3, 16, 16), device=device)
        z = z.repeat(2, 1, 1, 1)  # duplicate for cond & uncond
        scheduler.set_timesteps(num_inference_steps=1000)
        for t in tqdm(scheduler.timesteps):
            with torch.no_grad(), autocast():
                model_output = unet(z, timesteps=torch.tensor([t], device=device), context=cond)
                noise_pred_uncond, noise_pred_cond = model_output.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                z, _ = scheduler.step(noise_pred, t, z)

        decoded = autoencoderkl.decode(z[0].unsqueeze(0)) 
        # save output
        plt.imsave(f"{paths['model_output']}/sample_epoch{epoch}.png", decoded[0, 0].detach().cpu(), cmap="gray", vmin=0, vmax=1)

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
progress_bar.close()
torch.cuda.empty_cache()


