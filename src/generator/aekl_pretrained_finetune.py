import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter

from monai import transforms
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator


# clean up gpu cache before starting
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# set global seed
set_determinism(42)

# Load config
with open("config/pre_trained/autoencoder_finetune.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
columns = config["columns"]
training = config["training"]

print("Preparing paths ...")
# make sure model output path exists
os.makedirs(paths["model_output"], exist_ok=True)

# prepare paths and existace of folders for writers
writers_outputs = Path(paths["writers_outputs"])
writer_train_path = writers_outputs / "train"
writer_train_path.mkdir(parents=True, exist_ok=True)
writer_val_path = writers_outputs / "val"
writer_val_path.mkdir(parents=True, exist_ok=True)

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

# TRANSFORMS (same as from generative_chestxray MONAI project)
train_transforms = transforms.Compose([
    # Load the image file into a dictionary with key "image" and ensure [C, H, w]
    transforms.LoadImaged(keys=["image"], ensure_channel_first=True),
    # Rescale intensity from [0, 255] to [0.0, 1.0] — common normalization step
    transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    # Crop the central region to fixed size (e.g., 512x512 pixels)
    # Important if CT slices are larger than 512x512
    transforms.CenterSpatialCropd(keys=["image"], roi_size=(512, 512)),
    # Apply random affine transform (rotation, translation, scaling)
    # Helps regularize training by introducing geometric variation
    transforms.RandAffined(
        keys=["image"],
        rotate_range=(-np.pi / 36, np.pi / 36),  # ~±5 degrees
        translate_range=(-2, 2),                 # shift image by 2 pixels max
        scale_range=(-0.01, 0.01),               # ~±1% zoom
        spatial_size=[512, 512],                 # ensure final size stays constant
        prob=0.5,
    ),
    # Flip horizontally with 50% probability — good general augmentation
    transforms.RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),
    # Convert image to PyTorch tensor
    transforms.ToTensord(keys=["image"]),
])

val_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image"], ensure_channel_first=True),
    transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    transforms.CenterSpatialCropd(keys=["image"], roi_size=(512, 512)),
    transforms.ToTensord(keys=["image"]),
])

train_ds = Dataset(data=train_data, transform=train_transforms)
val_ds = Dataset(data=val_data, transform=val_transforms)
train_loader = DataLoader(train_ds, batch_size=training["batch_size"], shuffle=True, num_workers=4, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=training["batch_size"], shuffle=True, num_workers=4, persistent_workers=True)

######## DEFINE AUTOENCODER NET, LOSS, and OPT ########
device = torch.device("cuda")

# copy archictecture from MONAI pre-trained model-zoo chest-xray model
autoencoderkl = AutoencoderKL(
    spatial_dims=2,                                # 2D data (for CT slices or X-rays, use 3 for 3D volumes)
    in_channels=1,                                 # Input has 1 channel (grayscale image)
    out_channels=1,                                # Output will also be 1 channel (reconstruction)
    latent_channels=3,                             # Size of the latent space (number of channels in latent code z)
    num_channels=(64, 128, 128, 128),              # Number of feature channels at each encoder/decoder level
                                                   # The model has 4 levels; these are the feature map sizes
    num_res_blocks=2,                              # Number of residual blocks per level in encoder/decoder
    attention_levels=(False, False, False, False), # Whether to use attention at each level — all off here
    with_encoder_nonlocal_attn=False,              # Use non-local self-attention in the encoder (disabled)
    with_decoder_nonlocal_attn=False               # Use non-local self-attention in the decoder (disabled)
)
autoencoderkl = autoencoderkl.to(device)
# Load pretrained weights
autoencoderkl.load_state_dict(torch.load(paths["pretrained_model_path"])) # load the trained autoencoder model
print("Loaded pretrained autoencoder weights")

# Initialize a PatchGAN-style discriminator (same config from pre-trained)
discriminator = PatchDiscriminator(
    spatial_dims=2,            # 2D input images
    num_layers_d=3,            # Depth of the discriminator network
    num_channels=64,           # Starting number of filters
    in_channels=1,             # Grayscale input (1 channel)
    out_channels=1             # Outputs a prediction per patch
)
discriminator = discriminator.to(device)  # Move to GPU
adv_weight = 0.005 # How much the adversarial loss contributes to the total generator loss

# Same archutecture than pre-trained
# Initialize perceptual loss using a pretrained AlexNet model. Compares feature activations between reconstructed and real images.
perceptual_loss = PerceptualLoss(
    spatial_dims=2,            # 2D images (e.g., CT slices)
    network_type="alex"        # Use AlexNet as the pretrained feature extractor
)
perceptual_loss.to(device)     # Move to GPU
perceptual_weight = 0.002 # How much perceptual loss contributes to the total generator loss

# Optimizer for the autoencoder (generator)
optimizer_g = torch.optim.Adam(autoencoderkl.parameters(),lr=0.00005)
# Optimizer for the discriminator
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

# Mixed-precision training: Gradient scalers prevent underflow when using float16
scaler_g = GradScaler()   # Scaler for generator
scaler_d = GradScaler()   # Scaler for discriminator

############ TRAIN MODEL ############
# === Training settings ===
kl_weight = 0.00000001                  # Weight for KL divergence in the VAE loss
n_epochs = training['num_epochs']       # Total number of training epochs
eval_freq = training['eval_freq']    # Run validation every N epochs
adv_start = training["adv_start"]       # Epoch when the adversarial straining starts
# === Logging ===
writer_train = SummaryWriter(log_dir=writer_train_path)
writer_val = SummaryWriter(log_dir=writer_val_path)

print("Starting training ...")
# === Training loop ===
for epoch in range(n_epochs):
    # set adv_weight to 0 if adv does not start yet
    adv_weight = adv_weight if epoch >= adv_start else 0.0
    # train epoch aekl
    autoencoderkl.train()
    discriminator.train()
    # Use least-squares GAN (LSGAN) for stable training
    adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=True)
    
    # Progress bar through batches
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    pbar.set_description(f"Epoch {epoch}")
    for step, x in pbar:
        images = x["image"].to(device)

        # GENERATOR
        optimizer_g.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            reconstruction, z_mu, z_sigma = autoencoderkl(x=images)
            # calculate reconstruction loss
            l1_loss = F.l1_loss(reconstruction.float(), images.float())
            # calculate perceptual loss
            p_loss = perceptual_loss(reconstruction.float(), images.float())
            # KL divergence loss (VAE regularizer)
            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            # calculate generator loss (depends if adv already or not)
            if adv_weight > 0:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            else:
                generator_loss = torch.tensor([0.0]).to(device)

            # Combine losses with their weights
            loss = l1_loss + kl_weight * kl_loss + perceptual_weight * p_loss + adv_weight * generator_loss

            loss = loss.mean()
            l1_loss = l1_loss.mean()
            p_loss = p_loss.mean()
            kl_loss = kl_loss.mean()
            g_loss = generator_loss.mean()
            losses = OrderedDict(loss=loss, l1_loss=l1_loss, p_loss=p_loss, kl_loss=kl_loss, g_loss=g_loss)

        # Backprop for generator
        scaler_g.scale(losses["loss"]).backward()
        scaler_g.unscale_(optimizer_g)
        torch.nn.utils.clip_grad_norm_(autoencoderkl.parameters(), 1)
        scaler_g.step(optimizer_g)
        scaler_g.update()

        # DISCRIMINATOR
        if adv_weight > 0:
            optimizer_d.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                # Discriminator loss on fake (reconstructed) images
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                # Discriminator loss on real images
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                 # Average the two
                d_loss = adv_weight * discriminator_loss
                d_loss = d_loss.mean()

            # Backprop for discriminator
            scaler_d.scale(d_loss).backward()
            scaler_d.unscale_(optimizer_d)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1)
            scaler_d.step(optimizer_d)
            scaler_d.update()
        else:
            discriminator_loss = torch.tensor([0.0]).to(device)

        losses["d_loss"] = discriminator_loss

        # === Logging batch losses to writer ===
        for k, v in losses.items():
            writer_train.add_scalar(f"{k}", v.item(), epoch * len(train_loader) + step)

        pbar.set_postfix({
            "epoch": epoch,
            "loss": f"{losses['loss'].item():.6f}",
            "l1_loss": f"{losses['l1_loss'].item():.6f}",
            "p_loss": f"{losses['p_loss'].item():.6f}",
            "g_loss": f"{losses['g_loss'].item():.6f}",
            "d_loss": f"{losses['d_loss'].item():.6f}",
        },)


    # validation epoch
    print("Running validation epoch ...")
    if (epoch + 1) % eval_freq == 0:
        # set params
        step=len(train_loader) * epoch
        adv_weight=adv_weight if epoch >= adv_start else 0.0
        # eval aekl
        with torch.no_grad():
            autoencoderkl.eval()
            discriminator.eval()

            adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=True)
            total_losses = OrderedDict()
            
            for x in val_loader:
                images = x["image"].to(device)

                with autocast(enabled=True):
                    # GENERATOR
                    reconstruction, z_mu, z_sigma = autoencoderkl(x=images)
                    l1_loss = F.l1_loss(reconstruction.float(), images.float())
                    p_loss = perceptual_loss(reconstruction.float(), images.float())
                    kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
                    kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

                    if adv_weight > 0:
                        logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                        generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                    else:
                        generator_loss = torch.tensor([0.0]).to(device)

                    # DISCRIMINATOR
                    if adv_weight > 0:
                        logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                        loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                        logits_real = discriminator(images.contiguous().detach())[-1]
                        loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                    else:
                        discriminator_loss = torch.tensor([0.0]).to(device)

                    loss = l1_loss + kl_weight * kl_loss + perceptual_weight * p_loss + adv_weight * generator_loss

                    loss = loss.mean()
                    l1_loss = l1_loss.mean()
                    p_loss = p_loss.mean()
                    kl_loss = kl_loss.mean()
                    g_loss = generator_loss.mean()
                    d_loss = discriminator_loss.mean()
                    losses = OrderedDict(loss=loss, l1_loss=l1_loss, p_loss=p_loss, kl_loss=kl_loss, g_loss=g_loss, d_loss=d_loss)

                for k, v in losses.items():
                    total_losses[k] = total_losses.get(k, 0) + v.item() * images.shape[0]

            for k in total_losses.keys():
                total_losses[k] /= len(val_loader.dataset)

            for k, v in total_losses.items():
                writer_val.add_scalar(f"{k}", v, step)

            # log reconstructions
            img_npy_0 = np.clip(a=images[0, 0, :, :].cpu().numpy(), a_min=0, a_max=1)
            recons_npy_0 = np.clip(a=reconstruction[0, 0, :, :].cpu().numpy(), a_min=0, a_max=1)
            img_npy_1 = np.clip(a=images[1, 0, :, :].cpu().numpy(), a_min=0, a_max=1)
            recons_npy_1 = np.clip(a=reconstruction[1, 0, :, :].cpu().numpy(), a_min=0, a_max=1)

            img_row_0 = np.concatenate((img_npy_0,recons_npy_0,img_npy_1,recons_npy_1,), axis=1,)

            fig = plt.figure(dpi=300)
            plt.imshow(img_row_0, cmap="gray")
            plt.axis("off")
            writer_val.add_figure("RECONSTRUCTION", fig, step)

            val_loss = total_losses["l1_loss"]
            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")


print(f"Training finished!")

# Save model weigths
output_model_path = os.path.join(paths["model_output"], "finetuned_autoencoder.pth")
torch.save(autoencoderkl.state_dict(), output_model_path)
print(f"Saved final autoencoder model to: {output_model_path}")

# Cleanup after training
del discriminator
del perceptual_loss
torch.cuda.empty_cache()