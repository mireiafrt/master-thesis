import os
import shutil
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter

from monai import transforms
from monai.transforms import MapTransform
from monai.data import DataLoader, Dataset, ImageReader
from monai.utils import first, set_determinism
from generative.inferers import DiffusionInferer, LatentDiffusionInferer
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel

# clean up gpu cache before starting
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# set global seed
set_determinism(42)

# Load config
with open("config/pre_trained/generator_finetune.yaml", "r") as f:
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

# Create "report" col --> Sentence like "Female, age group under 20, healthy" or "x, x, with COVID-19"
def build_clip_prompt(row):
    sex_term = "female" if row["sex"] == "F" else "male"
    diagnosis_text = "healthy" if row["label"] == 0 else "with COVID-19"
    return f"A {sex_term} patient in the {row['age_group']} age group, {diagnosis_text}."
test_df["report"] = test_df.apply(build_clip_prompt, axis=1)
print("Reports created ...")

# split set into 80-20 train-val
train, val = train_test_split(test_df, train_size=0.8, stratify=test_df[columns["label"]], random_state=42)

# prepare the data for the autoencoder model
train_data = [{"image": row[columns["image_path"]], "report":row["report"]} for _, row in train.iterrows()]
val_data = [{"image": row[columns["image_path"]], "report": row["report"]} for _, row in val.iterrows()]

######################## TRANSFORMS (same as from generative_chestxray MONAI project)
# custom trandform to tokenize reports
class ApplyTokenizerd(MapTransform):
    """
    Map-style transform to apply CLIP tokenizer to a string field (e.g., 'report').
    Outputs token IDs tensor with shape [1, max_length].
    """

    def __init__(self, keys, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer"
        )

    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        for key in self.key_iterator(d):
            tokenized = self.tokenizer(
                d[key],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            d[key] = tokenized.input_ids  # shape: [1, max_length]
        return d

# train and val transforms
train_transforms = transforms.Compose([
    # Load the image file into a dictionary with key "image"
    transforms.LoadImaged(keys=["image"]),
    # Ensure the image has channel-first format [C, H, W]
    transforms.EnsureChannelFirstd(keys=["image"]),
    # Extract only the first channel and add back the channel dimension: [1, H, W] (keep the grayscale channel)
    # For CT slices (already 1 channel), this might be redundant
    transforms.Lambdad(keys=["image"], func=lambda x: x[0, :, :][None,],),
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
        prob=0.10,
    ),
    # Convert image to PyTorch tensor
    transforms.ToTensord(keys=["image"]),
    # Tokanize reports with custom function
    ApplyTokenizerd(keys=["report"]),
    # replaces the tokenized text 10% of the time with a special synthetic version (blank prompt)
    transforms.RandLambdad(keys=["report"],
        prob=0.10,
        func=lambda x: torch.cat((49406 * torch.ones(1, 1),
                                  49407 * torch.ones(1, x.shape[1] - 1)),
                                  1).long(),),  # 49406: BOS token 49407: PAD token
])

val_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image"]),
    transforms.EnsureChannelFirstd(keys=["image"]),
    transforms.Lambdad(keys=["image"],func=lambda x: x[0, :, :][None,],),
    transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
    transforms.CenterSpatialCropd(keys=["image"], roi_size=(512, 512)),
    transforms.ToTensord(keys=["image"]),
    ApplyTokenizerd(keys=["report"]),
])

train_ds = Dataset(data=train_data, transform=train_transforms)
val_ds = Dataset(data=val_data, transform=val_transforms)
train_loader = DataLoader(train_ds, batch_size=training["batch_size"], shuffle=True, num_workers=4, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=training["batch_size"], shuffle=True, num_workers=4, persistent_workers=True)
print("Loaders prepared ...")

######## LOAD AUTOENCODER NET AND DEFINE GENERATOR ARCH, LOSS, OPT, ... ########
device = torch.device("cuda")

# load finetuned autoencoder to produce the latent representations
class Stage1Wrapper(nn.Module):
    """Wrapper for stage 1 model as a workaround for the DataParallel usage in the training loop."""
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_mu, z_sigma = self.model.encode(x)
        z = self.model.sampling(z_mu, z_sigma)
        return z
    
print(f"Loading Autoencoder from {paths['autoencoder_path']}")
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
autoencoderkl.load_state_dict(torch.load(paths["autoencoder_path"])) # load the trained autoencoder model
autoencoderkl = Stage1Wrapper(model=autoencoderkl)
autoencoderkl.eval()

# Create the diffusion model and load pre-trained model
print("Creating unet model...")
diffusion = DiffusionModelUNet(
    spatial_dims=2,              # 2D CT slices
    in_channels=3,               # match AutoencoderKL latent_channels (latent space channels)
    out_channels=3,              # same as in_channels
    num_res_blocks=2,             # Number of residual blocks per level in the UNet
    num_channels=(256, 512, 768), # Number of channels at each level of the UNet encoder/decoder
    attention_levels=(False, True, True), # Whether to apply self-attention at each UNet level
    with_conditioning=True,       # Enables support for passing external conditioning information (e.g. embeddings)
    cross_attention_dim=training["emb_size"], # The size of the conditioning vector, must match the combined embedding vector dim
    num_head_channels=(0, 512, 768), # Number of channels per attention head at each level (if attention is enabled)
)
diffusion = diffusion.to(device)
diffusion.load_state_dict(torch.load(paths["pretrained_model_path"])) # load the pretrained generator model
print("Loaded pretrained model ...")

# set scheduler
scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195, prediction_type="v_prediction")

# set up text encoder
text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder")
text_encoder = text_encoder.to(device)
print("Loaded text encoder ...")

# setting up scaling_factor from autoencoder z std
eda_data = first(train_loader)["image"]
with torch.no_grad():
        z = autoencoderkl.forward(eda_data.to(device))
scale_factor = 1 / torch.std(z)
print(f"Scale factor: {scale_factor}")

# set optimizer
optimizer = torch.optim.Adam(diffusion.parameters(), lr=0.000025)

# set GradScaler
scaler = GradScaler()

############ TRAIN MODEL ############
# === Training settings ===
n_epochs = training['num_epochs']       # Total number of training epochs
eval_freq = training['eval_freq']       # Run validation every N epochs
guidance_scale = training['guidance_scale']
# === Logging ===
writer_train = SummaryWriter(log_dir=writer_train_path)
writer_val = SummaryWriter(log_dir=writer_val_path)

# === Training loop ===
for epoch in range(n_epochs):
    # train epoch
    diffusion.train()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    pbar.set_description(f"Epoch {epoch}")
    for step, x in pbar:
        images = x["image"].to(device)
        reports = x["report"].to(device)
        
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            with torch.no_grad():
                e = autoencoderkl(images) * scale_factor

            prompt_embeds = text_encoder(reports.squeeze(1))
            prompt_embeds = prompt_embeds[0]

            noise = torch.randn_like(e).to(device)
            noisy_e = scheduler.add_noise(original_samples=e, noise=noise, timesteps=timesteps)
            noise_pred = diffusion(x=noisy_e, timesteps=timesteps, context=prompt_embeds)

            if scheduler.prediction_type == "v_prediction":
                # Use v-prediction parameterization
                target = scheduler.get_velocity(e, noise, timesteps)
            elif scheduler.prediction_type == "epsilon":
                target = noise
            loss = F.mse_loss(noise_pred.float(), target.float())

        losses = OrderedDict(loss=loss)

        scaler.scale(losses["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()

        for k, v in losses.items():
            writer_train.add_scalar(f"{k}", v.item(), epoch * len(train_loader) + step)

        pbar.set_postfix({"epoch": epoch, "loss": f"{losses['loss'].item():.5f}"})

    # val epoch
    if (epoch + 1) % eval_freq == 0:
        print("Validation epoch ...")
        step=len(train_loader) * epoch
        # sample=True if (epoch + 1) % (eval_freq * 2) == 0 else False
        sample=True if (epoch + 1) % (eval_freq) == 0 else False # will make it sample always in validation
        # evaluate
        with torch.no_grad():
            diffusion.eval()
            total_losses = OrderedDict()

            for x in val_loader:
                images = x["image"].to(device)
                reports = x["report"].to(device)
                timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()

                with autocast(enabled=True):
                    e = autoencoderkl(images) * scale_factor

                    prompt_embeds = text_encoder(reports.squeeze(1))
                    prompt_embeds = prompt_embeds[0]

                    noise = torch.randn_like(e).to(device)
                    noisy_e = scheduler.add_noise(original_samples=e, noise=noise, timesteps=timesteps)
                    noise_pred = diffusion(x=noisy_e, timesteps=timesteps, context=prompt_embeds)

                    if scheduler.prediction_type == "v_prediction":
                        # Use v-prediction parameterization
                        target = scheduler.get_velocity(e, noise, timesteps)
                    elif scheduler.prediction_type == "epsilon":
                        target = noise
                    loss = F.mse_loss(noise_pred.float(), target.float())

                loss = loss.mean()
                losses = OrderedDict(loss=loss)

                for k, v in losses.items():
                    total_losses[k] = total_losses.get(k, 0) + v.item() * images.shape[0]

        for k in total_losses.keys():
            total_losses[k] /= len(val_loader.dataset)

        for k, v in total_losses.items():
            writer_val.add_scalar(f"{k}", v, step)

        # sample data if sample (uncoditional from moani pretrained??)
        if sample:
            print("Sampling images ...")
            spatial_shape=tuple(e.shape[1:])
            with torch.no_grad():
                latent = torch.randn((1,) + spatial_shape)
                latent = latent.to(device)

                prompt_embeds = torch.cat((49406 * torch.ones(1, 1), 49407 * torch.ones(1, 76)), 1).long()
                prompt_embeds = text_encoder(prompt_embeds.squeeze(1)).to(device)
                prompt_embeds = prompt_embeds[0]

                for t in tqdm(scheduler.timesteps, ncols=70):
                    noise_pred = diffusion(x=latent, timesteps=torch.asarray((t,)).to(device), context=prompt_embeds)
                    latent, _ = scheduler.step(noise_pred, t, latent)

                x_hat = autoencoderkl.model.decode(latent / scale_factor)
                img_0 = np.clip(a=x_hat[0, 0, :, :].cpu().numpy(), a_min=0, a_max=1)
                fig = plt.figure(dpi=300)
                plt.imshow(img_0, cmap="gray")
                plt.axis("off")
                writer_val.add_figure("SAMPLE", fig, step)

        val_loss = total_losses["loss"]
        print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")

print("Finished training")

# save model at the end to model_output + "finetuned_generator.pth"
output_model_path = os.path.join(paths["model_output"], "finetuned_generator.pth")
torch.save(diffusion.state_dict(), output_model_path)
print(f"Saved trained unet model to: {output_model_path}")

# scave scale_factor to use for generater sampling as well
torch.save({"scale_factor": scale_factor.item()}, os.path.join(paths["model_output"], "scale.pt"))

# Cleanup after training
torch.cuda.empty_cache()

        