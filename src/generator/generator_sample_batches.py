import os
import shutil
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from pathlib import Path

import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from monai.utils import set_determinism
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer


# set global seed
set_determinism(42)

# Load config
with open("config/generator/generator_sample_batch.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
num_inference_steps = config["num_inference_steps"]
guidance_scale = config["guidance_scale"]
batch_size = config["batch_size"]
sample_size = config["sample_size"]
filters = config["conditioning"]

# load scale factor from paths
scale_dict = torch.load(os.path.join(paths["generator_path"], "scale.pt"))
scale_factor = float(scale_dict["scale_factor"])

print("Preparing paths ...")
# make sure images output path exists
os.makedirs(paths["imgs_output"], exist_ok=True)

########## PREPARE PROMPTS and RESULT CSV ##########
metadata = pd.read_csv(paths["metadata_csv"])
metadata = metadata[metadata["use"] == True]

# filter metadata based on the conditioning
print("Filters:", filters)
df = metadata.copy()
for key, value in filters.items():
    if value is not None:
        df = df[df[key] == value]
df = df.reset_index(drop=True)
print("Filtered data...")

# Sample a subset instead of full data if it is none (random)
if sample_size is not None:
    df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    print(f"Subsampled to {sample_size} random rows with seed 42.")

# Create "report" col --> Sentence like "Female, age group under 20, healthy" or "x, x, with COVID-19"
def build_clip_prompt(row):
    sex_term = "female" if row["sex"] == "F" else "male"
    diagnosis_text = "healthy" if row["label"] == 0 else "with COVID-19"
    return f"A {sex_term} patient in the {row['age_group']} age group, {diagnosis_text}."
df["report"] = df.apply(build_clip_prompt, axis=1)
print("Reports created ...")

########## LOAD MODELS INTO ARCHITECTURES ##########
device = torch.device("cuda")
dtype = torch.float16  # or torch.float32 if you want to disable mixed precision
torch.backends.cuda.matmul.allow_tf32 = True # Enable TF32 globally

# Load autoencoder
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
autoencoderkl.load_state_dict(torch.load(paths["autoencoder_path"], map_location=device)) # load the trained autoencoder model
autoencoderkl = autoencoderkl.to(device, dtype=dtype).eval()

# Load Diffusion Model
gen_path = os.path.join(paths["generator_path"], paths["model_name"])
print(f"Loading Diffusion model from {gen_path}")
diffusion = DiffusionModelUNet(
    spatial_dims=2,              # 2D CT slices
    in_channels=3,               # match AutoencoderKL latent_channels (latent space channels)
    out_channels=3,              # same as in_channels
    num_res_blocks=2,             # Number of residual blocks per level in the UNet
    num_channels=(256, 512, 768), # Number of channels at each level of the UNet encoder/decoder
    attention_levels=(False, True, True), # Whether to apply self-attention at each UNet level
    with_conditioning=True,       # Enables support for passing external conditioning information (e.g. embeddings)
    cross_attention_dim=1024,     # The size of the conditioning vector, must match the combined embedding vector dim (from CLIP)
    num_head_channels=(0, 512, 768), # Number of channels per attention head at each level (if attention is enabled)
)
diffusion.load_state_dict(torch.load(gen_path, map_location=device)) # load the trained diffusion model
diffusion = diffusion.to(device, dtype=dtype).eval()

# load scheduler (now the DDIM instead of DDPM)
scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.0015, beta_end=0.0205, schedule="scaled_linear_beta", prediction_type="v_prediction", clip_sample=False)
scheduler.set_timesteps(num_inference_steps)

# set up tokenizer and text encoder
tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder").to(device, dtype=dtype).eval()         # eval = disable dropout


# function to iterate through prompts and generate one image per prompt (IN BATCHES)
def generate_images_from_reports(df: pd.DataFrame, batch_size: int = 4) -> pd.DataFrame:
    """
    Vectorised sampler: generates |df| synthetic images in batches of |batch_size|.
    Writes PNGs to paths['imgs_output'] and returns df with new column 'syn_path'.
    """
    H_latent, W_latent = 64, 64                 # latent resolution you trained with
    syn_paths = []
    
    # MAIN BATCH LOOP (iterate through df in batches)
    for start in tqdm(range(0, len(df), batch_size), desc="Generating batches"):
        end   = min(start + batch_size, len(df))
        batch = df.iloc[start:end].copy()

        # ------------------------------------------------------------------
        # 1  Tokenise ("" prompt + real prompt)   → (2*B, L)
        # ------------------------------------------------------------------
        prompts = batch["report"].tolist()
        prompt_pair = [""] * len(prompts) + prompts # ex:["","","","","", real, real, real, real, real] (like hugging face example)
        text_inputs = tokenizer(prompt_pair, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad(), torch.cuda.amp.autocast(dtype is torch.float16):
            prompt_embeds = text_encoder(text_inputs.input_ids)[0]   # (2B, L, C)
        prompt_embeds = prompt_embeds.to(dtype=dtype)

        # ------------------------------------------------------------------
        # 2  Initial latent noise     → (B, 3, 64, 64)
        # ------------------------------------------------------------------
        latents = torch.randn((len(prompts), 3, H_latent, W_latent), device=device, dtype=dtype)

        # ------------------------------------------------------------------
        # 3  DDIM sampling loop (vectorised over batch)
        # ------------------------------------------------------------------
        for t in tqdm(scheduler.timesteps.to(device), desc="Sampling"):
            noise_in  = torch.cat([latents]*2)               # (2B, 3, H, W)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype is torch.float16):
                model_output = diffusion(
                    noise_in, timesteps=torch.Tensor((t,)).to(latents.device).long(), context=prompt_embeds
                )
                # might need to change timestep because monai expects timesteps: timestep torch.Tensor (N,).
                #model_out = diffusion(
                    #noise_in, timesteps=timesteps = torch.full((2 * len(prompts),), t, device=device, dtype=torch.long), context=prompt_embeds
                #)
                
            noise_uncond, noise_cond = model_output.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            latents, _ = scheduler.step(noise_pred, t, latents)     # updates in-place

        # ------------------------------------------------------------------
        # 4  Decode to pixel space    → (B, 1, H, W)
        # ------------------------------------------------------------------
        with torch.no_grad(), torch.cuda.amp.autocast(dtype is torch.float16):
            imgs = autoencoderkl.decode_stage_2_outputs(latents / scale_factor)
        imgs = imgs.clamp(0, 1).mul_(255).byte().cpu().numpy()      # (B, 1, H, W)

        # ------------------------------------------------------------------
        # 5  Save PNGs & remember paths
        # ------------------------------------------------------------------
        for i, arr in enumerate(imgs):
            path = os.path.join(paths["imgs_output"], f"sample_{start + i}.png")
            Image.fromarray(arr[0]).save(path)
            syn_paths.append(path)

        torch.cuda.empty_cache()        # keeps footprint predictable

    # ----------------------------------------------------------------------
    # Final bookkeeping
    # ----------------------------------------------------------------------
    df["syn_path"] = syn_paths
    return df


# call function on appropiate subset of dataframe with prepared prompts
print(f"Generating {len(df)} synthetic images...")
df = generate_images_from_reports(df, batch_size=batch_size)

# create new datframe with only syn_path and report (columns) that generated it
cols_keep = ['sex', 'age_group', 'label', 'report', 'syn_path']
sub_df = df[cols_keep]
# save dataframe
sub_df.to_csv(paths["result_csv"], index=False)
print("Saved results csv...")

# Cleanup after training
torch.cuda.empty_cache()