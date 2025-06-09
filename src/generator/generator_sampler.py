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
from generative.networks.schedulers import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer


# set global seed
set_determinism(42)

# Load config
with open("config/generator/generator_sample.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
columns = config["columns"]
num_inference_steps = config["num_inference_steps"]
guidance_scale = config["guidance_scale"]
scale_factor = float(config["scale_factor"])
sample_size = config["sample_size"]
filters = config["conditioning"]

print("Preparing paths ...")
# make sure images output path exists
os.makedirs(paths["imgs_output"], exist_ok=True)

########## PREPARE PROMPTS and RESULT CSV ##########
metadata = pd.read_csv(paths["metadata_csv"])
metadata = metadata[metadata["use"] == True]
# filter metadata based on the conditioning
df = metadata.copy()
for key, value in filters.items():
    if value is not None:
        df = df[df[key] == value]
df = df.reset_index(drop=True)
print("Filtered data...")

# Sample a subset instead of full data if it is none
if sample_size is not None:
    df = df.iloc[:sample_size].reset_index(drop=True)
    print(f"Subsampled to {sample_size} rows.")

# Create "report" col --> Sentence like "Female, age group under 20, healthy" or "x, x, with COVID-19"
def build_clip_prompt(row):
    sex_term = "female" if row["sex"] == "F" else "male"
    diagnosis_text = "healthy" if row["label"] == 0 else "with COVID-19"
    return f"A {sex_term} patient in the {row['age_group']} age group, {diagnosis_text}."
df["report"] = df.apply(build_clip_prompt, axis=1)
print("Reports created ...")

########## LOAD MODELS INTO ARCHITECTURES ##########
device = torch.device("cuda")

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
autoencoderkl.load_state_dict(torch.load(paths["autoencoder_path"])) # load the trained autoencoder model
autoencoderkl = autoencoderkl.to(device)
autoencoderkl.eval()

# Load Diffusion Model
print(f"Loading Diffusion model from {paths['generator_path']}")
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
diffusion.load_state_dict(torch.load(paths["generator_path"])) # load the trained diffusion model
diffusion = diffusion.to(device)
diffusion.eval()

# load scheduler (now the DDIM instead of DDPM)
scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.0015, beta_end=0.0205, schedule="scaled_linear", prediction_type="v_prediction", clip_sample=False)
scheduler.set_timesteps(num_inference_steps)

# set up tokenizer and text encoder
tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder")

# function to iterate through prompts and generate one image per prompt
def generate_images_from_reports(df):
    
    syn_paths = []
    # loop through prompts
    for i, row in tqdm(df.iterrows(), total=len(df)):
        report = row["report"]
        # --- Tokenize with empty prompt and prompt for classifier-free guidance ---
        prompt = ["", report]
        text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_input_ids = text_inputs.input_ids

        prompt_embeds = text_encoder(text_input_ids.squeeze(1))
        prompt_embeds = prompt_embeds[0].to(device)

        # Sample random latent noise (3 channels, size 256 by 256 image)
        noise = torch.randn((1, 3, 256, 256)).to(device)

        # Start sampling process
        with torch.no_grad():
            progress_bar = tqdm(scheduler.timesteps)
            for t in progress_bar:
                noise_input = torch.cat([noise] * 2) # duplicate noise for cfg
                model_output = diffusion(
                    noise_input, timesteps=torch.Tensor((t,)).to(noise.device).long(), context=prompt_embeds
                )
                noise_pred_uncond, noise_pred_text = model_output.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                noise, _ = scheduler.step(noise_pred, t, noise)

        # Decode latent to image
        with torch.no_grad():
            sample = autoencoderkl.decode_stage_2_outputs(noise/scale_factor)

        # Save image
        sample = np.clip(sample.cpu().numpy(), 0, 1)
        sample = (sample * 255).astype(np.uint8)
        im = Image.fromarray(sample[0, 0])
        output_path = os.path.join(paths["imgs_output"], f"sample_{i}.png")
        im.save(output_path)

        # Log path to array of paths
        syn_paths.append(output_path)

    # save column of paths to synthetic image
    df['syn_path'] = syn_paths
    return df


# call function on appropiate subset of dataframe with prepared prompts
df = generate_images_from_reports(df)

# create new datframe with only syn_path and report (columns) that generated it
cols_keep = ['sex', 'age_group', 'label', 'report', 'syn_path']
sub_df = df[cols_keep]
# save dataframe
sub_df.to_csv(paths["result_csv"], index=False)

# Cleanup after training
torch.cuda.empty_cache()