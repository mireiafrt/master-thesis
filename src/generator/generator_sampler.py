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

device = torch.device("cuda")