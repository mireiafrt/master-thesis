import os
import pandas as pd
from monai.transforms import LoadImage
import nibabel as nib

# Config
NIFTI_ROOT = "data/preprocessed_nifti"
METADATA_PATH = "data/metadata.csv"
SPLITS = ["train", "val", "test", "ground_truth"]

# Load metadata (to cross-check patient IDs)
metadata = pd.read_csv(METADATA_PATH)
valid_ids = set(metadata["Patient ID"].astype(str))

# MONAI loader
loader = LoadImage(image_only=True, ensure_channel_first=True)

for split in SPLITS:
    print(f"\n🔍 Checking NIfTI volumes in split: {split}")
    split_dir = os.path.join(NIFTI_ROOT, split)
    if not os.path.exists(split_dir):
        print(f"❌ Split folder not found: {split_dir}")
        continue

    for fname in sorted(os.listdir(split_dir)):
        if not fname.endswith(".nii") and not fname.endswith(".nii.gz"):
            continue

        patient_id = fname.replace(".nii.gz", "").replace(".nii", "")
        nifti_path = os.path.join(split_dir, fname)

        if patient_id not in valid_ids:
            print(f"⚠️ {patient_id}: not found in metadata")

        try:
            image = loader(nifti_path)
        except Exception as e:
            print(f"❌ {patient_id}: Failed to load NIfTI — {e}")
            continue

        if image.ndim != 4 or image.shape[0] != 1:
            print(f"⚠️ {patient_id}: Unexpected shape: {image.shape}")
        else:
            print(f"✅ {patient_id}: shape = {image.shape}")
