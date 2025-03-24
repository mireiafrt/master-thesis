import os
import shutil
import pandas as pd
import yaml
from tqdm import tqdm

# Helper to remove macOS junk files
def remove_mac_metadata(folder_path):
    for dirpath, _, filenames in os.walk(folder_path):
        for fname in filenames:
            if fname.startswith("._") or fname == ".DS_Store":
                try:
                    os.remove(os.path.join(dirpath, fname))
                except Exception as e:
                    print(f"Could not delete {fname} in {dirpath}: {e}")

# Load Config
with open("config/data_split/data_split.yaml", "r") as f:
    config = yaml.safe_load(f)
metadata_path = config["metadata_path"]
source_root = config["source_root"]
target_root = config["target_root"]

# Load metadata
metadata = pd.read_csv(metadata_path)

# Create destination folders
split_names = metadata["split"].unique()
for split in split_names:
    os.makedirs(os.path.join(target_root, split), exist_ok=True)

# Copy patient folders to their corresponding split
for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Copying patient folders"):
    patient_id = str(row["Patient ID"])
    split = row["split"]

    src_folder = os.path.join(source_root, patient_id)
    dst_folder = os.path.join(target_root, split, patient_id)

    if os.path.exists(src_folder):
        try:
            shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)
            remove_mac_metadata(dst_folder)
        except Exception as e:
            print(f"Failed to copy {patient_id}: {e}")
    else:
        print(f"Patient folder not found: {src_folder}")
