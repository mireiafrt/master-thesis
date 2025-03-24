import pandas as pd
import yaml
from sklearn.model_selection import StratifiedShuffleSplit

# Load Config
with open("config/data_split/data_split.yaml", "r") as f:
    config = yaml.safe_load(f)

metadata_path = config["metadata_path"]
stratify_columns = config["stratify_columns"]
split_sizes = config["split_sizes"]
min_group_size = config.get("min_group_size", 4)

# Load Metadata
metadata = pd.read_csv(metadata_path)

# Create 'age_group' if it doesn't exist
if "age_group" not in metadata.columns:
    bins = [0, 20, 40, 60, 80, float("inf")]
    labels = ["Under 20", "20-40", "40-60", "60-80", "Over 80"]
    metadata["age_group"] = pd.cut(metadata["Patient Age"], bins=bins, labels=labels, right=False)

# Create startigied groups by combining "Patient Sex" and "age_group" which are in stratify_columns
metadata["stratify_group"] = metadata[stratify_columns].astype(str).agg("_".join, axis=1)

# Identify and separate small groups
group_counts = metadata["stratify_group"].value_counts()
small_groups = group_counts[group_counts < min_group_size].index
# Group all groups with low count into an "other" group
metadata["stratify_group"] = metadata["stratify_group"].apply(
    lambda x: "other" if x in small_groups else x
)

# Separate the "other" group
other_group = metadata[metadata["stratify_group"] == "other"].copy()
stratifiable = metadata[metadata["stratify_group"] != "other"].copy()


# -------- Stratified split on stratifiable patients --------

# Compute ground truth target size
total_n = len(metadata) # number of patients
gt_target_size = int(split_sizes["ground_truth"] * total_n)
gt_remaining_needed = gt_target_size - len(other_group)

if gt_remaining_needed < 0:
    raise ValueError("The 'other' group is larger than the desired ground_truth size!")

# Stratified sampling for ground_truth from remaining
sss_gt = StratifiedShuffleSplit(n_splits=1, test_size=gt_remaining_needed, random_state=42)
_, gt_idx = next(sss_gt.split(stratifiable, stratifiable["stratify_group"]))
gt_stratified = stratifiable.iloc[gt_idx].copy()
rest = stratifiable.drop(index=gt_stratified.index).copy()

# Final ground truth = other + stratified sample
ground_truth = pd.concat([other_group, gt_stratified])

# Split rest into train/val/test
remaining_ratio = 1.0 - split_sizes["ground_truth"]
train_ratio = split_sizes["train"] / remaining_ratio
val_ratio = split_sizes["val"] / remaining_ratio
test_ratio = split_sizes["test"] / remaining_ratio

# Split rest into train_val and test
sss_1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
train_val_idx, test_idx = next(sss_1.split(rest, rest["stratify_group"]))
train_val = rest.iloc[train_val_idx].copy()
test = rest.iloc[test_idx].copy()

# Split train_val into train and val
val_relative = val_ratio / (train_ratio + val_ratio)
sss_2 = StratifiedShuffleSplit(n_splits=1, test_size=val_relative, random_state=42)
train_idx, val_idx = next(sss_2.split(train_val, train_val["stratify_group"]))
train = train_val.iloc[train_idx].copy()
val = train_val.iloc[val_idx].copy()


# Final assignment
metadata["split"] = "unassigned"
metadata.loc[train.index, "split"] = "train"
metadata.loc[val.index, "split"] = "val"
metadata.loc[test.index, "split"] = "test"
metadata.loc[ground_truth.index, "split"] = "ground_truth"

# Drop helper column
metadata.drop(columns=["stratify_group"], inplace=True)

# Save output
print(metadata["split"].value_counts())
metadata.to_csv(metadata_path, index=False)
