import os
import yaml

import pandas as pd
import re
import json
from collections import defaultdict
from tqdm import tqdm
from collections import Counter


def load_config(path="config/data_filtering/data_filtering_covid.yaml"):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_age_bins(df):
    bins = [0, 20, 40, 60, 80, float('inf')]
    labels = ['Under 20', '20-40', '40-60', '60-80', 'Over 80']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    return df

def create_binary_diagnosis(df):
    df['binary_diagnosis'] = (df['finding'] == 'COVID-19').astype(int)
    return df

def filter_valid_patients(df, keep_pneumonia):
    df = df[df['verified finding']=='Yes'] # keep A images 
    df = df[df['sex'].isna()==False]
    df = df[df['age'].isna()==False]
    df = df[df['finding'].isna()==False]

    # keep pneumonia cases or niot
    if keep_pneumonia==False:
        df = df[df['finding']!='Pneumonia']

    return df

def verify_unique_files(df):
    # Flatten all file names into a single list
    all_filenames = []

    for file_list_str in df['file_names']:
        files = json.loads(file_list_str)  # convert stringified list back to Python list
        all_filenames.extend(files)

    # Count occurrences
    filename_counts = Counter(all_filenames)
    # Get duplicates
    duplicates = {fname: count for fname, count in filename_counts.items() if count > 1}

    # Print results
    if duplicates:
        print("❌ Found duplicate file names used in multiple volumes:")
        for fname, count in duplicates.items():
            print(f"{fname} → used {count} times")
    else:
        print("✅ No duplicate file names found — each slice is used in only one volume.")


def match_patient_to_image(df, image_folder):
    # List all PNG files once
    all_png_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    # Organize matched files by (patient_id, series_number)
    matched_records = []

    # iterate through the patients in df
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Matching patients"):
        pid = row['patient id']
        source = row['source']
        matched = []

        if source == "CNCB":
            pattern = re.compile(rf"^{re.escape(pid)}_(\d+)_\d+\.png$")
            for fname in all_png_files:
                m = pattern.match(fname)
                if m:
                    series = m.group(1)
                    matched.append((series, fname))

        elif source == "COVID-19-CT-Seg":
            if pid.startswith("radiopaedia_"):
                pattern = re.compile(rf"^{re.escape(pid)}_(\d+)-\d+\.png$")
                for fname in all_png_files:
                    m = pattern.match(fname)
                    if m:
                        series = m.group(1)
                        matched.append((series, fname))
            else:
                pattern = re.compile(rf"^{re.escape(pid)}-\d+\.png$")
                for fname in all_png_files:
                    if pattern.match(fname):
                        matched.append((None, fname))

        elif source == "COVID-CTset":
            pattern = re.compile(rf"^.+_{re.escape(pid)}_SR_(\d+)_IM\d+\.png$")
            for fname in all_png_files:
                m = pattern.match(fname)
                if m:
                    series = m.group(1)
                    matched.append((series, fname))

        elif source == "radiopaedia.org":
            pattern = re.compile(rf"^{re.escape(pid)}-\d+-(\d+)-\d+\.png$")
            for fname in all_png_files:
                m = pattern.match(fname)
                if m:
                    series = m.group(1)
                    matched.append((series, fname))

        elif source == "iCTCF":
            pattern = re.compile(rf"^{re.escape(pid)}-\d+\.png$")
            for fname in all_png_files:
                if pattern.match(fname):
                    matched.append((None, fname))

        elif source == "STOIC":
            pattern = re.compile(rf"^{re.escape(pid)}-\d+\.png$")
            for fname in all_png_files:
                if pattern.match(fname):
                    matched.append((None, fname))

        elif source == "COVID-CT-MD":
            pattern = re.compile(rf"^{re.escape(pid)}-IM\d+\.png$")
            for fname in all_png_files:
                if pattern.match(fname):
                    matched.append((None, fname))

        else:
            # Unknown source, skip
            continue

        # Group by series_number
        series_groups = defaultdict(list)
        for series, fname in matched:
            series_groups[series].append(fname)

        # One row per (patient, series)
        for series, files in series_groups.items():
            matched_records.append({
                **row.to_dict(),
                'series_number': series,
                'slice_count': len(files),
                'file_names': json.dumps(sorted(files))  # Sort filenames for consistency
            })

    # Create new DataFrame with the results added
    df_expanded = pd.DataFrame(matched_records)
    # verify files are unique
    verify_unique_files(df_expanded)

    # view slice numbers
    print(df_expanded['slice_count'].describe())
    # view non matched patients
    print(df_expanded['file_names'].isna().sum())

    return df_expanded

def extract_slice_number(fname):
    # Extract all digit groups from the filename
    numbers = re.findall(r'\d+', fname)
    if numbers:
        return int(numbers[-1])  # take the last one (typically the slice number) right before .png
    return None

def has_slice_gaps(file_list):
    # Extract slice numbers from all filenames
    slice_indices = []
    for fname in file_list:
        sn = extract_slice_number(fname)
        if sn is not None:
            slice_indices.append(sn)

    # Sort and check for spacing gaps
    slice_indices.sort()
    gaps = [j - i for i, j in zip(slice_indices[:-1], slice_indices[1:])]
    max_gap = max(gaps) if gaps else 0

    return max_gap > 1  # Flag if any jump > 1 between slices

def check_slice_gaps(df):
    df['has_slice_gap'] = df['file_names'].apply(
        lambda s: has_slice_gaps(json.loads(s))
    )
    # View results
    print(df['has_slice_gap'].value_counts())

def create_slice_level_metadata(df, image_folder):
    slice_records = []
    for _, row in df.iterrows():
        file_list = json.loads(row['file_names'])
        for fname in file_list:
            slice_records.append({
                "fname": fname,
                "image_path": os.path.join(image_folder, fname),
                "patient_id": row["patient id"],
                "series_num": row["series_number"],
                "age": row.get("age", None),
                "age_group": row.get("age_group", None),
                "sex": row.get("sex", None),
                "finding": row.get("finding", None),
                "label": row.get("binary_diagnosis", None)
            })

    df_slices = pd.DataFrame(slice_records)
    print(df_slices[['sex','age_group','label']].value_counts(normalize=True))
    return df_slices


def main():
    # Load the configuration
    config = load_config()
    # Access configuration parameters
    config_data_paths = config['data_paths']
    config_filtering = config['filtering']

    # Read metadata file and filter patients
    df = pd.read_csv(config_data_paths['metadata_csv'])
    df = filter_valid_patients(df, config_filtering['keep_pneumonia'])

    # create new variables
    df = create_age_bins(df)
    df = create_binary_diagnosis(df)

    # Match file names to patients
    df_expanded = match_patient_to_image(df, config_data_paths['image_folder'])

    # check slice gaps
    check_slice_gaps(df_expanded)

    # create slice level metadata
    df_slices = create_slice_level_metadata(df_expanded, config_data_paths['image_folder'])
    print(f"Final slice-level dataset shape: {df_slices.shape}")

    # save new metadata
    df_slices.to_csv(config_data_paths['result_csv'], index=False)


if __name__ == "__main__":
    main()