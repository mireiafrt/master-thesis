import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config(path="config/data_split/data_split_covid.yaml"):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def undersample_positive(df, groups_to_reduce):
    # Create unique patient-level DataFrame
    df_patients = df.drop_duplicates(subset="patient_id")[["patient_id", "sex", "age_group", "label"]].copy()
    df_patients["use"] = True

    # Downsample patients from large groups
    for group in groups_to_reduce:
        mask = (
            (df_patients["sex"] == group["sex"]) &
            (df_patients["age_group"] == group["age_group"]) &
            (df_patients["label"] == group["label"])
        )
        to_drop = df_patients[mask].sample(frac=group["drop_fraction"], random_state=42).index
        df_patients.loc[to_drop, "use"] = False

    # Merge `use` flag back to slice-level DataFrame
    df_filtered = df.merge(df_patients[["patient_id", "use"]], on="patient_id", how="left")
    #Filter final dataset
    df_filtered = df_filtered[df_filtered["use"] == True]

    # Show new class distribution
    print("Label distribution after filtering:")
    print(df_filtered["label"].value_counts(normalize=True))
    print("\nDetailed distribution:")
    print(df_filtered[['sex','age_group','label']].value_counts())
    print("\nNew size:")
    print(len(df_filtered))
    print("\nNumber of patients:")
    print(df_filtered['patient_id'].nunique())

    return df_filtered

def is_minority(row):
    return row['age_group'] in ['Under 20', '20-40', 'Over 80']

# Custom function to apply per patient
def extract_patient_info(patient_df):
    # Get mode age_group with fallback
    age_group_mode = patient_df['age_group'].mode()
    age_group_final = age_group_mode.iloc[0] if len(age_group_mode) == 1 else patient_df.iloc[0]['age_group']
    
    # Get mode gender with fallback
    sex_mode = patient_df['sex'].mode()
    sex_final = sex_mode.iloc[0] if len(sex_mode) == 1 else patient_df.iloc[0]['sex']

    # Get mode diagnosis with fallback
    diagnosis_mode = patient_df['label'].mode()
    diagnosis_final = diagnosis_mode.iloc[0] if len(diagnosis_mode) == 1 else patient_df.iloc[0]['label']
    
    return pd.Series({
        'sex': sex_final,
        'diagnosis': diagnosis_final,
        'age_group': age_group_final
    })

def split_patients(df, stratify_columns, split_percentages):
    # Get unique patients with their age/gender info, apply this function per patient on the age_group
    patient_info = df.groupby('patient_id').apply(extract_patient_info).reset_index()
    patient_info['is_minority'] = patient_info.apply(is_minority, axis=1)

    # Step 1: Split minority patients into GT and remaining
    minority_patients = patient_info[patient_info['is_minority']]
    majority_patients = patient_info[~patient_info['is_minority']]
    gt_patients, minority_remain = train_test_split(minority_patients,
        train_size=split_percentages['gt_split']['train_size'],
        test_size=split_percentages['gt_split']['test_size'],
        stratify=minority_patients[stratify_columns],
        random_state=42
    )

    # Step 2: Combine remaining minority + majority → Remaining patients
    remaining_patients = pd.concat([minority_remain, majority_patients])

    # Step 3: Split remaining into test, train-val, and then train-val
    temp_train_val, test_patients= train_test_split(remaining_patients,
        train_size=split_percentages['test_split']['train_size'],
        test_size=split_percentages['test_split']['test_size'],
        stratify=remaining_patients[stratify_columns],
        random_state=42
    )
    train_patients, val_patients = train_test_split(temp_train_val,
        train_size=split_percentages['train_split']['train_size'],
        test_size=split_percentages['train_split']['test_size'],
        stratify=temp_train_val[stratify_columns],
        random_state=42
    )

    # Step 4: Assign split labels
    def assign_split(df, split_name):
        df = df[['patient_id']].copy()
        df['split'] = split_name
        return df

    split_assignments = pd.concat([
        assign_split(gt_patients, 'ground_truth'),
        assign_split(test_patients, 'test'),
        assign_split(train_patients, 'train'),
        assign_split(val_patients, 'val')
    ])

    # Step 5: Merge back
    df_split = df.merge(split_assignments, on='patient_id', how='left')

    # print stats
    print("Patients w/out split:",df_split['split'].isna().sum())
    print("\nDistribution of split:")
    print(df_split['split'].value_counts(normalize=True))
    print("\nLabel:")
    print(df_split['finding'].value_counts(normalize=True))

    return df_split


def main():
    # Load the configuration
    config = load_config()
    # Access configuration parameters
    data_paths = config['data_paths']
    stratify_columns = config['stratify_columns']
    groups_to_reduce = config["groups_to_reduce"]
    split_percentages = config['split_percentages']

    # read metadata
    df = pd.read_csv(data_paths['metadata_path'])

    # undersample positivies
    df = undersample_positive(df, groups_to_reduce)

    # split dataset at patient level
    df = split_patients(df, stratify_columns, split_percentages)

    # save to output
    df.to_csv(data_paths['output_path'], index=False)


if __name__ == "__main__":
    main()