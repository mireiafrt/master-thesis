import yaml
import pandas as pd
import numpy as np

def load_config(path="config/data_filtering/data_filtering.yaml"):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_valid_patient_ids(config_data_paths, config_subgroup_attributes, config_col_binary_target):
    """
    This function filters the patient metadata to keep patients with non null age and gender, as well as a diagnosis.
    Also, we only consider instances with CT scans modality.
    It returns an array of patient ids fullfilling the above criteria.
    """
    # read patient metadata
    patient_df = pd.read_excel(config_data_paths['patient_metadata'])

    # convert the patients age to numeric
    convert_age_numeric(patient_df, 'Patient Age')
    
    # Initialize the filter to select rows where 'Modality' is 'CT'
    filter_condition = (patient_df['Modality'] == 'CT')
    # Extend the filter to exclude NaN values in configurable attributes
    for attribute in config_subgroup_attributes:
        filter_condition &= patient_df[attribute].notna()
    # Apply the filter to the DataFrame
    patient_df = patient_df[filter_condition]
    
    # read diagnosis metadata
    df_diagnosis = pd.read_excel(config_data_paths['patient_diagnosis'])

    # merge datasets to find inner join of patient ids
    df = pd.merge(patient_df, df_diagnosis, left_on='Patient ID', right_on='TCIA Patient ID', how='inner')

    # create new bianry diagnosis column
    non_binary_col = config_col_binary_target['old_non_binary_col']
    df[config_col_binary_target['new_col_name']] = np.where(df[non_binary_col] == 0,np.nan, np.where(df[non_binary_col].isin([2, 3]), 1, 0))
    
    # save the new filtered metadata file for later use
    df.to_csv(config_data_paths['new_metadata'], index=False)

    # return patient ids in filtered metadata
    return df['Patient ID'].unique()


def convert_age_numeric(df, age_col):
    # convert to numeric and keep nulls as nan
    df[age_col] = pd.to_numeric(df[age_col].str.rstrip('Y'), errors='coerce')


def main():
    # Load the configuration
    config = load_config()

    # Access data paths and subgroup attributes from the configuration
    config_data_paths = config['data_paths']
    config_subgroup_attributes = config['subgroup_attributes']
    config_col_binary_target = config['col_binary_target']

    # get the patient ids we will keep
    filtered_patient_ids = get_valid_patient_ids(config_data_paths, config_subgroup_attributes, config_col_binary_target)
    print(len(filtered_patient_ids))

if __name__ == "__main__":
    main()