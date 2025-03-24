import yaml
import pandas as pd
import numpy as np
import os
import shutil
from tqdm import tqdm


def load_config(path="config/data_filtering/data_filtering.yaml"):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def convert_age_numeric(df, age_col):
    # convert to numeric and keep nulls as nan
    df[age_col] = pd.to_numeric(df[age_col].str.rstrip('Y'), errors='coerce')


def get_valid_patients(config_data_paths, config_subgroup_attributes, config_col_binary_target):
    """
    This function filters the patient metadata to keep patients with non null age and gender, as well as a diagnosis.
    Also, we only consider instances with CT scans modality.
    It returns a dictionary with 'Patient ID' as key and 1 value attributes for 'Series Number'.
    This dictionary will help filter the folder of images per patient.
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
    df[config_col_binary_target['new_col_name']] = np.where(df[non_binary_col].isin([2, 3]), 1, 0) # 1 for 2 or 3, 0 for 0 and 1
    
    # save the new filtered metadata file for later use
    df.to_csv(config_data_paths['new_metadata'], index=False)

    # Filter the DataFrame to only include necessary columns to filter image folder
    sub_df = df[['Patient ID', 'Series Number']]
    # Convert to a dictionary with 'Patient ID' as the key and the rest as a nested dictionary
    filtering_criteria_dict = sub_df.set_index('Patient ID').apply(lambda row: row.to_dict(), axis=1).to_dict()
    return filtering_criteria_dict


def process_patients(patients_info, config_data_paths):
    """
    This function takes the original downloaded images dataset (with all patients, and image modalties), and filters it.
    It only keeps patients obtained from the filtering of the metadata on CT modality, non null sex, age, and diagnosis.
    It creates a new directory.

    To ensure the right modality of images is kept for each valid patient, we compare the sub-folder names to the "Series Number",

    Additionally, each XML of the patients is also kept.
    """

    src_dir = config_data_paths['patient_images']
    dst_dir = config_data_paths['new_images']

    # Create the destination directory if it does not exist
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    patient_folders = os.listdir(src_dir)

    # Process each patient directory with a progress bar
    for patient_folder in tqdm(patient_folders, desc='Processing patients', unit='folder'):
        patient_id = patient_folder
        # If the patient is in the valid patients dictionary
        if patient_id in patients_info:
            src_patient_path = os.path.join(src_dir, patient_folder)
            dst_patient_path = os.path.join(dst_dir, patient_folder)

            # Ensure destination patient folder exists
            if not os.path.exists(dst_patient_path):
                os.makedirs(dst_patient_path)

            # Process each subfolder in the patient directory
            for subfolder in os.listdir(src_patient_path):
                subfolder_path = os.path.join(src_patient_path, subfolder)
                if os.path.isdir(subfolder_path):
                    # Check subsubfolders
                    for subsubfolder in os.listdir(subfolder_path):
                        subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
                        if os.path.isdir(subsubfolder_path):
                            # Extract series number and check against dictionary of valid patients
                            series_number = int(float(subsubfolder.split('-')[0]))
                            # Find XML and DICOM files
                            xml_file = [f for f in os.listdir(subsubfolder_path) if f.endswith('.xml')]
                            dicom_files = [f for f in os.listdir(subsubfolder_path) if f.endswith('.dcm')]
                            
                            # Check if this subsubfolder matches the patient CT serie's info
                            patient_series_number = int(patients_info[patient_id]['Series Number'])
                            if series_number == patient_series_number:
                                # Copy all DICOM files and the XML file to the new destination
                                for file in dicom_files + xml_file:
                                    shutil.copy(os.path.join(subsubfolder_path, file), dst_patient_path)

    
    # Check for missing patients by comparing folder names in dst_dir with patient IDs in the dictionary
    processed_patient_folders = os.listdir(dst_dir)
    missing_patients = [id for id in patients_info.keys() if id not in processed_patient_folders]
    print(f"Data transfer complete. {len(processed_patient_folders)} patient folders present on destination directory.")

    if missing_patients:
        print("The following valid patient(s) were not succesfully processed:", missing_patients)



def main():
    # Load the configuration
    config = load_config()

    # Access configuration parameters
    config_data_paths = config['data_paths']
    config_subgroup_attributes = config['subgroup_attributes']
    config_col_binary_target = config['col_binary_target']

    # get the patient ids we will keep
    filtering_criteria_dict = get_valid_patients(config_data_paths, config_subgroup_attributes, config_col_binary_target)
    print(f"Patients detected to be kept: {len(filtering_criteria_dict)}")

    # filter the images with the appropiate patients and image modalities, and save it to new directory
    process_patients(filtering_criteria_dict, config_data_paths)

if __name__ == "__main__":
    main()