import os
import SimpleITK as sitk
from tqdm import tqdm

def convert_dicom_folder_to_nifti(input_folder, output_folder):
    """
    Converts all DICOM series (one folder per patient) into NIfTI files.
    Preserves train/val/test split structure.
    """
    for split in os.listdir(input_folder):
        if split.startswith('.'):
            continue  # Skip .DS_Store or hidden files

        split_input_path = os.path.join(input_folder, split)
        split_output_path = os.path.join(output_folder, split)
        os.makedirs(split_output_path, exist_ok=True)

        for patient_id in tqdm(os.listdir(split_input_path), desc=f"Converting {split}"):
            patient_folder = os.path.join(split_input_path, patient_id)

            if not os.path.isdir(patient_folder):
                continue  # Skip non-directory entries

            output_path = os.path.join(split_output_path, f"{patient_id}.nii.gz")

            # Read and convert
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(patient_folder)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            sitk.WriteImage(image, output_path)

if __name__ == "__main__":
    input_folder = "data/data_split"               # path to folders with DICOMs
    output_folder = "data/preprocessed_nifti"      # path to save NIfTI files
    convert_dicom_folder_to_nifti(input_folder, output_folder)
    print("DICOM to NIfTI conversion completed.")
