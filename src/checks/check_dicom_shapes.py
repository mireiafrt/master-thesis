import os
import pydicom
import numpy as np

def load_dicom_volume(base_path, patient_id):
    patient_path = os.path.join(base_path, patient_id)
    dicom_files = [
        os.path.join(patient_path, f)
        for f in os.listdir(patient_path)
        if f.lower().endswith(".dcm")
    ]

    if len(dicom_files) < 2:
        print(f"⚠️ {patient_id}: Only {len(dicom_files)} slices — skipping.")
        return

    try:
        dicoms = [pydicom.dcmread(f, force=True) for f in dicom_files]
        dicoms = [d for d in dicoms if hasattr(d, "ImagePositionPatient")]
        dicoms.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except Exception as e:
        print(f"❌ {patient_id}: Failed to read or sort DICOMs — {e}")
        return

    try:
        slices = [d.pixel_array for d in dicoms]
        volume = np.stack(slices).astype(np.float32)
    except Exception as e:
        print(f"❌ {patient_id}: Failed to stack slices — {e}")
        return

    # Check shape and consistency
    shape = volume.shape  # [D, H, W]
    if volume.ndim != 3:
        print(f"❌ {patient_id}: Volume shape invalid: {shape}")
    elif shape[0] < 10:
        print(f"⚠️ {patient_id}: Too few slices: {shape}")
    elif shape[0] == shape[1] or shape[0] == shape[2]:
        print(f"⚠️ {patient_id}: Suspect depth position — shape: {shape}")
    else:
        print(f"✅ {patient_id}: OK — shape {shape}")

def main():
    dicom_split_path = "data/data_split/train" # have to check for train, val, test, and gt
    print(f"🔍 Checking DICOM folders in: {dicom_split_path}\n")

    for patient_id in sorted(os.listdir(dicom_split_path)):
        full_path = os.path.join(dicom_split_path, patient_id)
        if not os.path.isdir(full_path):
            continue
        load_dicom_volume(dicom_split_path, patient_id)

if __name__ == "__main__":
    main()
