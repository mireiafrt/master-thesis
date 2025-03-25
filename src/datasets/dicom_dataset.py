import os
import pydicom
import numpy as np
from torch.utils.data import Dataset

class DICOM3DDataset(Dataset):
    def __init__(self, patient_ids, labels_dict, base_dir, transform=None):
        self.patient_ids = patient_ids
        self.labels_dict = labels_dict
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        label = self.labels_dict[patient_id]
        volume = self.load_volume(patient_id)

        if self.transform:
            volume = self.transform(volume)

        return volume, label

    def load_volume(self, patient_id):
        path = os.path.join(self.base_dir, patient_id)

        # Read all .dcm files in the patient folder
        dicom_files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(".dcm")
        ]

        if len(dicom_files) < 2:
            raise ValueError(f"{patient_id}: Only {len(dicom_files)} slices — skipping.")

        # Read files with force=True and filter out non-2D slices
        try:
            dicoms = [pydicom.dcmread(f, force=True) for f in dicom_files]
            dicoms = [d for d in dicoms if hasattr(d, "ImagePositionPatient") and len(d.pixel_array.shape) == 2]
        except Exception as e:
            raise RuntimeError(f"{patient_id}: Failed to read DICOMs — {e}")

        if len(dicoms) < 2:
            raise ValueError(f"{patient_id}: Not enough valid slices after filtering.")

        # Sort by Z-position
        try:
            dicoms.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except Exception as e:
            raise ValueError(f"{patient_id}: Missing or invalid ImagePositionPatient — {e}")

        # Stack pixel arrays into [D, H, W]
        try:
            volume = np.stack([d.pixel_array for d in dicoms]).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"{patient_id}: Failed to stack pixel arrays — {e}")

        # Sanity checks
        if volume.ndim != 3:
            raise ValueError(f"{patient_id}: Volume shape is invalid: {volume.shape}")
        if volume.shape[0] < 10:
            raise ValueError(f"{patient_id}: Volume too thin: {volume.shape}")

        # Add channel dimension → [C, D, H, W]
        volume = np.expand_dims(volume, axis=0)

        return volume

