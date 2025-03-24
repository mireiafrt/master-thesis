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
        dicom_files = [
            pydicom.dcmread(os.path.join(path, f))
            for f in os.listdir(path) if f.endswith(".dcm")
        ]
        dicom_files.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        volume = np.stack([f.pixel_array for f in dicom_files]).astype(np.float32)
        volume = np.expand_dims(volume, axis=0)  # [C, D, H, W]
        return volume
