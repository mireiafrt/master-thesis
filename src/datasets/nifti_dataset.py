import torch
from torch.utils.data import Dataset
from monai.transforms import Compose, LoadImage

class NiftiDataset(Dataset):
    def __init__(self, patient_ids, labels_dict, base_dir, transform=None):
        self.patient_ids = patient_ids
        self.labels_dict = labels_dict
        self.base_dir = base_dir
        self.transform = transform
        self.loader = LoadImage(image_only=True)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        image_path = f"{self.base_dir}/{patient_id}.nii.gz"
        label = self.labels_dict[patient_id]

        image = self.loader(image_path)
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)