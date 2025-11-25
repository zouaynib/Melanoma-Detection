import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ISIC2019Dataset(Dataset):
    """
    PyTorch Dataset for the ISIC 2019 skin lesion classification challenge.
    """

    def __init__(self, img_dir, csv_path, transform=None):
        """
        Args:
            img_dir (str): directory containing all JPEG images.
            csv_path (str): path to ISIC_2019_Training_GroundTruth.csv.
            transform (callable, optional): Optional transform applied to each image.
        """
        self.img_dir = img_dir
        self.data = pd.read_csv(csv_path)
        self.transform = transform

        # Extract image IDs
        self.image_ids = self.data["image"].values

        # Extract labels (one-hot → convert to integer class index)
        self.class_cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
        self.labels = self.data[self.class_cols].values.argmax(axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Apply transforms (if provided)
        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label
