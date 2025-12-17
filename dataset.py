import os
import csv
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset


class ISIC2019Dataset(Dataset):
    """
    Reads ISIC 2019 GroundTruth CSV in one-hot format.

    First column: image id
    Remaining columns: class one-hot (can be 8 or 9 depending on CSV, e.g. +UNK)

    Returns:
        (PIL.Image or transformed tensor, int_label)
    """

    def __init__(self, csv_path: str, img_dir: str, transform=None, drop_zero_labels: bool = True):
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.transform = transform
        self.drop_zero_labels = drop_zero_labels

        self.image_ids: List[str] = []
        self.images: List[str] = []
        self.labels: List[int] = []
        self.class_names: List[str] = []

        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        with open(self.csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            if len(header) < 2:
                raise ValueError("CSV header seems invalid (expected image_id + class columns).")

            self.class_names = header[1:]
            n_classes = len(self.class_names)
            if n_classes < 2:
                raise ValueError(f"Expected >=2 class columns, got {n_classes}. Header: {header}")

            for row in reader:
                if len(row) < 1 + n_classes:
                    continue

                image_id = row[0].strip()
                try:
                    label_vec = list(map(float, row[1:1 + n_classes]))
                except ValueError:
                    continue

                if self.drop_zero_labels and sum(label_vec) == 0:
                    continue

                label = int(max(range(n_classes), key=lambda i: label_vec[i]))

                img_path = os.path.join(self.img_dir, image_id + ".jpg")
                if not os.path.exists(img_path):
                    continue

                self.image_ids.append(image_id)
                self.images.append(img_path)
                self.labels.append(label)

        if len(self.images) == 0:
            raise RuntimeError("Dataset is empty after loading. Check CSV paths and image folder.")

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[object, int]:
        img = Image.open(self.images[idx]).convert("RGB")
        y = int(self.labels[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, y


class TransformSubset(Dataset):
    def __init__(self, subset: Dataset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        img, y = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, y
