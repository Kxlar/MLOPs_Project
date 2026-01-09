# data.py
import os
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class MVTecDataset(Dataset):
    def __init__(self, root, class_name, split="train", transform=None):
        super().__init__()
        assert split in ["train", "test"]
        self.root = root
        self.class_name = class_name
        self.split = split
        self.transform = transform

        pattern = os.path.join(root, class_name, split, "*", "*.*")
        paths = sorted(glob(pattern))

        if split == "train":
            paths = [p for p in paths if "good" in Path(p).parts]

        self.image_paths = paths

        if split == "test":
            self.labels = []
            for p in self.image_paths:
                self.labels.append(0 if "good" in Path(p).parts else 1)
        else:
            self.labels = [0] * len(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img_t = self.transform(img)
        else:
            img_t = T.ToTensor()(img)

        label = self.labels[idx]
        return img_t, label, img_path


def build_transform(img_size: int):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def build_dataloaders(data_root: str, class_name: str, img_size: int, batch_size: int = 8):
    transform = build_transform(img_size)

    train_dataset = MVTecDataset(data_root, class_name, split="train", transform=transform)
    test_dataset = MVTecDataset(data_root, class_name, split="test", transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,      # Windows safe
        pin_memory=True,
    )

    return train_dataset, test_dataset, train_loader


def get_defect_type_from_path(path, class_name):
    """
    path: string or Path to a test image
    returns: defect type string, e.g. 'color', 'cut', ... or 'good'
    """
    p = Path(path)
    parts = p.parts

    if "good" in parts:
        return "good"

    if "test" in parts:
        idx = parts.index("test")
        if idx + 1 < len(parts):
            return parts[idx + 1]

    return "unknown"


def load_ground_truth_mask(img_path, data_root, class_name, img_size):
    """
    returns: binary numpy array [img_size, img_size] with values 0/1, or None if good/no mask
    """
    img_path = Path(img_path)
    defect_type = get_defect_type_from_path(img_path, class_name)

    if defect_type == "good":
        return None

    stem = img_path.stem
    mask_name = stem + "_mask.png"

    mask_path = (
        Path(data_root)
        / class_name
        / "ground_truth"
        / defect_type
        / mask_name
    )

    if not mask_path.exists():
        print("No mask found for", img_path, "expected", mask_path)
        return None

    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((img_size, img_size), resample=Image.NEAREST)
    mask_np = np.array(mask)

    mask_bin = (mask_np > 0).astype(np.uint8)
    return mask_bin
