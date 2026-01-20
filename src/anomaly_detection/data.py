# data.py
import os
import argparse
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image

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

        # Filter for normal data only in training
        if split == "train":
            paths = [p for p in paths if "good" in Path(p).parts]

        self.image_paths = paths

        # Assign labels: 0 for good, 1 for anomaly
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


def build_transform(img_size: int, args=None, is_saving=False):
    """
    Builds the transform pipeline.
    Args:
        is_saving (bool): If True, returns PIL image (no normalization/tensor conversion).
    """
    transforms = [T.Resize((img_size, img_size))]

    # Apply augmentations only if requested
    if args and args.augment and hasattr(args, "aug_types"):
        if "rotation" in args.aug_types:
            transforms.append(T.RandomRotation(degrees=args.rot_degrees))

        if "color" in args.aug_types:
            transforms.append(
                T.ColorJitter(
                    brightness=args.color_brightness,
                    contrast=args.color_contrast,
                    saturation=args.color_saturation,
                )
            )

        if "blur" in args.aug_types:
            transforms.append(T.GaussianBlur(kernel_size=args.blur_kernel))

    # Add Normalization unless we are saving images to disk
    if not is_saving:
        transforms.append(T.ToTensor())
        transforms.append(
            T.Normalize(
                mean=(0.485, 0.456, 0.406),  # ImageNet stats
                std=(0.229, 0.224, 0.225),
            )
        )

    return T.Compose(transforms)


def build_dataloaders(args):
    # Train transform: Resize + Augment (if enabled) + Norm
    train_transform = build_transform(args.img_size, args=args, is_saving=False)

    # Test transform: Resize + Norm (No augmentation)
    test_transform = build_transform(args.img_size, args=None, is_saving=False)

    train_dataset = MVTecDataset(args.data_root, args.class_name, split="train", transform=train_transform)
    test_dataset = MVTecDataset(args.data_root, args.class_name, split="test", transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_dataset, test_dataset, train_loader, test_loader


def get_defect_type_from_path(path, class_name):
    p = Path(path)
    parts = p.parts

    if "good" in parts:
        return "good"

    if "test" in parts:
        if "test" in parts:
            idx = parts.index("test")
            if idx + 1 < len(parts):
                return parts[idx + 1]

    return "unknown"


def load_ground_truth_mask(img_path, data_root, class_name, img_size):
    img_path = Path(img_path)
    defect_type = get_defect_type_from_path(img_path, class_name)

    if defect_type == "good":
        return None

    stem = img_path.stem
    mask_name = stem + "_mask.png"

    mask_path = Path(data_root) / class_name / "ground_truth" / defect_type / mask_name

    if not mask_path.exists():
        return None

    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((img_size, img_size), resample=Image.NEAREST)
    mask_np = np.array(mask)

    mask_bin = (mask_np > 0).astype(np.uint8)
    return mask_bin


def save_augmented_dataset(args):
    """
    Generates and saves augmented images to disk without normalization.
    """
    print(f"Saving augmented dataset to: {args.save_aug_path}")

    # Transform: Resize + Augment (No Tensor/Norm)
    aug_transform = build_transform(args.img_size, args=args, is_saving=True)

    # Load raw dataset
    dataset = MVTecDataset(args.data_root, args.class_name, split="train", transform=None)

    output_dir = Path(args.save_aug_path) / f"{args.class_name}_augmented" / "train" / "good"
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for _, _, img_path in dataset:
        original_img = Image.open(img_path).convert("RGB")

        # Generate N augmentations per image
        for i in range(args.aug_multiplier):
            aug_img = aug_transform(original_img)

            stem = Path(img_path).stem
            save_name = f"{stem}_aug_{i}.png"

            aug_img.save(output_dir / save_name)
            count += 1

    print(f"Process complete. Saved {count} images.")


def get_args():
    parser = argparse.ArgumentParser()

    # Data params
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--class_name", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)

    # Augmentation params
    parser.add_argument("--augment", action="store_true", help="Enable augmentation")
    parser.add_argument("--aug_types", nargs="+", default=[], choices=["rotation", "color", "blur"])
    parser.add_argument(
        "--aug_multiplier",
        type=int,
        default=1,
        help="Num. augmentations per image (save mode only)",
    )

    # Augmentation hyperparameters
    parser.add_argument("--rot_degrees", type=float, default=10)
    parser.add_argument("--color_brightness", type=float, default=0.2)
    parser.add_argument("--color_contrast", type=float, default=0.2)
    parser.add_argument("--color_saturation", type=float, default=0.2)
    parser.add_argument("--blur_kernel", type=int, default=3)

    # Save mode
    parser.add_argument("--save_aug_path", type=str, default=None, help="Path to save augmented images")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.save_aug_path:
        save_augmented_dataset(args)
    else:
        # Debug / Loading test
        print(f"Initializing loaders for class: {args.class_name}")
        train_ds, test_ds, train_loader, test_loader = build_dataloaders(args)
        print(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")
