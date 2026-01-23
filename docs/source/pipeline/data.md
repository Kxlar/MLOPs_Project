

This project uses the **MVTec AD** dataset.

## 1. Downloading Dataset
Our dataset can be downloaded from the cloud

```bash
chmod +x setup.sh
```
```bash
./setup.sh
```

## 2. Data Module Documentation

The data.py script handles data loading, preprocessing, and synthetic data generation (drift simulation). It manages the MVTec AD dataset for both training (normal only) and testing (normal + anomalies).

- Dataset Handling: Manages train/test splits, ensuring only "good" data is used for training while the test set includes anomalies with binary labels.
- Preprocessing: Applies resizing and normalization using ImageNet statistics.
- Augmentation Pipeline: Supports on-the-fly or offline augmentations (Rotation, Color Jitter, Blur) to simulate data drift or increase robustness.

## What it does:
1. Scans the file system to load MVTec images (train/test splits).
2. Applies preprocessing (resizing, normalization) and optional augmentations (rotation, blur, color jitter).
3. Loads pixel-level ground-truth masks for defect evaluation.
4. (CLI mode) Generates and saves a synthetic "drifted" dataset to disk for robustness testing.

## Arguments
--data_root: Path to the root directory of the dataset.

--class_name: The object category to analyze (e.g., carpet).

--img_size: Target image size for resizing (default: 224).

--batch_size: Batch size for data loading (default: 8).

--augment: Flag to enable data augmentation.

--aug_types: List of augmentations to apply. Choices: rotation, color, blur.

--aug_multiplier: Number of augmentations per image (only used in save mode).

--rot_degrees: Maximum degrees for rotation augmentation.

--blur_kernel: Kernel size for Gaussian blur.

--save_aug_path: Path to save the augmented images. If provided, the script runs in "save mode" instead of loading mode.

--save_aug_dataset_name: Name of the output dataset folder.

--split: Which split to process/save (train or test).

--include_anomalies: Flag to include defect folders when generating drifted data for the test split.

## Verify Data Loading
Run this to initialize the dataloaders and check sample counts without saving any files. This confirms the dataset structure is correct.

```bash
uv run python src/anomaly_detection/data.py \
  --data_root ./data/ \
  --class_name carpet \
  --img_size 224 \
  --batch_size 8

```

---

## Folder structure

Dataset folder structure:

```text
data/
└── carpet/
    ├── train/
    │   └── good/
    │       ├── 000.png
    │       └── ...
    ├── test/
    │   ├── good/
    │   ├── cut/
    │   ├── hole/
    │   └── ...
    └── ground_truth/
        ├── cut/
        │   ├── 000_mask.png
        │   └── ...
        ├── hole/
        └── ...

```

---

## Generate a "Drifted" Dataset (data augmentation)
Run this to apply augmentations (color, blur and rotation) to the test set and save the images to disk. This is useful for testing model robustness against domain shifts.

```bash
uv run ./src/anomaly_detection/data.py \
  --data_root ./data/ \
  --class_name carpet \
  --save_aug_path ./data/augmented \
  --save_aug_dataset_name carpet_augmented \
  --split test \
  --augment \
  --aug_types rotation color blur \
  --aug_multiplier 1 \
  --rot_degrees 20 \
  --color_brightness 0.2 \
  --color_contrast 0.2 \
  --blur_kernel 3 \
  --include_anomalies
```
