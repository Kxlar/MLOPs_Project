

This project uses the **MVTec AD** dataset.

## 1. Downloading Dataset
Our dataset can be downloaded from the cloud

```bash
chmod +x setup.sh
```
```bash
./setup.sh
```

## 2. Data loading

The data module (`src/anomaly_detection/data.py`) is responsible for:

- Loading images from the dataset folders
- Creating PyTorch datasets and dataloaders
- Applying resizing, normalization, and optional augmentations
- Loading pixel-level ground-truth masks (for evaluation)
- (Optional) Saving an augmented (“drifted”) dataset to disk


To verify you have the dataset in the correct structure, load dataset

```bash
uv run python src/anomaly_detection/data.py \
  --data_root ./data/ \
  --class_name carpet \
  --img_size 224 \
  --batch_size 8

```

---

## 3. Folder structure

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

## 4. Data augmentation
Users can perform data augmentation on a chosen dataset:
- Modify the contrast
- Modify the brightness
- Modify the saturation
- Modify the blur
- Add rotations

```bash
uv run ./src/anomaly_detection/data.py \
  --data_root ./data/ \
  --class_name carpet \
  --augment \
  --aug_types rotation color blur \
  --aug_multiplier 1 \
  --rot_degrees 20 \
  --color_brightness 0.2 \
  --color_contrast 0.2 \
  --color_saturation 0.2 \
  --blur_kernel 3 \
  --save_aug_path ./data/augmented \
  --save_aug_dataset_name carpet_augmented \
  --split test \
  --include_anomalies
```
