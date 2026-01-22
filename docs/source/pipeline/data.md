
# Data pipeline

This project uses the **MVTec AD** dataset. The data module (`src/anomaly_detection/data.py`) is responsible for:

- Loading images from the dataset folders
- Creating PyTorch datasets and dataloaders
- Applying resizing, normalization, and optional augmentations
- Loading pixel-level ground-truth masks (for evaluation)
- (Optional) Saving an augmented (“drifted”) dataset to disk

---

## Expected folder structure

The code searches using this pattern:

- `data_root/<class_name>/<split>/*/*.*`

So your dataset should look like:

```text
data/raw/
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


## Quick check: test data loading

Use this to verify you have the dataset in the correct structure:

```bash
uv run python src/anomaly_detection/data.py \
  --data_root ./data/raw \
  --class_name carpet \
  --img_size 224 \
  --batch_size 8

```

## Save an augmented / drifted dataset (optional)

This mode writes a new dataset folder to disk. This is mainly used for drift demos.

```bash
uv run python src/anomaly_detection/data.py \
  --data_root ./data/raw \
  --class_name carpet \
  --split train \
  --augment \
  --aug_types rotation color \
  --aug_multiplier 2 \
  --save_aug_path ./data \
  --save_aug_dataset_name carpet_augmented


```

---