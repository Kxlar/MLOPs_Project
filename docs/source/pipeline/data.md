

This project uses the **MVTec AD** dataset.

## 1. Downloading Dataset
Our dataset can be downloaded from the cloud

```bash
chmod +x setup.sh
```
```bash
./setup.sh
```

##  2. Data module
The data module (`src/anomaly_detection/data.py`) is responsible for:

- Loading images from the dataset folders
- Creating PyTorch datasets and dataloaders
- Applying resizing, normalization, and optional augmentations
- Loading pixel-level ground-truth masks (for evaluation)
- (Optional) Saving an augmented (“drifted”) dataset to disk

---

## 3. older structure

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


## 4. Data loading

Use this to verify you have the dataset in the correct structure:

```bash
uv run python src/anomaly_detection/data.py \
  --data_root ./data/raw \
  --class_name carpet \
  --img_size 224 \
  --batch_size 8

```
