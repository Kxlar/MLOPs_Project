

# Training pipeline (memory bank)

Training in this project means: build a memory bank of normal (good) features using DINOv3 features.

Script: `src/anomaly_detection/train.py`

## What it does:

1. Loads training images (`train/good/`)
2. Loads DINOv3 weights
3. Extracts patch features
4. Builds a memory bank tensor
5. Saves it as a `.pt` file

---

## Inputs

**Required**

- `--data_root`: dataset root (e.g. `./data/raw`)
- `--class_name`: MVTec class (e.g. `carpet`)
- `--weights_path`: path to DINOv3 weights file

**Optional**

- `--save_path`: where to write the memory bank (default: `./models/memory_bank.pt`)
- `--img_size`, `--batch_size`

---

## Training
To build the memory bank, run:

```bash
uv run src/anomaly_detection/train.py \
  --data_root ./data/raw \
  --class_name carpet \
  --weights_path ./models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --save_path ./models/memory_bank.pt
```
