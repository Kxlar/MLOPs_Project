

# Training pipeline (memory bank)

In this Zero-Shot framework, "training" does not involve gradient updates. Instead, this script builds a Memory Bank of features from normal data.

- Feature Extraction: Runs the frozen DINOv3 model on the training set (only "good" images) to extract high-level feature representations.
- Memory Bank Construction: Aggregates these feature vectors into a single tensor, acting as the "reference" for normality.
- Persistence: Saves the resulting tensor (.pt) to disk, which is required by the inference and evaluation modules.

## What it does:

1. Loads training images (`train/good/`)
2. Loads DINOv3 weights
3. Extracts patch features
4. Builds a memory bank tensor
5. Saves it as a `.pt` file


## Arguments
--data_root: Path to the dataset root directory.

--class_name: The specific object category to model.

--weights_path: Path to the pretrained DINOv3 model weights.

--save_path: File path where the memory bank .pt file will be saved. Defaults to ./models/memory_bank.pt.

--img_size: Target image size for resizing (default: 224). Must match the size used during inference.

--batch_size: Number of images processed at once (default: 8).

--augment: (Optional) Enable data augmentation during memory bank creation. Typically False for standard banks, but can be used to build robust banks.

--aug_types: (Optional) List of augmentations to apply if --augment is set (e.g., rotation, color, blur).

## Standard Memory Bank Construction
This is the default usage: it processes the "good" training images and creates a standard reference bank.

```bash
uv run src/anomaly_detection/train.py \
  --data_root ./data/ \
  --class_name carpet \
  --weights_path ./models/dinov3_vitb16_pretrain.pth \
  --save_path ./models/checkpoints/memory_bank_carpet.pt \
  --img_size 224 \
  --batch_size 16

```

## Robust Memory Bank (with Augmentation)
If you expect the test environment to have specific variations (e.g., slight rotations or lighting changes) that should be considered "normal," you can inject them into the memory bank directly.

```bash
uv run src/anomaly_detection/train.py \
  --data_root ./data/ \
  --class_name carpet \
  --weights_path ./models/dinov3_vitb16_pretrain.pth \
  --save_path ./models/checkpoints/memory_bank_carpet_robust.pt \
  --augment \
  --aug_types rotation color \
  --rot_degrees 5 \
  --color_brightness 0.1

```

---
