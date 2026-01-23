

# Inference pipeline

This script runs the anomaly detection pipeline on new images using a pre-trained DINOv3 model and a pre-computed memory bank. It generates visual outputs (heatmaps and overlays) and quantitative data (anomaly scores) without retraining.

- Core Function: Loads a frozen DINOv3 model and a memory bank to compute anomaly scores for each image in the dataset.
- Visualization: Produces pixel-level anomaly heatmaps and overlays them onto the original images to localize defects.
- Scoring: Calculates a scalar anomaly score for each image (max value of the anomaly map) and can save these scores to a JSONL file for later analysis.

## What it does:

1. Loads a test image and the pre-computed memory bank (.pt file).
2. Loads the frozen DINOv3 feature extractor.
3. Computes anomaly maps by comparing image patches to the memory bank (k-NN).
4. Generates and saves visual outputs (heatmaps and overlays).
5. Calculates per-image anomaly scores and logs them to a JSONL file.

## Arguments
--data_root: Path to the root directory of the dataset.

--class_name: The object category to analyze.

--weights_path: Path to the pretrained DINOv3 model weights.

--memory_bank_path: Path to the saved memory bank (.pt file).

--output_dir: Directory where results will be saved; defaults to ./results/figures.

--output_name: (Optional) Custom name for the output folder; defaults to the class name.

--split: The dataset split to run inference on (train or test); defaults to test.

--save_heatmaps: Flag to enable saving of anomaly heatmaps (enabled by default).

--save_overlays: Flag to enable saving of heatmaps overlaid on original images (enabled by default).

--heatmaps_only: Convenience flag to disable overlays and save only the heatmaps.

--scores_jsonl: (Optional) Path to save per-image anomaly scores and labels in JSONL format.

--img_size: Target image size for resizing (default: 224).

--batch_size: Batch size for data loading (default: 8).

--k: Number of nearest neighbors to use for anomaly scoring (default: 10).

## Standard Inference (Visuals + Scores)

This is the most common use case: generating heatmaps, overlays, and a score log for the test set.

```bash
uv run src/anomaly_detection/inference.py \
  --data_root ./data/ \
  --class_name carpet \
  --weights_path ./models/dinov3_vitb16_pretrain.pth \
  --memory_bank_path ./models/memory_bank.pt \
  --output_dir ./results/figures \
  --scores_jsonl ./logs/inference_scores.jsonl \
  --save_heatmaps \
  --save_overlays \
  --k 10
```

## Heatmaps Only (Faster Visualization)
Use this when you only need to inspect the raw anomaly signal without the context of the original image (saves disk space and time).
```bash
uv run src/anomaly_detection/inference.py \
  --data_root ./data/ \
  --class_name carpet \
  --weights_path ./models/dinov3_vitb16_pretrain.pth \
  --memory_bank_path ./models/memory_bank.pt \
  --heatmaps_only
```

---
