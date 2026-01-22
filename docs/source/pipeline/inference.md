

# Inference pipeline

## Inference generates:
- anomaly heatmaps
- overlay images (optional)
- per-image anomaly scores (optional JSONL)

Script: `src/anomaly_detection/inference.py`

## How it works:

1. Loads test images
2. Loads DINOv3 + feature extractor
3. Loads memory bank (`.pt`)
4. Computes patch-level anomaly scores using kNN distances
5. Upsamples anomaly map to image size
6. Saves heatmaps / overlays + optional scores

---

## Inputs

**Required:**

- `--data_root`
- `--class_name`
- `--weights_path`
- `--memory_bank_path`

**Optional (common):**

- `--output_dir` (default `./results/figures`)
- `--split` (`test` by default)
- `--k` top-k neighbors (default 10)

**Saving options:**

- `--save_heatmaps`
- `--save_overlays`
- `--heatmaps_only`
- `--scores_jsonl <path>`

---

## Run inference

Example (test split):

```bash
uv run python src/anomaly_detection/inference.py \
  --data_root ./data/raw \
  --class_name carpet \
  --weights_path ./models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --memory_bank_path ./models/memory_bank.pt \
  --output_dir ./reports/figures \
  --img_size 224 \
  --batch_size 8 \
  --k 10
```