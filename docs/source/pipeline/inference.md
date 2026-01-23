

# Inference pipeline

The inference performs offline anomaly detection on unseen images using a frozen DINOv3 feature extractor and a precomputed memory bank. It compares patch-level image features against the memory bank via k-nearest-neighbor distances to generate anomaly heatmaps, optional overlay visualizations, and per-image anomaly scores for analysis, visualization, and downstream evaluation—without retraining the model.

## Run inference

Example (test split):

```bash
uv run python src/anomaly_detection/inference.py \
  --data_root ./data/raw \
  --class_name carpet \
  --weights_path ./models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --memory_bank_path ./models/memory_bank.pt \
  --output_dir ./results/figures \
  --img_size 224 \
  --batch_size 8 \
  --k 10
```



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

