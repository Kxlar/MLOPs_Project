

# Evaluation pipeline (ROC/AUC + histogram)

##Evaluation computes:
- Image-level ROC AUC
- Histogram of anomaly scores
- (Optional) pixel-level ROC AUC using ground-truth masks

Script: `src/anomaly_detection/evaluate.py`

There are **two evaluation modes**:

1) **From scores file (`--scores_jsonl`)**

   - Uses precomputed anomaly scores
   - Always produces histogram
   - Computes AUC only if both classes exist in the file

2) **Full evaluation (requires dataset + weights + memory bank)**

   - Recomputes anomaly maps
   - Image-level AUC + histogram
   - Pixel-level AUC if masks exist

---

### Logging

Evaluation logs key metrics such as image-level and pixel-level ROC AUC.
Each run produces a timestamped log file in `logs/` for traceability.


## Mode A: Evaluate from a scores JSONL file

Use this if you ran inference with `--scores_jsonl`.

```bash
uv run python src/anomaly_detection/evaluate.py \
  --class_name carpet \
  --scores_jsonl ./results/scores/carpet_scores.jsonl \
  --output_dir ./results
```


## Mode B: Full evaluation

Recompute maps

```bash
uv run python src/anomaly_detection/evaluate.py \
  --data_root ./data/raw \
  --class_name carpet \
  --weights_path ./models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --memory_bank_path ./models/memory_bank.pt \
  --output_dir ./results \
  --k 10
```

## Image-level evaluation

Skip pixel evaluation

```bash
uv run python src/anomaly_detection/evaluate.py \
  --data_root ./data/raw \
  --class_name carpet \
  --weights_path ./models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --memory_bank_path ./models/memory_bank.pt \
  --output_dir ./results \
  --hist_only
```


---