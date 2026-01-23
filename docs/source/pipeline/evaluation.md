

# Evaluation pipeline 

The evaluation script quantitatively assesses the performance of the anomaly detection model on a labeled dataset. It compares predicted anomaly scores against ground-truth labels to compute evaluation metrics, enabling objective comparison of models, hyperparameters, and configurations in a reproducible way.

We support two evaluation modes to balance speed and completeness: scores-only evaluation is fast for iteration, while full evaluation recomputes anomaly maps to enable pixel-level metrics and ensure reproducibility when configurations change.


There are **two evaluation modes**:

---

### Logging

Evaluation logs key metrics such as image-level and pixel-level ROC AUC.
Each run produces a timestamped log file in `logs/` for traceability.

---


## Mode A: Quick evaluation (scores-only)

Evaluate from a scores JSONL file. Use this if you ran inference with `--scores_jsonl`.

```bash
uv run python src/anomaly_detection/evaluate.py \
  --class_name carpet \
  --scores_jsonl ./results/scores/carpet_scores.jsonl \
  --output_dir ./results
```
   - Uses precomputed anomaly scores
   - Always produces histogram
   - Computes AUC only if both classes exist in the file


## Mode B: Full evaluation (Recompute maps)

Use for final reporting + pixel AUC. 

```bash
uv run python src/anomaly_detection/evaluate.py \
  --data_root ./data/raw \
  --class_name carpet \
  --weights_path ./models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --memory_bank_path ./models/memory_bank.pt \
  --output_dir ./results \
  --k 10
```

   - Recomputes anomaly maps
   - Image-level AUC + histogram
   - Pixel-level AUC if masks exist

---
