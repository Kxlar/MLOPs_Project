

# Evaluation pipeline

The evaluation script quantitatively assesses the performance of the anomaly detection model on a labeled dataset. It compares predicted anomaly scores against ground-truth labels to compute evaluation metrics, enabling objective comparison of models, hyperparameters, and configurations in a reproducible way. We used logs on evaluation with key metrics such as image-level and pixel-level ROC AUC. Each run produces a timestamped log file in `logs/` for traceability.

- Metric Calculation: Computes Image-Level ROC AUC and, if masks are available, Pixel-Level ROC AUC.
- Visualization: Generates histograms of anomaly scores (Good vs. Defective) and Pixel ROC curves.
- Dual Modes: Can run full inference using a model and memory bank, or quickly re-evaluate using pre-computed scores from a JSONL log.

### Arguments
--data_root: Path to the dataset root folder.

--class_name: The specific object category to evaluate.

--weights_path: Path to the pretrained DINOv3 model weights.

--memory_bank_path: Path to the computed memory bank (.pt file).

--output_dir: Directory to save results (plots and ROC curves); defaults to ./results.

--k: Number of nearest neighbors to use for anomaly scoring (default: 10).

--scores_jsonl: (Optional) Path to a JSONL file with pre-computed scores. If provided, skips inference and only generates histograms/metrics.

--hist_only: (Optional) Flag to skip pixel-level evaluation for faster execution.


### Full evaluation
This command runs the full pipeline: loading the model, computing anomaly maps for the test set, and calculating both image and pixel-level AUC.
```bash
uv run src/anomaly_detection/evaluate.py \
  --data_root ./data/ \
  --class_name carpet \
  --weights_path ./models/dinov3_vitb16_pretrain.pth \
  --memory_bank_path ./models/checkpoints/memory_bank_carpet.pt \
  --output_dir ./results \
  --img_size 224 \
  --k 10

```

### Fast Re-Evaluation (From Logs)
If you have already run inference and saved scores, use this to quickly regenerate histograms or check Image AUC without reloading the heavy model.
```bash
uv run src/anomaly_detection/evaluate.py \
  --class_name carpet \
  --scores_jsonl ./logs/inference_scores.jsonl \
  --output_dir ./results_reanalysis

```

---
