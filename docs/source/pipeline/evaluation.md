

# Evaluation pipeline 

The evaluation script quantitatively assesses the performance of the anomaly detection model on a labeled dataset. It compares predicted anomaly scores against ground-truth labels to compute evaluation metrics, enabling objective comparison of models, hyperparameters, and configurations in a reproducible way. We used logs on evaluation with key metrics such as image-level and pixel-level ROC AUC. Each run produces a timestamped log file in `logs/` for traceability.


### Run

```bash
uv run src/anomaly_detection/evaluate.py \
  --data_root ./data/raw \
  --class_name carpet \
  --weights_path ./models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --memory_bank_path ./models/memory_bank.pt

```
### Output
   - Recomputes anomaly maps
   - Image-level AUC + histogram
   - Pixel-level AUC if masks exist

---
