# Hydra

### Run (Hydra scripts)
```bash
uv sync
uv run python -m anomaly_detection.hydra.train_hydra
uv run python -m anomaly_detection.hydra.evaluate_hydra
uv run python -m anomaly_detection.hydra.inference_hydra
uv run python -m anomaly_detection.hydra.augment_hydra
```

### Override parameters (example)
```bash
uv run src/anomaly_detection/evaluate_hydra.py class_name=carpet k=5 output_dir=./results_k5
```
