# Hydra

### Run (Hydra scripts)
```bash
uv sync
uv run src/anomaly_detection/evaluate_hydra.py
uv run src/anomaly_detection/inference_hydra.py
uv run src/anomaly_detection/augment_hydra.py
```

### Override parameters (example)
```bash
uv run src/anomaly_detection/evaluate_hydra.py class_name=carpet k=5 output_dir=./results_k5
```
