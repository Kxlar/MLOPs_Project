# Data Drift & Data Augmentation Demo

This page describes our data drift experiment and how we evaluate the
effect of rebuilding the memory bank after distribution shifts in the data.

The goal is to demonstrate how data augmentation–induced drift affects
model outputs, and whether retraining on drifted data improves robustness.

---

## Motivation: Why Data Drift?

In real-world deployments, data distributions often change over time due to:

- Lighting changes
- Color shifts
- Sensor degradation
- Environmental variation
- Domain transfer

This phenomenon is known as **data drift**.

Our anomaly detection model is trained on **normal images only** using a
DINOv3-based memory bank. We want to study what happens when:

1. The test data is drifted
2. The training data is also drifted
3. The memory bank is rebuilt on drifted data

---

## Experimental Idea

We compare two scenarios:

### Before (No Adaptation)

- Test data is artificially drifted
- Inference uses a memory bank built on original training data
- Expectation: higher anomaly scores / distribution shift

### After (With Adaptation)

- Training data is drifted using the same transformation
- Memory bank is rebuilt on drifted training data
- Test inference is repeated
- Expectation: anomaly score distribution shifts back toward normal

---

## High-Level Pipeline

```text
Generate drifted test data
        ↓
Inference on drifted test data
        ↓
Histogram ("Before")
        ↓
Generate drifted training data
        ↓
Rebuild memory bank
        ↓
Inference on drifted test data again
        ↓
Histogram ("After")
        ↓
Side-by-side comparison (demo plot)
```

The docker build command must be run once to create the Docker image before the data drift experiment can be executed with docker run.



## Build  a Docker image 
```bash
docker build -f data_drift_demo.dockerfile -t mlops-data-drift-demo .
```

Run the container:

```bash
docker run --rm `
  -v "$((Get-Location).Path)\data:/app/data" `
  -v "$((Get-Location).Path)\models:/app/models" `
  -v "$((Get-Location).Path)\results:/app/results" `
  -e COLOR_CONTRAST=0.6 `
  -e CLASS_NAME=carpet `
  -e DATA_ROOT=/app/data `
  -e OUT_ROOT=/app/results/data_drift `
  mlops-data-drift-demo
```

### Outputs
After completion, the following artifacts are produced:

- Drifted datasets (train & test)
- Heatmaps for both runs
- Histogram before adaptation
- Histogram after adaptation
- Comparison plot (demo.py)

