
# Anomaly Detection (MVTec AD)

## Overview
This project implements a **zero-shot anomaly detection pipeline** for industrial visual inspection using the **MVTec AD dataset**, with a focus on the **Carpet** class.

The goal is to detect anomalous regions in images **without using any defective samples during training**, making the approach suitable for real-world industrial settings where defects are rare or unknown in advance.

---

## Why Zero-Shot Anomaly Detection?
In manufacturing, collecting and labeling defective samples is often expensive, time-consuming, or impractical.  
Zero-shot anomaly detection addresses this challenge by learning only from **normal (defect-free) data** and identifying deviations at inference time.

This project explores whether vision foundation models can enable robust anomaly detection without pre-training nor task-specific fine-tuning.

---

## Method at a Glance
The pipeline is built around **DINOv3**, a self-supervised Vision Transformer:

1. Extract patch-level features from normal training images
2. Build a **memory bank** of normal visual patterns
3. Compare test image features against the memory bank using distance-based scoring
4. Produce:
   - Image-level anomaly scores
   - Pixel-level anomaly heatmaps

---

## Features
The project supports:

- 🧠 **Zero-shot feature extraction** using DINOv3
- 🏦 **Memory bank construction** from normal samples
- 🔥 **Pixel-level anomaly heatmaps**
- 📊 **Image-level & pixel-level evaluation (ROC AUC)**
- 🧪 **Data augmentation and drift simulation**
- 🐳 **Dockerized training and inference**
- 🚀 **FastAPI backend and frontend**
- ☁️ **Cloud deployment (GCP-ready)**
- 🔁 **Reproducible experiments with Hydra**

---

## Dataset
We use the **MVTec Anomaly Detection (MVTec AD)** dataset, a standard benchmark for industrial anomaly detection.

- Training set: normal images only
- Test set: normal + defective images with pixel-level ground truth masks
- Focus class: **Carpet**

---

## Getting Started
If you are new to the project, start here:

➡️ **[Getting Started](getting-started.md)**  
➡️ **[Training Pipeline](pipeline/train.md)**  
➡️ **[Inference & Heatmaps](pipeline/inference.md)**

---

## Project Context
This project was developed as part of the **02476 MLOps course at DTU**, with an emphasis on:

- Clean code structure
- Reproducibility
- Automation
- Deployment-ready ML systems