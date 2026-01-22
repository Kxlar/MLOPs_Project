#!/bin/bash
set -e

BUCKET_NAME="mlops15-bucket"
MODEL_FILE="dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
DATA_DIR="data"

echo "Downloading data directory..."
gsutil -m cp -r gs://$BUCKET_NAME/$DATA_DIR .

echo "Downloading model file..."
mkdir -p models
gsutil cp gs://$BUCKET_NAME/$MODEL_FILE models/

echo "Download completed."
