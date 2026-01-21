#!/usr/bin/env bash
set -euo pipefail

# End-to-end drift experiment using the CLI scripts (no API).
# It will:
# 1) create a drifted test dataset
# 2) run inference.py over the drifted test set with the original memory bank (save heatmaps + scores)
# 3) generate a histogram from the saved scores
# 4) create a drifted train dataset
# 5) rebuild the memory bank on drifted train data
# 6) run inference.py again on the same drifted test set with the updated memory bank
# 7) generate a second histogram and create a side-by-side comparison image

CLASS_NAME=${CLASS_NAME:-carpet}
DATA_ROOT=${DATA_ROOT:-./data}
IMG_SIZE=${IMG_SIZE:-224}
AUG_MULTIPLIER=${AUG_MULTIPLIER:-1}

# Drift parameters (contrast-only by default)
COLOR_CONTRAST=${COLOR_CONTRAST:-0.3}
COLOR_BRIGHTNESS=${COLOR_BRIGHTNESS:-0.0}
COLOR_SATURATION=${COLOR_SATURATION:-0.0}

# Output dataset names
DRIFT_TEST_NAME=${DRIFT_TEST_NAME:-${CLASS_NAME}_drifted_test}
DRIFT_TRAIN_NAME=${DRIFT_TRAIN_NAME:-${CLASS_NAME}_drifted_train}

# Model paths
WEIGHTS_PATH=${WEIGHTS_PATH:-./models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}
MEMORY_BANK_PATH=${MEMORY_BANK_PATH:-./models/memory_bank.pt}

# Report folders
OUT_ROOT=${OUT_ROOT:-./results/data_drift}
BEFORE_OUT=${OUT_ROOT}/before
AFTER_OUT=${OUT_ROOT}/after

API_HOST=127.0.0.1
API_PORT=8000

hist_from_scores() {
  local scores_jsonl=$1
  local out_dir=$2
  mkdir -p "${out_dir}"

  uv run --no-sync -- python src/anomaly_detection/evaluate.py \
    --class_name "${CLASS_NAME}" \
    --output_dir "${out_dir}" \
    --scores_jsonl "${scores_jsonl}" \
    --hist_only
}

echo "[1/7] Generating drifted TEST dataset: ${DRIFT_TEST_NAME}"
uv run --no-sync --script src/anomaly_detection/data.py \
  --data_root "${DATA_ROOT}" \
  --class_name "${CLASS_NAME}" \
  --split test \
  --include_anomalies \
  --augment \
  --aug_types color \
  --color_contrast "${COLOR_CONTRAST}" \
  --color_brightness "${COLOR_BRIGHTNESS}" \
  --color_saturation "${COLOR_SATURATION}" \
  --aug_multiplier "${AUG_MULTIPLIER}" \
  --save_aug_path "${DATA_ROOT}" \
  --save_aug_dataset_name "${DRIFT_TEST_NAME}"

echo "[2/7] Inference on drifted test set with ORIGINAL memory bank"
uv run --no-sync -- python src/anomaly_detection/inference.py \
  --data_root "${DATA_ROOT}" \
  --class_name "${DRIFT_TEST_NAME}" \
  --split test \
  --weights_path "${WEIGHTS_PATH}" \
  --memory_bank_path "${MEMORY_BANK_PATH}" \
  --output_dir "${BEFORE_OUT}" \
  --output_name "inference" \
  --heatmaps_only \
  --scores_jsonl "${BEFORE_OUT}/inference/scores.jsonl" \
  --img_size "${IMG_SIZE}"

echo "[3/7] Histogram (before)"
hist_from_scores "${BEFORE_OUT}/inference/scores.jsonl" "${BEFORE_OUT}/eval"

echo "[4/7] Generating drifted TRAIN dataset: ${DRIFT_TRAIN_NAME}"
uv run --no-sync --script src/anomaly_detection/data.py \
  --data_root "${DATA_ROOT}" \
  --class_name "${CLASS_NAME}" \
  --split train \
  --augment \
  --aug_types color \
  --color_contrast "${COLOR_CONTRAST}" \
  --color_brightness "${COLOR_BRIGHTNESS}" \
  --color_saturation "${COLOR_SATURATION}" \
  --aug_multiplier "${AUG_MULTIPLIER}" \
  --save_aug_path "${DATA_ROOT}" \
  --save_aug_dataset_name "${DRIFT_TRAIN_NAME}"

echo "[5/7] Rebuilding memory bank on drifted TRAIN data (overwrite ${MEMORY_BANK_PATH})"
uv run --no-sync -- python src/anomaly_detection/train.py \
  --data_root "${DATA_ROOT}" \
  --class_name "${DRIFT_TRAIN_NAME}" \
  --weights_path "${WEIGHTS_PATH}" \
  --save_path "${MEMORY_BANK_PATH}" \
  --img_size "${IMG_SIZE}"

echo "[6/7] Inference on drifted test set with UPDATED memory bank"
uv run --no-sync -- python src/anomaly_detection/inference.py \
  --data_root "${DATA_ROOT}" \
  --class_name "${DRIFT_TEST_NAME}" \
  --split test \
  --weights_path "${WEIGHTS_PATH}" \
  --memory_bank_path "${MEMORY_BANK_PATH}" \
  --output_dir "${AFTER_OUT}" \
  --output_name "inference" \
  --heatmaps_only \
  --scores_jsonl "${AFTER_OUT}/inference/scores.jsonl" \
  --img_size "${IMG_SIZE}"

echo "[7/7] Histogram (after) + comparison image"
hist_from_scores "${AFTER_OUT}/inference/scores.jsonl" "${AFTER_OUT}/eval"

uv run --no-sync -- python src/anomaly_detection/demo.py \
  --hist1 "${BEFORE_OUT}/eval/${CLASS_NAME}/roc/histogram.png" \
  --hist2 "${AFTER_OUT}/eval/${CLASS_NAME}/roc/histogram.png" \
  --out "${OUT_ROOT}/hist_comparison.png"

echo "Done. Outputs in: ${OUT_ROOT}"
