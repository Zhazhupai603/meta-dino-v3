#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/inspire/hdd/project/exploration-topic/public/lzjjin/anaconda3/envs/ssl/bin/python"
PROJECT_ROOT="/inspire/qb-ilm/project/exploration-topic/jinluozhijie-CZXS25210075/dino"

# =========================
# Data prepare parameters
# =========================
# Options: cubs_v1,cubs_v2,cca (comma separated, can multi-select)
DATASETS="cubs_v1,cubs_v2"

# Paths
CUBS_V1_ROOT="data/extracted/DATASET for Carotid Ultrasound Boundary Study (CUBS) an open multi-center analysis of computerized intima-media thickness measurement systems and their clinical impact/DATASET for Carotid Ultrasound Boundary Study (CUBS) an open multi-center analysis of computerized intima-media thickness measurement systems and their clinical impact"
CUBS_V2_ROOT="data/extracted/m7ndn58sv6-1/m7ndn58sv6-1/DATASET_CUBS_tech/DATASET_CUBS_tech"
CCA_ROOT="data/extracted/Common Carotid Artery Ultrasound Images/Common Carotid Artery Ultrasound Images"
OUT_ROOT="data/processed/unified_dataset"

# Label source for CUBS
V1_MANUAL_SOURCE="Manual-A1"
V2_MANUAL_SOURCE="Manual-A1"

# Split
SEED=42
TRAIN_RATIO=0.8
VAL_RATIO=0.2

# Debug sampling (set to -1 means full dataset)
MAX_SAMPLES_V1=-1
MAX_SAMPLES_V2=-1
MAX_SAMPLES_CCA=-1

cd "${PROJECT_ROOT}"

CMD=(
  "${PYTHON_BIN}" data/prepare_data.py
  --datasets "${DATASETS}"
  --cubs_v1_root "${CUBS_V1_ROOT}"
  --cubs_v2_root "${CUBS_V2_ROOT}"
  --cca_root "${CCA_ROOT}"
  --out_root "${OUT_ROOT}"
  --v1_manual_source "${V1_MANUAL_SOURCE}"
  --v2_manual_source "${V2_MANUAL_SOURCE}"
  --seed "${SEED}"
  --train_ratio "${TRAIN_RATIO}"
  --val_ratio "${VAL_RATIO}"
)

if [[ "${MAX_SAMPLES_V1}" -ge 0 ]]; then
  CMD+=(--max_samples_v1 "${MAX_SAMPLES_V1}")
fi
if [[ "${MAX_SAMPLES_V2}" -ge 0 ]]; then
  CMD+=(--max_samples_v2 "${MAX_SAMPLES_V2}")
fi
if [[ "${MAX_SAMPLES_CCA}" -ge 0 ]]; then
  CMD+=(--max_samples_cca "${MAX_SAMPLES_CCA}")
fi

"${CMD[@]}" "$@"
