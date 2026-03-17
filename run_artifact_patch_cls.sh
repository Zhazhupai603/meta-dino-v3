#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

DATA_ROOT="data/processed/cca_artifact_dataset/artifact_patch_cls"
BACKBONE_NAME="vit_base_patch16_dinov3"
CHECKPOINT_PATH="download_ckpts/vit_base_patch16_dinov3/model_timm.pth"
OUTPUT_DIR="outputs/cca_artifact_patch_cls"
IMAGE_SIZE=128
BATCH_SIZE=32
NUM_WORKERS=4
EPOCHS=40
LR=1e-4
WEIGHT_DECAY=1e-4
SEED=42
HFLIP_PROB=0.5
FREEZE_BACKBONE=0

cd "${SCRIPT_DIR}"

CMD=(
  python train_artifact_patch_cls.py
  --data_root "${DATA_ROOT}"
  --backbone_name "${BACKBONE_NAME}"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --image_size "${IMAGE_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --seed "${SEED}"
  --hflip_prob "${HFLIP_PROB}"
)

if [[ "${FREEZE_BACKBONE}" -eq 1 ]]; then
  CMD+=(--freeze_backbone)
fi

"${CMD[@]}" "$@"
