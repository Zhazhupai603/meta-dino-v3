#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

INPUT_PATH="test_data/4.png"
DET_CHECKPOINT="outputs/cca_artifact_localizer/checkpoints/best.pt"
CLS_CHECKPOINT="outputs/cca_artifact_patch_cls/checkpoints/best.pt"
BACKBONE_NAME="vit_base_patch16_dinov3"
DET_IMAGE_SIZE=512
CLS_IMAGE_SIZE=128

cd "${SCRIPT_DIR}"

CMD=(
  python predict_artifact_pipeline.py
  --input "${INPUT_PATH}"
  --det_checkpoint "${DET_CHECKPOINT}"
  --cls_checkpoint "${CLS_CHECKPOINT}"
  --backbone_name "${BACKBONE_NAME}"
  --det_image_size "${DET_IMAGE_SIZE}"
  --cls_image_size "${CLS_IMAGE_SIZE}"
)

"${CMD[@]}" "$@"
