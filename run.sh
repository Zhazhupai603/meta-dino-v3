#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# =========================
# Train parameters (single GPU)
# =========================
DATA_ROOT="data/processed/unified_dataset"

# DINOv3 backbone + timm checkpoint (download via download_ckpts.py)
# Common choices:
#   vit_small_patch16_dinov3 + download_ckpts/vit_small_patch16_dinov3/model_timm.pth
#   vit_base_patch16_dinov3  + download_ckpts/vit_base_patch16_dinov3/model_timm.pth
#   vit_large_patch16_dinov3 + download_ckpts/vit_large_patch16_dinov3/model_timm.pth
BACKBONE_NAME="vit_base_patch16_dinov3"
CHECKPOINT_PATH="download_ckpts/vit_base_patch16_dinov3/model_timm.pth"

OUTPUT_DIR="outputs/cubs_dinov3_timm"

IMAGE_SIZE=512
BATCH_SIZE=8
NUM_WORKERS=4
EPOCHS=60
LR=1e-4
WEIGHT_DECAY=1e-4
CE_WEIGHT=1.0
DICE_WEIGHT=1.0
SEED=42
MIXED_PRECISION="no"   # no|fp16|bf16
GRAD_ACC=1
SAVE_INTERVAL=10
MAX_INTERVAL_CKPTS=3
VAL_GENERATE_COUNT=20
HFLIP_PROB=0.5
VFLIP_PROB=0.1
CROP_PROB=0.7
CROP_SCALE_MIN=0.7
CROP_SCALE_MAX=1.0
ROTATE_PROB=0.3
ROTATE_DEG=10.0
BRIGHTNESS_PROB=0.3
BRIGHTNESS_MIN=0.85
BRIGHTNESS_MAX=1.15
CONTRAST_PROB=0.3
CONTRAST_MIN=0.85
CONTRAST_MAX=1.15
BLUR_PROB=0.2
BLUR_KERNEL_SIZE=3
NOISE_PROB=0.2
NOISE_STD=0.03

# 1: freeze backbone, 0: train full model
FREEZE_BACKBONE=0

cd "${SCRIPT_DIR}"

# Force offline/local-only behavior for timm and HF hub.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TIMM_USE_HF_HUB=0

CMD=(
  python train.py
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
  --ce_weight "${CE_WEIGHT}"
  --dice_weight "${DICE_WEIGHT}"
  --seed "${SEED}"
  --mixed_precision "${MIXED_PRECISION}"
  --gradient_accumulation_steps "${GRAD_ACC}"
  --save_interval "${SAVE_INTERVAL}"
  --max_interval_ckpts "${MAX_INTERVAL_CKPTS}"
  --val_generate_count "${VAL_GENERATE_COUNT}"
  --hflip_prob "${HFLIP_PROB}"
  --vflip_prob "${VFLIP_PROB}"
  --crop_prob "${CROP_PROB}"
  --crop_scale_min "${CROP_SCALE_MIN}"
  --crop_scale_max "${CROP_SCALE_MAX}"
  --rotate_prob "${ROTATE_PROB}"
  --rotate_deg "${ROTATE_DEG}"
  --brightness_prob "${BRIGHTNESS_PROB}"
  --brightness_min "${BRIGHTNESS_MIN}"
  --brightness_max "${BRIGHTNESS_MAX}"
  --contrast_prob "${CONTRAST_PROB}"
  --contrast_min "${CONTRAST_MIN}"
  --contrast_max "${CONTRAST_MAX}"
  --blur_prob "${BLUR_PROB}"
  --blur_kernel_size "${BLUR_KERNEL_SIZE}"
  --noise_prob "${NOISE_PROB}"
  --noise_std "${NOISE_STD}"
)

if [[ "${FREEZE_BACKBONE}" -eq 1 ]]; then
  CMD+=(--freeze_backbone)
fi

"${CMD[@]}" "$@"
