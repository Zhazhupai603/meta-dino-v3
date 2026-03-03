# CUBS(v1+v2) -> DINOv3(timm) Pipeline

This pipeline is timm-only and covers:
- CUBS v1 + CUBS v2 boundary `.txt` -> segmentation mask `.png`
- merged dataset split generation
- DINOv3 segmentation training with timm backbone

## Project layout

```text
dino/
  data/
    prepare_data.py
    dataset.py
    raw_downloads/
    extracted/
    processed/
  model/
    dinov3_seg.py
  download_ckpts/        # downloaded model checkpoints
  train.py               # training entry
  prepare_data.sh
  run.sh
  run_multi.sh
```

## 1) Environment

Use your specified environment Python:

```bash
/inspire/hdd/project/exploration-topic/public/lzjjin/anaconda3/envs/ssl/bin/python -V
```

Install dependencies (you said you will install timm yourself):

```bash
/inspire/hdd/project/exploration-topic/public/lzjjin/anaconda3/envs/ssl/bin/python -m pip install -r requirements.txt
```

## 2) Prepare merged CUBS dataset

```bash
cd /inspire/qb-ilm/project/exploration-topic/jinluozhijie-CZXS25210075/dino
./prepare_data.sh
```

Output folder:

- `data/processed/unified_dataset/images`
- `data/processed/unified_dataset/masks`
- `data/processed/unified_dataset/splits/{train,val,test}.csv`
- `data/processed/unified_dataset/dataset_summary.json`

What the script does:
- reads `LI/MA` boundary point txt files from:
  - CUBS v1: `SEGMENTATIONS/Manual-A1`
  - CUBS v2: `LIMA-Profiles/Manual-A1`
- interpolates two boundaries and fills region between them as IMT mask
- converts source TIFF images to grayscale PNG

Optional quick debug (small sample):

```bash
./prepare_data.sh --max_samples_v1 50 --max_samples_v2 50
```

## 3) Train DINOv3 with timm

First download timm-format ckpt:

```bash
/inspire/hdd/project/exploration-topic/public/lzjjin/anaconda3/envs/ssl/bin/python download_ckpts.py \
  --backbone_name vit_base_patch16_dinov3
```

Default command:

```bash
./run.sh
```

Multi-GPU command (accelerate):

```bash
./run_multi.sh --mixed_precision bf16
```

Key defaults:
- data root: `data/processed/unified_dataset`
- backbone: `vit_base_patch16_dinov3`
- checkpoint: `download_ckpts/vit_base_patch16_dinov3/model_timm.pth`
- outputs: `outputs/cubs_dinov3_timm`
- interval checkpoints: every `10` epochs, keep latest `3` (`best.pt` and `last.pt` always kept)
- val visualization: each val run save `20` random predictions to `outputs/.../generate/{images,pred_masks,gt_masks}` (always latest only)

Useful overrides:

```bash
./run.sh \
  --backbone_name vit_large_patch16_dinov3 \
  --checkpoint_path download_ckpts/vit_large_patch16_dinov3/model_timm.pth \
  --image_size 512 \
  --batch_size 4 \
  --epochs 80
```

If you only want to train the segmentation head first:

```bash
./run.sh --freeze_backbone
```

Gradient accumulation example:

```bash
./run_multi.sh \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --mixed_precision fp16
```

Augmentation and checkpoint schedule example:

```bash
./run_multi.sh \
  --save_interval 5 \
  --max_interval_ckpts 6 \
  --hflip_prob 0.5 \
  --vflip_prob 0.1 \
  --crop_prob 0.7 \
  --crop_scale_min 0.7 \
  --crop_scale_max 1.0 \
  --rotate_prob 0.3 \
  --rotate_deg 10 \
  --brightness_prob 0.3 \
  --brightness_min 0.85 \
  --brightness_max 1.15 \
  --contrast_prob 0.3 \
  --contrast_min 0.85 \
  --contrast_max 1.15 \
  --blur_prob 0.2 \
  --blur_kernel_size 3 \
  --noise_prob 0.2 \
  --noise_std 0.03
```

## 4) Outputs

Training artifacts:
- `outputs/cubs_dinov3_timm/checkpoints/best.pt`
- `outputs/cubs_dinov3_timm/checkpoints/last.pt`
- `outputs/cubs_dinov3_timm/history.json`

## 5) Predict

Single image (writes `<input_name>_mask.png` next to input):

```bash
/inspire/hdd/project/exploration-topic/public/lzjjin/anaconda3/envs/ssl/bin/python predict.py \
  --input /path/to/image.png \
  --checkpoint outputs/dinov3_base_seg/checkpoints/best.pt \
  --backbone_name vit_base_patch16_dinov3
```

Directory mode (recursive, writes `*_mask.png` for each image):

```bash
/inspire/hdd/project/exploration-topic/public/lzjjin/anaconda3/envs/ssl/bin/python predict.py \
  --input /path/to/image_root_dir \
  --checkpoint outputs/dinov3_base_seg/checkpoints/best.pt \
  --backbone_name vit_base_patch16_dinov3
```

## 6) Notes

- The script is strictly timm-only as requested.
- Make sure the chosen timm model name exists in your timm version.
- If model name mismatch happens, inspect available names with:

```bash
/inspire/hdd/project/exploration-topic/public/lzjjin/anaconda3/envs/ssl/bin/python -c "import timm; print([m for m in timm.list_models('*dinov3*')])"
```
