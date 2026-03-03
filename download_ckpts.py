from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download timm-compatible DINOv3 checkpoint.")
    parser.add_argument(
        "--backbone_name",
        type=str,
        default="vit_large_patch16_dinov3",
        help="timm model name, e.g. vit_small_patch16_dinov3 / vit_base_patch16_dinov3 / vit_large_patch16_dinov3",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="download_ckpts",
        help="Root folder for downloaded checkpoints.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="model_timm.pth",
        help="Checkpoint file name.",
    )
    return parser.parse_args()


def enable_online_download() -> None:
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(key) == "1":
            os.environ.pop(key, None)
    os.environ["TIMM_USE_HF_HUB"] = "1"


def main() -> None:
    args = parse_args()
    enable_online_download()

    import timm

    out_dir = Path(args.output_root) / args.backbone_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / args.filename
    meta_path = out_dir / "download_meta.json"

    print(f"Downloading timm weights: {args.backbone_name}")
    model = timm.create_model(
        args.backbone_name,
        pretrained=True,
        num_classes=0,
        global_pool="",
    )

    torch.save(model.state_dict(), ckpt_path)
    meta = {
        "backbone_name": args.backbone_name,
        "checkpoint_path": str(ckpt_path),
        "format": "timm_state_dict",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved: {ckpt_path}")
    print(f"Saved: {meta_path}")


if __name__ == "__main__":
    main()
