from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from model.dinov3_seg import DINOv3SegModel, enforce_offline_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict segmentation mask(s) with trained DINOv3 model.")
    parser.add_argument("--input", type=Path, default='input.jpg', help="Input image file or directory.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/dinov3_base_seg/checkpoints/best.pt"),
        help="Path to trained checkpoint (.pt).",
    )
    parser.add_argument("--backbone_name", type=str, default="vit_base_patch16_dinov3")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=0)
    return parser.parse_args()


def collect_images(input_path: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if input_path.is_file():
        if input_path.suffix.lower() not in exts:
            raise ValueError(f"Unsupported input file extension: {input_path.suffix}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input not found: {input_path}")

    files = [
        path
        for path in sorted(input_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in exts and path.name != "mask.png" and "_mask" not in path.stem
    ]
    if not files:
        raise FileNotFoundError(f"No image files found under: {input_path}")
    return files


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model.", "_orig_mod."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def infer_num_classes(args: argparse.Namespace, checkpoint_obj: Dict[str, object]) -> int:
    if args.num_classes >= 2:
        return args.num_classes
    num_classes = checkpoint_obj.get("num_classes")
    if isinstance(num_classes, int) and num_classes >= 2:
        return num_classes
    inner_args = checkpoint_obj.get("args")
    if isinstance(inner_args, dict):
        maybe_num_classes = inner_args.get("num_classes")
        if isinstance(maybe_num_classes, int) and maybe_num_classes >= 2:
            return maybe_num_classes
    return 2


def mask_to_uint8(mask: np.ndarray, num_classes: int) -> np.ndarray:
    if num_classes == 3:
        out = np.zeros_like(mask, dtype=np.uint8)
        out[mask == 1] = 127
        out[mask == 2] = 255
        return out
    if num_classes <= 1:
        return mask.astype(np.uint8)
    scale = 255.0 / float(max(num_classes - 1, 1))
    return np.clip(mask.astype(np.float32) * scale, 0, 255).astype(np.uint8)


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[DINOv3SegModel, int]:
    enforce_offline_mode()
    try:
        raw = torch.load(args.checkpoint, map_location="cpu")
    except pickle.UnpicklingError:
        raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw and isinstance(raw["model_state_dict"], dict):
        state_dict = raw["model_state_dict"]
    elif isinstance(raw, dict):
        state_dict = raw
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(raw)}")

    num_classes = infer_num_classes(args, raw if isinstance(raw, dict) else {})
    model = DINOv3SegModel(
        backbone_name=args.backbone_name,
        checkpoint_path=None,
        num_classes=num_classes,
        freeze_backbone=False,
    )

    msg = model.load_state_dict(clean_state_dict(state_dict), strict=False)
    print("Checkpoint load message:", msg)
    model.to(device)
    model.eval()
    return model, num_classes


def preprocess(image_path: Path, image_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    image = Image.open(image_path).convert("L")
    orig_size = image.size
    image = TF.resize(
        image,
        [image_size, image_size],
        interpolation=TF.InterpolationMode.BILINEAR,
    )
    image_tensor = TF.to_tensor(image)
    image_tensor = image_tensor.repeat(3, 1, 1)
    image_tensor = TF.normalize(
        image_tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return image_tensor.unsqueeze(0), orig_size


def output_path_for(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}_mask.png")


@torch.inference_mode()
def predict_one(
    model: DINOv3SegModel,
    image_path: Path,
    out_path: Path,
    image_size: int,
    device: torch.device,
    num_classes: int,
) -> None:
    image_tensor, orig_size = preprocess(image_path, image_size)
    image_tensor = image_tensor.to(device, non_blocking=True)
    logits = model(image_tensor)
    pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    pred = mask_to_uint8(pred, num_classes)
    pred_img = Image.fromarray(pred, mode="L").resize(orig_size, resample=Image.NEAREST)
    pred_img.save(out_path)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script, but no GPU is available.")

    device = torch.device("cuda:0")
    images = collect_images(args.input)
    model, num_classes = load_model(args, device=device)

    for idx, image_path in enumerate(images, start=1):
        out_path = output_path_for(image_path)
        predict_one(
            model=model,
            image_path=image_path,
            out_path=out_path,
            image_size=args.image_size,
            device=device,
            num_classes=num_classes,
        )
        print(f"[{idx}/{len(images)}] {image_path} -> {out_path}")


if __name__ == "__main__":
    main()
