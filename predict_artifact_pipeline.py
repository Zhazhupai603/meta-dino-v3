from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF

from model.dinov3_artifact import DINOv3BoxRegModel, DINOv3PatchClsModel
from model.dinov3_seg import enforce_offline_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict artifact pipeline: localize then classify.")
    parser.add_argument("--input", type=Path, default=Path("input.jpg"))
    parser.add_argument("--det_checkpoint", type=Path, default=Path("outputs/cca_artifact_localizer/checkpoints/best.pt"))
    parser.add_argument("--cls_checkpoint", type=Path, default=Path("outputs/cca_artifact_patch_cls/checkpoints/best.pt"))
    parser.add_argument("--backbone_name", type=str, default="vit_base_patch16_dinov3")
    parser.add_argument("--det_image_size", type=int, default=512)
    parser.add_argument("--cls_image_size", type=int, default=128)
    parser.add_argument(
        "--artifact_gray_value",
        type=int,
        default=127,
        help="Gray value used in output mask when the localized region is classified as artifact.",
    )
    return parser.parse_args()


def collect_images(input_path: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if input_path.is_file():
        return [input_path]
    return [p for p in sorted(input_path.rglob("*")) if p.suffix.lower() in exts and p.is_file()]


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model.", "_orig_mod."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def normalize_image(image: Image.Image, size: int) -> torch.Tensor:
    image = TF.resize(image, [size, size], interpolation=TF.InterpolationMode.BILINEAR)
    tensor = TF.to_tensor(image).repeat(3, 1, 1)
    return TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0)


def cxcywh_to_xyxy(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    cx, cy, bw, bh = box.tolist()
    cx *= width
    cy *= height
    bw *= width
    bh *= height
    x1 = max(int(round(cx - bw / 2.0)), 0)
    y1 = max(int(round(cy - bh / 2.0)), 0)
    x2 = min(int(round(cx + bw / 2.0)), width - 1)
    y2 = min(int(round(cy + bh / 2.0)), height - 1)
    return x1, y1, x2, y2


def load_detector(args: argparse.Namespace, device: torch.device) -> DINOv3BoxRegModel:
    try:
        raw = torch.load(args.det_checkpoint, map_location="cpu")
    except pickle.UnpicklingError:
        raw = torch.load(args.det_checkpoint, map_location="cpu", weights_only=False)
    state_dict = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    model = DINOv3BoxRegModel(args.backbone_name, checkpoint_path=None, freeze_backbone=False)
    model.load_state_dict(clean_state_dict(state_dict), strict=False)
    model.to(device)
    model.eval()
    return model


def load_classifier(args: argparse.Namespace, device: torch.device) -> DINOv3PatchClsModel:
    try:
        raw = torch.load(args.cls_checkpoint, map_location="cpu")
    except pickle.UnpicklingError:
        raw = torch.load(args.cls_checkpoint, map_location="cpu", weights_only=False)
    state_dict = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    model = DINOv3PatchClsModel(args.backbone_name, checkpoint_path=None, num_classes=2, freeze_backbone=False)
    model.load_state_dict(clean_state_dict(state_dict), strict=False)
    model.to(device)
    model.eval()
    return model


def output_mask_path(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}_artifact_mask.png")


def output_json_path(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}_artifact.json")


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    enforce_offline_mode()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = load_detector(args, device)
    classifier = load_classifier(args, device)

    images = collect_images(args.input)
    class_mapping = {0: "artifact_fake", 1: "real_artery"}

    for idx, image_path in enumerate(images, start=1):
        image = Image.open(image_path).convert("L")
        orig_w, orig_h = image.size

        det_tensor = normalize_image(image, args.det_image_size).to(device)
        pred_box = detector(det_tensor)[0].detach().cpu().numpy()
        x1, y1, x2, y2 = cxcywh_to_xyxy(pred_box, orig_w, orig_h)

        crop = image.crop((x1, y1, x2 + 1, y2 + 1))
        cls_tensor = normalize_image(crop, args.cls_image_size).to(device)
        logits = classifier(cls_tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        class_id = int(np.argmax(probs))
        class_name = class_mapping[class_id]

        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        if class_id == 0:
            mask[y1 : y2 + 1, x1 : x2 + 1] = np.uint8(args.artifact_gray_value)
        Image.fromarray(mask, mode="L").save(output_mask_path(image_path))

        overlay = image.convert("RGB")
        drawer = ImageDraw.Draw(overlay)
        color = (160, 160, 160) if class_id == 0 else (0, 0, 0)
        drawer.rectangle([x1, y1, x2, y2], outline=color, width=2)
        drawer.text((x1, max(0, y1 - 12)), f"{class_name}:{float(probs[class_id]):.3f}", fill=color)
        overlay.save(image_path.with_name(f"{image_path.stem}_artifact_overlay.png"))

        payload = {
            "image": str(image_path),
            "bbox_xyxy": [x1, y1, x2, y2],
            "predicted_class": class_name,
            "scores": {
                "artifact_fake": float(probs[0]),
                "real_artery": float(probs[1]),
            },
            "output_mask_rule": {
                "artifact_fake": int(args.artifact_gray_value),
                "real_artery": 0,
                "background": 0,
            },
        }
        with output_json_path(image_path).open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        print(f"[{idx}/{len(images)}] {image_path} -> {output_mask_path(image_path)}")


if __name__ == "__main__":
    main()
