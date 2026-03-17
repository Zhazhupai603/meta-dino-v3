from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import List

import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode, RandomResizedCrop


def normalize_grayscale(image: Image.Image, image_size: int | None) -> torch.Tensor:
    if image_size is not None:
        image = TF.resize(image, [image_size, image_size], interpolation=InterpolationMode.BILINEAR)
    image_tensor = TF.to_tensor(image).repeat(3, 1, 1)
    return TF.normalize(
        image_tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )


@dataclass
class DetectionAugmentConfig:
    image_size: int = 512
    hflip_prob: float = 0.5
    vflip_prob: float = 0.1
    crop_prob: float = 0.7
    crop_scale_min: float = 0.7
    crop_context_scale: float = 1.15
    brightness_jitter: float = 0.15
    contrast_jitter: float = 0.2
    autocontrast_prob: float = 0.2
    blur_prob: float = 0.2
    blur_radius_min: float = 0.1
    blur_radius_max: float = 1.0
    noise_prob: float = 0.2
    noise_std: float = 0.03


def clamp_box_xyxy(box_xyxy: torch.Tensor, width: int, height: int) -> torch.Tensor:
    box_xyxy[0::2] = box_xyxy[0::2].clamp(0.0, float(width))
    box_xyxy[1::2] = box_xyxy[1::2].clamp(0.0, float(height))
    return box_xyxy


def cxcywh_to_xyxy_abs(bbox: torch.Tensor, width: int, height: int) -> torch.Tensor:
    cx = bbox[0] * width
    cy = bbox[1] * height
    bw = bbox[2] * width
    bh = bbox[3] * height
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    return clamp_box_xyxy(torch.tensor([x1, y1, x2, y2], dtype=torch.float32), width, height)


def xyxy_abs_to_cxcywh(box_xyxy: torch.Tensor, width: int, height: int) -> torch.Tensor:
    x1, y1, x2, y2 = box_xyxy.tolist()
    cx = ((x1 + x2) / 2.0) / max(width, 1)
    cy = ((y1 + y2) / 2.0) / max(height, 1)
    bw = (x2 - x1) / max(width, 1)
    bh = (y2 - y1) / max(height, 1)
    return torch.tensor([cx, cy, bw, bh], dtype=torch.float32).clamp(0.0, 1.0)


def maybe_apply_photometric_aug(
    image: Image.Image,
    brightness_jitter: float,
    contrast_jitter: float,
    autocontrast_prob: float,
    blur_prob: float,
    blur_radius_min: float,
    blur_radius_max: float,
    noise_prob: float,
    noise_std: float,
) -> Image.Image:
    if brightness_jitter > 0:
        factor = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * brightness_jitter
        image = ImageEnhance.Brightness(image).enhance(max(0.05, factor))
    if contrast_jitter > 0:
        factor = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * contrast_jitter
        image = ImageEnhance.Contrast(image).enhance(max(0.05, factor))
    if autocontrast_prob > 0 and torch.rand(1).item() < autocontrast_prob:
        image = ImageOps.autocontrast(image)
    if blur_prob > 0 and torch.rand(1).item() < blur_prob:
        radius = blur_radius_min + torch.rand(1).item() * (blur_radius_max - blur_radius_min)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    if noise_prob > 0 and torch.rand(1).item() < noise_prob and noise_std > 0:
        tensor = TF.to_tensor(image)
        tensor = (tensor + torch.randn_like(tensor) * noise_std).clamp(0.0, 1.0)
        image = TF.to_pil_image(tensor)
    return image


def random_bbox_preserving_crop(
    image: Image.Image,
    bbox: torch.Tensor,
    cfg: DetectionAugmentConfig,
) -> tuple[Image.Image, torch.Tensor]:
    width, height = image.size
    box_xyxy = cxcywh_to_xyxy_abs(bbox, width, height)
    x1, y1, x2, y2 = box_xyxy.tolist()
    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)

    min_crop_w = int(math.ceil(max(width * cfg.crop_scale_min, box_w * cfg.crop_context_scale)))
    min_crop_h = int(math.ceil(max(height * cfg.crop_scale_min, box_h * cfg.crop_context_scale)))
    min_crop_w = min(max(min_crop_w, int(math.ceil(box_w))), width)
    min_crop_h = min(max(min_crop_h, int(math.ceil(box_h))), height)
    if min_crop_w >= width and min_crop_h >= height:
        return image, bbox

    crop_w = int(torch.randint(min_crop_w, width + 1, (1,)).item())
    crop_h = int(torch.randint(min_crop_h, height + 1, (1,)).item())

    min_left = max(0, int(math.floor(x2 - crop_w)))
    max_left = min(int(math.floor(x1)), width - crop_w)
    min_top = max(0, int(math.floor(y2 - crop_h)))
    max_top = min(int(math.floor(y1)), height - crop_h)
    if min_left > max_left or min_top > max_top:
        return image, bbox

    left = int(torch.randint(min_left, max_left + 1, (1,)).item())
    top = int(torch.randint(min_top, max_top + 1, (1,)).item())
    image = TF.crop(image, top, left, crop_h, crop_w)
    new_box = clamp_box_xyxy(box_xyxy - torch.tensor([left, top, left, top], dtype=torch.float32), crop_w, crop_h)
    return image, xyxy_abs_to_cxcywh(new_box, crop_w, crop_h)


class ArtifactLocalizationDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        root_dir: str | Path,
        train: bool,
        augment: DetectionAugmentConfig,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.root_dir = Path(root_dir)
        self.train = train
        self.augment = augment
        self.df = pd.read_csv(self.csv_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_path = self.root_dir / row["image_relpath"]
        image = Image.open(image_path).convert("L")
        width = float(row["width"])
        height = float(row["height"])
        bbox = torch.tensor(
            [
                float(row["bbox_cx"]) / width,
                float(row["bbox_cy"]) / height,
                float(row["bbox_w"]) / width,
                float(row["bbox_h"]) / height,
            ],
            dtype=torch.float32,
        )

        if self.train and torch.rand(1).item() < self.augment.crop_prob:
            image, bbox = random_bbox_preserving_crop(image, bbox, self.augment)

        if self.train and torch.rand(1).item() < self.augment.hflip_prob:
            image = TF.hflip(image)
            bbox[0] = 1.0 - bbox[0]
        if self.train and torch.rand(1).item() < self.augment.vflip_prob:
            image = TF.vflip(image)
            bbox[1] = 1.0 - bbox[1]
        if self.train:
            image = maybe_apply_photometric_aug(
                image=image,
                brightness_jitter=self.augment.brightness_jitter,
                contrast_jitter=self.augment.contrast_jitter,
                autocontrast_prob=self.augment.autocontrast_prob,
                blur_prob=self.augment.blur_prob,
                blur_radius_min=self.augment.blur_radius_min,
                blur_radius_max=self.augment.blur_radius_max,
                noise_prob=self.augment.noise_prob,
                noise_std=self.augment.noise_std,
            )

        image_tensor = normalize_grayscale(image, self.augment.image_size)
        return {
            "image": image_tensor,
            "bbox": bbox,
            "image_path": str(image_path),
        }


@dataclass
class PatchAugmentConfig:
    image_size: int = 128
    hflip_prob: float = 0.5
    vflip_prob: float = 0.1
    crop_prob: float = 0.9
    crop_scale_min: float = 0.6
    crop_ratio_min: float = 0.75
    crop_ratio_max: float = 1.33
    rotate_deg: float = 15.0
    translate_frac: float = 0.08
    scale_min: float = 0.9
    scale_max: float = 1.1
    shear_deg: float = 8.0
    brightness_jitter: float = 0.2
    contrast_jitter: float = 0.25
    autocontrast_prob: float = 0.2
    equalize_prob: float = 0.1
    blur_prob: float = 0.2
    blur_radius_min: float = 0.1
    blur_radius_max: float = 1.2
    noise_prob: float = 0.25
    noise_std: float = 0.04


class ArtifactPatchDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        root_dir: str | Path,
        train: bool,
        augment: PatchAugmentConfig,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.root_dir = Path(root_dir)
        self.train = train
        self.augment = augment
        self.df = pd.read_csv(self.csv_path)
        self.image_paths: List[Path] = [self.root_dir / rel for rel in self.df["image_relpath"].tolist()]
        self.labels: List[int] = [int(item) for item in self.df["class_id"].tolist()]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path).convert("L")

        if self.train and torch.rand(1).item() < self.augment.crop_prob:
            top, left, height, width = RandomResizedCrop.get_params(
                image,
                scale=(self.augment.crop_scale_min, 1.0),
                ratio=(self.augment.crop_ratio_min, self.augment.crop_ratio_max),
            )
            image = TF.resized_crop(
                image,
                top,
                left,
                height,
                width,
                [self.augment.image_size, self.augment.image_size],
                interpolation=InterpolationMode.BILINEAR,
            )
        else:
            image = TF.resize(image, [self.augment.image_size, self.augment.image_size], interpolation=InterpolationMode.BILINEAR)

        if self.train and torch.rand(1).item() < self.augment.hflip_prob:
            image = TF.hflip(image)
        if self.train and torch.rand(1).item() < self.augment.vflip_prob:
            image = TF.vflip(image)
        if self.train and self.augment.rotate_deg > 0:
            angle = (torch.rand(1).item() * 2.0 - 1.0) * self.augment.rotate_deg
            image = TF.rotate(image, angle=angle, interpolation=InterpolationMode.BILINEAR, fill=0)
        if self.train and (
            self.augment.translate_frac > 0
            or self.augment.scale_min != 1.0
            or self.augment.scale_max != 1.0
            or self.augment.shear_deg > 0
        ):
            max_dx = int(round(self.augment.translate_frac * image.size[0]))
            max_dy = int(round(self.augment.translate_frac * image.size[1]))
            translate = (
                int(torch.randint(-max_dx, max_dx + 1, (1,)).item()) if max_dx > 0 else 0,
                int(torch.randint(-max_dy, max_dy + 1, (1,)).item()) if max_dy > 0 else 0,
            )
            scale = self.augment.scale_min + torch.rand(1).item() * (self.augment.scale_max - self.augment.scale_min)
            shear = (torch.rand(1).item() * 2.0 - 1.0) * self.augment.shear_deg
            image = TF.affine(
                image,
                angle=0.0,
                translate=translate,
                scale=scale,
                shear=[shear, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
        if self.train:
            image = maybe_apply_photometric_aug(
                image=image,
                brightness_jitter=self.augment.brightness_jitter,
                contrast_jitter=self.augment.contrast_jitter,
                autocontrast_prob=self.augment.autocontrast_prob,
                blur_prob=self.augment.blur_prob,
                blur_radius_min=self.augment.blur_radius_min,
                blur_radius_max=self.augment.blur_radius_max,
                noise_prob=self.augment.noise_prob,
                noise_std=self.augment.noise_std,
            )
        if self.train and self.augment.equalize_prob > 0 and torch.rand(1).item() < self.augment.equalize_prob:
            image = ImageOps.equalize(image)

        image_tensor = normalize_grayscale(image, None)
        return {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "image_path": str(image_path),
        }
