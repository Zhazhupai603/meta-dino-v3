from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF


@dataclass
class AugmentConfig:
    image_size: int = 512
    hflip_prob: float = 0.5
    vflip_prob: float = 0.0
    crop_prob: float = 0.7
    crop_scale_min: float = 0.7
    crop_scale_max: float = 1.0
    rotate_prob: float = 0.3
    rotate_deg: float = 10.0
    brightness_prob: float = 0.3
    brightness_min: float = 0.85
    brightness_max: float = 1.15
    contrast_prob: float = 0.3
    contrast_min: float = 0.85
    contrast_max: float = 1.15
    blur_prob: float = 0.2
    blur_kernel_size: int = 3
    noise_prob: float = 0.2
    noise_std: float = 0.03


class CubsSegDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        root_dir: str | Path,
        train: bool,
        augment: AugmentConfig,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.root_dir = Path(root_dir)
        self.train = train
        self.augment = augment
        self.df = pd.read_csv(self.csv_path)
        self.num_classes = self._infer_num_classes()
        self.samples: List[Tuple[Path, Path]] = []
        for _, row in self.df.iterrows():
            image_path = self.root_dir / row["image_relpath"]
            mask_path = self.root_dir / row["mask_relpath"]
            self.samples.append((image_path, mask_path))

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return TF.normalize(
            image_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def _infer_num_classes(self) -> int:
        summary_path = self.root_dir / "dataset_summary.json"
        if summary_path.exists():
            try:
                with summary_path.open("r", encoding="utf-8") as file:
                    summary = json.load(file)
                num_classes = int(summary.get("num_classes", 2))
                return max(num_classes, 2)
            except Exception:
                return 2
        return 2

    def _decode_mask(self, mask: Image.Image) -> np.ndarray:
        mask_np = np.array(mask, dtype=np.uint8)
        return np.clip(mask_np, 0, self.num_classes - 1).astype(np.int64)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.train:
            if torch.rand(1).item() < self.augment.crop_prob:
                i, j, h, w = T.RandomResizedCrop.get_params(
                    image,
                    scale=(self.augment.crop_scale_min, self.augment.crop_scale_max),
                    ratio=(0.9, 1.1),
                )
                image = TF.resized_crop(
                    image,
                    i,
                    j,
                    h,
                    w,
                    [self.augment.image_size, self.augment.image_size],
                    interpolation=TF.InterpolationMode.BILINEAR,
                )
                mask = TF.resized_crop(
                    mask,
                    i,
                    j,
                    h,
                    w,
                    [self.augment.image_size, self.augment.image_size],
                    interpolation=TF.InterpolationMode.NEAREST,
                )
            else:
                image = TF.resize(
                    image,
                    [self.augment.image_size, self.augment.image_size],
                    interpolation=TF.InterpolationMode.BILINEAR,
                )
                mask = TF.resize(
                    mask,
                    [self.augment.image_size, self.augment.image_size],
                    interpolation=TF.InterpolationMode.NEAREST,
                )
            if torch.rand(1).item() < self.augment.hflip_prob:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if torch.rand(1).item() < self.augment.vflip_prob:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            if torch.rand(1).item() < self.augment.rotate_prob:
                angle = float(torch.empty(1).uniform_(-self.augment.rotate_deg, self.augment.rotate_deg).item())
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0)
            if torch.rand(1).item() < self.augment.brightness_prob:
                factor = float(
                    torch.empty(1).uniform_(self.augment.brightness_min, self.augment.brightness_max).item()
                )
                image = TF.adjust_brightness(image, factor)
            if torch.rand(1).item() < self.augment.contrast_prob:
                factor = float(torch.empty(1).uniform_(self.augment.contrast_min, self.augment.contrast_max).item())
                image = TF.adjust_contrast(image, factor)
        else:
            image = TF.resize(
                image,
                [self.augment.image_size, self.augment.image_size],
                interpolation=TF.InterpolationMode.BILINEAR,
            )
            mask = TF.resize(
                mask,
                [self.augment.image_size, self.augment.image_size],
                interpolation=TF.InterpolationMode.NEAREST,
            )

        image_tensor = TF.to_tensor(image)  # [1,H,W]
        if self.train and torch.rand(1).item() < self.augment.blur_prob:
            k = int(self.augment.blur_kernel_size)
            if k % 2 == 0:
                k += 1
            k = max(k, 3)
            image_tensor = TF.gaussian_blur(image_tensor, kernel_size=[k, k], sigma=[0.1, 2.0])
        if self.train and torch.rand(1).item() < self.augment.noise_prob:
            noise = torch.randn_like(image_tensor) * float(self.augment.noise_std)
            image_tensor = torch.clamp(image_tensor + noise, 0.0, 1.0)
        image_tensor = image_tensor.repeat(3, 1, 1)  # [3,H,W] for ViT
        image_tensor = self._normalize(image_tensor)

        mask_np = self._decode_mask(mask)
        mask_tensor = torch.from_numpy(mask_np)  # [H,W], class ids

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }
