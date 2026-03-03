from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def enforce_offline_mode() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TIMM_USE_HF_HUB", "0")


def load_state_dict_maybe_nested(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    enforce_offline_mode()
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file as load_safetensors

        state_dict = load_safetensors(str(checkpoint_path))
    else:
        raw = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(raw, dict):
            if "state_dict" in raw and isinstance(raw["state_dict"], dict):
                state_dict = raw["state_dict"]
            elif "model" in raw and isinstance(raw["model"], dict):
                state_dict = raw["model"]
            else:
                state_dict = raw
        else:
            raise ValueError(f"Unsupported checkpoint object type: {type(raw)}")

    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model.", "backbone.", "encoder."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


class DINOv3SegModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        checkpoint_path: Path | None,
        num_classes: int = 2,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        enforce_offline_mode()
        import timm

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
            global_pool="",
        )
        if checkpoint_path is not None:
            state_dict = load_state_dict_maybe_nested(checkpoint_path)
            msg = self.backbone.load_state_dict(state_dict, strict=False)
            print("Checkpoint load message:", msg)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        embed_dim = getattr(self.backbone, "num_features", None)
        if embed_dim is None:
            raise RuntimeError("Unable to infer backbone embed dimension from timm model.")

        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, num_classes, kernel_size=1),
        )

    def _tokens_to_map(self, token_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, channels = token_tensor.shape
        side = int(math.isqrt(num_tokens))
        square_tokens = side * side
        if square_tokens != num_tokens:
            if square_tokens <= 0:
                raise RuntimeError(f"Invalid token count: {num_tokens}")
            num_extra = num_tokens - square_tokens
            if num_extra <= 0:
                raise RuntimeError(f"Token count {num_tokens} cannot be reshaped to square feature map.")
            token_tensor = token_tensor[:, num_extra:, :]
            num_tokens = token_tensor.shape[1]
            side = int(math.isqrt(num_tokens))
            if side * side != num_tokens:
                raise RuntimeError(f"Token count {num_tokens} cannot be reshaped to square feature map.")
        fmap = token_tensor.transpose(1, 2).reshape(batch_size, channels, side, side)
        return fmap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        if isinstance(features, dict):
            if "x_norm_patchtokens" in features:
                fmap = self._tokens_to_map(features["x_norm_patchtokens"])
            elif "x_prenorm" in features:
                fmap = self._tokens_to_map(features["x_prenorm"])
            else:
                raise RuntimeError(f"Unsupported feature dict keys: {list(features.keys())}")
        elif torch.is_tensor(features):
            if features.ndim == 4:
                fmap = features
            elif features.ndim == 3:
                fmap = self._tokens_to_map(features)
            else:
                raise RuntimeError(f"Unexpected feature tensor ndim: {features.ndim}")
        else:
            raise RuntimeError(f"Unsupported feature type: {type(features)}")

        logits = self.head(fmap)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits
