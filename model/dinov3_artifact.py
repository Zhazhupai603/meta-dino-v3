from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from model.dinov3_seg import enforce_offline_mode, load_state_dict_maybe_nested


class DINOv3ArtifactBackbone(nn.Module):
    def __init__(self, backbone_name: str, checkpoint_path: Path | None, freeze_backbone: bool) -> None:
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

        self.embed_dim = getattr(self.backbone, "num_features", None)
        if self.embed_dim is None:
            raise RuntimeError("Unable to infer backbone embed dimension from timm model.")

    def _tokens_to_pooled(self, token_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, channels = token_tensor.shape
        side = int(math.isqrt(num_tokens))
        square_tokens = side * side
        if square_tokens != num_tokens:
            num_extra = num_tokens - square_tokens
            token_tensor = token_tensor[:, num_extra:, :]
        return token_tensor.mean(dim=1)

    def forward_features_pooled(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        if isinstance(features, dict):
            if "x_norm_patchtokens" in features:
                return self._tokens_to_pooled(features["x_norm_patchtokens"])
            if "x_prenorm" in features:
                return self._tokens_to_pooled(features["x_prenorm"])
            raise RuntimeError(f"Unsupported feature dict keys: {list(features.keys())}")
        if torch.is_tensor(features):
            if features.ndim == 4:
                return features.mean(dim=(2, 3))
            if features.ndim == 3:
                return self._tokens_to_pooled(features)
        raise RuntimeError(f"Unsupported feature type: {type(features)}")


class DINOv3BoxRegModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        checkpoint_path: Path | None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = DINOv3ArtifactBackbone(backbone_name, checkpoint_path, freeze_backbone)
        embed_dim = self.encoder.embed_dim
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.encoder.forward_features_pooled(x)
        return self.head(pooled)


class DINOv3PatchClsModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        checkpoint_path: Path | None,
        num_classes: int = 2,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = DINOv3ArtifactBackbone(backbone_name, checkpoint_path, freeze_backbone)
        embed_dim = self.encoder.embed_dim
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.encoder.forward_features_pooled(x)
        return self.head(pooled)
