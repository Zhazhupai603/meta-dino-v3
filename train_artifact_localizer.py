from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data.artifact_datasets import ArtifactLocalizationDataset, DetectionAugmentConfig
from model.dinov3_artifact import DINOv3BoxRegModel


def cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box.unbind(dim=-1)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1)


def mean_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_xyxy = cxcywh_to_xyxy(pred)
    tgt_xyxy = cxcywh_to_xyxy(target)
    x1 = torch.maximum(pred_xyxy[:, 0], tgt_xyxy[:, 0])
    y1 = torch.maximum(pred_xyxy[:, 1], tgt_xyxy[:, 1])
    x2 = torch.minimum(pred_xyxy[:, 2], tgt_xyxy[:, 2])
    y2 = torch.minimum(pred_xyxy[:, 3], tgt_xyxy[:, 3])
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    pred_area = torch.clamp(pred_xyxy[:, 2] - pred_xyxy[:, 0], min=0) * torch.clamp(pred_xyxy[:, 3] - pred_xyxy[:, 1], min=0)
    tgt_area = torch.clamp(tgt_xyxy[:, 2] - tgt_xyxy[:, 0], min=0) * torch.clamp(tgt_xyxy[:, 3] - tgt_xyxy[:, 1], min=0)
    union = pred_area + tgt_area - inter
    return ((inter + eps) / (union + eps)).mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DINOv3 artifact localization model.")
    parser.add_argument("--data_root", type=Path, default=Path("data/processed/cca_artifact_dataset/artifact_localization"))
    parser.add_argument("--backbone_name", type=str, default="vit_base_patch16_dinov3")
    parser.add_argument("--checkpoint_path", type=Path, default=Path("download_ckpts/vit_base_patch16_dinov3/model_timm.pth"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/cca_artifact_localizer"))
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--hflip_prob", type=float, default=0.5)
    return parser.parse_args()


def run_epoch(
    model: DINOv3BoxRegModel,
    dataloader: DataLoader,
    optimizer: AdamW | None,
    device: torch.device,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    iou_sum = 0.0
    n = 0
    for batch in dataloader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["bbox"].to(device, non_blocking=True)
        pred = model(image)
        loss = F.smooth_l1_loss(pred, target)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            iou = mean_iou(pred, target)
        bs = image.shape[0]
        loss_sum += float(loss.item()) * bs
        iou_sum += float(iou.item()) * bs
        n += bs
    denom = max(n, 1)
    return {"loss": loss_sum / denom, "iou": iou_sum / denom}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    train_ds = ArtifactLocalizationDataset(
        csv_path=args.data_root / "splits/train.csv",
        root_dir=args.data_root,
        train=True,
        augment=DetectionAugmentConfig(image_size=args.image_size, hflip_prob=args.hflip_prob),
    )
    val_ds = ArtifactLocalizationDataset(
        csv_path=args.data_root / "splits/val.csv",
        root_dir=args.data_root,
        train=False,
        augment=DetectionAugmentConfig(
            image_size=args.image_size,
            hflip_prob=0.0,
            vflip_prob=0.0,
            crop_prob=0.0,
            brightness_jitter=0.0,
            contrast_jitter=0.0,
            autocontrast_prob=0.0,
            blur_prob=0.0,
            noise_prob=0.0,
        ),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = DINOv3BoxRegModel(
        backbone_name=args.backbone_name,
        checkpoint_path=args.checkpoint_path if args.checkpoint_path.exists() else None,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    best_iou = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, optimizer, device)
        with torch.no_grad():
            val_stats = run_epoch(model, val_loader, None, device)
        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})
        print(
            f"[Epoch {epoch:03d}] train_loss={train_stats['loss']:.4f} train_iou={train_stats['iou']:.4f} "
            f"val_loss={val_stats['loss']:.4f} val_iou={val_stats['iou']:.4f}"
        )
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_iou": val_stats["iou"],
            "args": vars(args),
            "task": "artifact_localization",
        }
        torch.save(checkpoint, args.output_dir / "checkpoints" / "last.pt")
        if val_stats["iou"] > best_iou:
            best_iou = val_stats["iou"]
            torch.save(checkpoint, args.output_dir / "checkpoints" / "best.pt")
        with (args.output_dir / "history.json").open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)
    print(f"Training finished. Best val IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
