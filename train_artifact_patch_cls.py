from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data.artifact_datasets import ArtifactPatchDataset, PatchAugmentConfig
from model.dinov3_artifact import DINOv3PatchClsModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DINOv3 artifact patch classifier.")
    parser.add_argument("--data_root", type=Path, default=Path("data/processed/cca_artifact_dataset/artifact_patch_cls"))
    parser.add_argument("--backbone_name", type=str, default="vit_base_patch16_dinov3")
    parser.add_argument("--checkpoint_path", type=Path, default=Path("download_ckpts/vit_base_patch16_dinov3/model_timm.pth"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/cca_artifact_patch_cls"))
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--hflip_prob", type=float, default=0.5)
    return parser.parse_args()


def run_epoch(
    model: DINOv3PatchClsModel,
    dataloader: DataLoader,
    optimizer: AdamW | None,
    device: torch.device,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    ce_loss = nn.CrossEntropyLoss()
    loss_sum = 0.0
    acc_sum = 0.0
    n = 0
    for batch in dataloader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        logits = model(image)
        loss = ce_loss(logits, target)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        pred = torch.argmax(logits, dim=1)
        acc = (pred == target).float().mean()
        bs = image.shape[0]
        loss_sum += float(loss.item()) * bs
        acc_sum += float(acc.item()) * bs
        n += bs
    denom = max(n, 1)
    return {"loss": loss_sum / denom, "acc": acc_sum / denom}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    train_ds = ArtifactPatchDataset(
        csv_path=args.data_root / "splits/train.csv",
        root_dir=args.data_root,
        train=True,
        augment=PatchAugmentConfig(image_size=args.image_size, hflip_prob=args.hflip_prob),
    )
    val_ds = ArtifactPatchDataset(
        csv_path=args.data_root / "splits/val.csv",
        root_dir=args.data_root,
        train=False,
        augment=PatchAugmentConfig(
            image_size=args.image_size,
            hflip_prob=0.0,
            vflip_prob=0.0,
            crop_prob=0.0,
            rotate_deg=0.0,
            translate_frac=0.0,
            scale_min=1.0,
            scale_max=1.0,
            shear_deg=0.0,
            brightness_jitter=0.0,
            contrast_jitter=0.0,
            autocontrast_prob=0.0,
            equalize_prob=0.0,
            blur_prob=0.0,
            noise_prob=0.0,
        ),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = DINOv3PatchClsModel(
        backbone_name=args.backbone_name,
        checkpoint_path=args.checkpoint_path if args.checkpoint_path.exists() else None,
        num_classes=2,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    best_acc = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, optimizer, device)
        with torch.no_grad():
            val_stats = run_epoch(model, val_loader, None, device)
        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})
        print(
            f"[Epoch {epoch:03d}] train_loss={train_stats['loss']:.4f} train_acc={train_stats['acc']:.4f} "
            f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['acc']:.4f}"
        )
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_stats["acc"],
            "args": vars(args),
            "task": "artifact_patch_classification",
            "class_mapping": {"artifact_fake": 0, "real_artery": 1},
        }
        torch.save(checkpoint, args.output_dir / "checkpoints" / "last.pt")
        if val_stats["acc"] > best_acc:
            best_acc = val_stats["acc"]
            torch.save(checkpoint, args.output_dir / "checkpoints" / "best.pt")
        with (args.output_dir / "history.json").open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)
    print(f"Training finished. Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
