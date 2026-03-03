from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data.dataset import AugmentConfig, CubsSegDataset
from model.dinov3_seg import DINOv3SegModel


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)[:, 1]
    target = target.float()
    inter = (probs * target).sum(dim=(1, 2))
    union = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def batch_metrics(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> Dict[str, torch.Tensor]:
    pred = torch.argmax(logits, dim=1)
    pred_f = pred.float()
    target_f = target.float()
    inter = (pred_f * target_f).sum(dim=(1, 2))
    pred_sum = pred_f.sum(dim=(1, 2))
    tgt_sum = target_f.sum(dim=(1, 2))
    union = pred_sum + tgt_sum - inter
    dice = ((2 * inter + eps) / (pred_sum + tgt_sum + eps)).mean()
    iou = ((inter + eps) / (union + eps)).mean()
    return {"dice": dice, "iou": iou}


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW | None,
    accelerator: Accelerator,
    ce_weight: float,
    dice_weight: float,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    ce_loss_fn = nn.CrossEntropyLoss()

    local_loss_sum = torch.tensor(0.0, device=accelerator.device)
    local_dice_sum = torch.tensor(0.0, device=accelerator.device)
    local_iou_sum = torch.tensor(0.0, device=accelerator.device)
    local_weight = torch.tensor(0.0, device=accelerator.device)

    for batch in dataloader:
        image = batch["image"].to(accelerator.device, non_blocking=True)
        target = batch["mask"].to(accelerator.device, non_blocking=True)
        batch_size = image.shape[0]

        if training:
            with accelerator.accumulate(model):
                logits = model(image)
                ce = ce_loss_fn(logits, target)
                dl = dice_loss(logits, target)
                loss = ce_weight * ce + dice_weight * dl
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        else:
            logits = model(image)
            ce = ce_loss_fn(logits, target)
            dl = dice_loss(logits, target)
            loss = ce_weight * ce + dice_weight * dl

        metrics = batch_metrics(logits.detach(), target)
        local_loss_sum += loss.detach() * batch_size
        local_dice_sum += metrics["dice"].detach() * batch_size
        local_iou_sum += metrics["iou"].detach() * batch_size
        local_weight += batch_size

    total_loss_sum = accelerator.reduce(local_loss_sum, reduction="sum")
    total_dice_sum = accelerator.reduce(local_dice_sum, reduction="sum")
    total_iou_sum = accelerator.reduce(local_iou_sum, reduction="sum")
    total_weight = accelerator.reduce(local_weight, reduction="sum")

    denom = max(float(total_weight.item()), 1.0)
    return {
        "loss": float(total_loss_sum.item() / denom),
        "dice": float(total_dice_sum.item() / denom),
        "iou": float(total_iou_sum.item() / denom),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DINOv3 segmentation model on merged CUBS dataset.")
    parser.add_argument("--data_root", type=Path, default=Path("data/processed/unified_dataset"))
    parser.add_argument("--backbone_name", type=str, default="vit_base_patch16_dinov3")
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("download_ckpts/dinov3-vitb16-pretrain-lvd1689m/model.safetensors"),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/cubs_dinov3_timm"))
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--max_interval_ckpts", type=int, default=3)
    parser.add_argument("--val_generate_count", type=int, default=20)
    parser.add_argument("--hflip_prob", type=float, default=0.5)
    parser.add_argument("--vflip_prob", type=float, default=0.1)
    parser.add_argument("--crop_prob", type=float, default=0.7)
    parser.add_argument("--crop_scale_min", type=float, default=0.7)
    parser.add_argument("--crop_scale_max", type=float, default=1.0)
    parser.add_argument("--rotate_prob", type=float, default=0.3)
    parser.add_argument("--rotate_deg", type=float, default=10.0)
    parser.add_argument("--brightness_prob", type=float, default=0.3)
    parser.add_argument("--brightness_min", type=float, default=0.85)
    parser.add_argument("--brightness_max", type=float, default=1.15)
    parser.add_argument("--contrast_prob", type=float, default=0.3)
    parser.add_argument("--contrast_min", type=float, default=0.85)
    parser.add_argument("--contrast_max", type=float, default=1.15)
    parser.add_argument("--blur_prob", type=float, default=0.2)
    parser.add_argument("--blur_kernel_size", type=int, default=3)
    parser.add_argument("--noise_prob", type=float, default=0.2)
    parser.add_argument("--noise_std", type=float, default=0.03)
    return parser.parse_args()


def _denorm_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().float().clone()
    if image.ndim == 3 and image.shape[0] >= 1:
        image = image[0]
    image = image * 0.229 + 0.485
    image = image.clamp(0, 1)
    return (image.numpy() * 255.0).astype(np.uint8)


@torch.no_grad()
def generate_val_predictions(
    model: nn.Module,
    dataset: CubsSegDataset,
    output_dir: Path,
    accelerator: Accelerator,
    epoch: int,
    count: int,
    seed: int,
) -> None:
    if count <= 0 or len(dataset) == 0:
        return
    model.eval()
    images_dir = output_dir / "images"
    pred_masks_dir = output_dir / "pred_masks"
    gt_masks_dir = output_dir / "gt_masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    pred_masks_dir.mkdir(parents=True, exist_ok=True)
    gt_masks_dir.mkdir(parents=True, exist_ok=True)

    for folder in (images_dir, pred_masks_dir, gt_masks_dir):
        for old_file in folder.glob("*.png"):
            old_file.unlink(missing_ok=True)

    rng = random.Random(seed + epoch)
    num_pick = min(count, len(dataset))
    indices = rng.sample(list(range(len(dataset))), num_pick)

    for rank, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(accelerator.device, non_blocking=True)
        logits = model(image)
        pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8) * 255
        gt = sample["mask"].detach().cpu().numpy().astype(np.uint8) * 255
        img = _denorm_to_uint8(sample["image"])

        stem = Path(sample["image_path"]).stem
        file_name = f"{rank:02d}_{stem}.png"
        Image.fromarray(img, mode="L").save(images_dir / file_name)
        Image.fromarray(pred, mode="L").save(pred_masks_dir / file_name)
        Image.fromarray(gt, mode="L").save(gt_masks_dir / file_name)


def prune_interval_checkpoints(checkpoints_dir: Path, max_interval_ckpts: int) -> None:
    if max_interval_ckpts < 0:
        return
    ckpts = sorted(checkpoints_dir.glob("epoch_*.pt"))
    if len(ckpts) <= max_interval_ckpts:
        return
    for old_ckpt in ckpts[: len(ckpts) - max_interval_ckpts]:
        old_ckpt.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    accelerator.print(f"Using device: {accelerator.device}")

    train_csv = args.data_root / "splits/train.csv"
    val_csv = args.data_root / "splits/val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            f"Missing split files under {args.data_root}/splits. "
            "Please run data/prepare_data.py first."
        )

    train_ds = CubsSegDataset(
        csv_path=train_csv,
        root_dir=args.data_root,
        train=True,
        augment=AugmentConfig(
            image_size=args.image_size,
            hflip_prob=args.hflip_prob,
            vflip_prob=args.vflip_prob,
            crop_prob=args.crop_prob,
            crop_scale_min=args.crop_scale_min,
            crop_scale_max=args.crop_scale_max,
            rotate_prob=args.rotate_prob,
            rotate_deg=args.rotate_deg,
            brightness_prob=args.brightness_prob,
            brightness_min=args.brightness_min,
            brightness_max=args.brightness_max,
            contrast_prob=args.contrast_prob,
            contrast_min=args.contrast_min,
            contrast_max=args.contrast_max,
            blur_prob=args.blur_prob,
            blur_kernel_size=args.blur_kernel_size,
            noise_prob=args.noise_prob,
            noise_std=args.noise_std,
        ),
    )
    val_ds = CubsSegDataset(
        csv_path=val_csv,
        root_dir=args.data_root,
        train=False,
        augment=AugmentConfig(image_size=args.image_size, hflip_prob=0.0, vflip_prob=0.0),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = DINOv3SegModel(
        backbone_name=args.backbone_name,
        checkpoint_path=args.checkpoint_path if args.checkpoint_path.exists() else None,
        num_classes=2,
        freeze_backbone=args.freeze_backbone,
    )

    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    best_dice = -1.0
    history = []
    checkpoints_dir = args.output_dir / "checkpoints"
    generate_root = args.output_dir / "generate"
    last_val_stats: Dict[str, float] | None = None
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            accelerator=accelerator,
            ce_weight=args.ce_weight,
            dice_weight=args.dice_weight,
        )
        should_validate = args.val_interval > 0 and (epoch % args.val_interval == 0 or epoch == args.epochs)
        val_stats: Dict[str, float] | None = None
        if should_validate:
            with torch.no_grad():
                val_stats = run_epoch(
                    model=model,
                    dataloader=val_loader,
                    optimizer=None,
                    accelerator=accelerator,
                    ce_weight=args.ce_weight,
                    dice_weight=args.dice_weight,
                )
            last_val_stats = val_stats

        record = {"epoch": epoch, "train": train_stats, "val": val_stats}
        if accelerator.is_main_process:
            history.append(record)
        if val_stats is None:
            accelerator.print(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_stats['loss']:.4f} train_dice={train_stats['dice']:.4f} "
                "(skip val)"
            )
        else:
            accelerator.print(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_stats['loss']:.4f} train_dice={train_stats['dice']:.4f} "
                f"val_loss={val_stats['loss']:.4f} val_dice={val_stats['dice']:.4f} val_iou={val_stats['iou']:.4f}"
            )

        if accelerator.is_main_process:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": None if last_val_stats is None else last_val_stats["dice"],
                "args": vars(args),
            }
            torch.save(checkpoint, checkpoints_dir / "last.pt")
            if val_stats is not None and val_stats["dice"] > best_dice:
                best_dice = val_stats["dice"]
                torch.save(checkpoint, checkpoints_dir / "best.pt")
            if args.save_interval > 0 and epoch % args.save_interval == 0:
                torch.save(checkpoint, checkpoints_dir / f"epoch_{epoch:03d}.pt")
                prune_interval_checkpoints(checkpoints_dir, args.max_interval_ckpts)

            if val_stats is not None:
                generate_val_predictions(
                    model=accelerator.unwrap_model(model),
                    dataset=val_ds,
                    output_dir=generate_root,
                    accelerator=accelerator,
                    epoch=epoch,
                    count=args.val_generate_count,
                    seed=args.seed,
                )

            with (args.output_dir / "history.json").open("w", encoding="utf-8") as file:
                json.dump(history, file, indent=2)
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"Training finished. Best val dice: {best_dice:.4f}")
        print(f"Artifacts saved at: {args.output_dir}")


if __name__ == "__main__":
    main()
