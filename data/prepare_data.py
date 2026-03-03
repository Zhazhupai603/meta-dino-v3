from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


@dataclass
class SampleRecord:
    sample_id: str
    patient_id: str
    source: str
    image_relpath: str
    mask_relpath: str
    li_relpath: str
    ma_relpath: str
    width: int
    height: int


def read_points(path: Path) -> np.ndarray:
    points: List[Tuple[float, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            points.append((x, y))
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array(points, dtype=np.float32)


def make_imt_mask(width: int, height: int, li_points: np.ndarray, ma_points: np.ndarray) -> Image.Image:
    if len(li_points) < 2 or len(ma_points) < 2:
        return Image.new("L", (width, height), color=0)

    li_sorted = li_points[np.argsort(li_points[:, 0])]
    ma_sorted = ma_points[np.argsort(ma_points[:, 0])]

    x_start = int(max(li_sorted[:, 0].min(), ma_sorted[:, 0].min()))
    x_end = int(min(li_sorted[:, 0].max(), ma_sorted[:, 0].max()))
    if x_end <= x_start:
        return Image.new("L", (width, height), color=0)

    xs = np.arange(x_start, x_end + 1, dtype=np.float32)
    y_li = np.interp(xs, li_sorted[:, 0], li_sorted[:, 1])
    y_ma = np.interp(xs, ma_sorted[:, 0], ma_sorted[:, 1])

    y_top = np.minimum(y_li, y_ma)
    y_bottom = np.maximum(y_li, y_ma)

    polygon_top = list(zip(xs.tolist(), y_top.tolist()))
    polygon_bottom = list(zip(xs[::-1].tolist(), y_bottom[::-1].tolist()))
    polygon = polygon_top + polygon_bottom

    mask = Image.new("L", (width, height), color=0)
    drawer = ImageDraw.Draw(mask)
    drawer.polygon(polygon, fill=255)
    return mask


def save_png_grayscale(src_tiff: Path, dst_png: Path) -> Tuple[int, int]:
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(src_tiff).convert("L")
    width, height = image.size
    image.save(dst_png, format="PNG")
    return width, height


def save_png_mask_binary(src_mask: Path, dst_png: Path) -> Tuple[int, int]:
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    mask = Image.open(src_mask).convert("L")
    width, height = mask.size
    mask_np = (np.array(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    out = Image.fromarray(mask_np, mode="L")
    out.save(dst_png, format="PNG")
    return width, height


def collect_cubs_v1(
    cubs_v1_root: Path,
    out_root: Path,
    manual_source: str,
    max_samples: Optional[int],
) -> List[SampleRecord]:
    images_dir = cubs_v1_root / "IMAGES"
    seg_dir = cubs_v1_root / "SEGMENTATIONS" / manual_source
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing CUBS v1 images directory: {images_dir}")
    if not seg_dir.exists():
        raise FileNotFoundError(f"Missing CUBS v1 segmentation directory: {seg_dir}")

    records: List[SampleRecord] = []
    image_files = sorted(images_dir.glob("*.tiff"))
    if max_samples is not None:
        image_files = image_files[:max_samples]

    for image_file in image_files:
        stem = image_file.stem
        li_file = seg_dir / f"{stem}-LI.txt"
        ma_file = seg_dir / f"{stem}-MA.txt"
        if not li_file.exists() or not ma_file.exists():
            continue

        sample_id = f"v1_{stem}"
        patient_id = "_".join(stem.split("_")[:2]) if stem.startswith("clin_") else stem
        image_rel = Path("images") / f"{sample_id}.png"
        mask_rel = Path("masks") / f"{sample_id}.png"

        width, height = save_png_grayscale(image_file, out_root / image_rel)
        li_points = read_points(li_file)
        ma_points = read_points(ma_file)
        mask = make_imt_mask(width, height, li_points, ma_points)
        (out_root / mask_rel).parent.mkdir(parents=True, exist_ok=True)
        mask.save(out_root / mask_rel, format="PNG")

        records.append(
            SampleRecord(
                sample_id=sample_id,
                patient_id=patient_id,
                source="cubs_v1",
                image_relpath=image_rel.as_posix(),
                mask_relpath=mask_rel.as_posix(),
                li_relpath=li_file.relative_to(cubs_v1_root).as_posix(),
                ma_relpath=ma_file.relative_to(cubs_v1_root).as_posix(),
                width=width,
                height=height,
            )
        )
    return records


def collect_cca(
    cca_root: Path,
    out_root: Path,
    max_samples: Optional[int],
) -> List[SampleRecord]:
    images_dir = cca_root / "US images"
    masks_dir = cca_root / "Expert mask images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing CCA image directory: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Missing CCA mask directory: {masks_dir}")

    records: List[SampleRecord] = []
    image_files = sorted(images_dir.glob("*.png"))
    if max_samples is not None:
        image_files = image_files[:max_samples]

    for image_file in image_files:
        stem = image_file.stem
        mask_file = masks_dir / f"{stem}.png"
        if not mask_file.exists():
            continue

        sample_id = f"cca_{stem}"
        patient_id = stem.split("_slice_")[0] if "_slice_" in stem else stem
        image_rel = Path("images") / f"{sample_id}.png"
        mask_rel = Path("masks") / f"{sample_id}.png"

        width, height = save_png_grayscale(image_file, out_root / image_rel)
        save_png_mask_binary(mask_file, out_root / mask_rel)

        records.append(
            SampleRecord(
                sample_id=sample_id,
                patient_id=patient_id,
                source="cca",
                image_relpath=image_rel.as_posix(),
                mask_relpath=mask_rel.as_posix(),
                li_relpath="",
                ma_relpath="",
                width=width,
                height=height,
            )
        )
    return records


def collect_cubs_v2(
    cubs_v2_root: Path,
    out_root: Path,
    manual_source: str,
    max_samples: Optional[int],
) -> List[SampleRecord]:
    images_dir = cubs_v2_root / "images"
    profile_dir = cubs_v2_root / "LIMA-Profiles" / manual_source
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing CUBS v2 images directory: {images_dir}")
    if not profile_dir.exists():
        raise FileNotFoundError(f"Missing CUBS v2 profile directory: {profile_dir}")

    records: List[SampleRecord] = []
    image_files = sorted(images_dir.glob("*.tiff"))
    if max_samples is not None:
        image_files = image_files[:max_samples]

    for image_file in image_files:
        stem = image_file.stem
        li_file = profile_dir / f"{stem}-LI.txt"
        ma_file = profile_dir / f"{stem}-MA.txt"
        if not li_file.exists() or not ma_file.exists():
            continue

        sample_id = f"v2_{stem}"
        patient_id = stem
        image_rel = Path("images") / f"{sample_id}.png"
        mask_rel = Path("masks") / f"{sample_id}.png"

        width, height = save_png_grayscale(image_file, out_root / image_rel)
        li_points = read_points(li_file)
        ma_points = read_points(ma_file)
        mask = make_imt_mask(width, height, li_points, ma_points)
        (out_root / mask_rel).parent.mkdir(parents=True, exist_ok=True)
        mask.save(out_root / mask_rel, format="PNG")

        records.append(
            SampleRecord(
                sample_id=sample_id,
                patient_id=patient_id,
                source="cubs_v2",
                image_relpath=image_rel.as_posix(),
                mask_relpath=mask_rel.as_posix(),
                li_relpath=li_file.relative_to(cubs_v2_root).as_posix(),
                ma_relpath=ma_file.relative_to(cubs_v2_root).as_posix(),
                width=width,
                height=height,
            )
        )
    return records


def split_by_group(
    records: List[SampleRecord],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, List[SampleRecord]]:
    group_to_records: Dict[str, List[SampleRecord]] = {}
    for record in records:
        group_to_records.setdefault(record.patient_id, []).append(record)

    groups = list(group_to_records.keys())
    random.Random(seed).shuffle(groups)

    n_groups = len(groups)
    if n_groups <= 2:
        n_train = max(1, n_groups - 1)
        n_val = 0
    else:
        n_train = max(1, int(n_groups * train_ratio))
        n_val = max(1, int(n_groups * val_ratio))
        if n_train + n_val >= n_groups:
            n_val = max(1, n_groups - n_train - 1)
            if n_train + n_val >= n_groups:
                n_train = max(1, n_groups - 2)
                n_val = 1
    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train : n_train + n_val])
    test_groups = set(groups[n_train + n_val :])

    train_records: List[SampleRecord] = []
    val_records: List[SampleRecord] = []
    test_records: List[SampleRecord] = []
    for group_name, group_records in group_to_records.items():
        if group_name in train_groups:
            train_records.extend(group_records)
        elif group_name in val_groups:
            val_records.extend(group_records)
        else:
            test_records.extend(group_records)

    return {"train": train_records, "val": val_records, "test": test_records}


def to_dataframe(records: Iterable[SampleRecord]) -> pd.DataFrame:
    return pd.DataFrame([record.__dict__ for record in records])


def parse_args() -> argparse.Namespace:
    default_v1 = (
        "data/extracted/"
        "DATASET for Carotid Ultrasound Boundary Study (CUBS) an open multi-center analysis of computerized intima-media thickness measurement systems and their clinical impact/"
        "DATASET for Carotid Ultrasound Boundary Study (CUBS) an open multi-center analysis of computerized intima-media thickness measurement systems and their clinical impact"
    )
    default_v2 = (
        "data/extracted/"
        "m7ndn58sv6-1/m7ndn58sv6-1/"
        "DATASET_CUBS_tech/DATASET_CUBS_tech"
    )
    default_cca = (
        "data/extracted/"
        "Common Carotid Artery Ultrasound Images/"
        "Common Carotid Artery Ultrasound Images"
    )
    parser = argparse.ArgumentParser(
        description="Prepare unified segmentation dataset from selected sources (cubs_v1,cubs_v2,cca)."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="cubs_v1,cubs_v2",
        help="Comma-separated list from {cubs_v1,cubs_v2,cca}. e.g. cubs_v1,cubs_v2,cca",
    )
    parser.add_argument("--cubs_v1_root", type=Path, default=Path(default_v1))
    parser.add_argument("--cubs_v2_root", type=Path, default=Path(default_v2))
    parser.add_argument("--cca_root", type=Path, default=Path(default_cca))
    parser.add_argument("--out_root", type=Path, default=Path("data/processed/unified_dataset"))
    parser.add_argument("--v1_manual_source", type=str, default="Manual-A1")
    parser.add_argument("--v2_manual_source", type=str, default="Manual-A1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_samples_v1", type=int, default=None)
    parser.add_argument("--max_samples_v2", type=int, default=None)
    parser.add_argument("--max_samples_cca", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [item.strip().lower() for item in args.datasets.split(",") if item.strip()]
    valid = {"cubs_v1", "cubs_v2", "cca"}
    invalid = [item for item in selected if item not in valid]
    if invalid:
        raise ValueError(f"Unsupported datasets {invalid}. Valid options: {sorted(valid)}")
    if not selected:
        raise ValueError("No dataset selected. Use --datasets with at least one item.")

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "masks").mkdir(parents=True, exist_ok=True)
    (out_root / "splits").mkdir(parents=True, exist_ok=True)

    v1_records: List[SampleRecord] = []
    v2_records: List[SampleRecord] = []
    cca_records: List[SampleRecord] = []

    if "cubs_v1" in selected:
        v1_records = collect_cubs_v1(
            cubs_v1_root=args.cubs_v1_root,
            out_root=out_root,
            manual_source=args.v1_manual_source,
            max_samples=args.max_samples_v1,
        )
    if "cubs_v2" in selected:
        v2_records = collect_cubs_v2(
            cubs_v2_root=args.cubs_v2_root,
            out_root=out_root,
            manual_source=args.v2_manual_source,
            max_samples=args.max_samples_v2,
        )
    if "cca" in selected:
        cca_records = collect_cca(
            cca_root=args.cca_root,
            out_root=out_root,
            max_samples=args.max_samples_cca,
        )

    all_records = v1_records + v2_records + cca_records
    if not all_records:
        raise RuntimeError("No valid samples were produced. Please check input roots and manual source folders.")

    split_records = split_by_group(
        records=all_records,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    all_df = to_dataframe(all_records)
    all_df.to_csv(out_root / "metadata_all.csv", index=False)
    for split_name, records in split_records.items():
        split_df = to_dataframe(records)
        split_df.to_csv(out_root / "splits" / f"{split_name}.csv", index=False)

    summary = {
        "selected_datasets": selected,
        "num_total": len(all_records),
        "num_v1": len(v1_records),
        "num_v2": len(v2_records),
        "num_cca": len(cca_records),
        "num_train": len(split_records["train"]),
        "num_val": len(split_records["val"]),
        "num_test": len(split_records["test"]),
        "v1_root": str(args.cubs_v1_root),
        "v2_root": str(args.cubs_v2_root),
        "cca_root": str(args.cca_root),
        "output_root": str(out_root),
    }
    with (out_root / "dataset_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved metadata to: {out_root / 'metadata_all.csv'}")
    print(f"Saved splits to: {out_root / 'splits'}")


if __name__ == "__main__":
    main()
