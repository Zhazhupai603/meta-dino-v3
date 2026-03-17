from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


@dataclass
class DetectionRecord:
    sample_id: str
    patient_id: str
    source: str
    subset: str
    image_relpath: str
    bbox_relpath: str
    width: int
    height: int
    bbox_cx: float
    bbox_cy: float
    bbox_w: float
    bbox_h: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float


@dataclass
class PatchRecord:
    sample_id: str
    patient_id: str
    source: str
    subset: str
    image_relpath: str
    class_name: str
    class_id: int
    width: int
    height: int


def infer_patient_id(stem: str) -> str:
    if "_slice_" in stem:
        return stem.split("_slice_")[0]
    if "_" in stem:
        return stem.split("_")[0]
    if " " in stem:
        parts = stem.rsplit(" ", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
    return stem


def parse_bbox_txt(path: Path) -> Tuple[float, float, float, float]:
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [item for item in re.split(r"[;,\s]+", line) if item]
            if len(parts) >= 4:
                return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
    raise ValueError(f"Unable to parse bbox from: {path}")


def save_png_grayscale(src_image: Path, dst_png: Path) -> Tuple[int, int]:
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(src_image).convert("L")
    width, height = image.size
    image.save(dst_png, format="PNG")
    return width, height


def split_by_group(records: List[DetectionRecord], seed: int, train_ratio: float) -> Dict[str, List[DetectionRecord]]:
    group_to_records: Dict[str, List[DetectionRecord]] = {}
    for record in records:
        group_to_records.setdefault(record.patient_id, []).append(record)

    groups = list(group_to_records.keys())
    random.Random(seed).shuffle(groups)
    n_train = max(1, int(len(groups) * train_ratio))
    if n_train >= len(groups):
        n_train = max(1, len(groups) - 1)
    train_groups = set(groups[:n_train])

    train_records: List[DetectionRecord] = []
    val_records: List[DetectionRecord] = []
    for group_name, group_records in group_to_records.items():
        if group_name in train_groups:
            train_records.extend(group_records)
        else:
            val_records.extend(group_records)
    return {"train": train_records, "val": val_records}


def split_patch_records(
    records: List[PatchRecord],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, List[PatchRecord]]:
    rng = random.Random(seed)
    by_class: Dict[int, List[PatchRecord]] = {}
    for record in records:
        by_class.setdefault(record.class_id, []).append(record)

    split_records: Dict[str, List[PatchRecord]] = {"train": [], "val": [], "test": []}
    for _, class_records in by_class.items():
        items = class_records[:]
        rng.shuffle(items)
        n_total = len(items)
        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio))
        if n_train + n_val >= n_total:
            n_val = max(1, n_total - n_train - 1)
            if n_train + n_val >= n_total:
                n_train = max(1, n_total - 2)
                n_val = 1
        split_records["train"].extend(items[:n_train])
        split_records["val"].extend(items[n_train : n_train + n_val])
        split_records["test"].extend(items[n_train + n_val :])
    return split_records


def to_dataframe(records: Iterable[object]) -> pd.DataFrame:
    return pd.DataFrame([record.__dict__ for record in records])


def prepare_artifact_datasets(
    artifact_root: Path,
    out_root: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    max_samples_main: int | None,
    max_samples_positive: int | None,
    max_samples_negative: int | None,
) -> Dict[str, object]:
    transversal_root = artifact_root / "ARTERY_TRANSVERSAL"
    if not transversal_root.exists():
        raise FileNotFoundError(f"Missing ARTERY_TRANSVERSAL directory: {transversal_root}")

    detection_root = out_root / "artifact_localization"
    patch_root = out_root / "artifact_patch_cls"
    for root in (detection_root, patch_root):
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "splits").mkdir(parents=True, exist_ok=True)

    print("[artifact] Preparing localization dataset...")

    train_subsets = ["Ultrasonix_train"]
    test_subsets = ["Ultrasonix_test", "Toshiba_test"]

    detection_records: List[DetectionRecord] = []
    for subset_name in train_subsets + test_subsets:
        subset_root = transversal_root / subset_name
        image_dir = subset_root / "img"
        txt_dir = subset_root / "txt"
        if not image_dir.exists() or not txt_dir.exists():
            raise FileNotFoundError(f"Missing img/txt directories under: {subset_root}")

        image_files = sorted(path for path in image_dir.iterdir() if path.is_file())
        if max_samples_main is not None:
            image_files = image_files[:max_samples_main]

        for image_file in tqdm(image_files, desc=f"Localization {subset_name}", unit="img"):
            txt_file = txt_dir / f"{image_file.stem}.txt"
            if not txt_file.exists():
                continue
            sample_id = f"artifact_det_{subset_name.lower()}_{image_file.stem}"
            image_rel = Path("images") / f"{sample_id}.png"
            width, height = save_png_grayscale(image_file, detection_root / image_rel)
            cx, cy, bw, bh = parse_bbox_txt(txt_file)
            x1 = max(cx - bw / 2.0, 0.0)
            y1 = max(cy - bh / 2.0, 0.0)
            x2 = min(cx + bw / 2.0, float(width))
            y2 = min(cy + bh / 2.0, float(height))
            detection_records.append(
                DetectionRecord(
                    sample_id=sample_id,
                    patient_id=infer_patient_id(image_file.stem),
                    source="cca_artifact",
                    subset=subset_name.lower(),
                    image_relpath=image_rel.as_posix(),
                    bbox_relpath=txt_file.relative_to(artifact_root).as_posix(),
                    width=width,
                    height=height,
                    bbox_cx=float(cx),
                    bbox_cy=float(cy),
                    bbox_w=float(bw),
                    bbox_h=float(bh),
                    bbox_x1=float(x1),
                    bbox_y1=float(y1),
                    bbox_x2=float(x2),
                    bbox_y2=float(y2),
                )
            )

    print("[artifact] Writing localization splits...")
    detection_train_val = split_by_group(
        [record for record in detection_records if record.subset == "ultrasonix_train"],
        seed=seed,
        train_ratio=train_ratio,
    )
    detection_test = [record for record in detection_records if record.subset in {"ultrasonix_test", "toshiba_test"}]
    detection_splits = {
        "train": detection_train_val["train"],
        "val": detection_train_val["val"],
        "test": detection_test,
    }
    to_dataframe(detection_records).to_csv(detection_root / "metadata_all.csv", index=False)
    for split_name, split_items in detection_splits.items():
        to_dataframe(split_items).to_csv(detection_root / "splits" / f"{split_name}.csv", index=False)

    print("[artifact] Preparing patch classification dataset...")
    patch_records: List[PatchRecord] = []
    patch_sources = [
        ("positive", "real_artery", 1, max_samples_positive),
        ("negative", "artifact_fake", 0, max_samples_negative),
    ]
    train_root = transversal_root / "Ultrasonix_train"
    for folder_name, class_name, class_id, max_samples in patch_sources:
        patch_dir = train_root / folder_name
        if not patch_dir.exists():
            raise FileNotFoundError(f"Missing patch directory: {patch_dir}")
        patch_files = sorted(path for path in patch_dir.iterdir() if path.is_file())
        if max_samples is not None:
            patch_files = patch_files[:max_samples]
        for patch_file in tqdm(patch_files, desc=f"Patch {folder_name}", unit="patch"):
            sample_id = f"artifact_patch_{folder_name}_{patch_file.stem}"
            image_rel = Path("images") / f"{sample_id}.png"
            width, height = save_png_grayscale(patch_file, patch_root / image_rel)
            patch_records.append(
                PatchRecord(
                    sample_id=sample_id,
                    patient_id=patch_file.stem,
                    source="cca_artifact",
                    subset=folder_name,
                    image_relpath=image_rel.as_posix(),
                    class_name=class_name,
                    class_id=class_id,
                    width=width,
                    height=height,
                )
            )

    print("[artifact] Writing patch classification splits...")
    patch_splits = split_patch_records(
        patch_records,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    to_dataframe(patch_records).to_csv(patch_root / "metadata_all.csv", index=False)
    for split_name, split_items in patch_splits.items():
        to_dataframe(split_items).to_csv(patch_root / "splits" / f"{split_name}.csv", index=False)

    detection_summary = {
        "task": "artifact_localization",
        "num_total": len(detection_records),
        "num_train": len(detection_splits["train"]),
        "num_val": len(detection_splits["val"]),
        "num_test": len(detection_splits["test"]),
        "class_mapping": {"real_artery": 1},
        "output_root": str(detection_root),
    }
    patch_summary = {
        "task": "artifact_patch_classification",
        "num_total": len(patch_records),
        "num_train": len(patch_splits["train"]),
        "num_val": len(patch_splits["val"]),
        "num_test": len(patch_splits["test"]),
        "class_mapping": {"artifact_fake": 0, "real_artery": 1},
        "output_root": str(patch_root),
    }
    with (detection_root / "dataset_summary.json").open("w", encoding="utf-8") as file:
        json.dump(detection_summary, file, ensure_ascii=False, indent=2)
    with (patch_root / "dataset_summary.json").open("w", encoding="utf-8") as file:
        json.dump(patch_summary, file, ensure_ascii=False, indent=2)

    top_summary = {
        "selected_datasets": ["cca_artifact"],
        "design": "dual_model_pipeline",
        "artifact_root": str(artifact_root),
        "detection_root": str(detection_root),
        "patch_root": str(patch_root),
        "detection_summary": detection_summary,
        "patch_summary": patch_summary,
    }
    with (out_root / "dataset_summary.json").open("w", encoding="utf-8") as file:
        json.dump(top_summary, file, ensure_ascii=False, indent=2)
    return top_summary
