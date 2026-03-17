from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.signal import correlate2d
from tqdm.auto import tqdm

try:
    from data.prepare_artifact_data import prepare_artifact_datasets
except ModuleNotFoundError:
    from prepare_artifact_data import prepare_artifact_datasets

ARTIFACT_FAKE_MASK_VALUE = 1
ARTIFACT_REAL_MASK_VALUE = 2


@dataclass
class SampleRecord:
    sample_id: str
    patient_id: str
    source: str
    subset: str
    task_type: str
    label_name: str
    label_id: int
    image_relpath: str
    mask_relpath: str
    li_relpath: str
    ma_relpath: str
    width: int
    height: int


@dataclass
class PatchMatchStats:
    positive_total: int = 0
    positive_matched: int = 0
    negative_total: int = 0
    negative_matched: int = 0


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


def save_png_grayscale(src_image: Path, dst_png: Path) -> Tuple[int, int]:
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(src_image).convert("L")
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


def save_mask(mask_array: np.ndarray, dst_png: Path) -> Tuple[int, int]:
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    mask = Image.fromarray(mask_array.astype(np.uint8), mode="L")
    width, height = mask.size
    mask.save(dst_png, format="PNG")
    return width, height


def make_rect_mask(
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    box_width: float,
    box_height: float,
    fill_value: int,
) -> Image.Image:
    mask = Image.new("L", (width, height), color=0)
    if box_width <= 0 or box_height <= 0:
        return mask

    left = max(int(round(center_x - box_width / 2.0)), 0)
    top = max(int(round(center_y - box_height / 2.0)), 0)
    right = min(int(round(center_x + box_width / 2.0)), width - 1)
    bottom = min(int(round(center_y + box_height / 2.0)), height - 1)
    if right < left or bottom < top:
        return mask

    drawer = ImageDraw.Draw(mask)
    drawer.rectangle([left, top, right, bottom], fill=int(fill_value))
    return mask


def parse_bbox_txt(path: Path) -> Tuple[float, float, float, float]:
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [item for item in re.split(r"[;,\s]+", line) if item]
            if len(parts) < 4:
                continue
            return tuple(float(item) for item in parts[:4])  # type: ignore[return-value]
    raise ValueError(f"Unable to parse bbox from: {path}")


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


def load_grayscale_array(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def resize_array(array: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return array
    width = max(1, int(round(array.shape[1] / factor)))
    height = max(1, int(round(array.shape[0] / factor)))
    resized = Image.fromarray(array, mode="L").resize((width, height), resample=Image.BILINEAR)
    return np.array(resized, dtype=np.uint8)


def best_template_match(image: np.ndarray, patch: np.ndarray, downsample_factor: int = 4) -> Tuple[int, int, float]:
    img_small = resize_array(image, downsample_factor).astype(np.float32)
    patch_small = resize_array(patch, downsample_factor).astype(np.float32)
    patch_small = patch_small - patch_small.mean()
    patch_std = float(patch_small.std())
    if patch_std < 1e-6:
        patch_small = patch_small * 0.0
    else:
        patch_small = patch_small / patch_std
    img_small = img_small - img_small.mean()
    img_std = float(img_small.std())
    if img_std >= 1e-6:
        img_small = img_small / img_std

    response = correlate2d(img_small, patch_small[::-1, ::-1], mode="valid", boundary="fill", fillvalue=0.0)
    y_small, x_small = np.unravel_index(np.argmax(response), response.shape)

    x0 = int(x_small * downsample_factor)
    y0 = int(y_small * downsample_factor)
    x0 = min(max(x0, 0), max(image.shape[1] - patch.shape[1], 0))
    y0 = min(max(y0, 0), max(image.shape[0] - patch.shape[0], 0))

    search_radius = max(2, downsample_factor * 2)
    best_score = float("inf")
    best_xy = (x0, y0)
    for y in range(max(0, y0 - search_radius), min(image.shape[0] - patch.shape[0], y0 + search_radius) + 1):
        for x in range(max(0, x0 - search_radius), min(image.shape[1] - patch.shape[1], x0 + search_radius) + 1):
            crop = image[y : y + patch.shape[0], x : x + patch.shape[1]].astype(np.float32)
            mse = float(np.mean((crop - patch.astype(np.float32)) ** 2))
            if mse < best_score:
                best_score = mse
                best_xy = (x, y)
    return best_xy[0], best_xy[1], best_score


def overlay_patch_mask(
    mask_array: np.ndarray,
    patch: np.ndarray,
    top_left_x: int,
    top_left_y: int,
    label_value: int,
    min_intensity: int = 20,
) -> None:
    patch_h, patch_w = patch.shape
    region = mask_array[top_left_y : top_left_y + patch_h, top_left_x : top_left_x + patch_w]
    patch_region = patch > min_intensity
    if label_value == ARTIFACT_FAKE_MASK_VALUE:
        region[(patch_region) & (region == 0)] = label_value
    else:
        region[patch_region] = label_value


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

    for image_file in tqdm(image_files, desc="CUBS v1", unit="img"):
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
                subset=manual_source,
                task_type="segmentation",
                label_name="imt",
                label_id=1,
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

    for image_file in tqdm(image_files, desc="CCA masks", unit="img"):
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
                subset="expert_mask",
                task_type="segmentation",
                label_name="cca_mask",
                label_id=1,
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

    for image_file in tqdm(image_files, desc="CUBS v2", unit="img"):
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
                subset=manual_source,
                task_type="segmentation",
                label_name="imt",
                label_id=1,
                image_relpath=image_rel.as_posix(),
                mask_relpath=mask_rel.as_posix(),
                li_relpath=li_file.relative_to(cubs_v2_root).as_posix(),
                ma_relpath=ma_file.relative_to(cubs_v2_root).as_posix(),
                width=width,
                height=height,
            )
        )
    return records


def collect_cca_artifact_bbox_subset(
    subset_root: Path,
    out_root: Path,
    subset_name: str,
    max_samples: Optional[int],
    patch_overlays: Optional[Dict[str, List[Tuple[np.ndarray, int, int, int]]]] = None,
) -> List[SampleRecord]:
    image_dir = subset_root / "img"
    txt_dir = subset_root / "txt"
    if not image_dir.exists() or not txt_dir.exists():
        raise FileNotFoundError(f"Missing img/txt directories under {subset_root}")

    image_files = sorted(path for path in image_dir.iterdir() if path.is_file())
    if max_samples is not None:
        image_files = image_files[:max_samples]

    records: List[SampleRecord] = []
    for image_file in tqdm(image_files, desc=f"{subset_name} images", unit="img"):
        txt_file = txt_dir / f"{image_file.stem}.txt"
        if not txt_file.exists():
            continue

        sample_id = f"cca_artifact_{subset_name}_{image_file.stem}"
        patient_id = infer_patient_id(image_file.stem)
        image_rel = Path("images") / f"{sample_id}.png"
        mask_rel = Path("masks") / f"{sample_id}.png"

        width, height = save_png_grayscale(image_file, out_root / image_rel)
        center_x, center_y, box_width, box_height = parse_bbox_txt(txt_file)
        mask = make_rect_mask(width, height, center_x, center_y, box_width, box_height, fill_value=ARTIFACT_REAL_MASK_VALUE)
        mask_np = np.array(mask, dtype=np.uint8)
        for patch_array, label_value, x, y in patch_overlays.get(image_file.name, []) if patch_overlays else []:
            overlay_patch_mask(mask_np, patch_array, x, y, label_value)
        save_mask(mask_np, out_root / mask_rel)

        records.append(
            SampleRecord(
                sample_id=sample_id,
                patient_id=patient_id,
                source="cca_artifact",
                subset=subset_name,
                task_type="artifact_segmentation",
                label_name="real_artery",
                label_id=2,
                image_relpath=image_rel.as_posix(),
                mask_relpath=mask_rel.as_posix(),
                li_relpath=txt_file.relative_to(subset_root.parent.parent).as_posix(),
                ma_relpath="",
                width=width,
                height=height,
            )
        )
    return records


def build_ultrasonix_patch_overlays(
    train_root: Path,
    max_samples_positive: Optional[int],
    max_samples_negative: Optional[int],
) -> Tuple[Dict[str, List[Tuple[np.ndarray, int, int, int]]], PatchMatchStats]:
    image_dir = train_root / "img"
    positive_dir = train_root / "positive"
    negative_dir = train_root / "negative"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not positive_dir.exists() or not negative_dir.exists():
        raise FileNotFoundError(f"Missing positive/negative directories under: {train_root}")

    image_arrays: Dict[str, np.ndarray] = {
        path.name: load_grayscale_array(path) for path in sorted(image_dir.glob("*.png"))
    }
    overlays: Dict[str, List[Tuple[np.ndarray, int, int, int]]] = {name: [] for name in image_arrays}
    stats = PatchMatchStats()

    def assign_patches(patch_dir: Path, max_samples: Optional[int], label_value: int, label_name: str) -> None:
        patch_files = sorted(path for path in patch_dir.iterdir() if path.is_file())
        if max_samples is not None:
            patch_files = patch_files[:max_samples]
        if label_name == "positive":
            stats.positive_total = len(patch_files)
        else:
            stats.negative_total = len(patch_files)

        for patch_file in tqdm(patch_files, desc=f"Match {label_name} patches", unit="patch"):
            patch_array = load_grayscale_array(patch_file)
            best_image_name = None
            best_score = float("inf")
            best_xy = (0, 0)
            for image_name, image_array in image_arrays.items():
                x, y, score = best_template_match(image_array, patch_array)
                if score < best_score:
                    best_score = score
                    best_image_name = image_name
                    best_xy = (x, y)
            if best_image_name is None:
                continue
            overlays.setdefault(best_image_name, []).append((patch_array, label_value, best_xy[0], best_xy[1]))
            if label_name == "positive":
                stats.positive_matched += 1
            else:
                stats.negative_matched += 1

    assign_patches(positive_dir, max_samples_positive, ARTIFACT_REAL_MASK_VALUE, "positive")
    assign_patches(negative_dir, max_samples_negative, ARTIFACT_FAKE_MASK_VALUE, "negative")
    return overlays, stats


def collect_cca_artifact(
    cca_artifact_root: Path,
    out_root: Path,
    max_samples_main: Optional[int],
    max_samples_positive: Optional[int],
    max_samples_negative: Optional[int],
) -> Tuple[List[SampleRecord], PatchMatchStats]:
    transversal_root = cca_artifact_root / "ARTERY_TRANSVERSAL"
    if not transversal_root.exists():
        raise FileNotFoundError(f"Missing ARTERY_TRANSVERSAL directory: {transversal_root}")

    print("[cca_artifact] Building patch overlays from Ultrasonix_train/positive and negative ...")
    train_patch_overlays, patch_stats = build_ultrasonix_patch_overlays(
        transversal_root / "Ultrasonix_train",
        max_samples_positive=max_samples_positive,
        max_samples_negative=max_samples_negative,
    )
    print(
        "[cca_artifact] Patch matching done: "
        f"positive {patch_stats.positive_matched}/{patch_stats.positive_total}, "
        f"negative {patch_stats.negative_matched}/{patch_stats.negative_total}"
    )

    records: List[SampleRecord] = []
    records.extend(
        collect_cca_artifact_bbox_subset(
            subset_root=transversal_root / "Toshiba_test",
            out_root=out_root,
            subset_name="toshiba_test",
            max_samples=max_samples_main,
        )
    )
    records.extend(
        collect_cca_artifact_bbox_subset(
            subset_root=transversal_root / "Ultrasonix_test",
            out_root=out_root,
            subset_name="ultrasonix_test",
            max_samples=max_samples_main,
        )
    )
    records.extend(
        collect_cca_artifact_bbox_subset(
            subset_root=transversal_root / "Ultrasonix_train",
            out_root=out_root,
            subset_name="ultrasonix_train",
            max_samples=max_samples_main,
            patch_overlays=train_patch_overlays,
        )
    )
    return records, patch_stats


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
    default_cca_artifact = "data/extracted/Common Carotid Artery Data Set"
    parser = argparse.ArgumentParser(
        description="Prepare unified segmentation dataset from selected sources."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="cubs_v1,cubs_v2",
        help="Comma-separated list from {cubs_v1,cubs_v2,cca,cca_artifact}.",
    )
    parser.add_argument("--cubs_v1_root", type=Path, default=Path(default_v1))
    parser.add_argument("--cubs_v2_root", type=Path, default=Path(default_v2))
    parser.add_argument("--cca_root", type=Path, default=Path(default_cca))
    parser.add_argument("--cca_artifact_root", type=Path, default=Path(default_cca_artifact))
    parser.add_argument("--out_root", type=Path, default=Path("data/processed/unified_dataset"))
    parser.add_argument("--v1_manual_source", type=str, default="Manual-A1")
    parser.add_argument("--v2_manual_source", type=str, default="Manual-A1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_samples_v1", type=int, default=None)
    parser.add_argument("--max_samples_v2", type=int, default=None)
    parser.add_argument("--max_samples_cca", type=int, default=None)
    parser.add_argument("--max_samples_cca_artifact_main", type=int, default=None)
    parser.add_argument("--max_samples_cca_artifact_positive", type=int, default=None)
    parser.add_argument("--max_samples_cca_artifact_negative", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [item.strip().lower() for item in args.datasets.split(",") if item.strip()]
    valid = {"cubs_v1", "cubs_v2", "cca", "cca_artifact"}
    invalid = [item for item in selected if item not in valid]
    if invalid:
        raise ValueError(f"Unsupported datasets {invalid}. Valid options: {sorted(valid)}")
    if not selected:
        raise ValueError("No dataset selected. Use --datasets with at least one item.")
    if "cca_artifact" in selected and len(selected) > 1:
        raise ValueError(
            "cca_artifact now prepares a dual-model pipeline (localization + patch classification). "
            "Please run it into a separate output root instead of mixing with segmentation datasets."
        )
    if selected == ["cca_artifact"]:
        summary = prepare_artifact_datasets(
            artifact_root=args.cca_artifact_root,
            out_root=args.out_root,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            max_samples_main=args.max_samples_cca_artifact_main,
            max_samples_positive=args.max_samples_cca_artifact_positive,
            max_samples_negative=args.max_samples_cca_artifact_negative,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Saved artifact localization dataset to: {args.out_root / 'artifact_localization'}")
        print(f"Saved artifact patch dataset to: {args.out_root / 'artifact_patch_cls'}")
        return

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "masks").mkdir(parents=True, exist_ok=True)
    (out_root / "splits").mkdir(parents=True, exist_ok=True)

    v1_records: List[SampleRecord] = []
    v2_records: List[SampleRecord] = []
    cca_records: List[SampleRecord] = []
    cca_artifact_records: List[SampleRecord] = []
    cca_artifact_patch_stats = PatchMatchStats()

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
    if "cca_artifact" in selected:
        cca_artifact_records, cca_artifact_patch_stats = collect_cca_artifact(
            cca_artifact_root=args.cca_artifact_root,
            out_root=out_root,
            max_samples_main=args.max_samples_cca_artifact_main,
            max_samples_positive=args.max_samples_cca_artifact_positive,
            max_samples_negative=args.max_samples_cca_artifact_negative,
        )

    all_records = v1_records + v2_records + cca_records + cca_artifact_records
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

    class_mapping = {"background": 0, "foreground": 1}
    if "cca_artifact" in selected:
        class_mapping = {"background": 0, "artifact_fake": 1, "real_artery": 2}

    summary = {
        "selected_datasets": selected,
        "num_total": len(all_records),
        "num_v1": len(v1_records),
        "num_v2": len(v2_records),
        "num_cca": len(cca_records),
        "num_cca_artifact": len(cca_artifact_records),
        "num_train": len(split_records["train"]),
        "num_val": len(split_records["val"]),
        "num_test": len(split_records["test"]),
        "v1_root": str(args.cubs_v1_root),
        "v2_root": str(args.cubs_v2_root),
        "cca_root": str(args.cca_root),
        "cca_artifact_root": str(args.cca_artifact_root),
        "num_classes": len(class_mapping),
        "class_mapping": class_mapping,
        "cca_artifact_patch_stats": {
            "positive_total": cca_artifact_patch_stats.positive_total,
            "positive_matched": cca_artifact_patch_stats.positive_matched,
            "negative_total": cca_artifact_patch_stats.negative_total,
            "negative_matched": cca_artifact_patch_stats.negative_matched,
        },
        "output_root": str(out_root),
    }
    with (out_root / "dataset_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved metadata to: {out_root / 'metadata_all.csv'}")
    print(f"Saved splits to: {out_root / 'splits'}")


if __name__ == "__main__":
    main()
