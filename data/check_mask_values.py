from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect unique values inside mask PNG files.")
    parser.add_argument(
        "--mask_dir",
        type=Path,
        default=Path("data/processed/cca_artifact_dataset/masks"),
        help="Directory containing mask PNG files.",
    )
    parser.add_argument(
        "--decode_artifact_mask",
        action="store_true",
        help="Deprecated compatibility flag. Current artifact masks are already stored as 0/1/2.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="How many example files to print.",
    )
    return parser.parse_args()


def decode_mask(mask: np.ndarray) -> np.ndarray:
    return np.clip(mask, 0, 2).astype(np.uint8)


def main() -> None:
    args = parse_args()
    mask_dir = args.mask_dir
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    mask_files = sorted(mask_dir.glob("*.png"))
    if not mask_files:
        raise FileNotFoundError(f"No PNG masks found under: {mask_dir}")

    pixel_counts: dict[int, int] = {}
    file_counts: dict[int, int] = {}
    file_examples: list[dict[str, object]] = []
    all_values: set[int] = set()

    for mask_file in mask_files:
        mask = np.array(Image.open(mask_file).convert("L"), dtype=np.uint8)
        if args.decode_artifact_mask:
            mask = decode_mask(mask)

        unique_vals, counts = np.unique(mask, return_counts=True)
        unique_list = [int(v) for v in unique_vals.tolist()]
        all_values.update(unique_list)

        for value, count in zip(unique_list, counts.tolist()):
            pixel_counts[value] = pixel_counts.get(value, 0) + int(count)
            file_counts[value] = file_counts.get(value, 0) + 1

        if len(file_examples) < args.limit:
            file_examples.append(
                {
                    "file": mask_file.name,
                    "unique_values": unique_list,
                }
            )

    summary = {
        "mask_dir": str(mask_dir),
        "num_masks": len(mask_files),
        "all_unique_values": sorted(all_values),
        "contains_only_012": sorted(all_values) in ([0], [0, 1], [0, 2], [0, 1, 2], [1], [2], [1, 2]),
        "has_value_2": 2 in all_values,
        "pixel_counts": {str(k): v for k, v in sorted(pixel_counts.items())},
        "file_counts_with_value": {str(k): v for k, v in sorted(file_counts.items())},
        "example_files": file_examples,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
