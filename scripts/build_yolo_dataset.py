#!/usr/bin/env python3
"""Convert gold_standard.jsonl polygon annotations -> YOLOv12-format dataset.

YOLOv12 (ultralytics) trains from a directory layout where each image has a
sibling .txt file with one bounding box per line:

    class_id center_x center_y width height

(all four coords normalized to [0, 1]).

Our gold_standard.jsonl stores POLYGONS with coords in [0, 100] percentages.
This script:

  1. Reads gold_standard.jsonl + splits.json.
  2. For each field's polygon, computes the axis-aligned bounding rectangle
     (min_x, min_y, max_x, max_y), normalizes by /100, converts to YOLO
     (cx, cy, w, h) format.
  3. Maps field name -> integer class id (brand_name=0 ... warnings=11).
  4. Writes labels to {out_dir}/labels/{split}/IMG_NNNN.txt and symlinks
     images to {out_dir}/images/{split}/IMG_NNNN.jpeg.
  5. Generates {out_dir}/data.yaml — the ultralytics dataset descriptor.

Skips images with zero valid polygons (avoids the empty-label-file error
from ultralytics).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import yaml

# Class-id ordering. Must stay locked across all splits + train+eval.
FIELD_NAMES: list[str] = [
    "brand_name",
    "drug_name",
    "generic_name",
    "strength",
    "quantity",
    "company",
    "manufacturer",
    "batch_number",
    "mfg_date",
    "expiry_date",
    "mrp",
    "warnings",
]

FIELD_TO_CLASS_ID: dict[str, int] = {name: i for i, name in enumerate(FIELD_NAMES)}


def polygon_to_yolo_bbox(polygon: Iterable[Iterable[float]]) -> tuple[float, float, float, float]:
    """Convert a polygon in [0, 100] percentages to YOLO bbox (cx, cy, w, h) in [0, 1].

    Computes the axis-aligned bounding rectangle of the polygon, normalizes by /100,
    and emits center_x, center_y, width, height — all in [0, 1].
    """
    pts = list(polygon)
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    # Polygon coords are percentages of image dims, so /100 is the normalization.
    min_x, max_x = min_x / 100.0, max_x / 100.0
    min_y, max_y = min_y / 100.0, max_y / 100.0
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    w = max_x - min_x
    h = max_y - min_y
    return cx, cy, w, h


def _is_valid_polygon(polygon) -> bool:
    """A polygon is usable if it's a non-empty list with >= 3 (x, y) pairs."""
    if not polygon:
        return False
    if not isinstance(polygon, list):
        return False
    if len(polygon) < 3:
        return False
    for p in polygon:
        if not (isinstance(p, (list, tuple)) and len(p) >= 2):
            return False
    return True


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _build_label_lines(fields: dict) -> list[str]:
    """Return YOLO-format label lines for one image, skipping invalid polygons.

    Unknown field names (not in FIELD_TO_CLASS_ID) are also skipped.
    """
    lines: list[str] = []
    for field_name, payload in fields.items():
        if field_name not in FIELD_TO_CLASS_ID:
            continue
        if not isinstance(payload, dict):
            continue
        polygon = payload.get("polygon")
        if not _is_valid_polygon(polygon):
            continue
        cls_id = FIELD_TO_CLASS_ID[field_name]
        cx, cy, w, h = polygon_to_yolo_bbox(polygon)
        # Clamp into [0, 1] just in case of floating-point overshoot at the boundary.
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)
        if w <= 0.0 or h <= 0.0:
            continue
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def _symlink_force(src: Path, dst: Path) -> None:
    """Create dst -> src symlink, replacing any existing dst."""
    if dst.is_symlink() or dst.exists():
        try:
            dst.unlink()
        except IsADirectoryError:
            # Defensive: shouldn't happen for image leaf paths.
            pass
    dst.symlink_to(src)


def build_yolo_dataset(
    gold_standard_path: Path,
    splits_path: Path,
    images_root: Path,
    out_dir: Path,
) -> dict[str, int]:
    """Build the YOLO-format dataset directory. Returns per-split image counts."""
    gold_standard_path = Path(gold_standard_path)
    splits_path = Path(splits_path)
    images_root = Path(images_root)
    out_dir = Path(out_dir).resolve()

    records = _read_jsonl(gold_standard_path)
    by_image = {r["image_file"]: r for r in records}
    splits = json.loads(splits_path.read_text())

    counts: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        images_dir = out_dir / "images" / split_name
        labels_dir = out_dir / "labels" / split_name
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        kept = 0
        for image_file in splits.get(split_name, []):
            if image_file not in by_image:
                print(
                    f"[build_yolo_dataset] WARN: {image_file} in splits but not in gold_standard; skipping",
                    file=sys.stderr,
                )
                continue
            src_img = images_root / image_file
            if not src_img.is_file():
                print(
                    f"[build_yolo_dataset] WARN: {image_file} on disk missing at {src_img}; skipping",
                    file=sys.stderr,
                )
                continue
            fields = by_image[image_file].get("fields", {}) or {}
            label_lines = _build_label_lines(fields)
            if not label_lines:
                print(
                    f"[build_yolo_dataset] WARN: {image_file} has zero valid polygons; skipping",
                    file=sys.stderr,
                )
                continue

            stem = Path(image_file).stem
            ext = Path(image_file).suffix or ".jpeg"
            dst_img = images_dir / f"{stem}{ext}"
            dst_lbl = labels_dir / f"{stem}.txt"

            _symlink_force(src_img.resolve(), dst_img)
            dst_lbl.write_text("\n".join(label_lines) + "\n")
            kept += 1

        counts[split_name] = kept
        print(f"[build_yolo_dataset] split={split_name} kept={kept} images")

    # Write data.yaml for ultralytics.
    data_yaml = {
        "path": str(out_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(FIELD_NAMES),
        "names": list(FIELD_NAMES),
    }
    yaml_path = out_dir / "data.yaml"
    with yaml_path.open("w") as fh:
        yaml.safe_dump(data_yaml, fh, sort_keys=False)
    print(f"[build_yolo_dataset] wrote {yaml_path}")

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gold-standard", type=Path, required=True,
                        help="Path to gold_standard.jsonl")
    parser.add_argument("--splits", type=Path, required=True,
                        help="Path to splits.json with {train,val,test} image-filename lists")
    parser.add_argument("--images-root", type=Path, required=True,
                        help="Directory holding the raw images (e.g. raw/raw_images)")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory, e.g. data/yolo")
    args = parser.parse_args()

    build_yolo_dataset(
        gold_standard_path=args.gold_standard,
        splits_path=args.splits,
        images_root=args.images_root,
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
