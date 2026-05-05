#!/usr/bin/env python3
"""Convert golden_set + splits.json + raw images -> data/processed/{train,val,test}.jsonl.

Adds per-record:
  image_id    — IMG_5246 (filename without extension)
  image_path  — relative to bucket root, e.g., raw/raw_images/IMG_5246.jpeg
  image_hash  — sha256:... of the image bytes (for test-set firewall)
  split       — train | val | test

Idempotent: re-running with the same inputs writes byte-identical outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def compute_image_hash(image_path: Path, chunk_size: int = 1 << 16) -> str:
    """Return sha256:<hex64> of the image file bytes."""
    h = hashlib.sha256()
    with image_path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _strip_extension(filename: str) -> str:
    return Path(filename).stem


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_processed(
    gold_standard_path: Path,
    splits_path: Path,
    images_root: Path,
    out_dir: Path,
    image_path_prefix: str = "raw/raw_images",
) -> dict[str, int]:
    """Build train/val/test JSONL files. Returns counts per split."""
    gold_standard_path = Path(gold_standard_path)
    splits_path = Path(splits_path)
    images_root = Path(images_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl(gold_standard_path)
    by_image = {r["image_file"]: r for r in records}
    splits = json.loads(splits_path.read_text())

    counts = {}
    for split_name in ("train", "val", "test"):
        out_lines: list[str] = []
        for image_file in splits.get(split_name, []):
            if image_file not in by_image:
                print(
                    f"[build_processed] WARN: {image_file} in splits but not in gold_standard; skipping",
                    file=sys.stderr,
                )
                continue
            img_path = images_root / image_file
            if not img_path.is_file():
                print(
                    f"[build_processed] WARN: {image_file} on disk missing at {img_path}; skipping",
                    file=sys.stderr,
                )
                continue
            base = by_image[image_file]
            rec = {
                "image_id": _strip_extension(image_file),
                "image_file": image_file,
                "image_path": f"{image_path_prefix.rstrip('/')}/{image_file}",
                "image_hash": compute_image_hash(img_path),
                "split": split_name,
                **base,
            }
            # Sort-keyed JSON dump => deterministic (idempotent).
            out_lines.append(json.dumps(rec, sort_keys=True, ensure_ascii=False))
        out_path = out_dir / f"{split_name}.jsonl"
        out_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
        counts[split_name] = len(out_lines)
        print(f"[build_processed] wrote {out_path} ({counts[split_name]} records)")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gold-standard", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--image-path-prefix",
        default="raw/raw_images",
        help="Prefix written into each record's image_path field",
    )
    args = parser.parse_args()
    build_processed(
        gold_standard_path=args.gold_standard,
        splits_path=args.splits,
        images_root=args.images_root,
        out_dir=args.out_dir,
        image_path_prefix=args.image_path_prefix,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
