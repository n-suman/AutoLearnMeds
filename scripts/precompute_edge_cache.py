#!/usr/bin/env python3
"""Pre-compute per-patch Sobel edge density for an image directory.

Used by Phase 7b text-aware MAE to skip the cache-build step at training
start. Writes a single (N, num_patches) float32 numpy memmap file so the
MAE dataset can mmap it.

Usage:
    uv run python scripts/precompute_edge_cache.py \\
        --images-root /content/raw_images_local \\
        --out checkpoints/mae/text-aware-seed44/edge_cache.npy \\
        --image-size 224 --patch-size 16 \\
        [--exclude-jsonls data/processed/val.jsonl data/processed/test.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--exclude-jsonls", nargs="*", default=[],
                        help="JSONL files whose image_path basenames are excluded")
    parser.add_argument("--include-jsonls", nargs="*", default=[],
                        help="JSONL files whose image_path basenames are included (inverse of exclude)")
    args = parser.parse_args()

    # Heavy imports inside main (matches project convention).
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pretrain_mae import compute_edge_density_per_patch

    # Discover images.
    root = args.images_root
    paths = sorted(p for p in root.glob("**/*") if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
    print(f"[edge_cache] discovered {len(paths)} images under {root}", flush=True)

    # Apply include / exclude filters (mirror build_pretrain_dataset semantics).
    if args.include_jsonls:
        included: set[str] = set()
        for jp in args.include_jsonls:
            for line in Path(jp).read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                img = entry.get("image_path") or entry.get("image") or ""
                if img:
                    included.add(Path(img).name)
        before = len(paths)
        paths = [p for p in paths if p.name in included]
        print(f"[edge_cache] include filter: {before} -> {len(paths)}", flush=True)

    if args.exclude_jsonls:
        excluded: set[str] = set()
        for jp in args.exclude_jsonls:
            for line in Path(jp).read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                img = entry.get("image_path") or entry.get("image") or ""
                if img:
                    excluded.add(Path(img).name)
        before = len(paths)
        paths = [p for p in paths if p.name not in excluded]
        print(f"[edge_cache] exclude filter: {before} -> {len(paths)}", flush=True)

    if not paths:
        print("[edge_cache] FATAL: 0 images after filtering", file=sys.stderr)
        return 2

    num_patches = (args.image_size // args.patch_size) ** 2
    args.out.parent.mkdir(parents=True, exist_ok=True)

    cache = np.memmap(args.out, dtype=np.float32, mode="w+", shape=(len(paths), num_patches))

    start = time.time()
    for i, p in enumerate(paths):
        cache[i] = compute_edge_density_per_patch(p, args.image_size, args.patch_size)
        if (i + 1) % 200 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(paths) - i - 1) / rate
            print(f"[edge_cache] {i+1}/{len(paths)} elapsed={elapsed:.0f}s rate={rate:.1f}/s eta={eta:.0f}s", flush=True)

    cache.flush()
    del cache  # close memmap

    # Also write a paths.json sidecar so the consumer can verify image-order alignment.
    paths_json = args.out.with_suffix(".paths.json")
    paths_json.write_text(json.dumps([str(p) for p in paths], indent=1))

    elapsed = time.time() - start
    print(f"[edge_cache] DONE wrote {args.out} ({len(paths)} x {num_patches} float32) "
          f"+ paths.json sidecar in {elapsed:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
