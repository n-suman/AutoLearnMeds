#!/usr/bin/env python3
"""One-time sync of user's Drive data folders to the canonical GCS bucket.

Architecture: Drive is user-facing source of truth (where the user uploads).
GCS is the canonical persistent home that Colab and remote workers read from.
This script bridges the two: it pulls each named Drive folder by ID and
mirrors it into a stable subpath under the GCS bucket. Idempotent — skips
folders whose target subpath already contains objects, unless --force.

Required (env vars or flags):
  --bucket | AUTOLEARNMEDS_GCS_BUCKET            gs://bucket-name
  --golden-set-id | AUTOLEARNMEDS_DRIVE_GOLDEN_SET_ID    Drive folder ID for label JSON
  --raw-images-id | AUTOLEARNMEDS_DRIVE_RAW_IMAGES_ID    Drive folder ID for images

Usage:
  python scripts/sync_drive_to_gcs.py                # idempotent
  python scripts/sync_drive_to_gcs.py --force        # force re-sync everything
  python scripts/sync_drive_to_gcs.py --skip-images  # only sync golden_set
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# GCS subpath layout — kept stable so Phase-1 build_processed.py can rely on it.
GOLDEN_SET_SUBPATH = "raw/golden_set"
RAW_IMAGES_SUBPATH = "raw/images"


def gcs_path_has_objects(bucket: str, subpath: str) -> bool:
    """Return True iff gs://bucket/subpath/ contains at least one object."""
    target = f"{bucket.rstrip('/')}/{subpath.strip('/')}/"
    result = subprocess.run(
        ["gsutil", "ls", target],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def sync_one(folder_id: str, bucket: str, subpath: str, force: bool, tmp_root: Path) -> None:
    """Download one Drive folder by ID, then upload to gs://bucket/subpath."""
    target = f"{bucket.rstrip('/')}/{subpath.strip('/')}"
    print(f"[sync] {folder_id} -> {target}")

    if not force and gcs_path_has_objects(bucket, subpath):
        print("[sync]   already populated, skipping (use --force to re-sync)")
        return

    try:
        import gdown
    except ImportError:
        print(
            "[sync] FATAL: gdown not installed. On Colab, run: uv sync --extra colab",
            file=sys.stderr,
        )
        sys.exit(1)

    local = tmp_root / subpath.replace("/", "_")
    local.mkdir(parents=True, exist_ok=True)
    print(f"[sync]   downloading from Drive folder {folder_id} to {local} ...")
    try:
        result = gdown.download_folder(
            id=folder_id,
            output=str(local),
            quiet=False,
            use_cookies=False,
        )
    except Exception as e:
        print(
            f"[sync]   FAIL: gdown raised {type(e).__name__}: {e}\n"
            "[sync]   If folder is large or shared, try mounting Drive and "
            "using gsutil cp directly from /content/drive/MyDrive/<folder>.",
            file=sys.stderr,
        )
        sys.exit(2)
    if not result:
        print(
            "[sync]   FAIL: gdown returned no files. Possible causes: "
            "Drive API quota, the folder is shared (not in MyDrive), or the ID is wrong.",
            file=sys.stderr,
        )
        sys.exit(3)

    n = sum(1 for p in local.rglob("*") if p.is_file())
    print(f"[sync]   downloaded {n} files locally")

    print(f"[sync]   uploading to {target} ...")
    cp = subprocess.run(["gsutil", "-m", "cp", "-r", f"{local}/.", f"{target}/"])
    if cp.returncode != 0:
        print(f"[sync]   FAIL: gsutil cp exit {cp.returncode}", file=sys.stderr)
        sys.exit(4)
    print(f"[sync]   uploaded {n} files to {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--bucket",
        default=os.environ.get("AUTOLEARNMEDS_GCS_BUCKET", ""),
        help="gs://bucket-name (default: AUTOLEARNMEDS_GCS_BUCKET env var)",
    )
    parser.add_argument(
        "--golden-set-id",
        default=os.environ.get("AUTOLEARNMEDS_DRIVE_GOLDEN_SET_ID", ""),
        help="Drive folder ID for label JSON (default: AUTOLEARNMEDS_DRIVE_GOLDEN_SET_ID env var)",
    )
    parser.add_argument(
        "--raw-images-id",
        default=os.environ.get("AUTOLEARNMEDS_DRIVE_RAW_IMAGES_ID", ""),
        help="Drive folder ID for images (default: AUTOLEARNMEDS_DRIVE_RAW_IMAGES_ID env var)",
    )
    parser.add_argument("--force", action="store_true", help="re-sync even if GCS subpath is populated")
    parser.add_argument("--skip-golden-set", action="store_true")
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument(
        "--tmp-root",
        default=Path("/tmp/autolearnmeds-drive-sync"),
        type=Path,
        help="local staging dir (default: /tmp/autolearnmeds-drive-sync)",
    )
    args = parser.parse_args()

    if not args.bucket:
        print(
            "[sync] FATAL: --bucket / AUTOLEARNMEDS_GCS_BUCKET is required (e.g. gs://my-bucket)",
            file=sys.stderr,
        )
        return 1

    args.tmp_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_golden_set:
        if not args.golden_set_id:
            print("[sync] golden-set folder ID not set, skipping", file=sys.stderr)
        else:
            sync_one(
                args.golden_set_id, args.bucket, GOLDEN_SET_SUBPATH, args.force, args.tmp_root
            )

    if not args.skip_images:
        if not args.raw_images_id:
            print("[sync] raw-images folder ID not set, skipping", file=sys.stderr)
        else:
            sync_one(
                args.raw_images_id, args.bucket, RAW_IMAGES_SUBPATH, args.force, args.tmp_root
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
