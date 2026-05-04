#!/usr/bin/env python3
"""One-time sync of user's Drive data folders to the canonical GCS bucket.

Architecture: Drive is user-facing source of truth (where the user uploads).
GCS is the canonical persistent home that Colab and remote workers read from.
This script bridges the two: it pulls each named Drive folder by ID and
mirrors it into a stable subpath under the GCS bucket. Idempotent — skips
folders whose target subpath already contains objects, unless --force.

Two strategies, tried in order:
  1. Drive mount (fastest, works for folders in the user's MyDrive when the
     calling notebook has run google.colab.auth.authenticate_user() and
     google.colab.drive.mount). Resolves folder ID -> name via the Drive
     API, locates the folder under /content/drive/MyDrive/, then copies
     directly with gsutil.
  2. gdown fallback (works only for shared "Anyone with the link" folders).

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

DRIVE_MOUNT_ROOT = Path("/content/drive/MyDrive")


def gcs_path_has_objects(bucket: str, subpath: str) -> bool:
    """Return True iff gs://bucket/subpath/ contains at least one object."""
    target = f"{bucket.rstrip('/')}/{subpath.strip('/')}/"
    result = subprocess.run(
        ["gsutil", "ls", target],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def find_folder_on_mount(folder_id: str) -> Path | None:
    """Resolve a Drive folder ID to a local path under the mounted Drive.

    Uses the Drive API (with whatever ADC the calling environment has, set up
    by google.colab.auth.authenticate_user() in the notebook) to look up the
    folder's name and parent chain, then walks /content/drive/MyDrive/ for
    the matching path. Returns None if the API call fails or the resolved
    path does not exist on disk.
    """
    if not DRIVE_MOUNT_ROOT.exists():
        return None
    try:
        import google.auth
        from googleapiclient.discovery import build
    except ImportError as e:
        print(f"[sync]   Drive API libs not available ({e}); falling back to gdown.", file=sys.stderr)
        return None

    try:
        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception as e:
        print(f"[sync]   Drive API auth failed: {e}", file=sys.stderr)
        return None

    try:
        meta = service.files().get(fileId=folder_id, fields="name,parents").execute()
    except Exception as e:
        print(f"[sync]   Drive API folder lookup failed for {folder_id}: {e}", file=sys.stderr)
        return None

    name = meta.get("name")
    if not name:
        return None

    # Simple case first: folder is directly in MyDrive root.
    candidate = DRIVE_MOUNT_ROOT / name
    if candidate.is_dir():
        return candidate

    # Walk up the parent chain to build the full path.
    path_parts: list[str] = [name]
    parents = meta.get("parents") or []
    parent_id = parents[0] if parents else None
    visited = {folder_id}
    while parent_id and parent_id not in visited:
        visited.add(parent_id)
        try:
            pmeta = service.files().get(fileId=parent_id, fields="name,parents").execute()
        except Exception:
            break
        pname = pmeta.get("name", "")
        if pname == "My Drive":
            break
        path_parts.insert(0, pname)
        pp = pmeta.get("parents") or []
        parent_id = pp[0] if pp else None

    full_path = DRIVE_MOUNT_ROOT.joinpath(*path_parts)
    return full_path if full_path.is_dir() else None


def gdown_into(folder_id: str, dest: Path) -> int:
    """Download a Drive folder via gdown. Returns number of files written, or -1 on failure."""
    try:
        import gdown
    except ImportError:
        print(
            "[sync] FATAL: gdown not installed. On Colab, run: uv sync --extra colab",
            file=sys.stderr,
        )
        return -1
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[sync]   trying gdown to download folder {folder_id} into {dest} ...")
    try:
        result = gdown.download_folder(
            id=folder_id,
            output=str(dest),
            quiet=False,
            use_cookies=False,
        )
    except Exception as e:
        print(
            f"[sync]   gdown FAIL: {type(e).__name__}: {e}\n"
            "[sync]     gdown only handles folders shared 'Anyone with the link'.\n"
            "[sync]     For private MyDrive folders, the Drive-mount strategy is used instead.",
            file=sys.stderr,
        )
        return -1
    if not result:
        print("[sync]   gdown FAIL: returned no files (folder likely private)", file=sys.stderr)
        return -1
    return sum(1 for p in dest.rglob("*") if p.is_file())


def gsutil_upload(src: Path, target: str) -> bool:
    """Upload a local directory to gs://target/ using gsutil -m cp -r. Returns True on success."""
    cp = subprocess.run(["gsutil", "-m", "cp", "-r", f"{src}/.", f"{target}/"])
    return cp.returncode == 0


def sync_one(folder_id: str, bucket: str, subpath: str, force: bool, tmp_root: Path) -> None:
    """Mirror one Drive folder ID to gs://bucket/subpath/ via the best-available strategy."""
    target = f"{bucket.rstrip('/')}/{subpath.strip('/')}"
    print(f"[sync] {folder_id} -> {target}")

    if not force and gcs_path_has_objects(bucket, subpath):
        print("[sync]   already populated, skipping (use --force to re-sync)")
        return

    # Strategy 1: Drive mount + gsutil (private MyDrive folders, fastest).
    mount_path = find_folder_on_mount(folder_id)
    if mount_path is not None:
        print(f"[sync]   found on mounted Drive at {mount_path}")
        if gsutil_upload(mount_path, target):
            print("[sync]   uploaded via Drive mount + gsutil")
            return
        print("[sync]   gsutil upload failed for mount path; falling back to gdown ...", file=sys.stderr)

    # Strategy 2: gdown (only works for folders shared with anyone-with-the-link).
    local = tmp_root / subpath.replace("/", "_")
    n = gdown_into(folder_id, local)
    if n < 0:
        print(
            "[sync]   FAIL: neither Drive-mount nor gdown could fetch this folder.\n"
            "[sync]   Workarounds:\n"
            "[sync]     a) move the folder into your MyDrive root, OR\n"
            "[sync]     b) share the folder publicly ('Anyone with the link'), OR\n"
            "[sync]     c) upload to GCS manually: gsutil -m cp -r ./folder/ "
            f"{target}/",
            file=sys.stderr,
        )
        sys.exit(2)
    print(f"[sync]   downloaded {n} files via gdown")
    if not gsutil_upload(local, target):
        print("[sync]   FAIL: gsutil cp from local stage failed", file=sys.stderr)
        sys.exit(4)
    print(f"[sync]   uploaded {n} files via gdown + gsutil")


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
