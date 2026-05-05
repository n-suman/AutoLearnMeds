"""Disaster-recovery drill — Colab-marked.

Workflow:
  1. Snapshot SHA-256 of all files under experiments/runs/.
  2. Force a sync to GCS.
  3. Nuke experiments/runs/ locally (simulating a Colab runtime wipe).
  4. Run scripts/disaster_recover.sh (pulls from GCS).
  5. Re-snapshot and assert SHAs match the pre-nuke set.

Run only on Colab (needs gsutil + the live GCS bucket).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.colab


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_tree(root: Path, rel_to: Path) -> dict[str, str]:
    """Map relative-path -> sha256 for every regular file under root."""
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(rel_to))] = _sha256_file(p)
    return out


def test_disaster_recovery_restores_experiments_runs(project_root: Path) -> None:
    """End-to-end: snapshot -> nuke -> recover -> hash-compare."""
    bucket = os.environ.get("AUTOLEARNMEDS_GCS_BUCKET", "gs://auto_learn_meds")
    runs_dir = project_root / "experiments" / "runs"
    if not runs_dir.is_dir() or not any(runs_dir.iterdir()):
        pytest.skip("experiments/runs/ is empty; nothing to test recovery of")

    # 1. Snapshot pre-nuke hashes.
    pre_hashes = _hash_tree(runs_dir, project_root)
    assert pre_hashes, "expected non-empty hash set"

    # 2. Force a sync to GCS so it has the latest. Direct gsutil call mirrors
    # what sync_to_gcs.sh does in its loop.
    sync_result = subprocess.run(
        ["gsutil", "-m", "rsync", "-r", "-d",
         str(runs_dir),
         f"{bucket.rstrip('/')}/experiments/runs"],
        capture_output=True, text=True, timeout=120,
    )
    assert sync_result.returncode == 0, f"pre-test sync failed: {sync_result.stderr}"

    # 3. Nuke local copy (simulated Colab wipe).
    shutil.rmtree(runs_dir)
    assert not runs_dir.exists()

    # 4. Run disaster_recover.sh.
    env = os.environ.copy()
    env["AUTOLEARNMEDS_GCS_BUCKET"] = bucket
    env["AUTOLEARNMEDS_WORKSPACE"] = str(project_root)
    rec = subprocess.run(
        ["bash", str(project_root / "scripts" / "disaster_recover.sh")],
        capture_output=True, text=True, timeout=300, env=env,
    )
    assert rec.returncode == 0, f"recover script failed: stderr={rec.stderr}"
    assert runs_dir.is_dir(), "experiments/runs/ should have been restored"

    # 5. Re-hash and compare.
    post_hashes = _hash_tree(runs_dir, project_root)
    missing = set(pre_hashes) - set(post_hashes)
    extra = set(post_hashes) - set(pre_hashes)
    mismatched = {k for k in pre_hashes if k in post_hashes and pre_hashes[k] != post_hashes[k]}
    assert not missing, f"recovered tree is missing files: {sorted(missing)[:5]}"
    assert not extra, f"recovered tree has extra files: {sorted(extra)[:5]}"
    assert not mismatched, f"recovered tree has hash mismatches: {sorted(mismatched)[:5]}"
