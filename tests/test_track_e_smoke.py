"""End-to-end smoke: Track E gold-only config loads and the trainer reaches the first eval.

This test requires:
- GPU (CUDA-capable)
- HuggingFace model cache (SigLIP-large-384 weights, ~600MB first run)
- The project's data files (data/processed/train.jsonl, val.jsonl)

On CPU-only machines (e.g. Codex's sandbox, CI without GPU), the test is auto-skipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.skipif(not _torch_cuda_available(), reason="requires CUDA GPU")
def test_track_e_gold_only_first_step(project_root: Path, tmp_path: Path) -> None:
    """Run train.py with max_steps=10 to verify the pipeline doesn't crash."""
    import subprocess
    import yaml

    # Use the existing baseline.yaml as base, then override the Track E flags
    cfg_overrides = {
        "max_steps": 10,
        "eval_every": 10,
        "encoder_model": "google/siglip-large-patch16-384",
        "image_size": 384,
        "batch_size": 1,
    }
    # Write a tmp config that merges baseline + overrides
    base = yaml.safe_load((project_root / "experiments" / "configs" / "baseline.yaml").read_text())
    base.update(cfg_overrides)
    tmp_cfg = tmp_path / "smoke.yaml"
    tmp_cfg.write_text(yaml.safe_dump(base))

    result = subprocess.run(
        ["python", "train.py", "--config", str(tmp_cfg)],
        cwd=project_root, capture_output=True, text=True, timeout=600,
    )
    # Trainer should exit 0 and emit `final_macro_f1=...` on its last stdout line
    assert result.returncode == 0, f"train.py exited {result.returncode}\nstderr:\n{result.stderr}"
    last_line = result.stdout.strip().splitlines()[-1]
    assert last_line.startswith("final_macro_f1="), f"Last stdout line: {last_line!r}"


@pytest.mark.skipif(not _torch_cuda_available(), reason="requires CUDA GPU")
def test_track_e_yaml_via_dry_run(project_root: Path) -> None:
    """Less-expensive sanity: run_experiment.sh --track E --dry-run should succeed."""
    import subprocess
    result = subprocess.run(
        ["bash", "scripts/run_experiment.sh", "track-e-smoke", "--track", "E", "--dry-run"],
        cwd=project_root, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "track_e_highres.yaml" in (result.stdout + result.stderr)
