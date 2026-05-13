"""Tests for backfill_safety4_baseline.py — recomputes safety-4 from existing per-field JSONs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_backfill_script_exists(project_root: Path) -> None:
    assert (project_root / "scripts" / "backfill_safety4_baseline.py").exists()


def test_backfill_produces_expected_keys(project_root: Path, tmp_path: Path) -> None:
    """Script should produce a JSON with one entry per existing per_field_*.json."""
    import subprocess
    out_path = tmp_path / "safety4_baseline.json"
    result = subprocess.run(
        ["python", "scripts/backfill_safety4_baseline.py", "--out", str(out_path)],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    # Should at minimum contain Track A and Track B baselines
    assert "track_a" in data
    assert "track_b" in data
    for entry in data.values():
        assert "safety4_macro_f1" in entry
        assert "safety4_macro_edit_f1" in entry
        assert "per_field" in entry
