"""Test that run_experiment.sh accepts --track E and picks the right defaults."""
from __future__ import annotations

import subprocess
from pathlib import Path


def test_track_e_default_config_resolves(project_root: Path) -> None:
    """Invoking with --track E --dry-run should not error with 'unknown track'."""
    result = subprocess.run(
        ["bash", "scripts/run_experiment.sh", "test-run-id", "--track", "E", "--dry-run"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert "unknown track" not in result.stderr.lower(), result.stderr


def test_track_e_dry_run_prints_command(project_root: Path) -> None:
    """--dry-run should print what would be run without invoking python."""
    result = subprocess.run(
        ["bash", "scripts/run_experiment.sh", "test-run-id-dry", "--track", "E", "--dry-run"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    # Should mention the Track E config path
    assert "track_e_highres.yaml" in result.stdout or "track_e_highres.yaml" in result.stderr


def test_track_a_dry_run_still_works(project_root: Path) -> None:
    """Adding --dry-run + Track E support must not break Track A."""
    result = subprocess.run(
        ["bash", "scripts/run_experiment.sh", "test-run-id-a", "--track", "A", "--dry-run"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "baseline.yaml" in result.stdout or "baseline.yaml" in result.stderr


def test_unknown_track_still_errors(project_root: Path) -> None:
    """Unknown tracks still produce an error."""
    result = subprocess.run(
        ["bash", "scripts/run_experiment.sh", "test-id", "--track", "Z", "--dry-run"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "unknown" in result.stderr.lower()
