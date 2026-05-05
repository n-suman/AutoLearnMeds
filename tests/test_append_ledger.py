"""Tests for scripts/append_ledger.py."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_module(project_root: Path):
    spec = importlib.util.spec_from_file_location(
        "append_ledger", project_root / "scripts" / "append_ledger.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_run(workspace: Path, run_id: str, final_macro_f1: float, with_notes: bool = True) -> Path:
    run_dir = workspace / "experiments" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps({
        "run_id": run_id,
        "git_sha": "deadbeef",
        "config": "experiments/configs/baseline.yaml",
        "final_macro_f1": final_macro_f1,
        "wall_clock_seconds": 1800,
        "exit_code": 0,
    }))
    (run_dir / "config.yaml").write_text("seed: 42\nhidden_dim: 512\n")
    (run_dir / "train.py").write_text("# snapshot\n")
    if with_notes:
        (run_dir / "notes.md").write_text(
            "# Experiment: " + run_id + "\n\n"
            "## Hypothesis\nTest hypothesis.\n\n"
            "## Citation\n- press_2022_alibi.pdf §3.1\n\n"
            "## Adjacent ideas\nN/A\n\n"
            "## Predicted direction\n+0.01\n"
        )
    return run_dir


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "experiments" / "runs").mkdir(parents=True)
    (tmp_path / "experiments" / "ledger.jsonl").touch()
    return tmp_path


def test_append_creates_ledger_entry(project_root: Path, workspace: Path) -> None:
    mod = _load_module(project_root)
    _make_run(workspace, "run-001", 0.0710)
    mod.append_ledger(
        run_id="run-001",
        workspace=workspace,
        phase="explore",
        keep_threshold=0.001,
    )
    line = (workspace / "experiments" / "ledger.jsonl").read_text().strip()
    entry = json.loads(line)
    assert entry["run_id"] == "run-001"
    assert entry["phase"] == "explore"
    assert entry["metrics"]["final_macro_f1"] == 0.0710
    assert "kept" in entry
    assert "best_at_time_of_run" in entry
    assert entry["timestamp"].endswith("Z") or "+" in entry["timestamp"]


def test_append_kept_when_above_threshold(project_root: Path, workspace: Path) -> None:
    mod = _load_module(project_root)
    # Seed a prior best at 0.0710
    _make_run(workspace, "run-baseline", 0.0710)
    mod.append_ledger(run_id="run-baseline", workspace=workspace, phase="explore", keep_threshold=0.001)
    # New run beats by 0.005
    _make_run(workspace, "run-better", 0.0760)
    mod.append_ledger(run_id="run-better", workspace=workspace, phase="explore", keep_threshold=0.001)
    lines = [json.loads(l) for l in (workspace / "experiments" / "ledger.jsonl").read_text().splitlines() if l.strip()]
    assert lines[-1]["kept"] is True
    assert lines[-1]["best_at_time_of_run"] == pytest.approx(0.0710)


def test_append_reverted_when_below_threshold(project_root: Path, workspace: Path) -> None:
    mod = _load_module(project_root)
    _make_run(workspace, "run-baseline", 0.0710)
    mod.append_ledger(run_id="run-baseline", workspace=workspace, phase="explore", keep_threshold=0.001)
    _make_run(workspace, "run-worse", 0.0700)
    mod.append_ledger(run_id="run-worse", workspace=workspace, phase="explore", keep_threshold=0.001)
    lines = [json.loads(l) for l in (workspace / "experiments" / "ledger.jsonl").read_text().splitlines() if l.strip()]
    assert lines[-1]["kept"] is False


def test_append_extracts_hypothesis_from_notes(project_root: Path, workspace: Path) -> None:
    mod = _load_module(project_root)
    _make_run(workspace, "run-001", 0.0710)
    mod.append_ledger(run_id="run-001", workspace=workspace, phase="explore", keep_threshold=0.001)
    entry = json.loads((workspace / "experiments" / "ledger.jsonl").read_text().splitlines()[-1])
    assert entry["hypothesis"] == "Test hypothesis."
    assert "press_2022_alibi.pdf" in entry["citation"][0]


def test_append_handles_missing_notes(project_root: Path, workspace: Path) -> None:
    """If notes.md is absent, ledger entry still writes with hypothesis='(missing)' and a warning."""
    mod = _load_module(project_root)
    _make_run(workspace, "run-001", 0.0710, with_notes=False)
    mod.append_ledger(run_id="run-001", workspace=workspace, phase="explore", keep_threshold=0.001)
    entry = json.loads((workspace / "experiments" / "ledger.jsonl").read_text().splitlines()[-1])
    assert entry["hypothesis"].startswith("(missing")
