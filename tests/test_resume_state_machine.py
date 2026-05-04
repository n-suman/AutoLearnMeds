"""Tests for scripts/resume_or_start.py — atomic experiment state machine.

State table (from spec §7.3):
| Last visible state                                     | Action                              |
| no _active.json                                        | START_NEW                           |
| _active.json status=PLANNING                           | RESTART (this experiment)           |
| _active.json status=ACTIVE, no metrics.json            | RESTART (training was killed)       |
| _active.json with metrics.json but no ledger entry     | RESUME_FROM_EVALUATED               |
| _active.json with ledger entry                         | CLEAR_AND_START_NEW                 |
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_module(project_root: Path):
    spec = importlib.util.spec_from_file_location(
        "resume_or_start",
        project_root / "scripts" / "resume_or_start.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A fresh fake workspace tree with the expected layout."""
    (tmp_path / "experiments" / "runs").mkdir(parents=True)
    (tmp_path / "experiments" / "ledger.jsonl").touch()
    return tmp_path


def _write_active(workspace: Path, status: str, run_id: str = "run-001") -> None:
    payload = {"run_id": run_id, "status": status, "started_at": "2026-05-05T12:00:00Z"}
    (workspace / "experiments" / "_active.json").write_text(json.dumps(payload))


def _write_metrics(workspace: Path, run_id: str = "run-001", final_macro_f1: float = 0.5) -> None:
    run_dir = workspace / "experiments" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps({"final_macro_f1": final_macro_f1}))


def _append_ledger(workspace: Path, run_id: str = "run-001") -> None:
    line = json.dumps({"run_id": run_id, "kept": False, "metrics": {"final_macro_f1": 0.5}})
    with (workspace / "experiments" / "ledger.jsonl").open("a") as fh:
        fh.write(line + "\n")


def test_no_active_returns_start_new(workspace: Path, project_root: Path) -> None:
    mod = _load_module(project_root)
    assert mod.next_action(workspace) == "START_NEW"


def test_planning_status_returns_restart(workspace: Path, project_root: Path) -> None:
    mod = _load_module(project_root)
    _write_active(workspace, "PLANNING")
    assert mod.next_action(workspace) == "RESTART"


def test_active_without_metrics_returns_restart(workspace: Path, project_root: Path) -> None:
    mod = _load_module(project_root)
    _write_active(workspace, "ACTIVE")
    assert mod.next_action(workspace) == "RESTART"


def test_active_with_metrics_no_ledger_returns_resume_from_evaluated(
    workspace: Path, project_root: Path
) -> None:
    mod = _load_module(project_root)
    _write_active(workspace, "ACTIVE")
    _write_metrics(workspace)
    assert mod.next_action(workspace) == "RESUME_FROM_EVALUATED"


def test_active_with_ledger_entry_returns_clear_and_start_new(
    workspace: Path, project_root: Path
) -> None:
    mod = _load_module(project_root)
    _write_active(workspace, "COMMITTED")
    _write_metrics(workspace)
    _append_ledger(workspace)
    assert mod.next_action(workspace) == "CLEAR_AND_START_NEW"
