"""Tests for scripts/leaderboard.py."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_module(project_root: Path):
    spec = importlib.util.spec_from_file_location(
        "leaderboard", project_root / "scripts" / "leaderboard.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_ledger(workspace: Path, entries: list[dict]) -> Path:
    p = workspace / "experiments" / "ledger.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(e) for e in entries))
    return p


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


def test_renders_table_in_descending_order(project_root: Path, workspace: Path) -> None:
    mod = _load_module(project_root)
    _write_ledger(workspace, [
        {"run_id": "a", "phase": "explore", "metrics": {"final_macro_f1": 0.05}, "kept": False, "rationale": "x", "wall_clock_min": 30},
        {"run_id": "b", "phase": "explore", "metrics": {"final_macro_f1": 0.10}, "kept": True, "rationale": "y", "wall_clock_min": 30},
        {"run_id": "c", "phase": "explore", "metrics": {"final_macro_f1": 0.07}, "kept": False, "rationale": "z", "wall_clock_min": 30},
    ])
    out = workspace / "experiments" / "leaderboard.md"
    mod.regenerate(workspace, out)
    text = out.read_text()
    # b (0.10) appears before c (0.07) appears before a (0.05)
    assert text.index("| b ") < text.index("| c ")
    assert text.index("| c ") < text.index("| a ")


def test_marks_kept_with_marker(project_root: Path, workspace: Path) -> None:
    mod = _load_module(project_root)
    _write_ledger(workspace, [
        {"run_id": "kept_run", "phase": "explore", "metrics": {"final_macro_f1": 0.10}, "kept": True, "rationale": "x", "wall_clock_min": 30},
        {"run_id": "rev_run",  "phase": "explore", "metrics": {"final_macro_f1": 0.05}, "kept": False, "rationale": "y", "wall_clock_min": 30},
    ])
    out = workspace / "experiments" / "leaderboard.md"
    mod.regenerate(workspace, out)
    text = out.read_text()
    # Some marker — checking ✓ for kept and ✗ for reverted
    kept_line = [l for l in text.splitlines() if "kept_run" in l][0]
    rev_line = [l for l in text.splitlines() if "rev_run" in l][0]
    assert "✓" in kept_line or "yes" in kept_line.lower()
    assert "✗" in rev_line or "no" in rev_line.lower()


def test_handles_empty_ledger(project_root: Path, workspace: Path) -> None:
    mod = _load_module(project_root)
    (workspace / "experiments").mkdir(exist_ok=True, parents=True)
    (workspace / "experiments" / "ledger.jsonl").write_text("")
    out = workspace / "experiments" / "leaderboard.md"
    mod.regenerate(workspace, out)
    text = out.read_text()
    assert "no experiments yet" in text.lower() or "0 experiments" in text.lower()
