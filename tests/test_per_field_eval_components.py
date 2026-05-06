"""Lightweight tests for the per-field evaluation infrastructure.

Two tests, both no-Colab:
 1. ``test_per_field_eval_module_imports_lazily`` - asserts torch/transformers
    are NOT pulled into per_field_eval at module import time. (CLI scripts
    that mirror train_qwen.py's lazy-import convention should pass this.)
 2. ``test_compare_per_field_output_shape`` - feeds two synthetic per-field
    JSONs into compare_per_field.main() and validates the output markdown
    structure (table contents, "Track B wins" / "Track A wins" sections).
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def repo_root_on_path() -> Path:
    """Ensure the project root is on sys.path; return its Path."""
    p = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(p))
    sys.path.insert(0, str(p / "scripts"))
    return p


def test_per_field_eval_module_imports_lazily(repo_root_on_path: Path) -> None:
    """scripts/per_field_eval.py must import without dragging in heavy ml deps."""
    # Make sure we re-import fresh so we observe a clean module namespace.
    for modname in ("per_field_eval",):
        if modname in sys.modules:
            del sys.modules[modname]

    import per_field_eval

    assert not hasattr(per_field_eval, "torch"), (
        "torch leaked into per_field_eval module namespace - move to lazy import"
    )
    assert not hasattr(per_field_eval, "transformers"), (
        "transformers leaked into per_field_eval module namespace"
    )
    assert not hasattr(per_field_eval, "peft"), (
        "peft leaked into per_field_eval module namespace"
    )
    # Sanity: required cheap functions are still exposed.
    assert callable(per_field_eval.main)
    assert callable(per_field_eval.build_track_a)
    assert callable(per_field_eval.build_track_b)


def _track_a_fixture() -> dict:
    """A wins on field_x f1, ties on field_y, loses on field_z (B wins f1+edit)."""
    return {
        "track": "A",
        "ckpt": "fixture/best.pt",
        "config": "experiments/configs/baseline.yaml",
        "val_jsonl": "data/processed/val.jsonl",
        "n_samples": 111,
        "macro_f1": 0.50,
        "macro_edit_f1": 0.60,
        "per_field_f1": {
            "field_x": 0.80,
            "field_y": 0.40,
            "field_z": 0.10,
        },
        "per_field_edit_f1": {
            "field_x": 0.85,
            "field_y": 0.40,
            "field_z": 0.20,
        },
    }


def _track_b_fixture() -> dict:
    """B beats A on field_z f1+edit; ties field_y; loses field_x."""
    return {
        "track": "B",
        "ckpt": "fixture/best_lora",
        "config": "experiments/configs/qwen_baseline.yaml",
        "val_jsonl": "data/processed/val.jsonl",
        "n_samples": 111,
        "macro_f1": 0.30,
        "macro_edit_f1": 0.40,
        "per_field_f1": {
            "field_x": 0.20,
            "field_y": 0.40,
            "field_z": 0.50,
        },
        "per_field_edit_f1": {
            "field_x": 0.30,
            "field_y": 0.40,
            "field_z": 0.55,
        },
    }


def test_compare_per_field_output_shape(
    repo_root_on_path: Path, tmp_path: Path
) -> None:
    """End-to-end: compare_per_field.main writes a markdown with the right shape."""
    # Drop any cached import so the test runs fresh.
    if "compare_per_field" in sys.modules:
        importlib.reload(sys.modules["compare_per_field"])
    import compare_per_field

    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    out_path = tmp_path / "out.md"

    a_path.write_text(json.dumps(_track_a_fixture()))
    b_path.write_text(json.dumps(_track_b_fixture()))

    rc = compare_per_field.main(
        ["--a", str(a_path), "--b", str(b_path), "--out", str(out_path)]
    )
    assert rc == 0
    assert out_path.is_file()
    md = out_path.read_text()

    # All three fields appear in the table.
    for fname in ("field_x", "field_y", "field_z"):
        assert f"| {fname} |" in md, f"missing field {fname} in table"

    # The macro row is present.
    assert "**macro**" in md

    # Track B wins section: should ONLY contain field_z.
    # (field_x: A wins both; field_y: tie; field_z: B wins both.)
    b_wins_section = md.split("## Track B wins")[1].split("## Track A wins")[0]
    assert "- field_z" in b_wins_section
    assert "- field_x" not in b_wins_section
    assert "- field_y" not in b_wins_section

    # Track A wins section: should contain field_x AND field_y (ties go to A's bucket).
    a_wins_section = md.split("## Track A wins")[1].split("## What this tells")[0]
    assert "- field_x" in a_wins_section
    assert "- field_y" in a_wins_section
    assert "- field_z" not in a_wins_section

    # Paper paragraph mentions the winning field.
    assert "field_z" in md.split("## What this tells the paper")[1]
