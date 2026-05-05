"""Tests for scripts/build_data_card.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module(project_root: Path):
    spec = importlib.util.spec_from_file_location(
        "build_data_card", project_root / "scripts" / "build_data_card.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_module_loads(project_root: Path) -> None:
    mod = _load_module(project_root)
    assert hasattr(mod, "build_data_card")


def test_build_data_card_writes_md(
    project_root: Path,
    synthetic_jsonl_dir: Path,
    synthetic_images_dir: Path,
    tiny_processed_dir: Path,
    tmp_path: Path,
) -> None:
    # Need processed JSONLs first.
    import sys
    sys.path.insert(0, str(project_root / "scripts"))
    import build_processed
    build_processed.build_processed(
        gold_standard_path=synthetic_jsonl_dir / "gold_standard.jsonl",
        splits_path=synthetic_jsonl_dir / "splits.json",
        images_root=synthetic_images_dir,
        out_dir=tiny_processed_dir,
        image_path_prefix="raw/raw_images",
    )

    out_path = tmp_path / "data_card.md"
    mod = _load_module(project_root)
    mod.build_data_card(processed_dir=tiny_processed_dir, out_path=out_path)
    assert out_path.is_file()
    text = out_path.read_text()

    # Required sections per Gebru 2018 datasheet conventions
    assert "# Data Card" in text
    assert "## Composition" in text
    assert "## Splits" in text
    assert "## Field presence" in text
    # Per-split sizes
    assert "train" in text
    assert "val" in text
    assert "test" in text
    # The actual sizes from synthetic fixtures
    assert "3" in text  # train size
    assert "1" in text  # val size
    # Field names
    assert "brand_name" in text


def test_build_data_card_handles_empty_split(
    project_root: Path, tmp_path: Path
) -> None:
    """Should not crash if a split JSONL is empty."""
    proc = tmp_path / "processed"
    proc.mkdir()
    for sp in ("train", "val", "test"):
        (proc / f"{sp}.jsonl").write_text("")

    mod = _load_module(project_root)
    out_path = tmp_path / "data_card.md"
    mod.build_data_card(processed_dir=proc, out_path=out_path)
    text = out_path.read_text()
    assert "train" in text
    assert "0" in text  # zero-size split
