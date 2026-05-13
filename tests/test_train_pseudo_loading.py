"""Tests for pseudo-label dataset construction in train.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def train_mod(project_root: Path):
    import importlib
    sys.path.insert(0, str(project_root))
    if "train" in sys.modules:
        importlib.reload(sys.modules["train"])
    import train
    return train


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_build_combined_dataset_gold_only_when_no_pseudo(train_mod, tmp_path: Path) -> None:
    """If cfg.pseudo_jsonl is empty, combined dataset = gold only."""
    gold = tmp_path / "gold.jsonl"
    _write_jsonl(gold, [{"image_path": "x.jpg", "xml_label": "<medication><brand_name>A</brand_name></medication>"}])
    cfg = train_mod.Config(train_jsonl=str(gold), pseudo_jsonl="", pseudo_weight=0.0)
    rows, weights = train_mod.build_combined_train_rows(cfg)
    assert len(rows) == 1
    assert weights == [1.0]


def test_build_combined_mixes_with_weights(train_mod, tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    pseudo = tmp_path / "pseudo.jsonl"
    # Use fully-populated XML for pseudo so the all-empty filter doesn't drop it
    full_xml = "<medication><brand_name>P</brand_name><drug_name>P</drug_name><generic_name>P</generic_name><strength>5</strength><quantity>10</quantity><company>X</company><manufacturer>X</manufacturer><batch_number>B</batch_number><mfg_date>11/2023</mfg_date><expiry_date>10/2025</expiry_date><mrp>85</mrp><warnings>w</warnings></medication>"
    _write_jsonl(gold, [{"image_path": "g.jpg", "xml_label": "<medication><brand_name>G</brand_name></medication>"}])
    _write_jsonl(pseudo, [{"image_path": "p.jpg", "xml_label": full_xml}])
    cfg = train_mod.Config(train_jsonl=str(gold), pseudo_jsonl=str(pseudo), pseudo_weight=0.3)
    rows, weights = train_mod.build_combined_train_rows(cfg)
    assert len(rows) == 2
    assert weights == [1.0, 0.3]


def test_build_combined_drops_filtered_pseudo(train_mod, tmp_path: Path) -> None:
    """Pseudo rows failing filter_pseudo_rows (all-empty) should not appear."""
    gold = tmp_path / "gold.jsonl"
    pseudo = tmp_path / "pseudo.jsonl"
    _write_jsonl(gold, [{"image_path": "g.jpg", "xml_label": "<medication><brand_name>G</brand_name></medication>"}])
    # All-empty pseudo row (all 12 empty) — should be dropped by all_empty filter (default max_empty_fields=12)
    empty_xml = "<medication><brand_name></brand_name><drug_name></drug_name><generic_name></generic_name><strength></strength><quantity></quantity><company></company><manufacturer></manufacturer><batch_number></batch_number><mfg_date></mfg_date><expiry_date></expiry_date><mrp></mrp><warnings></warnings></medication>"
    _write_jsonl(pseudo, [{"image_path": "p.jpg", "xml_label": empty_xml}])
    cfg = train_mod.Config(train_jsonl=str(gold), pseudo_jsonl=str(pseudo), pseudo_weight=0.3)
    rows, weights = train_mod.build_combined_train_rows(cfg)
    assert len(rows) == 1
    assert rows[0]["image_path"] == "g.jpg"
