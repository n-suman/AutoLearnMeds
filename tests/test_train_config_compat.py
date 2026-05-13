"""Regression: Track A's baseline.yaml must still load + produce a Config with no new behavior enabled."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def train_mod(project_root: Path):
    sys.path.insert(0, str(project_root))
    if "train" in sys.modules:
        importlib.reload(sys.modules["train"])
    import train
    return train


def test_baseline_yaml_loads_into_config(project_root: Path, train_mod) -> None:
    cfg_path = project_root / "experiments" / "configs" / "baseline.yaml"
    raw = yaml.safe_load(cfg_path.read_text())
    cfg = train_mod.Config(**raw)
    assert cfg.encoder_model == "google/siglip-base-patch16-224"
    assert cfg.max_steps == 1000


def test_baseline_does_not_enable_new_features(project_root: Path, train_mod) -> None:
    """All Track E features must default to off when not specified in YAML."""
    cfg_path = project_root / "experiments" / "configs" / "baseline.yaml"
    raw = yaml.safe_load(cfg_path.read_text())
    cfg = train_mod.Config(**raw)
    # New flags should default to off (None / empty / 0.0)
    assert cfg.pseudo_jsonl == ""
    assert cfg.pseudo_weight == 0.0
    assert cfg.pseudo_weight_mode == "const"
    assert cfg.pseudo_min_confidence == "medium"
    assert cfg.image_size == 224
    assert cfg.stage_schedule == []
    assert cfg.compute_calibration is False


def test_track_e_yaml_overrides_defaults(project_root: Path, train_mod, tmp_path: Path) -> None:
    """A Track E-style config should set the new flags to their non-default values."""
    track_e_raw = {
        "seed": 44,
        "encoder_model": "google/siglip-large-patch16-384",
        "image_size": 384,
        "pseudo_jsonl": "data/pseudo_labels/round_001.jsonl",
        "pseudo_weight": 0.3,
        "compute_calibration": True,
        "max_steps": 3000,
    }
    cfg = train_mod.Config(**track_e_raw)
    assert cfg.image_size == 384
    assert cfg.pseudo_weight == 0.3
    assert cfg.compute_calibration is True


def test_preflight_rejects_pseudo_with_val_leak(project_root: Path, tmp_path: Path, train_mod) -> None:
    """If pseudo_jsonl contains an image_path from val, preflight_leak_check should raise."""
    val_jsonl = tmp_path / "val.jsonl"
    pseudo_jsonl = tmp_path / "pseudo.jsonl"
    val_jsonl.write_text('{"image_path": "raw/raw_images/IMG_001.JPG", "xml_label": "<medication><brand_name>X</brand_name></medication>"}\n')
    pseudo_jsonl.write_text('{"image_path": "raw/raw_images/IMG_001.JPG", "xml_label": "<medication><brand_name>X</brand_name></medication>"}\n')

    cfg = train_mod.Config(val_jsonl=str(val_jsonl), pseudo_jsonl=str(pseudo_jsonl), pseudo_weight=0.3)
    with pytest.raises(AssertionError, match="leak"):
        train_mod.preflight_leak_check(cfg)


def test_preflight_passes_when_no_overlap(project_root: Path, tmp_path: Path, train_mod) -> None:
    val_jsonl = tmp_path / "val.jsonl"
    pseudo_jsonl = tmp_path / "pseudo.jsonl"
    val_jsonl.write_text('{"image_path": "raw/raw_images/IMG_001.JPG", "xml_label": "x"}\n')
    pseudo_jsonl.write_text('{"image_path": "raw/raw_images/IMG_999.JPG", "xml_label": "x"}\n')
    cfg = train_mod.Config(val_jsonl=str(val_jsonl), pseudo_jsonl=str(pseudo_jsonl), pseudo_weight=0.3)
    train_mod.preflight_leak_check(cfg)  # should not raise


def test_preflight_noop_when_pseudo_disabled(project_root: Path, train_mod) -> None:
    """If pseudo_weight=0.0 or pseudo_jsonl='', preflight is a no-op."""
    cfg = train_mod.Config()
    train_mod.preflight_leak_check(cfg)  # should not raise (no pseudo configured)
