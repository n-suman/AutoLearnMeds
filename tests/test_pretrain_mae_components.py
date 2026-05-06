"""Lightweight (non-Colab) unit tests for pretrain_mae.py pure-logic helpers."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture
def pretrain_mae_mod():
    """Import pretrain_mae.py from the project root, fresh per test.
    Lazy ml imports inside pretrain_mae.py mean this works in the torch-less Mac venv.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if "pretrain_mae" in sys.modules:
        importlib.reload(sys.modules["pretrain_mae"])
    import pretrain_mae
    return pretrain_mae


def test_mae_config_loads_yaml(pretrain_mae_mod, project_root: Path) -> None:
    cfg = pretrain_mae_mod.MAEConfig.from_yaml(
        project_root / "experiments" / "configs" / "mae_pretrain.yaml"
    )
    assert cfg.mask_ratio == 0.75
    assert cfg.decoder_depth == 4
    assert cfg.total_epochs == 200
    assert cfg.adam_betas == (0.9, 0.95)
    assert cfg.random_resized_crop_scale == (0.5, 1.0)


def test_pretrain_mae_module_imports_without_torch() -> None:
    """pretrain_mae.py must import in a torch-less venv (lazy ml imports).
    Belt-and-suspenders: confirm torch and transformers aren't in the module's
    top-level namespace.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if "pretrain_mae" in sys.modules:
        del sys.modules["pretrain_mae"]
    import pretrain_mae
    assert not hasattr(pretrain_mae, "torch")
    assert not hasattr(pretrain_mae, "transformers")


def test_mae_config_excludes_val_test_jsonls(pretrain_mae_mod, project_root: Path) -> None:
    """Strict-paper claim: val + test image pixels are excluded from MAE pretraining."""
    cfg = pretrain_mae_mod.MAEConfig.from_yaml(
        project_root / "experiments" / "configs" / "mae_pretrain.yaml"
    )
    assert isinstance(cfg.exclude_jsonls, tuple)
    assert "data/processed/val.jsonl" in cfg.exclude_jsonls
    assert "data/processed/test.jsonl" in cfg.exclude_jsonls


def test_build_pretrain_dataset_filters_excluded(pretrain_mae_mod, tmp_path: Path) -> None:
    """build_pretrain_dataset must drop images whose basename appears in exclude_jsonls."""
    import json as _json
    import types

    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"]:
        (raw / name).write_bytes(b"\xff\xd8\xff")  # minimal jpg-ish bytes
    excl = tmp_path / "exclude.jsonl"
    excl.write_text("\n".join([
        _json.dumps({"image_path": "raw/b.jpg"}),
        _json.dumps({"image_path": "raw/c.jpg"}),
    ]))

    # build_pretrain_dataset only reads cfg.images_root and cfg.exclude_jsonls,
    # so a SimpleNamespace with those two fields is enough — no need to fill the
    # full MAEConfig.
    cfg = types.SimpleNamespace(
        images_root=str(raw),
        exclude_jsonls=(str(excl),),
    )

    paths = pretrain_mae_mod.build_pretrain_dataset(cfg)
    names = sorted(p.name for p in paths)
    assert names == ["a.jpg", "d.jpg", "e.jpg"], f"expected b.jpg+c.jpg excluded, got {names}"
