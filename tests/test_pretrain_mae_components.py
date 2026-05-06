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


# === Phase 7b — text-aware MAE + TAPT ===

def test_edge_density_per_patch_shape(pretrain_mae_mod, tmp_path: Path) -> None:
    """Sobel edges + per-patch mean produces shape (num_patches,) with no NaN, all positive."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    # Synthetic 224x224 image with vertical lines every 8 px — yields edges in
    # every 16x16 patch.
    img = np.full((224, 224), 255, dtype=np.uint8)
    img[:, ::8] = 0
    img_path = tmp_path / "synth.png"
    cv2.imwrite(str(img_path), img)
    edges = pretrain_mae_mod.compute_edge_density_per_patch(img_path, 224, 16)
    assert edges.shape == (196,)
    assert np.all(edges > 0)
    assert not np.isnan(edges).any()


def test_random_masking_with_weights_biases_high_weight(pretrain_mae_mod) -> None:
    """High-weight patches should be masked at higher rate; low-weight at lower rate.

    Per the Phase 7b spec: weight=1 (text-like) should be masked > 0.75 average,
    weight=0 (bg-like) should be masked < 0.75 average, total mean ~ 0.75.
    """
    torch = pytest.importorskip("torch")
    torch.manual_seed(42)
    B, L, D = 100, 196, 16
    x = torch.randn(B, L, D)
    weights = torch.cat([torch.ones(B, 50), torch.zeros(B, 146)], dim=1)
    visible, mask, _ = pretrain_mae_mod.random_masking(x, mask_ratio=0.75, weights=weights)
    text_mask_rate = mask[:, :50].mean().item()
    bg_mask_rate = mask[:, 50:].mean().item()
    assert text_mask_rate > 0.75, f"text patches not masked enough: {text_mask_rate}"
    assert bg_mask_rate < 0.75, f"bg patches over-masked: {bg_mask_rate}"
    assert text_mask_rate > bg_mask_rate
    assert abs(mask.mean().item() - 0.75) < 0.05, f"total mean drifted: {mask.mean().item()}"


def test_text_aware_yaml_loads(pretrain_mae_mod, project_root: Path) -> None:
    """The text-aware MAE pretraining yaml loads with the expected fields set."""
    cfg = pretrain_mae_mod.MAEConfig.from_yaml(
        project_root / "experiments" / "configs" / "mae_pretrain_text_aware.yaml"
    )
    assert cfg.text_aware_masking is True
    assert cfg.text_mask_rate == 0.90
    assert cfg.bg_mask_rate == 0.60


def test_tapt_yaml_loads(pretrain_mae_mod, project_root: Path) -> None:
    """The TAPT MAE pretraining yaml loads include_jsonls + 50 epochs + uniform masking."""
    cfg = pretrain_mae_mod.MAEConfig.from_yaml(
        project_root / "experiments" / "configs" / "mae_pretrain_tapt.yaml"
    )
    assert cfg.include_jsonls == ("data/processed/train.jsonl",)
    assert cfg.total_epochs == 50
    assert cfg.text_aware_masking is False
