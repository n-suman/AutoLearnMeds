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


# === Phase 7c (F8a) — resumable checkpointing ===

def test_mae_config_save_every_epoch_default(pretrain_mae_mod) -> None:
    """save_every_epoch should default to 10 (more frequent than the literal yaml values)."""
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(pretrain_mae_mod.MAEConfig)}
    assert fields["save_every_epoch"].default == 10
    assert fields["save_to_latest"].default is True


def test_mae_config_yamls_still_load_with_save_every_epoch(pretrain_mae_mod, project_root):
    """The three existing yamls all set save_every_epoch explicitly; verify they still parse."""
    for yaml_name in ("mae_pretrain.yaml", "mae_pretrain_text_aware.yaml", "mae_pretrain_tapt.yaml"):
        cfg = pretrain_mae_mod.MAEConfig.from_yaml(
            project_root / "experiments" / "configs" / yaml_name
        )
        # Text-aware yaml was reduced to 10 in commit 24d351b for VM-reclaim resilience.
        assert cfg.save_every_epoch in (50, 25, 10), f"{yaml_name}: unexpected save_every_epoch={cfg.save_every_epoch}"
        assert cfg.save_to_latest is True


def test_save_full_state_writes_loadable_files(pretrain_mae_mod, tmp_path):
    """The save helper produces files that can be loaded back."""
    torch = pytest.importorskip("torch")
    import dataclasses

    # Minimal cfg-shaped object (the helper only reads cfg via dataclasses.asdict).
    @dataclasses.dataclass
    class _MiniCfg:
        save_every_epoch: int = 10

    pytest.importorskip("transformers")
    # Don't actually load SigLIP in the unit test; use a tiny stand-in.
    import torch.nn as nn
    class _StubEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)
        def save_pretrained(self, path):
            # Mimic HF's save_pretrained: just write a sentinel file.
            Path(path).mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), Path(path) / "weights.pt")

    enc = _StubEncoder()
    dec = nn.Linear(8, 8)
    optim = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=1e-4)

    epoch_dir = tmp_path / "epoch-test"
    pretrain_mae_mod._save_full_state(epoch_dir, enc, dec, optim, step=42, last_loss=0.123, cfg=_MiniCfg())

    assert (epoch_dir / "weights.pt").is_file()
    assert (epoch_dir / "trainer_state.pt").is_file()
    state = torch.load(epoch_dir / "trainer_state.pt", weights_only=False)
    assert state["step"] == 42
    assert abs(state["last_loss"] - 0.123) < 1e-9
    assert "decoder_state_dict" in state
    assert "optimizer_state_dict" in state
