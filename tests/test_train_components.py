"""Lightweight (non-Colab) unit tests for train.py pure-logic helpers."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def train_mod():
    """Import train.py from the project root, fresh per test.
    Lazy ml imports inside train.py mean this works in the torch-less Mac venv.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if "train" in sys.modules:
        importlib.reload(sys.modules["train"])
    import train
    return train


def test_config_loads_baseline_yaml(train_mod, project_root: Path) -> None:
    cfg = train_mod.Config.from_yaml(project_root / "experiments" / "configs" / "baseline.yaml")
    assert cfg.seed == 42
    assert cfg.encoder_model == "google/siglip-base-patch16-224"
    assert cfg.hidden_dim == 512
    assert cfg.n_decoder_layers == 6
    assert cfg.n_decoder_heads == 8
    assert cfg.batch_size == 16
    assert cfg.peak_lr == pytest.approx(3e-4)
    assert cfg.precision == "bf16"


def test_set_seed_is_deterministic(train_mod) -> None:
    """Calling set_seed twice with the same seed produces the same first int from random."""
    import random as _r
    train_mod.set_seed(42)
    a = _r.randint(0, 1_000_000)
    train_mod.set_seed(42)
    b = _r.randint(0, 1_000_000)
    assert a == b


def test_module_imports_without_torch(train_mod) -> None:
    """train.py must import in a torch-less venv (lazy ml imports).
    Belt-and-suspenders: confirm torch isn't in the module's top-level namespace.
    """
    assert not hasattr(train_mod, "torch")


def test_cfg_has_encoder_init_path_field(train_mod, project_root: Path) -> None:
    """Phase 7: train.py must support cfg.encoder_init_path so Track A can
    load an MAE-pretrained encoder."""
    cfg = train_mod.Config()
    assert hasattr(cfg, "encoder_init_path")
    assert cfg.encoder_init_path == ""  # default: empty -> falls back to HF hub
    # Source-level grep: confirm the field is actually wired into the encoder load.
    src = (project_root / "train.py").read_text()
    assert "encoder_init_path" in src
    assert "cfg.encoder_init_path" in src


def test_baseline_mae_init_yaml_loads(train_mod, project_root: Path) -> None:
    """The Phase 7 Track A config with MAE init must load and set the new field."""
    cfg = train_mod.Config.from_yaml(
        project_root / "experiments" / "configs" / "baseline_mae_init.yaml"
    )
    assert cfg.encoder_init_path == "checkpoints/mae/run-seed44/final"
    # Sanity: other fields still come through identical to baseline.
    assert cfg.encoder_model == "google/siglip-base-patch16-224"
    assert cfg.hidden_dim == 512


def test_build_encoder_default_returns_base_dim(project_root: Path) -> None:
    """Default config produces SigLIP-base with hidden_size=768."""
    torch = pytest.importorskip("torch")  # skip in torch-less envs
    pytest.importorskip("transformers")   # skip if transformers not installed

    import sys
    sys.path.insert(0, str(project_root))
    import train

    cfg = train.Config()
    enc, hidden_dim = train.build_encoder(cfg)
    assert hidden_dim == 768  # SigLIP-base hidden size
    # Encoder should be frozen
    assert all(not p.requires_grad for p in enc.parameters())


def test_build_encoder_large_returns_large_dim(project_root: Path) -> None:
    """When cfg.encoder_model = SigLIP-large, hidden_dim should be 1024.

    NOTE: this test actually downloads SigLIP-large the first time it runs (~600 MB).
    Mark as a slow/colab test.
    """
    torch = pytest.importorskip("torch")  # skip in torch-less envs
    pytest.importorskip("transformers")   # skip if transformers not installed

    import sys
    sys.path.insert(0, str(project_root))
    import train

    cfg = train.Config(encoder_model="google/siglip-large-patch16-384", image_size=384)
    enc, hidden_dim = train.build_encoder(cfg)
    assert hidden_dim == 1024


def test_dual_best_checkpoint_tracker(project_root: Path) -> None:
    """The BestCheckpointTracker tracks two metrics independently and decides when to save."""
    import sys
    sys.path.insert(0, str(project_root))
    import train

    t = train.BestCheckpointTracker()
    assert t.is_new_best("edit_f1", 0.20)  # first measurement → new best
    assert t.is_new_best("strict_f1", 0.05)
    assert not t.is_new_best("edit_f1", 0.15)  # regression
    assert t.is_new_best("edit_f1", 0.25)  # improvement
    assert t.best_value("edit_f1") == 0.25
    assert t.best_value("strict_f1") == 0.05


def test_best_checkpoint_tracker_unseen_metric_returns_negative_inf(project_root: Path) -> None:
    """Querying best_value for a metric never observed returns -inf (not a crash)."""
    import sys
    sys.path.insert(0, str(project_root))
    import train

    t = train.BestCheckpointTracker()
    assert t.best_value("never_seen") == -float("inf")


def test_best_checkpoint_tracker_tie_is_not_new_best(project_root: Path) -> None:
    """A value equal to the best should not be reported as new best (strict >)."""
    import sys
    sys.path.insert(0, str(project_root))
    import train

    t = train.BestCheckpointTracker()
    assert t.is_new_best("m", 0.5)
    assert not t.is_new_best("m", 0.5)
