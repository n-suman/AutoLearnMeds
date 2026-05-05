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
