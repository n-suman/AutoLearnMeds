"""Colab-marked smoke tests for train.py — instantiate components and run small forward passes.

Skipped on local Mac (default pytest config). Run on Colab via `pytest -m colab`.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.colab


@pytest.fixture
def train_mod():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if "train" in sys.modules:
        importlib.reload(sys.modules["train"])
    import train
    return train


def test_encoder_is_frozen(train_mod) -> None:
    """Encoder.parameters() returns parameters with requires_grad=False."""
    enc = train_mod.Encoder("google/siglip-base-patch16-224")
    assert all(not p.requires_grad for p in enc.parameters())


def test_encoder_forward_shape(train_mod) -> None:
    """Encoder(batch_images) returns (B, 196, 768) for 224x224 inputs."""
    import torch
    enc = train_mod.Encoder("google/siglip-base-patch16-224")
    enc.train(False)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = enc(x)
    assert out.shape == (2, 196, 768)
