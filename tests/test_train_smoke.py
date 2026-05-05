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


def test_apply_rope_preserves_shape(train_mod) -> None:
    """RoPE rotates the last dim in pairs; shape is preserved."""
    import torch
    head_dim = 64
    seq = 32
    q = torch.randn(2, 4, seq, head_dim)  # (B, H, T, D)
    cos, sin = train_mod.rope_cache(seq, head_dim, device=q.device, dtype=q.dtype)
    q_rot = train_mod.apply_rope(q, cos, sin)
    assert q_rot.shape == q.shape


def test_decoder_block_forward_shape(train_mod) -> None:
    """DecoderBlock returns (B, T, hidden) given (B, T, hidden) and (B, S, hidden) memory."""
    import torch
    block = train_mod.DecoderBlock(hidden_dim=128, n_heads=4, ffn_ratio=4, dropout=0.0)
    x = torch.randn(2, 7, 128)
    mem = torch.randn(2, 196, 128)
    causal = train_mod.causal_mask(7, x.device)
    out = block(x, mem, causal)
    assert out.shape == (2, 7, 128)


def test_decoder_forward_shape(train_mod) -> None:
    """Decoder(token_ids, memory) returns (B, T, vocab_size) logits."""
    import torch
    vocab_size = 256
    decoder = train_mod.Decoder(
        vocab_size=vocab_size,
        hidden_dim=128,
        n_layers=2,
        n_heads=4,
        ffn_ratio=4,
        dropout=0.0,
        tied_embeddings=True,
    )
    token_ids = torch.randint(0, vocab_size, (2, 7))
    memory = torch.randn(2, 196, 128)
    logits = decoder(token_ids, memory)
    assert logits.shape == (2, 7, vocab_size)


def test_decoder_tied_embeddings_share_weight(train_mod) -> None:
    """When tied_embeddings=True, output projection weight IS the embedding weight."""
    decoder = train_mod.Decoder(
        vocab_size=128, hidden_dim=64, n_layers=2, n_heads=4,
        ffn_ratio=4, dropout=0.0, tied_embeddings=True,
    )
    assert decoder.embedding.weight is decoder.output_proj_weight


def test_pharma_vlm_predict_text_returns_strings(train_mod, project_root: Path) -> None:
    """An untrained model still produces strings of length B (one per image)."""
    import torch

    import prepare
    tok = prepare.get_tokenizer(project_root / "data" / "processed" / "tokenizer.json")
    model = train_mod.PharmaVLM(
        encoder_model="google/siglip-base-patch16-224",
        vocab_size=tok.get_vocab_size(),
        hidden_dim=128,
        n_layers=2,
        n_heads=4,
        ffn_ratio=4,
        dropout=0.0,
        tied_embeddings=True,
        tokenizer=tok,
    )
    model.train(False)
    x = torch.randn(2, 3, 224, 224)
    out = model.predict_text(x, max_new_tokens=16)
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(s, str) for s in out)


def test_pharma_vlm_forward_returns_loss(train_mod, project_root: Path) -> None:
    """forward(images, target_ids) returns a scalar loss tensor."""
    import torch

    import prepare
    tok = prepare.get_tokenizer(project_root / "data" / "processed" / "tokenizer.json")
    model = train_mod.PharmaVLM(
        encoder_model="google/siglip-base-patch16-224",
        vocab_size=tok.get_vocab_size(),
        hidden_dim=128,
        n_layers=2,
        n_heads=4,
        ffn_ratio=4,
        dropout=0.0,
        tied_embeddings=True,
        tokenizer=tok,
    )
    images = torch.randn(2, 3, 224, 224)
    target_ids = torch.randint(0, tok.get_vocab_size(), (2, 10))
    loss = model.forward(images, target_ids)
    assert loss.ndim == 0
    assert loss.item() > 0
