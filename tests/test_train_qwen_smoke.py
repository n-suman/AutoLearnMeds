"""Track B smoke tests — require Colab GPU + ml extras (torch + transformers + peft + bnb).

Marked @pytest.mark.colab so they're skipped by default (`-m 'not colab'` in pytest.ini).
"""
from __future__ import annotations

import pytest


@pytest.mark.colab
def test_load_qwen_with_lora_returns_model(project_root):
    """Loads Qwen2-VL-2B with 4-bit + LoRA r=8 and confirms trainable param count is in range."""
    import sys
    sys.path.insert(0, str(project_root))
    import train_qwen

    cfg = train_qwen.QwenConfig.from_yaml(project_root / "experiments/configs/qwen_baseline.yaml")
    bundle = train_qwen.load_qwen_with_lora(cfg)
    assert "model" in bundle and "processor" in bundle

    trainable = sum(p.numel() for p in bundle["model"].parameters() if p.requires_grad)
    total = sum(p.numel() for p in bundle["model"].parameters())
    # Qwen2-VL-2B is ~2.2B params; LoRA r=8 on q/k/v/o_proj is typically a few million.
    assert 100_000 < trainable, f"trainable too low: {trainable:,}"
    assert trainable < total / 100, f"trainable too high: {trainable:,} / {total:,}"


@pytest.mark.colab
def test_qwen_predict_text_returns_strings(project_root):
    """End-to-end: load Qwen+LoRA, call predict_text on a 2-image batch, get list[str] of len 2."""
    import sys

    import torch

    sys.path.insert(0, str(project_root))
    import train_qwen

    cfg = train_qwen.QwenConfig.from_yaml(project_root / "experiments/configs/qwen_baseline.yaml")
    bundle = train_qwen.load_qwen_with_lora(cfg)
    wrapper = train_qwen.QwenWrapper(bundle["model"], bundle["processor"], cfg)

    # 2-image batch of random SigLIP-normalized noise (range [-1, 1]).
    fake_images = torch.empty(2, 3, 224, 224).uniform_(-1, 1)
    out = wrapper.predict_text(fake_images, max_new_tokens=16)
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(s, str) for s in out)
