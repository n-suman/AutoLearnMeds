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
