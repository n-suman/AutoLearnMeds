"""Phase-7 MAE smoke tests — require Colab GPU + ml extras (torch + transformers + torchvision).

Marked @pytest.mark.colab so they're skipped by default (`-m 'not colab'` in pytest.ini).
"""
from __future__ import annotations

import pytest


@pytest.mark.colab
def test_build_mae_model_runs_forward(project_root):
    """Build model, run a single masked forward, check decoder output shape."""
    import sys

    import torch

    sys.path.insert(0, str(project_root))
    import pretrain_mae

    cfg = pretrain_mae.MAEConfig.from_yaml(
        project_root / "experiments/configs/mae_pretrain.yaml"
    )
    bundle = pretrain_mae.build_mae_model(cfg)

    # Sanity: encoder can be moved to CPU; decoder accepts decoder_dim tensors.
    fake_imgs = torch.zeros(2, 3, cfg.image_size, cfg.image_size)
    with torch.no_grad():
        enc_out = bundle["encoder"](pixel_values=fake_imgs).last_hidden_state
    assert enc_out.shape == (2, cfg.num_patches, bundle["encoder"].config.hidden_size)

    visible, mask, ids_restore = pretrain_mae.random_masking(enc_out, cfg.mask_ratio)
    visible_proj = bundle["decoder"].proj(visible)
    B, L_visible, D = visible_proj.shape
    num_masked = cfg.num_patches - L_visible
    mask_tokens = bundle["decoder"].mask_token.expand(B, num_masked, -1)
    full = torch.cat([visible_proj, mask_tokens], dim=1)
    full = torch.gather(
        full,
        dim=1,
        index=ids_restore.unsqueeze(-1).expand(-1, -1, full.size(-1)),
    )

    pred = bundle["decoder"](full)
    assert pred.shape == (2, cfg.num_patches, cfg.patch_size ** 2 * 3)
