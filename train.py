"""Phase-2 module: SigLIP encoder + Donut-style decoder + training loop.

This file is the ONE the autoresearch agent edits. Outputs the literal line
`final_macro_f1=X.XXXX` on stdout as its last line so the harness can parse it.

Section banners (# === Name ===) demarcate scopes the agent should respect:
they stay; the code inside is fair game.
"""
from __future__ import annotations

# === Imports ===
import argparse
import dataclasses
import os
import random
import sys
from pathlib import Path
from typing import Any


# === Args dataclass ===

@dataclasses.dataclass
class Config:
    """Hyperparameters for one training run.

    Loaded from a YAML in experiments/configs/. The autoresearch agent
    rewrites these defaults inside train.py when it iterates; the YAML
    file is the canonical baseline checkpoint.
    """
    seed: int = 42
    encoder_model: str = "google/siglip-base-patch16-224"
    hidden_dim: int = 512
    n_decoder_layers: int = 6
    n_decoder_heads: int = 8
    ffn_ratio: int = 4
    dropout: float = 0.1
    tied_embeddings: bool = True
    max_target_length: int = 256
    batch_size: int = 16
    peak_lr: float = 3.0e-4
    min_lr: float = 3.0e-5
    warmup_steps: int = 100
    max_steps: int = 1000
    weight_decay: float = 0.05
    adam_betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    eval_every: int = 200
    log_every: int = 25
    precision: str = "bf16"
    train_jsonl: str = "data/processed/train.jsonl"
    val_jsonl: str = "data/processed/val.jsonl"
    images_root: str = "/mnt/gcs/raw/raw_images"
    path_strip_prefix: str = "raw/raw_images/"
    tokenizer_path: str = "data/processed/tokenizer.json"
    checkpoint_dir: str = "checkpoints/runs/baseline"
    wandb_project: str = "autolearnmeds"
    wandb_mode: str = "online"

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Config":
        import yaml
        data = yaml.safe_load(Path(path).read_text())
        if isinstance(data.get("adam_betas"), list):
            data["adam_betas"] = tuple(data["adam_betas"])
        return cls(**data)


# === Seeding ===

def set_seed(seed: int) -> None:
    """Seed Python, numpy, torch (if available), and CUDA RNGs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# === Encoder (SigLIP wrapper, frozen) ===

def _import_torch():
    """Lazy torch import. Returns the torch module."""
    import torch
    return torch


class Encoder:
    """SigLIP-base-patch16-224 vision encoder, FROZEN.

    Agent-editable: model name, normalization mean/std (via processor),
    whether to freeze. Default: pretrained, frozen, no [CLS] (only patch tokens).
    """

    def __init__(self, model_name: str = "google/siglip-base-patch16-224") -> None:
        from transformers import AutoModel
        torch = _import_torch()
        self._torch = torch
        self.model = AutoModel.from_pretrained(model_name).vision_model
        for p in self.model.parameters():
            p.requires_grad_(False)

    def parameters(self):
        return self.model.parameters()

    def to(self, device):
        self.model.to(device)
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def __call__(self, pixel_values):
        """pixel_values: (B, 3, H, W). Returns (B, 196, 768) patch features."""
        torch = self._torch
        with torch.no_grad():
            out = self.model(pixel_values=pixel_values, output_hidden_states=False)
        return out.last_hidden_state


# === Main ===

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/baseline.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed from the config",
    )
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    if args.seed is not None:
        cfg = dataclasses.replace(cfg, seed=args.seed)

    set_seed(cfg.seed)

    final_macro_f1 = 0.0
    print(f"final_macro_f1={final_macro_f1:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
