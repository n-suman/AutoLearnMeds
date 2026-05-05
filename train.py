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


# === Rotary position embedding ===

def rope_cache(seq_len: int, head_dim: int, device, dtype):
    """Precompute RoPE cos/sin tables. head_dim must be even."""
    torch = _import_torch()
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE; got {head_dim}")
    half = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1).to(dtype=dtype)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1).to(dtype=dtype)
    return cos, sin


def apply_rope(x, cos, sin):
    """Apply RoPE to (B, H, T, D) tensor x. cos, sin: (T, D)."""
    torch = _import_torch()
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack([-x2, x1], dim=-1).flatten(-2)
    cos_b = cos[None, None, :, :]
    sin_b = sin[None, None, :, :]
    return x * cos_b + x_rot * sin_b


def causal_mask(seq_len: int, device):
    """Returns a (T, T) bool mask where True = "may attend"."""
    torch = _import_torch()
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


# === Decoder block (self-attn + cross-attn + FFN) ===

class DecoderBlock:
    """Pre-LN transformer decoder block with RoPE self-attn + cross-attn + FFN.

    Agent-editable: head count, FFN ratio, activation, layer-norm placement
    (currently pre-LN), dropout, attention pattern.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        torch = _import_torch()
        nn = torch.nn

        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} not divisible by n_heads {n_heads}")
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.dropout = dropout

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.ln2 = nn.LayerNorm(hidden_dim)
        self.xq_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.xk_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.xv_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.xo_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * ffn_ratio)
        self.fc2 = nn.Linear(hidden_dim * ffn_ratio, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        self._modules = [
            self.ln1, self.q_proj, self.k_proj, self.v_proj, self.o_proj,
            self.ln2, self.xq_proj, self.xk_proj, self.xv_proj, self.xo_proj,
            self.ln3, self.fc1, self.fc2, self.act, self.drop,
        ]

    def parameters(self):
        for m in self._modules:
            yield from m.parameters()

    def to(self, device):
        for m in self._modules:
            m.to(device)
        return self

    def train(self, mode: bool = True):
        for m in self._modules:
            if hasattr(m, "train"):
                m.train(mode)
        return self

    def __call__(self, x, memory, causal):
        """x: (B, T, H), memory: (B, S, H). Returns (B, T, H)."""
        torch = _import_torch()
        B, T, H = x.shape
        S = memory.shape[1]

        h = self.ln1(x)
        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        cos, sin = rope_cache(T, self.head_dim, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        attn_mask = causal.view(1, 1, T, T)
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.fc1.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, T, H)
        x = x + self.o_proj(attn_out)

        h = self.ln2(x)
        xq = self.xq_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        xk = self.xk_proj(memory).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        xv = self.xv_proj(memory).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        xattn_out = torch.nn.functional.scaled_dot_product_attention(
            xq, xk, xv,
            dropout_p=self.dropout if self.fc1.training else 0.0,
        )
        xattn_out = xattn_out.transpose(1, 2).reshape(B, T, H)
        x = x + self.xo_proj(xattn_out)

        h = self.ln3(x)
        h = self.fc2(self.drop(self.act(self.fc1(h))))
        x = x + h
        return x


# === Decoder (stack of blocks, tied embeddings) ===

class Decoder:
    """Donut-style autoregressive decoder.

    Token embeddings -> N x DecoderBlock -> final LayerNorm -> output projection
    (optionally tied to the input embedding weight).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        ffn_ratio: int,
        dropout: float,
        tied_embeddings: bool,
    ) -> None:
        torch = _import_torch()
        nn = torch.nn

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.blocks = [
            DecoderBlock(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ]
        self.final_ln = nn.LayerNorm(hidden_dim)

        if tied_embeddings:
            self.output_proj_weight = self.embedding.weight
            self.output_proj_bias = None
        else:
            self.output_proj_weight = nn.Parameter(torch.empty(vocab_size, hidden_dim))
            nn.init.normal_(self.output_proj_weight, mean=0.0, std=0.02)
            self.output_proj_bias = nn.Parameter(torch.zeros(vocab_size))

    def parameters(self):
        seen: set[int] = set()
        for m in (self.embedding, self.final_ln, *self.blocks):
            for p in m.parameters():
                if id(p) in seen:
                    continue
                seen.add(id(p))
                yield p
        if self.output_proj_weight is not self.embedding.weight:
            yield self.output_proj_weight
            if self.output_proj_bias is not None:
                yield self.output_proj_bias

    def to(self, device):
        self.embedding.to(device)
        self.final_ln.to(device)
        for b in self.blocks:
            b.to(device)
        if self.output_proj_weight is not self.embedding.weight:
            self.output_proj_weight = self.output_proj_weight.to(device)
            if self.output_proj_bias is not None:
                self.output_proj_bias = self.output_proj_bias.to(device)
        return self

    def train(self, mode: bool = True):
        self.embedding.train(mode)
        self.final_ln.train(mode)
        for b in self.blocks:
            b.train(mode)
        return self

    def __call__(self, token_ids, memory):
        """token_ids: (B, T) int64, memory: (B, S, H) float. Returns (B, T, V) logits."""
        torch = _import_torch()
        B, T = token_ids.shape
        x = self.embedding(token_ids)
        mask = causal_mask(T, x.device)
        for block in self.blocks:
            x = block(x, memory, mask)
        x = self.final_ln(x)
        logits = torch.nn.functional.linear(x, self.output_proj_weight, self.output_proj_bias)
        return logits


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
