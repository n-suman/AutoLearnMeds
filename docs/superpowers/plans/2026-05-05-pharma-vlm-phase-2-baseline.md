# Phase 2 — Baseline Model + Training Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the full SigLIP-base + Donut-style decoder baseline in `train.py` (the single file the autoresearch agent edits), train it end-to-end on Colab A100 against `data/processed/train.jsonl`, and confirm `final_macro_f1` on val is above-random (>0.30) and reproducible to ±0.01 across 3 seeds. The model class implements `predict_text(batch_images, max_new_tokens) -> list[str]` so it slots into `prepare.evaluate()` unchanged.

**Architecture:** Frozen `google/siglip-base-patch16-224` produces 196 patch tokens (768-dim). A trainable linear projection maps 768 → 512. A 6-layer GPT-style decoder with RoPE positional encoding, causal self-attention, cross-attention to the 196 image tokens, and tied input/output embeddings predicts the canonical XML output sequence. Greedy decoding generates output strings at evaluation time. Teacher-forced cross-entropy loss for training. AdamW with cosine LR schedule, bf16 mixed precision, gradient clipping. W&B for tracking.

**Tech Stack:** PyTorch 2.x, Transformers (for SigLIP encoder + processor), tokenizers (BPE — already trained in Phase 1), accelerate, wandb. All ml extras already in `pyproject.toml`. The `tokenizer.json` and processed JSONLs from Phase 1 are reused unchanged.

**Spec:** [`docs/superpowers/specs/2026-05-05-pharma-vlm-autoresearch-design.md`](../specs/2026-05-05-pharma-vlm-autoresearch-design.md), §4 (Model), §1.3 (metric), §9.2 (Phase 2 exit criteria).

**Naming convention in this plan:** PyTorch's `Module.eval()` method is referred to and called as `.train(False)` throughout. The two are semantically identical (both put the module in inference mode, disabling dropout etc.). Using `.train(False)` keeps all training-mode toggles symmetric: `.train(True)` enables, `.train(False)` disables.

**Phase 1 prerequisites met:** `prepare.py` v1 frozen, `data/processed/{train,val,test}.jsonl` exist (564/111/162), `data/processed/tokenizer.json` is in git, `prepare.evaluate()` runs end-to-end with stub model on Colab. 60 lightweight + 8 colab-marked tests pass.

---

## File Structure

| Path | Created/Modified | Responsibility |
|---|---|---|
| `train.py` | **REWRITTEN from stub** | Single-file baseline: encoder + decoder + model + training loop. Agent-editable in autoresearch loop. |
| `tests/test_train_components.py` | Create | Lightweight unit tests for the few pure-logic helpers (config dataclass, seeding, torch-less import). |
| `tests/test_train_smoke.py` | Create | Colab-marked: instantiates the model, runs 1 forward + 1 backward + 1 step + 1 generation. Ensures shapes/types are wired correctly. |
| `tests/test_train_overfit.py` | Create | Colab-marked: trains for ~50 steps on 3 fixed records and asserts loss decreases. Catches gross architectural bugs. |
| `scripts/run_experiment.sh` | Create | Canonical single-experiment launcher used by autoresearch. Calls `python train.py` with config from a YAML. |
| `experiments/configs/baseline.yaml` | Create | Hyperparameters for the Phase 2 baseline run (not for the agent — these are the locked baseline). |

**`train.py` internal structure** (single file, sections delimited by `# === ... ===` banners):

```
# === Imports ===
# === Args dataclass ===
# === Seeding ===
# === Encoder (SigLIP wrapper, frozen) ===
# === Rotary position embedding ===
# === Decoder block (self-attn + cross-attn + FFN) ===
# === Decoder (stack of blocks, tied embeddings) ===
# === Model (encoder + projection + decoder + predict_text) ===
# === Tokenization helpers ===
# === Training loop ===
# === Main ===
```

The agent (in autoresearch) edits any of these sections except imports/main glue. Section banners stay (the agent reads them as scope markers).

---

## Task 1: train.py skeleton + dataclass config + seeding + main glue

**Files:**
- Modify: `train.py` (rewrite from stub)
- Create: `experiments/configs/baseline.yaml`
- Create: `experiments/configs/.gitignore`
- Create: `tests/test_train_components.py`

The first task lays down the structure: argparse → dataclass config → seed everything → empty `main` that prints `final_macro_f1=0.0000` so subsequent tasks can replace the body.

- [ ] **Step 1: Create `experiments/configs/.gitignore` and the baseline YAML**

Create `/Users/apple/AutoLearnMeds/experiments/configs/.gitignore`:

```
# Allow committed canonical configs.
!*.yaml
!.gitignore
```

Create `/Users/apple/AutoLearnMeds/experiments/configs/baseline.yaml`:

```yaml
# Phase 2 baseline. The autoresearch agent does NOT edit this — the agent
# edits train.py instead. This config is the "factory default".
seed: 42
encoder_model: "google/siglip-base-patch16-224"
hidden_dim: 512
n_decoder_layers: 6
n_decoder_heads: 8
ffn_ratio: 4
dropout: 0.1
tied_embeddings: true
max_target_length: 256
batch_size: 16
peak_lr: 3.0e-4
min_lr: 3.0e-5
warmup_steps: 100
max_steps: 1000
weight_decay: 0.05
adam_betas: [0.9, 0.95]
grad_clip: 1.0
eval_every: 200
log_every: 25
precision: "bf16"
train_jsonl: "data/processed/train.jsonl"
val_jsonl: "data/processed/val.jsonl"
images_root: "/mnt/gcs/raw/raw_images"
path_strip_prefix: "raw/raw_images/"
tokenizer_path: "data/processed/tokenizer.json"
checkpoint_dir: "checkpoints/runs/baseline"
wandb_project: "autolearnmeds"
wandb_mode: "online"
```

- [ ] **Step 2: Write the failing test for the config dataclass + seeding**

Create `/Users/apple/AutoLearnMeds/tests/test_train_components.py`:

```python
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
```

- [ ] **Step 3: Run, verify failures**

```bash
uv run pytest tests/test_train_components.py -v
```

Expected: 3 FAIL — `train.py` is still the NotImplementedError stub.

- [ ] **Step 4: Verify pyyaml is available**

```bash
uv run python -c "import yaml; print(yaml.__version__)"
```

Expected: prints a version. (`pyyaml>=6.0` is already in core deps.)

- [ ] **Step 5: Rewrite `train.py` with skeleton — config + seeding + main**

Use Write tool to overwrite `/Users/apple/AutoLearnMeds/train.py`:

```python
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
```

- [ ] **Step 6: Run, verify tests pass**

```bash
uv run pytest tests/test_train_components.py -v
```

Expected: all 3 PASSED.

- [ ] **Step 7: Run the full local suite**

```bash
uv run pytest -v 2>&1 | tail -3
```

Expected: 63 PASSED (60 prior + 3 new), 8 deselected.

- [ ] **Step 8: Commit**

```bash
git add train.py experiments/configs/ tests/test_train_components.py
git commit -m "feat(train): skeleton — Config dataclass + seeding + main stub

Replaces train.py's NotImplementedError stub with the agent-editable
Phase-2 skeleton: argparse, Config dataclass loaded from yaml, set_seed
that covers Python/numpy/torch/CUDA, and a main that prints
final_macro_f1=0.0000 so autoresearch parsers work end-to-end.

experiments/configs/baseline.yaml locks the baseline hyperparameters
the autoresearch agent starts from. The agent edits train.py defaults,
not this YAML.

Lazy ml imports throughout: torch is imported only inside set_seed,
numpy similarly. Module imports cleanly in the local Mac venv.

3 lightweight tests cover yaml load, seed determinism, and torch-less
import.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: SigLIP encoder wrapper (frozen) + projection

**Files:**
- Modify: `train.py` (append encoder section)
- Create: `tests/test_train_smoke.py` (Colab-marked)

The encoder is `google/siglip-base-patch16-224` from transformers, run with `requires_grad=False`.

- [ ] **Step 1: Write the failing test (Colab-marked)**

Create `/Users/apple/AutoLearnMeds/tests/test_train_smoke.py`:

```python
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
```

- [ ] **Step 2: Verify deselected locally**

```bash
uv run pytest tests/test_train_smoke.py -v
```

Expected: 2 deselected.

- [ ] **Step 3: Append the Encoder class to `train.py`**

Use Edit tool to append after the `set_seed` function:

```python


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
```

- [ ] **Step 4: Run, verify locally**

```bash
uv run pytest -v 2>&1 | tail -3
```

Expected: 63 PASSED, 10 deselected (8 prior + 2 new).

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_train_smoke.py
git commit -m "feat(train): SigLIP encoder wrapper, frozen

Encoder loads google/siglip-base-patch16-224's vision_model and
sets requires_grad=False on every parameter. Output is the model's
last_hidden_state — (B, 196, 768) for 224x224 inputs, with no CLS
token.

Lazy torch + transformers imports keep the module loadable on a
torch-less venv.

2 colab-marked tests cover the frozen requirement and the (B, 196, 768)
output shape.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Decoder block — RoPE + self-attention + cross-attention + FFN

**Files:**
- Modify: `train.py` (append decoder block section)
- Modify: `tests/test_train_smoke.py` (append shape tests)

The decoder block has: pre-LN, causal self-attention with RoPE, pre-LN, cross-attention to image tokens, pre-LN, FFN.

- [ ] **Step 1: Append failing tests for the decoder block**

Append to `/Users/apple/AutoLearnMeds/tests/test_train_smoke.py`:

```python


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
```

- [ ] **Step 2: Append the decoder block to `train.py`**

Use Edit tool to append after the `Encoder` class:

```python


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
```

- [ ] **Step 3: Verify locally**

```bash
uv run pytest -v 2>&1 | tail -3
```

Expected: 63 PASSED, 12 deselected.

- [ ] **Step 4: Commit**

```bash
git add train.py tests/test_train_smoke.py
git commit -m "feat(train): RoPE + DecoderBlock with self-attn + cross-attn + FFN

Pre-LN transformer decoder block:
- Self-attention with RoPE over the head_dim, causal mask, fused SDPA kernel.
- Cross-attention to image memory tokens (no positional encoding on memory —
  SigLIP patches are already spatially encoded).
- FFN: 2-layer MLP with GELU + dropout, ffn_ratio configurable.
- Residual connections around each sub-block.

RoPE helpers: rope_cache (precomputes cos/sin tables), apply_rope (rotates
adjacent dim pairs), causal_mask (lower-triangular bool).

2 new colab-marked tests cover RoPE shape preservation and full block
forward shape.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Decoder — stack of blocks + tied embeddings + output head

**Files:**
- Modify: `train.py` (append decoder section)
- Modify: `tests/test_train_smoke.py` (append decoder test)

- [ ] **Step 1: Append failing test**

Append to `/Users/apple/AutoLearnMeds/tests/test_train_smoke.py`:

```python


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
```

- [ ] **Step 2: Append the Decoder class to `train.py`**

Use Edit tool to append after the `DecoderBlock` class:

```python


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
```

- [ ] **Step 3: Verify locally**

```bash
uv run pytest -v 2>&1 | tail -3
```

Expected: 63 PASSED, 14 deselected.

- [ ] **Step 4: Commit**

```bash
git add train.py tests/test_train_smoke.py
git commit -m "feat(train): Decoder — stack of blocks + final LN + tied output head

Donut-style autoregressive decoder. Token embedding -> N DecoderBlocks
(self-attn + cross-attn + FFN, RoPE, pre-LN) -> final LN -> output
projection.

tied_embeddings=True (default per spec §4.2) shares the embedding
weight with the output projection — common for small-vocab decoders,
saves params, has a regularizing effect (Press & Wolf 2017,
papers/press_2017_tied_embeddings.pdf).

parameters() iterator deduplicates by id() so the tied weight isn't
double-counted by the optimizer.

2 new colab-marked tests cover (B, T, V) logits shape and the
tied-weight identity invariant.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Model wrapper — encoder + projection + decoder + greedy `predict_text`

**Files:**
- Modify: `train.py` (append model section + tokenization helpers)
- Modify: `tests/test_train_smoke.py` (append model + predict_text test)

- [ ] **Step 1: Append failing tests**

Append to `/Users/apple/AutoLearnMeds/tests/test_train_smoke.py`:

```python


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
```

- [ ] **Step 2: Append the Model class + tokenization helpers to `train.py`**

Use Edit tool to append after the `Decoder` class:

```python


# === Tokenization helpers ===

def encode_targets(tokenizer, target_texts: list[str], max_length: int):
    """Encode XML target strings to (input_ids, labels) tensors for teacher forcing.

    target_texts already include BOS and EOS (format_output wraps them).
    input_ids = ids[:-1] (teacher input).
    labels    = ids[1:]  (next-token target), with -100 where padded.
    """
    torch = _import_torch()
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        pad_id = 0

    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")

    encs = [tokenizer.encode(t).ids for t in target_texts]
    encs = [ids[: max_length] for ids in encs]
    encs = [ids if len(ids) >= 2 else [bos_id, eos_id] for ids in encs]

    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    max_t = 0
    for ids in encs:
        in_ids = ids[:-1]
        lab = ids[1:]
        input_ids.append(in_ids)
        labels.append(lab)
        max_t = max(max_t, len(in_ids))

    for i in range(len(input_ids)):
        pad_n = max_t - len(input_ids[i])
        input_ids[i] = input_ids[i] + [pad_id] * pad_n
        labels[i] = labels[i] + [-100] * pad_n

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


# === Model (encoder + projection + decoder + predict_text) ===

class PharmaVLM:
    """Frozen SigLIP encoder + projection + Donut-style decoder.

    Public methods:
      forward(images, target_ids) -> scalar loss (teacher-forced cross-entropy)
      predict_text(images, max_new_tokens) -> list[str] (greedy decoding)

    The autoresearch agent's predict_text is what plugs into prepare.evaluate().
    """

    def __init__(
        self,
        encoder_model: str,
        vocab_size: int,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        ffn_ratio: int,
        dropout: float,
        tied_embeddings: bool,
        tokenizer,
    ) -> None:
        torch = _import_torch()
        nn = torch.nn

        self.encoder = Encoder(encoder_model)
        encoder_hidden = self.encoder.model.config.hidden_size
        self.proj = nn.Linear(encoder_hidden, hidden_dim, bias=False)
        self.decoder = Decoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
            tied_embeddings=tied_embeddings,
        )
        self.tokenizer = tokenizer
        self.bos_id = tokenizer.token_to_id("<s>")
        self.eos_id = tokenizer.token_to_id("</s>")
        self._device = None

    def trainable_parameters(self):
        """Yields only parameters with requires_grad=True (proj + decoder)."""
        for p in self.proj.parameters():
            yield p
        for p in self.decoder.parameters():
            yield p

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.proj.parameters()
        yield from self.decoder.parameters()

    def to(self, device):
        self.encoder.to(device)
        self.proj.to(device)
        self.decoder.to(device)
        self._device = device
        return self

    def train(self, mode: bool = True):
        self.proj.train(mode)
        self.decoder.train(mode)
        self.encoder.train(False)  # encoder always in inference mode
        return self

    def _encode_images(self, images):
        feats = self.encoder(images)
        return self.proj(feats)

    def forward(self, images, target_ids):
        """Teacher-forced training step. target_ids: (B, T). Returns scalar loss."""
        torch = _import_torch()
        memory = self._encode_images(images)
        in_ids = target_ids[:, :-1].contiguous()
        lab_ids = target_ids[:, 1:].contiguous()
        logits = self.decoder(in_ids, memory)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            lab_ids.reshape(-1),
            ignore_index=-100,
        )
        return loss

    def predict_text(self, images, max_new_tokens: int = 256) -> list[str]:
        """Greedy autoregressive decoding. Returns one decoded string per image."""
        torch = _import_torch()
        self.train(False)
        memory = self._encode_images(images)
        B = images.size(0)
        cur = torch.full((B, 1), self.bos_id, dtype=torch.long, device=images.device)
        finished = torch.zeros(B, dtype=torch.bool, device=images.device)
        for _ in range(max_new_tokens):
            logits = self.decoder(cur, memory)
            next_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, next_ids], dim=1)
            finished |= next_ids.squeeze(-1) == self.eos_id
            if finished.all():
                break

        outputs: list[str] = []
        for row in cur.tolist():
            trimmed = []
            for tok_id in row:
                trimmed.append(tok_id)
                if tok_id == self.eos_id and len(trimmed) > 1:
                    break
            outputs.append(self.tokenizer.decode(trimmed, skip_special_tokens=False))
        return outputs
```

- [ ] **Step 3: Verify locally**

```bash
uv run pytest -v 2>&1 | tail -3
```

Expected: 63 PASSED, 16 deselected.

- [ ] **Step 4: Commit**

```bash
git add train.py tests/test_train_smoke.py
git commit -m "feat(train): PharmaVLM model + greedy predict_text + tokenization helpers

PharmaVLM wraps frozen SigLIP encoder + trainable projection
(encoder_hidden=768 -> hidden_dim) + trainable Donut-style decoder.

forward(images, target_ids) does teacher-forced cross-entropy on shifted
target_ids (input is target_ids[:, :-1], labels target_ids[:, 1:]).

predict_text(images, max_new_tokens) implements greedy autoregressive
decoding with EOS-stopping. Returns one string per image. The output
strings are exactly what prepare.evaluate consumes via parse_output.

trainable_parameters() yields only proj + decoder (the encoder is frozen).

encode_targets() builds (input_ids, labels) tensors with right-padding
(-100 in labels for ignore_index in CE loss). Used by the training loop.

2 new colab-marked tests cover the predict_text contract (returns list
of strings) and the forward-returns-loss invariant.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Training loop — optimizer, scheduler, eval-at-interval, W&B logging

**Files:**
- Modify: `train.py` (append training loop + replace main body)

- [ ] **Step 1: Append the training loop and replace main body in `train.py`**

Use Edit tool to append after the `PharmaVLM` class:

```python


# === Training loop ===

def cosine_with_warmup(step: int, warmup_steps: int, max_steps: int, peak_lr: float, min_lr: float) -> float:
    """Linear warmup over `warmup_steps`, then cosine decay to `min_lr`."""
    import math
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _infinite(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def train_loop(model, cfg: Config, device, wandb_run=None) -> dict[str, Any]:
    """Run cfg.max_steps of training. Returns the final metrics dict."""
    import prepare
    torch = _import_torch()

    train_dl = prepare.get_dataloader(
        jsonl_path=cfg.train_jsonl,
        images_root=cfg.images_root,
        path_strip_prefix=cfg.path_strip_prefix,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
    )

    optimizer = torch.optim.AdamW(
        list(model.trainable_parameters()),
        lr=cfg.peak_lr,
        betas=cfg.adam_betas,
        weight_decay=cfg.weight_decay,
    )

    autocast_dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float32

    trainable_params = sum(p.numel() for p in model.trainable_parameters())
    print(f"[train] trainable_params={trainable_params:,}")

    train_iter = _infinite(train_dl)
    last_metrics: dict[str, Any] = {"macro_f1": 0.0, "per_field_f1": {}, "n_examples": 0}
    best_macro_f1 = 0.0
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    for step in range(cfg.max_steps):
        model.train(True)
        batch = next(train_iter)
        images = batch["image"].to(device)

        in_ids, lab_ids = encode_targets(model.tokenizer, batch["target_text"], cfg.max_target_length)
        in_ids = in_ids.to(device)
        lab_ids = lab_ids.to(device)

        with torch.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            dtype=autocast_dtype,
            enabled=(cfg.precision == "bf16"),
        ):
            memory = model._encode_images(images)
            logits = model.decoder(in_ids, memory)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                lab_ids.reshape(-1),
                ignore_index=-100,
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), cfg.grad_clip)
        lr = cosine_with_warmup(step, cfg.warmup_steps, cfg.max_steps, cfg.peak_lr, cfg.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % cfg.log_every == 0:
            print(f"[train] step={step:5d} loss={loss.item():.4f} lr={lr:.2e}")
            if wandb_run is not None:
                wandb_run.log({"train/loss": loss.item(), "train/lr": lr, "step": step})

        if step > 0 and step % cfg.eval_every == 0:
            model.train(False)
            with torch.no_grad():
                metrics = prepare.evaluate(
                    model=model,
                    jsonl_path=cfg.val_jsonl,
                    images_root=cfg.images_root,
                    path_strip_prefix=cfg.path_strip_prefix,
                    batch_size=cfg.batch_size,
                    max_new_tokens=cfg.max_target_length,
                )
            print(f"[ val ] step={step:5d} macro_f1={metrics['macro_f1']:.4f} (n={metrics['n_examples']})")
            if wandb_run is not None:
                wandb_run.log({
                    "val/macro_f1": metrics["macro_f1"],
                    "val/n_examples": metrics["n_examples"],
                    **{f"val/per_field/{f}": v for f, v in metrics["per_field_f1"].items()},
                    "step": step,
                })
            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                ckpt = {"step": step, "macro_f1": best_macro_f1}
                torch.save(ckpt, Path(cfg.checkpoint_dir) / "best.meta.pt")
            last_metrics = metrics

    model.train(False)
    with torch.no_grad():
        last_metrics = prepare.evaluate(
            model=model,
            jsonl_path=cfg.val_jsonl,
            images_root=cfg.images_root,
            path_strip_prefix=cfg.path_strip_prefix,
            batch_size=cfg.batch_size,
            max_new_tokens=cfg.max_target_length,
        )
    print(f"[final] macro_f1={last_metrics['macro_f1']:.4f}")
    if wandb_run is not None:
        wandb_run.log({"final/macro_f1": last_metrics["macro_f1"]})
    return last_metrics
```

- [ ] **Step 2: Replace the main function body**

Use Edit tool. Find the existing `main` body (which currently just prints `final_macro_f1=0.0000`) and replace its body. The replacement:

```python
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/baseline.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    if args.seed is not None:
        cfg = dataclasses.replace(cfg, seed=args.seed)

    set_seed(cfg.seed)

    import prepare
    torch = _import_torch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] device={device} seed={cfg.seed}")

    tokenizer = prepare.get_tokenizer(cfg.tokenizer_path)
    model = PharmaVLM(
        encoder_model=cfg.encoder_model,
        vocab_size=tokenizer.get_vocab_size(),
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_decoder_layers,
        n_heads=cfg.n_decoder_heads,
        ffn_ratio=cfg.ffn_ratio,
        dropout=cfg.dropout,
        tied_embeddings=cfg.tied_embeddings,
        tokenizer=tokenizer,
    ).to(device)

    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                config=dataclasses.asdict(cfg),
                mode=cfg.wandb_mode,
            )
        except Exception as e:
            print(f"[main] WARN: wandb init failed ({e}); continuing without it")

    final_metrics = train_loop(model, cfg, device, wandb_run=wandb_run)

    if wandb_run is not None:
        wandb_run.finish()

    print(f"final_macro_f1={final_metrics['macro_f1']:.4f}")
    return 0
```

- [ ] **Step 3: Verify locally**

```bash
uv run pytest -v 2>&1 | tail -3
```

Expected: 63 PASSED, 16 deselected.

- [ ] **Step 4: Commit**

```bash
git add train.py
git commit -m "feat(train): training loop + main with W&B + cosine LR + eval-at-interval

train_loop runs cfg.max_steps iterations:
- AdamW with cfg.adam_betas + cfg.weight_decay over trainable_parameters
  (frozen encoder excluded).
- Cosine LR schedule with linear warmup (warmup_steps -> peak_lr -> min_lr).
- bf16 autocast when cfg.precision == 'bf16'.
- Grad clip at cfg.grad_clip.
- Every cfg.log_every: print loss + lr, log to W&B.
- Every cfg.eval_every: run prepare.evaluate on val, log macro_f1 +
  per-field F1 to W&B, save best checkpoint metadata.
- Final eval after max_steps; print 'final_macro_f1=X.XXXX' on stdout
  (the literal token the autoresearch harness parses).

main wires up: argparse -> Config.from_yaml -> set_seed -> tokenizer ->
model -> W&B init -> train_loop -> wandb.finish.

--no-wandb flag for offline / no-telemetry runs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Local smoke test — overfit on 5 examples

**Files:**
- Create: `tests/test_train_overfit.py` (Colab-marked)

This is a Colab-only integration test that catches gross architectural bugs by training the model for ~50 steps on a tiny dataset (the 5 synthetic fixtures) and asserting loss decreases.

- [ ] **Step 1: Create the failing test**

Create `/Users/apple/AutoLearnMeds/tests/test_train_overfit.py`:

```python
"""Colab-marked overfitting test: trains for 50 steps on the 5 synthetic
fixtures and asserts loss decreases by at least 50%. Catches gross architectural
bugs (wrong loss reduction, broken backward graph, dead gradients, etc.).
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


def test_overfit_on_synthetic_fixtures(
    train_mod,
    project_root: Path,
    synthetic_jsonl_dir: Path,
    synthetic_images_dir: Path,
    tiny_processed_dir: Path,
) -> None:
    import sys
    sys.path.insert(0, str(project_root / "scripts"))
    import build_processed
    build_processed.build_processed(
        gold_standard_path=synthetic_jsonl_dir / "gold_standard.jsonl",
        splits_path=synthetic_jsonl_dir / "splits.json",
        images_root=synthetic_images_dir,
        out_dir=tiny_processed_dir,
        image_path_prefix="raw/raw_images",
    )

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dl = prepare.get_dataloader(
        jsonl_path=tiny_processed_dir / "train.jsonl",
        images_root=synthetic_images_dir,
        path_strip_prefix="raw/raw_images/",
        batch_size=3,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(list(model.trainable_parameters()), lr=1e-3)
    train_iter = train_mod._infinite(dl)
    losses: list[float] = []
    for step in range(50):
        model.train(True)
        batch = next(train_iter)
        images = batch["image"].to(device).float()
        in_ids, lab_ids = train_mod.encode_targets(tok, batch["target_text"], 256)
        in_ids = in_ids.to(device)
        lab_ids = lab_ids.to(device)
        memory = model._encode_images(images)
        logits = model.decoder(in_ids, memory)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            lab_ids.reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    initial = sum(losses[:5]) / 5
    final = sum(losses[-5:]) / 5
    print(f"[overfit_test] initial_loss={initial:.4f} final_loss={final:.4f}")
    assert final < 0.5 * initial, (
        f"Expected loss to drop >50% in 50 steps on 5 synthetic examples, "
        f"got {initial:.4f} -> {final:.4f}"
    )
```

- [ ] **Step 2: Verify deselected locally**

```bash
uv run pytest tests/test_train_overfit.py -v
```

Expected: 1 deselected.

- [ ] **Step 3: Commit**

```bash
git add tests/test_train_overfit.py
git commit -m "test(train): overfit-on-synthetic-fixtures Colab integration test

Trains the model for 50 steps on the 5-record synthetic fixtures and
asserts loss drops by >50%. Catches gross architectural bugs:
- wrong loss reduction (mean vs sum, scalar shape)
- broken backward graph (detached tensors)
- dead gradients (zero learning rate, frozen-by-mistake)
- shape mismatches that didn't show in the unit tests

Colab-marked. Run via 'pytest -m colab' from /workspace on the A100.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: `scripts/run_experiment.sh` — canonical experiment launcher

**Files:**
- Create: `scripts/run_experiment.sh`

The autoresearch loop's per-experiment lifecycle starts here. The agent calls this script after editing `train.py`. It runs train.py, captures stdout/stderr, parses `final_macro_f1`, and writes a per-run directory.

- [ ] **Step 1: Create the script**

Create `/Users/apple/AutoLearnMeds/scripts/run_experiment.sh`:

```bash
#!/usr/bin/env bash
# Canonical autoresearch experiment runner.
#
# Usage:
#   bash scripts/run_experiment.sh [run_id] [--config experiments/configs/baseline.yaml] [--seed N]
#
# What it does:
#   1. Snapshots the current train.py into experiments/runs/<run_id>/train.py
#   2. Records the config used into experiments/runs/<run_id>/config.yaml
#   3. Runs `python train.py --config <cfg> [--seed N]`, capturing stdout/stderr
#      to experiments/runs/<run_id>/stdout.log
#   4. Parses the final `final_macro_f1=X.XXXX` line and writes
#      experiments/runs/<run_id>/metrics.json with {final_macro_f1, run_id,
#      git_sha, exit_code, wall_clock_seconds}.

set -euo pipefail

RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%S)-$(printf '%04x' $RANDOM)}"
shift || true

CONFIG="experiments/configs/baseline.yaml"
SEED_ARG=""
EXTRA_ARGS=()
while (($#)); do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --seed) SEED_ARG="--seed $2"; shift 2;;
    *) EXTRA_ARGS+=("$1"); shift;;
  esac
done

RUN_DIR="experiments/runs/${RUN_ID}"
mkdir -p "$RUN_DIR"

echo "[run_experiment] run_id=$RUN_ID config=$CONFIG ${SEED_ARG}"

cp train.py "$RUN_DIR/train.py"
cp "$CONFIG" "$RUN_DIR/config.yaml"

GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
START_TS=$(date -u +%s)

set +e
uv run python train.py --config "$CONFIG" $SEED_ARG "${EXTRA_ARGS[@]}" \
    > "$RUN_DIR/stdout.log" 2>&1
EXIT_CODE=$?
set -e

END_TS=$(date -u +%s)
WALL_CLOCK=$((END_TS - START_TS))

FINAL_F1="$(grep -oE 'final_macro_f1=[0-9.]+' "$RUN_DIR/stdout.log" | tail -1 | cut -d= -f2 || echo '-1.0')"
[[ -z "$FINAL_F1" ]] && FINAL_F1="-1.0"

cat > "$RUN_DIR/metrics.json" <<EOF
{
  "run_id": "$RUN_ID",
  "git_sha": "$GIT_SHA",
  "config": "$CONFIG",
  "final_macro_f1": $FINAL_F1,
  "wall_clock_seconds": $WALL_CLOCK,
  "exit_code": $EXIT_CODE
}
EOF

echo "[run_experiment] DONE final_macro_f1=$FINAL_F1 wall_clock=${WALL_CLOCK}s exit=$EXIT_CODE"
echo "[run_experiment] artifacts at $RUN_DIR/"
exit $EXIT_CODE
```

- [ ] **Step 2: Make executable + verify bash syntax**

```bash
chmod +x /Users/apple/AutoLearnMeds/scripts/run_experiment.sh
bash -n /Users/apple/AutoLearnMeds/scripts/run_experiment.sh && echo OK
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_experiment.sh
git commit -m "feat(scripts): run_experiment.sh — canonical autoresearch launcher

Runs one training experiment and writes a self-contained run directory
under experiments/runs/<run_id>/:
- train.py: snapshot of the script the agent just edited (reproducibility)
- config.yaml: snapshot of the YAML config used
- stdout.log: full captured output
- metrics.json: parsed {run_id, git_sha, config, final_macro_f1,
  wall_clock_seconds, exit_code}

The agent calls this after every edit to train.py. The harness reads
metrics.json to decide keep-or-discard.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Phase 2 exit gate on Colab

**Files:** none — orchestration only.

Run on Colab via the SSH tunnel (the controller's responsibility, not a fresh subagent's).

- [ ] **Step 1: Push commits**

```bash
cd /Users/apple/AutoLearnMeds && git push origin phase-0-plumbing
```

- [ ] **Step 2: Pull on Colab + run colab-marked tests**

```bash
SSHPASS="$AUTOLEARNMEDS_SSH_PASSWORD" sshpass -e ssh -o ConnectTimeout=20 autolearnmeds-colab "
set -e
cd /workspace
git pull --ff-only origin phase-0-plumbing
uv sync --extra ml --extra colab 2>&1 | tail -3
echo '--- run colab-marked tests ---'
uv run pytest -m colab 2>&1 | tail -8
"
```

Expected: 12+ colab tests pass (8 prior + 4-5 new from train.py: encoder freeze, encoder shape, decoder shape, model predict_text, overfit drop). The overfit test takes ~30-60 seconds.

- [ ] **Step 3: Run the baseline experiment (single seed, ~1 hour)**

```bash
SSHPASS="$AUTOLEARNMEDS_SSH_PASSWORD" sshpass -e ssh -o ConnectTimeout=120 autolearnmeds-colab "
set -e
cd /workspace
echo '--- baseline run, seed=42 ---'
bash scripts/run_experiment.sh baseline-seed42 --config experiments/configs/baseline.yaml --seed 42
"
```

Expected: ~30-60 min runtime; `final_macro_f1` printed; `experiments/runs/baseline-seed42/metrics.json` written. **Phase 2 success criterion #1: macro_f1 > 0.30.**

- [ ] **Step 4: Run two more seeds for reproducibility check**

```bash
SSHPASS="$AUTOLEARNMEDS_SSH_PASSWORD" sshpass -e ssh -o ConnectTimeout=120 autolearnmeds-colab "
set -e
cd /workspace
echo '--- baseline run, seed=43 ---'
bash scripts/run_experiment.sh baseline-seed43 --config experiments/configs/baseline.yaml --seed 43
echo '--- baseline run, seed=44 ---'
bash scripts/run_experiment.sh baseline-seed44 --config experiments/configs/baseline.yaml --seed 44
"
```

Expected: two additional runs; macro_f1 within ±0.01 of seed-42. **Phase 2 success criterion #2: reproducibility ±0.01 across 3 seeds.**

- [ ] **Step 5: Read all three metrics.json from Colab and consolidate**

Local from Mac:

```bash
SSHPASS="$AUTOLEARNMEDS_SSH_PASSWORD" sshpass -e ssh -o ConnectTimeout=20 autolearnmeds-colab "
cd /workspace
for s in 42 43 44; do
  echo \"=== seed=\$s ===\"
  cat experiments/runs/baseline-seed\${s}/metrics.json
  echo ''
done
"
```

Expected: three metrics.json files; all `final_macro_f1` values within ±0.01.

- [ ] **Step 6: Write Phase 2 review**

Create `/Users/apple/AutoLearnMeds/docs/superpowers/reviews/phase2_review.md`:

```markdown
# Phase 2 Review — Baseline Model + Training Loop

**Tag:** `phase-2-complete`
**Date:** YYYY-MM-DD (fill at commit)
**Status:** GREEN — baseline trains end-to-end on Colab A100 with macro_f1 > 0.30 reproducible to ±0.01 across 3 seeds.

## What was built

- `train.py` — single-file SigLIP+Donut baseline (the agent-editable file).
- `experiments/configs/baseline.yaml` — locked baseline hyperparameters.
- `scripts/run_experiment.sh` — canonical autoresearch launcher.
- 9 colab-marked tests + 3 lightweight tests.

## What was verified

- 63 lightweight pytest tests pass locally.
- All colab-marked tests pass on A100.
- baseline-seed42: macro_f1 = X.XXXX (>0.30).
- baseline-seed43: macro_f1 = X.XXXX (within ±0.01).
- baseline-seed44: macro_f1 = X.XXXX (within ±0.01).

## Open items

- Tied embeddings + small vocab (1853 tokens, plateaued in Phase 1) means the embedding matrix is tiny. Decoder dominates parameter count.
- Greedy decoding only — beam search is a Phase-7 sweep candidate.
- No augmentation in baseline — explore-phase will add RandAugment / AugMix.

## Next step

Phase 3 — Disaster-recovery drill (per spec §9.3).
```

- [ ] **Step 7: Tag and push**

```bash
cd /Users/apple/AutoLearnMeds
git add docs/superpowers/reviews/phase2_review.md
git commit -m "docs: phase 2 review

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
git tag -a phase-2-complete -m "Phase 2 baseline + training loop complete"
git push origin phase-0-plumbing phase-2-complete
```

---

## Self-Review

**Spec coverage check** (against spec §4 Model + §9.2 Phase 2):

| Spec requirement | Implemented in task |
|---|---|
| §4.1 SigLIP-base frozen encoder | T2 |
| §4.2 6-layer decoder, hidden 512, 8 heads, ffn_ratio 4, GELU | T3, T4 |
| §4.2 RoPE positional encoding | T3 |
| §4.2 Cross-attention every layer | T3 |
| §4.2 Tied input/output embeddings | T4 |
| §4.2 Pre-LN | T3 |
| §4.4 Cross-entropy loss, teacher-forced | T5, T6 |
| §4.5 AdamW betas, wd, peak_lr, min_lr | T1, T6 |
| §4.5 Cosine + linear warmup | T6 |
| §4.5 Grad clip 1.0 | T6 |
| §4.5 Batch 16, bf16 | T1, T6 |
| §1.3 Print `final_macro_f1=X.XXXX` last line | T1 stub, T6 real |
| §9.2 macro_f1 > random (>~0.30) | T9 verifies |
| §9.2 reproducible ±0.01 across 3 seeds | T9 verifies |

All requirements covered.

**Placeholder scan:** the Phase-2-review template uses `YYYY-MM-DD` and `X.XXXX` — these are runtime fills (date, actual macro_f1 from the run), not plan placeholders.

**Type consistency:**
- `Config` fields used in `train_loop` match the dataclass definition.
- `Encoder`, `DecoderBlock`, `Decoder`, `PharmaVLM` interface (`__call__` shapes, `parameters()`, `to(device)`, `train(mode)`) consistent across tasks.
- `predict_text(images, max_new_tokens) -> list[str]` interface matches what `prepare.evaluate()` calls in Phase 1.
- `encode_targets(tokenizer, texts, max_length) -> (input_ids, labels)` consistent between Task 5 (definition) and Task 6 (use in train_loop) and Task 7 (use in overfit test).

---

## Phase 2 → Phase 3 transition

When this plan is fully executed and tagged `phase-2-complete`:

1. `train.py` is the canonical agent-editable file with all sections.
2. `experiments/configs/baseline.yaml` is the locked hyperparameter source.
3. `scripts/run_experiment.sh` is the canonical experiment launcher.
4. The model trains end-to-end; macro_f1 > 0.30 reproducible across 3 seeds.
5. Phase 3 plan implements the disaster-recovery drill per spec §9.3.
