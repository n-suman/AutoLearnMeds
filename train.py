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
import json
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
    encoder_init_path: str = ""  # If set, load SigLIP weights from this path instead of HF hub.
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

    # --- Track E flags (defaults preserve Track A behavior) ---
    image_size: int = 224                          # 384 for Track E
    pseudo_jsonl: str = ""                         # path to data/pseudo_labels/round_001.jsonl
    pseudo_weight: float = 0.0                     # 0.0 = ignore pseudo even if jsonl given
    pseudo_weight_mode: str = "const"              # "const" | "adaptive_confidence"
    pseudo_min_confidence: str = "medium"          # "low" | "medium" | "high"
    track_e_tokenizer_path: str = ""               # if set, use this instead of cfg.tokenizer_path
    stage_schedule: list = dataclasses.field(default_factory=list)  # [{name, steps, data, lr_mult}, ...]; empty = single-stage
    compute_calibration: bool = False              # if True, eval also emits per-field entropy

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

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        encoder_init_path: str = "",
    ) -> None:
        from transformers import AutoModel, SiglipVisionModel
        torch = _import_torch()
        self._torch = torch
        # Phase 7: if encoder_init_path is set, load SigLIP weights from that
        # local path (e.g. an MAE-pretrained checkpoint dir). Otherwise pull
        # the stock pretrained encoder from the HF hub.
        #
        # Two cases:
        #   (a) stock "google/siglip-base-patch16-224": full SiglipModel
        #       config (with both vision + text towers); we want only the
        #       vision tower, hence the .vision_model attribute access.
        #   (b) MAE-pretrained checkpoint from pretrain_mae.py's
        #       encoder.save_pretrained(...): config is vision-only
        #       (model_type=siglip_vision_model). AutoModel.from_pretrained
        #       returns a SiglipVisionModel directly — no .vision_model attr.
        # Detect via config.model_type and dispatch to the right loader.
        encoder_init = encoder_init_path if encoder_init_path else model_name
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(encoder_init)
        if getattr(cfg, "model_type", "") == "siglip_vision_model":
            self.model = SiglipVisionModel.from_pretrained(encoder_init)
        else:
            self.model = AutoModel.from_pretrained(encoder_init).vision_model
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


def build_encoder(cfg: "Config"):
    """Construct the (frozen) vision encoder per cfg.

    Returns (encoder, hidden_dim) where:
      - encoder is an Encoder instance wrapping a frozen SiglipVisionModel
      - hidden_dim is encoder.model.config.hidden_size (768 for base, 1024 for large)

    Behavior:
    - If cfg.encoder_init_path is set, loads SigLIP weights from that local path
      (Track C / Phase 7 MAE-pretrained encoder). The Encoder class handles the
      model_type dispatch (siglip_vision_model vs full SiglipModel).
    - Otherwise loads from HuggingFace hub via cfg.encoder_model.
    - Always freezes all encoder parameters (enforced inside Encoder.__init__).
    """
    encoder = Encoder(cfg.encoder_model, encoder_init_path=cfg.encoder_init_path)
    hidden_dim = encoder.model.config.hidden_size
    return encoder, hidden_dim


# === Pre-flight checks ===

def preflight_leak_check(cfg: Config) -> None:
    """Refuse to train if pseudo_jsonl contains any image_path also in val or test.

    No-op when pseudo is disabled (cfg.pseudo_jsonl == "" or cfg.pseudo_weight == 0.0).
    """
    if not cfg.pseudo_jsonl or cfg.pseudo_weight == 0.0:
        return

    def _paths_from_jsonl(path: str) -> set[str]:
        out: set[str] = set()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                out.add(row.get("image_path", ""))
        return out

    val_paths = _paths_from_jsonl(cfg.val_jsonl) if cfg.val_jsonl else set()
    pseudo_paths = _paths_from_jsonl(cfg.pseudo_jsonl)

    overlap = val_paths & pseudo_paths
    assert not overlap, f"pseudo leak: {len(overlap)} val images found in pseudo_jsonl: {sorted(overlap)[:3]}..."

    # Test-set leak check is gated by file existence — test_jsonl is conventionally val.jsonl path with "test" substituted
    test_path = cfg.val_jsonl.replace("val.jsonl", "test.jsonl") if cfg.val_jsonl else ""
    if test_path and Path(test_path).exists():
        test_paths = _paths_from_jsonl(test_path)
        test_overlap = test_paths & pseudo_paths
        assert not test_overlap, f"PSEUDO LEAK (TEST): {len(test_overlap)} test images in pseudo_jsonl"


# === Combined training rows builder ===

def build_combined_train_rows(cfg: Config) -> tuple[list[dict], list[float]]:
    """Return (rows, per_row_weights) for the training loop.

    - Always includes all rows from cfg.train_jsonl with weight 1.0 and `is_pseudo: False`.
    - If cfg.pseudo_jsonl is non-empty AND cfg.pseudo_weight > 0:
      reads pseudo, runs prepare.normalize_pseudo_xml + filter_pseudo_rows
      against val/test, appends survivors with the configured weight and `is_pseudo: True`.

    The `is_pseudo` field on each row is used by StageScheduler (Task D4) to filter
    examples by stage (gold-only vs pseudo-only).
    """
    import prepare

    def _read(path: str) -> list[dict]:
        rows: list[dict] = []
        if not path:
            return rows
        p = Path(path)
        if not p.exists():
            return rows
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    gold = _read(cfg.train_jsonl)
    # Annotate gold rows
    for r in gold:
        r["is_pseudo"] = False
    rows = list(gold)
    weights = [1.0] * len(gold)

    if cfg.pseudo_jsonl and cfg.pseudo_weight > 0.0:
        pseudo_raw = _read(cfg.pseudo_jsonl)
        pseudo = [prepare.normalize_pseudo_xml(r) for r in pseudo_raw]
        val_paths = {r["image_path"] for r in _read(cfg.val_jsonl)} if cfg.val_jsonl else set()
        test_path = cfg.val_jsonl.replace("val.jsonl", "test.jsonl") if cfg.val_jsonl else ""
        test_paths = {r["image_path"] for r in _read(test_path)} if (test_path and Path(test_path).exists()) else set()
        kept, dropped = prepare.filter_pseudo_rows(pseudo, val_paths=val_paths, test_paths=test_paths)
        print(f"[train] pseudo: kept={len(kept)} dropped={dropped}", flush=True)
        # Annotate pseudo rows
        for r in kept:
            r["is_pseudo"] = True
        rows.extend(kept)

        if cfg.pseudo_weight_mode == "adaptive_confidence":
            # weight = avg per-field confidence * cfg.pseudo_weight
            conf_map = {"low": 0.3, "medium": 0.7, "high": 1.0}
            for row in kept:
                conf = row.get("per_field_confidence", {})
                avg = sum(conf_map.get(v, 0.5) for v in conf.values()) / max(1, len(conf))
                weights.append(cfg.pseudo_weight * avg)
        else:  # const
            weights.extend([cfg.pseudo_weight] * len(kept))

    return rows, weights


# === Stage scheduler ===

@dataclasses.dataclass(frozen=True)
class Stage:
    name: str
    steps: int
    data: str       # "all" | "gold" | "pseudo"
    lr_mult: float


class StageScheduler:
    """Maps a global step → (current stage, lr_mult, data filter).

    Used by the trainer's main loop to drive multi-stage Track E training.
    With cfg.stage_schedule=[], behaves as a single 'baseline' stage covering
    cfg.max_steps with data='all' and lr_mult=1.0 (Track A-compatible default).
    """

    def __init__(self, cfg: Config):
        if not cfg.stage_schedule:
            self._stages = [Stage(name="baseline", steps=cfg.max_steps, data="all", lr_mult=1.0)]
        else:
            self._stages = [Stage(**s) for s in cfg.stage_schedule]
        # Precompute cumulative step boundaries
        self._cum: list[int] = []
        acc = 0
        for s in self._stages:
            acc += s.steps
            self._cum.append(acc)

    def stage_at_step(self, step: int) -> Stage:
        for i, boundary in enumerate(self._cum):
            if step < boundary:
                return self._stages[i]
        # past last stage: return the last (allows for max_steps overshoot tolerance)
        return self._stages[-1]

    def total_steps(self) -> int:
        return self._cum[-1]

    def example_mask_at_step(self, step: int, rows: list[dict]) -> list[bool]:
        """Return a bool list parallel to `rows`: True = keep this example in this stage."""
        s = self.stage_at_step(step)
        if s.data == "all":
            return [True] * len(rows)
        if s.data == "gold":
            return [not r.get("is_pseudo", False) for r in rows]
        if s.data == "pseudo":
            return [r.get("is_pseudo", False) for r in rows]
        raise ValueError(f"Unknown stage data: {s.data}")


class BestCheckpointTracker:
    """Per-metric best tracker. Independent tracking lets dual checkpoints diverge.

    Usage:
        tracker = BestCheckpointTracker()
        if tracker.is_new_best("edit_f1", current_edit):
            save_checkpoint(model, path / "best_edit_f1.pt")
        if tracker.is_new_best("strict_f1", current_strict):
            save_checkpoint(model, path / "best_strict_f1.pt")

    A "new best" is strict improvement (>). Ties don't trigger a save.
    First-ever measurement of any metric always counts as new best (compared to -inf).
    """

    def __init__(self):
        self._best: dict[str, float] = {}

    def is_new_best(self, metric: str, value: float) -> bool:
        prev = self._best.get(metric, -float("inf"))
        if value > prev:
            self._best[metric] = value
            return True
        return False

    def best_value(self, metric: str) -> float:
        return self._best.get(metric, -float("inf"))


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
        encoder_init_path: str = "",
    ) -> None:
        torch = _import_torch()
        nn = torch.nn

        self.encoder = Encoder(encoder_model, encoder_init_path=encoder_init_path)
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
        """Greedy autoregressive decoding. Returns one decoded string per image.

        Auto-moves images to the model's device — prepare.evaluate() passes
        CPU tensors from the DataLoader and doesn't know the model's device.
        """
        torch = _import_torch()
        self.train(False)
        # Infer device from the trainable projection layer; move images to match.
        model_device = next(self.proj.parameters()).device
        if images.device != model_device:
            images = images.to(model_device)
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


def self_named_params(module) -> list[tuple[str, Any]]:
    """Yield (name, tensor) for every parameter in `module`. Works for both
    nn.Modules (uses .named_parameters()) and our DecoderBlock pseudo-module
    (synthesizes names from _modules + their attributes).
    """
    if hasattr(module, "named_parameters") and callable(module.named_parameters):
        return list(module.named_parameters())
    # DecoderBlock case: walk _modules with stable names per spec.
    out: list[tuple[str, Any]] = []
    name_map = {
        id(module.ln1): "ln1", id(module.q_proj): "q_proj",
        id(module.k_proj): "k_proj", id(module.v_proj): "v_proj",
        id(module.o_proj): "o_proj", id(module.ln2): "ln2",
        id(module.xq_proj): "xq_proj", id(module.xk_proj): "xk_proj",
        id(module.xv_proj): "xv_proj", id(module.xo_proj): "xo_proj",
        id(module.ln3): "ln3", id(module.fc1): "fc1", id(module.fc2): "fc2",
    }
    for m in module._modules:
        prefix = name_map.get(id(m), m.__class__.__name__.lower())
        for pname, p in m.named_parameters() if hasattr(m, "named_parameters") else []:
            out.append((f"{prefix}.{pname}", p))
    return out


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
                # Full state dict for proj + decoder (encoder is frozen — re-load
                # from HF on resume). Saved as a regular torch dict so confirm-
                # phase / paper-prep can load even from a different machine.
                full_ckpt = {
                    "step": step,
                    "macro_f1": best_macro_f1,
                    "proj_state_dict": model.proj.state_dict(),
                    "decoder_blocks_state_dicts": [
                        {n: p.detach().cpu() for n, p in self_named_params(b)}
                        for b in model.decoder.blocks
                    ],
                    "decoder_embedding_state_dict": model.decoder.embedding.state_dict(),
                    "decoder_final_ln_state_dict": model.decoder.final_ln.state_dict(),
                    "config": dataclasses.asdict(cfg),
                }
                torch.save(full_ckpt, Path(cfg.checkpoint_dir) / "best.pt")
                # Also keep the lightweight meta for backward compatibility
                torch.save(
                    {"step": step, "macro_f1": best_macro_f1},
                    Path(cfg.checkpoint_dir) / "best.meta.pt",
                )
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


# === Main ===

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
    preflight_leak_check(cfg)

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
        encoder_init_path=cfg.encoder_init_path,
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

    # Primary metric (autoresearch optimization target).
    print(f"final_macro_f1={final_metrics['macro_f1']:.4f}", flush=True)
    # Secondary metric (paper-grade lenient F1; partial credit via edit distance).
    print(f"final_macro_edit_f1={final_metrics.get('macro_edit_f1', 0.0):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
