"""Phase 7 — MAE continue-pretraining of SigLIP-base on pharma images.

Sibling of train.py / train_qwen.py — single-file trainer, agent-editable,
prints final_pretrain_loss=<float> on stdout for the autoresearch harness
to parse.

Heavy ML imports (torch / transformers / torchvision / PIL) are deliberately
deferred to inside the functions that use them, so this module imports cleanly
in the Mac dev venv (no ml extras) for unit tests and config-load smoke tests.
"""
from __future__ import annotations

# === Imports (lightweight only — heavy ml imports go inside functions) ===
import argparse
import dataclasses
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import yaml


# === Config dataclass ===

@dataclasses.dataclass
class MAEConfig:
    """Hyperparameters for one MAE pretraining run.

    Loaded from a YAML in experiments/configs/. Mirrors the schema of
    experiments/configs/mae_pretrain.yaml exactly. Field order matches
    that YAML's logical grouping: identity, image preprocessing, MAE,
    training, augmentation, I/O.
    """
    phase: int
    run_kind: str
    encoder_init: str
    image_size: int
    patch_size: int
    num_patches: int
    mask_ratio: float
    decoder_depth: int
    decoder_dim: int
    decoder_heads: int
    norm_pix_loss: bool
    seed: int
    batch_size: int
    grad_accum_steps: int
    peak_lr: float
    min_lr: float
    warmup_epochs: int
    total_epochs: int
    weight_decay: float
    adam_betas: tuple[float, float]
    grad_clip: float
    log_every: int
    precision: str
    random_resized_crop_scale: tuple[float, float]
    random_horizontal_flip: float
    images_root: str
    checkpoint_dir: str
    wandb_project: str
    wandb_mode: str
    # Optional: jsonl files whose `image_path` entries should be EXCLUDED from
    # the pretraining pool. Used to keep val/test image PIXELS out of MAE
    # pretraining for a cleaner paper claim ("zero leakage at any level").
    exclude_jsonls: tuple[str, ...] = ()
    # Phase 7b — text-aware (edge-density-weighted) masking. When True the
    # dataset computes a per-patch Sobel-edge density per image and the
    # mask sampler in random_masking() biases mask probability toward
    # high-edge ("text/graphic") patches. Default False keeps Phase 7
    # uniform-random behavior.
    text_aware_masking: bool = False
    text_mask_rate: float = 0.90
    bg_mask_rate: float = 0.60
    # Phase 7b — TAPT support. When non-empty, build_pretrain_dataset filters
    # the image pool to ONLY images whose basename appears in any include
    # jsonl (inverse of exclude_jsonls). When both include and exclude are
    # set, include is applied first then exclude.
    include_jsonls: tuple[str, ...] = ()
    # Phase 7c — resumable checkpointing (F8a). Default 10 (was effectively 50
    # via yaml override) to bound worst-case interruption loss to ~5 minutes
    # of A100 time. Existing yamls (mae_pretrain.yaml, mae_pretrain_text_aware.yaml,
    # mae_pretrain_tapt.yaml) all set this explicitly, so they keep their
    # current cadence; new configs that omit the field inherit 10.
    save_every_epoch: int = 10
    # Phase 7c — when True, every per-epoch save also mirrors to
    # {checkpoint_dir}/latest/ so resume-on-launch finds it. Disable for
    # tiny/test runs where the mirror would be wasted disk I/O.
    save_to_latest: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MAEConfig":
        data = yaml.safe_load(Path(path).read_text())
        if isinstance(data.get("adam_betas"), list):
            data["adam_betas"] = tuple(data["adam_betas"])
        if isinstance(data.get("random_resized_crop_scale"), list):
            data["random_resized_crop_scale"] = tuple(data["random_resized_crop_scale"])
        if isinstance(data.get("exclude_jsonls"), list):
            data["exclude_jsonls"] = tuple(data["exclude_jsonls"])
        if isinstance(data.get("include_jsonls"), list):
            data["include_jsonls"] = tuple(data["include_jsonls"])
        return cls(**data)


# === Seeding ===

def set_seed(seed: int) -> None:
    """Seed Python, numpy, torch (if installed) for reproducibility.

    Mirrors train.py:set_seed — numpy and torch imports are lazy so this
    function is safe to call from a torch-less venv.
    """
    random.seed(seed)
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


# === MAE model: SigLIP encoder + ViT decoder + mask token ===

def build_mae_model(cfg: "MAEConfig") -> dict[str, Any]:
    """Build the MAE model bundle.

    Encoder: SigLIP-base vision tower (initialized from cfg.encoder_init).
        SigLIP-base does NOT prepend a [CLS] token, so encoder.last_hidden_state
        is shape (B, num_patches, hidden_size) directly.
    Decoder: 4-layer ViT (per cfg.decoder_depth) operating in `decoder_dim`
        space. Includes a linear projection from encoder_hidden → decoder_dim,
        learned positional embeddings, transformer encoder stack, final
        LayerNorm, and a pixel-prediction head (decoder_dim → patch_size**2 * 3).
    mask_token: learnable (1, 1, decoder_dim) parameter inserted at masked
        positions before unshuffling.

    Returns: {"encoder": ..., "decoder": ...}. (decoder.mask_token is registered
    as a Parameter so .to(device) and .parameters() handle it correctly.)
    """
    import torch
    import torch.nn as nn
    from transformers import SiglipVisionModel

    encoder = SiglipVisionModel.from_pretrained(cfg.encoder_init)
    encoder_hidden = encoder.config.hidden_size

    class ViTDecoder(nn.Module):
        def __init__(self, encoder_hidden: int, cfg: "MAEConfig") -> None:
            super().__init__()
            self.proj = nn.Linear(encoder_hidden, cfg.decoder_dim)
            self.pos_embed = nn.Parameter(torch.zeros(1, cfg.num_patches, cfg.decoder_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            # mask_token registered as a Parameter on the decoder so .to(device)
            # moves it AND .parameters() includes it AND it stays a leaf tensor
            # (passing a .to(device)'d non-Module Parameter to AdamW fails with
            # "can't optimize a non-leaf Tensor" — the .to() call detaches it).
            self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder_dim))
            nn.init.trunc_normal_(self.mask_token, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.decoder_dim,
                nhead=cfg.decoder_heads,
                dim_feedforward=4 * cfg.decoder_dim,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.decoder_depth)
            self.norm = nn.LayerNorm(cfg.decoder_dim)
            self.pred = nn.Linear(cfg.decoder_dim, cfg.patch_size ** 2 * 3)

        def forward(self, x):
            x = x + self.pos_embed
            x = self.transformer(x)
            x = self.norm(x)
            return self.pred(x)

    decoder = ViTDecoder(encoder_hidden, cfg)

    return {"encoder": encoder, "decoder": decoder}


def patchify(images, patch_size: int):
    """(B, 3, H, W) -> (B, num_patches, patch_size**2 * 3).

    Reshape pattern: B C h p w p -> B (h w) (p p C).
    """
    import torch  # noqa: F401  (ensure torch is imported when called)
    p = patch_size
    B, C, H, W = images.shape
    h, w = H // p, W // p
    x = images.reshape(B, C, h, p, w, p)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    x = x.reshape(B, h * w, p * p * C)
    return x


def compute_edge_density_per_patch(
    image_path: "Path",
    image_size: int,
    patch_size: int,
):
    """Returns (num_patches,) float32 ndarray of mean Sobel edge magnitude per patch.

    High edge magnitude => text / graphic boundaries / informative content.
    Low edge magnitude => uniform background, blurred regions.

    Used by text-aware MAE masking (Phase 7b) to bias toward masking
    informative patches more aggressively.
    """
    import cv2
    import numpy as np
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        # corrupt/missing file — return uniform "informative" for safety
        n = (image_size // patch_size) ** 2
        return np.full(n, 0.5, dtype=np.float32)
    img = cv2.resize(img, (image_size, image_size))
    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.sqrt(sobelx ** 2 + sobely ** 2)
    h = image_size // patch_size
    # Reshape (H, W) -> (h, patch_size, h, patch_size) -> mean per patch.
    edges = edges.reshape(h, patch_size, h, patch_size).transpose(0, 2, 1, 3)
    per_patch = edges.mean(axis=(2, 3))  # (h, h)
    return per_patch.flatten().astype(np.float32)


def random_masking(x, mask_ratio: float, weights=None):
    """Random masking per He et al. 2021, with optional per-patch weights.

    Args:
        x: (B, L, D) token features.
        mask_ratio: target average mask ratio across the batch.
        weights: optional (B, L) per-patch weights in [0, 1]. Higher weight
            => higher P(mask). If None, fall back to uniform random
            (original He 2021 recipe).

    Returns:
        visible_x: (B, L_keep, D) the kept tokens (in shuffle order).
        mask: (B, L) float, 1 at masked positions and 0 at kept positions
              (in original order — already unshuffled).
        ids_restore: (B, L) inverse permutation to undo the shuffle on a
              full sequence (visible + mask_tokens).
    """
    import torch
    B, L, D = x.shape
    keep = int(L * (1 - mask_ratio))
    if weights is None:
        # Original He 2021 recipe — uniform Bernoulli via argsort of uniform noise.
        noise = torch.rand(B, L, device=x.device)
    else:
        # Weighted: ids_shuffle[:keep] = LOWEST-noise positions become VISIBLE
        # (kept). So to bias high-weight patches toward being MASKED, give
        # them HIGH noise. Formula: noise = weights + (1-weights)*uniform
        # gives: weight=1 → noise=1 (max → masked), weight=0 → noise~U(0,1)
        # (regular random). Preserves uniform-random fallback when all
        # weights are zero, and respects the documented semantic: higher
        # weight => higher P(mask).
        w = weights.to(x.device)
        noise = w + (1.0 - w) * torch.rand(B, L, device=x.device)

    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :keep]
    visible_x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
    mask = torch.ones(B, L, device=x.device)
    mask[:, :keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return visible_x, mask, ids_restore


# === Dataset ===

def _basenames_from_jsonls(jsonl_paths: tuple[str, ...]) -> set[str]:
    """Read each JSONL and return the union of `image_path` basenames.

    Tolerant of malformed lines / missing files: WARN-only, skip silently.
    """
    names: set[str] = set()
    for jsonl_path in jsonl_paths:
        jp = Path(jsonl_path)
        if not jp.is_file():
            print(f"[mae] WARN: jsonl {jp} not found; skipping", flush=True)
            continue
        for line in jp.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            img_path = entry.get("image_path") or entry.get("image") or ""
            if img_path:
                names.add(Path(img_path).name)
    return names


def build_pretrain_dataset(cfg: "MAEConfig") -> list[Path]:
    """List image files under cfg.images_root, applying include/exclude jsonl filters.

    No labels needed — MAE self-supervised pretraining only consumes pixels.

    Filter pipeline (applied in order):
      1. cfg.include_jsonls (Phase 7b TAPT): when non-empty, KEEP ONLY images
         whose basename appears in any include jsonl. This is the inverse of
         exclude — used by TAPT to restrict pretraining to the labeled
         train.jsonl images.
      2. cfg.exclude_jsonls (Phase 7 strict-paper): drop images whose basename
         appears in any exclude jsonl. Used to keep val/test pixels out of
         MAE pretraining for the "zero leakage at any level" claim.

    Both filters use JSONL with `image_path` entries — same format as
    data/processed/{train,val,test}.jsonl.
    """
    root = Path(cfg.images_root)
    paths = sorted(p for p in root.glob("**/*") if p.suffix.lower() in {".jpeg", ".jpg", ".png"})

    include_jsonls = getattr(cfg, "include_jsonls", ()) or ()
    if include_jsonls:
        included = _basenames_from_jsonls(tuple(include_jsonls))
        if included:
            before = len(paths)
            paths = [p for p in paths if p.name in included]
            print(f"[mae] included {len(paths)}/{before} images via {len(include_jsonls)} include jsonl(s)", flush=True)

    if cfg.exclude_jsonls:
        excluded_basenames = _basenames_from_jsonls(tuple(cfg.exclude_jsonls))
        if excluded_basenames:
            before = len(paths)
            paths = [p for p in paths if p.name not in excluded_basenames]
            print(f"[mae] excluded {before - len(paths)} images via {len(cfg.exclude_jsonls)} jsonl(s); pool={len(paths)}", flush=True)

    return paths


# === Phase 7c — resumable checkpointing helpers (F8a) ===

def _save_full_state(epoch_dir, encoder, decoder, optimizer, step: int, last_loss: float, cfg: "MAEConfig") -> None:
    """Save encoder (HF format) + decoder weights + optimizer state + step + last_loss + cfg snapshot.

    The encoder is written via HuggingFace `save_pretrained` (config.json +
    model.safetensors) so the result is loadable with
    `SiglipVisionModel.from_pretrained(epoch_dir)`. Everything else lives
    in `trainer_state.pt` next to the encoder files.

    Module-level (not nested) so unit tests can call it with a stand-in
    encoder/decoder/optimizer.
    """
    import torch
    epoch_dir = Path(epoch_dir)
    epoch_dir.mkdir(parents=True, exist_ok=True)
    encoder.save_pretrained(epoch_dir)
    torch.save({
        "decoder_state_dict": decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "last_loss": last_loss,
        "cfg": dataclasses.asdict(cfg),
    }, epoch_dir / "trainer_state.pt")


def _try_push_checkpoint_to_gcs(cfg: "MAEConfig", latest_dir, epoch_dir) -> None:
    """Best-effort gsutil rsync of checkpoint dirs to GCS.

    Failures are swallowed silently — the persistent sync_to_gcs daemon
    will retry on its own cadence. Bucket is configurable via the
    AUTOLEARNMEDS_GCS_BUCKET env var (defaults to gs://auto_learn_meds).

    No-ops if gsutil isn't on PATH and isn't at the standard Colab location.
    """
    import os
    import shutil
    import subprocess
    bucket = os.environ.get("AUTOLEARNMEDS_GCS_BUCKET", "gs://auto_learn_meds")
    gsutil = shutil.which("gsutil") or "/tools/google-cloud-sdk/bin/gsutil"
    if not Path(gsutil).is_file():
        return
    for src in (epoch_dir, latest_dir):
        src = Path(src)
        if not src.is_dir():
            continue
        dst = f"{bucket}/{src}"
        try:
            subprocess.run(
                [gsutil, "-m", "rsync", "-r", str(src), dst],
                check=False, capture_output=True, timeout=120,
            )
        except Exception:
            pass  # silent — sync_to_gcs daemon will retry


# === Training loop ===

def pretrain_mae_loop(bundle: dict, cfg: "MAEConfig", wandb_run) -> dict:
    """MAE pretraining: mask 75% of patches, reconstruct masked patches.

    Heavy imports (torch, torchvision, PIL) are deferred to here. Algorithm:
    1. Encoder runs on the FULL image (all 196 patches).
    2. Mask 75% in token space via random_masking(...).
    3. Project visible tokens to decoder_dim, concat with learned mask_tokens
       at the masked positions, unshuffle via ids_restore, decode.
    4. Pixel-MSE loss on masked positions only (per-patch normalization
       enabled by cfg.norm_pix_loss).
    5. AdamW + linear warmup (cfg.warmup_epochs) + cosine decay to cfg.min_lr.
    6. Save encoder via encoder.save_pretrained(...) every cfg.save_every_epoch
       epochs and once at the end (final/).

    Returns {"final_loss": float}.
    """
    import torch
    from PIL import Image
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = bundle["encoder"].to(device).train()
    decoder = bundle["decoder"].to(device).train()
    # decoder.mask_token is a registered Parameter; decoder.to(device) moved it.

    # Phase 7c — resumable checkpointing (F8a). Look for a previous run's
    # latest/ mirror and reload encoder + decoder + optimizer state if found.
    # Optimizer state is loaded LATER (after the optimizer is constructed
    # from cfg.peak_lr/betas/etc.); we stash it in `_resume_optim_state`.
    start_step = 1
    last_loss = 0.0
    _resume_optim_state = None
    latest_dir = Path(cfg.checkpoint_dir) / "latest"
    state_path = latest_dir / "trainer_state.pt"
    if state_path.is_file():
        print(f"[mae] RESUME: found {state_path}; loading...", flush=True)
        from transformers import SiglipVisionModel
        encoder = SiglipVisionModel.from_pretrained(latest_dir).to(device).train()
        bundle["encoder"] = encoder  # downstream code reads via local `encoder`, but keep bundle consistent

        state = torch.load(state_path, map_location=device, weights_only=False)
        decoder.load_state_dict(state["decoder_state_dict"])
        _resume_optim_state = state["optimizer_state_dict"]
        start_step = int(state["step"]) + 1
        last_loss = float(state.get("last_loss", 0.0))
        print(
            f"[mae] RESUME: starting from step={start_step} last_loss={last_loss:.4f}",
            flush=True,
        )

    image_paths = build_pretrain_dataset(cfg)
    print(f"[mae] dataset n={len(image_paths)} (using all images, no label needed)", flush=True)

    aug = transforms.Compose([
        transforms.RandomResizedCrop(cfg.image_size, scale=tuple(cfg.random_resized_crop_scale)),
        transforms.RandomHorizontalFlip(cfg.random_horizontal_flip),
        transforms.ToTensor(),
        # SigLIP normalization (matches train.py preprocessor):
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Use a proper DataLoader with workers for parallel image decode + augmentation —
    # without this the GPU sits at 0% util while a single Python thread bottlenecks
    # on JPEG decode + torchvision resize. With 4 workers + persistent_workers the
    # GPU stays well-fed (50-90% util on A100 for SigLIP-base + small ViT decoder).
    from torch.utils.data import Dataset, DataLoader

    # Phase 7b — for text-aware masking we precompute per-patch edge density
    # for every image once, store in a NumPy memmap, and serve from it in
    # the dataset. ~10-15s init for 2787 images at 224x224. Eliminates
    # per-step CPU edge-detect cost.
    edge_cache_array = None
    if cfg.text_aware_masking:
        import numpy as np
        ckpt_dir = Path(cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        edge_cache_path = ckpt_dir / "edge_cache.npy"
        n_images = len(image_paths)
        if edge_cache_path.is_file():
            edge_cache_array = np.load(edge_cache_path, mmap_mode="r")
            if edge_cache_array.shape[0] != n_images or edge_cache_array.shape[1] != cfg.num_patches:
                print(f"[mae] edge_cache shape mismatch (got {edge_cache_array.shape}, expected ({n_images}, {cfg.num_patches})); recomputing", flush=True)
                edge_cache_array = None
            else:
                print(f"[mae] loaded edge_cache from {edge_cache_path} shape={edge_cache_array.shape}", flush=True)
        if edge_cache_array is None:
            print(f"[mae] computing edge_cache for {n_images} images...", flush=True)
            t0 = time.time()
            arr = np.zeros((n_images, cfg.num_patches), dtype=np.float32)
            for i, p in enumerate(image_paths):
                arr[i] = compute_edge_density_per_patch(p, cfg.image_size, cfg.patch_size)
            np.save(edge_cache_path, arr)
            edge_cache_array = np.load(edge_cache_path, mmap_mode="r")
            print(f"[mae] edge_cache saved to {edge_cache_path} ({time.time()-t0:.1f}s)", flush=True)

    class _ImagePathDataset(Dataset):
        def __init__(self, paths: list, transform, cfg, edge_cache=None) -> None:
            self.paths = paths
            self.transform = transform
            self.cfg = cfg
            self.edge_cache = edge_cache  # NumPy memmap (N, num_patches) or None

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, idx: int):
            img = self.transform(Image.open(self.paths[idx]).convert("RGB"))
            if self.cfg.text_aware_masking and self.edge_cache is not None:
                # Convert memmap row to a regular tensor (avoids mmap pickling issues
                # across DataLoader worker boundary).
                import torch
                w = torch.from_numpy(self.edge_cache[idx].copy())
                return img, w
            return img

    pretrain_ds = _ImagePathDataset(image_paths, aug, cfg, edge_cache=edge_cache_array)
    loader = DataLoader(
        pretrain_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    # decoder.parameters() now includes decoder.mask_token (registered as Parameter)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optim = torch.optim.AdamW(
        params,
        lr=cfg.peak_lr,
        betas=tuple(cfg.adam_betas),
        weight_decay=cfg.weight_decay,
    )
    # Phase 7c — resume optimizer state if a checkpoint was loaded above.
    # AdamW state (running m/v moments, step counters per param) is what we
    # care about; loading it preserves momentum so we don't waste warmup.
    if _resume_optim_state is not None:
        optim.load_state_dict(_resume_optim_state)
        print(f"[mae] RESUME: optimizer state loaded ({len(optim.state)} param entries)", flush=True)

    # Each loader iteration yields one batch; with grad_accum we consume grad_accum_steps
    # batches per optimizer step.
    batches_per_epoch = len(loader)  # already accounts for drop_last
    steps_per_epoch = batches_per_epoch // cfg.grad_accum_steps
    total_steps = steps_per_epoch * cfg.total_epochs
    print(f"[mae] batches_per_epoch={batches_per_epoch} steps_per_epoch={steps_per_epoch} total_steps={total_steps}", flush=True)

    def lr_at(step: int, steps_per_epoch: int) -> float:
        epoch = step / steps_per_epoch
        if epoch < cfg.warmup_epochs:
            return cfg.peak_lr * (epoch / max(1, cfg.warmup_epochs))
        progress = (epoch - cfg.warmup_epochs) / max(1, cfg.total_epochs - cfg.warmup_epochs)
        progress = min(1.0, max(0.0, progress))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.min_lr + (cfg.peak_lr - cfg.min_lr) * cos

    # `last_loss` may have been seeded from a resumed checkpoint above; only
    # overwrite if we did NOT resume.
    if _resume_optim_state is None:
        last_loss = 0.0
    start = time.time()
    loader_iter = iter(loader)

    for step in range(start_step, total_steps + 1):
        optim.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for _ in range(cfg.grad_accum_steps):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)  # next epoch
                batch = next(loader_iter)
            if cfg.text_aware_masking:
                imgs, batch_weights = batch
                batch_weights = batch_weights.to(device, non_blocking=True)
            else:
                imgs = batch
                batch_weights = None
            imgs = imgs.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16):
                target = patchify(imgs, cfg.patch_size)  # (B, L, p*p*3)
                if cfg.norm_pix_loss:
                    mean = target.mean(dim=-1, keepdim=True)
                    var = target.var(dim=-1, keepdim=True, unbiased=False)
                    target = (target - mean) / (var + 1e-6).sqrt()

                # Encoder sees ALL patches; we then mask in token space.
                enc_out = encoder(pixel_values=imgs).last_hidden_state  # (B, L, D)

                # Random mask in patch space (uniform OR text-aware-weighted).
                if cfg.text_aware_masking:
                    # Normalize per-image to [0, 1].
                    weights = batch_weights.float()
                    w_min = weights.min(dim=1, keepdim=True).values
                    w_max = weights.max(dim=1, keepdim=True).values
                    weights = (weights - w_min) / (w_max - w_min + 1e-6)
                    # Map to [bg_mask_rate, text_mask_rate].
                    weights = cfg.bg_mask_rate + weights * (cfg.text_mask_rate - cfg.bg_mask_rate)
                    visible, mask, ids_restore = random_masking(enc_out, cfg.mask_ratio, weights=weights)
                else:
                    visible, mask, ids_restore = random_masking(enc_out, cfg.mask_ratio)
                B, L_visible, _D_enc = visible.shape
                num_masked = cfg.num_patches - L_visible

                # Project visible (encoder_dim) → decoder_dim BEFORE concat.
                visible_proj = decoder.proj(visible)
                mask_tokens = decoder.mask_token.expand(B, num_masked, -1)
                full = torch.cat([visible_proj, mask_tokens], dim=1)
                # Unshuffle so positions are correct.
                full = torch.gather(
                    full,
                    dim=1,
                    index=ids_restore.unsqueeze(-1).expand(-1, -1, full.size(-1)),
                )

                # Decoder runs on full sequence; output = pixel reconstructions per patch.
                pred = decoder(full)  # (B, L, p*p*3)

                # MSE on masked patches only.
                loss = ((pred - target) ** 2).mean(dim=-1)
                loss = (loss * mask).sum() / mask.sum()

            loss_to_backward = loss / cfg.grad_accum_steps
            loss_to_backward.backward()
            loss_accum += loss.item() / cfg.grad_accum_steps

        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        cur_lr = lr_at(step, steps_per_epoch)
        for g in optim.param_groups:
            g["lr"] = cur_lr
        optim.step()
        last_loss = loss_accum

        if step % cfg.log_every == 0:
            elapsed = time.time() - start
            print(
                f"[mae] step={step}/{total_steps} loss={last_loss:.4f} "
                f"lr={cur_lr:.2e} elapsed={elapsed:.0f}s",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log({"train/loss": last_loss, "train/lr": cur_lr, "step": step})

        # Phase 7c — save FULL state every save_every_epoch epochs (encoder +
        # decoder + optimizer + step + last_loss + cfg snapshot). Mirror to
        # latest/ so a fresh launch can resume. Best-effort GCS push so we
        # don't have to wait for the sync_to_gcs daemon's poll cycle.
        if step % (cfg.save_every_epoch * steps_per_epoch) == 0:
            epoch = step // steps_per_epoch
            ckpt_dir = Path(cfg.checkpoint_dir) / f"epoch-{epoch:04d}"
            _save_full_state(ckpt_dir, encoder, decoder, optim, step, last_loss, cfg)
            if cfg.save_to_latest:
                latest_dir = Path(cfg.checkpoint_dir) / "latest"
                _save_full_state(latest_dir, encoder, decoder, optim, step, last_loss, cfg)
                print(
                    f"[mae] saved full state to {ckpt_dir} (epoch={epoch} step={step}); "
                    f"latest mirror at {latest_dir}",
                    flush=True,
                )
                _try_push_checkpoint_to_gcs(cfg, latest_dir, ckpt_dir)
            else:
                print(
                    f"[mae] saved full state to {ckpt_dir} (epoch={epoch} step={step}); "
                    f"latest mirror disabled",
                    flush=True,
                )
                _try_push_checkpoint_to_gcs(cfg, ckpt_dir, ckpt_dir)

    # Save final encoder.
    final_dir = Path(cfg.checkpoint_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    encoder.save_pretrained(final_dir)
    wall = time.time() - start
    print(
        f"[mae] DONE final_loss={last_loss:.4f} wall={wall:.0f}s saved={final_dir}",
        flush=True,
    )
    return {"final_loss": last_loss}


# === Main ===

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase-7 MAE pretraining of SigLIP-base on pharma images.")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/mae_pretrain.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for cfg.seed.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging.",
    )
    args = parser.parse_args(argv)

    cfg = MAEConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg = dataclasses.replace(cfg, seed=args.seed)
    set_seed(cfg.seed)

    print(
        f"[mae] start phase={cfg.phase} seed={cfg.seed} encoder_init={cfg.encoder_init}",
        flush=True,
    )

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
            print(f"[mae] WARN: wandb init failed ({e}); continuing without it", flush=True)

    bundle = build_mae_model(cfg)
    metrics = pretrain_mae_loop(bundle, cfg, wandb_run)

    if wandb_run is not None:
        wandb_run.finish()

    final_loss = float(metrics.get("final_loss", 0.0))
    print(f"final_pretrain_loss={final_loss:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
