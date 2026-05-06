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
    save_every_epoch: int
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

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MAEConfig":
        data = yaml.safe_load(Path(path).read_text())
        if isinstance(data.get("adam_betas"), list):
            data["adam_betas"] = tuple(data["adam_betas"])
        if isinstance(data.get("random_resized_crop_scale"), list):
            data["random_resized_crop_scale"] = tuple(data["random_resized_crop_scale"])
        if isinstance(data.get("exclude_jsonls"), list):
            data["exclude_jsonls"] = tuple(data["exclude_jsonls"])
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


def random_masking(x, mask_ratio: float):
    """Random masking per He et al. 2021.

    Args:
        x: (B, L, D) token features.
        mask_ratio: fraction of tokens to mask out.

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
    noise = torch.rand(B, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :keep]
    visible_x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
    mask = torch.ones(B, L, device=x.device)
    mask[:, :keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return visible_x, mask, ids_restore


# === Dataset ===

def build_pretrain_dataset(cfg: "MAEConfig") -> list[Path]:
    """List image files under cfg.images_root, excluding any in cfg.exclude_jsonls.

    No labels needed — MAE self-supervised pretraining only consumes pixels.

    The exclude mechanism reads each path in cfg.exclude_jsonls (each is a JSONL
    file with `image_path` entries — same format as data/processed/{train,val,
    test}.jsonl) and filters out images whose basename appears in the exclude
    set. This keeps val/test pixels out of MAE pretraining so the paper can
    claim zero leakage at any level (label OR pixel).
    """
    root = Path(cfg.images_root)
    paths = sorted(p for p in root.glob("**/*") if p.suffix.lower() in {".jpeg", ".jpg", ".png"})

    if cfg.exclude_jsonls:
        excluded_basenames: set[str] = set()
        for jsonl_path in cfg.exclude_jsonls:
            jp = Path(jsonl_path)
            if not jp.is_file():
                print(f"[mae] WARN: exclude_jsonl {jp} not found; skipping (no exclusion applied)", flush=True)
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
                    excluded_basenames.add(Path(img_path).name)
        if excluded_basenames:
            before = len(paths)
            paths = [p for p in paths if p.name not in excluded_basenames]
            print(f"[mae] excluded {before - len(paths)} images via {len(cfg.exclude_jsonls)} jsonl(s); pool={len(paths)}", flush=True)

    return paths


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

    image_paths = build_pretrain_dataset(cfg)
    print(f"[mae] dataset n={len(image_paths)} (using all images, no label needed)", flush=True)

    aug = transforms.Compose([
        transforms.RandomResizedCrop(cfg.image_size, scale=tuple(cfg.random_resized_crop_scale)),
        transforms.RandomHorizontalFlip(cfg.random_horizontal_flip),
        transforms.ToTensor(),
        # SigLIP normalization (matches train.py preprocessor):
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # decoder.parameters() now includes decoder.mask_token (registered as Parameter)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optim = torch.optim.AdamW(
        params,
        lr=cfg.peak_lr,
        betas=tuple(cfg.adam_betas),
        weight_decay=cfg.weight_decay,
    )

    effective_batch = cfg.batch_size * cfg.grad_accum_steps
    steps_per_epoch = (len(image_paths) + effective_batch - 1) // effective_batch
    total_steps = steps_per_epoch * cfg.total_epochs
    print(f"[mae] steps_per_epoch={steps_per_epoch} total_steps={total_steps}", flush=True)

    def lr_at(step: int, steps_per_epoch: int) -> float:
        epoch = step / steps_per_epoch
        if epoch < cfg.warmup_epochs:
            return cfg.peak_lr * (epoch / max(1, cfg.warmup_epochs))
        progress = (epoch - cfg.warmup_epochs) / max(1, cfg.total_epochs - cfg.warmup_epochs)
        progress = min(1.0, max(0.0, progress))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.min_lr + (cfg.peak_lr - cfg.min_lr) * cos

    indices = list(range(len(image_paths)))
    random.shuffle(indices)
    cursor = 0
    last_loss = 0.0
    start = time.time()

    for step in range(1, total_steps + 1):
        optim.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for _ in range(cfg.grad_accum_steps):
            if cursor + cfg.batch_size > len(indices):
                random.shuffle(indices)
                cursor = 0
            batch_paths = [image_paths[indices[cursor + i]] for i in range(cfg.batch_size)]
            cursor += cfg.batch_size
            imgs = torch.stack([aug(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)

            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16):
                target = patchify(imgs, cfg.patch_size)  # (B, L, p*p*3)
                if cfg.norm_pix_loss:
                    mean = target.mean(dim=-1, keepdim=True)
                    var = target.var(dim=-1, keepdim=True, unbiased=False)
                    target = (target - mean) / (var + 1e-6).sqrt()

                # Encoder sees ALL patches; we then mask in token space.
                enc_out = encoder(pixel_values=imgs).last_hidden_state  # (B, L, D)

                # Random mask in patch space.
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

        # Save encoder every save_every_epoch epochs.
        if step % (cfg.save_every_epoch * steps_per_epoch) == 0:
            epoch = step // steps_per_epoch
            ckpt_dir = Path(cfg.checkpoint_dir) / f"epoch-{epoch:04d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            encoder.save_pretrained(ckpt_dir)  # HF format
            print(f"[mae] saved encoder to {ckpt_dir} (epoch={epoch})", flush=True)

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
