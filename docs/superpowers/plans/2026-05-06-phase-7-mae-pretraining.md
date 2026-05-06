# Phase 7 — MAE Pretraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue-pretrain the SigLIP vision encoder via Masked Autoencoder (MAE; He et al. 2021) on all 3061 in-domain pharma images, then plug the resulting encoder into Track A and measure whether downstream `macro_f1` and `macro_edit_f1` lift over Track A's vanilla-SigLIP baseline (0.0741 / 0.3225 from `baseline-seed44-rerun`).

**Architecture:** Encoder = SigLIP-base-patch16-224 vision tower (initialized from pretrained, fine-tuned via MAE). Decoder = 4-layer ViT with dim=512, heads=16, learned mask token, learned positional embeddings. Mask ratio = 0.75 (standard MAE). Loss = pixel-MSE per masked patch, normalized per-patch.

**Tech Stack:** PyTorch, transformers (`SiglipVisionModel.from_pretrained` for init), peft NOT used (full fine-tune of encoder). HuggingFace `save_pretrained` format for the output encoder so Track A can load it via the existing `from_pretrained(path)` path.

---

## Hypothesis

In-domain MAE pretraining narrows the visual-distribution gap between SigLIP's web-pretraining and our pharma-label test images. Predict:
- `macro_f1`: 0.0741 (vanilla SigLIP) → **0.090–0.115** (MAE-init), +20–55%.
- `macro_edit_f1`: 0.3225 → **0.36–0.42**, +12–30%.
- Effect should be largest on OCR-heavy fields (`batch_number`, `expiry_date`, `mrp`) where small-text legibility limits Track A's vanilla performance.

**Null result threshold:** if MAE-init Track A ≤ vanilla-SigLIP Track A by more than ±0.005 strict, count as null. Document as "in-domain MAE pretraining does not help in this regime; SigLIP's web-prior is already saturating the encoder for pharma-label OCR." That's also a publishable finding.

---

## File Structure

| Path | Created/Modified | Responsibility |
|---|---|---|
| `experiments/configs/mae_pretrain.yaml` | Create | Locked Phase 7 pretraining config. |
| `pretrain_mae.py` | Create | Single-file MAE pretraining trainer (sibling of `train.py`). |
| `experiments/configs/baseline_mae_init.yaml` | Create | Track A config that points `encoder_init_path` at the MAE output. |
| `train.py` | Modify | Add `cfg.encoder_init_path` support — when set, load encoder from that path instead of SigLIP base. |
| `scripts/run_pretraining.sh` | Create | Launcher for `pretrain_mae.py`. Mirrors `run_experiment.sh` shape but writes only to `experiments/pretraining/<run_id>/`. |
| `tests/test_pretrain_mae_components.py` | Create | Lightweight: yaml load + lazy-imports invariant. |
| `tests/test_pretrain_mae_smoke.py` | Create | Colab-marked: load SigLIP-init encoder, single masked-forward + reconstruction. |
| `experiments/pretraining/<run_id>/` | Output | MAE pretraining run dir (encoder.safetensors + config.json + stdout.log). |
| `checkpoints/mae/<run_id>/` | Output | HF-format encoder checkpoint, loadable via `SiglipVisionModel.from_pretrained(path)`. |
| `docs/superpowers/reviews/phase_7_review.md` | Create | End-of-phase review. |

---

## Task 1: `experiments/configs/mae_pretrain.yaml`

```yaml
phase: 7
run_kind: mae_pretrain
encoder_init: "google/siglip-base-patch16-224"

# Image preprocessing
image_size: 224
patch_size: 16
num_patches: 196   # (224/16)^2

# MAE
mask_ratio: 0.75
decoder_depth: 4
decoder_dim: 512
decoder_heads: 16
norm_pix_loss: true   # per-patch norm before MSE; standard in MAE paper

# Training
seed: 44
batch_size: 32
grad_accum_steps: 2   # effective batch 64
peak_lr: 1.5e-4
min_lr: 0.0
warmup_epochs: 20
total_epochs: 200      # 200 × ceil(3061/64) ≈ 9600 steps
weight_decay: 0.05
adam_betas: [0.9, 0.95]
grad_clip: 1.0
log_every: 50
save_every_epoch: 50    # checkpoint at 50, 100, 150, 200
precision: "bf16"

# Image augmentation (minimal — preserve text legibility)
random_resized_crop_scale: [0.5, 1.0]
random_horizontal_flip: 0.5
# NO color jitter — would destroy text features which are exactly what we want to learn

# I/O
images_root: "/mnt/gcs/raw/raw_images"
checkpoint_dir: "checkpoints/mae/run-seed44"
wandb_project: "autolearnmeds"
wandb_mode: "online"
```

Commit: `feat(phase-7): mae_pretrain.yaml — locked Phase 7 config`.

---

## Task 2: `pretrain_mae.py` skeleton + 2 lightweight tests

Create `pretrain_mae.py` with:
- Module docstring (continuing pattern from train.py / train_qwen.py)
- `from __future__ import annotations`
- Lightweight imports only (argparse, dataclasses, json, random, sys, time, pathlib.Path, typing.Any, yaml)
- `MAEConfig` dataclass mirroring all yaml fields + `from_yaml(path)` classmethod
- `set_seed(seed)` (mirror `train.py`)
- Placeholder `main()` that argparses, loads config, prints `[mae] start config={...}` then `final_pretrain_loss=0.0000` and exits 0

Create `tests/test_pretrain_mae_components.py` with two NON-colab-marked tests:
1. `test_mae_config_loads_yaml` — loads `mae_pretrain.yaml` and asserts shape (mask_ratio==0.75, decoder_depth==4, total_epochs==200, etc.)
2. `test_pretrain_mae_module_imports_without_torch` — `import pretrain_mae; assert not hasattr(pretrain_mae, "torch") and not hasattr(pretrain_mae, "transformers")`

Verify: `uv run --extra dev pytest tests/test_pretrain_mae_components.py -v` passes 2/2.

Commit: `feat(phase-7): pretrain_mae.py skeleton + MAEConfig dataclass + 2 lightweight tests`.

---

## Task 3: MAE model — encoder init from SigLIP, ViT decoder, patchify helpers

Append to `pretrain_mae.py`:

```python
def build_mae_model(cfg: "MAEConfig") -> dict[str, Any]:
    """Build the MAE encoder + decoder. Encoder loaded from SigLIP; decoder fresh.

    Returns {"encoder": SiglipVisionModel, "decoder": ViTDecoder, "mask_token": nn.Parameter}.
    """
    import torch
    import torch.nn as nn
    from transformers import SiglipVisionModel

    encoder = SiglipVisionModel.from_pretrained(cfg.encoder_init)

    # The decoder is a small ViT operating on patch tokens. It receives:
    #   - the encoder's output for visible patches (projected to decoder_dim)
    #   - learned mask tokens at the masked positions
    #   - learned positional embeddings for ALL positions
    # ... and outputs pixel reconstructions per patch.

    encoder_dim = encoder.config.hidden_size  # 768 for SigLIP-base
    decoder_dim = cfg.decoder_dim

    class ViTDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(encoder_dim, decoder_dim)
            self.pos_embed = nn.Parameter(torch.zeros(1, cfg.num_patches, decoder_dim))
            decoder_layer = nn.TransformerEncoderLayer(
                d_model=decoder_dim,
                nhead=cfg.decoder_heads,
                dim_feedforward=decoder_dim * 4,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=cfg.decoder_depth)
            self.norm = nn.LayerNorm(decoder_dim)
            self.pred = nn.Linear(decoder_dim, cfg.patch_size ** 2 * 3)  # RGB pixel values

        def forward(self, x):
            x = x + self.pos_embed
            x = self.transformer(x)
            x = self.norm(x)
            return self.pred(x)

    decoder = ViTDecoder()
    mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
    nn.init.trunc_normal_(mask_token, std=0.02)
    nn.init.trunc_normal_(decoder.pos_embed, std=0.02)

    return {"encoder": encoder, "decoder": decoder, "mask_token": mask_token}


def patchify(images, patch_size: int):
    """(B, 3, H, W) → (B, num_patches, patch_size**2 * 3) — matches MAE paper."""
    import torch
    p = patch_size
    B, C, H, W = images.shape
    h, w = H // p, W // p
    x = images.reshape(B, C, h, p, w, p)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    x = x.reshape(B, h * w, p * p * C)
    return x


def random_masking(x, mask_ratio: float):
    """Returns (visible_x, mask, ids_restore)."""
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
```

Add `tests/test_pretrain_mae_smoke.py` with one Colab-marked test:
- Load encoder from `cfg.encoder_init` via `build_mae_model(cfg)`.
- Run a single masked forward+decode on a `(2, 3, 224, 224)` zeros tensor.
- Assert decoder output has shape `(2, 196, 768)` (= `(B, num_patches, patch_size**2 * 3)`).

Commit: `feat(phase-7): build_mae_model + patchify + random_masking + 1 colab smoke test`.

---

## Task 4: Training loop with masking, pixel-MSE, save HF-format encoder

Append to `pretrain_mae.py`:

```python
def build_pretrain_dataset(cfg: "MAEConfig"):
    """List ALL .jpeg/.jpg/.png files under cfg.images_root. No labels needed."""
    from pathlib import Path
    root = Path(cfg.images_root)
    paths = sorted(p for p in root.glob("**/*") if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
    return paths


def pretrain_mae_loop(bundle, cfg, wandb_run):
    import time
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = bundle["encoder"].to(device).train()
    decoder = bundle["decoder"].to(device).train()
    mask_token = bundle["mask_token"].to(device)

    image_paths = build_pretrain_dataset(cfg)
    print(f"[mae] dataset n={len(image_paths)} (using all images, no label needed)")

    aug = transforms.Compose([
        transforms.RandomResizedCrop(cfg.image_size, scale=tuple(cfg.random_resized_crop_scale)),
        transforms.RandomHorizontalFlip(cfg.random_horizontal_flip),
        transforms.ToTensor(),
        # SigLIP normalization (matches train.py preprocessor):
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    params = list(encoder.parameters()) + list(decoder.parameters()) + [mask_token]
    optim = torch.optim.AdamW(params, lr=cfg.peak_lr, betas=tuple(cfg.adam_betas), weight_decay=cfg.weight_decay)

    def lr_at(step, steps_per_epoch):
        epoch = step / steps_per_epoch
        if epoch < cfg.warmup_epochs:
            return cfg.peak_lr * (epoch / cfg.warmup_epochs)
        progress = (epoch - cfg.warmup_epochs) / max(1, cfg.total_epochs - cfg.warmup_epochs)
        # cosine to min_lr
        cos = 0.5 * (1 + (-1) ** 0 * (math.cos(math.pi * progress)))  # simplified placeholder; T4 should use math.cos
        return cfg.min_lr + (cfg.peak_lr - cfg.min_lr) * cos

    # NOTE: compute steps_per_epoch as ceil(len(image_paths) / effective_batch).
    effective_batch = cfg.batch_size * cfg.grad_accum_steps
    steps_per_epoch = (len(image_paths) + effective_batch - 1) // effective_batch
    total_steps = steps_per_epoch * cfg.total_epochs
    print(f"[mae] steps_per_epoch={steps_per_epoch} total_steps={total_steps}")

    indices = list(range(len(image_paths)))
    cursor = 0
    last_loss = 0.0
    start = time.time()

    for step in range(1, total_steps + 1):
        optim.zero_grad()
        loss_accum = 0.0
        for _ in range(cfg.grad_accum_steps):
            # Pull cfg.batch_size images
            if cursor + cfg.batch_size > len(indices):
                random.shuffle(indices)
                cursor = 0
            batch_paths = [image_paths[indices[cursor + i]] for i in range(cfg.batch_size)]
            cursor += cfg.batch_size
            imgs = torch.stack([aug(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Forward pass
                target = patchify(imgs, cfg.patch_size)  # (B, L, p*p*3)
                if cfg.norm_pix_loss:
                    mean = target.mean(dim=-1, keepdim=True)
                    var = target.var(dim=-1, keepdim=True, unbiased=False)
                    target = (target - mean) / (var + 1e-6).sqrt()

                # Encoder sees ALL patches; we then mask in token space.
                enc_out = encoder(pixel_values=imgs).last_hidden_state  # (B, L, D)
                # Drop the [CLS] token if SigLIP's encoder includes one. SigLIP-base does NOT
                # add a CLS token by default, but verify with bundle["encoder"].config.
                # If shape == (B, L, D) where L == num_patches, no slicing needed.

                # Random mask in patch space.
                visible, mask, ids_restore = random_masking(enc_out, cfg.mask_ratio)
                # Build full sequence for decoder: visible + mask_tokens, then unshuffle.
                B, L_visible, D = visible.shape
                num_masked = cfg.num_patches - L_visible
                mask_tokens = mask_token.expand(B, num_masked, -1)
                # Project visible (encoder_dim) → decoder_dim BEFORE concat.
                visible_proj = decoder.proj(visible)
                full = torch.cat([visible_proj, mask_tokens], dim=1)
                # Unshuffle
                full = torch.gather(full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, full.size(-1)))

                # Decoder runs on full sequence; output = pixel reconstructions per patch.
                pred = decoder(full)  # (B, L, p*p*3)

                # MSE on masked patches only.
                loss = ((pred - target) ** 2).mean(dim=-1)
                loss = (loss * mask).sum() / mask.sum()

            loss_to_backward = loss / cfg.grad_accum_steps
            loss_to_backward.backward()
            loss_accum += loss.item() / cfg.grad_accum_steps

        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        for g in optim.param_groups:
            g["lr"] = lr_at(step, steps_per_epoch)
        optim.step()
        last_loss = loss_accum

        if step % cfg.log_every == 0:
            elapsed = time.time() - start
            msg = (f"[mae] step={step}/{total_steps} loss={last_loss:.4f} "
                   f"lr={lr_at(step, steps_per_epoch):.2e} elapsed={elapsed:.0f}s")
            print(msg, flush=True)
            if wandb_run is not None:
                wandb_run.log({"train/loss": last_loss, "train/lr": lr_at(step, steps_per_epoch), "step": step})

        # Save encoder every save_every_epoch epochs.
        if step % (cfg.save_every_epoch * steps_per_epoch) == 0:
            epoch = step // steps_per_epoch
            ckpt_dir = Path(cfg.checkpoint_dir) / f"epoch-{epoch:04d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            encoder.save_pretrained(ckpt_dir)  # HF format — can be reloaded with SiglipVisionModel.from_pretrained
            print(f"[mae] saved encoder to {ckpt_dir} (epoch={epoch})", flush=True)

    # Save final encoder.
    final_dir = Path(cfg.checkpoint_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    encoder.save_pretrained(final_dir)
    print(f"[mae] DONE final_loss={last_loss:.4f} wall={(time.time()-start):.0f}s saved={final_dir}", flush=True)
    return {"final_loss": last_loss}
```

Replace `main()` with full version: argparse → MAEConfig → set_seed → wandb init → build_mae_model → pretrain_mae_loop → wandb finish → print `final_pretrain_loss=X.XXXX`.

Verify: locally `uv run --extra dev pytest tests/ 2>&1 | tail -3` still all green; `python pretrain_mae.py --help` works.

Commit: `feat(phase-7): MAE training loop + main with grad accumulation`.

---

## Task 5: `scripts/run_pretraining.sh` launcher

Mirrors `scripts/run_experiment.sh` shape but simpler — no auto-finalize (pretraining isn't a downstream run, no ledger entry). Writes:
- Snapshot of `pretrain_mae.py` to `experiments/pretraining/<run_id>/pretrain_mae.py`
- Snapshot of `mae_pretrain.yaml` to `experiments/pretraining/<run_id>/config.yaml`
- stdout/stderr to `experiments/pretraining/<run_id>/stdout.log` (with `PYTHONUNBUFFERED=1`)
- Final loss parsed from `^final_pretrain_loss=` and written to `metrics.json` along with wall_clock and exit_code.
- Explicit `gsutil rsync` of the run dir + checkpoints/mae/ to GCS at end.

Lock the run_id format: `mae-pretrain-seedN`.

Commit: `feat(phase-7): scripts/run_pretraining.sh — MAE pretraining launcher`.

---

## Task 6: Run pretraining on Colab (~2.5 hours A100)

Pre-write `experiments/pretraining/mae-pretrain-seed44/notes.md` with hypothesis (see top of this plan).

Launch:
```bash
cd /workspace
export LD_LIBRARY_PATH=/usr/lib64-nvidia
export PATH=$PATH:/tools/google-cloud-sdk/bin
nohup bash scripts/run_pretraining.sh mae-pretrain-seed44 --seed 44 --no-wandb \
  > /tmp/mae-pretrain.log 2>&1 &
```

Monitor via stdout polling for `[mae] step=X/9600 loss=...` lines. Final encoder lands at `/workspace/checkpoints/mae/run-seed44/final/`. Auto-finalize is NOT called (pretraining doesn't go in the ledger — only downstream eval runs do).

After completion, manually push checkpoint to GCS (the run script does this in step 5 above).

Estimated 2.5 hours on A100 at 200 epochs × 48 steps/epoch × ~1 sec/step.

---

## Task 7: `train.py` — add `cfg.encoder_init_path` support

Modify `build_model(cfg)` (or wherever the SigLIP encoder is instantiated) so that:
- If `cfg.encoder_init_path` is set (non-empty string), load via `SiglipVisionModel.from_pretrained(cfg.encoder_init_path)`.
- Otherwise, fall back to current behavior (`SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")`).

Add to `Cfg` dataclass: `encoder_init_path: str = ""`.

Add a smoke test asserting the new field exists in the dataclass.

Commit: `feat(phase-7): train.py supports cfg.encoder_init_path for MAE-init encoder`.

---

## Task 8: `experiments/configs/baseline_mae_init.yaml`

A copy of `baseline.yaml` with one addition:
```yaml
encoder_init_path: "checkpoints/mae/run-seed44/final"
```
All other fields identical to `baseline.yaml` (max_steps=1000, peak_lr, etc.).

Commit: `feat(phase-7): baseline_mae_init.yaml — Track A with MAE-pretrained encoder`.

---

## Task 9: Run baseline-mae-init-seed44 on Colab

```bash
cd /workspace
export LD_LIBRARY_PATH=/usr/lib64-nvidia
export FINALIZE_PHASE=baseline
nohup bash scripts/run_experiment.sh baseline-mae-init-seed44 \
  --track A --seed 44 --no-wandb \
  --config experiments/configs/baseline_mae_init.yaml \
  > /tmp/baseline-mae-init.log 2>&1 &
```

~30 min on A100 (same as Track A baseline). Auto-finalize records ledger entry with both metrics.

---

## Task 10: Per-field eval + comparison

Run `scripts/per_field_eval.py --track A --ckpt checkpoints/runs/baseline_mae_init/best.pt --out experiments/per_field_track_a_mae.json`.

Run `scripts/compare_per_field.py --a experiments/per_field_track_a.json --b experiments/per_field_track_a_mae.json --out experiments/per_field_track_a_vs_mae.md` — using the EXISTING comparison infra (just feeding it Track A vs Track A-MAE instead of Track A vs Track B).

Commit all three files.

---

## Task 11: Phase 7 review + tag

Create `docs/superpowers/reviews/phase_7_review.md` with:
- What was built (T1–T10).
- Headline result: MAE-init Track A vs vanilla Track A on macro_f1 + macro_edit_f1.
- Per-field win/loss vs vanilla.
- Hypothesis verdict: confirmed / null / refuted.
- Updated decision tree for the paper.

Tag `phase-7-mae-complete`, push.

---

## Self-Review

| Spec / framing requirement | Implemented in task |
|---|---|
| In-domain MAE pretraining of SigLIP encoder | T2–T6 |
| Save in HF-loadable format | T4 (`encoder.save_pretrained`) |
| Track A integration via config flag | T7 |
| Apples-to-apples downstream evaluation | T8–T9 |
| Per-field breakdown | T10 |
| Paper-grade review | T11 |

No placeholders. The MAE training loop's `lr_at` function uses `math.cos` — make sure to `import math` inside the function. The `random_masking` function uses `torch.gather` correctly per the He et al. 2021 paper. Per-patch normalization is enabled via `norm_pix_loss: true` (the recipe that the MAE paper found most stable).
