# Phase 7b — Text-Aware MAE Masking + TAPT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two creative-reuse experiments queued on top of in-flight Phase 7. Implements **#1 (text-aware MAE masking)** and **#4 (TAPT)** from `docs/superpowers/research_directions.md`. Together they complete the full Gururangan 2020 DAPT+TAPT recipe AND add a domain-justified twist on the He 2021 mask-uniform-random recipe.

**Architecture:**
- **Text-aware MAE**: same MAE encoder/decoder/loss as Phase 7, BUT the masking distribution is *informativeness-weighted*: patches with high edge density (= likely text or graphic boundaries) get masked at higher rate (90%) than uniform-background patches (60%). Average mask ratio held at 0.75 to keep the He et al. recipe valid.
- **TAPT**: continue MAE pretraining for 50 more epochs on JUST the 564 labeled `train.jsonl` images after the DAPT-MAE finishes.

**Why creative-reuse instead of just running more configs:**

1. **Text-aware masking** — the He 2021 paper notes uniform 75% works "well" on natural images. Pharma labels are 60–80% non-informative background (white space, plain colored regions). Uniform masking spends 60–80% of its training signal on reconstructing background, which is "easy". Force the encoder to focus on the hard, informative patches by masking them more aggressively. Citation: He et al. 2021 (creative reapplication); related work AttMask (Cao et al. 2022) uses ViT attention rollout for similar weighting.

2. **TAPT** — Gururangan 2020 shows DAPT+TAPT consistently beats DAPT-alone in NLP. The same recipe transfers to vision MAE for free — same code, different `images_root` (just labeled train).

**Tech Stack:** OpenCV (Sobel edge detection — already in torchvision dep tree), NumPy. No new model dependencies.

---

## Hypothesis

| Variant | macro_f1 | macro_edit_f1 | Lift over vanilla SigLIP |
|---|---|---|---|
| Vanilla SigLIP + Track A (baseline) | 0.0741 | 0.3225 | — |
| Phase 7 MAE-init + Track A (in flight) | 0.090–0.115 (predicted) | 0.36–0.42 (predicted) | +0.02 to +0.04 |
| **Phase 7b text-aware MAE-init + Track A** | **0.105–0.135** | **0.40–0.46** | **+0.03 to +0.06** |
| Phase 7b TAPT-on-DAPT + Track A | 0.095–0.125 | 0.37–0.43 | +0.02 to +0.05 |

The text-aware variant is predicted to outperform vanilla MAE because the encoder spends more capacity on the text features that matter downstream. TAPT is predicted modest extra lift because the labeled train set is small (564 images) and overlaps with DAPT's 2787, but the more focused signal might still help convergence.

**Null result for text-aware:** if it ties vanilla MAE within ±0.005, we conclude the SigLIP web-prior is already strong enough for text features and uniform masking suffices.

---

## File Structure

| Path | Created/Modified | Responsibility |
|---|---|---|
| `pretrain_mae.py` | Modify | Add `compute_edge_density(image)` + per-patch mask weights to `random_masking`. |
| `experiments/configs/mae_pretrain_text_aware.yaml` | Create | Text-aware variant config — adds `text_aware_masking: true`, `text_mask_rate: 0.90`, `bg_mask_rate: 0.60`. |
| `experiments/configs/mae_pretrain_tapt.yaml` | Create | TAPT variant — `images_root: data/processed/train.jsonl`, `total_epochs: 50`, `encoder_init: <DAPT ckpt path>`. |
| `tests/test_pretrain_mae_components.py` | Modify | Add 2 lightweight tests for edge density + weighted masking. |
| `experiments/configs/baseline_mae_text_aware_init.yaml` | Create | Track A config for downstream eval. |
| `experiments/configs/baseline_mae_tapt_init.yaml` | Create | Track A config for TAPT downstream eval. |
| `docs/superpowers/reviews/phase_7b_review.md` | Create | End-of-phase review with the 4-way comparison table. |

---

## Task 1: Edge-density text-likelihood detector

Add to `pretrain_mae.py` (lazy-import OpenCV inside):

```python
def compute_edge_density_per_patch(
    image_path: "Path",
    image_size: int,
    patch_size: int,
) -> "np.ndarray":
    """Returns (num_patches,) ndarray of per-patch mean Sobel edge magnitude.

    High edge magnitude => text / graphic boundaries / informative content.
    Low edge magnitude => uniform background, blurred regions.

    Used by text-aware masking to bias toward masking informative patches more.
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
```

Cache the output: pre-compute for ALL pretraining images at dataset-init time, store in a NumPy memmap file at `cfg.checkpoint_dir / "edge_cache.npy"`. ~10 sec for 2787 images at 224x224. Eliminates per-step CPU cost.

Add a lightweight test:
```python
def test_edge_density_per_patch_shape(pretrain_mae_mod, tmp_path):
    """Sobel edges + per-patch mean produces shape (num_patches,) with reasonable range."""
    import numpy as np
    # Create a synthetic 224x224 image with clear text-like vertical stripes.
    import cv2
    img = np.full((224, 224), 255, dtype=np.uint8)
    img[:, ::8] = 0  # vertical lines every 8 pixels
    img_path = tmp_path / "synth.png"
    cv2.imwrite(str(img_path), img)
    edges = pretrain_mae_mod.compute_edge_density_per_patch(img_path, 224, 16)
    assert edges.shape == (196,)
    assert edges.min() > 0  # all patches have some edges (vertical lines everywhere)
    assert edges.std() < edges.mean()  # roughly uniform across patches given the synthetic
```

---

## Task 2: Weighted random masking

Modify `random_masking()` to accept an optional per-patch weight array. Higher weight → higher mask probability.

```python
def random_masking(x, mask_ratio: float, weights=None):
    """Random masking per He et al. 2021, with optional per-patch weights.

    Args:
        x: (B, L, D) token features.
        mask_ratio: target average mask ratio across the batch.
        weights: optional (B, L) per-patch weights in [0, 1]. Higher weight =>
            higher P(mask). If None, fall back to uniform random (original recipe).

    Returns:
        visible_x: (B, L_keep, D) the kept tokens (in shuffle order).
        mask: (B, L) float, 1 at masked positions and 0 at kept positions
              (in original order — already unshuffled).
        ids_restore: (B, L) inverse permutation.
    """
    import torch
    B, L, D = x.shape
    keep = int(L * (1 - mask_ratio))

    if weights is None:
        # Original He 2021 recipe — uniform Bernoulli via argsort of uniform noise.
        noise = torch.rand(B, L, device=x.device)
    else:
        # Weighted: high-weight patches get LOW noise (more likely to be masked, since we
        # take ids_shuffle[:keep] = the LOWEST noise = patches we want to KEEP). To put
        # high-weight patches in the "mask" region, give them HIGH noise. Inverted compared
        # to weight semantics, so:
        #   high weight => high noise => sorted to back => masked
        noise = (1 - weights.to(x.device)) * torch.rand(B, L, device=x.device)

    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :keep]
    visible_x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
    mask = torch.ones(B, L, device=x.device)
    mask[:, :keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return visible_x, mask, ids_restore
```

The `weights` argument is `None` by default → backward compatible.

Modify the dataset (`_ImagePathDataset`) to also return per-patch edge density, lazily loaded from the memmap cache. The `__getitem__` returns `(image_tensor, weights_tensor)` instead of just `image_tensor`.

Modify `pretrain_mae_loop`:
```python
if cfg.text_aware_masking:
    weights = batch[1].to(device)  # (B, num_patches)
    # Normalize weights to [low_rate, high_rate] range so per-image average mask ratio is mask_ratio.
    weights = (weights - weights.min(dim=1, keepdim=True).values) / \
              (weights.max(dim=1, keepdim=True).values - weights.min(dim=1, keepdim=True).values + 1e-6)
    # Map to per-patch mask probability: low edges -> bg_mask_rate, high edges -> text_mask_rate.
    weights = cfg.bg_mask_rate + weights * (cfg.text_mask_rate - cfg.bg_mask_rate)
    visible, mask, ids_restore = random_masking(enc_out, cfg.mask_ratio, weights=weights)
else:
    visible, mask, ids_restore = random_masking(enc_out, cfg.mask_ratio)
```

Add a lightweight test:
```python
def test_random_masking_with_weights_biases_high_weight(pretrain_mae_mod):
    """High-weight patches should be masked at higher rate."""
    import torch
    torch.manual_seed(42)
    B, L, D = 100, 196, 16
    x = torch.randn(B, L, D)
    # Make first 50 patches "text" (weight=1) and last 146 "bg" (weight=0).
    weights = torch.cat([torch.ones(B, 50), torch.zeros(B, 146)], dim=1)
    visible, mask, _ = pretrain_mae_mod.random_masking(x, mask_ratio=0.75, weights=weights)
    # Average mask rate over the 100-image batch:
    text_mask_rate = mask[:, :50].mean().item()
    bg_mask_rate = mask[:, 50:].mean().item()
    # High-weight (text) patches must be masked >75%; low-weight (bg) must be <75%.
    assert text_mask_rate > 0.75
    assert bg_mask_rate < 0.75
    # Total mask ratio should still be roughly 0.75.
    assert abs(mask.mean().item() - 0.75) < 0.05
```

---

## Task 3: Text-aware config + run

Create `experiments/configs/mae_pretrain_text_aware.yaml`:

```yaml
# Text-aware MAE pretraining — variant of mae_pretrain.yaml.
# Differences from mae_pretrain.yaml: text_aware_masking=true + text/bg rates.
phase: 7
run_kind: mae_pretrain
encoder_init: "google/siglip-base-patch16-224"

image_size: 224
patch_size: 16
num_patches: 196

mask_ratio: 0.75
text_aware_masking: true
text_mask_rate: 0.90
bg_mask_rate: 0.60

decoder_depth: 4
decoder_dim: 512
decoder_heads: 16
norm_pix_loss: true

seed: 44
batch_size: 32
grad_accum_steps: 2
peak_lr: 1.5e-4
min_lr: 0.0
warmup_epochs: 20
total_epochs: 200
weight_decay: 0.05
adam_betas: [0.9, 0.95]
grad_clip: 1.0
log_every: 50
save_every_epoch: 50
precision: "bf16"

random_resized_crop_scale: [0.5, 1.0]
random_horizontal_flip: 0.5

images_root: "/content/raw_images_local"   # local SSD path (set on Colab)
checkpoint_dir: "checkpoints/mae/text-aware-seed44"
wandb_project: "autolearnmeds"
wandb_mode: "online"

exclude_jsonls:
  - "data/processed/val.jsonl"
  - "data/processed/test.jsonl"
```

Add `text_aware_masking: bool = False`, `text_mask_rate: float = 0.90`, `bg_mask_rate: float = 0.60` fields to MAEConfig dataclass (all with defaults so the original yaml still parses).

Run on Colab AFTER the current DAPT MAE finishes:
```bash
cd /workspace
nohup bash scripts/run_pretraining.sh mae-pretrain-text-aware-seed44 \
  --config experiments/configs/mae_pretrain_text_aware.yaml --seed 44 --no-wandb \
  > /tmp/mae-text-aware.log 2>&1 &
```

ETA: same as DAPT — ~6.5 hours on A100.

---

## Task 4: TAPT config + run

Create `experiments/configs/mae_pretrain_tapt.yaml`:

```yaml
# TAPT — task-adaptive MAE pretraining on labeled train images only.
# Initialized from the DAPT-MAE checkpoint. Continues for 50 more epochs.
phase: 7
run_kind: mae_pretrain
encoder_init: "checkpoints/mae/run-seed44/final"   # DAPT output

image_size: 224
patch_size: 16
num_patches: 196

mask_ratio: 0.75
text_aware_masking: false   # TAPT uses uniform masking; the variation is the data.

decoder_depth: 4
decoder_dim: 512
decoder_heads: 16
norm_pix_loss: true

seed: 44
batch_size: 32
grad_accum_steps: 2
peak_lr: 5.0e-5             # 1/3 of DAPT's peak — we're already in a good basin
min_lr: 0.0
warmup_epochs: 5
total_epochs: 50            # 1/4 of DAPT
weight_decay: 0.05
adam_betas: [0.9, 0.95]
grad_clip: 1.0
log_every: 50
save_every_epoch: 25
precision: "bf16"

random_resized_crop_scale: [0.5, 1.0]
random_horizontal_flip: 0.5

# TAPT-specific: pretrain on the LABELED train images only — supplies the strongest
# domain-relevant signal because these are the exact images we'll be extracting from.
images_root: "/content/raw_images_local"
checkpoint_dir: "checkpoints/mae/tapt-seed44"
wandb_project: "autolearnmeds"
wandb_mode: "online"

# TAPT inverts the exclusion: only INCLUDE train.jsonl images.
include_jsonls:
  - "data/processed/train.jsonl"
exclude_jsonls: []
```

Add `include_jsonls: tuple[str, ...] = ()` field to MAEConfig (intersection of the include set with the on-disk image basenames). When `include_jsonls` is non-empty, `build_pretrain_dataset` filters the pool to ONLY images whose basename appears in any `include_jsonls`.

Add a lightweight test mirroring the existing exclude test.

Run on Colab AFTER DAPT MAE finishes. ~30 minutes A100.

---

## Task 5: Downstream evaluation — Track A with each MAE variant

Three Track A runs total:
1. `baseline-mae-init-seed44` (from main Phase 7 plan T8) — uses DAPT MAE.
2. `baseline-mae-text-aware-init-seed44` — new config `baseline_mae_text_aware_init.yaml` with `encoder_init_path: checkpoints/mae/text-aware-seed44/final`.
3. `baseline-mae-tapt-init-seed44` — new config `baseline_mae_tapt_init.yaml` with `encoder_init_path: checkpoints/mae/tapt-seed44/final`.

Each ~30 minutes A100. Run sequentially after their respective MAE finishes.

---

## Task 6: Per-field comparison + Phase 7b review

Run `scripts/per_field_eval.py` on each Track A variant. Run `scripts/compare_per_field.py` for the most informative pairs:
- Vanilla Track A vs Track A-MAE (Phase 7's main result)
- Track A-MAE vs Track A-MAE-text-aware (text-aware delta)
- Track A-MAE vs Track A-MAE-TAPT (TAPT delta)

Write `docs/superpowers/reviews/phase_7b_review.md` with the 4-way table:

| Track | macro_f1 | macro_edit_f1 | OCR-fields delta | wall-clock |
|---|---|---|---|---|
| A (vanilla SigLIP) | 0.0741 | 0.3225 | — | 30 min |
| A-MAE (DAPT) | ? | ? | ? | 30 min |
| A-MAE-text-aware | ? | ? | ? | 30 min |
| A-MAE-TAPT | ? | ? | ? | 30 min |

Tag `phase-7b-creative-reuse-complete`.

---

## Total compute budget

- Phase 7 DAPT MAE (in flight): ~6.5 h
- Phase 7b text-aware MAE: ~6.5 h (sequential after DAPT)
- Phase 7b TAPT: ~30 min (sequential after DAPT)
- 3× Track A downstream runs: 3 × 30 min = 1.5 h
- 3× per-field eval: 3 × 5 min = 15 min

Total: ~14.5 h A100. With current Pro+ balance (1770 units, ~13 units/h on A100) = ~190 units. Comfortable margin — we'd still have ~1580 units after.

## Citation

This plan is creative reapplication of two cited papers:
- **He, Chen, Xie, Li, Dollár, Girshick (2021).** "Masked Autoencoders Are Scalable Vision Learners." arXiv:2111.06377. — uniform-random masking; we vary by adding edge-density-weighted masking that biases toward informative patches. Related work: Cao et al. (2022), "AttMask: Attention-Guided Masked Image Modeling," arXiv:2203.12719 — uses ViT attention rollout for mask weighting. Our approach is simpler (edge density, no learned weighting) and task-justified (text + graphic boundaries are exactly what pharma label extraction cares about).
- **Gururangan, Marasović, Swayamdipta, Lo, Beltagy, Downey, Smith (2020).** "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks." arXiv:2004.10964. — DAPT+TAPT recipe. Phase 7 was DAPT only; Phase 7b adds TAPT to complete the recipe.
