# Run: mae-pretrain-seed44

**Phase:** 7 (MAE continue-pretraining)
**Track:** C (Track A's encoder, MAE-pretrained on in-domain images)
**Parent:** baseline-seed44-rerun (the vanilla-SigLIP comparison)

## Hypothesis

Masked Autoencoder (He et al. 2021) continue-pretraining of the SigLIP-base-patch16-224 vision encoder on **2787 in-domain pharma images** (3061 raw images minus 273 in val + test, kept out for leakage hygiene) will lift downstream Track A performance:

- `macro_f1`: 0.0741 (vanilla SigLIP) → **0.090–0.115** (MAE-init), +20–55%.
- `macro_edit_f1`: 0.3225 → **0.36–0.42**, +12–30%.
- Largest effect expected on OCR-heavy fields (`batch_number`, `expiry_date`, `mrp`) where small-text legibility limits Track A's vanilla performance.

**Null result threshold:** if MAE-init Track A is within ±0.005 strict of vanilla, count as null. That itself is publishable: "in-domain MAE pretraining does not help when the off-the-shelf encoder is already strong enough."

## Why MAE for this task

The encoder must learn features useful for reading pharma labels (small-text contrast, layout, color schemes typical of medicine packaging). Vanilla SigLIP was pretrained on web image-text pairs — strong general visual prior, but minimal pharma-specific signal. MAE on in-domain images closes that gap *without needing labels*: the model is forced to learn pharma-label structure to reconstruct the 75% of patches we hide.

## Strict-leakage hygiene

The pretraining pool excludes **val.jsonl** and **test.jsonl** image basenames (273 images). The encoder thus never sees val/test PIXELS during pretraining (it never saw their labels either). Combined with deterministic seed reproduction in baseline-seed44-rerun, the paper claim becomes "no leakage at any level — labels OR pixels." This is stricter than common practice for self-supervised pretraining and avoids any footnote in Table 1.

## Configuration

See `config.yaml` (snapshot of `experiments/configs/mae_pretrain.yaml`).

Key knobs (all from the He et al. recipe):
- Encoder: `google/siglip-base-patch16-224`, full fine-tune (no LoRA).
- Decoder: 4-layer ViT, dim=512, heads=16, learnable mask token + positional embeddings.
- Mask ratio = 0.75.
- Loss = pixel-MSE on the 147 masked patches per image (per-patch normalized via `norm_pix_loss=true`).
- AdamW with peak_lr=1.5e-4, warmup 20 epochs, cosine decay to 0, weight_decay=0.05, betas=(0.9, 0.95).
- 200 epochs × 44 steps/epoch = **8800 total steps**, effective batch 64 (32 × grad_accum 2).
- bf16 autocast on A100.
- Augmentation: random_resized_crop scale ∈ [0.5, 1.0], hflip 0.5. **No color jitter** (would destroy the small-text features we want the encoder to learn).
- Saves encoder via `encoder.save_pretrained(...)` every 50 epochs (= 4 checkpoints) plus a `final/` snapshot. Each is HF-loadable so train.py can plug it in via `cfg.encoder_init_path`.

## Citation

**Direct prior work on the same dataset (added 2026-05-07 after Malepati 2026 was added to papers/):**

- Malepati, Nandamury, Manjunath, Rajan, Prabhune (2026). "Comparative Evaluation of YOLOv12 and SAHI for Medication Identification in Hospital Pharmacies." *2026 IITCEE*, IEEE. DOI: 10.1109/IITCEE67948.2026.11394638. — same 48 MP smartphone-photo dataset (837 annotated images; our 564/111/111 split is the same set). Establishes that the four OCR-critical classes (Batch, MRP, Manufacturing Date, Expiry) cannot be reliably extracted at standard YOLOv12 resolution (mAP@0.5=0.035) but recover dramatically (0.609 = 17× lift) via SAHI tiled inference at 1024-px tiles. **This is the prior work our Phase 7 MAE pretraining is implicitly competing with**: SAHI gets there via *resolution* (tile and merge); we are trying to get there via *representation* (better encoder features from in-domain pretraining). Both target the same root cause — small text is unreadable at 224×224 SigLIP input. See research_directions.md #11–#13 for SAHI-style ideas now queued as a result.

**Primary:**

- He, Chen, Xie, Li, Dollár, Girshick (2021). "Masked Autoencoders Are Scalable Vision Learners." arXiv:2111.06377. CVPR 2022. — **mask ratio 0.75, asymmetric encoder-decoder, pixel-MSE per masked patch, per-patch normalization, small-decoder design — every architectural choice traces here.**

**Supporting:**

- Dosovitskiy et al. (2020). "An Image is Worth 16x16 Words." arXiv:2010.11929. ICLR 2021. — ViT architecture used by both encoder and decoder.
- Zhai, Mustafa, Kolesnikov, Beyer (2023). "Sigmoid Loss for Language Image Pre-training." arXiv:2303.15343. — the SigLIP-base-patch16-224 encoder we're continue-pretraining; matches the encoder used in vanilla Track A.
- Devlin et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." arXiv:1810.04805. NAACL 2019. — origin of masked-prediction self-supervision; MAE adapts the idea to images.
- Gururangan, Marasović, Swayamdipta, Lo, Beltagy, Downey, Smith (2020). "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks." arXiv:2004.10964. ACL 2020. — **load-bearing reference for our entire Track C strategy: take a strong general-purpose self-supervised model and continue-pretrain it on in-domain data before the supervised fine-tune.** Our Phase 7 transfers their NLP recipe to vision.

**Alternatives considered and not chosen** (paper should mention briefly):

- Bao et al. (2021). "BEiT: BERT Pre-Training of Image Transformers." arXiv:2106.08254. — uses VQ-VAE discrete tokens as reconstruction targets instead of pixels. Not chosen because (a) MAE is simpler with fewer hyperparameters, (b) MAE consistently matches or exceeds BEiT on downstream linear-probe + fine-tune benchmarks.
- Xie et al. (2022). "SimMIM: A Simple Framework for Masked Image Modeling." arXiv:2111.09886. — uses a simple linear-projection decoder instead of MAE's transformer decoder. Not chosen because the asymmetric design lets us use a richer decoder during pretraining and discard it for downstream — better cost/benefit at our scale.

## Result

_(filled in after the run lands)_

```
final_pretrain_loss=
wall_clock_seconds=
exit_code=
total_steps=8800
saved_checkpoints=
```

## Retrospective

_(filled in after the run lands and the downstream `baseline-mae-init-seed44` run completes — the actual paper-relevant comparison)_
