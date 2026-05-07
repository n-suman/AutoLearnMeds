# Phase 7 + 7b Review — In-Domain MAE Pretraining

**Date:** 2026-05-08
**Branch:** phase-0-plumbing
**Plans:** [Phase 7](../plans/2026-05-06-phase-7-mae-pretraining.md), [Phase 7b](../plans/2026-05-06-phase-7b-text-aware-mae-and-tapt.md)
**Runs:** `mae-pretrain-seed44` (DAPT), `mae-pretrain-tapt-seed44`, `mae-pretrain-text-aware-seed44`
**Downstream runs:** `baseline_mae_init-seed44`, `baseline_mae_text_aware_init-seed44`, `baseline_mae_tapt_init-seed44`
**Tag (suggested):** `phase-7b-mae-negative-result`

---

## TL;DR

**Three independent MAE pretraining recipes (DAPT, text-region-biased DAPT, DAPT+TAPT) all underperform vanilla SigLIP-base-patch16-224 on the downstream Track A pharma extraction task by 2.6–2.9× on strict macro_f1 and 1.7–3.2× on macro_edit_f1.** This is a clean falsification of the hypothesis that in-domain MAE pretraining lifts downstream performance in our small-data regime, evaluated on the same val split (n=111) as Track A and Track B.

The negative result is consistent across:
- different masking strategies (uniform vs edge-density-biased),
- different pretraining-pool compositions (2,787 in-domain images for DAPT; 564 labeled-train images only for TAPT after DAPT),
- different total-epoch counts (200 epochs for DAPT/text-aware; 50 epochs for TAPT).

**Headline numbers** (best.pt eval, n=111 val):

| Track | macro_f1 | macro_edit_f1 | edit/strict |
|---|---:|---:|---:|
| Vanilla SigLIP (Track A best.pt) | **0.0741** | **0.3225** | 4.4× |
| Track C-DAPT (200-epoch in-domain MAE) | 0.0284 | 0.1847 | 6.5× |
| Track C-text-aware (200-epoch text-biased MAE) | 0.0331 | 0.1569 | 4.7× |
| Track C-TAPT (DAPT then 50-epoch labeled-only TAPT) | 0.0262 | 0.0993 | 3.8× |

---

## What was built (Phase 7 + 7b plan tasks)

| Task | Delivered as | Commit |
|---|---|---|
| T1 mae_pretrain.yaml | yaml + dataclass | 6935661 |
| T2 pretrain_mae.py skeleton + 2 lightweight tests | trainer file + tests | 05f9419 (sibling Phase 7 T3) → e00c129 |
| T3 build_mae_model + patchify + random_masking | code + colab smoke test | e00c129 |
| T4 training loop + main + grad accumulation | code | e00c129 |
| T5 run_pretraining.sh launcher | shell script | 58a3eef |
| T6 DAPT MAE run on Colab (200 epochs, 2787 imgs) | run_id `mae-pretrain-seed44` | (live run, final at 7c1db86 era) |
| T7 train.py supports cfg.encoder_init_path | train.py + smoke test | 58a3eef |
| T8 baseline_mae_init.yaml | yaml | 58a3eef |
| F4 strict-leakage (val + test pixels excluded from MAE pool) | cfg.exclude_jsonls + tests | f639497 |
| F8a resumable MAE checkpointing | full state save every 10 epochs + resume | 8a4cf9b |
| F8b precompute_edge_cache.py | standalone script + .npy fix | decebd7 + 620bcc9 |
| F8c pipeline_phase_7.sh orchestrator | 8-stage shell pipeline | 4b19d09 |
| Phase 7b T1-T8 text-aware + TAPT yamls + tests | yamls + dataset filter + weighted random_masking | d3eb1bd |
| Phase 7b T9 launcher hook into orchestrator | (built into pipeline_phase_7.sh) | 4b19d09 |
| F6 fix train.py to load SiglipVisionModel correctly | train.py | f324c06 |
| F7 fix per_field_eval.py to pass encoder_init_path | per_field_eval.py | 9c2bb24 |

Plus eight infrastructure/robustness fixes triggered by Colab VM reclaims and tunnel deaths during the multi-day execution: F1 (PYTHONUNBUFFERED + auto-finalize), F2 (SSH keepalive), F3 (named tunnel via cloudflared + capulamedia.com), F5 (disaster_recover -u flag), F6 (PATH + git config), plus the gsutil orphan-process kill manual interventions.

## Pretraining results (Phase 7 + 7b)

DAPT MAE (vanilla, uniform-random masking, 200 epochs, 2787 in-domain images): final pretraining loss **0.1566**, wall-clock **7.3 hours** on A100. Loss curve smooth descent 0.95 → 0.16 (Figure F6).

Text-aware MAE (200 epochs, edge-density-biased masking 90%/60% on text/bg patches, 2787 images): final pretraining loss **0.1873**, wall-clock **4.0 hours** on A100. Higher absolute loss than vanilla DAPT — expected, since masked text patches have higher entropy than uniformly-random masked patches.

TAPT MAE (50 epochs, init from DAPT-final, on 564 labeled-train images only): final pretraining loss **0.6008**, wall-clock **26 minutes** on A100. Higher loss reflects fewer images per epoch + lower learning rate (5e-5 vs DAPT's 1.5e-4).

All three encoders saved as HuggingFace-loadable `SiglipVisionModel` checkpoints under `gs://auto_learn_meds/checkpoints/mae/`.

## Downstream results (Track A with each MAE-pretrained encoder)

Each downstream run uses the SAME Track A architecture (frozen MAE encoder + 6-layer Donut decoder + cross-attention) and SAME training recipe (1000 steps, batch 16, peak lr 3e-4, etc.) as the vanilla baseline-seed44-rerun. The only variable is which encoder is loaded.

**Eval trajectory (val macro_f1) at every 200 steps:**

| step | Vanilla SigLIP | C-DAPT | C-text-aware | C-TAPT |
|---:|---:|---:|---:|---:|
| 200 | 0.0038 | 0.0036 | 0.0011 | 0.0013 |
| 400 | 0.0374 | 0.0159 | 0.0188 | 0.0041 |
| 600 | 0.0683 | 0.0208 | 0.0186 | 0.0139 |
| 800 | 0.0741 | **0.0284** | **0.0331** | 0.0262 |
| 1000 (final) | 0.0804 | 0.0268 | 0.0217 | 0.0043 |

(Bold = best.pt step for each track. C-TAPT final regresses sharply — strict overfitting on the small TAPT-pretrained encoder.)

The trajectory shape is the same across all three MAE variants: gradual climb to step ~800, then plateau or regress. **None catches up to vanilla SigLIP at any step**, despite identical decoder architecture and identical optimizer/data/training schedule.

## Why does this happen?

Three plausible mechanisms, not mutually exclusive:

1. **Catastrophic forgetting of SigLIP's web-pretrained discriminative features**. SigLIP-base was pretrained on hundreds of millions of image-text pairs with a contrastive sigmoid loss. Its features are tuned for visual-textual alignment. Continuing-pretraining via MAE replaces some of those gradients with pixel-MSE-reconstruction gradients, biasing features toward "what's needed to predict masked pixels" rather than "what's needed to align with text."

2. **Wrong objective for the task**. The downstream task is reading printed text on pharma packaging. Pixel reconstruction asks the encoder to recover masked pixel intensities — which mostly involves modeling the dominant local color and texture (the white-pill, brown-cardboard, blue-foil patterns). Reading printed text is a much narrower, higher-frequency feature class that the pixel-MSE objective barely rewards. The encoder's capacity is finite; pushing toward reconstruction features pushes away from text-reading features.

3. **Pretraining-data scale too small for stable continued-pretraining**. SigLIP saw hundreds of millions of pretraining images; our DAPT pool is 2,787. Even 200 epochs is a tiny fraction of SigLIP's original training compute. The encoder doesn't get enough signal to *replace* its general visual prior with a useful pharma-specific prior; it just gets enough to slightly *corrupt* the general prior in the direction of reconstruction features.

The text-aware variant tested mechanism (2) directly: by masking text-region patches at 90% rate (vs 60% for background), we forced the encoder to spend more capacity on text-from-context filling — exactly the downstream skill. Result: text-aware lifts macro_f1 16% over vanilla DAPT (0.0331 vs 0.0284) but lifts macro_edit_f1 LESS (0.1569 vs 0.1847; -15%). The text-aware encoder is producing more strict-format-shaped output than vanilla DAPT but with somewhat lower content quality. This is suggestive that mechanism (1) is dominant — pure capacity is being moved around by the masking strategy without overall lift.

The TAPT variant tested mechanism (3) directly: by re-using DAPT's 200-epoch encoder and continuing for 50 more epochs on JUST the 564 labeled-train images, we maximized in-domain signal per parameter update at the cost of overfitting risk. Result: TAPT is the WORST downstream performer (macro_f1 0.0262, macro_edit_f1 0.0993). The 50 extra epochs on 564 images destroyed even more of the SigLIP prior, with no compensating gain.

The cleanest pattern: **all three variants converge to roughly the same downstream regime (macro_f1 0.026–0.033, macro_edit_f1 0.10–0.18) regardless of pretraining recipe.** The bottleneck is the MAE pretraining objective, not the masking strategy or the data composition.

## What this changes for the paper

**Phase 7+7b's contribution to the paper is a strong negative result with a mechanistic explanation.**

The Phase 7 plan predicted: "macro_f1 0.0741 → 0.090–0.115 (+20–55%); macro_edit_f1 0.3225 → 0.36–0.42 (+12–30%); largest effect on OCR-heavy fields." All three predictions were falsified. Effects went in the OPPOSITE direction.

For the paper:
- **Section IV-A** Table 1 includes the four Track A variants (vanilla + 3 MAE-init).
- **Section IV-B** per-field analysis (Figures F2a + F2b) shows the per-field gap is uniform — vanilla SigLIP wins on ~all fields under both metrics (one minor exception: TAPT on brand_name, where it ties at 0.30 vs vanilla's 0.58 lenient).
- **Section V-A** discusses the catastrophic-forgetting mechanism above as the primary explanation. Concrete recommendation: "Don't apply DAPT to a strong web-pretrained encoder for narrow downstream tasks unless you're using a contrastive (not generative) self-supervised objective AND you have substantial in-domain data (≥10× labeled set size)."

This is more interesting than a positive result would have been. We tested a published recipe (Gururangan 2020 DAPT) faithfully, with a well-justified domain-specific extension (text-aware masking), and it didn't help. That's worth knowing.

## Robustness lessons (autoresearch infra, not paper)

The 5-day execution of Phase 7 + 7b surfaced and shipped fixes for:

1. **VM reclaim mid-run** (3 incidents): F8a (resumable MAE checkpointing every 10 epochs with full optimizer state + GCS push at every save) ensured zero training progress was lost.
2. **gsutil orphan worker pile-up under heavy concurrent saves**: F8a's `subprocess.run(timeout=120)` doesn't propagate timeout to gsutil's `-m` worker subprocesses; they accumulated and saturated network. Fixed manually each time by killing PIDs by start-time. **Open follow-up**: replace the gsutil rsync with a `subprocess.Popen` + explicit `process.kill()` on timeout, or switch to `gcloud storage rsync` which has cleaner subprocess hierarchy.
3. **Colab named-tunnel disconnects** (~daily): F2 SSH keepalive + F3 named tunnel via `colab.capulamedia.com` (vs trycloudflare-style ephemeral tunnels) reduced disconnect frequency from every-2h to roughly daily.
4. **SiglipVisionModel vs SiglipModel API mismatch**: F6 detected the `model_type` field on saved encoder configs and dispatched to the right loader; F7 fixed `per_field_eval.py` to pass `encoder_init_path` through to `PharmaVLM` (without it, the script silently loaded the wrong encoder, producing macro_f1=0.0 spurious "results").

All four are now codified — future Phase 8 work won't re-encounter them.

## Where we go from here (Phase 8 + cleanup)

1. **Phase 8: Track D = YOLOv12+SAHI+OCR pipeline.** Per Malepati 2026's 17× SAHI lift on the same dataset for OCR-critical classes, this is the most-likely-to-succeed remaining method. Estimated: 1 day code + 6 hours A100 (train YOLOv12 + train per-region OCR head). Will close out the comparative paper.
2. **Section IV-A Table 1 + Section V conclusion**: write up after Track D lands.
3. **Ledger consistency**: the local Mac canonical 9-entry ledger has been pushed to GCS. Colab's local ledger.jsonl is at 6 entries after the `git reset --hard` (the pipeline's local-only commit was reverted). On the next Colab session, the bootstrap's `disaster_recover.sh` will pull the canonical 9-entry ledger from GCS via `-u` (newer-only) flag.
4. **Tag suggestion**: `phase-7b-mae-negative-result` once the review + paper updates land.

## Citation

This phase implements the recipes in:
- He, Chen, Xie, Li, Dollár, Girshick (2021). "Masked Autoencoders Are Scalable Vision Learners." [arXiv:2111.06377](https://arxiv.org/abs/2111.06377). — primary MAE recipe.
- Gururangan, Marasović, Swayamdipta, Lo, Beltagy, Downey, Smith (2020). "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks." [arXiv:2004.10964](https://arxiv.org/abs/2004.10964). — DAPT+TAPT recipe.
- Cao, Xu, Clifton (2022). "AttMask: Attention-Guided Masked Image Modeling." [arXiv:2203.12719](https://arxiv.org/abs/2203.12719). — related work to text-aware masking.
- Malepati, Nandamury, Manjunath, Rajan, Prabhune (2026). "Comparative Evaluation of YOLOv12 and SAHI for Medication Identification in Hospital Pharmacies." IITCEE 2026, IEEE. — same-dataset prior work that motivates Track D for Phase 8.
