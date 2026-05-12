# Track E — Deployable Pharmaceutical Label Reader

**Status:** Design / pre-implementation
**Created:** 2026-05-13
**Author:** Claude (brainstormed with Suman Nandamury)
**Supersedes:** "No more experiments unless redirected" guidance in `memory/project_state_checkpoint_2026_05_08.md`. The user has redirected.
**Related:** `2026-05-05-pharma-vlm-autoresearch-design.md` (the original 5-track comparative design)

## 1. Goal

Build a pharma-label structured-extraction model that the user can plausibly deploy in production — or stop only when improvements clearly plateau and document why. The existing best model (Track A) achieves macro F1 = 0.074 (strict) / 0.32 (lenient) on the 111-image val split; that is not deployable. This spec describes the next experimental track ("Track E") plus contingency paths if Track E doesn't lift the envelope enough.

The user's accuracy bar: **best-effort, no fixed target.** Decision rule: when 3 consecutive runs fail to lift `macro_edit_f1` over safety-critical 4 fields by ≥ 0.02 absolute, declare plateau and reassess.

## 2. Success criteria

The **headline metric** is `macro_edit_f1` over the four safety-critical fields:

- `batch_number` (recall risk if wrong)
- `expiry_date` (dispensing-error risk)
- `mrp` (pricing fraud risk)
- `manufacturing_date` (recall risk)

Computed by the existing `prepare.evaluate(model, val_jsonl, ...)` harness on the gold val split (n=111). The test split (n=111) is touched once, by the final declared-best run, for the paper-grade claim.

**Secondary metrics** (tracked every eval, not optimized): strict F1 on safety-4, both metrics over all 12 fields, per-field breakdown, edit/strict ratio (Track B's diagnostic of format-vs-content overfitting), confidence calibration curves.

**Deploy-readiness** is operationalized via abstention curves rather than a single number: at decoder-entropy threshold T, the model answers Y% of safety-4 fields with accuracy Z%. The paper / deployment story is "at T=X, the system gives a confident answer on Y% of fields with Z% accuracy and flags the rest for review."

## 3. Architecture

| Component | Track A (baseline) | Track E |
|---|---|---|
| Vision encoder | `google/siglip-base-patch16-224` (~93M, frozen) | `google/siglip-large-patch16-384` (~316M, frozen) |
| Patch grid | 14 × 14 = 196 tokens | 24 × 24 = 576 tokens |
| Encoder hidden dim | 768 | 1024 |
| Decoder | 6-layer Donut-style autoregressive, d_model=384, vocab BPE 1,853 | **Unchanged** |
| Cross-attention bridge | Linear 768 → 384 | Linear 1024 → 384 (init from scratch) |
| Tokenizer | BPE trained on `train.jsonl` (1,853 vocab) | BPE retrained on `train.jsonl ∪ pseudo_labels.jsonl` (~2,500 vocab); saved to `experiments/tokenizers/track_e_bpe.json` |

**Why SigLIP-large and not DINOv2 / Qwen2-VL:**
- SigLIP-large has the right inductive bias (text–image contrastive pretraining on web images includes packaging-like content).
- DINOv2 lacks text-discriminative pretraining — forces re-learning. Available as ablation #8 if needed.
- Qwen2-VL was Track B; it failed at 2.2B parameters because LoRA at r=8 wasn't capacity-enough to bridge structural format gap on 564 training images. Bigger isn't the answer here.

**Why frozen encoder:** the data regime (564 + ~2,275 pseudo ≈ 2,839 images) is small for fine-tuning a 316M-parameter encoder. The decoder learns task-specific format; encoder provides general features.

**Memory budget:** SigLIP-large-384 frozen in bf16 + 576-token cross-attention + 6-layer decoder + batch 4 grad-accum 4 fits in ~14–18 GB on A100 40 GB. Fallback to batch 2 grad-accum 8 if OOM.

## 4. Data strategy

### Sources

| Source | Size | Provenance |
|---|--:|---|
| `train.jsonl` (gold) | 564 | pharmacist-labeled, seed-42 split |
| `data/pseudo_labels/round_001.jsonl` | ~2,275 | Qwen2-VL-7B-Instruct via Codex brief 003 (open-source, run on Colab) |
| `val.jsonl` | 111 | gold; never mixed with training |
| `test.jsonl` | 111 | gold; touched once for final paper claim |

The original Brief 001 (GPT-4o) was superseded by Brief 003 (Qwen2-VL on Colab) because the user has no standalone OpenAI API key — only a Codex $20 subscription which doesn't expose raw API calls for bulk inference. Brief 003 uses Pro+ compute units (negligible cost) for the labeling pass.

### Filtering pseudo-labels

1. **Format hygiene:** drop rows whose `xml_label` doesn't parse into our 12-field schema.
2. **Confidence threshold:** drop rows where Qwen2-VL's self-rated `per_field_confidence` averages below `medium`. Threshold tuned post-calibration.
3. **All-empty filter:** drop rows where ≥ 8 of 12 fields are empty (signal of non-pharma images that snuck into `raw/`).
4. **Leak check:** drop any pseudo-label whose `image_path` appears in val or test. Belt-and-suspenders trainer assertion.

### Sample weighting (config-flagged)

The trainer reads:
```yaml
pseudo_jsonl: data/pseudo_labels/round_001.jsonl  # null disables pseudo
pseudo_weight: 0.3                                 # const mode
pseudo_weight_mode: const                          # or "adaptive_confidence"
pseudo_min_confidence: medium                      # low | medium | high
```

Defaults (`pseudo_jsonl: null`, `pseudo_weight: 0.0`) mean old configs (`baseline.yaml`, etc.) reproduce their original numbers — additive-only, per `feedback_experiments_additive_only.md`.

### Augmentation (high-res-tuned)

- Keep RandAugment M=5 N=2 from Track A.
- Add `RandomResizedCrop(384, scale=(0.7, 1.0))`.
- Add mild `ColorJitter(brightness=0.2, contrast=0.2)`.
- No horizontal flip, no rotation > 5° (label text is direction-sensitive).

### Pack-type balance

Leave the natural distribution (32% strip-foil, 16% ampoule, ..., 0.6% carton). Track per-pack F1 in eval, not training. Revisit if rare-pack F1 is alarmingly low.

## 5. Training regime

### Three-stage schedule

| Stage | Steps | Data | Why |
|---|--:|---|---|
| 1 — pseudo warmup | 500 | pseudo only, weight 1.0 | Decoder learns format + broad vocabulary from the larger but noisier set. |
| 2 — joint | 2,000 | gold + pseudo, weights per Section 4 | Bulk learning. Gold anchors, pseudo provides volume. |
| 3 — gold-only finetune | 500, LR ×0.1 | gold only, weight 1.0 | Calibrates to true distribution; strips Qwen2-VL stylistic quirks. |

**Total:** 3,000 steps vs Track A's 1,000. Wall-clock ~3–4× per run.

**Pseudo-unavailable fallback:** if Brief 003 reports calibration F1 below threshold or labels aren't produced for any reason, the run collapses to a single-stage **gold-only schedule of 1,500 steps** — identical to Section 8 run #1 (`track_e_highres_gold_only-seed44`). Same config, same artifact, same ledger row; don't double-count.

### Optimizer & schedule

- AdamW, peak_lr=3e-4 (matches Track A), linear warmup 200 steps, cosine decay to 1e-5.
- Stage 3 LR ×0.1 (peak 3e-5).
- Effective batch 16 (batch 4 × grad-accum 4), bf16 autocast.
- weight_decay=0.05, betas=(0.9, 0.95), grad-clip 1.0.

### Eval cadence

- Every 200 steps on val.
- Save `best_edit_f1.pt` and `best_strict_f1.pt` separately at each eval (Track B's lesson: the two can diverge).
- Save `step_<N>.pt` every 500 steps for trajectory plots.
- Save `eval_<N>.json` per checkpoint with full per-field breakdown.

### Resume-from-checkpoint

`_progress.json` written every 50 steps. On Colab VM reclaim and restart, the script reads it, rebuilds dataloader to skip consumed examples, restores optimizer state, continues. Pattern lifted from existing `pretrain_mae.py` (F8a in the original design).

### One-command lifecycle

```bash
bash scripts/run_experiment.sh experiments/configs/track_e_highres.yaml
```

Extended to call: tokenizer build → pre-flight checks (pseudo file exists; no val/test leak) → training → final eval → `finalize_experiment.sh` (append ledger row — **never insert**; regenerate leaderboard; generate per-field bar chart + trajectory plot to `paper/figures/<run_id>/`; git commit) → GCS sync to `gs://auto_learn_meds/experiments/runs/<run_id>/`.

## 6. Evaluation & calibration

### Headline & secondary metrics

See Section 2.

### Anchored comparison points

| Anchor | Source | Headline metric |
|---|---|---|
| Track A vanilla | `experiments/per_field_track_a.json` | Back-computed safety-4 macro_edit_f1 (one-time job) |
| Qwen2-VL zero-shot ceiling | Brief 003 calibration | TBD |
| Each Track E run | new row appended to ledger | TBD |

### Calibration / abstention curves

At eval time, the decoder also emits per-field next-token entropy. The eval harness computes accuracy-vs-coverage curves over an entropy threshold sweep. Deploy-readiness phrasing: "at threshold T=X, model answers Y% of fields with Z% accuracy on safety-4."

Implemented as `cfg.compute_calibration=true` in eval; no extra training cost.

### Figures generated per run

- `eval_trajectory.png` (twin-axis safety-4 edit + train loss vs step)
- `per_field_safety4.png`, `per_field_all12.png`
- `calibration_curve.png`
- `error_samples.png` (9-panel: still-wrong / fixed-by-track-E / regressed-vs-Track-A)

Plus the running envelope plot `paper/figures/_envelope.png`, regenerated after each run.

## 7. Risks & contingencies

| # | Risk | Detection | Plan B |
|--:|---|---|---|
| 1 | Pseudo-labels too noisy (Qwen2-VL val lenient F1 < 0.30 safety-4) | Brief 003 calibration | Skip pseudo path; only run gold-only-control |
| 2 | SigLIP-large OOM at batch 4 | Pre-flight memory check | Batch 2 grad-accum 8; if still OOM, SigLIP-base-384 |
| 3 | Resolution alone doesn't help (run 1 ≈ Track A) | Gate after run 1 | Pivot to Approach 2 (OCR-conditioned end-to-end) |
| 4 | Pseudo-mix helps lenient but hurts strict | Eval trajectory shows divergence | Informative, not a bug: ship best_edit_f1 checkpoint; document; revisit stage 3 length |
| 5 | Colab VM reclaim eats progress | `_progress.json` restart loop handles it | No new mitigation needed |
| 6 | Non-pharma images get pseudo-labeled | Brief 003 spot-check + all-empty filter | Drop rows with ≥ 8 empty fields at load time |
| 7 | Format drift in pseudo-labels (₹85 vs 85) | Spot-check + regex audit | `prepare.normalize_pseudo_xml(row)` at load |
| 8 | Pseudo includes val/test images | Trainer startup assertion | Fail loudly, refuse to train |
| 9 | Tokenizer over-learns pseudo-only tokens | After build: `tokenizer.encode(gold_targets)` unused-rate check | If > 5%, retrain tokenizer on gold only |
| 10 | Qwen2-VL calibration ≥ 0.60 lenient on safety-4 | Brief 003 calibration | Good problem: decide between custom-model path vs use-Qwen2-VL-as-deployment-model |

**Cost stop:** if total Track E wall-clock exceeds 50 A100-hours without ≥ 0.10 absolute lift on safety-4 macro_edit_f1, pause and reassess approach.

**Two-contingency rule:** if two contingencies fire on the same run, write `experiments/runs/<run_id>/INCIDENT.md` documenting both, surface to user, do not silently hack around.

## 8. Experiment sequencing

Stage 0 (in motion):
- Codex Brief 002 (GCS inventory) — passive safety net
- Codex Brief 003 (Qwen2-VL pseudo-labeling script + Colab runbook)
- Claude (me): trainer config flags + new YAMLs; back-compute Track A safety-4 baseline

Stage 1:

| # | Run ID | Config | Decision after |
|--:|---|---|---|
| 1 | `track_e_highres_gold_only-seed44` | SigLIP-large-384, gold only, 1,500 steps | Gate 1: lift ≥ 0.05 vs Track A? |
| 2a | (Colab) Qwen2-VL calibrate on val | — | Gate 2: safety-4 lenient ≥ 0.30? |
| 2b | (Colab) Qwen2-VL full labeling pass | produces `round_001.jsonl` | — |
| 3 | `track_e_highres_full-seed44` | SigLIP-large-384, gold + pseudo @ 0.3, 3-stage 3,000 steps | Gate 3: lift ≥ 0.03 vs run #1? |

Stage 2 ablations (only if best of #1/#3 clears gates):

| # | Run ID | Varies | Why |
|--:|---|---|---|
| 4 | `track_e_highres_full_w05-seed44` | pseudo_weight=0.5 | Map weight curve |
| 5 | `track_e_highres_full_w10-seed44` | pseudo_weight=1.0 | "Pseudo as good as gold" test |
| 6 | `track_e_highres_full_no_stage3-seed44` | drop stage 3 | Does gold-only finetune matter? |
| 7 | `track_e_512_full-seed44` | image_size=512 | More resolution help? |
| 8 | `track_e_dinov2_full-seed44` | DINOv2-large-518 encoder | Encoder ablation |

Stage 3 (paper claim):

| # | Run ID | What |
|--:|---|---|
| 9 | `track_e_best_test_eval` | Best checkpoint evaluated on **test** (n=111). The number that goes in the paper. |

Reassess gates (decision discipline):
- After #1: lift < 0.05 → resolution wasn't the bottleneck; pivot to Approach 2.
- After #3: no lift over #1 → pseudo isn't helping; run #4 to confirm, then stop pseudo path.
- After 5 runs: envelope hasn't passed 0.15 safety-4 lenient → structural reassess with user.
- 50 A100 hours total → hard stop.

## 9. Paper integration

The existing 5-track comparative paper (`paper/main.md`) stays intact. Its contribution is the three falsified hypotheses; that's a paper-grade finding regardless of what Track E does.

Track E becomes either:
- A new Section VII appendix to the existing paper, or
- A separate follow-up paper.

Decision point: **after Stage 1 runs #1 and #3 have both landed** (we'll have signal on whether resolution alone helps and whether pseudo adds value). If Track E's safety-4 lift over Track A is ≥ 0.10 absolute, it's appendix-worthy at minimum; if ≥ 0.20, it's a separate paper. Codex will get a parallel brief to finish the existing paper's `[FILL]` sections — independent work, doesn't block Track E experiments.

## 10. Anti-goals — what NOT to do

1. **Do not edit existing ledger rows** in `experiments/ledger.jsonl`. Append-only.
2. **Do not overwrite existing run-ids.** Track E uses `track_e_*-seed44` namespace.
3. **Do not edit existing config YAMLs** (`baseline.yaml`, `qwen_baseline.yaml`, `track_d.yaml`, etc.). New configs only.
4. **Do not overwrite existing figures** in `paper/figures/`. New figures get new IDs.
5. **Do not change the eval harness** (`prepare.evaluate`, `prepare.compute_metrics`). New runs must be comparable to existing tracks.
6. **Do not retrain the existing tokenizer in place.** Save Track E's BPE to `experiments/tokenizers/track_e_bpe.json`.
7. **Do not propose Track F before Track E reports** — single-track focus until plateau or breakthrough.
8. **Do not skip the gold-only-control run** even if pseudo-labels are ready first. Section IV-style comparability requires it.

## 11. Files this design will touch (planning, not implementing yet)

- `train.py` — add config flags (encoder_model_name, image_size, pseudo_*, stage_schedule); old configs hit no-op paths
- `experiments/configs/track_e_highres.yaml` (new)
- `experiments/configs/track_e_highres_gold_only.yaml` (new)
- `experiments/configs/track_e_highres_full_w05.yaml`, `..._w10.yaml`, `..._no_stage3.yaml`, `..._512.yaml`, `..._dinov2.yaml` (new, as runs sequence)
- `scripts/run_experiment.sh` — add tokenizer-build step, pre-flight checks
- `scripts/finalize_experiment.sh` — already handles append-only ledger; verify
- `scripts/plots/calibration_curve.py` (new)
- `scripts/plots/envelope.py` (new — running envelope across runs)
- `scripts/plots/error_samples.py` (new — 9-panel qualitative grid)
- `paper/figures/_envelope.png` (new, regenerated)
- `paper/figures/<run_id>/` (new per-run directories)
- `experiments/tokenizers/track_e_bpe.json` (new file)
- `experiments/runs/<run_id>/` (new per run)

The implementation plan (next step: invoke `superpowers:writing-plans`) will sequence the actual code edits.

## 12. Open questions for user review

- Anything in Section 7 (risks) that I haven't named?
- Pack-type imbalance left alone (Section 4) — agree or oversample minorities?
- Run ordering in Section 8 — any reason to prefer a different first run?
- Paper integration in Section 9 — appendix to existing, or separate paper?
