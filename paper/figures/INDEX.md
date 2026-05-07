# Paper Figures — Inventory

Every figure that will appear in the final paper lives here, generated from
the project's actual experiment artifacts (metrics.json, stdout.log,
per_field_*.json). Treat this index as a contract: if a figure has a row
here, the corresponding PNG/SVG MUST exist alongside the row's source data.

**Discipline (per memory `feedback_paper_visualizations.md`):** every
experiment produces its plots inline, written to `paper/figures/<id>/`.
Plot scripts live in `scripts/plots/` and are rerunnable post-hoc.

## Status legend

- ✅ figure exists and is committed
- 🔧 source data exists but figure not yet generated
- ⏳ source experiment still running / not started
- ❌ blocked on something (annotated)

## Figure inventory

### Per-field comparisons

| ID | Figure | Source data | Status | Caption draft |
|---|---|---|---|---|
| F1a | Per-field strict F1, Track A vs Track B | `experiments/per_field_track_a.json` + `..._b.json` | ✅ | "Per-field strict F1 — Track A wins or ties Track B on every field." |
| F1b | Per-field edit-distance F1, Track A vs Track B | same | ✅ | "Per-field lenient F1 — Track A leads by ~2× on most fields; Track B catches up most on `drug_name` (0.61 → 0.28)." |
| F2a | Per-field strict F1, 4-track (vanilla A + 3 MAE variants) | `experiments/per_field_track_a.json` + 3 `per_field_baseline_mae_*-seed44.json` | ✅ | "Per-field strict F1, 4-track. Vanilla SigLIP wins or ties on every field except brand_name where TAPT briefly competes (0.30 vs 0.58 still won by vanilla)." |
| F2b | Per-field edit-distance F1, 4-track | same | ✅ | "Per-field lenient F1, 4-track. Confirms negative result: all 3 MAE variants underperform vanilla on every field except minor wins on brand_name (TAPT) and drug_name (text-aware)." |
| F2c | Per-field strict F1, 5-track (incl. Track D) | + `experiments/per_field_track_d-seed44.json` | ✅ | "Per-field strict F1 across all 5 tracks. Track D (YOLO+SAHI+TrOCR) close to zero on every field; Track A wins." |
| F2d | Per-field edit-distance F1, 5-track | same | ✅ | "Per-field lenient F1 across all 5 tracks. Track A wins every field; Track D is non-trivial only on named-entity fields (brand 0.24, drug 0.26, generic 0.23) but loses 2-3× to Track A there. On OCR-critical fields (batch_number, mrp, mfg_date, expiry_date) where Track D was supposed to win per Malepati 2026's 17× SAHI lift, Track A wins by 22-41×." |
| F3 | Confusion matrix per track (per Malepati 2026 Fig. 4-6 style) | TBD | ⏳ | "Predicted-vs-actual matrices for each track on the brand_name field; diagonal saturation indicates per-class accuracy." |

### Training trajectories

| ID | Figure | Source data | Status | Caption draft |
|---|---|---|---|---|
| F4 | Track A `baseline-seed44-rerun` eval trajectory (twin axes: val macro_f1 + train loss) | `experiments/runs/baseline-seed44-rerun/stdout.log` | ✅ | "Track A vanilla SigLIP: val macro_f1 climbs to 0.074 at step 800, plateaus. Train loss drops 4 orders of magnitude on a log scale — clear overfitting signature on a 564-image train set." |
| F5 | Track B `qwen-baseline-seed42` val trajectory (val macro_f1 + macro_edit_f1) | `experiments/runs/qwen-baseline-seed42/stdout.log` | ✅ | **Headline figure**: Track B's macro_edit_f1 peaks at step 300 (0.2167) then DROPS to 0.1165 at step 400 while strict macro_f1 stays flat. Visual proof of specialization-induced overfitting under choice of stopping metric. |
| F6 | Phase 7 DAPT MAE loss curve (loss vs step, full 8600 steps) | `experiments/pretraining/mae-pretrain-seed44/stdout.log` | ✅ | "MAE pretraining loss 0.95 → 0.16 over 8600 steps (200 epochs, 2787 in-domain pharma images, val + test pixels excluded). Smooth descent, no instabilities. Final loss=0.1566 wall-clock=7.3h on A100." |
| F7 | Phase 7 LR schedule (peak 1.5e-4, warmup 20 epochs, cosine to 0) | derived | ⏳ | "Learning rate schedule for the DAPT MAE run." |

### Qualitative examples

| ID | Figure | Source data | Status | Caption draft |
|---|---|---|---|---|
| F8 | Annotated dataset samples — 8 representative images, one per packaging type, with polygon overlays + field labels (Malepati 2026 Fig. 1, 2 style) | `gs://auto_learn_meds/raw/golden_set/gold_standard.jsonl` (polygon coords) + `raw_images/` | ✅ | "Representative samples from the 837-image pharmaceutical packaging dataset (shared with Malepati 2026): strip-foil, ampoule, syrup-bottle, vial, blister-pack, tube, jar, sachet. Each polygon shows ground-truth field bounding regions; colors per field-name. Best example: F8_blister_pack_IMG_5582 with 6 visible fields including the OCR-critical batch_number / mrp / expiry_date." |
| F9 | MAE reconstruction examples (input | masked | reconstructed) at epoch-50 / 100 / 200 | needs instrumenting pretrain_mae.py to dump examples | ⏳ | "MAE reconstruction quality at epoch 50, 100, 200. Mask ratio 0.75. Pixel-MSE per masked patch (per-patch normalized)." |
| F10 | Per-track failure modes (image, ground-truth XML, predicted XML, error annotation) | TBD | ⏳ | "Failure modes per track. Track A: short-XML truncation. Track B: format-drift outputs. Track D: missed small-text regions." |
| F11 | Track D SAHI tile boundaries + per-tile detections (Malepati 2026 Fig. 1, 2 style) | TBD when Phase 8 runs | ⏳ | "YOLOv12+SAHI inference on a 4032×3024 photo: 1024-px tiles with 0.35 overlap, base size 1280, per-tile detections merged via NMS." |
| F12 | Edge-density heatmap for text-aware MAE (image | edge density | sampled mask) | when text-aware MAE runs | ⏳ | "Sobel edge density used to bias MAE masking toward text + graphic boundaries (Phase 7b creative reuse #1)." |

### Cross-track summary

| ID | Figure | Source data | Status | Caption draft |
|---|---|---|---|---|
| F13 | Master Table 1: macro_f1 + macro_edit_f1 across all tracks | `experiments/ledger.jsonl` | 🔧 | "Cross-track comparison on the held-out val split (n=111). Trainable params, wall-clock, both metrics." |
| F14 | Decision tree by data size (research_directions.md "decision tree" section) | research_directions.md | 🔧 | "Recommended method by labeled-data size: ≥5000 → Track A; 500–5000 → Track B with edit_f1 early-stop; <500 → predicted regime, untested." |
| F15 | Ablation grid (rank 8/16, mask ratio 0.5/0.75, etc.) | as ablations land | ⏳ | "Ablation grid showing the sensitivity of each track to its key hyperparameters." |

## Discipline checklist for adding a new experiment

When writing a new `run_*.sh` or experiment plan:

1. List the figures the experiment produces (which IDs in this index).
2. Add a `scripts/plots/<plot_name>.py` that takes `metrics.json` / `stdout.log` and writes the figure(s).
3. Wire the plot script into the experiment's finalize step (run_experiment.sh / run_pretraining.sh).
4. The figure file gets committed alongside the metrics.json.
5. Update this INDEX.md row to ✅.

## How to regenerate everything

```bash
# After all experiments land:
bash scripts/plots/regenerate_all.sh
# Each subscript reads from experiments/ and writes to paper/figures/.
```
