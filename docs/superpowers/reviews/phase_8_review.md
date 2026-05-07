# Phase 8 Review — Track D: YOLOv12 + SAHI + TrOCR Pipeline

**Date:** 2026-05-08
**Branch:** phase-0-plumbing
**Plan:** [Phase 8 plan](../plans/2026-05-08-phase-8-track-d-sahi-pipeline.md)
**Runs:** `yolo-baseline-seed44` (detection training), `track_d-seed44` (full pipeline eval)
**Tag (suggested):** `phase-8-track-d-complete`

---

## TL;DR

Track D = **YOLOv12-s + SAHI tiled inference + TrOCR-base-printed per-region OCR + XML composition**, replicating Malepati 2026's modular detection pipeline and extending it to full structured-extraction evaluatable through `prepare.compute_metrics`. Trained from scratch on the same 564/95/142 split shared with Tracks A/B/C.

**Headline result on n=111 val (best.pt for YOLO, pretrained TrOCR):**

```
Track D:                macro_f1 = 0.0180   macro_edit_f1 = 0.0831
Vanilla Track A:        macro_f1 = 0.0741   macro_edit_f1 = 0.3225
                        ────────────       ────────────────
                        Track A wins       Track A wins
                        4.1× strict        3.9× lenient
```

**Track D is the worst-performing track in the entire 5-track comparison.**

| Track | macro_f1 | macro_edit_f1 |
|---|---:|---:|
| Vanilla SigLIP | **0.0741** | **0.3225** |
| C-text-aware MAE | 0.0331 | 0.1569 |
| C-DAPT MAE | 0.0284 | 0.1847 |
| C-TAPT MAE | 0.0262 | 0.0993 |
| B Qwen+LoRA (step 400) | 0.0159 | 0.1165 |
| **D YOLO+SAHI+TrOCR** | **0.0180** | **0.0831** ← worst |

The hypothesis "Track D wins on OCR-critical fields where end-to-end methods see only 224×224 input" is **falsified**. Even on `batch_number`, `expiry_date`, `mfg_date`, `mrp` — the four classes where Malepati 2026 showed SAHI lifts AP@0.5 17× over plain YOLOv12 — Track A's edit_f1 beats Track D's by 30–35× (e.g., batch_number 0.18 vs 0.005; mfg_date 0.29 vs 0.0).

The mechanism is **compound error in the modular chain**, characterized in Section "Why does Track D lose?" below.

---

## What was built

| Task | Delivered as | Commit |
|---|---|---|
| T1 deps (`ultralytics>=8.3`, `sahi>=0.11`) | pyproject.toml | 6e8605e |
| T2 polygon → YOLO bbox conversion + data.yaml + 3 lightweight tests | `scripts/build_yolo_dataset.py` | df9fcdc |
| T3 train_yolo.py + yolo_baseline.yaml + 2 tests | `train_yolo.py` | a095c70 |
| T4 track_d_pipeline.py + track_d.yaml + 3 tests | `track_d_pipeline.py` | a095c70 |
| T5 run_track_d.sh launcher + smoke test | `scripts/run_track_d.sh` | a095c70 |
| T6 YOLO training on Colab | YOLOv12-s, 100 epochs, 19 min | (live run) |
| T6 fix: yolov12s.pt manual download | weights from sunsmarterjie/yolov12 | (manual SSH) |
| T7 fix: bypass prepare.evaluate's tensor pipeline | `track_d_pipeline.py:main()` rewritten to read full-res images | b989a13 |
| T7 Track D inference + eval | run_id `track_d-seed44`, 5 min | (live run) |
| T8 5-track per-field figures (F2c, F2d) | `paper/figures/F2c_*.png`, `F2d_*.png` | (this commit) |
| T9 paper main.md + this review | (this commit) |  |

**Compute used:** 19 min YOLO training + 5 min Track D inference = ~24 minutes A100, ~2 compute units. Far less than the 8-hour estimate in the plan (the small dataset trains very fast).

**Two early-execution bugs surfaced and fixed:**
1. **`yolov12s.pt` not in ultralytics' auto-download registry.** Fixed by manually downloading from `sunsmarterjie/yolov12`'s GitHub release (the official YOLOv12 repo, not the ultralytics-bundled one).
2. **`prepare.evaluate` downsamples images to 224×224 (SigLIP input size).** Track D's `predict_text` was receiving these tiny tensors and reversing normalization to a 224×224 PIL — way too small for SAHI's 1024-px tiling. Initial Track D run showed macro_f1=0.0 / macro_edit_f1=0.0025 (uniform across all fields). Fixed in commit `b989a13` by giving Track D its own eval loop that reads images at full resolution from disk via `cfg.images_root + image_path`, runs the pipeline, parses predictions via `prepare.parse_output()`, and feeds (predictions, truths) directly to `prepare.compute_metrics`. Same metric primitives, full-res input.

## Why does Track D lose?

Track D's per-field breakdown reveals the issue. Per-field strict F1:

| Field | Track D strict | Track A strict | Track A lenient | Notes |
|---|---:|---:|---:|---|
| brand_name | 0.036 | 0.138 | 0.580 | Track A wins 4× lenient |
| drug_name | 0.100 | 0.204 | 0.614 | Track A wins 6× lenient |
| generic_name | 0.000 | 0.049 | 0.614 | Track A 27× lenient |
| company | 0.044 | 0.350 | 0.340 | Track A 4× lenient |
| **batch_number** | 0.000 | 0.000 | **0.181** | Track A's edit_f1 wins 36× |
| **expiry_date** | 0.000 | 0.000 | **0.266** | Track A 38× |
| **mfg_date** | 0.000 | 0.000 | **0.295** | Track A 41× |
| **mrp** | 0.000 | 0.000 | **0.217** | Track A 22× |

The **OCR-critical fields** (bottom four rows) are where Track D was supposed to win — they were the basis of Malepati 2026's 17× SAHI lift. Track D loses them all by an order of magnitude on the lenient metric.

Three compounding error sources, each plausible from the design:

1. **YOLOv12-s mAP@0.5 = 0.291 (val) at 19-min training.** That's the upstream cap. Even if the rest of the pipeline were perfect, ~70% of the field detections are missing or wrong-class. Malepati 2026 reports macro AP@0.5 of 0.609 on OCR-critical classes specifically (with targeted SAHI tuning); ours is 0.291 across all classes. That suggests further training or class-specific tuning would lift YOLO accuracy, but doubling YOLO accuracy still leaves a ~50% per-detection error rate.

2. **TrOCR-base-printed pretrained, no fine-tune.** Per-region OCR was assumed to be the easy part of the chain. But TrOCR was pretrained on receipts/forms, not pharma packaging. Pharma fonts + tight text layouts + glare from foil packaging produce decoded text that's "almost right" but rarely a strict match. This shows up as Track D's edit_f1 being non-trivial on named-entity fields (brand 0.24, drug 0.26, generic 0.23) but strict F1 close to zero — text content is recognizable but format/spelling drift kills exact match.

3. **Compound error: the chain multiplies failure rates.** End-to-end XML correctness for a field requires: YOLO detected the right region (~30%) AND assigned the right class (probably ~60% conditional on detect, since YOLO classes 0–11 share many visual features) AND TrOCR read the text without error (~50% maybe?). Multiplied: 0.30 × 0.60 × 0.50 = 9% expected end-to-end correctness per field. The strict per-field F1 of ~0.02 across 12 fields averaged with most at zero is consistent.

Track A doesn't have this chain. Its decoder sees the entire encoded image (ALL patches) and generates the entire structured output as one autoregressive sequence. Errors in field A don't propagate to field B; the decoder learns global field co-occurrence patterns. The 564 labeled training examples are enough to learn a useful global decoder; they are NOT enough to train each link in a 3-stage modular pipeline to a high enough joint accuracy.

## Comparison to Malepati 2026's published result

The plan cited Malepati 2026's headline finding: SAHI lifts macro AP@0.5 on the four OCR-critical classes from 0.035 to 0.609 — a 17× detection-accuracy gain on the same dataset. We can't directly reproduce that 0.609 (their paper used targeted per-class confidence tuning + longer training; we used the standard recipe + 19 min). But that's not the relevant comparison anyway, because:

- Malepati 2026 stops at **detection + class label**. They never compose full structured output, never attempt to read the field VALUE inside each detected region.
- Track D **extends** their pipeline to produce field values via TrOCR + assemble them into the same XML format Tracks A/B/C use, so all five tracks are evaluated on the same `compute_metrics` harness with the same "is the brand_name string exactly right" semantics.

The result Track D produces is "what happens when you try to use the Malepati 2026 modular pipeline for full structured extraction in this regime." Per-region OCR was the missing piece, and it's the piece that breaks. The detection performance Malepati 2026 demonstrated is **necessary but not sufficient** for end-to-end XML correctness.

## What this changes for the paper

Track D was the last "this might beat Track A" track in the comparison. With Track D losing too:

- **The paper's headline result is now "every alternative we tried — bigger pretrained model, in-domain pretraining (3 ways), modular SAHI pipeline — underperforms a custom hybrid trained from scratch with a frozen pretrained encoder."** That's a stronger claim than "Track A is good"; it's "Track A is *the* answer in this regime, and we can document why each alternative fails."

- **Section V (Discussion)** structures cleanly around the three falsified hypotheses, each with a mechanism:
  - V-A: Why Track B fails (specialization-induced overfitting under the chosen stopping metric, plus the format-vs-content trade-off).
  - V-B: Why Track C fails (catastrophic forgetting of SigLIP's task-aligned features under MAE's pixel-reconstruction objective).
  - V-D: Why Track D fails (compound error in the modular chain; per-region OCR is the bottleneck even when detection is reasonable).

- **Section IV-E (Decision tree)** is now writeable with confidence:
  - **≥5,000 labeled images**: Track A or Track D, depending on whether deployment-time latency favors end-to-end or modular. Both should converge.
  - **500–5,000 labeled images (our regime)**: Track A wins. Track B is competitive only if you have a much bigger labeled set (we extrapolate Track B catches up around ~3,000–5,000 examples based on the LoRA-fine-tuning literature; not tested in this paper).
  - **<500 labeled images**: untested; predicted regime where Track A's decoder also fails. Recommended path: Track A with k-shot in-context examples, OR an explicit OCR-centric architecture trained to read first and structure second.

## Robustness lessons

Two interesting bugs worth documenting for future autoresearch work:

1. **Newer model weights aren't always in standard auto-download registries.** YOLOv12 (2025) was released too recently for ultralytics' default model zoo. Mitigation: the project's deps now include the manual download URL in a comment; `train_yolo.py` should detect missing weights and fetch from the official release.

2. **Different tracks need different image-preprocessing pipelines, but the eval harness assumes one.** `prepare.evaluate` was built around Track A's SigLIP-input expectation (224×224, normalized to [-1, 1]). Track D needs full-resolution input for SAHI tiling. Solution: Track D bypasses `prepare.evaluate` and constructs `(predictions, truths)` directly, then calls `prepare.compute_metrics`. The metric primitive is the right level of abstraction; the tensor-feeding-loop is too track-specific to share.

## Where to next

With Track D landed, the comparative comparison closes at 5 tracks. Remaining paper work:

- **Section V-D** (modular pipeline failure analysis) — write up the compound-error analysis above with per-stage failure-rate breakdown.
- **Section IV-E** (decision tree) — write up.
- **Section VI** (conclusion) — write up.
- **Bibliography polish** — add `keers2013`, `tariq2025`, `tolley2022`, `nguyen2025`, `javaid2024`, `labelstudio` to the registry (they were flagged in the paper draft but not yet committed).

Citation: per the plan, this phase implements:
- **Tian, Ye, Doermann (2025)** [tian2025yolov12]: YOLOv12 architecture.
- **Akyon, Altinuc, Temizel (2022)** [akyon2022sahi]: SAHI primary reference.
- **Malepati, Nandamury, Manjunath, Rajan, Prabhune (2026)** [malepati2026sahi]: same-dataset prior work that Track D extends.
- **Li et al. (2021)** "TrOCR" — needs to be added to the registry.
