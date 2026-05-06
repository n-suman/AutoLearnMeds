# Track B Review — Qwen2-VL-2B + LoRA Baseline

**Date:** 2026-05-06
**Branch:** phase-0-plumbing
**Plan:** [docs/superpowers/plans/2026-05-05-pharma-vlm-track-b-qwen-lora.md](../plans/2026-05-05-pharma-vlm-track-b-qwen-lora.md)
**Run:** experiments/runs/qwen-baseline-seed42 (REVERTED on strict macro_f1)
**Tag:** track-b-baseline-complete

---

## What was built

Nine implementation tasks (T1–T9) and three robustness fixes (F1–F3). Total: 13 commits on `phase-0-plumbing`.

| Commit | Task | What |
|---|---|---|
| 3ded379 | T1 | `pyproject.toml` — `peft>=0.13`, `bitsandbytes>=0.44`, `qwen-vl-utils>=0.0.10`, `transformers>=4.46` |
| ff4e7a5 | T2 | `experiments/configs/qwen_baseline.yaml` — locked Track B config |
| 05f9419 | T3 | `train_qwen.py` skeleton + `QwenConfig` dataclass + 2 lightweight tests |
| a2f10b4 | T4 | `load_qwen_with_lora` — 4-bit Qwen2-VL + LoRA r=8 + 1 colab smoke test |
| b2cd1be | T5 | `QwenWrapper.predict_text` — pluggable into `prepare.evaluate` (same contract as Track A's `PharmaVLM.predict_text`) |
| e666c1f | T6 | training loop + `main` with grad accumulation |
| 4e1d64e | T6-fix | print last-eval metrics (match Track A's contract; not best-on-val) |
| 2525296 | T7 | `scripts/run_experiment.sh` accepts `--track A\|B` dispatch |
| f3ef1a3 | (T8 prep) | pre-write notes.md for qwen-baseline-seed42 |
| f78637d | F2 | SSH keepalive (`ServerAliveInterval 60` / `ServerAliveCountMax 10`) |
| cde9fda | F1 | self-contained run — `PYTHONUNBUFFERED`, auto-finalize, GCS push on exit |
| 6e4a5b4 | F3 | switch Colab bootstrap to Cloudflare **named** tunnel (`colab.capulamedia.com`) |
| (this) | T8 result | experiment ledger entry — REVERTED on strict macro_f1 |

The seven plan tasks (T1–T7) were each executed by a fresh implementation subagent following `superpowers:subagent-driven-development`, with a spec compliance review for the foundational T3 (since T4–T6 build on it). All 99 lightweight tests + 20 colab-deselected tests pass on `phase-0-plumbing` HEAD.

The three F-tasks were not in the original plan — they were added after the **first** Track B baseline attempt failed silently (block-buffered stdout + tunnel death + no auto-finalize). F1 + F2 + F3 ensured the **second** attempt produced full per-step trajectory data even though the runtime was again reclaimed mid-eval.

---

## Track B baseline result

```
qwen-baseline-seed42 (Track B):
  step 400  macro_f1=0.0159  macro_edit_f1=0.1165   (last completed eval; serves as final)
  step 300  macro_f1=0.0123  macro_edit_f1=0.2167   (peak on macro_edit_f1)
  wall_clock ≈ 78 min on A100 (training)
  effective batch = 16 (4 × grad_accum 4); LoRA r=8 on q/k/v/o_proj
  trainable = ~5 M params of ~2.2 B total = 0.23%
```

The run was synthesized from `stdout.log` because the Colab VM was reclaimed during the step-500 eval before `train_qwen.py` could write `metrics.json`. The step-400 LoRA adapters are intact at `gs://auto_learn_meds/checkpoints/runs/qwen_baseline/best_lora/` (8.7 MB safetensors + adapter_config.json).

### Eval trajectory

| Step | macro_f1 (strict) | macro_edit_f1 (lenient) | Δ over prior eval |
|------|-------------------|--------------------------|-------------------|
| 100  | 0.0000 | 0.0000 | — |
| 200  | 0.0115 | 0.0152 | +0.012 / +0.015 |
| 300  | 0.0123 | **0.2167** | +0.001 / **+0.202** |
| 400  | **0.0159** | 0.1165 | +0.004 / **−0.105** |
| 500  | did-not-complete | did-not-complete | — |

The non-monotonic `macro_edit_f1` curve (peaks at step 300, drops 47% relative at step 400) is the load-bearing finding from this experiment.

---

## Head-to-head with Track A

| Method | trainable params | macro_f1 (strict) | macro_edit_f1 (lenient) | wall (min) | source |
|---|---|---|---|---|---|
| **Track A**: SigLIP+Donut, 1000 steps, seed 44 | 26.5 M (all decoder) | **0.0804** | not measured (metric added post hoc) | 30 | baseline-seed44 |
| Track A: same, seed 42 | 26.5 M | 0.0679 | not measured | 31 | baseline-seed42 |
| Track A: same, seed 43 | 26.5 M | 0.0646 | not measured | 30 | baseline-seed43 |
| Track A: + RandAugment M=5, seed 42 | 26.5 M | 0.0679 | not measured | 28 | randaug-m5-seed42 |
| **Track B**: Qwen2-VL-2B + LoRA r=8, 500 steps, seed 42 | ~5 M | 0.0159 (step 400) | 0.1165 (step 400) | 78 | qwen-baseline-seed42 |
| **Track B (peak edit_f1)**: same, at step 300 | ~5 M | 0.0123 | **0.2167** | 60 | qwen-baseline-seed42, ckpt-300 |

**Apples-to-apples on strict**: Track A wins by 5.0× (0.0804 / 0.0159).
**Apples-to-apples on lenient**: not yet possible — Track A's `macro_edit_f1` is uninstrumented. Backfilling those numbers is the highest-priority next experiment.
**Apples-to-best-form on lenient (Track B at its own optimal step 300)**: Track B's `macro_edit_f1=0.2167` would need to be compared against Track A's lenient number to decide. Rough interpolation: Track A's strict 0.0804 with a similar ratio (1.5×–3× lenient/strict typical for span extraction) suggests Track A's lenient is in the 0.12–0.24 band — i.e. **likely comparable to Track B's peak**.

---

## Paper-relevant findings

### 1. The choice of stopping metric matters more than the choice of method

`macro_edit_f1` peaked at step 300 (0.2167) and dropped 47% relative at step 400 (0.1165), while strict `macro_f1` kept slowly climbing (0.0123 → 0.0159 over the same window). This is the textbook signature of **specialization-induced overfitting**: the model is learning to produce strict-match-shaped output at the cost of the more general "label content extraction" quality that lenient matching rewards.

A practitioner who naively optimizes `macro_f1` on a 564-image train set with a pretrained VLM will end up with a model that is technically "better" on the loss they tracked, but is producing measurably worse extracted information. **Early-stopping on `macro_edit_f1` would have given a model with 1.86× the lenient performance** of the model at strict-peak.

This is a different framing than the typical "early stopping prevents overfitting" advice — here the validation **loss is still decreasing** while the metric we actually care about is regressing. The recommendation is metric-specific, not loss-specific.

### 2. Pretrained VLMs unlock content-correctness fast, format-correctness slow

The step 200 → step 300 jump in `macro_edit_f1` (0.0152 → 0.2167; **14×**) is larger than any single inter-checkpoint jump in Track A's full history. The corresponding strict `macro_f1` went 0.0115 → 0.0123 (essentially flat). Read together: **between step 200 and step 300, Qwen2-VL learned to read pharmaceutical labels** (the lenient metric jumped). It was already producing *some* output before that, but the output had no structure the strict matcher could lock onto.

The strict metric then started catching up between step 300 and step 400 (0.0123 → 0.0159; +30%). The model was now learning the *XML format*, not the *labels*. And the cost of that format learning was a regression in lenient performance — meaning the labels themselves were getting worse to make the format more matchable. **The strict metric and the lenient metric anti-correlate during this regime.**

### 3. 5M trainable params are NOT enough to bridge the strict-format gap

Track B's strict `macro_f1` (0.0159) at the end of 500 steps is still 5× below Track A's (0.0804). LoRA r=8 contributed ~5M trainable params; Track A trains 26.5M from scratch. Hypothesis: **higher-rank LoRA (r=16 or r=32, ~10M–20M trainable) closes the strict gap without sacrificing the lenient performance** — that's the natural follow-up experiment.

If r=16/r=32 closes the strict gap to ~0.05+ AND keeps `macro_edit_f1` ≥ 0.2, the Track B v2 result becomes: *equal or better content quality than Track A, with 1/3 to 1/5 the trainable parameters.* That's a publication-grade finding.

If r=16/r=32 closes the strict gap but tanks the lenient (more specialization-overfitting), the finding becomes: *parameter count is fungible with overfitting risk; the specialization-overfitting failure mode is invariant to rank.* Also publication-grade, just a different paper.

---

## Decision tree (the paper's payload)

For pharma label extraction with N labeled images, here's what the data so far recommends:

```
                  IF you have ≥ 5000 labeled images:
                  ├── train Track A (custom hybrid SigLIP+Donut decoder)
                  │     - both metrics likely to converge
                  │     - format-correctness is "free" since decoder learns it from N
                  │     - cheaper inference (~30M params vs 2.2B)
                  │
                  IF you have 500–5000 labeled images (our regime):
                  ├── start Track B (Qwen2-VL + LoRA r=8 to r=16)
                  │     - lenient performance saturates fast (~step 300 of 500)
                  │     - early-stop on macro_edit_f1, NOT on macro_f1
                  │     - if strict format matters at deployment, consider r=16+
                  │     - DO NOT chase strict macro_f1 past the edit_f1 peak
                  │
                  IF you have < 500 labeled images:
                  └── (untested in this paper; prediction:)
                        Track B with r=4 or r=8, max_steps=100–200
                        + adversarial format prompt augmentation
                        Strict macro_f1 likely undefined regime;
                        report only edit_f1.
```

The paper's contribution: **the small-data-pharma pipeline is metric-bound, not method-bound, in the 500–5000-image regime**. The same model fine-tuned with the same recipe gives a 14× lenient quality difference depending on **when you stop**. This re-shapes the standard "Track A vs Track B" comparison from "which model is better" to "which model + which stopping rule is better."

---

## Robustness lessons (autoresearch infra, not paper)

The first Track B baseline attempt (run launched ~18:48 UTC on 2026-05-05) failed silently due to:
1. Python's stdout block-buffered to 4KB; first ~step= log never flushed; we had no idea training was progressing past model load.
2. Cloudflared "quick tunnel" (`*.trycloudflare.com`) died with `websocket: bad handshake` mid-run; we couldn't query state.
3. No auto-finalize; even if training completed cleanly, manual SSH was required to write the ledger entry.

The F1 + F2 + F3 patches addressed all three:
- **F1**: `PYTHONUNBUFFERED=1` on the trainer (real-time per-step logs); `finalize_experiment.sh` auto-called on clean exit; explicit `gsutil rsync` of run dir + checkpoints + ledger to GCS in the same script.
- **F2**: `ServerAliveInterval 60` / `ServerAliveCountMax 10` in the SSH config (defensive against any future idle-disconnect).
- **F3**: Cloudflare **named** tunnel via authenticated token, persistent hostname `colab.capulamedia.com`. Survives runtime restarts; no URL rotation.

The second Track B attempt (this run) confirmed F1 worked end-to-end: we had the full per-step trajectory in `stdout.log` and on GCS in real time. F2 + F3 didn't fully save us when the Colab VM itself was reclaimed (the tunnel died because its host died), but they DID give us a clean reproducible setup.

**Open robustness work** (deferred — not blocking the paper):

- **Per-eval metrics.json snapshot.** Currently `metrics.json` is only written at process exit. Add a `--metrics-snapshot` mode to `train_qwen.py` that writes (and gsutil rsyncs) `metrics.json` AFTER every completed eval. Then any future VM reclaim still leaves a real, parseable, ledger-ingestible metrics.json corresponding to the last completed eval. ~10 min of work; eliminates the failure mode this run hit.
- **Best-checkpoint metric in metrics.json.** Currently we report only the last completed eval's metric as `final_macro_f1`. Also persist `best_macro_f1`, `best_macro_edit_f1`, and the step at which each peaked. This makes the "early-stop on edit_f1" decision discoverable from the ledger alone. ~5 min.

---

## Next experiments (priority order)

The paper-track ranking after this baseline:

1. **`qwen-edit-stop-seed42`** — Track B with `max_steps=300`, identical config otherwise. Verify that `macro_edit_f1=0.2167` at step 300 is reproducible (not a one-seed artifact); also gives the paper's headline Track B number under the "best stopping rule" framing. ~50 min A100.
2. **`qwen-r16-seed42`** — Track B with `lora_rank=16`, `max_steps=500`. Tests whether higher rank closes the strict gap without sacrificing lenient performance. ~80 min A100.
3. **`track-a-edit-backfill`** — re-evaluate `checkpoints/best/best.pt` (Track A's best) against val with the new `compute_metrics(...)` to backfill `macro_edit_f1` for all 4 Track A entries. Apples-to-apples comparative table becomes possible. ~40 min on A100, all Python (no training).
4. **Per-field breakdown** — split `macro_f1` and `macro_edit_f1` by field, both tracks. Identify which fields each method wins on (likely: Qwen wins OCR-heavy fields like `batch_number`/`expiry_date`/`mrp`; Track A wins structured fields where the strict format alignment is naturally tighter). This is the load-bearing per-field decomposition for the paper. ~30 min on A100, mostly metric code.
5. **`qwen-r4-edit-stop-seed42`** — Track B with `lora_rank=4`, `max_steps=200`. The "minimal-data minimal-params" point on the ablation grid; hypothesis is that fewer params = less specialization-overfitting headroom = lenient peak holds for longer. ~30 min A100.

The Phase 7 self-supervised pretraining (MAE on 2223 unlabeled images) remains in the queue but is now lower-priority than (1)–(5), since the comparative-framing finding is already in hand and pretraining-then-fine-tuning is a 4-hour experiment.

---

## Status

- T1–T8 complete; ledger entry written; leaderboard regenerated.
- T9 = this document.
- Next: tag `track-b-baseline-complete`, push, then schedule (1)–(3) above.

---

## Postscript — 2026-05-06: Track A edit_f1 backfill changes the conclusion

After this review was written, we re-ran Track A baseline-seed44 (run_id `baseline-seed44-rerun`) to capture `macro_edit_f1` for the same model — necessary because Track A's original Phase 2 runs predate the edit metric. The result substantially revises the comparative framing above.

**Re-run result:** `macro_f1=0.0804` (byte-identical to original; deterministic seed reproduced exactly), `macro_edit_f1=0.4094`.

**Updated comparative table:**

| Method | macro_f1 (strict) | macro_edit_f1 (lenient) | edit/strict ratio |
|---|---|---|---|
| **Track A** baseline-seed44 | **0.0804** | **0.4094** | **5.1×** |
| Track B step 300 (peak edit) | 0.0123 | 0.2167 | 17.6× |
| Track B step 400 (peak strict so far) | 0.0159 | 0.1165 | 7.3× |

**Track A wins on BOTH metrics.** Strict: by 5.1× (0.0804 vs 0.0159). Lenient: by 1.9× (0.4094 vs Track B's peak 0.2167). The "Track B wins lenient" claim earlier in this review was based on the unstated assumption that Track A's edit_f1 was much lower — it isn't.

**Refined narrative:**

The original framing (sections "Paper-relevant findings" #1 and "Decision tree" above) treated this as a "method-by-metric tradeoff" — Track A wins strict, Track B wins lenient, choose your weapon. That framing is now wrong. The corrected framing:

1. **Track A wins by every metric we can measure in the 500–5000 image regime.**
2. **Track B's much-larger backbone (2.2B vs 26.5M params) does NOT compensate for its overfitting-to-format dynamic in this small-data regime.**
3. **The strict-vs-lenient gap is real, but it's about METRIC SENSITIVITY, not about which method to pick.** Both methods produce content with formatting drift; lenient matching always rewards both more generously than strict; the LENIENT/STRICT RATIO measures how aggressive the formatting-drift overhead is.
4. **Track B at step 300 is in a genuinely degenerate regime** — 17.6× ratio is much higher than Track A's 5.1× because Track B is outputting "almost-right content in completely-wrong format". Step 400 is recovering toward a more typical 7.3× ratio.

The decision tree above still stands for one specific recommendation: **early-stop on edit_f1, not strict_f1, when fine-tuning a pretrained VLM with LoRA on small data.** That's still useful guidance for practitioners. But the broader "use Track B for lenient quality" claim is retracted.

**What this means for the paper:**

The paper's contribution is now subtler than "the choice of stopping metric matters more than the choice of method" — that statement IS still true (within Track B), but at the method level, Track A unambiguously wins. The paper's headline result becomes:

> **In the 500–5000 image small-data pharma regime, a custom hybrid model trained from scratch on top of a frozen pretrained vision encoder (SigLIP+Donut, 26.5M trainable params) outperforms LoRA-fine-tuned pretrained VLMs (Qwen2-VL-2B, ~5M trainable LoRA params) on every metric. The pretrained-VLM approach exhibits a non-monotonic eval trajectory under lenient metrics that requires careful early-stopping but does not change the relative ordering.**

That's still a publishable finding — it's a useful negative result for practitioners who would otherwise default to "just LoRA-fine-tune a big VLM" thinking. And the per-field breakdown (queued experiment #4 in this review) may yet reveal field-specific wins for Track B that change the picture.

**Important prior-work connection (added 2026-05-07):**

Malepati, Nandamury, Manjunath, Rajan, Prabhune (2026), "Comparative Evaluation of YOLOv12 and SAHI for Medication Identification in Hospital Pharmacies" (IITCEE 2026, IEEE), is on the SAME 48 MP smartphone-photo dataset that our 564/111/111 split is drawn from, and reports a 17× lift on the four OCR-critical classes (Batch, MRP, Manufacturing Date, Expiry) from 0.035 to 0.609 macro AP@0.5 by adding SAHI tiled inference (1024-px tiles, 0.35 overlap, base 1280) on top of YOLOv12. This is direct empirical evidence that the small-text-at-low-resolution problem we observe in Tracks A and B is **solvable by resolution-aware inference, not just better representation**. Three new entries added to research_directions.md (#11 SAHI tiled inference for Track A at inference time; #12 multi-tile MAE pretraining; #13 Track D = full SAHI+OCR pipeline as a competitor track). Track D is the largest paper-realignment move and reframes the contribution from "Track A vs Track B vs Track C end-to-end" to "end-to-end VLMs vs modular SAHI+OCR pipeline." The Malepati 2026 paper becomes our published prior-work baseline.

**Re-prioritized next experiments:**

1. **Per-field breakdown of Track A vs Track B** — same models, same val set, broken down by field. Find any fields where Track B beats Track A. Likely candidates: `batch_number`, `expiry_date`, `mrp` (OCR-heavy fields where Qwen's pretrained vision encoder might shine on small text).
2. **`qwen-r16-seed42`** — Track B with `lora_rank=16`, max_steps=500. Tests if higher rank closes the strict gap to under 0.05 — even partial closure makes the per-field story more interesting.
3. **`qwen-edit-stop-seed42`** — Track B with `max_steps=300` to verify the step-300 edit_f1=0.2167 is reproducible. Now also tests whether stopping at the edit peak yields a *useful* model (vs just a peak-on-this-metric one).
4. (Lower priority, was previously priority 3) Backfill Track A baseline-seed42, baseline-seed43 with edit_f1. Confirms the 5.1× ratio is consistent across seeds. Less urgent now that we have one clean apples-to-apples data point.

The Phase 7 self-supervised pretraining (MAE on 2223 unlabeled images) becomes more relevant under this revised framing — if Track A is the strong baseline, an MAE-pretrained Track A is the natural next move, expected to push macro_f1 from 0.080 toward 0.10+.
