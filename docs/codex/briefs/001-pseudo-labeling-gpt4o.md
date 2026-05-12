# Brief 001: Pseudo-label the unlabeled pharma images with GPT-4o vision

**Owner:** Codex
**Created:** 2026-05-13 by Claude
**Estimated effort:** M (single-day, mostly waiting on API)
**Estimated cost:** $20–$80 in OpenAI API spend

## Context (self-contained)

The project is `AutoLearnMeds` — a research effort + production-leaning system to extract structured information (12 fields) from photographs of pharmaceutical product labels (ampoules, blister packs, strip foils, syrup bottles, vials, sachets, cartons, sprays, tubes, jars).

**Where we are:** the user has 786 labeled images (564 train / 111 val / 111 test, deterministic seed 42). After five model-architecture experiments (Tracks A–D documented in `paper/main.md`), our best model achieves only **macro_f1 = 0.074 (strict) / 0.32 (lenient)** on the val split. The bottleneck is data: 564 train images is too few for a 12-field structured extractor.

**The unlabeled bucket:** `gs://auto_learn_meds/raw/raw_images/` contains **~3,061 raw images total**, of which **786 are labeled** (used in train/val/test). The remaining **~2,275 are unlabeled** and currently used only for MAE self-supervised pretraining (Track C — which underperformed). They have never been used as labeled training data because pharmacist-grade annotation is expensive.

**The opportunity:** GPT-4o vision has strong OCR + reasoning over packaging text. If we have it generate 12-field XML labels on the 2,275 unlabeled images and filter for quality, we can plausibly 3–4× the labeled training set with one paid API run for $50-ish. This is the highest-leverage move available to us right now.

## The 12-field XML schema

The exact format used by the existing training pipeline (do NOT deviate — the trainer's tokenizer is built around this):

```xml
<medication>
  <brand_name>Crocin</brand_name>
  <generic_name>Paracetamol</generic_name>
  <drug_name>Crocin Advance</drug_name>
  <strength>500 mg</strength>
  <quantity>15 tablets</quantity>
  <company>GSK Consumer Healthcare</company>
  <manufacturer>GSK Asia Pvt Ltd, India</manufacturer>
  <batch_number>BC4521A</batch_number>
  <manufacturing_date>11/2023</manufacturing_date>
  <expiry_date>10/2025</expiry_date>
  <mrp>30.00</mrp>
  <warnings>Keep out of reach of children. Do not exceed recommended dose.</warnings>
</medication>
```

**Field semantics — critical for prompting GPT-4o correctly:**

- `brand_name`: marketed product name. Often largest text on the pack.
- `generic_name`: pharmacological / INN name (e.g. "Paracetamol").
- `drug_name`: full marketed name including variant (e.g. "Crocin Advance"); when unclear, can equal brand_name.
- `strength`: dose per unit including units (e.g. "500 mg", "10 mg/5 ml").
- `quantity`: pack contents (e.g. "10 tablets", "60 ml", "1 vial").
- `company`: top-level pharma company (the brand owner; often differs from manufacturer).
- `manufacturer`: contract manufacturer / "Mfd by" entity, often with address.
- `batch_number`: alphanumeric, usually small print. Labels may say "B.No.", "Batch", "Lot".
- `manufacturing_date`: format varies — MM/YYYY most common. Labels say "Mfg.", "Mfd.", "MFG".
- `expiry_date`: similar formats. Labels say "Exp.", "EXP", "Use by".
- `mrp`: Indian Maximum Retail Price; numeric INR. Often prefixed "₹" or "MRP Rs."
- `warnings`: any cautionary text printed on the pack.

**If a field is not visible / not present on the pack: emit an empty tag like `<batch_number></batch_number>`.** Do not invent values.

## Step-by-step

1. **Establish access** to `gs://auto_learn_meds/` (user has gcloud auth set up locally; if not, ask before starting). Verify with `gsutil ls gs://auto_learn_meds/raw/raw_images/ | head`.

2. **Get the labeled-image manifest:**
   - `gs://auto_learn_meds/raw/train.jsonl`, `val.jsonl`, `test.jsonl` — each row has `image_path` and `xml_label`.
   - Local copies likely exist in the repo under `raw/` or similar; check `ls raw/ 2>/dev/null` and `cat raw/train.jsonl | head -1`.
   - The set of labeled image filenames is the union of these three.

3. **Compute the unlabeled set:** `unlabeled = all_in_bucket - labeled`. Expected size ≈ 2,275.

4. **Calibration first (do this BEFORE the full run):** Run your prompt + GPT-4o pipeline on the **111 validation images** that have gold labels. Compute strict + lenient F1 of GPT-4o predictions vs gold using the project's existing eval function `prepare.compute_metrics(predictions, truths)` — see `prepare.py`. **Stop and report if GPT-4o's lenient F1 on val is < 0.40.** If it is below that, the pseudo-labels are noisier than the current model's predictions and won't help training. Use this calibration to write `status/001-...md` with concrete numbers before spending the full budget.

5. **Run on the full unlabeled set.** Suggested parallelism: 8 concurrent API calls. Use exponential backoff on 429s. Resume-from-checkpoint via a sidecar `progress.jsonl` so a mid-run failure doesn't lose work.

6. **Per-image, also capture:**
   - `model_version`: e.g. `"gpt-4o-2024-11-20"`
   - `prompt_hash`: SHA of the prompt for traceability
   - `confidence`: GPT-4o's self-rated confidence per field (ask for it in the response schema; bucket as low/med/high)
   - `cost_usd`: per-image API cost
   - `latency_sec`

7. **Output format** — write to `data/pseudo_labels/round_001.jsonl`, one row per unlabeled image:
   ```json
   {"image_path": "...", "xml_label": "<medication>...</medication>", "per_field_confidence": {...}, "model_version": "gpt-4o-2024-11-20", "generated_at": "2026-05-13T14:00:00Z", "cost_usd": 0.018}
   ```

8. **Quality spot-check:** sample 30 random rows, open the images visually, eyeball the predicted XML against what's on the pack. Note disagreements in the status file.

9. **Write `scripts/codex/pseudo_label_gpt4o.py`** as a reusable, idempotent script. Future rounds (Round 002 with a different model, Round 003 after human correction, etc.) reuse the same code.

## Prompting tips

- Include 3 gold-labeled examples (input image + correct XML) in the prompt as few-shot context. Pull them from `train.jsonl` choosing 3 different pack types.
- Tell GPT-4o to output **valid XML and nothing else** — no preamble, no explanation.
- For the date fields, include the line "If the date format is unclear, emit it in MM/YYYY."
- For `mrp`: "Emit as a bare number in INR. Strip ₹ or 'Rs.' prefixes."
- For `warnings`: "Concatenate multiple warning lines with '. ' separator."
- Ask for per-field confidence as a final JSON block AFTER the XML.

## Acceptance criteria

1. ✅ `data/pseudo_labels/round_001.jsonl` exists with ~2,275 rows.
2. ✅ Calibration report at the TOP of `status/001-pseudo-labeling-gpt4o.md`: GPT-4o macro_f1 (strict + lenient) on the 111 val images, plus per-field breakdown for the safety-critical 4 (batch_number, expiry_date, mrp, manufacturing_date).
3. ✅ `scripts/codex/pseudo_label_gpt4o.py` committed — reusable for future rounds.
4. ✅ Total cost reported in the final status block.
5. ✅ Spot-check notes for 30 random samples (which fields GPT-4o got reliably, which it failed on).

## Anti-goals — explicitly DO NOT

- ❌ Do not train any model on the pseudo-labels. That's Claude's next task.
- ❌ Do not modify `train.py`, `train_qwen.py`, `pretrain_mae.py`, `prepare.py`, or any config in `experiments/configs/`.
- ❌ Do not modify `experiments/ledger.jsonl` (Claude's running experiment record).
- ❌ Do not commit pseudo-labels into `experiments/` or any path that the existing trainers will accidentally pick up. Keep them in `data/pseudo_labels/`.
- ❌ Do not use OpenAI models other than GPT-4o without checking first (cost / quality tradeoff calibration needed).
- ❌ Do not exceed $80 total spend without checking. If calibration on val suggests poor quality, stop early and report.

## Where to write

| File | Purpose |
|---|---|
| `scripts/codex/pseudo_label_gpt4o.py` | The reusable labeling script |
| `data/pseudo_labels/round_001.jsonl` | The pseudo-labels |
| `data/pseudo_labels/round_001.progress.jsonl` | Resume-from-checkpoint sidecar |
| `data/pseudo_labels/round_001.calibration.json` | The val-set F1 numbers |
| `status/001-pseudo-labeling-gpt4o.md` | Status updates + final summary |

## After completion

Claude will:
- Read the calibration F1.
- If quality is acceptable, add a new training config that mixes gold + pseudo-labels with a sample-weight or confidence-threshold filter.
- Run a Track A retrain experiment (call it Track E or A++) on the expanded data.
- Report results back to the user.
