# Phase 1 Review — Data Ingestion & Schema Lock

**Tag:** `phase-1-complete`
**Date:** 2026-05-05
**Status:** GREEN — `evaluate()` runs end-to-end on real val (111 records) with a stub model; 68 tests pass on Colab (60 lightweight + 8 colab-marked).

## What was built

- `scripts/build_processed.py` — converts `gs://auto_learn_meds/raw/golden_set/{gold_standard.jsonl,splits.json}` + `raw_images/` into `data/processed/{train,val,test}.jsonl` with `image_id`, `image_path`, `image_hash` (sha256), and `split` fields added. Idempotent (sort-keyed JSON dump).
- `prepare.py` v1 (FROZEN — autoresearch agent must not modify):
  - **Constants:** `FIELD_ORDER` (12), `HIGH_FREQUENCY_FIELDS` (10), `SPECIAL_TOKENS` (26), `IMAGE_SIZE=224`, `TOKENIZER_VOCAB_SIZE=8192`.
  - **Output format:** `format_output(record)` → Donut-style XML; `parse_output(text)` → `{field: text}` via single compiled regex; tolerates truncation and unknown tags.
  - **Tokenizer:** `train_tokenizer(corpus, out_path, vocab_size)`, `get_tokenizer(path)` (BPE with all special tokens registered as added tokens).
  - **Metric:** `normalize_field_value` (lowercase + strip + whitespace collapse), `compute_field_f1` (with the "neither side has the field → no contribution" convention), `compute_metrics` (macro_f1 over HIGH_FREQUENCY_FIELDS only; per_field_f1 over all 12).
  - **Image pipeline:** `preprocess_image` (224x224 letterbox + SigLIP normalization; uses `AutoImageProcessor` to skip the unneeded SigLIP tokenizer / SentencePiece), `PharmaLabelDataset`, `get_dataloader`.
  - **Public API:** `evaluate(model, jsonl_path, ...)` and `evaluate_test(... explicit_consent=...)` (gated + audited).
  - All torch/transformers/Pillow/tokenizers imports are lazy (inside function bodies), so the module imports cleanly in a torch-less venv.
- `scripts/build_data_card.py` — auto-generates `data/data_card.md` (Gebru 2018 datasheet style: motivation, composition, splits, field presence from train, categorical distributions for view_type/pack_type/medicine_name).

## What was verified

- ✅ 60 lightweight pytest tests pass locally (Mac, no torch).
- ✅ 8 colab-marked pytest tests pass on Colab A100 (image preprocessing, DataLoader, evaluate, evaluate_test gated + audited).
- ✅ `build_processed.py` against real data: **564/111/162** train/val/test (matches the source `splits.json` exactly).
- ✅ BPE tokenizer trained on training-set field values (1745 unique values → vocab plateaus at 1853 tokens; all 26 special tokens registered).
- ✅ `data/data_card.md` auto-generated with field-presence and categorical distributions from real data.
- ✅ `evaluate()` runs on real val (111 records) with a stub `predict_text -> ""` model, returns `macro_f1=0.0000`, `n_examples=111`. **Phase 1 exit gate.**

## Issues encountered + root-cause fixes

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | T2 — `data/processed/.gitignore` couldn't be added | Root `.gitignore` had `data/processed/` ignoring the whole directory | Added `!data/processed/.gitignore` exception in root |
| 2 | Phase-1 exit gate — `SiglipTokenizer requires SentencePiece` | `AutoProcessor.from_pretrained("google/siglip-...")` loads BOTH the image processor AND the tokenizer; SigLIP's tokenizer needs SentencePiece (not in our deps) | Use `AutoImageProcessor` (image-only) — we have our own BPE for the decoder anyway |

Both bugs fixed inline during Phase 1; the latter was caught only by the on-Colab integration run (the colab-marked tests are deselected in the local Mac suite, so this was a real value-add of running the exit gate).

## Open items / deferrals

- **BPE vocab plateaus at 1853 tokens** (target was 8192). Corpus is small (1745 unique field values, ~50KB total). Phase 2 must decide: live with the smaller vocab, or augment the corpus with substring/character-level entries. Either is fine for a small decoder.
- Field-aware normalization (stripping "B.No.:", "M.R.P.Rs.:", parsing dates) deferred to Phase-7 sweep.
- 2223 unlabeled raw images sit unused; potential MAE/BYOL pretraining of the encoder is in the Phase-7 backlog.
- Polygons available in source but not used by the model; potential auxiliary localization loss is in the Phase-7 backlog.
- The Makefile's `check_drive` target prints "skipping" because the project moved off Drive in Phase 0; cosmetic, deferred to a later cleanup.

## Compute used

A few minutes on the A100 — 1 build_processed run, 1 BPE training, 1 stub evaluate(). Well under 0.1 GPU-hour.

## Next step

**Phase 2 — Baseline Model + Training Loop.** Implement the SigLIP-base + Donut-style decoder in `train.py`, run a single full training pass, confirm `final_macro_f1` is reproducible to ±0.005 across 3 seeds. The model wraps everything behind `predict_text(batch_images, max_new_tokens) -> list[str]` so it slots into the `evaluate()` interface.
