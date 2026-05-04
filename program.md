# Pharma-VLM Autoresearch — Baseline `program.md`

You are an autonomous research agent operating on this repo. Your job: iterate on `train.py` to maximize a single scalar metric.

## Goal

Maximize `val_macro_f1` of structured field extraction from pharmaceutical product label images, by editing `train.py`.

## Metric

Single scalar: `val_macro_f1`. Higher is better. Reported by `train.py` as the literal stdout line `final_macro_f1=X.XXXX` on the last line of training. Macro-F1 over fields, with field-aware normalization. See spec §1.3.

## Time budget

- **Phase 1 (explore):** 15 minutes wall-clock per experiment (excluding compile/startup).
- **Phase 2 (confirm):** 60 minutes wall-clock per experiment, 3 seeds.

## Allowed (you may modify in `train.py`)

- Architecture: layers, hidden dim, heads, FFN ratio, activation, dropout, position encoding, cross-attention pattern, layer-norm placement (pre vs post), tied embeddings.
- Optimizer: AdamW vs Lion vs Muon, betas, weight decay, schedule shape, warmup, peak LR.
- Loss: label smoothing, token weighting, auxiliary losses.
- Augmentations & data: RandAugment, AugMix, hard-example mining, curriculum, oversampling.
- Tokenization variants that don't retrain the BPE.

## Forbidden

- Modifying `prepare.py`.
- Modifying `evaluate()`.
- Retraining the BPE tokenizer.
- Changing the test set or the metric definition.
- Reading `data/test.jsonl` or invoking `evaluate_test()`.
- Swapping the encoder out of the SigLIP family without a top-level run-tag.

## Source vetting

Every change MUST cite at least one paper from `papers/` or add a new paper to `papers/` with a one-paragraph summary in `papers/README.md`. If a change is heuristic (no paper), label it `--exploratory--` and run it under a separate run-tag.

## Workflow per experiment

1. Read `experiments/ledger.jsonl` tail; pick a direction not recently tried.
2. Edit `train.py` with a minimal diff (one knob).
3. Add citation + hypothesis to `experiments/runs/<run_id>/notes.md`.
4. Run: `bash scripts/run_experiment.sh`.
5. Read final metric from `experiments/runs/<run_id>/metrics.json`.
6. If improved over current best: `bash scripts/promote.sh <run_id>`. Else: revert `train.py`.
7. Append ledger entry; commit; push; sync to GCS.

## Backup

After every experiment, `scripts/sync_to_gcs.sh` runs automatically. If Colab disconnects or your token quota approaches its limit, write `RESUME_NEEDED.md` with `next_planned_action`, then exit cleanly. The next session reads it and picks up.

## Do not

- Combine multiple changes in one experiment ("one knob per experiment").
- Continue past the time budget.
- Skip the `notes.md` or the citation.
- Touch the test set.
