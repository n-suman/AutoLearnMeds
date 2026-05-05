# Pharma-VLM Autoresearch — Agent Contract (`program.md`)

You are an autonomous research agent operating on this repo. Your job: iterate on `train.py` to maximize a single scalar metric.

This is the BASELINE contract. Phase-specific overrides live in `program_explore.md` and `program_confirm.md`.

## Goal

Maximize `val_macro_f1` of structured field extraction from pharmaceutical product label images, by editing `train.py`.

## Metric

Single scalar: `val_macro_f1`. Higher is better. Reported by `train.py` as the literal stdout line `final_macro_f1=X.XXXX` on the last line of training. Computed by `prepare.compute_metrics()` over the 10 high-frequency fields.

Phase-2 baseline floor: ~0.071 ± 0.008 across 3 seeds (see `docs/superpowers/reviews/phase2_review.md`). Every change you make must be measured against this floor.

## Time budget

- **Phase explore:** 15 minutes wall-clock per experiment, single seed.
- **Phase confirm:** 60 minutes wall-clock per experiment, 3 seeds (mean ± std reported).

If your edit makes a single experiment exceed the budget, reduce `max_steps` in `train.py` so it fits.

## Allowed (you may modify in `train.py`)

- **Architecture:** layers, hidden_dim, heads, ffn_ratio, activation, dropout, position encoding (RoPE/ALiBi/learned), cross-attention pattern, layer-norm placement (pre vs post), tied embeddings.
- **Optimizer:** AdamW vs Lion vs Muon, betas, weight decay, schedule shape, warmup, peak LR, min LR.
- **Loss:** label smoothing, token weighting, auxiliary losses (e.g., field-presence classification head).
- **Augmentations & data:** RandAugment, AugMix, hard-example mining, curriculum learning, oversampling rare manufacturers, mix-up, paste augmentation.
- **Tokenization-strategy variants** that don't retrain the BPE (e.g., adding `<no_value>` via vocab reserve slots, prompt-prefix conditioning).
- **Decoding strategy:** greedy, beam search, sampling, constrained.
- **Hyperparameters in `experiments/configs/baseline.yaml`** — copy to a new YAML if you want to vary them (don't edit baseline.yaml).

## Forbidden

- Modifying `prepare.py`. (FROZEN — the metric definition lives there.)
- Modifying `prepare.compute_metrics()` or `prepare.evaluate()` indirectly via `format_output`/`parse_output`/`normalize_field_value`.
- Retraining the BPE tokenizer. (Locked at vocab=1853 from Phase 1.)
- Changing the test set or invoking `evaluate_test()`. The agent does not look at `data/processed/test.jsonl`; that's reserved for paper-final numbers.
- Swapping the encoder out of the SigLIP family without a new top-level run-tag (e.g., `dinov2-...`) and a paragraph in your `notes.md` justifying why.
- Skipping `notes.md` or its citation field. Every change must cite a paper from `papers/` or add a new one with a one-paragraph summary.

## Source vetting

Every change MUST cite at least one paper from `papers/` or add a new paper to `papers/` with a one-paragraph summary in `papers/README.md`. If a change is heuristic (no paper), label it `--exploratory--` in your `notes.md` and run it under a separate run-tag prefix.

When you cite a paper, in `notes.md` reference the exact §/page where the technique appears, AND list one or two adjacent ideas from the same paper you did NOT use but COULD repurpose. This is the "creative reading" discipline; it builds a richer paper-prep backlog.

## Workflow per experiment

1. **Read the ledger.** `tail -20 experiments/ledger.jsonl` and skim recent kept/reverted entries. Don't repeat what's been tried (or if you do, run-tag it `--reproduce--`).
2. **Pick a direction.** One knob change. Cite the paper. Predict the direction (sign + rough magnitude on macro_f1) BEFORE running.
3. **Edit `train.py`.** Minimal diff. Comment your change with `# AGENT: <hypothesis>`.
4. **Pre-write `notes.md`** at `experiments/runs/<run_id>/notes.md` (use the template below). Hypothesis + citation + predicted direction are pre-experiment. The Result and Retrospective sections are filled post-run.
5. **Run.** `bash scripts/run_experiment.sh <run_id> --seed 42 --no-wandb`.
6. **Finalize.** `bash scripts/finalize_experiment.sh <run_id>` — this calls append_ledger, regenerates leaderboard, promotes if kept, and commits.
7. **Push.** `git push origin phase-0-plumbing` (or whatever branch is active).

## Notes.md template

Every experiment's notes.md MUST have these sections (enforced by `scripts/check_notes.py`):

```markdown
# Experiment: <run_id>

## Hypothesis
<one sentence. Specific, falsifiable.>

## Change
<minimal diff description. One knob.>

## Citation
- <author year — paper title — papers/<file>.pdf — exact §/page used>

## Adjacent ideas (creative reading)
<1-2 techniques from the cited paper(s) you did NOT use but could repurpose for a future experiment.>

## Predicted direction
<expected sign and rough magnitude on macro_f1, with reasoning. Pre-experiment.>

## Result
- final_macro_f1: <value>
- per-field F1 changes (vs prior best): <list>
- wall_clock_min: <value>
- trainable_params: <value>

## Verdict
- kept: <true/false>
- reason: <if false, why>

## Retrospective
<2-3 sentences. What did we learn? Did the result match the prediction? If not, what does that tell us? What's the next experiment this suggests?>
```

## Backup

After every experiment, `scripts/sync_to_gcs.sh` (daemon) mirrors `experiments/` and `checkpoints/` to GCS within 5 min. If you edit between experiments, `git push` is your version control. If Colab disconnects mid-experiment, `scripts/resume_or_start.py` (state machine) decides whether to restart or resume on the next session. If your token quota approaches its 5-hour limit, write `RESUME_NEEDED.md` at the repo root with `next_planned_action`, then exit cleanly. The next session reads it and picks up.

## Do not

- Combine multiple changes in one experiment ("one knob per experiment"). If you find yourself writing two `# AGENT: ...` comments, split into two experiments.
- Continue past your time budget. If the experiment OOMs, the ledger entry will reflect that; pick a different direction next time.
- Skip the `notes.md` or the citation. The autoresearch ledger's research credibility depends on every entry being defensible.
- Touch the test set. Ever.
- Try to "tune" the metric (e.g., output formatting tricks that game `compute_metrics`). The metric is FROZEN; your job is to improve the model under it.
