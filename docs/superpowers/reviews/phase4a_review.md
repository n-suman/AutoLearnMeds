# Phase 4a Review — Explore-Phase Rails

**Tag:** `phase-4a-rails-complete`
**Date:** 2026-05-05
**Status:** GREEN — explore loop runs end-to-end on Colab; one experiment (randaug-m5-seed42) ran via the new rails and was correctly reverted.

## What was built

- **`program.md`** — full agent contract (10 sections including the notes.md template + creative-reading discipline). Replaces the Phase-0 skeleton.
- **`program_explore.md`** — explore-phase override: 15-min budget, +0.001 keep threshold, 9-item priority backlog drawn from Phase 2 review.
- **`train.py`** — `train_loop` now saves a full state dict (proj + decoder blocks + embedding + final_ln + config) at best macro_f1, not just `{step, macro_f1}` metadata. New `self_named_params` helper handles the DecoderBlock pseudo-module's parameter naming.
- **`scripts/append_ledger.py`** — atomic ledger append per spec §5.4. Reads `metrics.json` + `notes.md`, computes `kept` based on `current_best + threshold`, writes one JSON line to `experiments/ledger.jsonl`.
- **`scripts/leaderboard.py`** — regenerates `experiments/leaderboard.md` from the ledger, sorted by macro_f1 descending.
- **`scripts/promote.sh`** — copies a kept run's `train.py` + `config.yaml` + `best.pt` to `checkpoints/best/` and appends to a manifest.
- **`scripts/finalize_experiment.sh`** — orchestrates append_ledger → leaderboard → promote-if-kept → git commit. The agent's only post-run command.
- **11 lightweight tests** (5 append_ledger, 3 leaderboard, 2 smoke for promote+finalize, 1 full-contract program.md). Total local suite: 74 PASSED, 18 colab-marked deselected.

## What was verified end-to-end

A real explore experiment ran through the rails on Colab A100:

1. **Pre-experiment:** `notes.md` written with hypothesis + citation + adjacent ideas + predicted direction.
2. **Edit:** `train.py` patched to add `RandAugment(M=5, N=2)` in the `forward()` hook (training mode only). Diff was self-documenting via `# AGENT (randaug-m5-seed42):` markers.
3. **Run:** `bash scripts/run_experiment.sh randaug-m5-seed42 --config experiments/configs/baseline.yaml --seed 42 --no-wandb` → wrote `metrics.json` after 1674s (~28 min).
4. **Finalize:** `bash scripts/finalize_experiment.sh randaug-m5-seed42 --phase explore --parent baseline-seed44`:
   - `append_ledger.py` → REVERTED (-0.0125 below current floor 0.0804)
   - `leaderboard.py` → regenerated; randaug-m5-seed42 ranked 3rd
   - `promote.sh` → correctly skipped (entry kept=false)
   - `git add -f + commit` → after the `-f` fix (see below)
5. **Revert:** `git checkout train.py` on Colab restored the baseline.

The 4-row leaderboard now reads:

| rank | run_id | phase | macro_f1 | kept |
|---|---|---|---|---|
| 1 | baseline-seed44 | baseline | 0.0804 | ✓ |
| 2 | baseline-seed42 | baseline | 0.0679 | ✓ |
| 3 | randaug-m5-seed42 | explore | 0.0679 | ✗ |
| 4 | baseline-seed43 | baseline | 0.0646 | ✗ |

## Issues encountered + root-cause fixes

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | leaderboard.py crashed on entries with empty `hypothesis` | `"".splitlines()` returns `[]` not `[""]`; indexing `[0]` raises | Implementer caught and fixed inline during T4 |
| 2 | `finalize_experiment.sh` reported "nothing staged to commit" despite new files in run dir | `git add` (no `-f`) defers to `.gitignore`; the `experiments/` parent rules suppressed everything despite the `!experiments/runs/*/notes.md` etc. negations not catching during plain add | Added `-f` and explicit per-file paths in `finalize_experiment.sh`; staged the missed files manually for this commit |

## RandAugment M=5 result — what we learned

- **Hypothesis:** "+0.05 to +0.15 macro_f1 over the 0.0710 baseline" (predicted).
- **Result:** 0.0679 — exact match with baseline-seed42's score; **delta 0.0**.

This is itself a research finding:

1. **The training-noise floor is around 0.07** for this configuration. RandAugment M=5 didn't shift it.
2. **Plausible reasons for null result** (per the experiment's `notes.md` retrospective):
   - M=5 may be too weak for this regime; RandAugment paper noted optimal M correlates with model capacity, but our fundamental constraint is the FROZEN encoder. Augmenting inputs to a frozen encoder produces different SigLIP features, but those features still get "memorized" by the small decoder.
   - Some RandAugment ops (color jitter, contrast, posterize) may *corrupt* color-coded packaging features that distinguish medicines. RandAugment's default op set wasn't designed for fine-grained color-sensitive labels.
   - 1000 steps may be insufficient to see augmentation's regularization benefit before the decoder memorizes.
3. **Next experiments queued from the retrospective:** sweep M=8/9/12; try a per-op-subset RandAugment that excludes color manipulations; combine RandAugment with longer training.

This is exactly the kind of falsification-then-iterate cycle the autoresearch methodology is designed for. Phase 4b — the actual sweep — will run dozens of these.

## Open items / deferrals

- **Crash-mid-experiment recovery** still requires `train.py` to write `experiments/_active.json`. Wired up to `resume_or_start.py` only as a state-machine; the `train.py` writer side is Phase 4b nice-to-have.
- **Self-driving loop.** Currently the agent's iteration is human-driven (a Claude Code session reads `program.md` and runs scripts). Spec §7.4 mentions Option-B scheduled wake-ups via the `schedule` skill; defer to Phase 4b once we want sustained overnight runs.
- **`RESUME_NEEDED.md` flow** for token-quota approach not implemented; agent doesn't introspect Claude rate limits. Manual user re-run on quota refresh is the current fallback.
- **Colab git push** still fails (the bootstrap doesn't configure GitHub auth). Workflow: scp artifacts to Mac, commit + push from Mac. Acceptable for now; could be Phase-4b improvement to install a deploy key on the Colab side.

## Next step

**Phase 4b — actual explore sweep.** The rails are operational; the human (or a long-running Claude Code session) can now sustain experimentation. To start:

> Open Claude Code in the repo and prompt: *"Read program.md and program_explore.md. Run the next experiment from the priority backlog. Use the workflow per experiment exactly."*

The agent then iterates. Each iteration is one ledger entry. Phase 4b's exit criterion: ≥1 confirmed (3-seed) win that beats baseline by ≥0.02 macro_f1 — that triggers Phase 5 (confirm phase).

Until then, the leaderboard's `current floor` is **0.0804**. The agent's job is to beat it.
