# Phase 3 Review — Disaster-Recovery Drill

**Tag:** `phase-3-complete`
**Date:** 2026-05-05
**Status:** GREEN — automated drill passes on Colab in 26s.

## What was built

- `scripts/disaster_recover.sh` — pulls `experiments/` and `checkpoints/` from `gs://${BUCKET}` after a Colab nuke. Complementary to `colab_bootstrap.sh` (which restores code from GitHub, deps via uv, and processed JSONLs from GCS golden_set). gsutil-discovery handles Colab's non-standard install path (`/tools/google-cloud-sdk/bin/gsutil`). Non-destructive `gsutil -m rsync` (no `-d`) so local newer files survive.
- `tests/test_disaster_recovery.py` — Colab-marked end-to-end drill: snapshot SHA-256 of every file under `experiments/runs/` → force-sync to GCS → `rm -rf` local → run `disaster_recover.sh` → re-snapshot → assert no missing/extra/mismatched files.

## What was verified

- ✅ Pre-existing `sync_to_gcs.sh` daemon successfully mirrored all three Phase-2 baseline run directories (`baseline-seed42/43/44/`) to `gs://auto_learn_meds/experiments/runs/`.
- ✅ `scripts/disaster_recover.sh` runs cleanly on Colab — pulls 3 run dirs back from GCS in seconds.
- ✅ Hash-compare confirms byte-identity: every file's SHA-256 in the recovered tree matches its pre-nuke value.
- ✅ All 18 colab-marked tests still pass after the drill (no collateral damage).
- ✅ `make verify` would still be green (we didn't disturb the local Mac state).

The drill ran on the Phase 2 ledger (3 seeds × 4 files each = 12 files plus directory metadata), end-to-end in 26.49s.

## Issues encountered + root-cause fixes

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | First test attempt failed with `FileNotFoundError: 'gsutil'` | Colab installs gsutil at `/tools/google-cloud-sdk/bin/` but doesn't put that directory on the default PATH for new shells or `uv run` subprocesses. The existing `sync_to_gcs.sh` daemon found it because the bootstrap step had it on PATH at launch and the daemon inherited that environment | Added gsutil discovery in both `disaster_recover.sh` and `test_disaster_recovery.py`: try `command -v gsutil` first, then fall back to known Colab install paths. Fail loudly if not found. |

## Gaps documented (deferred to Phase 4+)

- **Crash-mid-experiment drill not implemented.** Spec §9.3 listed it as a sub-item, but `resume_or_start.py` is currently called by the autoresearch loop only — `train.py` doesn't write `experiments/_active.json` mid-run. When the autoresearch loop wires this up in Phase 4, the crash drill becomes a natural unit test on top of the existing `test_resume_state_machine.py`. Currently a `train.py` crash mid-run produces no `metrics.json`; `resume_or_start.next_action()` returns `START_NEW`, which means we re-run from scratch — correct behavior, just less efficient than checkpoint-resume.
- Daily snapshot script (`scripts/daily_snapshot.sh`) referenced in spec §7.7 not yet built. Defers to Phase 7 (it's a periodic-cron concern, not blocking research).
- The disaster-recovery test currently doesn't restore checkpoints/ because we haven't saved any model checkpoints yet (Phase 2 only saved `best.meta.pt`, not the actual weights). Phase 4 should add full state-dict save; the drill would then naturally cover checkpoint recovery.

## Compute used

Negligible. ~30 seconds A100-attached time for the test run; no GPU work.

## Observations from the drill

The recovery is fundamentally **tier-redundant**: we have 4 layers of persistence:
1. **Local** Colab disk (`/workspace`)
2. **GitHub** (code + plan/spec/review docs + ledger metadata committed periodically)
3. **GCS bucket** (live mirror of `experiments/` and `checkpoints/` via `sync_to_gcs.sh` every 5 min)
4. **Daily snapshots** (cron — not yet implemented per gap above)

A "Colab nuke" loses tier 1 only. `colab_bootstrap.sh` restores code + deps from tier 2, processed data from the GCS source-of-truth, and `disaster_recover.sh` restores the ledger + checkpoints from tier 3. This means **the most we can lose is whatever was written in the last 5 minutes** before the nuke (the sync_to_gcs daemon polling interval).

For the autoresearch loop, this is acceptable: the worst-case data loss is one in-flight experiment's mid-run state, which is recoverable by re-running. The atomic-experiment unit (`_active.json` + ledger entry) was designed in spec §7.3 to make this a graceful restart, not a crash.

## Next step

**Phase 4 — explore-phase autoresearch sweep** (per spec §9.4). The agent runs ~30 short experiments at 15 min each, optimizing macro_f1 from the 0.0710 baseline floor. Starting backlog (from Phase 2 review):

1. RandAugment (M=5-7 for our small data) — direct hit on overfitting
2. AugMix
3. Label smoothing 0.1
4. Decoder dropout sweep (0.1 / 0.2 / 0.3)
5. Beam search width=4
6. Untied embeddings (testing whether tied helps with our tiny vocab)
7. Pre-LN vs post-LN
8. RoPE on subset of head_dim
9. Donut "task tokens" (`<pack=...>` / `<view=...>` decoder prefix)

Phase 4 also needs:
- `program.md` finalized (autoresearch agent contract; spec §5.3 has the skeleton)
- `program_explore.md` override (spec §5.3)
- An autoresearch driver — either Claude Code self-loop via the `loop` skill or scheduled via `schedule` skill
- Full state-dict save in `train_loop` (so confirm-phase can resume from best checkpoint)

Phase 4 is where the autoresearch methodology actually starts producing scientific output.
