# experiments/

The autoresearch ledger — single source of truth for all experiments.

- `ledger.jsonl` — append-only, one JSON line per experiment. See spec §5.4 for schema.
- `runs/<run_id>/` — per-experiment artifacts: `train.py` snapshot, `config.yaml`, `metrics.json`, `stdout.log`, `notes.md`.
- `_active.json` — current planned/active experiment (transient; cleared after commit).
- `leaderboard.md` — auto-rendered ranked table of all kept experiments.

All files in this folder ARE tracked in git (small, critical for reproducibility); per-run checkpoints are NOT (large, mirrored to GCS instead).
