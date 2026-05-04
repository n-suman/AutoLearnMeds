# scripts/

Operational scripts for the project.

## Storage architecture

Three tiers, in order from cold to hot:

| Tier | Location | Access pattern | What lives here |
|---|---|---|---|
| **User-facing source of truth** | Google Drive (your folders) | You upload here | Raw images + golden_set JSON labels |
| **Canonical persistent home** | GCS bucket `gs://auto_learn_meds/` | Reads from any worker | `raw/` (mirror of Drive), `processed/`, `experiments/`, `checkpoints/` |
| **Hot training cache** | Colab local SSD `/content/cache/` | Random reads at training time | Pre-warmed copy of training data; lost on session restart |

Drive is the only tier where you upload by hand. Everything else is automated. Training never reads Drive directly — Drive's API throttles aggressively and gdrive-FUSE drops connections under random access. GCS-then-local-SSD is the production-grade pattern.

The bootstrap performs a one-time idempotent Drive→GCS mirror via `sync_drive_to_gcs.py` on first session, then reads from GCS thereafter.

## Scripts

| Script | Purpose |
|---|---|
| `colab_bootstrap.sh` | One-cell Colab session bootstrap (mounts, deps, Drive→GCS sync, SSH tunnel) |
| `sync_drive_to_gcs.py` | One-time idempotent copy of user's Drive folders to GCS (called by bootstrap) |
| `keepalive.py` | Heartbeat to prevent Colab idle disconnect |
| `sync_to_gcs.sh` | Periodic backup of `experiments/` and `checkpoints/` to GCS |
| `update_ssh_config.sh` | Refreshes Mac `~/.ssh/config` with new Colab cloudflared host |
| `resume_or_start.py` | State-machine-based experiment launcher |
| `build_processed.py` | Phase 1: raw images + labels → `processed/{train,val,test}.jsonl` |
| `build_data_card.py` | Auto-generate `data/data_card.md` |
| `build_paper_artifacts.py` | Auto-generate paper tables + figures |
| `build_bibtex.py` | `papers/README.md` → `paper/refs.bib` |
| `build_research_log.py` | Weekly auto-generated narrative summary |
| `leaderboard.py` | Re-render `experiments/leaderboard.md` from ledger |
| `promote.sh` | Promote a winning experiment's `train.py` to `checkpoints/best/` |
| `run_experiment.sh` | Canonical experiment runner (one experiment, full lifecycle) |
| `disaster_recover.sh` | Pull from GCS + GitHub after catastrophic loss |
| `daily_snapshot.sh` | Daily tar of `experiments/` + `checkpoints/best/` to GCS |
| `check_test_lock.py` | Verify test set hash matches recorded value |
| `check_paper_claims.py` | Find unsourced numerical claims in LaTeX |
| `check_significance.py` | Statistical significance tests for confirm-phase wins |
