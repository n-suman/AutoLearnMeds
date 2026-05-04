# scripts/

Operational scripts for the project.

| Script | Purpose |
|---|---|
| `colab_bootstrap.sh` | One-cell Colab session bootstrap (mounts, deps, SSH tunnel) |
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
