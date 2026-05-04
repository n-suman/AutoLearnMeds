# checkpoints/

Model checkpoints from training runs. On Colab, this is a symlink to a GDrive folder so checkpoints survive runtime restarts.

Layout:
- `best/` — current best-by-macro-F1 model
- `runs/<run_id>/` — per-experiment checkpoints

Mirrored to GCS every 5 min by `scripts/sync_to_gcs.sh`.
