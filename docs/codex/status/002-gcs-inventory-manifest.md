---
brief: 002-gcs-inventory-manifest
status: done
started: 2026-05-12 19:29 UTC
updated: 2026-05-12 19:48 UTC
---

## 2026-05-12 19:29 UTC — opening note

I read the brief and will build a read-only, reusable GCS inventory script plus machine/human manifests under the requested paths.

Prerequisites checked before starting:

- `gsutil` installed: `gsutil version: 5.33`.
- `gcloud auth list` works outside the sandbox; active account is `suman.nandamury@gmail.com`, and read-only `gsutil ls` against `gs://auto_learn_meds/raw/raw_images/` succeeds.
- Python available: `Python 3.11.9`.
- Repo branch is `phase-0-plumbing`.
- Worktree is dirty, but the existing changes are unrelated to this brief: modified `uv.lock` and an untracked Malepati/YOLOv12+SAHI PDF. I will not touch or stage them.
- No GCS write is needed; this brief will use listing/metadata commands only.

## 2026-05-12 19:48 UTC — final summary

Produced the reusable read-only inventory script and both manifest files:

- `scripts/codex/build_gcs_manifest.py`
- `experiments/gcs_manifest.json`
- `experiments/gcs_manifest.md`

Validation:

- `python3 -m json.tool experiments/gcs_manifest.json` succeeds.
- `python3 scripts/codex/build_gcs_manifest.py --help` succeeds.
- The script was run against GCS with read-only `gsutil ls` / `gsutil du` calls; no checkpoint files were downloaded and no GCS writes were performed.

Key numbers:

- Total inventoried objects: 160.
- Total inventoried size: 21.01 GiB.
- Ledger entries: 11, representing 10 unique run IDs (`track_d-seed44` appears twice in the ledger).
- GCS run IDs: 10.
- Local run IDs: 6.
- `ledger_missing_from_gcs`: 0. No alarm condition.
- `gcs_orphans`: 0.
- `ledger_missing_locally`: 4 (`baseline_mae_init-seed44`, `baseline_mae_tapt_init-seed44`, `baseline_mae_text_aware_init-seed44`, `track_d-seed44`), matching the expected missing local Track C/Track D run directories.
- Raw image prefix: 3,061 objects including `.DS_Store`, 3,060 image files, 7.60 GiB.
- Raw manifests present: `raw/golden_set/gold_standard.jsonl` and `raw/golden_set/splits.json`. The `raw/train.jsonl`, `raw/val.jsonl`, and `raw/test.jsonl` paths named in the brief are not present.

Checkpoint inventory:

- `checkpoints/` contains 78 objects totaling 21.01 GiB.
- Track A/Track C best checkpoints are present under `checkpoints/runs/*/best.pt`.
- Qwen LoRA adapter artifacts are present under `checkpoints/runs/qwen_baseline/best_lora/`.
- MAE checkpoints are present under `checkpoints/mae/{run-seed44,tapt-seed44,text-aware-seed44}/`.
