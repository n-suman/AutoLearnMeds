---
brief: 001-pseudo-labeling-gpt4o
status: blocked
started: 2026-05-12 19:29 UTC
updated: 2026-05-12 19:29 UTC
---

## 2026-05-12 19:29 UTC — prerequisite check

I read the brief and confirmed the task is to build an idempotent GPT-4o pseudo-labeling pipeline, calibrate on the validation set first, then generate labels for the unlabeled raw images only if validation lenient F1 is acceptable.

Prerequisites checked before starting API work:

- Branch is `phase-0-plumbing`.
- Python is available: `Python 3.11.9`.
- `gsutil` is installed: `gsutil version: 5.33`.
- GCS read access works for `gs://auto_learn_meds/raw/raw_images/`.
- `gcloud auth list` works outside the sandbox; active account is `suman.nandamury@gmail.com`.
- Existing dirty worktree items are unrelated to this brief: modified `uv.lock` and an untracked Malepati/YOLOv12+SAHI PDF.
- `OPENAI_API_KEY` is missing from the environment, so paid OpenAI API calibration/full labeling cannot start.
- The exact brief paths `gs://auto_learn_meds/raw/train.jsonl`, `val.jsonl`, and `test.jsonl` do not exist. The repo's documented equivalent is available at `gs://auto_learn_meds/raw/golden_set/gold_standard.jsonl` plus `gs://auto_learn_meds/raw/golden_set/splits.json`, which is enough to reconstruct train/val/test manifests when API credentials are available.

Blocked until the user provides an OpenAI API key in the environment or explicitly points me to another approved credential source.
