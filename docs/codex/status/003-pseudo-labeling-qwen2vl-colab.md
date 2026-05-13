---
brief: 003-pseudo-labeling-qwen2vl-colab
status: done
started: 2026-05-12 21:57 UTC
updated: 2026-05-12 22:00 UTC
---

## 2026-05-12 21:57 UTC — opening note

I read the brief and confirmed it supersedes brief 001. I will leave brief 001's blocked status alone.

Prerequisites checked before starting:

- Python is available: `Python 3.11.9`.
- Repo branch is `phase-0-plumbing`.
- Worktree has only the previously known unrelated dirty items: modified `uv.lock` and untracked Malepati/YOLOv12+SAHI PDF.
- `gsutil` is installed (`gsutil version: 5.33`; it prints a sandbox-local multiprocessing warning, but exits 0).
- GCS read access works for `gs://auto_learn_meds/raw/golden_set/`.
- `gcloud auth list` works outside the sandbox; active account is `suman.nandamury@gmail.com`.
- No GPU or paid API key is needed for Codex-side work.

Model choice: defaulting to `Qwen/Qwen2-VL-7B-Instruct` as requested. No fallback selected.

## 2026-05-12 22:00 UTC — final summary

Produced the assigned Codex-side handoff files:

- `scripts/codex/pseudo_label_qwen2vl.py`
- `scripts/codex/qwen2vl_colab_runbook.md`

Script behavior:

- CLI supports `--mode calibrate` and `--mode label`.
- Heavy ML imports are lazy; importing helper functions does not load Qwen.
- Default model is `Qwen/Qwen2-VL-7B-Instruct`.
- Uses full 12-tag canonical XML output with project field name `mfg_date`; parser also accepts `manufacturing_date` and normalizes it to `mfg_date`.
- Label mode is resume-aware via `data/pseudo_labels/round_001.progress.jsonl`.
- Calibration writes `data/pseudo_labels/round_001.calibration.json`, including macro metrics, per-field metrics, safety-four edit F1, and pack-type breakdown.
- If safety-four edit F1 is below 0.30, calibrate mode exits non-zero after writing the report.

Codex-side smoke tests passed:

- `python3 -m py_compile scripts/codex/pseudo_label_qwen2vl.py`
- `python3 scripts/codex/pseudo_label_qwen2vl.py --help`
- Import-only prompt/parser smoke test, including malformed response handling, without loading the model.

Runbook handoff:

- `scripts/codex/qwen2vl_colab_runbook.md` is ready for the user to run on Colab.
- It reconstructs `raw/{train,val,test}_reconstructed.jsonl` from `raw/golden_set/gold_standard.jsonl` and `splits.json`, because separate `raw/train.jsonl`, `raw/val.jsonl`, and `raw/test.jsonl` files are not present.
- Calibration should be run first. Full unlabeled labeling should proceed only if `safety4_macro_edit_f1 >= 0.30`.
