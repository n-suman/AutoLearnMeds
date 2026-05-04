# data/

Dataset working directory.

- `raw/` — symlink (on Colab) to the GDrive mount of the original images + YOLO labels + transcriptions. Not tracked in git (see `.gitignore`).
- `processed/` — generated `train.jsonl`, `val.jsonl`, `test.jsonl` produced by `scripts/build_processed.py`. Not tracked in git.
- `data_card.md` — auto-generated dataset documentation (Gebru et al. 2018 datasheet style).

The canonical schema for `processed/*.jsonl` is locked in Phase 1 once the user shares the golden_set. See spec §3.2.
