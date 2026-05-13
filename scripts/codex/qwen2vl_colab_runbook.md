# Qwen2-VL-7B Pseudo-Labeling Runbook

Run these commands on the Colab A100 runtime after the normal project bootstrap.
This path does not use paid APIs.

## 1. Pull Latest Code

```bash
cd /content/AutoLearnMeds
git pull origin phase-0-plumbing
uv sync --extra ml --extra colab
mkdir -p raw/golden_set data/pseudo_labels
```

## 2. Reconstruct Gold Split JSONLs

The bucket has `raw/golden_set/gold_standard.jsonl` and `splits.json`, not
separate `raw/train.jsonl`, `raw/val.jsonl`, and `raw/test.jsonl` files.

```bash
gsutil cp gs://auto_learn_meds/raw/golden_set/gold_standard.jsonl raw/golden_set/gold_standard.jsonl
gsutil cp gs://auto_learn_meds/raw/golden_set/splits.json raw/golden_set/splits.json

python - <<'PY'
import json
from pathlib import Path

gold_path = Path("raw/golden_set/gold_standard.jsonl")
splits_path = Path("raw/golden_set/splits.json")

records = {}
with gold_path.open() as f:
    for line in f:
        row = json.loads(line)
        image_file = row.get("image_file") or Path(row.get("image_path", "")).name
        row["image_path"] = f"gs://auto_learn_meds/raw/raw_images/{image_file}"
        records[image_file] = row

splits = json.loads(splits_path.read_text())
for split_name in ["train", "val", "test"]:
    out = Path(f"raw/{split_name}_reconstructed.jsonl")
    with out.open("w") as f:
        for image_file in splits[split_name]:
            row = records[image_file]
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(split_name, sum(1 for _ in out.open()), out)
PY
```

## 3. Calibrate On Validation Split First

```bash
python scripts/codex/pseudo_label_qwen2vl.py \
  --mode calibrate \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --gold-jsonl raw/val_reconstructed.jsonl \
  --out-jsonl data/pseudo_labels/round_001.jsonl \
  --batch-size 1 \
  --max-new-tokens 512
```

Inspect the calibration:

```bash
cat data/pseudo_labels/round_001.calibration.json
```

Proceed to full labeling only if `safety4_macro_edit_f1 >= 0.30`. The safety
four fields are `batch_number`, `expiry_date`, `mrp`, and `mfg_date`.

## 4. Build The Unlabeled List

```bash
python - <<'PY'
import json
import subprocess
from pathlib import Path

labeled = set()
with open("raw/golden_set/gold_standard.jsonl") as f:
    for line in f:
        row = json.loads(line)
        image_file = row.get("image_file") or Path(row.get("image_path", "")).name
        labeled.add(f"gs://auto_learn_meds/raw/raw_images/{image_file}")

all_paths = subprocess.check_output(
    ["gsutil", "ls", "gs://auto_learn_meds/raw/raw_images/"],
    text=True,
).splitlines()
unlabeled = sorted(
    p for p in all_paths
    if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"))
    and p not in labeled
)

out = Path("data/pseudo_labels/unlabeled_paths.txt")
out.write_text("\n".join(unlabeled) + "\n")
print(f"{len(unlabeled)} unlabeled images -> {out}")
PY
```

Expected count is about 2,200-2,300 images.

## 5. Run Full Pseudo-Labeling

```bash
python scripts/codex/pseudo_label_qwen2vl.py \
  --mode label \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --unlabeled-list data/pseudo_labels/unlabeled_paths.txt \
  --out-jsonl data/pseudo_labels/round_001.jsonl \
  --batch-size 1 \
  --max-new-tokens 512 \
  --resume
```

The script appends one completed label row at a time to:

```text
data/pseudo_labels/round_001.jsonl
```

It also appends progress rows to:

```text
data/pseudo_labels/round_001.progress.jsonl
```

If the Colab runtime disconnects, rerun the command with `--resume`; completed
image paths marked `done` in the progress file will be skipped.

## 6. Sync Results Back To GCS

```bash
gsutil cp data/pseudo_labels/round_001.calibration.json gs://auto_learn_meds/pseudo_labels/
gsutil cp data/pseudo_labels/round_001.jsonl gs://auto_learn_meds/pseudo_labels/
gsutil cp data/pseudo_labels/round_001.progress.jsonl gs://auto_learn_meds/pseudo_labels/
```

Report the calibration metrics back before launching any downstream training.
