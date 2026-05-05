# Phase 1 — Data Ingestion & Schema Lock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the user's golden_set + raw_images on GCS into hash-locked `data/processed/{train,val,test}.jsonl` files; implement the canonical, frozen `prepare.py` (BPE-8192 tokenizer + Donut-style XML output format + image preprocessing + DataLoader + `evaluate()` returning per-field and macro F1) so Phase 2 can train a model against it. Auto-generate `data/data_card.md`. Exit gate: `evaluate()` runs on a stub model on Colab and returns a sensible (low) macro-F1.

**Architecture:** Single `prepare.py` (autoresearch convention — agent never edits this) provides `get_dataloader`, `evaluate`, `FIELD_ORDER`, `format_output`, `parse_output`, `train_tokenizer`. Heavy deps (`torch`, `transformers`, `tokenizers`) are imported lazily inside functions so the module imports cleanly in a torch-less local Mac venv (lightweight tests can run there). `scripts/build_processed.py` is run once on Colab against `/mnt/gcs/raw/` to produce the JSONL splits.

**Tech Stack:** Python 3.11, `tokenizers` (HuggingFace BPE), `torch`, `transformers` (SigLIP `AutoProcessor`), `Pillow`, `pyyaml`, `pytest`. Drive is not in the loop — all data lives in `gs://auto_learn_meds/`.

**Spec:** [`docs/superpowers/specs/2026-05-05-pharma-vlm-autoresearch-design.md`](../specs/2026-05-05-pharma-vlm-autoresearch-design.md), §3 (Data Pipeline), §1.3 (metric), §3.4 (output format).

**Schema locked from real data on 2026-05-05:**
- 837 labeled records, 564/111/162 train/val/test, stratified by `medicine_name` (47 medicines, all 47 in train, 45 in val, 44 in test).
- 12 OCR fields total. **Primary macro-F1 averages 9 high-frequency fields** (≥5% presence): `drug_name, generic_name, brand_name, batch_number, mrp, expiry_date, mfg_date, company, warnings, strength` — wait, that's 10. Re-counting from the data: the 9 fields ≥5% are `drug_name (56%), generic_name (54%), brand_name (49%), batch_number (31%), mrp (31%), expiry_date (29%), mfg_date (28%), company (21%), warnings (8%), strength (6%)` — that's 10 fields ≥5%. The 2 ultra-rare excluded fields are `quantity (0.7%)` and `manufacturer (0.6%)`. So `HIGH_FREQUENCY_FIELDS` has 10 entries, `ALL_FIELDS` has 12.

**Data sources** (Colab-only paths):
- `/mnt/gcs/raw/golden_set/gold_standard.jsonl` — 837 records
- `/mnt/gcs/raw/golden_set/splits.json` — pre-built splits
- `/mnt/gcs/raw/raw_images/IMG_*.jpeg` — 3060 images (837 referenced)

**Phase 0 prerequisites met:** repo cloned to `/content/AutoLearnMeds`, GCS mounted at `/mnt/gcs`, venv synced with `--extra ml`, VSCode SSH connected, `make verify` GREEN.

---

## File Structure

| Path | Created/Modified | Responsibility |
|---|---|---|
| `prepare.py` | **REWRITTEN from stub** | Canonical FROZEN module: tokenizer, dataloader, `evaluate()`. Lazy torch imports. |
| `scripts/build_processed.py` | Create | Reads `/mnt/gcs/raw/golden_set/*` + raw_images → writes `data/processed/{train,val,test}.jsonl` with `image_hash`. |
| `scripts/build_data_card.py` | Create | Reads `data/processed/*.jsonl` → writes `data/data_card.md` (per-field stats, splits, distributions). |
| `tests/conftest.py` | Modify | Add `synthetic_records` fixture and `tiny_processed_dir` fixture. |
| `tests/fixtures/synthetic_jsonl/gold_standard.jsonl` | Create | 5 synthetic records mimicking real schema for tests. |
| `tests/fixtures/synthetic_jsonl/splits.json` | Create | Synthetic splits (3 train, 1 val, 1 test). |
| `tests/fixtures/synthetic_images/IMG_synth_*.jpeg` | Create | 5 tiny solid-color JPEGs (under 1KB each) for image-pipeline tests. |
| `tests/test_build_processed.py` | Create | Tests `scripts/build_processed.py` against synthetic fixtures. |
| `tests/test_prepare_constants.py` | Create | Tests `FIELD_ORDER`, `HIGH_FREQUENCY_FIELDS`, special tokens. |
| `tests/test_xml_format.py` | Create | Tests `format_output` / `parse_output` round-trip. |
| `tests/test_tokenizer.py` | Create | Tests BPE tokenizer training + load. Lightweight (no torch). |
| `tests/test_metric.py` | Create | Tests `normalize_field_value`, `compute_field_f1`, `compute_metrics` on hand-crafted examples. |
| `tests/test_build_data_card.py` | Create | Tests data-card generation. |
| `tests/test_torch_pipeline.py` | Create | Marked `@pytest.mark.colab` — image preprocessing + DataLoader. Skipped locally; run on Colab. |
| `data/processed/.gitignore` | Create | Ignore generated JSONL + tokenizer.json (generated artifacts, not source code). |

**`prepare.py` internal structure** (single file, sections delimited by `# === ... ===` banners):

```
# === Constants ===          FIELD_ORDER, ALL_FIELDS, HIGH_FREQUENCY_FIELDS, SPECIAL_TOKENS, IMAGE_SIZE
# === Output format ===      format_output(record), parse_output(text)
# === Tokenizer ===          train_tokenizer(...), get_tokenizer(...)
# === Normalization+F1 ===   normalize_field_value, compute_field_f1, compute_metrics
# === Image preprocessing ===preprocess_image (lazy torch import)
# === Dataset+DataLoader === PharmaLabelDataset, get_dataloader (lazy torch import)
# === Public eval API ===    evaluate(model, split), evaluate_test(...) [gated]
# === Module hash check ===  __module_hash__, _verify_module_integrity()
```

---

## Task 1: Test fixtures and synthetic data

**Files:**
- Create: `tests/fixtures/__init__.py` (empty)
- Create: `tests/fixtures/synthetic_jsonl/gold_standard.jsonl`
- Create: `tests/fixtures/synthetic_jsonl/splits.json`
- Create: `tests/fixtures/synthetic_images/IMG_synth_001.jpeg` through `IMG_synth_005.jpeg`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Create the empty fixtures package marker**

```bash
mkdir -p /Users/apple/AutoLearnMeds/tests/fixtures/synthetic_jsonl
mkdir -p /Users/apple/AutoLearnMeds/tests/fixtures/synthetic_images
touch /Users/apple/AutoLearnMeds/tests/fixtures/__init__.py
```

- [ ] **Step 2: Create the synthetic gold_standard.jsonl**

Create `/Users/apple/AutoLearnMeds/tests/fixtures/synthetic_jsonl/gold_standard.jsonl` with exactly these 5 lines (each is one JSON object, no trailing newline on the last):

```json
{"image_file": "IMG_synth_001.jpeg", "medicine_name": "TAB SYNTHA", "view_type": "front", "pack_type": "strip_foil", "fields": {"brand_name": {"text": "Synthacid", "polygon": [[10.0,10.0],[90.0,10.0],[90.0,30.0],[10.0,30.0]]}, "generic_name": {"text": "Paracetamol", "polygon": [[10.0,40.0],[90.0,40.0],[90.0,55.0],[10.0,55.0]]}, "batch_number": {"text": "B.No.:S001", "polygon": [[10.0,60.0],[50.0,60.0],[50.0,70.0],[10.0,70.0]]}}, "product_polygon": [[5.0,5.0],[95.0,5.0],[95.0,95.0],[5.0,95.0]], "product_bbox": [5.0,5.0,90.0,90.0]}
{"image_file": "IMG_synth_002.jpeg", "medicine_name": "INJ SYNTHB", "view_type": "angled", "pack_type": "ampoule", "fields": {"drug_name": {"text": "Synthbol", "polygon": [[10.0,10.0],[90.0,10.0],[90.0,30.0],[10.0,30.0]]}, "mrp": {"text": "M.R.P.Rs.: 99.50", "polygon": [[10.0,40.0],[90.0,40.0],[90.0,55.0],[10.0,55.0]]}}, "product_polygon": [[5.0,5.0],[95.0,5.0],[95.0,95.0],[5.0,95.0]], "product_bbox": [5.0,5.0,90.0,90.0]}
{"image_file": "IMG_synth_003.jpeg", "medicine_name": "TAB SYNTHA", "view_type": "back", "pack_type": "strip_foil", "fields": {"brand_name": {"text": "Synthacid", "polygon": [[10.0,10.0],[90.0,10.0],[90.0,30.0],[10.0,30.0]]}, "expiry_date": {"text": "Exp. Dt.: 12/2027", "polygon": [[10.0,40.0],[90.0,40.0],[90.0,55.0],[10.0,55.0]]}, "mfg_date": {"text": "Mfg. Dt.: 01/2025", "polygon": [[10.0,60.0],[50.0,60.0],[50.0,70.0],[10.0,70.0]]}}, "product_polygon": [[5.0,5.0],[95.0,5.0],[95.0,95.0],[5.0,95.0]], "product_bbox": [5.0,5.0,90.0,90.0]}
{"image_file": "IMG_synth_004.jpeg", "medicine_name": "SYRUP SYNTHC", "view_type": "side", "pack_type": "syrup_bottle", "fields": {"drug_name": {"text": "Synthcol", "polygon": [[10.0,10.0],[90.0,10.0],[90.0,30.0],[10.0,30.0]]}, "company": {"text": "ABC LABS PVT LTD", "polygon": [[10.0,40.0],[90.0,40.0],[90.0,55.0],[10.0,55.0]]}, "warnings": {"text": "FOR EXTERNAL USE ONLY", "polygon": [[10.0,60.0],[90.0,60.0],[90.0,75.0],[10.0,75.0]]}}, "product_polygon": [[5.0,5.0],[95.0,5.0],[95.0,95.0],[5.0,95.0]], "product_bbox": [5.0,5.0,90.0,90.0]}
{"image_file": "IMG_synth_005.jpeg", "medicine_name": "INJ SYNTHB", "view_type": "front", "pack_type": "vial", "fields": {"strength": {"text": "10MG / 5 ML", "polygon": [[10.0,10.0],[90.0,10.0],[90.0,30.0],[10.0,30.0]]}, "generic_name": {"text": "Synthboline", "polygon": [[10.0,40.0],[90.0,40.0],[90.0,55.0],[10.0,55.0]]}}, "product_polygon": [[5.0,5.0],[95.0,5.0],[95.0,95.0],[5.0,95.0]], "product_bbox": [5.0,5.0,90.0,90.0]}
```

- [ ] **Step 3: Create the synthetic splits.json**

Create `/Users/apple/AutoLearnMeds/tests/fixtures/synthetic_jsonl/splits.json`:

```json
{
  "train": ["IMG_synth_001.jpeg", "IMG_synth_002.jpeg", "IMG_synth_003.jpeg"],
  "val": ["IMG_synth_004.jpeg"],
  "test": ["IMG_synth_005.jpeg"],
  "seed": 42,
  "split_ratio": "60/20/20",
  "stratified_by": "none (synthetic)"
}
```

- [ ] **Step 4: Generate the 5 tiny synthetic JPEGs via Python**

Run from `/Users/apple/AutoLearnMeds`:

```bash
python3 -c "
from pathlib import Path
import struct, zlib
# Minimal 1-pixel JPEG; we'll create 5 distinct ones via different solid-color RGB.
# Easier: use PIL if available; if not, write a tiny valid JPEG bytes literal.
try:
    from PIL import Image
    out = Path('tests/fixtures/synthetic_images')
    out.mkdir(parents=True, exist_ok=True)
    for i, color in enumerate([(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255)], start=1):
        Image.new('RGB', (32,32), color=color).save(out / f'IMG_synth_{i:03d}.jpeg', 'JPEG')
    print('wrote 5 JPEGs')
except ImportError:
    print('PIL not available locally; skipping. Will need ml extras to populate.')
"
```

If PIL isn't installed in your local venv, run instead:

```bash
uv run --with pillow python3 -c "
from PIL import Image
from pathlib import Path
out = Path('tests/fixtures/synthetic_images')
out.mkdir(parents=True, exist_ok=True)
for i, color in enumerate([(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255)], start=1):
    Image.new('RGB', (32,32), color=color).save(out / f'IMG_synth_{i:03d}.jpeg', 'JPEG')
print('wrote 5 JPEGs')
"
```

Verify: `ls -la tests/fixtures/synthetic_images/` should show 5 files of ~600-900 bytes each.

- [ ] **Step 5: Add fixtures to conftest.py**

Append to `/Users/apple/AutoLearnMeds/tests/conftest.py`:

```python


@pytest.fixture(scope="session")
def synthetic_jsonl_dir(project_root: Path) -> Path:
    """Path to tests/fixtures/synthetic_jsonl/ (gold_standard.jsonl + splits.json)."""
    p = project_root / "tests" / "fixtures" / "synthetic_jsonl"
    assert (p / "gold_standard.jsonl").is_file()
    assert (p / "splits.json").is_file()
    return p


@pytest.fixture(scope="session")
def synthetic_images_dir(project_root: Path) -> Path:
    """Path to tests/fixtures/synthetic_images/ (5 small JPEGs)."""
    p = project_root / "tests" / "fixtures" / "synthetic_images"
    assert p.is_dir()
    return p


@pytest.fixture
def tiny_processed_dir(tmp_path: Path) -> Path:
    """Empty dir for write-test JSONL outputs."""
    d = tmp_path / "processed"
    d.mkdir()
    return d
```

- [ ] **Step 6: Verify fixtures load**

Run:

```bash
uv run pytest tests/test_smoke.py -v 2>&1 | tail -3
ls tests/fixtures/synthetic_jsonl/
ls tests/fixtures/synthetic_images/
wc -l tests/fixtures/synthetic_jsonl/gold_standard.jsonl
```

Expected:
- All prior smoke tests still PASS (20 passed).
- `gold_standard.jsonl` has 5 lines.
- `synthetic_images/` has 5 .jpeg files.

- [ ] **Step 7: Commit**

```bash
git add tests/fixtures/ tests/conftest.py
git commit -m "test: synthetic fixtures for Phase 1 (5 records, 5 tiny JPEGs)

5 hand-crafted records covering all view_types and several pack_types,
~10 fields total spread across the 5 records. 5 tiny solid-color JPEGs
(<1KB each) for image-pipeline tests. Two new conftest fixtures expose
the synthetic JSONL dir and synthetic images dir.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `scripts/build_processed.py` — convert raw → processed JSONL

**Files:**
- Create: `scripts/build_processed.py`
- Create: `tests/test_build_processed.py`
- Create: `data/processed/.gitignore`

- [ ] **Step 1: Add `data/processed/.gitignore` to keep generated outputs out of git**

Create `/Users/apple/AutoLearnMeds/data/processed/.gitignore`:

```
# Generated by scripts/build_processed.py — not source.
*.jsonl
tokenizer.json

# But keep the .gitignore itself.
!.gitignore
```

- [ ] **Step 2: Write the failing test**

Create `/Users/apple/AutoLearnMeds/tests/test_build_processed.py`:

```python
"""Tests for scripts/build_processed.py — raw -> processed JSONL conversion."""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


def _load_module(project_root: Path):
    spec = importlib.util.spec_from_file_location(
        "build_processed", project_root / "scripts" / "build_processed.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_module_loads(project_root: Path) -> None:
    mod = _load_module(project_root)
    assert hasattr(mod, "build_processed")
    assert hasattr(mod, "compute_image_hash")


def test_compute_image_hash_is_deterministic(
    project_root: Path, synthetic_images_dir: Path
) -> None:
    mod = _load_module(project_root)
    img = synthetic_images_dir / "IMG_synth_001.jpeg"
    h1 = mod.compute_image_hash(img)
    h2 = mod.compute_image_hash(img)
    assert h1 == h2
    assert h1.startswith("sha256:")
    assert len(h1) == len("sha256:") + 64


def test_compute_image_hash_distinct_for_distinct_images(
    project_root: Path, synthetic_images_dir: Path
) -> None:
    mod = _load_module(project_root)
    h1 = mod.compute_image_hash(synthetic_images_dir / "IMG_synth_001.jpeg")
    h2 = mod.compute_image_hash(synthetic_images_dir / "IMG_synth_002.jpeg")
    assert h1 != h2


def test_build_processed_creates_three_jsonl_files(
    project_root: Path,
    synthetic_jsonl_dir: Path,
    synthetic_images_dir: Path,
    tiny_processed_dir: Path,
) -> None:
    mod = _load_module(project_root)
    mod.build_processed(
        gold_standard_path=synthetic_jsonl_dir / "gold_standard.jsonl",
        splits_path=synthetic_jsonl_dir / "splits.json",
        images_root=synthetic_images_dir,
        out_dir=tiny_processed_dir,
        image_path_prefix="raw/raw_images",
    )
    assert (tiny_processed_dir / "train.jsonl").is_file()
    assert (tiny_processed_dir / "val.jsonl").is_file()
    assert (tiny_processed_dir / "test.jsonl").is_file()


def test_build_processed_split_sizes(
    project_root: Path,
    synthetic_jsonl_dir: Path,
    synthetic_images_dir: Path,
    tiny_processed_dir: Path,
) -> None:
    mod = _load_module(project_root)
    mod.build_processed(
        gold_standard_path=synthetic_jsonl_dir / "gold_standard.jsonl",
        splits_path=synthetic_jsonl_dir / "splits.json",
        images_root=synthetic_images_dir,
        out_dir=tiny_processed_dir,
        image_path_prefix="raw/raw_images",
    )
    train = (tiny_processed_dir / "train.jsonl").read_text().splitlines()
    val = (tiny_processed_dir / "val.jsonl").read_text().splitlines()
    test = (tiny_processed_dir / "test.jsonl").read_text().splitlines()
    assert len(train) == 3
    assert len(val) == 1
    assert len(test) == 1


def test_build_processed_record_schema(
    project_root: Path,
    synthetic_jsonl_dir: Path,
    synthetic_images_dir: Path,
    tiny_processed_dir: Path,
) -> None:
    mod = _load_module(project_root)
    mod.build_processed(
        gold_standard_path=synthetic_jsonl_dir / "gold_standard.jsonl",
        splits_path=synthetic_jsonl_dir / "splits.json",
        images_root=synthetic_images_dir,
        out_dir=tiny_processed_dir,
        image_path_prefix="raw/raw_images",
    )
    line0 = (tiny_processed_dir / "train.jsonl").read_text().splitlines()[0]
    rec = json.loads(line0)
    # Required added fields
    assert "image_id" in rec
    assert "image_file" in rec
    assert "image_path" in rec
    assert "image_hash" in rec
    assert "split" in rec
    assert rec["image_hash"].startswith("sha256:")
    assert rec["image_path"].startswith("raw/raw_images/")
    assert rec["image_id"].startswith("IMG_synth_")
    # Original fields preserved
    assert "fields" in rec
    assert "medicine_name" in rec
    assert rec["split"] == "train"


def test_build_processed_idempotent(
    project_root: Path,
    synthetic_jsonl_dir: Path,
    synthetic_images_dir: Path,
    tiny_processed_dir: Path,
) -> None:
    mod = _load_module(project_root)
    args = dict(
        gold_standard_path=synthetic_jsonl_dir / "gold_standard.jsonl",
        splits_path=synthetic_jsonl_dir / "splits.json",
        images_root=synthetic_images_dir,
        out_dir=tiny_processed_dir,
        image_path_prefix="raw/raw_images",
    )
    mod.build_processed(**args)
    train_v1 = (tiny_processed_dir / "train.jsonl").read_text()
    mod.build_processed(**args)
    train_v2 = (tiny_processed_dir / "train.jsonl").read_text()
    assert train_v1 == train_v2


def test_build_processed_cli_invocation(
    project_root: Path,
    synthetic_jsonl_dir: Path,
    synthetic_images_dir: Path,
    tiny_processed_dir: Path,
) -> None:
    """The script must be runnable as a CLI."""
    result = subprocess.run(
        [
            "python",
            str(project_root / "scripts" / "build_processed.py"),
            "--gold-standard", str(synthetic_jsonl_dir / "gold_standard.jsonl"),
            "--splits", str(synthetic_jsonl_dir / "splits.json"),
            "--images-root", str(synthetic_images_dir),
            "--out-dir", str(tiny_processed_dir),
            "--image-path-prefix", "raw/raw_images",
        ],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert (tiny_processed_dir / "train.jsonl").is_file()
```

- [ ] **Step 3: Run tests, verify they fail (no script yet)**

```bash
uv run pytest tests/test_build_processed.py -v
```

Expected: all 7 tests FAIL with `FileNotFoundError` or `ModuleNotFoundError`.

- [ ] **Step 4: Implement `scripts/build_processed.py`**

Create `/Users/apple/AutoLearnMeds/scripts/build_processed.py`:

```python
#!/usr/bin/env python3
"""Convert golden_set + splits.json + raw images -> data/processed/{train,val,test}.jsonl.

Adds per-record:
  image_id    — IMG_5246 (filename without extension)
  image_path  — relative to bucket root, e.g., raw/raw_images/IMG_5246.jpeg
  image_hash  — sha256:... of the image bytes (for test-set firewall)
  split       — train | val | test

Idempotent: re-running with the same inputs writes byte-identical outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def compute_image_hash(image_path: Path, chunk_size: int = 1 << 16) -> str:
    """Return sha256:<hex64> of the image file bytes."""
    h = hashlib.sha256()
    with image_path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _strip_extension(filename: str) -> str:
    return Path(filename).stem


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_processed(
    gold_standard_path: Path,
    splits_path: Path,
    images_root: Path,
    out_dir: Path,
    image_path_prefix: str = "raw/raw_images",
) -> dict[str, int]:
    """Build train/val/test JSONL files. Returns counts per split."""
    gold_standard_path = Path(gold_standard_path)
    splits_path = Path(splits_path)
    images_root = Path(images_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl(gold_standard_path)
    by_image = {r["image_file"]: r for r in records}
    splits = json.loads(splits_path.read_text())

    counts = {}
    for split_name in ("train", "val", "test"):
        out_lines: list[str] = []
        for image_file in splits.get(split_name, []):
            if image_file not in by_image:
                print(
                    f"[build_processed] WARN: {image_file} in splits but not in gold_standard; skipping",
                    file=sys.stderr,
                )
                continue
            img_path = images_root / image_file
            if not img_path.is_file():
                print(
                    f"[build_processed] WARN: {image_file} on disk missing at {img_path}; skipping",
                    file=sys.stderr,
                )
                continue
            base = by_image[image_file]
            rec = {
                "image_id": _strip_extension(image_file),
                "image_file": image_file,
                "image_path": f"{image_path_prefix.rstrip('/')}/{image_file}",
                "image_hash": compute_image_hash(img_path),
                "split": split_name,
                **base,
            }
            # Sort-keyed JSON dump => deterministic (idempotent).
            out_lines.append(json.dumps(rec, sort_keys=True, ensure_ascii=False))
        out_path = out_dir / f"{split_name}.jsonl"
        out_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
        counts[split_name] = len(out_lines)
        print(f"[build_processed] wrote {out_path} ({counts[split_name]} records)")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gold-standard", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--image-path-prefix",
        default="raw/raw_images",
        help="Prefix written into each record's image_path field",
    )
    args = parser.parse_args()
    build_processed(
        gold_standard_path=args.gold_standard,
        splits_path=args.splits,
        images_root=args.images_root,
        out_dir=args.out_dir,
        image_path_prefix=args.image_path_prefix,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Make the script executable**

```bash
chmod +x /Users/apple/AutoLearnMeds/scripts/build_processed.py
```

- [ ] **Step 6: Run tests, verify they pass**

```bash
uv run pytest tests/test_build_processed.py -v
```

Expected: all 7 tests PASSED.

- [ ] **Step 7: Run the broader suite to confirm nothing regressed**

```bash
uv run pytest -v 2>&1 | tail -3
```

Expected: 27 passed (20 prior + 7 new).

- [ ] **Step 8: Commit**

```bash
git add scripts/build_processed.py tests/test_build_processed.py data/processed/.gitignore
git commit -m "feat(scripts): build_processed.py — raw → processed JSONL with hashing

Reads gold_standard.jsonl + splits.json + raw_images/, writes
data/processed/{train,val,test}.jsonl with image_id, image_path,
image_hash (sha256), split fields added. Sort-keyed JSON dump for
idempotent re-runs. 7 tests cover module load, hash determinism,
hash distinction, file creation, split sizes, schema, idempotency,
and CLI invocation.

data/processed/.gitignore excludes generated *.jsonl and
tokenizer.json from git (they're build artifacts, not source).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `prepare.py` — constants + XML output format/parse

**Files:**
- Modify: `prepare.py` (rewrite from stub to v1)
- Create: `tests/test_prepare_constants.py`
- Create: `tests/test_xml_format.py`

- [ ] **Step 1: Write the failing test for constants**

Create `/Users/apple/AutoLearnMeds/tests/test_prepare_constants.py`:

```python
"""Tests for prepare.py constants — field universe + special tokens."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def prepare_mod():
    """Import prepare from the project root, fresh per test."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if "prepare" in sys.modules:
        importlib.reload(sys.modules["prepare"])
    import prepare
    return prepare


def test_field_order_is_canonical_12(prepare_mod) -> None:
    """FIELD_ORDER lists exactly 12 OCR fields in canonical decoder order."""
    expected = [
        "brand_name", "drug_name", "generic_name",
        "strength", "quantity",
        "company", "manufacturer",
        "batch_number", "mfg_date", "expiry_date", "mrp",
        "warnings",
    ]
    assert prepare_mod.FIELD_ORDER == expected
    assert len(prepare_mod.FIELD_ORDER) == 12


def test_high_frequency_fields_is_10(prepare_mod) -> None:
    """HIGH_FREQUENCY_FIELDS includes only fields with ≥5% presence (everything except quantity, manufacturer)."""
    high = prepare_mod.HIGH_FREQUENCY_FIELDS
    assert len(high) == 10
    assert "quantity" not in high
    assert "manufacturer" not in high
    # The 10 retained fields:
    for f in [
        "brand_name", "drug_name", "generic_name", "strength",
        "company", "batch_number", "mfg_date", "expiry_date", "mrp", "warnings",
    ]:
        assert f in high


def test_special_tokens_cover_all_fields(prepare_mod) -> None:
    """For every field there's an open and close special token."""
    tokens = prepare_mod.SPECIAL_TOKENS
    assert prepare_mod.BOS_TOKEN in tokens
    assert prepare_mod.EOS_TOKEN in tokens
    for field in prepare_mod.FIELD_ORDER:
        assert f"<{field}>" in tokens
        assert f"</{field}>" in tokens


def test_image_size(prepare_mod) -> None:
    """SigLIP-base native resolution."""
    assert prepare_mod.IMAGE_SIZE == 224


def test_module_imports_without_torch(prepare_mod) -> None:
    """prepare.py must be importable in a torch-less venv (lazy imports)."""
    # If we got here, the import worked. Belt-and-suspenders: confirm torch
    # isn't in the module's top-level namespace.
    assert not hasattr(prepare_mod, "torch")
```

- [ ] **Step 2: Run, verify failure**

```bash
uv run pytest tests/test_prepare_constants.py -v
```

Expected: tests FAIL because `prepare.py` is the stub raising `NotImplementedError`.

- [ ] **Step 3: Write the failing test for XML output format**

Create `/Users/apple/AutoLearnMeds/tests/test_xml_format.py`:

```python
"""Tests for prepare.py XML format — format_output / parse_output round-trip."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def prepare_mod():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if "prepare" in sys.modules:
        importlib.reload(sys.modules["prepare"])
    import prepare
    return prepare


def test_format_output_emits_only_present_fields(prepare_mod) -> None:
    record = {
        "fields": {
            "brand_name": {"text": "Pantocid DSR"},
            "batch_number": {"text": "B.No.:GTF3406A"},
        }
    }
    out = prepare_mod.format_output(record)
    assert "<brand_name>Pantocid DSR</brand_name>" in out
    assert "<batch_number>B.No.:GTF3406A</batch_number>" in out
    # Absent fields not emitted
    assert "<generic_name>" not in out
    assert "<mrp>" not in out
    # Wrapped in BOS/EOS
    assert out.startswith(prepare_mod.BOS_TOKEN)
    assert out.endswith(prepare_mod.EOS_TOKEN)


def test_format_output_canonical_field_order(prepare_mod) -> None:
    """Fields must appear in FIELD_ORDER, regardless of dict insertion order."""
    record = {
        "fields": {
            "warnings": {"text": "X"},
            "brand_name": {"text": "Y"},
            "mrp": {"text": "Z"},
        }
    }
    out = prepare_mod.format_output(record)
    # brand_name precedes mrp precedes warnings in FIELD_ORDER
    pos_brand = out.index("<brand_name>")
    pos_mrp = out.index("<mrp>")
    pos_warn = out.index("<warnings>")
    assert pos_brand < pos_mrp < pos_warn


def test_parse_output_extracts_fields(prepare_mod) -> None:
    text = (
        prepare_mod.BOS_TOKEN
        + "<brand_name>Pantocid DSR</brand_name>"
        + "<mrp>M.R.P.Rs.: 245.00</mrp>"
        + prepare_mod.EOS_TOKEN
    )
    parsed = prepare_mod.parse_output(text)
    assert parsed == {"brand_name": "Pantocid DSR", "mrp": "M.R.P.Rs.: 245.00"}


def test_round_trip(prepare_mod) -> None:
    """format_output -> parse_output recovers the original {field: text}."""
    record = {
        "fields": {
            "brand_name": {"text": "Pantocid DSR"},
            "generic_name": {"text": "Pantoprazole Gastro-Resistant and Domperidone"},
            "batch_number": {"text": "B.No.:GTF3406A"},
        }
    }
    out = prepare_mod.format_output(record)
    parsed = prepare_mod.parse_output(out)
    assert parsed == {
        "brand_name": "Pantocid DSR",
        "generic_name": "Pantoprazole Gastro-Resistant and Domperidone",
        "batch_number": "B.No.:GTF3406A",
    }


def test_parse_output_handles_missing_eos(prepare_mod) -> None:
    """A truncated decode (no </s>) should still extract whatever fields are complete."""
    text = (
        prepare_mod.BOS_TOKEN
        + "<brand_name>Pantocid DSR</brand_name>"
        + "<mrp>M.R.P.Rs.: 245.00</"  # truncated mid-tag
    )
    parsed = prepare_mod.parse_output(text)
    assert parsed.get("brand_name") == "Pantocid DSR"
    # mrp is incomplete — should NOT be present (or present with empty/partial)
    # The exact behavior is up to the implementation, but it shouldn't crash.


def test_parse_output_ignores_unknown_tags(prepare_mod) -> None:
    """Tags not in FIELD_ORDER are ignored."""
    text = (
        prepare_mod.BOS_TOKEN
        + "<brand_name>X</brand_name>"
        + "<not_a_field>Y</not_a_field>"
        + prepare_mod.EOS_TOKEN
    )
    parsed = prepare_mod.parse_output(text)
    assert "brand_name" in parsed
    assert "not_a_field" not in parsed
```

- [ ] **Step 4: Run, verify failures**

```bash
uv run pytest tests/test_xml_format.py tests/test_prepare_constants.py -v 2>&1 | tail -10
```

Expected: all FAIL.

- [ ] **Step 5: Rewrite `prepare.py` with constants + XML format/parse**

Replace the stub at `/Users/apple/AutoLearnMeds/prepare.py` with:

```python
"""Phase-1 module: data prep, tokenizer, dataloader, evaluate().

This file is FIXED in the autoresearch loop — the agent must not modify it.
Defines: FIELD_ORDER, HIGH_FREQUENCY_FIELDS, SPECIAL_TOKENS, format_output,
parse_output, train_tokenizer, get_tokenizer, normalize_field_value,
compute_field_f1, compute_metrics, preprocess_image, PharmaLabelDataset,
get_dataloader, evaluate, evaluate_test (gated).

torch / transformers / Pillow / tokenizers are imported lazily inside the
functions that need them, so this module imports cleanly in a torch-less
venv (used for lightweight tests).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# === Constants ===

# Canonical decoder output order — all 12 OCR fields. Order chosen to keep
# semantically related fields adjacent (drug names, then dosing, then origin,
# then dates+price, then warnings). Locked: changing this invalidates trained
# tokenizer + checkpoints.
FIELD_ORDER: list[str] = [
    "brand_name", "drug_name", "generic_name",
    "strength", "quantity",
    "company", "manufacturer",
    "batch_number", "mfg_date", "expiry_date", "mrp",
    "warnings",
]

ALL_FIELDS: frozenset[str] = frozenset(FIELD_ORDER)

# High-frequency fields (≥5% presence in the 837-record golden_set).
# Excludes 'quantity' (0.7%) and 'manufacturer' (0.6%) — too few samples to
# give a reliable per-field F1. The primary macro-F1 averages over these 10.
HIGH_FREQUENCY_FIELDS: frozenset[str] = frozenset([
    "brand_name", "drug_name", "generic_name", "strength",
    "company", "batch_number", "mfg_date", "expiry_date", "mrp", "warnings",
])

BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

# Open and close tags for every field, plus the BOS/EOS sentinels.
SPECIAL_TOKENS: list[str] = (
    [BOS_TOKEN, EOS_TOKEN]
    + [f"<{f}>" for f in FIELD_ORDER]
    + [f"</{f}>" for f in FIELD_ORDER]
)

# SigLIP-base-patch16-224 native resolution. Locked.
IMAGE_SIZE = 224

# Tokenizer vocabulary size (BPE). Locked.
TOKENIZER_VOCAB_SIZE = 8192


# === Output format ===

def format_output(record: dict) -> str:
    """Render a record's `fields` dict as the canonical Donut-style XML output.

    Only present fields are emitted, in FIELD_ORDER. The text inside each tag
    is the raw OCR text (no normalization).
    """
    fields = record.get("fields") or {}
    parts: list[str] = [BOS_TOKEN]
    for fname in FIELD_ORDER:
        if fname not in fields:
            continue
        text = (fields[fname] or {}).get("text")
        if text is None:
            continue
        parts.append(f"<{fname}>{text}</{fname}>")
    parts.append(EOS_TOKEN)
    return "".join(parts)


# Compiled once at import time: matches a single complete <fname>value</fname>
# for any fname in FIELD_ORDER. Capture group 1 = field name, group 2 = value.
_FIELD_TAG_RE = re.compile(
    r"<(" + "|".join(re.escape(f) for f in FIELD_ORDER) + r")>(.*?)</\1>",
    re.DOTALL,
)


def parse_output(text: str) -> dict[str, str]:
    """Inverse of format_output: extract {field_name: text} from XML output.

    Tolerates missing BOS/EOS, ignores unknown tags, and skips truncated tags
    (where the closing </name> is missing).
    """
    return {m.group(1): m.group(2) for m in _FIELD_TAG_RE.finditer(text)}
```

- [ ] **Step 6: Run, verify both test files pass**

```bash
uv run pytest tests/test_prepare_constants.py tests/test_xml_format.py -v
```

Expected: all PASSED.

- [ ] **Step 7: Run the smoke tests too — `prepare.py` is now valid Python that imports cleanly, but is it still considered a "stub"?**

The smoke test `test_three_file_discipline_stubs` only checks that `prepare.py` exists. It doesn't check for `NotImplementedError`. Verify:

```bash
uv run pytest -v 2>&1 | tail -3
```

Expected: 39 PASSED (27 prior + 12 new from constants + xml).

- [ ] **Step 8: Commit**

```bash
git add prepare.py tests/test_prepare_constants.py tests/test_xml_format.py
git commit -m "feat(prepare): constants + Donut-style XML format/parse

Locks the 12-field universe (FIELD_ORDER), the 10 high-frequency fields
(HIGH_FREQUENCY_FIELDS — primary macro-F1 averages over these), special
tokens (BOS/EOS + open/close per field), and image/vocab sizes
(IMAGE_SIZE=224, TOKENIZER_VOCAB_SIZE=8192).

format_output(record) emits canonical-ordered XML of only present fields.
parse_output(text) extracts {field_name: text} via a single compiled regex
that ignores unknown tags and tolerates truncated decodes.

12 tests cover constant values, special-token coverage, ordering, round-trip,
truncation tolerance, and unknown-tag rejection. Module imports without torch.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `prepare.py` — BPE tokenizer

**Files:**
- Modify: `prepare.py` (append tokenizer section)
- Create: `tests/test_tokenizer.py`

- [ ] **Step 1: Write the failing test**

Create `/Users/apple/AutoLearnMeds/tests/test_tokenizer.py`:

```python
"""Tests for prepare.py tokenizer — BPE-8192 train + load."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def prepare_mod():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if "prepare" in sys.modules:
        importlib.reload(sys.modules["prepare"])
    import prepare
    return prepare


@pytest.fixture
def tiny_field_corpus() -> list[str]:
    """A tiny but realistic corpus of field texts to train BPE on."""
    return [
        "Pantocid DSR",
        "Pantoprazole Gastro-Resistant and Domperidone",
        "Synthacid",
        "Paracetamol",
        "Synthbol",
        "M.R.P.Rs.: 99.50",
        "B.No.:GTF3406A",
        "Exp. Dt.: 09/2026",
        "Mfg. Dt.: 11/2024",
        "NEON LABORATORIES LIMITED",
        "ABC LABS PVT LTD",
        "SCHEDULE H PRESCRIPTION DRUG-CAUTION",
        "FOR EXTERNAL USE ONLY",
        "500MG / 2 ML",
        "10MG / 5 ML",
    ] * 10  # repeat for enough merges


def test_train_tokenizer_writes_file(
    prepare_mod, tiny_field_corpus: list[str], tmp_path: Path
) -> None:
    out = tmp_path / "tokenizer.json"
    prepare_mod.train_tokenizer(
        corpus=tiny_field_corpus,
        out_path=out,
        vocab_size=512,  # small for fast testing
    )
    assert out.is_file()
    # tokenizer.json should be valid JSON with a vocab and merges.
    data = json.loads(out.read_text())
    assert "model" in data
    assert "added_tokens" in data


def test_get_tokenizer_loads_from_file(
    prepare_mod, tiny_field_corpus: list[str], tmp_path: Path
) -> None:
    out = tmp_path / "tokenizer.json"
    prepare_mod.train_tokenizer(corpus=tiny_field_corpus, out_path=out, vocab_size=512)
    tok = prepare_mod.get_tokenizer(out)
    # Encode a known string; decode should round-trip (modulo whitespace BPE adds).
    enc = tok.encode("Pantocid")
    assert len(enc.ids) > 0


def test_special_tokens_are_in_vocab(
    prepare_mod, tiny_field_corpus: list[str], tmp_path: Path
) -> None:
    """All SPECIAL_TOKENS must be added as added_tokens during training."""
    out = tmp_path / "tokenizer.json"
    prepare_mod.train_tokenizer(corpus=tiny_field_corpus, out_path=out, vocab_size=512)
    tok = prepare_mod.get_tokenizer(out)
    for st in prepare_mod.SPECIAL_TOKENS:
        token_id = tok.token_to_id(st)
        assert token_id is not None, f"Special token {st!r} missing from tokenizer"


def test_tokenizer_round_trip_preserves_special_tokens(
    prepare_mod, tiny_field_corpus: list[str], tmp_path: Path
) -> None:
    """An XML output sequence encodes and decodes losslessly (no special-token mangling)."""
    out = tmp_path / "tokenizer.json"
    prepare_mod.train_tokenizer(corpus=tiny_field_corpus, out_path=out, vocab_size=512)
    tok = prepare_mod.get_tokenizer(out)
    seq = (
        prepare_mod.BOS_TOKEN
        + "<brand_name>Pantocid</brand_name>"
        + prepare_mod.EOS_TOKEN
    )
    enc = tok.encode(seq)
    decoded = tok.decode(enc.ids, skip_special_tokens=False)
    # The decoded sequence should still contain all special tokens we put in.
    assert "<s>" in decoded
    assert "</s>" in decoded
    assert "<brand_name>" in decoded
    assert "</brand_name>" in decoded
    assert "Pantocid" in decoded
```

- [ ] **Step 2: Run, verify failure**

```bash
uv run pytest tests/test_tokenizer.py -v 2>&1 | tail -20
```

Expected: tests FAIL — `train_tokenizer` and `get_tokenizer` not yet defined.

- [ ] **Step 3: Add `tokenizers` to dev extras (if not already)**

Inspect `pyproject.toml`. The `tokenizers` package is currently in `ml` (along with torch). For the lightweight pytest run on the Mac venv (which only has dev extras), we need `tokenizers` available too.

Edit `/Users/apple/AutoLearnMeds/pyproject.toml`. Move `tokenizers>=0.19` from `ml` to a new shared group, or simply add it to `dev`:

Find the `dev` block and update it from:

```toml
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.5",
]
```

to:

```toml
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.5",
    "tokenizers>=0.19",  # used by prepare.py (also in ml; here for local tests)
]
```

Then re-sync:

```bash
uv sync --extra dev
```

Expected: `+ tokenizers==0.x.x` (or already present from a prior sync).

- [ ] **Step 4: Add the tokenizer section to `prepare.py`**

Append to `/Users/apple/AutoLearnMeds/prepare.py`:

```python


# === Tokenizer ===

def train_tokenizer(
    corpus: list[str],
    out_path: Path | str,
    vocab_size: int = TOKENIZER_VOCAB_SIZE,
) -> None:
    """Train a BPE tokenizer on `corpus` and save to `out_path` (tokenizer.json).

    All SPECIAL_TOKENS are registered as added_tokens (single-id, not splittable
    by BPE), so they survive encode/decode unchanged.

    `corpus` is a list of strings — typically the field text values from
    train.jsonl (NOT the XML-formatted output, since SPECIAL_TOKENS handle the
    structural part).
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<pad>", *SPECIAL_TOKENS],
        show_progress=False,
    )
    tok.train_from_iterator(corpus, trainer=trainer)
    tok.save(str(out_path))


def get_tokenizer(path: Path | str):
    """Load a saved tokenizer.json from `path`. Returns a tokenizers.Tokenizer."""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(Path(path)))
```

- [ ] **Step 5: Run, verify tokenizer tests pass**

```bash
uv run pytest tests/test_tokenizer.py -v
```

Expected: all 4 PASSED.

- [ ] **Step 6: Run the full suite**

```bash
uv run pytest -v 2>&1 | tail -3
```

Expected: 43 PASSED (39 prior + 4 new).

- [ ] **Step 7: Commit**

```bash
git add prepare.py pyproject.toml uv.lock tests/test_tokenizer.py
git commit -m "feat(prepare): BPE-8192 tokenizer with all special tokens registered

train_tokenizer(corpus, out_path, vocab_size=8192) trains a BPE on raw
field texts and registers all SPECIAL_TOKENS (BOS/EOS plus open/close
per field) as added tokens. get_tokenizer(path) loads it back.
Special tokens survive encode/decode round-trip unchanged.

tokenizers>=0.19 moved into dev extras so local lightweight tests can
exercise the tokenizer without ml extras.

4 tests cover file write, load, special-token vocab membership, and
encode/decode round-trip for an XML-formatted sequence.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `prepare.py` — normalization + per-field F1 + macro F1

**Files:**
- Modify: `prepare.py` (append normalization + metric section)
- Create: `tests/test_metric.py`

- [ ] **Step 1: Write the failing test**

Create `/Users/apple/AutoLearnMeds/tests/test_metric.py`:

```python
"""Tests for prepare.py normalization + F1 metric."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def prepare_mod():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if "prepare" in sys.modules:
        importlib.reload(sys.modules["prepare"])
    import prepare
    return prepare


# --- normalize_field_value ---

def test_normalize_lowercases(prepare_mod) -> None:
    assert prepare_mod.normalize_field_value("Pantocid DSR") == "pantocid dsr"


def test_normalize_strips(prepare_mod) -> None:
    assert prepare_mod.normalize_field_value("  Pantocid  ") == "pantocid"


def test_normalize_collapses_whitespace(prepare_mod) -> None:
    assert prepare_mod.normalize_field_value("Pantocid    DSR\t\nfoo") == "pantocid dsr foo"


def test_normalize_handles_none_and_empty(prepare_mod) -> None:
    assert prepare_mod.normalize_field_value(None) == ""
    assert prepare_mod.normalize_field_value("") == ""


# --- compute_field_f1 ---

def test_field_f1_perfect(prepare_mod) -> None:
    """All matches → F1 = 1.0."""
    preds = [
        {"brand_name": "Pantocid"},
        {"brand_name": "Synthacid"},
    ]
    truths = [
        {"brand_name": "Pantocid"},
        {"brand_name": "Synthacid"},
    ]
    f1 = prepare_mod.compute_field_f1(preds, truths, "brand_name")
    assert f1 == 1.0


def test_field_f1_all_wrong(prepare_mod) -> None:
    """Zero matches → F1 = 0.0."""
    preds = [{"brand_name": "X"}, {"brand_name": "Y"}]
    truths = [{"brand_name": "Pantocid"}, {"brand_name": "Synthacid"}]
    f1 = prepare_mod.compute_field_f1(preds, truths, "brand_name")
    assert f1 == 0.0


def test_field_f1_normalization_applied(prepare_mod) -> None:
    """Casing differences should not penalize."""
    preds = [{"brand_name": "PANTOCID"}]
    truths = [{"brand_name": "pantocid"}]
    f1 = prepare_mod.compute_field_f1(preds, truths, "brand_name")
    assert f1 == 1.0


def test_field_f1_partial_credit(prepare_mod) -> None:
    """4 records, 2 with matching brand_name, 2 with mismatched.
    TP=2, FP=2 (predicted but wrong), FN=2 (truth but wrong predicted).
    Precision=2/4=0.5, Recall=2/4=0.5, F1=0.5.
    """
    preds = [
        {"brand_name": "A"},
        {"brand_name": "X"},  # wrong
        {"brand_name": "B"},
        {"brand_name": "Y"},  # wrong
    ]
    truths = [
        {"brand_name": "A"},
        {"brand_name": "B"},
        {"brand_name": "B"},
        {"brand_name": "C"},
    ]
    f1 = prepare_mod.compute_field_f1(preds, truths, "brand_name")
    assert f1 == pytest.approx(0.5, abs=1e-6)


def test_field_f1_handles_missing_field_in_pred(prepare_mod) -> None:
    """If pred is missing a field but truth has it, that's a false negative."""
    preds = [{}]  # no brand_name predicted
    truths = [{"brand_name": "A"}]
    f1 = prepare_mod.compute_field_f1(preds, truths, "brand_name")
    assert f1 == 0.0


def test_field_f1_handles_missing_field_in_truth(prepare_mod) -> None:
    """If truth doesn't have a field and pred predicts something, that's a FP."""
    preds = [{"brand_name": "A"}]
    truths = [{}]
    f1 = prepare_mod.compute_field_f1(preds, truths, "brand_name")
    assert f1 == 0.0


def test_field_f1_both_missing_is_zero_not_one(prepare_mod) -> None:
    """If neither has the field, no TP/FP/FN — F1 is undefined; return 0.0 by convention."""
    preds = [{}]
    truths = [{}]
    f1 = prepare_mod.compute_field_f1(preds, truths, "brand_name")
    assert f1 == 0.0


# --- compute_metrics ---

def test_compute_metrics_returns_macro_and_per_field(prepare_mod) -> None:
    preds = [
        {"brand_name": "A", "mrp": "99"},
        {"brand_name": "B"},
    ]
    truths = [
        {"brand_name": "A", "mrp": "99"},
        {"brand_name": "B", "warnings": "X"},
    ]
    m = prepare_mod.compute_metrics(preds, truths)
    assert "macro_f1" in m
    assert "per_field_f1" in m
    assert isinstance(m["per_field_f1"], dict)
    # brand_name perfect (2/2) -> 1.0
    assert m["per_field_f1"]["brand_name"] == pytest.approx(1.0)


def test_compute_metrics_macro_averages_high_frequency_only(prepare_mod) -> None:
    """compute_metrics's macro_f1 averages only HIGH_FREQUENCY_FIELDS (10 fields)."""
    # Make every high-freq field perfect (1.0). manufacturer/quantity should NOT be averaged in.
    preds = [{f: "x" for f in prepare_mod.HIGH_FREQUENCY_FIELDS}] * 5
    truths = [{f: "x" for f in prepare_mod.HIGH_FREQUENCY_FIELDS}] * 5
    m = prepare_mod.compute_metrics(preds, truths)
    assert m["macro_f1"] == pytest.approx(1.0)


def test_compute_metrics_macro_excludes_quantity_and_manufacturer(prepare_mod) -> None:
    """Even if quantity/manufacturer are perfect, they don't influence macro_f1
    (they aren't in HIGH_FREQUENCY_FIELDS)."""
    # All high-freq are wrong (F1=0), but quantity is perfect.
    preds = [{"quantity": "x"}] * 3
    truths = [{"quantity": "x"}] * 3
    m = prepare_mod.compute_metrics(preds, truths)
    assert m["macro_f1"] == 0.0
    assert m["per_field_f1"].get("quantity") == 1.0
```

- [ ] **Step 2: Run, verify failure**

```bash
uv run pytest tests/test_metric.py -v 2>&1 | tail -10
```

Expected: tests FAIL — functions not defined.

- [ ] **Step 3: Add the normalization + metric section to `prepare.py`**

Append to `/Users/apple/AutoLearnMeds/prepare.py`:

```python


# === Normalization + F1 ===

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_field_value(s: str | None) -> str:
    """Light normalization: lowercase + strip + collapse internal whitespace.

    Phase-1 deliberately does NOT strip prefixes like "B.No.:" or parse dates;
    those are Phase-7 sweep candidates.
    """
    if not s:
        return ""
    return _WHITESPACE_RE.sub(" ", s.strip()).lower()


def compute_field_f1(
    predictions: list[dict[str, str]],
    truths: list[dict[str, str]],
    field: str,
) -> float:
    """F1 for a single field across N records.

    A predicted text for `field` matches truth iff their normalized values
    are equal AND truth contains the field. Records where neither has the
    field contribute nothing (no TP/FP/FN).

    Returns 0.0 when precision+recall == 0.
    """
    if len(predictions) != len(truths):
        raise ValueError(f"len mismatch: {len(predictions)} vs {len(truths)}")

    tp = fp = fn = 0
    for pred, truth in zip(predictions, truths):
        p = pred.get(field)
        t = truth.get(field)
        if p is None and t is None:
            continue
        if p is not None and t is None:
            fp += 1
            continue
        if p is None and t is not None:
            fn += 1
            continue
        # Both present.
        if normalize_field_value(p) == normalize_field_value(t):
            tp += 1
        else:
            fp += 1
            fn += 1

    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_metrics(
    predictions: list[dict[str, str]],
    truths: list[dict[str, str]],
) -> dict[str, Any]:
    """Compute macro_f1 (over HIGH_FREQUENCY_FIELDS) + per-field F1 (over all 12).

    Returns:
        {
            "macro_f1": float in [0, 1],
            "per_field_f1": {field: float, ...} for ALL 12 fields,
            "n_examples": int,
        }
    """
    per_field = {f: compute_field_f1(predictions, truths, f) for f in FIELD_ORDER}
    high_freq = [v for f, v in per_field.items() if f in HIGH_FREQUENCY_FIELDS]
    macro = sum(high_freq) / len(high_freq) if high_freq else 0.0
    return {
        "macro_f1": macro,
        "per_field_f1": per_field,
        "n_examples": len(predictions),
    }
```

- [ ] **Step 4: Run, verify tests pass**

```bash
uv run pytest tests/test_metric.py -v
```

Expected: all 13 PASSED.

- [ ] **Step 5: Run the full suite**

```bash
uv run pytest -v 2>&1 | tail -3
```

Expected: 56 PASSED (43 prior + 13 new).

- [ ] **Step 6: Commit**

```bash
git add prepare.py tests/test_metric.py
git commit -m "feat(prepare): normalize + per-field F1 + macro_f1 metric

normalize_field_value: lowercase + strip + whitespace collapse. No
prefix stripping in Phase 1 (deferred to Phase-7 sweep).

compute_field_f1(preds, truths, field): standard F1 with the convention
that records where NEITHER side has the field contribute nothing.
Returns 0 on empty/zero-precision-or-recall.

compute_metrics: macro_f1 averaged over HIGH_FREQUENCY_FIELDS (10
fields), per_field_f1 reported over ALL 12 fields. Two fields
(quantity, manufacturer) are excluded from macro_f1 because their
0.7%/0.6% presence rate makes per-field F1 a noise floor.

13 tests cover normalization, perfect/zero/partial F1, missing-side
handling, and the macro/per-field structure.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `prepare.py` — image preprocessing + Dataset + DataLoader

**Files:**
- Modify: `prepare.py` (append image + dataset section)
- Create: `tests/test_torch_pipeline.py`
- Modify: `pyproject.toml` (add a `colab` pytest marker)

This task introduces torch dependencies. The tests are marked `@pytest.mark.colab` and skipped locally; the controller runs them on Colab.

- [ ] **Step 1: Configure pytest's "colab" marker**

Edit `/Users/apple/AutoLearnMeds/pyproject.toml`. Find the `[tool.pytest.ini_options]` block and update it from:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"
```

to:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short -m 'not colab'"
markers = [
    "colab: requires the Colab runtime (torch + ml extras + GPU)",
]
```

This makes default `pytest` skip Colab-only tests; running `pytest -m colab` runs only those.

- [ ] **Step 2: Write the failing test**

Create `/Users/apple/AutoLearnMeds/tests/test_torch_pipeline.py`:

```python
"""Tests for prepare.py image preprocessing + Dataset + DataLoader.

Marked @pytest.mark.colab — requires torch, transformers, Pillow. Skipped
in local Mac dev runs (default pytest config); run on Colab via
`pytest -m colab` from /workspace.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.colab


@pytest.fixture
def prepare_mod():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if "prepare" in sys.modules:
        importlib.reload(sys.modules["prepare"])
    import prepare
    return prepare


def test_preprocess_image_returns_tensor(
    prepare_mod, synthetic_images_dir: Path
) -> None:
    import torch  # type: ignore[import-not-found]
    img = synthetic_images_dir / "IMG_synth_001.jpeg"
    t = prepare_mod.preprocess_image(img)
    assert isinstance(t, torch.Tensor)
    assert t.shape == (3, prepare_mod.IMAGE_SIZE, prepare_mod.IMAGE_SIZE)


def test_preprocess_image_letterbox_aspect(
    prepare_mod, tmp_path: Path
) -> None:
    """Non-square images should be letterboxed to IMAGE_SIZE x IMAGE_SIZE."""
    from PIL import Image

    img_path = tmp_path / "wide.jpeg"
    Image.new("RGB", (256, 64), color=(127, 127, 127)).save(img_path, "JPEG")
    t = prepare_mod.preprocess_image(img_path)
    assert t.shape == (3, prepare_mod.IMAGE_SIZE, prepare_mod.IMAGE_SIZE)


def test_pharma_label_dataset_loads(
    prepare_mod,
    synthetic_jsonl_dir: Path,
    synthetic_images_dir: Path,
    tiny_processed_dir: Path,
    project_root: Path,
) -> None:
    # Build processed JSONL from synthetic fixtures
    import sys
    sys.path.insert(0, str(project_root / "scripts"))
    import build_processed
    build_processed.build_processed(
        gold_standard_path=synthetic_jsonl_dir / "gold_standard.jsonl",
        splits_path=synthetic_jsonl_dir / "splits.json",
        images_root=synthetic_images_dir,
        out_dir=tiny_processed_dir,
        image_path_prefix="raw/raw_images",
    )
    ds = prepare_mod.PharmaLabelDataset(
        jsonl_path=tiny_processed_dir / "train.jsonl",
        images_root=synthetic_images_dir,
        path_strip_prefix="raw/raw_images/",
    )
    assert len(ds) == 3
    item = ds[0]
    assert "image" in item
    assert "target_text" in item
    assert "image_id" in item
    assert item["image"].shape == (3, prepare_mod.IMAGE_SIZE, prepare_mod.IMAGE_SIZE)
    assert isinstance(item["target_text"], str)


def test_get_dataloader_iterates(
    prepare_mod,
    synthetic_jsonl_dir: Path,
    synthetic_images_dir: Path,
    tiny_processed_dir: Path,
    project_root: Path,
) -> None:
    import sys
    sys.path.insert(0, str(project_root / "scripts"))
    import build_processed
    build_processed.build_processed(
        gold_standard_path=synthetic_jsonl_dir / "gold_standard.jsonl",
        splits_path=synthetic_jsonl_dir / "splits.json",
        images_root=synthetic_images_dir,
        out_dir=tiny_processed_dir,
        image_path_prefix="raw/raw_images",
    )
    dl = prepare_mod.get_dataloader(
        jsonl_path=tiny_processed_dir / "train.jsonl",
        images_root=synthetic_images_dir,
        path_strip_prefix="raw/raw_images/",
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(dl))
    assert batch["image"].shape[0] == 2  # batch dim
    assert len(batch["target_text"]) == 2
    assert len(batch["image_id"]) == 2
```

- [ ] **Step 3: Verify pytest skips the colab tests locally**

```bash
uv run pytest tests/test_torch_pipeline.py -v
```

Expected: `4 deselected` (because of `-m 'not colab'`). No failures.

- [ ] **Step 4: Add image + dataset section to `prepare.py`**

Append to `/Users/apple/AutoLearnMeds/prepare.py`:

```python


# === Image preprocessing ===

# Cached after first call so we don't redownload SigLIP processor each batch.
_PROCESSOR_CACHE: dict[str, Any] = {}


def _get_siglip_processor():
    """Lazy-load and cache the SigLIP image processor."""
    if "processor" not in _PROCESSOR_CACHE:
        from transformers import AutoProcessor

        _PROCESSOR_CACHE["processor"] = AutoProcessor.from_pretrained(
            "google/siglip-base-patch16-224"
        )
    return _PROCESSOR_CACHE["processor"]


def preprocess_image(image_path: Path | str):
    """Resize + letterbox + normalize an image. Returns a (3, IMAGE_SIZE, IMAGE_SIZE) torch tensor.

    Letterboxing preserves aspect ratio with mid-gray padding so non-square
    medicine boxes don't get squashed.
    """
    from PIL import Image
    import torch  # type: ignore[import-not-found]

    img = Image.open(str(image_path)).convert("RGB")
    w, h = img.size
    scale = IMAGE_SIZE / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(127, 127, 127))
    canvas.paste(img, ((IMAGE_SIZE - nw) // 2, (IMAGE_SIZE - nh) // 2))

    proc = _get_siglip_processor()
    out = proc(images=canvas, return_tensors="pt")
    return out["pixel_values"].squeeze(0)  # (3, H, W)


# === Dataset + DataLoader ===

class PharmaLabelDataset:
    """Iterates a processed JSONL file. Yields {image, target_text, image_id} dicts.

    `image` is the preprocessed tensor.
    `target_text` is format_output(record) — the canonical XML decoder target.
    `image_id` is the record's image_id (for evaluation cross-reference).
    """

    def __init__(
        self,
        jsonl_path: Path | str,
        images_root: Path | str,
        path_strip_prefix: str = "raw/raw_images/",
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.images_root = Path(images_root)
        self.path_strip_prefix = path_strip_prefix
        self.records: list[dict] = [
            json.loads(line)
            for line in self.jsonl_path.read_text().splitlines()
            if line.strip()
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        rel = rec["image_path"]
        if rel.startswith(self.path_strip_prefix):
            rel = rel[len(self.path_strip_prefix):]
        img_path = self.images_root / rel
        return {
            "image": preprocess_image(img_path),
            "target_text": format_output(rec),
            "image_id": rec["image_id"],
            "fields_truth": {f: (rec.get("fields") or {}).get(f, {}).get("text")
                              for f in FIELD_ORDER
                              if (rec.get("fields") or {}).get(f, {}).get("text") is not None},
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Default-style collate that stacks tensors and lists everything else."""
    import torch  # type: ignore[import-not-found]

    out: dict[str, Any] = {}
    out["image"] = torch.stack([b["image"] for b in batch])
    out["target_text"] = [b["target_text"] for b in batch]
    out["image_id"] = [b["image_id"] for b in batch]
    out["fields_truth"] = [b["fields_truth"] for b in batch]
    return out


def get_dataloader(
    jsonl_path: Path | str,
    images_root: Path | str,
    path_strip_prefix: str = "raw/raw_images/",
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
):
    """PyTorch DataLoader over PharmaLabelDataset."""
    from torch.utils.data import DataLoader  # type: ignore[import-not-found]

    ds = PharmaLabelDataset(
        jsonl_path=jsonl_path,
        images_root=images_root,
        path_strip_prefix=path_strip_prefix,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate,
    )
```

- [ ] **Step 5: Lightweight tests still pass locally (no torch)**

```bash
uv run pytest -v 2>&1 | tail -3
```

Expected: 56 PASSED, 4 deselected. (Colab tests are skipped.)

- [ ] **Step 6: Commit**

```bash
git add prepare.py pyproject.toml tests/test_torch_pipeline.py
git commit -m "feat(prepare): image preprocess + Dataset + DataLoader (Colab-marked tests)

preprocess_image: load JPEG -> RGB -> aspect-preserving resize to fit
in IMAGE_SIZE x IMAGE_SIZE -> letterbox with mid-gray (rgb 127,127,127) ->
SigLIP AutoProcessor normalization. Returns (3, 224, 224) tensor.
Processor is cached after first call.

PharmaLabelDataset: iterates a processed JSONL, yielding
{image, target_text, image_id, fields_truth} where target_text is the
canonical XML decoder target and fields_truth is a {field: text} dict
of the present-only ground-truth fields (consumed by evaluate()).

get_dataloader: thin wrapper around torch.utils.data.DataLoader with a
custom collate that stacks images and lists everything else.

torch / transformers / Pillow are all lazy-imported inside functions so
the module imports cleanly in the local Mac venv.

Pytest 'colab' marker added; default config skips colab-marked tests
locally. The 4 torch-pipeline tests are marked colab; controller runs
them on the Colab A100 via SSH.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: `prepare.py` — public `evaluate()` and test-set firewall

**Files:**
- Modify: `prepare.py` (append eval API + firewall section)
- Create: `tests/test_evaluate.py` (Colab-marked)

- [ ] **Step 1: Write the failing test**

Create `/Users/apple/AutoLearnMeds/tests/test_evaluate.py`:

```python
"""Tests for prepare.py public eval API: evaluate() + evaluate_test() firewall."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.colab


@pytest.fixture
def prepare_mod():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if "prepare" in sys.modules:
        importlib.reload(sys.modules["prepare"])
    import prepare
    return prepare


class _StubModelAlwaysEmpty:
    """Returns an empty (BOS+EOS) string for every input — simulates an
    untrained model."""

    def predict_text(self, batch_images, max_new_tokens: int = 256) -> list[str]:
        return [""] * len(batch_images)


def test_evaluate_runs_on_stub_returns_zero(
    prepare_mod,
    synthetic_jsonl_dir: Path,
    synthetic_images_dir: Path,
    tiny_processed_dir: Path,
    project_root: Path,
) -> None:
    import sys
    sys.path.insert(0, str(project_root / "scripts"))
    import build_processed
    build_processed.build_processed(
        gold_standard_path=synthetic_jsonl_dir / "gold_standard.jsonl",
        splits_path=synthetic_jsonl_dir / "splits.json",
        images_root=synthetic_images_dir,
        out_dir=tiny_processed_dir,
        image_path_prefix="raw/raw_images",
    )
    metrics = prepare_mod.evaluate(
        model=_StubModelAlwaysEmpty(),
        jsonl_path=tiny_processed_dir / "val.jsonl",
        images_root=synthetic_images_dir,
        path_strip_prefix="raw/raw_images/",
        batch_size=4,
    )
    assert "macro_f1" in metrics
    assert metrics["macro_f1"] == 0.0  # stub predicts nothing
    assert "per_field_f1" in metrics
    assert metrics["n_examples"] == 1


def test_evaluate_runs_on_stub_with_partial_predictions(
    prepare_mod,
    synthetic_jsonl_dir: Path,
    synthetic_images_dir: Path,
    tiny_processed_dir: Path,
    project_root: Path,
) -> None:
    """A stub model that always returns the correct brand_name text gets a
    boost on that field's F1."""
    import sys
    sys.path.insert(0, str(project_root / "scripts"))
    import build_processed
    build_processed.build_processed(
        gold_standard_path=synthetic_jsonl_dir / "gold_standard.jsonl",
        splits_path=synthetic_jsonl_dir / "splits.json",
        images_root=synthetic_images_dir,
        out_dir=tiny_processed_dir,
        image_path_prefix="raw/raw_images",
    )

    class _StubBrandOnly:
        def predict_text(self, batch_images, max_new_tokens=256):
            xml = (
                prepare_mod.BOS_TOKEN
                + "<drug_name>Synthcol</drug_name>"
                + prepare_mod.EOS_TOKEN
            )
            return [xml] * len(batch_images)

    metrics = prepare_mod.evaluate(
        model=_StubBrandOnly(),
        jsonl_path=tiny_processed_dir / "val.jsonl",
        images_root=synthetic_images_dir,
        path_strip_prefix="raw/raw_images/",
        batch_size=4,
    )
    # val has 1 record (IMG_synth_004) with drug_name="Synthcol".
    # Stub's prediction matches → drug_name F1 is 1.0.
    assert metrics["per_field_f1"]["drug_name"] == pytest.approx(1.0)


def test_evaluate_test_is_gated(prepare_mod, tiny_processed_dir: Path) -> None:
    """evaluate_test must require an explicit confirmation flag."""
    with pytest.raises(RuntimeError, match="explicit_consent"):
        prepare_mod.evaluate_test(
            model=_StubModelAlwaysEmpty(),
            jsonl_path=tiny_processed_dir / "test.jsonl",
            images_root=tiny_processed_dir,
            path_strip_prefix="raw/",
        )


def test_evaluate_test_logs_audit_entry(
    prepare_mod,
    synthetic_jsonl_dir: Path,
    synthetic_images_dir: Path,
    tiny_processed_dir: Path,
    project_root: Path,
    tmp_path: Path,
) -> None:
    import sys
    sys.path.insert(0, str(project_root / "scripts"))
    import build_processed
    build_processed.build_processed(
        gold_standard_path=synthetic_jsonl_dir / "gold_standard.jsonl",
        splits_path=synthetic_jsonl_dir / "splits.json",
        images_root=synthetic_images_dir,
        out_dir=tiny_processed_dir,
        image_path_prefix="raw/raw_images",
    )
    audit = tmp_path / "evaluate_test_audit.log"
    prepare_mod.evaluate_test(
        model=_StubModelAlwaysEmpty(),
        jsonl_path=tiny_processed_dir / "test.jsonl",
        images_root=synthetic_images_dir,
        path_strip_prefix="raw/raw_images/",
        explicit_consent="i_understand_this_consumes_the_test_set",
        audit_log=audit,
    )
    assert audit.is_file()
    assert "evaluate_test invoked" in audit.read_text()
```

- [ ] **Step 2: Verify these are deselected locally**

```bash
uv run pytest tests/test_evaluate.py -v
```

Expected: `4 deselected` (Colab marker).

- [ ] **Step 3: Add the eval API + firewall section to `prepare.py`**

Append to `/Users/apple/AutoLearnMeds/prepare.py`:

```python


# === Public eval API ===

def evaluate(
    model,
    jsonl_path: Path | str,
    images_root: Path | str,
    path_strip_prefix: str = "raw/raw_images/",
    batch_size: int = 16,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    """Run `model.predict_text(batch_images, max_new_tokens) -> list[str]`
    over the given JSONL split, parse the predicted XML, compute
    {macro_f1, per_field_f1, n_examples}.

    `model` must implement `predict_text(batch_images, max_new_tokens)` returning
    a list of strings (one per image). The model itself is opaque — train.py
    decides how to wrap its decoder behind this method.
    """
    dl = get_dataloader(
        jsonl_path=jsonl_path,
        images_root=images_root,
        path_strip_prefix=path_strip_prefix,
        batch_size=batch_size,
        shuffle=False,
    )
    predictions: list[dict[str, str]] = []
    truths: list[dict[str, str]] = []
    for batch in dl:
        outputs = model.predict_text(batch["image"], max_new_tokens=max_new_tokens)
        for out_text, fields_truth in zip(outputs, batch["fields_truth"]):
            predictions.append(parse_output(out_text))
            truths.append(fields_truth)
    return compute_metrics(predictions, truths)


_TEST_FIREWALL_CONSENT = "i_understand_this_consumes_the_test_set"


def evaluate_test(
    model,
    jsonl_path: Path | str,
    images_root: Path | str,
    path_strip_prefix: str = "raw/raw_images/",
    batch_size: int = 16,
    max_new_tokens: int = 256,
    explicit_consent: str = "",
    audit_log: Path | str | None = None,
) -> dict[str, Any]:
    """Gated wrapper around evaluate() for the held-out TEST split.

    Refuses to run unless `explicit_consent == "i_understand_this_consumes_the_test_set"`.
    Every successful call appends a timestamped line to `audit_log`
    (default: ./experiments/evaluate_test_audit.log).
    """
    import datetime as _dt

    if explicit_consent != _TEST_FIREWALL_CONSENT:
        raise RuntimeError(
            "evaluate_test requires explicit_consent="
            f"{_TEST_FIREWALL_CONSENT!r}. Test-set evals must be rare and intentional."
        )

    audit_path = Path(audit_log) if audit_log else Path("experiments/evaluate_test_audit.log")
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = evaluate(
        model=model,
        jsonl_path=jsonl_path,
        images_root=images_root,
        path_strip_prefix=path_strip_prefix,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    line = (
        f"{_dt.datetime.now(_dt.UTC).isoformat()} evaluate_test invoked "
        f"jsonl={jsonl_path} macro_f1={metrics['macro_f1']:.4f} "
        f"n={metrics['n_examples']}\n"
    )
    with audit_path.open("a") as fh:
        fh.write(line)
    return metrics
```

- [ ] **Step 4: Run the full local suite**

```bash
uv run pytest -v 2>&1 | tail -3
```

Expected: 56 PASSED, 8 deselected (4 prior Colab + 4 new Colab).

- [ ] **Step 5: Commit**

```bash
git add prepare.py tests/test_evaluate.py
git commit -m "feat(prepare): public evaluate() + gated evaluate_test firewall

evaluate(model, jsonl_path, ...) drives a DataLoader over the split, calls
model.predict_text(batch_images, max_new_tokens) -> list[str], parses each
output, and returns compute_metrics(...) — i.e., {macro_f1, per_field_f1,
n_examples}. The model interface is intentionally minimal.

evaluate_test wraps evaluate() with two safety rails:
1. Refuses to run unless caller passes
   explicit_consent='i_understand_this_consumes_the_test_set'.
2. On success, appends a timestamped line to
   experiments/evaluate_test_audit.log (configurable).

Together these enforce the spec §5.7 / §1.6 'test set is sacred' rule:
the agent's train.py must not import evaluate_test, and any human use is
recorded.

4 colab-marked tests cover stub-zero, stub-partial, gating, and audit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: `scripts/build_data_card.py` — auto-generated dataset card

**Files:**
- Create: `scripts/build_data_card.py`
- Create: `tests/test_build_data_card.py`

- [ ] **Step 1: Write the failing test**

Create `/Users/apple/AutoLearnMeds/tests/test_build_data_card.py`:

```python
"""Tests for scripts/build_data_card.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module(project_root: Path):
    spec = importlib.util.spec_from_file_location(
        "build_data_card", project_root / "scripts" / "build_data_card.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_module_loads(project_root: Path) -> None:
    mod = _load_module(project_root)
    assert hasattr(mod, "build_data_card")


def test_build_data_card_writes_md(
    project_root: Path,
    synthetic_jsonl_dir: Path,
    synthetic_images_dir: Path,
    tiny_processed_dir: Path,
    tmp_path: Path,
) -> None:
    # Need processed JSONLs first.
    import sys
    sys.path.insert(0, str(project_root / "scripts"))
    import build_processed
    build_processed.build_processed(
        gold_standard_path=synthetic_jsonl_dir / "gold_standard.jsonl",
        splits_path=synthetic_jsonl_dir / "splits.json",
        images_root=synthetic_images_dir,
        out_dir=tiny_processed_dir,
        image_path_prefix="raw/raw_images",
    )

    out_path = tmp_path / "data_card.md"
    mod = _load_module(project_root)
    mod.build_data_card(processed_dir=tiny_processed_dir, out_path=out_path)
    assert out_path.is_file()
    text = out_path.read_text()

    # Required sections per Gebru 2018 datasheet conventions
    assert "# Data Card" in text
    assert "## Composition" in text
    assert "## Splits" in text
    assert "## Field presence" in text
    # Per-split sizes
    assert "train" in text
    assert "val" in text
    assert "test" in text
    # The actual sizes from synthetic fixtures
    assert "3" in text  # train size
    assert "1" in text  # val size
    # Field names
    assert "brand_name" in text


def test_build_data_card_handles_empty_split(
    project_root: Path, tmp_path: Path
) -> None:
    """Should not crash if a split JSONL is empty."""
    proc = tmp_path / "processed"
    proc.mkdir()
    for sp in ("train", "val", "test"):
        (proc / f"{sp}.jsonl").write_text("")

    mod = _load_module(project_root)
    out_path = tmp_path / "data_card.md"
    mod.build_data_card(processed_dir=proc, out_path=out_path)
    text = out_path.read_text()
    assert "train" in text
    assert "0" in text  # zero-size split
```

- [ ] **Step 2: Run, verify failure**

```bash
uv run pytest tests/test_build_data_card.py -v
```

Expected: 3 FAIL.

- [ ] **Step 3: Implement `scripts/build_data_card.py`**

Create `/Users/apple/AutoLearnMeds/scripts/build_data_card.py`:

```python
#!/usr/bin/env python3
"""Auto-generate data/data_card.md from data/processed/{train,val,test}.jsonl.

Follows Gebru et al. 2018 'Datasheets for Datasets' conventions: motivation,
composition, splits, field-presence stats, distribution of categorical
metadata. Designed to be regenerated whenever processed/* changes.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _read_split(processed_dir: Path, split: str) -> list[dict]:
    p = processed_dir / f"{split}.jsonl"
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _field_presence(records: list[dict]) -> Counter:
    c: Counter = Counter()
    for r in records:
        for f in (r.get("fields") or {}):
            if (r["fields"][f] or {}).get("text"):
                c[f] += 1
    return c


def _categorical_dist(records: list[dict], key: str) -> Counter:
    return Counter(r.get(key) for r in records)


def build_data_card(
    processed_dir: Path | str,
    out_path: Path | str,
) -> None:
    processed_dir = Path(processed_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train = _read_split(processed_dir, "train")
    val = _read_split(processed_dir, "val")
    test = _read_split(processed_dir, "test")
    all_recs = train + val + test

    # --- Header
    lines: list[str] = [
        "# Data Card",
        "",
        "Auto-generated by `scripts/build_data_card.py`. Re-run after every "
        "data refresh.",
        "",
        "Following Gebru et al. 2018, *Datasheets for Datasets* "
        "(`papers/gebru_2018_datasheets.pdf`).",
        "",
    ]

    # --- Motivation
    lines += [
        "## Motivation",
        "",
        "This dataset supports training and evaluating a vision-language model "
        "for structured information extraction from pharmaceutical product "
        "label images. See the design spec, `docs/superpowers/specs/`, for the "
        "research goal.",
        "",
    ]

    # --- Composition
    lines += [
        "## Composition",
        "",
        f"- **Total labeled records:** {len(all_recs)}",
        f"- **Image format:** JPEG, RGB, varying resolutions (resized at "
        f"training time to 224×224 with letterbox padding).",
        f"- **Fields per record:** variable; up to 12 OCR fields.",
        "",
    ]

    # --- Splits
    lines += [
        "## Splits",
        "",
        "| Split | Records | Notes |",
        "|---|---|---|",
        f"| train | {len(train)} | Used for parameter updates. |",
        f"| val   | {len(val)} | Used by the autoresearch loop's macro_f1 metric. |",
        f"| test  | {len(test)} | **Frozen.** Only `evaluate_test()` may read these. |",
        "",
    ]

    # --- Field presence
    lines += [
        "## Field presence",
        "",
        "Per-field presence counts in train (the agent never sees val/test field stats).",
        "",
        "| Field | Count | % |",
        "|---|---|---|",
    ]
    fp = _field_presence(train)
    for f, count in sorted(fp.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * count / max(len(train), 1)
        lines.append(f"| {f} | {count} | {pct:.1f}% |")
    lines.append("")

    # --- view_type / pack_type / medicine_name
    for key in ("view_type", "pack_type", "medicine_name"):
        lines += [f"## Distribution: `{key}` (train)", ""]
        dist = _categorical_dist(train, key)
        if not dist:
            lines += ["(no values)", ""]
            continue
        if len(dist) > 20:
            lines += [f"(showing top 20 of {len(dist)} unique values)", ""]
        lines += ["| Value | Count |", "|---|---|"]
        for v, c in dist.most_common(20):
            lines.append(f"| `{v!r}` | {c} |")
        lines.append("")

    # --- Maintenance
    lines += [
        "## Maintenance",
        "",
        "- Source of truth: `gs://auto_learn_meds/raw/golden_set/` "
        "(`gold_standard.jsonl` + `splits.json`).",
        "- Regenerate this file: "
        "`uv run python scripts/build_data_card.py "
        "--processed-dir data/processed --out data/data_card.md`",
        "",
    ]

    out_path.write_text("\n".join(lines))
    print(f"[build_data_card] wrote {out_path} ({len(lines)} lines)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build_data_card(processed_dir=args.processed_dir, out_path=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Make executable**

```bash
chmod +x /Users/apple/AutoLearnMeds/scripts/build_data_card.py
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_build_data_card.py -v
```

Expected: all 3 PASSED.

- [ ] **Step 6: Run the full suite**

```bash
uv run pytest -v 2>&1 | tail -3
```

Expected: 59 PASSED, 8 deselected.

- [ ] **Step 7: Commit**

```bash
git add scripts/build_data_card.py tests/test_build_data_card.py
git commit -m "feat(scripts): build_data_card.py — auto-generated dataset card

Reads data/processed/{train,val,test}.jsonl, emits data/data_card.md
following Gebru 2018 datasheet conventions: motivation, composition,
splits, per-field presence (train only), categorical distributions
(view_type, pack_type, medicine_name).

Reports field-presence stats from train ONLY; val/test field stats are
not exposed (avoids leaking eval data into agent context).

3 tests cover module load, full-record generation against synthetic
fixtures, and empty-split robustness.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Phase 1 exit gate — run on Colab

**Files:** none — orchestration only.

This task is run by the controller (the agent driving subagent-driven-development) via the SSH tunnel to Colab.

- [ ] **Step 1: Push commits**

```bash
cd /Users/apple/AutoLearnMeds && git push origin phase-0-plumbing
```

- [ ] **Step 2: Pull on Colab + run build_processed.py against the real golden_set**

```bash
SSHPASS='cWbDRcUsDzyWkYXRdhno8IzD4yqwTjVl' sshpass -e ssh -o ConnectTimeout=20 autolearnmeds-colab "
cd /workspace
git pull --ff-only origin phase-0-plumbing
mkdir -p data/processed
uv run python scripts/build_processed.py \
  --gold-standard /mnt/gcs/raw/golden_set/gold_standard.jsonl \
  --splits        /mnt/gcs/raw/golden_set/splits.json \
  --images-root   /mnt/gcs/raw/raw_images \
  --out-dir       data/processed \
  --image-path-prefix raw/raw_images
echo '--- counts ---'
wc -l data/processed/*.jsonl
"
```

Expected output (approximate):

```
--- counts ---
564 data/processed/train.jsonl
111 data/processed/val.jsonl
162 data/processed/test.jsonl
837 total
```

- [ ] **Step 3: Train the BPE tokenizer on the real corpus**

```bash
SSHPASS='cWbDRcUsDzyWkYXRdhno8IzD4yqwTjVl' sshpass -e ssh -o ConnectTimeout=20 autolearnmeds-colab "
cd /workspace
uv run python -c \"
import json
from pathlib import Path
import prepare

corpus = []
for line in Path('data/processed/train.jsonl').read_text().splitlines():
    rec = json.loads(line)
    for f in prepare.FIELD_ORDER:
        text = (rec.get('fields') or {}).get(f, {}).get('text')
        if text:
            corpus.append(text)
print(f'corpus size: {len(corpus)} field values')
prepare.train_tokenizer(corpus=corpus, out_path='data/processed/tokenizer.json')
tok = prepare.get_tokenizer('data/processed/tokenizer.json')
print(f'vocab size: {tok.get_vocab_size()}')
print(f'<s> id: {tok.token_to_id(prepare.BOS_TOKEN)}')
print(f'<brand_name> id: {tok.token_to_id(\\\"<brand_name>\\\")}')
\"
"
```

Expected: prints corpus size (~thousands of field values), vocab size (8192), and IDs for special tokens.

- [ ] **Step 4: Generate data_card.md against real data**

```bash
SSHPASS='cWbDRcUsDzyWkYXRdhno8IzD4yqwTjVl' sshpass -e ssh -o ConnectTimeout=20 autolearnmeds-colab "
cd /workspace
uv run python scripts/build_data_card.py \
  --processed-dir data/processed \
  --out data/data_card.md
echo '--- head of data_card.md ---'
head -40 data/data_card.md
"
```

Expected: data_card.md is written; head shows the canonical sections.

- [ ] **Step 5: Run the full test suite on Colab (lightweight + colab-marked)**

```bash
SSHPASS='cWbDRcUsDzyWkYXRdhno8IzD4yqwTjVl' sshpass -e ssh -o ConnectTimeout=20 autolearnmeds-colab "
cd /workspace
uv run pytest -v 2>&1 | tail -10
echo '--- now with colab marker ---'
uv run pytest -m colab -v 2>&1 | tail -15
"
```

Expected:
- Default run: 59 PASSED, 8 deselected.
- Colab-marker run: 8 PASSED.

- [ ] **Step 6: Run evaluate() against a stub model on real val data**

```bash
SSHPASS='cWbDRcUsDzyWkYXRdhno8IzD4yqwTjVl' sshpass -e ssh -o ConnectTimeout=20 autolearnmeds-colab "
cd /workspace
uv run python -c \"
import prepare

class StubAlwaysEmpty:
    def predict_text(self, batch_images, max_new_tokens=256):
        return [''] * len(batch_images)

metrics = prepare.evaluate(
    model=StubAlwaysEmpty(),
    jsonl_path='data/processed/val.jsonl',
    images_root='/mnt/gcs/raw/raw_images',
    path_strip_prefix='raw/raw_images/',
    batch_size=8,
)
print(f'macro_f1 = {metrics[\\\"macro_f1\\\"]:.4f}')
print(f'n_examples = {metrics[\\\"n_examples\\\"]}')
print('per_field_f1:')
for f, v in sorted(metrics['per_field_f1'].items()):
    print(f'  {f:18s}: {v:.4f}')
\"
"
```

Expected: macro_f1 = 0.0000 (stub predicts nothing); n_examples = 111; all per_field_f1 = 0. **This is the Phase 1 exit gate.**

- [ ] **Step 7: Commit the generated tokenizer + data card and write the Phase-1 review**

The `data/processed/tokenizer.json` and `data/data_card.md` are useful to track in git for reproducibility. Override `.gitignore` for these specific files.

Edit `/Users/apple/AutoLearnMeds/data/processed/.gitignore` to add an explicit allow for `tokenizer.json`:

```
# Generated by scripts/build_processed.py — not source.
*.jsonl

# But DO track the trained tokenizer (required for reproducibility).
!tokenizer.json

# And keep the .gitignore itself.
!.gitignore
```

Then on Colab side, commit + push:

```bash
SSHPASS='cWbDRcUsDzyWkYXRdhno8IzD4yqwTjVl' sshpass -e ssh -o ConnectTimeout=20 autolearnmeds-colab "
cd /workspace
git add data/processed/tokenizer.json data/data_card.md
git diff --staged --stat
git commit -m 'data: tokenizer.json + data_card.md from real golden_set

Trained BPE-8192 on training-set field texts, generated data card with
per-field presence and categorical distributions. These are the locked
artifacts the autoresearch loop and paper-generation scripts depend on.'
git push origin phase-0-plumbing
"
```

- [ ] **Step 8: Write Phase 1 review**

Create `/Users/apple/AutoLearnMeds/docs/superpowers/reviews/phase1_review.md`:

```markdown
# Phase 1 Review — Data Ingestion & Schema Lock

**Tag:** `phase-1-complete`
**Date:** YYYY-MM-DD (fill in at commit time)
**Status:** GREEN — `evaluate()` runs end-to-end on real val data with a stub model.

## What was built

- `scripts/build_processed.py` — converts `gs://auto_learn_meds/raw/golden_set/{gold_standard.jsonl,splits.json}` + `raw_images/` into `data/processed/{train,val,test}.jsonl` with `image_id`, `image_path`, `image_hash` (sha256), and `split` fields added. Idempotent (sort-keyed JSON dump).
- `prepare.py` v1 (FROZEN — autoresearch agent must not modify):
  - **Constants:** `FIELD_ORDER` (12), `HIGH_FREQUENCY_FIELDS` (10), `SPECIAL_TOKENS`, `IMAGE_SIZE=224`, `TOKENIZER_VOCAB_SIZE=8192`.
  - **Output format:** `format_output(record)` -> Donut-style XML; `parse_output(text)` -> `{field: text}`.
  - **Tokenizer:** `train_tokenizer(corpus, out_path, vocab_size)`, `get_tokenizer(path)` (BPE with all special tokens registered).
  - **Metric:** `normalize_field_value`, `compute_field_f1`, `compute_metrics` (macro_f1 over HIGH_FREQUENCY_FIELDS).
  - **Image pipeline:** `preprocess_image` (224x224 letterbox + SigLIP normalization), `PharmaLabelDataset`, `get_dataloader`.
  - **Public API:** `evaluate(model, jsonl_path, ...)` and `evaluate_test(... explicit_consent=...)` (gated + audited).
- `scripts/build_data_card.py` — auto-generates `data/data_card.md` (Gebru 2018 datasheet style).

## What was verified

- ✅ 59 lightweight pytest tests pass locally (Mac, no torch).
- ✅ 8 colab-marked pytest tests pass on Colab A100 (image preprocessing, DataLoader, evaluate, evaluate_test).
- ✅ `build_processed.py` against real data: 564/111/162 train/val/test (matches the source `splits.json`).
- ✅ BPE-8192 tokenizer trained on training-set field values; all 26 special tokens registered.
- ✅ `data/data_card.md` auto-generated with field-presence and categorical distributions matching the source.
- ✅ `evaluate()` runs on real val (111 records) with a stub `predict_text -> ""` model, returns `macro_f1=0.0000`, `n_examples=111`. Phase 1 exit gate.

## Open items / deferrals

- Field-aware normalization (stripping "B.No.:", "M.R.P.Rs.:", parsing dates) deferred to Phase-7 sweep.
- `manufacturer` and `quantity` excluded from primary macro_f1 (excluded due to <1% presence). Per-field F1 still reported.
- 2223 unlabeled raw images sit unused. Potential MAE/BYOL pretraining of the encoder is in the Phase-7 backlog.
- Polygons available in source but not currently used by the model. Potential auxiliary localization loss is in the Phase-7 backlog.

## Compute used

A few minutes on the A100 — 1 build_processed run, 1 BPE training, 1 stub evaluate. <0.1 GPU-hour.

## Next step

**Phase 2 — Baseline Model + Training Loop.** Implement the SigLIP-base + Donut-style decoder in `train.py`, run a single full training pass, confirm `final_macro_f1` is reproducible to ±0.005 across 3 seeds.
```

- [ ] **Step 9: Tag and push**

```bash
cd /Users/apple/AutoLearnMeds
git add docs/superpowers/reviews/phase1_review.md
git commit -m "docs: phase 1 review

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
git tag -a phase-1-complete -m "Phase 1 data ingestion + schema lock complete"
git push origin phase-0-plumbing phase-1-complete
```

---

## Self-Review

**Spec coverage check** (against spec §3 Data Pipeline + §1.3 metric):

| Spec requirement | Implemented in task |
|---|---|
| §3.1 GCS as source of truth | T9 (build_processed reads /mnt/gcs) |
| §3.2 Canonical JSONL with image_hash, split | T2 |
| §3.3 80/10/10 stratified split (use provided) | T9 (build_processed honors splits.json) |
| §3.4 XML-tagged output format | T3 |
| §3.4 BPE-8192 tokenizer | T4 |
| §3.5 224×224 letterbox preprocessing | T6 |
| §3.6 `get_dataloader`, `evaluate` | T6, T7 |
| §3.7 `data/data_card.md` (Gebru 2018) | T8 |
| §1.3 macro_f1 primary metric | T5 |
| §1.4 secondary per-field F1 | T5 |
| §1.5 normalization | T5 |
| §1.6 test-set firewall | T7 (evaluate_test) |
| §5.7 metric tampering hash | (deferred — not strictly needed for Phase 1 exit; would add to Phase 2 if requested) |

**Placeholder scan:** None. Every code block is complete. The Phase 1 review template uses `YYYY-MM-DD` as the date placeholder, which is filled at commit time — that's a runtime fill, not a plan placeholder.

**Type consistency:**
- `predictions: list[dict[str, str]]` and `truths: list[dict[str, str]]` consistently typed across `compute_field_f1` and `compute_metrics`.
- `model.predict_text(batch_images, max_new_tokens) -> list[str]` interface consistent across `evaluate` and the stub fixtures.
- `format_output(record)` and `parse_output(text)` are inverse-shaped (round-trip tested).
- Path arguments accept both `Path | str` consistently in `train_tokenizer`, `get_tokenizer`, `preprocess_image`, `PharmaLabelDataset`.

**Scope check:** Phase 1 produces working, testable software on its own (tokenizer + dataloader + evaluate work end-to-end against a stub). The actual model training is Phase 2.

---

## Phase 1 → Phase 2 transition

When this plan is fully executed and tagged `phase-1-complete`:

1. `data/processed/{train,val,test}.jsonl` exist on Colab + GCS.
2. `data/processed/tokenizer.json` is in git, locked.
3. `data/data_card.md` is in git, regenerable.
4. `prepare.py` v1 is the canonical, never-modified eval module.
5. Phase 2 plan implements the SigLIP+Donut baseline in `train.py`, with `final_macro_f1` printed on stdout per spec §1.3.
