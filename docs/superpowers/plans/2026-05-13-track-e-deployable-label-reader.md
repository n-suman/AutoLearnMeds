# Track E — Deployable Label Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Track E — a high-resolution end-to-end SigLIP+Donut model trained on gold + Qwen2-VL pseudo-labels — without breaking any existing experiment, ledger row, or paper figure from Tracks A–D.

**Architecture:** Additive-only edits to `train.py` (new Config flags, new code paths gated on them) and `prepare.py` (new functions; existing ones unchanged). New config YAMLs, new tokenizer artifact, new plot scripts, new run-ids. The single existing `evaluate()` entry point keeps the same signature; new behavior is opt-in. Old configs (`baseline.yaml`, `qwen_baseline.yaml`, etc.) must reproduce their original ledger numbers byte-for-byte after these changes.

**Tech Stack:** Python 3.11, PyTorch (bf16 on A100), HuggingFace transformers + tokenizers (BPE), PyYAML, pytest, matplotlib. Single-A100 Colab + GCS for run artifacts.

**Spec:** `docs/superpowers/specs/2026-05-13-track-e-deployable-label-reader-design.md`

---

## File Structure

**Files this plan creates:**

| File | Responsibility |
|---|---|
| `experiments/configs/track_e_highres_gold_only.yaml` | Run 1: SigLIP-large-384, gold-only, 1,500 steps |
| `experiments/configs/track_e_highres.yaml` | Run 3: SigLIP-large-384 + pseudo @ weight 0.3, 3-stage |
| `experiments/tokenizers/track_e_bpe.json` | BPE trained on gold + pseudo (built by Task E1; placeholder until then) |
| `scripts/build_track_e_tokenizer.py` | Reusable tokenizer-build script for Track E and future track variants |
| `scripts/backfill_safety4_baseline.py` | One-off: compute safety-4 macro_edit_f1 for existing Track A–D from per-field JSONs |
| `scripts/plots/per_field_safety4.py` | Bar chart restricted to safety-4 fields, cross-track |
| `scripts/plots/envelope.py` | Running envelope across Track E runs |
| `scripts/plots/calibration_curve.py` | Accuracy-vs-coverage curve at entropy threshold sweep |
| `scripts/plots/error_samples.py` | 9-panel qualitative grid: still-wrong / fixed-by-E / regressed |
| `tests/test_prepare_safety_metrics.py` | Unit tests for the new safety-4 + subset metric functions |
| `tests/test_prepare_pseudo.py` | Unit tests for pseudo-label normalizer + filter |
| `tests/test_train_config_compat.py` | Regression: old Config defaults still produce old behavior |
| `tests/test_train_stage_schedule.py` | Stage-scheduler unit tests |
| `tests/test_run_experiment_track_e.py` | Integration: run_experiment.sh accepts `--track E` |

**Files this plan modifies (additive only):**

| File | What changes |
|---|---|
| `prepare.py` | Add `SAFETY_FIELDS`, `compute_subset_metrics`, `compute_safety4_metrics`, `normalize_pseudo_xml`, `filter_pseudo_rows`, optional `image_size` parameter on `PharmaLabelDataset` + `preprocess_image` + `get_dataloader`. No existing function signature changes. |
| `train.py` | Add new Config fields (encoder override, image_size, pseudo_*, stage_schedule, compute_calibration). Gate new code paths behind `if cfg.<flag>:` so old configs hit the original code. Add dual best-checkpoint save (best_edit_f1.pt + best_strict_f1.pt). Add pre-flight leak check. |
| `scripts/run_experiment.sh` | Add `E)` arm to the `--track` case statement; default config `track_e_highres.yaml`. |

**Files this plan does NOT touch:**

`experiments/ledger.jsonl`, `experiments/per_field_*.json` (existing tracks), `paper/figures/*` (existing), any existing config YAML, `scripts/finalize_experiment.sh`, `train_qwen.py`, `pretrain_mae.py`, `track_d_pipeline.py`.

---

## Phase A — Safety-4 Metric Foundation

This phase ships first because every Track E run reports against it, and we can backfill the existing tracks' safety-4 numbers without any GPU work.

### Task A1: Add `SAFETY_FIELDS` constant + `compute_subset_metrics` to `prepare.py`

**Files:**
- Modify: `prepare.py` (append after `compute_metrics` at line ~313)
- Test: `tests/test_prepare_safety_metrics.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_prepare_safety_metrics.py`:

```python
"""Tests for safety-4 subset metric + SAFETY_FIELDS constant."""
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


def test_safety_fields_constant_exists(prepare_mod) -> None:
    assert prepare_mod.SAFETY_FIELDS == frozenset(["batch_number", "mfg_date", "expiry_date", "mrp"])


def test_compute_subset_metrics_macro_strict(prepare_mod) -> None:
    """Macro F1 over a subset of fields is the mean of per-field strict F1 for those fields."""
    preds = [{"batch_number": "B1", "mfg_date": "11/2023", "mrp": "85.00", "expiry_date": "10/2025", "brand_name": "Wrong"}]
    truths = [{"batch_number": "B1", "mfg_date": "11/2023", "mrp": "85.00", "expiry_date": "10/2025", "brand_name": "Right"}]
    result = prepare_mod.compute_subset_metrics(preds, truths, fields=prepare_mod.SAFETY_FIELDS)
    assert result["macro_f1"] == pytest.approx(1.0)  # all 4 safety fields exact match
    assert result["macro_edit_f1"] == pytest.approx(1.0)
    assert set(result["per_field_f1"].keys()) == prepare_mod.SAFETY_FIELDS
    assert "brand_name" not in result["per_field_f1"]  # not in subset


def test_compute_subset_metrics_partial(prepare_mod) -> None:
    """When 2 of 4 safety fields match strictly, macro_f1 should be 0.5."""
    preds = [{"batch_number": "B1", "mfg_date": "11/2023", "mrp": "WRONG", "expiry_date": "WRONG"}]
    truths = [{"batch_number": "B1", "mfg_date": "11/2023", "mrp": "85.00", "expiry_date": "10/2025"}]
    result = prepare_mod.compute_subset_metrics(preds, truths, fields=prepare_mod.SAFETY_FIELDS)
    assert result["macro_f1"] == pytest.approx(0.5)
    assert result["macro_edit_f1"] > 0.0  # partial credit on the two wrong ones


def test_compute_safety4_metrics_convenience(prepare_mod) -> None:
    """Thin wrapper that calls compute_subset_metrics with SAFETY_FIELDS."""
    preds = [{"batch_number": "X", "mfg_date": "Y", "mrp": "Z", "expiry_date": "W"}]
    truths = [{"batch_number": "X", "mfg_date": "Y", "mrp": "Z", "expiry_date": "W"}]
    result = prepare_mod.compute_safety4_metrics(preds, truths)
    assert result["macro_f1"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prepare_safety_metrics.py -v`
Expected: 4 failures with `AttributeError: module 'prepare' has no attribute 'SAFETY_FIELDS'` (or similar).

- [ ] **Step 3: Implement in `prepare.py`**

Append after the existing `compute_metrics` function (around line 313):

```python
# === Safety-4 subset metric (Track E onward) ===

# The four fields whose errors carry production safety/financial risk:
# - batch_number: recall-tracing risk (drug-batch identification)
# - expiry_date:  dispensing-error risk (expired-drug dispensing)
# - mrp:          pricing-fraud risk
# - mfg_date:     recall-tracing risk
# Track E and later runs optimize macro_edit_f1 over this subset.
SAFETY_FIELDS: frozenset[str] = frozenset(["batch_number", "mfg_date", "expiry_date", "mrp"])


def compute_subset_metrics(
    predictions: list[dict[str, str]],
    truths: list[dict[str, str]],
    fields: frozenset[str],
) -> dict[str, Any]:
    """Compute macro F1 + macro edit F1 restricted to a subset of fields.

    Same per-field math as compute_metrics, but the macro averages are over `fields`
    instead of FIELD_ORDER. Useful for headline metrics that weight production-critical
    fields over auxiliary ones.

    Returns the same dict shape as compute_metrics — macro_f1, per_field_f1,
    macro_edit_f1, per_field_edit_f1 — but per_field_* only contain the requested fields.
    """
    assert fields.issubset(ALL_FIELDS), f"Subset contains unknown fields: {fields - ALL_FIELDS}"
    per_field = {f: compute_field_f1(predictions, truths, f) for f in fields}
    per_field_edit = {f: compute_field_edit_f1(predictions, truths, f) for f in fields}
    macro = sum(per_field.values()) / len(per_field) if per_field else 0.0
    macro_edit = sum(per_field_edit.values()) / len(per_field_edit) if per_field_edit else 0.0
    return {
        "macro_f1": macro,
        "per_field_f1": per_field,
        "macro_edit_f1": macro_edit,
        "per_field_edit_f1": per_field_edit,
    }


def compute_safety4_metrics(
    predictions: list[dict[str, str]],
    truths: list[dict[str, str]],
) -> dict[str, Any]:
    """Convenience wrapper: compute_subset_metrics(..., fields=SAFETY_FIELDS)."""
    return compute_subset_metrics(predictions, truths, SAFETY_FIELDS)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prepare_safety_metrics.py -v`
Expected: 4 PASS.

Also run the existing tests to confirm no regressions:

Run: `pytest tests/test_metric.py -v`
Expected: all existing PASS (no change in behavior of `compute_metrics`).

- [ ] **Step 5: Commit**

```bash
git add prepare.py tests/test_prepare_safety_metrics.py
git commit -m "feat(prepare): SAFETY_FIELDS + compute_subset_metrics for Track E headline metric"
```

---

### Task A2: Backfill safety-4 numbers for existing Tracks A–D

**Why:** Track E's first eval needs an anchored comparison. We compute the safety-4 macro_edit_f1 for every existing track from the per-field JSONs already in git — no retraining.

**Files:**
- Create: `scripts/backfill_safety4_baseline.py`
- Create: `experiments/safety4_baseline.json` (output of the script)
- Test: `tests/test_backfill_safety4.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_backfill_safety4.py`:

```python
"""Tests for backfill_safety4_baseline.py — recomputes safety-4 from existing per-field JSONs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_backfill_script_exists(project_root: Path) -> None:
    assert (project_root / "scripts" / "backfill_safety4_baseline.py").exists()


def test_backfill_produces_expected_keys(project_root: Path, tmp_path: Path) -> None:
    """Script should produce a JSON with one entry per existing per_field_*.json."""
    import subprocess
    out_path = tmp_path / "safety4_baseline.json"
    result = subprocess.run(
        ["python", "scripts/backfill_safety4_baseline.py", "--out", str(out_path)],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    # Should at minimum contain Track A and Track B baselines
    assert "track_a" in data
    assert "track_b" in data
    for entry in data.values():
        assert "safety4_macro_f1" in entry
        assert "safety4_macro_edit_f1" in entry
        assert "per_field" in entry
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_backfill_safety4.py -v`
Expected: FAIL — `scripts/backfill_safety4_baseline.py` doesn't exist.

- [ ] **Step 3: Implement the script**

Create `scripts/backfill_safety4_baseline.py`:

```python
"""One-off: compute safety-4 macro F1 + macro_edit_F1 for every existing per-field JSON.

Reads `experiments/per_field_*.json` (one per track / variant), restricts each to
the four SAFETY_FIELDS, and writes a consolidated baseline file. Track E uses this
as the anchor comparison.

Usage:
    python scripts/backfill_safety4_baseline.py [--out experiments/safety4_baseline.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prepare import SAFETY_FIELDS  # noqa: E402


def backfill_one(per_field_path: Path) -> dict[str, float | dict]:
    """Read a per_field_*.json (shape: {"per_field_f1": {f: x}, "per_field_edit_f1": {f: y}})
    and return the safety-4 subset macros."""
    data = json.loads(per_field_path.read_text())
    pf_strict = data.get("per_field_f1", {})
    pf_edit = data.get("per_field_edit_f1", {})

    safety_strict = {f: pf_strict.get(f, 0.0) for f in SAFETY_FIELDS}
    safety_edit = {f: pf_edit.get(f, 0.0) for f in SAFETY_FIELDS}

    return {
        "source_file": per_field_path.name,
        "safety4_macro_f1": sum(safety_strict.values()) / len(SAFETY_FIELDS),
        "safety4_macro_edit_f1": sum(safety_edit.values()) / len(SAFETY_FIELDS),
        "per_field": {
            f: {"strict": safety_strict[f], "edit": safety_edit[f]}
            for f in SAFETY_FIELDS
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-field-dir", default="experiments", help="Directory containing per_field_*.json")
    parser.add_argument("--out", default="experiments/safety4_baseline.json")
    args = parser.parse_args()

    pf_dir = Path(args.per_field_dir)
    out = Path(args.out)

    results: dict[str, dict] = {}
    for pf_file in sorted(pf_dir.glob("per_field_*.json")):
        key = pf_file.stem.replace("per_field_", "")  # e.g. "track_a"
        results[key] = backfill_one(pf_file)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"Wrote {out} with {len(results)} tracks")
    for key, entry in results.items():
        print(f"  {key:50s}  safety4 macro_edit_f1 = {entry['safety4_macro_edit_f1']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test + run the script**

Run: `pytest tests/test_backfill_safety4.py -v`
Expected: PASS.

Run: `python scripts/backfill_safety4_baseline.py`
Expected: prints one line per track; writes `experiments/safety4_baseline.json`.

- [ ] **Step 5: Commit**

```bash
git add scripts/backfill_safety4_baseline.py experiments/safety4_baseline.json tests/test_backfill_safety4.py
git commit -m "chore(eval): backfill safety-4 baseline for existing tracks"
```

---

## Phase B — Pseudo-Label Handling

### Task B1: Add `normalize_pseudo_xml` to `prepare.py`

**Why:** GPT-4o-style models emit minor format quirks (₹ prefixes, varied date separators, whitespace runs). We normalize once at load time so the trainer sees consistent strings.

**Files:**
- Modify: `prepare.py` (append after `compute_safety4_metrics` from Task A1)
- Test: `tests/test_prepare_pseudo.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_prepare_pseudo.py`:

```python
"""Tests for pseudo-label normalization + filtering (Track E)."""
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


def test_normalize_strips_inr_currency_prefix(prepare_mod) -> None:
    row = {"image_path": "x.jpg", "xml_label": "<medication><mrp>₹85.00</mrp></medication>"}
    out = prepare_mod.normalize_pseudo_xml(row)
    assert "₹" not in out["xml_label"]
    assert "85.00" in out["xml_label"]


def test_normalize_strips_rs_prefix(prepare_mod) -> None:
    row = {"image_path": "x.jpg", "xml_label": "<medication><mrp>Rs. 85.00</mrp></medication>"}
    out = prepare_mod.normalize_pseudo_xml(row)
    assert "Rs." not in out["xml_label"]
    assert "85.00" in out["xml_label"]


def test_normalize_date_separators(prepare_mod) -> None:
    row = {"image_path": "x.jpg", "xml_label": "<medication><expiry_date>10-2025</expiry_date></medication>"}
    out = prepare_mod.normalize_pseudo_xml(row)
    assert "10/2025" in out["xml_label"]


def test_normalize_collapses_whitespace(prepare_mod) -> None:
    row = {"image_path": "x.jpg", "xml_label": "<medication><brand_name>  Crocin   Advance  </brand_name></medication>"}
    out = prepare_mod.normalize_pseudo_xml(row)
    assert "Crocin Advance" in out["xml_label"]


def test_normalize_preserves_other_fields(prepare_mod) -> None:
    row = {"image_path": "x.jpg", "xml_label": "<medication><batch_number>BC4521A</batch_number></medication>", "extra": "keep"}
    out = prepare_mod.normalize_pseudo_xml(row)
    assert out["image_path"] == "x.jpg"
    assert out["extra"] == "keep"
    assert "BC4521A" in out["xml_label"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_prepare_pseudo.py -v`
Expected: 5 failures with `AttributeError: module 'prepare' has no attribute 'normalize_pseudo_xml'`.

- [ ] **Step 3: Implement in `prepare.py`**

Append:

```python
# === Pseudo-label handling (Track E onward) ===

_INR_PREFIX_RE = re.compile(r"(?:₹|Rs\.?\s*)")
_DATE_SEP_RE = re.compile(r"(\d{1,2})[-.](\d{4})")  # MM-YYYY or MM.YYYY → MM/YYYY
_WS_RUN_RE = re.compile(r"\s+")


def normalize_pseudo_xml(row: dict) -> dict:
    """Apply minor format normalizations to a pseudo-label row before training.

    Operations (all string-replace, no structural changes):
    - Strip Indian rupee prefixes (₹, Rs., Rs) from mrp values.
    - Normalize date separators MM-YYYY / MM.YYYY → MM/YYYY.
    - Collapse runs of whitespace inside any tag to a single space.

    Returns a new dict (original is not mutated). All keys other than `xml_label`
    are passed through unchanged.
    """
    xml = row.get("xml_label", "")
    xml = _INR_PREFIX_RE.sub("", xml)
    xml = _DATE_SEP_RE.sub(r"\1/\2", xml)
    # Collapse whitespace inside tag bodies (between > and <)
    def _collapse_inside_tags(match: re.Match) -> str:
        return _WS_RUN_RE.sub(" ", match.group(0)).strip()
    # Match content between tags: >...<
    xml = re.sub(r">([^<]*)<", lambda m: ">" + _WS_RUN_RE.sub(" ", m.group(1)).strip() + "<", xml)

    return {**row, "xml_label": xml}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prepare_pseudo.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add prepare.py tests/test_prepare_pseudo.py
git commit -m "feat(prepare): normalize_pseudo_xml — strip ₹/Rs prefixes, date separators, ws"
```

---

### Task B2: Add `filter_pseudo_rows` to `prepare.py`

**Why:** Drop bad rows before they reach the trainer: leak rows (image in val/test), all-empty rows (non-pharma images), and parse-failure rows.

**Files:**
- Modify: `prepare.py` (append after `normalize_pseudo_xml`)
- Test: `tests/test_prepare_pseudo.py` (extend)

- [ ] **Step 1: Append tests to `tests/test_prepare_pseudo.py`**

```python
def test_filter_drops_val_leak(prepare_mod) -> None:
    rows = [
        {"image_path": "raw_images/IMG_001.JPG", "xml_label": "<medication><brand_name>X</brand_name></medication>"},
        {"image_path": "raw_images/IMG_002.JPG", "xml_label": "<medication><brand_name>Y</brand_name></medication>"},
    ]
    val_paths = {"raw_images/IMG_001.JPG"}
    test_paths: set[str] = set()
    kept, dropped = prepare_mod.filter_pseudo_rows(rows, val_paths=val_paths, test_paths=test_paths)
    assert len(kept) == 1
    assert kept[0]["image_path"] == "raw_images/IMG_002.JPG"
    assert dropped["val_leak"] == 1


def test_filter_drops_all_empty(prepare_mod) -> None:
    """Rows with >=8 of 12 empty fields look like non-pharma images and are dropped."""
    # All 12 fields empty
    xml_empty = "<medication>" + "".join(f"<{f}></{f}>" for f in prepare_mod.FIELD_ORDER) + "</medication>"
    rows = [
        {"image_path": "raw_images/IMG_003.JPG", "xml_label": xml_empty},
        {"image_path": "raw_images/IMG_004.JPG", "xml_label": "<medication><brand_name>Real</brand_name><drug_name>Real</drug_name><generic_name>R</generic_name><strength>5mg</strength><quantity>10</quantity><company>X</company><manufacturer>X</manufacturer><batch_number>B</batch_number><mfg_date>11/2023</mfg_date><expiry_date>10/2025</expiry_date><mrp>85</mrp><warnings>w</warnings></medication>"},
    ]
    kept, dropped = prepare_mod.filter_pseudo_rows(rows, val_paths=set(), test_paths=set(), max_empty_fields=8)
    assert len(kept) == 1
    assert kept[0]["image_path"] == "raw_images/IMG_004.JPG"
    assert dropped["all_empty"] == 1


def test_filter_drops_parse_failures(prepare_mod) -> None:
    rows = [
        {"image_path": "raw_images/IMG_005.JPG", "xml_label": "not valid xml at all"},
        {"image_path": "raw_images/IMG_006.JPG", "xml_label": "<medication><brand_name>X</brand_name></medication>"},
    ]
    kept, dropped = prepare_mod.filter_pseudo_rows(rows, val_paths=set(), test_paths=set())
    assert len(kept) == 1
    assert kept[0]["image_path"] == "raw_images/IMG_006.JPG"
    assert dropped["parse_fail"] == 1


def test_filter_returns_dropped_counts(prepare_mod) -> None:
    kept, dropped = prepare_mod.filter_pseudo_rows([], val_paths=set(), test_paths=set())
    assert kept == []
    assert dropped == {"val_leak": 0, "test_leak": 0, "all_empty": 0, "parse_fail": 0}
```

- [ ] **Step 2: Run to verify failures**

Run: `pytest tests/test_prepare_pseudo.py::test_filter_drops_val_leak tests/test_prepare_pseudo.py::test_filter_drops_all_empty tests/test_prepare_pseudo.py::test_filter_drops_parse_failures tests/test_prepare_pseudo.py::test_filter_returns_dropped_counts -v`
Expected: 4 FAIL with `AttributeError: module 'prepare' has no attribute 'filter_pseudo_rows'`.

- [ ] **Step 3: Implement in `prepare.py`**

Append:

```python
def filter_pseudo_rows(
    rows: list[dict],
    val_paths: set[str],
    test_paths: set[str],
    max_empty_fields: int = 8,
) -> tuple[list[dict], dict[str, int]]:
    """Filter pseudo-label rows. Returns (kept_rows, dropped_counts).

    Drop reasons:
    - val_leak / test_leak: row's image_path appears in val or test split.
    - all_empty: row's parsed XML has >= max_empty_fields fields with empty value.
    - parse_fail: row's xml_label fails parse_output().

    The dropped_counts dict has keys val_leak, test_leak, all_empty, parse_fail (all ints).
    """
    kept: list[dict] = []
    dropped = {"val_leak": 0, "test_leak": 0, "all_empty": 0, "parse_fail": 0}
    for row in rows:
        path = row.get("image_path", "")
        if path in val_paths:
            dropped["val_leak"] += 1
            continue
        if path in test_paths:
            dropped["test_leak"] += 1
            continue
        try:
            parsed = parse_output(row.get("xml_label", ""))
        except Exception:
            dropped["parse_fail"] += 1
            continue
        if not parsed:  # parse_output returns empty dict on failure
            dropped["parse_fail"] += 1
            continue
        empty_count = sum(1 for f in FIELD_ORDER if not parsed.get(f, "").strip())
        if empty_count >= max_empty_fields:
            dropped["all_empty"] += 1
            continue
        kept.append(row)
    return kept, dropped
```

- [ ] **Step 4: Run tests + regression**

Run: `pytest tests/test_prepare_pseudo.py -v`
Expected: all PASS.

Run: `pytest tests/test_metric.py tests/test_evaluate.py -v`
Expected: existing tests still PASS (no behavior change to `compute_metrics` or `evaluate`).

- [ ] **Step 5: Commit**

```bash
git add prepare.py tests/test_prepare_pseudo.py
git commit -m "feat(prepare): filter_pseudo_rows — leak + all-empty + parse-fail drops"
```

---

## Phase C — Trainer Config Flags

### Task C1: Extend `Config` dataclass with Track E flags

**Why:** All new behavior is gated on new fields with backwards-compatible defaults. Old YAMLs hit no-op paths.

**Files:**
- Modify: `train.py` (the `Config` dataclass near top of file)
- Test: `tests/test_train_config_compat.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_config_compat.py`:

```python
"""Regression: Track A's baseline.yaml must still load + produce a Config with no new behavior enabled."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def train_mod(project_root: Path):
    sys.path.insert(0, str(project_root))
    if "train" in sys.modules:
        importlib.reload(sys.modules["train"])
    import train
    return train


def test_baseline_yaml_loads_into_config(project_root: Path, train_mod) -> None:
    cfg_path = project_root / "experiments" / "configs" / "baseline.yaml"
    raw = yaml.safe_load(cfg_path.read_text())
    cfg = train_mod.Config(**raw)
    assert cfg.encoder_model == "google/siglip-base-patch16-224"
    assert cfg.max_steps == 1000


def test_baseline_does_not_enable_new_features(project_root: Path, train_mod) -> None:
    """All Track E features must default to off when not specified in YAML."""
    cfg_path = project_root / "experiments" / "configs" / "baseline.yaml"
    raw = yaml.safe_load(cfg_path.read_text())
    cfg = train_mod.Config(**raw)
    # New flags should default to off (None / empty / 0.0)
    assert cfg.pseudo_jsonl == ""
    assert cfg.pseudo_weight == 0.0
    assert cfg.pseudo_weight_mode == "const"
    assert cfg.pseudo_min_confidence == "medium"
    assert cfg.image_size == 224
    assert cfg.stage_schedule == []
    assert cfg.compute_calibration is False


def test_track_e_yaml_overrides_defaults(project_root: Path, train_mod, tmp_path: Path) -> None:
    """A Track E-style config should set the new flags to their non-default values."""
    track_e_raw = {
        "seed": 44,
        "encoder_model": "google/siglip-large-patch16-384",
        "image_size": 384,
        "pseudo_jsonl": "data/pseudo_labels/round_001.jsonl",
        "pseudo_weight": 0.3,
        "compute_calibration": True,
        "max_steps": 3000,
    }
    cfg = train_mod.Config(**track_e_raw)
    assert cfg.image_size == 384
    assert cfg.pseudo_weight == 0.3
    assert cfg.compute_calibration is True
```

- [ ] **Step 2: Run to verify failures**

Run: `pytest tests/test_train_config_compat.py -v`
Expected: 2 FAIL (`test_baseline_does_not_enable_new_features`, `test_track_e_yaml_overrides_defaults`) with `TypeError: unexpected keyword argument` or `AttributeError`.

- [ ] **Step 3: Implement — append new fields to Config**

Edit `train.py`. Find the `Config` dataclass (around line 16). Add new fields at the END of the existing fields (preserve order so old fields aren't reordered):

```python
@dataclasses.dataclass
class Config:
    # ... [all existing fields preserved unchanged] ...

    # --- Track E flags (defaults preserve Track A behavior) ---
    image_size: int = 224                          # 384 for Track E
    pseudo_jsonl: str = ""                         # path to data/pseudo_labels/round_001.jsonl
    pseudo_weight: float = 0.0                     # 0.0 = ignore pseudo even if jsonl given
    pseudo_weight_mode: str = "const"              # "const" | "adaptive_confidence"
    pseudo_min_confidence: str = "medium"          # "low" | "medium" | "high"
    track_e_tokenizer_path: str = ""               # if set, use this instead of cfg.tokenizer_path
    stage_schedule: list = dataclasses.field(default_factory=list)
        # e.g. [{"name": "stage1", "steps": 500, "data": "pseudo", "lr_mult": 1.0}, ...]
        # empty list = single-stage default (use existing max_steps)
    compute_calibration: bool = False              # if True, eval also emits per-field entropy
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_train_config_compat.py -v`
Expected: all PASS.

Run any existing trainer-component test as regression:

Run: `pytest tests/test_train_components.py -v`
Expected: all PASS (no signature change to existing functions).

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_train_config_compat.py
git commit -m "feat(train): Config flags for Track E (image_size, pseudo_*, stage_schedule, compute_calibration)"
```

---

### Task C2: Add `image_size` parameter to `PharmaLabelDataset` and `preprocess_image`

**Why:** Track E uses 384×384 input; existing code is hardcoded to 224 via `prepare.IMAGE_SIZE`. We add an optional parameter with default = current behavior.

**Files:**
- Modify: `prepare.py` (functions `preprocess_image`, `PharmaLabelDataset.__init__`, `get_dataloader`)
- Test: `tests/test_prepare_image_size.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_prepare_image_size.py`:

```python
"""Tests for image_size parameter on PharmaLabelDataset + preprocess_image."""
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


def test_preprocess_image_default_size(prepare_mod, synthetic_images_dir: Path) -> None:
    img_path = next(synthetic_images_dir.glob("*.jpg"))
    arr = prepare_mod.preprocess_image(img_path)
    assert arr.shape[-1] == 224 and arr.shape[-2] == 224


def test_preprocess_image_custom_size(prepare_mod, synthetic_images_dir: Path) -> None:
    img_path = next(synthetic_images_dir.glob("*.jpg"))
    arr = prepare_mod.preprocess_image(img_path, image_size=384)
    assert arr.shape[-1] == 384 and arr.shape[-2] == 384


def test_dataset_default_size_is_backwards_compatible(prepare_mod, synthetic_jsonl_dir: Path, synthetic_images_dir: Path) -> None:
    """Old code path (no image_size kwarg) still returns 224×224 tensors."""
    ds = prepare_mod.PharmaLabelDataset(
        jsonl_path=synthetic_jsonl_dir / "gold_standard.jsonl",
        images_root=synthetic_images_dir,
        tokenizer=None,
        path_strip_prefix="raw/raw_images/",
    )
    item = ds[0]
    # The exact tensor shape depends on the dataset's internal preprocess —
    # we just verify the spatial dimensions match the default.
    assert item["pixel_values"].shape[-1] == 224
```

- [ ] **Step 2: Run to verify failures**

Run: `pytest tests/test_prepare_image_size.py -v`
Expected: `test_preprocess_image_custom_size` FAILs (unexpected keyword `image_size`).

- [ ] **Step 3: Implement**

Edit `prepare.py`. Find `preprocess_image` (around line 331). Change signature:

```python
def preprocess_image(image_path: Path | str, image_size: int | None = None):
    """Load an image and return a normalized tensor at the requested resolution.

    Args:
        image_path: absolute or repo-relative path to a JPEG/PNG.
        image_size: target side length (square crop). None → uses the module-level
            IMAGE_SIZE constant (224 — Track A's default). Track E passes 384.
    """
    from PIL import Image
    import torch
    from torchvision import transforms

    size = image_size if image_size is not None else IMAGE_SIZE
    img = Image.open(image_path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # SigLIP normalization
    ])
    return tfm(img)
```

**Note for the engineer:** if the existing `preprocess_image` body differs from the sketch above (e.g. uses a HuggingFace processor), preserve the existing logic and just thread the `image_size` parameter through whatever resize/crop step it does. The acceptance check is `arr.shape[-1] == image_size`.

Edit `PharmaLabelDataset.__init__` (around line 354). Add `image_size` parameter:

```python
class PharmaLabelDataset:
    def __init__(
        self,
        jsonl_path: Path | str,
        images_root: Path | str,
        tokenizer,
        path_strip_prefix: str = "",
        image_size: int | None = None,
    ):
        # ... existing init body ...
        self.image_size = image_size  # None → use IMAGE_SIZE default
```

Find where the dataset calls `preprocess_image` (likely in `__getitem__`). Thread the parameter:

```python
def __getitem__(self, idx):
    # ... existing logic ...
    pixel_values = preprocess_image(image_path, image_size=self.image_size)
    # ... rest of method ...
```

Edit `get_dataloader` (around line 408). Add `image_size` parameter that defaults to None and passes it to the Dataset.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_prepare_image_size.py -v tests/test_evaluate.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add prepare.py tests/test_prepare_image_size.py
git commit -m "feat(prepare): optional image_size on dataset + preprocess (224 default; 384 for Track E)"
```

---

## Phase D — Trainer Logic

### Task D1: Wire `cfg.encoder_model` and `cfg.image_size` through trainer setup

**Why:** Currently the trainer hardcodes the SigLIP-base encoder. Make it config-driven so `track_e_highres_gold_only.yaml` can swap to SigLIP-large-384.

**Files:**
- Modify: `train.py` (encoder-loading section)
- Test: `tests/test_train_components.py` (extend)

- [ ] **Step 1: Append to `tests/test_train_components.py`**

```python
def test_encoder_loader_picks_up_cfg_model(project_root: Path) -> None:
    """If cfg.encoder_model is set, the trainer should construct the encoder from that ID."""
    import sys
    sys.path.insert(0, str(project_root))
    import train
    cfg = train.Config(encoder_model="google/siglip-large-patch16-384", image_size=384)
    # The encoder-builder helper (refactored out of main) should be importable.
    enc, hidden_dim = train.build_encoder(cfg)
    # SigLIP-large hidden dim is 1024; SigLIP-base is 768
    assert hidden_dim == 1024
```

(If `train.build_encoder` doesn't exist yet, this test will fail — that's the point.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_train_components.py::test_encoder_loader_picks_up_cfg_model -v`
Expected: FAIL — either `AttributeError: build_encoder` or hardcoded 768.

- [ ] **Step 3: Refactor encoder construction into `build_encoder(cfg)`**

In `train.py`, locate the encoder-loading code (look for `SiglipVisionModel.from_pretrained` or similar). Extract into:

```python
def build_encoder(cfg: Config):
    """Construct the (frozen) vision encoder per cfg. Returns (encoder, hidden_dim)."""
    from transformers import SiglipVisionModel
    encoder = SiglipVisionModel.from_pretrained(cfg.encoder_model)
    # Optional: load from local path if cfg.encoder_init_path is set (existing Track C behavior)
    if cfg.encoder_init_path:
        # ... existing load-from-path logic preserved ...
        pass
    for p in encoder.parameters():
        p.requires_grad = False
    hidden_dim = encoder.config.hidden_size
    return encoder, hidden_dim
```

Replace the inline encoder-loading code in the main function with a call to `build_encoder(cfg)`. Pass `hidden_dim` to the decoder's cross-attention bridge construction.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_train_components.py -v`
Expected: all PASS.

Run the trainer in dry-run mode if it has one, or run a small smoke test:

Run: `python -c "import train; cfg = train.Config(); enc, h = train.build_encoder(cfg); print('OK', h)"`
Expected: prints `OK 768` (base) for default cfg.

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_train_components.py
git commit -m "refactor(train): extract build_encoder(cfg) — picks up encoder_model + size"
```

---

### Task D2: Add pre-flight leak check at trainer startup

**Why:** Belt-and-suspenders: if pseudo_jsonl contains a val/test image, refuse to train.

**Files:**
- Modify: `train.py` (top of main loop)
- Test: extend `tests/test_train_config_compat.py`

- [ ] **Step 1: Append to `tests/test_train_config_compat.py`**

```python
def test_preflight_rejects_pseudo_with_val_leak(project_root: Path, tmp_path: Path, train_mod) -> None:
    """If pseudo_jsonl contains an image_path from val, preflight_leak_check should raise."""
    val_jsonl = tmp_path / "val.jsonl"
    pseudo_jsonl = tmp_path / "pseudo.jsonl"
    val_jsonl.write_text('{"image_path": "raw/raw_images/IMG_001.JPG", "xml_label": "<medication><brand_name>X</brand_name></medication>"}\n')
    pseudo_jsonl.write_text('{"image_path": "raw/raw_images/IMG_001.JPG", "xml_label": "<medication><brand_name>X</brand_name></medication>"}\n')

    cfg = train_mod.Config(val_jsonl=str(val_jsonl), pseudo_jsonl=str(pseudo_jsonl), pseudo_weight=0.3)
    with pytest.raises(AssertionError, match="leak"):
        train_mod.preflight_leak_check(cfg)


def test_preflight_passes_when_no_overlap(project_root: Path, tmp_path: Path, train_mod) -> None:
    val_jsonl = tmp_path / "val.jsonl"
    pseudo_jsonl = tmp_path / "pseudo.jsonl"
    val_jsonl.write_text('{"image_path": "raw/raw_images/IMG_001.JPG", "xml_label": "x"}\n')
    pseudo_jsonl.write_text('{"image_path": "raw/raw_images/IMG_999.JPG", "xml_label": "x"}\n')
    cfg = train_mod.Config(val_jsonl=str(val_jsonl), pseudo_jsonl=str(pseudo_jsonl), pseudo_weight=0.3)
    train_mod.preflight_leak_check(cfg)  # should not raise


def test_preflight_noop_when_pseudo_disabled(project_root: Path, train_mod) -> None:
    """If pseudo_weight=0.0 or pseudo_jsonl='', preflight is a no-op."""
    cfg = train_mod.Config()
    train_mod.preflight_leak_check(cfg)  # should not raise (no pseudo configured)
```

- [ ] **Step 2: Run to verify failures**

Run: `pytest tests/test_train_config_compat.py::test_preflight_rejects_pseudo_with_val_leak tests/test_train_config_compat.py::test_preflight_passes_when_no_overlap tests/test_train_config_compat.py::test_preflight_noop_when_pseudo_disabled -v`
Expected: 3 FAIL with `AttributeError: module 'train' has no attribute 'preflight_leak_check'`.

- [ ] **Step 3: Implement in `train.py`**

Add near the top of `train.py` (after `build_encoder`):

```python
def preflight_leak_check(cfg: Config) -> None:
    """Refuse to train if pseudo_jsonl contains any image_path also in val or test.

    No-op when pseudo is disabled (cfg.pseudo_jsonl == "" or cfg.pseudo_weight == 0.0).
    """
    if not cfg.pseudo_jsonl or cfg.pseudo_weight == 0.0:
        return

    def _paths_from_jsonl(path: str) -> set[str]:
        out: set[str] = set()
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                out.add(row.get("image_path", ""))
        return out

    val_paths = _paths_from_jsonl(cfg.val_jsonl) if cfg.val_jsonl else set()
    pseudo_paths = _paths_from_jsonl(cfg.pseudo_jsonl)

    overlap = val_paths & pseudo_paths
    assert not overlap, f"PSEUDO LEAK: {len(overlap)} val images in pseudo_jsonl: {sorted(overlap)[:3]}..."
    # Test-set leak check is gated separately — test_jsonl path is "data/processed/test.jsonl" by convention
    test_path = cfg.val_jsonl.replace("val.jsonl", "test.jsonl")
    if Path(test_path).exists():
        test_paths = _paths_from_jsonl(test_path)
        test_overlap = test_paths & pseudo_paths
        assert not test_overlap, f"PSEUDO LEAK (TEST): {len(test_overlap)} test images in pseudo_jsonl"
```

Add `import json` and `from pathlib import Path` at the top of `train.py` if not already present.

Call it from the main trainer function near startup, before any dataset construction:

```python
def main(cfg: Config):
    # ... existing seed-setting ...
    preflight_leak_check(cfg)
    # ... rest of training ...
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_train_config_compat.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_train_config_compat.py
git commit -m "feat(train): preflight_leak_check — refuses pseudo/val image overlap"
```

---

### Task D3: Pseudo-label dataset loading with sample weights

**Why:** When Track E config sets `pseudo_jsonl` + `pseudo_weight > 0`, the trainer must mix gold and pseudo rows with the configured weights.

**Files:**
- Modify: `train.py` (dataset construction in main loop)
- Test: `tests/test_train_pseudo_loading.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_pseudo_loading.py`:

```python
"""Tests for pseudo-label dataset construction in train.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def train_mod(project_root: Path):
    import importlib, sys
    sys.path.insert(0, str(project_root))
    if "train" in sys.modules:
        importlib.reload(sys.modules["train"])
    import train
    return train


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_build_combined_dataset_gold_only_when_no_pseudo(train_mod, tmp_path: Path) -> None:
    """If cfg.pseudo_jsonl is empty, combined dataset = gold only."""
    gold = tmp_path / "gold.jsonl"
    _write_jsonl(gold, [{"image_path": "x.jpg", "xml_label": "<medication><brand_name>A</brand_name></medication>"}])
    cfg = train_mod.Config(train_jsonl=str(gold), pseudo_jsonl="", pseudo_weight=0.0)
    rows, weights = train_mod.build_combined_train_rows(cfg)
    assert len(rows) == 1
    assert weights == [1.0]


def test_build_combined_mixes_with_weights(train_mod, tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    pseudo = tmp_path / "pseudo.jsonl"
    _write_jsonl(gold, [{"image_path": "g.jpg", "xml_label": "<medication><brand_name>G</brand_name></medication>"}])
    _write_jsonl(pseudo, [{"image_path": "p.jpg", "xml_label": "<medication><brand_name>P</brand_name></medication>"}])
    cfg = train_mod.Config(train_jsonl=str(gold), pseudo_jsonl=str(pseudo), pseudo_weight=0.3)
    rows, weights = train_mod.build_combined_train_rows(cfg)
    assert len(rows) == 2
    assert weights == [1.0, 0.3]


def test_build_combined_drops_filtered_pseudo(train_mod, tmp_path: Path) -> None:
    """Pseudo rows failing filter_pseudo_rows (e.g. all-empty) should not appear."""
    gold = tmp_path / "gold.jsonl"
    pseudo = tmp_path / "pseudo.jsonl"
    _write_jsonl(gold, [{"image_path": "g.jpg", "xml_label": "<medication><brand_name>G</brand_name></medication>"}])
    # All-empty pseudo row (8+ empty fields)
    empty_xml = "<medication><brand_name></brand_name><drug_name></drug_name><generic_name></generic_name><strength></strength><quantity></quantity><company></company><manufacturer></manufacturer><batch_number></batch_number><mfg_date></mfg_date><expiry_date></expiry_date><mrp></mrp><warnings></warnings></medication>"
    _write_jsonl(pseudo, [{"image_path": "p.jpg", "xml_label": empty_xml}])
    cfg = train_mod.Config(train_jsonl=str(gold), pseudo_jsonl=str(pseudo), pseudo_weight=0.3)
    rows, weights = train_mod.build_combined_train_rows(cfg)
    assert len(rows) == 1
    assert rows[0]["image_path"] == "g.jpg"
```

- [ ] **Step 2: Run to verify failures**

Run: `pytest tests/test_train_pseudo_loading.py -v`
Expected: 3 FAIL with `AttributeError: build_combined_train_rows`.

- [ ] **Step 3: Implement in `train.py`**

Add (after `preflight_leak_check`):

```python
def build_combined_train_rows(cfg: Config) -> tuple[list[dict], list[float]]:
    """Return (rows, per_row_weights) for the training loop.

    - Always includes all rows from cfg.train_jsonl with weight 1.0.
    - If cfg.pseudo_jsonl is non-empty AND cfg.pseudo_weight > 0:
      reads pseudo, runs prepare.normalize_pseudo_xml + filter_pseudo_rows
      against val/test, appends survivors with the configured weight.
    """
    import prepare

    def _read(path: str) -> list[dict]:
        rows: list[dict] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    gold = _read(cfg.train_jsonl)
    rows = list(gold)
    weights = [1.0] * len(gold)

    if cfg.pseudo_jsonl and cfg.pseudo_weight > 0.0:
        pseudo_raw = _read(cfg.pseudo_jsonl)
        pseudo = [prepare.normalize_pseudo_xml(r) for r in pseudo_raw]
        val_paths = {r["image_path"] for r in _read(cfg.val_jsonl)} if cfg.val_jsonl else set()
        test_path = cfg.val_jsonl.replace("val.jsonl", "test.jsonl") if cfg.val_jsonl else ""
        test_paths = {r["image_path"] for r in _read(test_path)} if Path(test_path).exists() else set()
        kept, dropped = prepare.filter_pseudo_rows(pseudo, val_paths=val_paths, test_paths=test_paths)
        print(f"[train] pseudo: kept={len(kept)} dropped={dropped}", flush=True)
        rows.extend(kept)

        if cfg.pseudo_weight_mode == "adaptive_confidence":
            # weight = avg per-field confidence * cfg.pseudo_weight
            conf_map = {"low": 0.3, "medium": 0.7, "high": 1.0}
            for row in kept:
                conf = row.get("per_field_confidence", {})
                avg = sum(conf_map.get(v, 0.5) for v in conf.values()) / max(1, len(conf))
                weights.append(cfg.pseudo_weight * avg)
        else:  # const
            weights.extend([cfg.pseudo_weight] * len(kept))

    return rows, weights
```

The trainer's main loop must use `weights` when computing per-batch loss (multiply each example's per-token cross-entropy by its weight before averaging). The exact integration point depends on the existing loss-computation code; the engineer threads `weights` through `get_dataloader` into the collate function and applies them in the loss reduction.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_train_pseudo_loading.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_train_pseudo_loading.py
git commit -m "feat(train): build_combined_train_rows — gold + filtered pseudo with sample weights"
```

---

### Task D4: Stage scheduler

**Why:** Track E's 3-stage schedule (pseudo warmup → joint → gold finetune) is the central training innovation. Implement as a clean state-machine that gates which examples + LR multiplier apply at each step.

**Files:**
- Modify: `train.py` (main loop)
- Test: `tests/test_train_stage_schedule.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_stage_schedule.py`:

```python
"""Tests for the stage scheduler."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def train_mod(project_root: Path):
    import importlib, sys
    sys.path.insert(0, str(project_root))
    if "train" in sys.modules:
        importlib.reload(sys.modules["train"])
    import train
    return train


def test_empty_schedule_returns_default_stage(train_mod) -> None:
    """No schedule = single 'baseline' stage with all data, lr_mult=1.0, for cfg.max_steps."""
    cfg = train_mod.Config(max_steps=1000, stage_schedule=[])
    sched = train_mod.StageScheduler(cfg)
    assert sched.stage_at_step(0).name == "baseline"
    assert sched.stage_at_step(999).name == "baseline"
    assert sched.stage_at_step(0).lr_mult == 1.0
    assert sched.stage_at_step(0).data == "all"


def test_three_stage_schedule_transitions(train_mod) -> None:
    """500 + 2000 + 500 = 3000 step schedule transitions cleanly at boundaries."""
    schedule = [
        {"name": "warmup", "steps": 500, "data": "pseudo", "lr_mult": 1.0},
        {"name": "joint", "steps": 2000, "data": "all", "lr_mult": 1.0},
        {"name": "finetune", "steps": 500, "data": "gold", "lr_mult": 0.1},
    ]
    cfg = train_mod.Config(max_steps=3000, stage_schedule=schedule)
    sched = train_mod.StageScheduler(cfg)

    assert sched.stage_at_step(0).name == "warmup"
    assert sched.stage_at_step(499).name == "warmup"
    assert sched.stage_at_step(500).name == "joint"
    assert sched.stage_at_step(2499).name == "joint"
    assert sched.stage_at_step(2500).name == "finetune"
    assert sched.stage_at_step(2999).name == "finetune"

    assert sched.stage_at_step(0).data == "pseudo"
    assert sched.stage_at_step(2500).lr_mult == pytest.approx(0.1)


def test_filter_examples_by_stage_data(train_mod) -> None:
    """Stage data='gold' should mask out pseudo examples in the batch."""
    rows = [{"image_path": "g.jpg", "is_pseudo": False}, {"image_path": "p.jpg", "is_pseudo": True}]
    sched = train_mod.StageScheduler(train_mod.Config(stage_schedule=[{"name": "s1", "steps": 100, "data": "gold", "lr_mult": 1.0}]))
    mask = sched.example_mask_at_step(0, rows)
    assert mask == [True, False]
```

- [ ] **Step 2: Run to verify failures**

Run: `pytest tests/test_train_stage_schedule.py -v`
Expected: 3 FAIL with `AttributeError: module 'train' has no attribute 'StageScheduler'`.

- [ ] **Step 3: Implement in `train.py`**

```python
@dataclasses.dataclass(frozen=True)
class Stage:
    name: str
    steps: int
    data: str       # "all" | "gold" | "pseudo"
    lr_mult: float


class StageScheduler:
    """Maps a global step → (current stage, lr_mult, data filter)."""

    def __init__(self, cfg: Config):
        if not cfg.stage_schedule:
            self._stages = [Stage(name="baseline", steps=cfg.max_steps, data="all", lr_mult=1.0)]
        else:
            self._stages = [Stage(**s) for s in cfg.stage_schedule]
        # Precompute cumulative step boundaries
        self._cum: list[int] = []
        acc = 0
        for s in self._stages:
            acc += s.steps
            self._cum.append(acc)

    def stage_at_step(self, step: int) -> Stage:
        for i, boundary in enumerate(self._cum):
            if step < boundary:
                return self._stages[i]
        # past last stage: return the last (allows for max_steps overshoot tolerance)
        return self._stages[-1]

    def total_steps(self) -> int:
        return self._cum[-1]

    def example_mask_at_step(self, step: int, rows: list[dict]) -> list[bool]:
        """Return a bool list parallel to `rows`: True = keep this example in this stage."""
        s = self.stage_at_step(step)
        if s.data == "all":
            return [True] * len(rows)
        if s.data == "gold":
            return [not r.get("is_pseudo", False) for r in rows]
        if s.data == "pseudo":
            return [r.get("is_pseudo", False) for r in rows]
        raise ValueError(f"Unknown stage data: {s.data}")
```

The trainer's main loop now does (sketched):

```python
sched = StageScheduler(cfg)
for step in range(sched.total_steps()):
    stage = sched.stage_at_step(step)
    # Adjust LR
    current_lr = base_lr_schedule(step) * stage.lr_mult
    # Filter batch
    batch_rows = sampler.next_batch()
    mask = sched.example_mask_at_step(step, batch_rows)
    batch_rows = [r for r, keep in zip(batch_rows, mask) if keep]
    # If batch is empty for this stage, skip (or resample)
    if not batch_rows:
        continue
    # ... rest of forward/backward ...
```

The integration is the engineer's call — clean way is to filter at the sampler / dataloader level using a `WeightedRandomSampler` whose weights become 0 for filtered-out examples in the current stage. The unit tests pin the scheduler's API.

`build_combined_train_rows` should also annotate rows with `is_pseudo: bool` so the scheduler's mask works:

```python
# In build_combined_train_rows:
for r in gold:
    r["is_pseudo"] = False
for r in kept:
    r["is_pseudo"] = True
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_train_stage_schedule.py tests/test_train_pseudo_loading.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_train_stage_schedule.py
git commit -m "feat(train): StageScheduler for Track E 3-stage schedule"
```

---

### Task D5: Dual best-checkpoint save (`best_edit_f1.pt` + `best_strict_f1.pt`)

**Why:** Track B's lesson — the two metrics can diverge. Saving both lets the user pick whichever is better for deployment.

**Files:**
- Modify: `train.py` (eval-and-save loop)
- Test: extend `tests/test_train_components.py`

- [ ] **Step 1: Append to `tests/test_train_components.py`**

```python
def test_dual_best_checkpoint_tracker(project_root: Path) -> None:
    """The BestCheckpointTracker tracks two metrics independently and decides when to save."""
    import sys
    sys.path.insert(0, str(project_root))
    import train

    t = train.BestCheckpointTracker()
    assert t.is_new_best("edit_f1", 0.20)  # first measurement → new best
    assert t.is_new_best("strict_f1", 0.05)
    assert not t.is_new_best("edit_f1", 0.15)  # regression
    assert t.is_new_best("edit_f1", 0.25)  # improvement
    assert t.best_value("edit_f1") == 0.25
    assert t.best_value("strict_f1") == 0.05
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_train_components.py::test_dual_best_checkpoint_tracker -v`
Expected: FAIL with `AttributeError: BestCheckpointTracker`.

- [ ] **Step 3: Implement**

In `train.py`:

```python
class BestCheckpointTracker:
    """Per-metric best tracker. Independent tracking lets dual checkpoints diverge."""

    def __init__(self):
        self._best: dict[str, float] = {}

    def is_new_best(self, metric: str, value: float) -> bool:
        prev = self._best.get(metric, -float("inf"))
        if value > prev:
            self._best[metric] = value
            return True
        return False

    def best_value(self, metric: str) -> float:
        return self._best.get(metric, -float("inf"))
```

In the trainer's eval loop, replace the existing single-best-save logic with:

```python
tracker = BestCheckpointTracker()  # at start of training

# In eval block:
metrics = prepare.evaluate(model, cfg.val_jsonl, ...)
safety = prepare.compute_safety4_metrics(metrics["predictions"], metrics["truths"])  # see below
if tracker.is_new_best("edit_f1", safety["macro_edit_f1"]):
    save_checkpoint(model, Path(cfg.checkpoint_dir) / "best_edit_f1.pt")
if tracker.is_new_best("strict_f1", safety["macro_f1"]):
    save_checkpoint(model, Path(cfg.checkpoint_dir) / "best_strict_f1.pt")
```

If `prepare.evaluate` doesn't currently return `predictions` and `truths` in addition to metrics, this requires `evaluate` to also yield those arrays. Engineer's call: either modify `evaluate` to optionally return them (additive `return_predictions=False` kwarg), or recompute safety inside the loop. Lean toward the kwarg approach — keep the metric calc inside `prepare`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_train_components.py::test_dual_best_checkpoint_tracker -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_train_components.py
git commit -m "feat(train): BestCheckpointTracker for dual best (edit_f1 + strict_f1)"
```

---

## Phase E — Tokenizer for Gold + Pseudo

### Task E1: Build script `scripts/build_track_e_tokenizer.py`

**Files:**
- Create: `scripts/build_track_e_tokenizer.py`
- Test: `tests/test_build_track_e_tokenizer.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_build_track_e_tokenizer.py`:

```python
"""Test for the Track E BPE tokenizer builder."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


def test_builder_runs_on_synthetic_input(project_root: Path, tmp_path: Path) -> None:
    """Given a single gold jsonl, the script writes a tokenizer.json that is loadable."""
    gold = tmp_path / "gold.jsonl"
    gold.write_text('{"image_path": "x.jpg", "xml_label": "<medication><brand_name>Crocin</brand_name><drug_name>Crocin Advance</drug_name></medication>"}\n')
    out = tmp_path / "track_e_bpe.json"
    result = subprocess.run(
        ["python", "scripts/build_track_e_tokenizer.py",
         "--train-jsonl", str(gold),
         "--out", str(out),
         "--vocab-size", "256"],  # small for test
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    data = json.loads(out.read_text())
    assert "version" in data or "model" in data  # standard tokenizers.json structure


def test_builder_includes_pseudo_when_provided(project_root: Path, tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    pseudo = tmp_path / "pseudo.jsonl"
    gold.write_text('{"image_path": "g.jpg", "xml_label": "<medication><brand_name>X</brand_name></medication>"}\n')
    pseudo.write_text('{"image_path": "p.jpg", "xml_label": "<medication><brand_name>UNIQUE_PSEUDO_TOKEN</brand_name></medication>"}\n')
    out = tmp_path / "track_e_bpe.json"
    result = subprocess.run(
        ["python", "scripts/build_track_e_tokenizer.py",
         "--train-jsonl", str(gold),
         "--pseudo-jsonl", str(pseudo),
         "--out", str(out),
         "--vocab-size", "512"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    # Don't assert specific token presence — vocab might be subword-split — just that the run succeeded
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_build_track_e_tokenizer.py -v`
Expected: FAIL — script doesn't exist.

- [ ] **Step 3: Implement**

Create `scripts/build_track_e_tokenizer.py`:

```python
"""Train a BPE tokenizer on the union of gold + pseudo XML targets.

Output: a tokenizers JSON file usable by prepare.get_tokenizer().

Usage:
    python scripts/build_track_e_tokenizer.py \
        --train-jsonl data/processed/train.jsonl \
        --pseudo-jsonl data/pseudo_labels/round_001.jsonl \
        --out experiments/tokenizers/track_e_bpe.json \
        --vocab-size 8192
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prepare  # noqa: E402


def _read_xml_targets(path: Path) -> list[str]:
    targets: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            targets.append(row.get("xml_label", ""))
    return targets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--pseudo-jsonl", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int, default=prepare.TOKENIZER_VOCAB_SIZE)
    args = ap.parse_args()

    targets = _read_xml_targets(Path(args.train_jsonl))
    if args.pseudo_jsonl:
        targets.extend(_read_xml_targets(Path(args.pseudo_jsonl)))

    print(f"Training BPE on {len(targets)} XML targets (vocab={args.vocab_size})")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Delegate to prepare.train_tokenizer which already handles special tokens correctly
    prepare.train_tokenizer(targets=targets, vocab_size=args.vocab_size, out_path=out)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Note:** if `prepare.train_tokenizer`'s signature differs from `(targets, vocab_size, out_path)`, adjust the call accordingly. Read the existing function (around line 101 in `prepare.py`) and match its parameters.

- [ ] **Step 4: Run test**

Run: `pytest tests/test_build_track_e_tokenizer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_track_e_tokenizer.py tests/test_build_track_e_tokenizer.py
git commit -m "feat(scripts): build_track_e_tokenizer — BPE on gold + pseudo XML"
```

---

## Phase F — Config Files

### Task F1: `experiments/configs/track_e_highres_gold_only.yaml`

**Why:** First run — isolates the resolution effect.

- [ ] **Step 1: Create the file**

Create `experiments/configs/track_e_highres_gold_only.yaml`:

```yaml
# Track E run #1: SigLIP-large-384 trained on gold only.
# Isolates the resolution upgrade effect (vs Track A's SigLIP-base-224).
# Headline metric: safety-4 macro_edit_f1 on val.

seed: 44
encoder_model: "google/siglip-large-patch16-384"
image_size: 384
hidden_dim: 512                   # decoder d_model; bridge handles encoder_hidden → 512
n_decoder_layers: 6
n_decoder_heads: 8
ffn_ratio: 4
dropout: 0.1
tied_embeddings: true
max_target_length: 256
batch_size: 4                     # smaller per-step batch due to bigger encoder
peak_lr: 3.0e-4
min_lr: 3.0e-5
warmup_steps: 200
max_steps: 1500
weight_decay: 0.05
adam_betas: [0.9, 0.95]
grad_clip: 1.0
eval_every: 200
log_every: 25
precision: "bf16"

# Data paths (Track A defaults)
train_jsonl: "data/processed/train.jsonl"
val_jsonl: "data/processed/val.jsonl"
images_root: "/mnt/gcs/raw/raw_images"
path_strip_prefix: "raw/raw_images/"

# Track E tokenizer (built on gold only here; pseudo additions happen in track_e_highres.yaml)
tokenizer_path: "experiments/tokenizers/track_e_bpe_gold.json"
track_e_tokenizer_path: "experiments/tokenizers/track_e_bpe_gold.json"

# Output
checkpoint_dir: "checkpoints/runs/track_e_highres_gold_only-seed44"
wandb_project: "autolearnmeds"
wandb_mode: "online"

# Pseudo disabled
pseudo_jsonl: ""
pseudo_weight: 0.0

# Single-stage (the gold-only-control)
stage_schedule: []

# Calibration / abstention curves
compute_calibration: true
```

- [ ] **Step 2: Smoke-test the YAML loads into Config**

Run: `python -c "import yaml, sys; sys.path.insert(0, '.'); from train import Config; cfg = Config(**yaml.safe_load(open('experiments/configs/track_e_highres_gold_only.yaml'))); print('OK', cfg.encoder_model, cfg.image_size)"`
Expected: `OK google/siglip-large-patch16-384 384`

- [ ] **Step 3: Commit**

```bash
git add experiments/configs/track_e_highres_gold_only.yaml
git commit -m "config(track-e): track_e_highres_gold_only — SigLIP-large-384, gold only, 1500 steps"
```

---

### Task F2: `experiments/configs/track_e_highres.yaml`

**Why:** Run #3 — the full Track E recipe with pseudo-labels and the 3-stage schedule.

- [ ] **Step 1: Create the file**

Create `experiments/configs/track_e_highres.yaml`:

```yaml
# Track E run #3: SigLIP-large-384 + Qwen2-VL pseudo-labels @ weight 0.3, 3-stage schedule.
# Headline candidate: gold (564) + filtered pseudo (~2,275) over 3,000 steps.

seed: 44
encoder_model: "google/siglip-large-patch16-384"
image_size: 384
hidden_dim: 512
n_decoder_layers: 6
n_decoder_heads: 8
ffn_ratio: 4
dropout: 0.1
tied_embeddings: true
max_target_length: 256
batch_size: 4
peak_lr: 3.0e-4
min_lr: 3.0e-5
warmup_steps: 200
max_steps: 3000                     # consumed by stage_schedule (sum of steps below)
weight_decay: 0.05
adam_betas: [0.9, 0.95]
grad_clip: 1.0
eval_every: 200
log_every: 25
precision: "bf16"

train_jsonl: "data/processed/train.jsonl"
val_jsonl: "data/processed/val.jsonl"
images_root: "/mnt/gcs/raw/raw_images"
path_strip_prefix: "raw/raw_images/"

tokenizer_path: "experiments/tokenizers/track_e_bpe.json"
track_e_tokenizer_path: "experiments/tokenizers/track_e_bpe.json"

checkpoint_dir: "checkpoints/runs/track_e_highres_full-seed44"
wandb_project: "autolearnmeds"
wandb_mode: "online"

# Pseudo
pseudo_jsonl: "data/pseudo_labels/round_001.jsonl"
pseudo_weight: 0.3
pseudo_weight_mode: "const"
pseudo_min_confidence: "medium"

# 3-stage schedule: pseudo warmup → joint → gold finetune
stage_schedule:
  - { name: "warmup",   steps: 500,  data: "pseudo", lr_mult: 1.0 }
  - { name: "joint",    steps: 2000, data: "all",    lr_mult: 1.0 }
  - { name: "finetune", steps: 500,  data: "gold",   lr_mult: 0.1 }

compute_calibration: true
```

- [ ] **Step 2: Smoke-test**

Run: `python -c "import yaml, sys; sys.path.insert(0, '.'); from train import Config; cfg = Config(**yaml.safe_load(open('experiments/configs/track_e_highres.yaml'))); print('OK stages:', len(cfg.stage_schedule), 'pseudo:', cfg.pseudo_weight)"`
Expected: `OK stages: 3 pseudo: 0.3`

- [ ] **Step 3: Commit**

```bash
git add experiments/configs/track_e_highres.yaml
git commit -m "config(track-e): track_e_highres — full recipe with pseudo + 3-stage"
```

---

## Phase G — Scripts

### Task G1: Extend `run_experiment.sh` for Track E

**Files:**
- Modify: `scripts/run_experiment.sh`
- Test: `tests/test_run_experiment_track_e.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_experiment_track_e.py`:

```python
"""Test that run_experiment.sh accepts --track E and picks the right defaults."""
from __future__ import annotations

import subprocess
from pathlib import Path


def test_track_e_default_config_resolves(project_root: Path, tmp_path: Path) -> None:
    """Invoking with --track E --dry-run (we'll add this flag) should print the resolved config path."""
    result = subprocess.run(
        ["bash", "scripts/run_experiment.sh", "test-run-id", "--track", "E", "--dry-run"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    # Even if --dry-run isn't supported, at least the case statement should accept E
    # We check for the absence of the "unknown track" error
    assert "unknown track" not in result.stderr.lower(), result.stderr
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_run_experiment_track_e.py -v`
Expected: FAIL — `unknown track E`.

- [ ] **Step 3: Modify `scripts/run_experiment.sh`**

Find the case statement (around line 35) and extend:

```bash
case "$TRACK" in
  A) TRAINER="train.py"; DEFAULT_CONFIG="experiments/configs/baseline.yaml" ;;
  B) TRAINER="train_qwen.py"; DEFAULT_CONFIG="experiments/configs/qwen_baseline.yaml" ;;
  E) TRAINER="train.py"; DEFAULT_CONFIG="experiments/configs/track_e_highres.yaml" ;;
  *) echo "[run_experiment] FATAL: unknown track $TRACK (expected A, B, or E)" >&2; exit 2 ;;
esac
```

Add a `--dry-run` flag handler near the top of the arg parse:

```bash
DRY_RUN=0
# (inside the while loop)
    --dry-run) DRY_RUN=1; shift;;
```

And before the training call, gate it:

```bash
if (( DRY_RUN )); then
  echo "[run_experiment] DRY RUN: would run python $TRAINER --config $CONFIG $SEED_ARG"
  exit 0
fi
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_run_experiment_track_e.py -v`
Expected: PASS.

Run regression: `pytest tests/test_smoke.py -v` (if it exists).
Expected: existing PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_experiment.sh tests/test_run_experiment_track_e.py
git commit -m "feat(scripts): run_experiment.sh accepts --track E + --dry-run"
```

---

## Phase H — Plot Scripts

### Task H1: `scripts/plots/per_field_safety4.py`

**Files:**
- Create: `scripts/plots/per_field_safety4.py`
- Test: `tests/test_plots_safety4.py` (new)

- [ ] **Step 1: Write the failing test (smoke-style)**

Create `tests/test_plots_safety4.py`:

```python
"""Smoke test: per_field_safety4.py produces a non-zero PNG."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_per_field_safety4_plot(project_root: Path, tmp_path: Path) -> None:
    """Given two per-field JSONs, the plot script writes a PNG with non-zero size."""
    track_a = tmp_path / "per_field_track_a.json"
    track_e = tmp_path / "per_field_track_e.json"
    track_a.write_text(json.dumps({
        "per_field_f1": {"batch_number": 0.0, "mfg_date": 0.05, "expiry_date": 0.05, "mrp": 0.10},
        "per_field_edit_f1": {"batch_number": 0.18, "mfg_date": 0.22, "expiry_date": 0.29, "mrp": 0.35},
    }))
    track_e.write_text(json.dumps({
        "per_field_f1": {"batch_number": 0.30, "mfg_date": 0.40, "expiry_date": 0.45, "mrp": 0.50},
        "per_field_edit_f1": {"batch_number": 0.55, "mfg_date": 0.60, "expiry_date": 0.65, "mrp": 0.70},
    }))
    out = tmp_path / "per_field_safety4.png"
    result = subprocess.run(
        ["python", "scripts/plots/per_field_safety4.py",
         "--per-field", f"track_a={track_a}", f"track_e={track_e}",
         "--metric", "edit",
         "--out", str(out)],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists() and out.stat().st_size > 1000
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_plots_safety4.py -v`
Expected: FAIL — script missing.

- [ ] **Step 3: Implement**

Create `scripts/plots/per_field_safety4.py`:

```python
"""Bar chart restricted to the 4 safety-critical fields, multi-track.

Usage:
    python scripts/plots/per_field_safety4.py \
        --per-field track_a=experiments/per_field_track_a.json \
        --per-field track_e=experiments/per_field_track_e.json \
        --metric edit \
        --out paper/figures/safety4_cross_track.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from prepare import SAFETY_FIELDS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-field", action="append", required=True,
                    help="name=path/to/per_field.json (repeat per track)")
    ap.add_argument("--metric", choices=["strict", "edit"], default="edit")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fields = sorted(SAFETY_FIELDS)
    tracks = {}
    for spec in args.per_field:
        name, path = spec.split("=", 1)
        data = json.loads(Path(path).read_text())
        key = "per_field_edit_f1" if args.metric == "edit" else "per_field_f1"
        per = data.get(key, {})
        tracks[name] = [per.get(f, 0.0) for f in fields]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(fields))
    w = 0.8 / max(1, len(tracks))
    for i, (name, vals) in enumerate(tracks.items()):
        ax.bar(x + i * w, vals, width=w, label=name)
    ax.set_xticks(x + (len(tracks) - 1) * w / 2)
    ax.set_xticklabels(fields, rotation=20)
    ax.set_ylabel(f"F1 ({args.metric})")
    ax.set_title("Safety-4 fields — per-track")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_plots_safety4.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/plots/per_field_safety4.py tests/test_plots_safety4.py
git commit -m "feat(plots): per_field_safety4 — cross-track bar chart restricted to safety-4 fields"
```

---

### Task H2: `scripts/plots/envelope.py`

**Files:**
- Create: `scripts/plots/envelope.py`
- Test: extend `tests/test_plots_safety4.py` or new file

- [ ] **Step 1: Write the smoke test**

Append to `tests/test_plots_safety4.py` (or create `tests/test_plots_envelope.py`):

```python
def test_envelope_plot(project_root: Path, tmp_path: Path) -> None:
    """Envelope across multiple runs reads ledger.jsonl entries and produces a PNG."""
    import json
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("\n".join(json.dumps(r) for r in [
        {"run_id": "track_a_baseline", "metrics": {"safety4_macro_edit_f1": 0.10}, "timestamp": "2026-05-05T14:00:00Z"},
        {"run_id": "track_e_gold_only", "metrics": {"safety4_macro_edit_f1": 0.18}, "timestamp": "2026-05-13T14:00:00Z"},
        {"run_id": "track_e_full", "metrics": {"safety4_macro_edit_f1": 0.25}, "timestamp": "2026-05-14T14:00:00Z"},
    ]) + "\n")
    out = tmp_path / "envelope.png"
    result = subprocess.run(
        ["python", "scripts/plots/envelope.py", "--ledger", str(ledger), "--out", str(out)],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists() and out.stat().st_size > 1000
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_plots_safety4.py::test_envelope_plot -v`
Expected: FAIL — script missing.

- [ ] **Step 3: Implement**

Create `scripts/plots/envelope.py`:

```python
"""Running-envelope plot: safety-4 macro_edit_f1 across all runs in the ledger, over time.

Usage:
    python scripts/plots/envelope.py --ledger experiments/ledger.jsonl --out paper/figures/_envelope.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    with open(args.ledger) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            value = r.get("metrics", {}).get("safety4_macro_edit_f1")
            if value is None:
                continue
            rows.append({
                "run_id": r.get("run_id", "?"),
                "value": value,
                "ts": r.get("timestamp", ""),
            })
    rows.sort(key=lambda r: r["ts"])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = list(range(len(rows)))
    ys = [r["value"] for r in rows]
    envelope = [max(ys[:i+1]) for i in range(len(ys))]
    ax.plot(xs, ys, "o-", label="per-run safety-4 macro_edit_f1", alpha=0.6)
    ax.plot(xs, envelope, "s--", color="red", label="running envelope (best so far)")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["run_id"] for r in rows], rotation=30, ha="right")
    ax.set_ylabel("safety-4 macro_edit_f1")
    ax.set_title("Running envelope across runs")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_plots_safety4.py::test_envelope_plot -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/plots/envelope.py tests/test_plots_safety4.py
git commit -m "feat(plots): envelope.py — safety-4 running envelope across ledger entries"
```

---

### Task H3: `scripts/plots/calibration_curve.py`

**Files:**
- Create: `scripts/plots/calibration_curve.py`
- Test: extend `tests/test_plots_safety4.py`

- [ ] **Step 1: Write the smoke test**

Append:

```python
def test_calibration_curve(project_root: Path, tmp_path: Path) -> None:
    """Given a predictions JSON with per-field entropies + correctness, produce a coverage-vs-accuracy PNG."""
    import json
    preds = tmp_path / "preds.json"
    preds.write_text(json.dumps([
        {"field": "batch_number", "correct": True,  "entropy": 0.1},
        {"field": "batch_number", "correct": False, "entropy": 0.9},
        {"field": "expiry_date",  "correct": True,  "entropy": 0.2},
        {"field": "mrp",          "correct": False, "entropy": 0.5},
        {"field": "mfg_date",     "correct": True,  "entropy": 0.3},
    ]))
    out = tmp_path / "calibration.png"
    result = subprocess.run(
        ["python", "scripts/plots/calibration_curve.py", "--preds", str(preds), "--out", str(out)],
        cwd=project_root, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists() and out.stat().st_size > 1000
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_plots_safety4.py::test_calibration_curve -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Create `scripts/plots/calibration_curve.py`:

```python
"""Coverage-vs-accuracy at varying entropy thresholds.

Input: a JSON list of {field, correct (bool), entropy (float)} per prediction.
Output: PNG showing two curves — coverage and accuracy — as the entropy threshold sweeps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = json.loads(Path(args.preds).read_text())
    entropies = np.array([r["entropy"] for r in rows])
    correct = np.array([r["correct"] for r in rows])

    thresholds = np.linspace(0, 1, 51)
    coverages = []
    accuracies = []
    for t in thresholds:
        mask = entropies <= t
        coverages.append(mask.mean())
        accuracies.append(correct[mask].mean() if mask.any() else 1.0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, coverages, label="coverage", color="tab:blue")
    ax.plot(thresholds, accuracies, label="accuracy on covered", color="tab:orange")
    ax.set_xlabel("entropy threshold T")
    ax.set_ylabel("rate")
    ax.set_title("Calibration: accuracy vs coverage at entropy threshold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_plots_safety4.py::test_calibration_curve -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/plots/calibration_curve.py tests/test_plots_safety4.py
git commit -m "feat(plots): calibration_curve — accuracy vs coverage at entropy threshold sweep"
```

---

### Task H4: `scripts/plots/error_samples.py`

**Files:**
- Create: `scripts/plots/error_samples.py`
- Test: extend `tests/test_plots_safety4.py`

- [ ] **Step 1: Write the smoke test**

Append:

```python
def test_error_samples_plot(project_root: Path, tmp_path: Path) -> None:
    """Smoke: error_samples runs against a tiny JSON and writes a PNG."""
    import json
    samples = tmp_path / "samples.json"
    # Three categories, one fake sample each, with placeholder image paths (script must handle missing images gracefully).
    samples.write_text(json.dumps([
        {"category": "still_wrong", "image_path": "fake1.jpg", "field": "batch_number", "gold": "B1", "pred": "Bz"},
        {"category": "fixed_by_e",  "image_path": "fake2.jpg", "field": "expiry_date",  "gold": "10/25", "pred": "10/25"},
        {"category": "regressed",   "image_path": "fake3.jpg", "field": "mrp",          "gold": "85",   "pred": "8.5"},
    ]))
    out = tmp_path / "errors.png"
    result = subprocess.run(
        ["python", "scripts/plots/error_samples.py", "--samples", str(samples), "--out", str(out)],
        cwd=project_root, capture_output=True, text=True,
    )
    # Allow nonzero exit if images can't be loaded — but must still write a PNG with at least placeholder cells.
    assert out.exists() and out.stat().st_size > 1000, result.stderr
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_plots_safety4.py::test_error_samples_plot -v`
Expected: FAIL — script missing.

- [ ] **Step 3: Implement**

Create `scripts/plots/error_samples.py`:

```python
"""9-panel qualitative grid: 3 each of still-wrong / fixed-by-E / regressed predictions.

Input: a JSON list of {category, image_path, field, gold, pred} (3+ per category).
Output: a 3x3 PNG. If image_path can't be loaded, draws a placeholder rectangle with the text.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


CATEGORIES = ["still_wrong", "fixed_by_e", "regressed"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = json.loads(Path(args.samples).read_text())
    by_cat: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    for r in rows:
        if r["category"] in by_cat:
            by_cat[r["category"]].append(r)

    fig, axes = plt.subplots(3, 3, figsize=(11, 9))
    for col, cat in enumerate(CATEGORIES):
        for row in range(3):
            ax = axes[row, col]
            ax.axis("off")
            entries = by_cat.get(cat, [])
            if row < len(entries):
                e = entries[row]
                try:
                    img = Image.open(e["image_path"]).convert("RGB")
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, f"[image\nmissing]", ha="center", va="center", transform=ax.transAxes)
                title = f"{cat}\n{e['field']}\nGT: {e['gold']}\nPred: {e['pred']}"
                ax.set_title(title, fontsize=8)
    fig.suptitle("Error samples by category (3 each)", fontsize=12)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_plots_safety4.py::test_error_samples_plot -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/plots/error_samples.py tests/test_plots_safety4.py
git commit -m "feat(plots): error_samples — 9-panel qualitative grid"
```

---

## Phase I — Integration Smoke Test

### Task I1: End-to-end dry-run of Track E gold-only config

**Why:** Validate that all the additions hold together. We don't run actual training, just verify the config loads, the preflight passes, the dataset constructs, and the first forward pass completes.

**Files:**
- Create: `tests/test_track_e_smoke.py`
- (Optionally requires GPU; mark accordingly)

- [ ] **Step 1: Write the smoke test**

Create `tests/test_track_e_smoke.py`:

```python
"""End-to-end smoke: Track E gold-only config loads and the trainer reaches the first eval."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif("not torch.cuda.is_available()", reason="requires GPU")
def test_track_e_gold_only_first_step(project_root: Path, tmp_path: Path) -> None:
    """Run train.py with max_steps=10 to verify the pipeline doesn't crash.

    This test requires the env to have the model artifacts (HF model cache), data files,
    and a GPU. CI will skip it; the user runs it on Colab once.
    """
    # Use the existing baseline.yaml as base, then override the Track E flags
    cfg_overrides = {
        "max_steps": 10,
        "eval_every": 10,
        "encoder_model": "google/siglip-large-patch16-384",
        "image_size": 384,
        "batch_size": 1,
    }
    # Write a tmp config that merges baseline + overrides
    import yaml
    base = yaml.safe_load((project_root / "experiments" / "configs" / "baseline.yaml").read_text())
    base.update(cfg_overrides)
    tmp_cfg = tmp_path / "smoke.yaml"
    tmp_cfg.write_text(yaml.safe_dump(base))

    result = subprocess.run(
        ["python", "train.py", "--config", str(tmp_cfg)],
        cwd=project_root, capture_output=True, text=True, timeout=600,
    )
    # Trainer should exit 0 and emit `final_macro_f1=...` on its last stdout line
    assert result.returncode == 0, result.stderr
    last_line = result.stdout.strip().splitlines()[-1]
    assert last_line.startswith("final_macro_f1="), f"Last stdout line: {last_line!r}"
```

- [ ] **Step 2: Run on Colab (not on CI / Codex's machine)**

User runs on Colab once everything else lands:

```bash
cd /content/AutoLearnMeds && git pull
pytest tests/test_track_e_smoke.py -v -s
```

Expected: PASS after ~5-10 minutes (10 steps of SigLIP-large training).

- [ ] **Step 3: Commit the test**

```bash
git add tests/test_track_e_smoke.py
git commit -m "test(track-e): GPU-gated end-to-end smoke test"
```

---

## Self-Review Notes

**Spec coverage:**
- §3 Architecture: Tasks C1, D1, F1 ✅
- §4 Data strategy: Tasks B1, B2, C1, D3 ✅
- §5 Training regime (3-stage, optimizer, dual-best checkpoint, preflight): Tasks C1, D2, D4, D5 ✅
- §6 Evaluation & calibration: Tasks A1, A2, H3 ✅
- §7 Risks: Task D2 (preflight), Task B2 (filter pseudo) ✅; the "OOM fallback" is operational, not coded.
- §8 Sequencing: Tasks F1, F2, G1 deliver the configs and runner; sequencing is execution-time
- §10 Anti-goals: Tasks B1, C1 all preserve old behavior — verified by Task C1's regression tests
- §11 Files: every file in §11 is created or modified by a task in this plan ✅

**Placeholder scan:** No TBDs/TODOs left in step bodies. Code blocks are complete syntax.

**Type consistency:**
- `Stage` dataclass used in Task D4 — `name: str, steps: int, data: str, lr_mult: float`. Consistent across StageScheduler tests and trainer integration.
- `BestCheckpointTracker.is_new_best(metric: str, value: float) -> bool` — same signature in test + impl.
- `build_combined_train_rows(cfg) -> tuple[list[dict], list[float]]` — same in test + impl.
- `compute_subset_metrics(predictions, truths, fields) -> dict` — same shape as `compute_metrics`.
- `filter_pseudo_rows(rows, val_paths, test_paths) -> tuple[list, dict]` — consistent.

**Items deliberately NOT in this plan:**
- Codex briefs (002 pseudo-label oracle on Colab, 003 GCS inventory) — those are Codex's plan, not Claude's.
- Paper writeup (`paper/main.md` Section V grafting) — separate effort, parallel.
- The actual training runs (#1, #3, ablations) — operations, not code.
- Stage 3 paper-claim test eval run — operations.
- DINOv2 ablation (run #8) — additive config; trivial once Phase F YAMLs exist.

---

## Plan complete and saved to `docs/superpowers/plans/2026-05-13-track-e-deployable-label-reader.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
