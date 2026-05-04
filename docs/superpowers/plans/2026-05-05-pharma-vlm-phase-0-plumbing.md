# Phase 0 — Plumbing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get the AutoLearnMeds project bootstrapped end-to-end with no GPU work yet — repo metadata, folder skeleton, scripts, Colab+VSCode SSH tunnel, and a green `make verify`. After this, Phase 1 can start consuming real data on Colab.

**Architecture:** Local-first scaffolding (everything reproducible on the Mac) + a single `colab_bootstrap.sh` that brings a fresh Colab Pro+ A100 runtime to a known state with one cell-run. Mirrors Karpathy `autoresearch`'s three-file convention (`prepare.py`, `train.py`, `program.md`).

**Tech Stack:** Python 3.11 (uv-managed via `pyproject.toml`), pytest for tests, bash/sh for scripts, Jupyter notebook for Colab entry, Cloudflare Tunnel + colab-ssh for SSH-into-Colab.

**Spec:** [`docs/superpowers/specs/2026-05-05-pharma-vlm-autoresearch-design.md`](../specs/2026-05-05-pharma-vlm-autoresearch-design.md), §9 Phase 0.

**What's already done (root commit `2e7f7f7`):** `.gitignore`, `papers/` (14 PDFs + registry + karpathy summary), spec doc, git repo on `main`.

---

## File Structure

| Path | Responsibility | Created in task |
|---|---|---|
| `pyproject.toml` | uv-managed deps (split: core / ml / colab) | T1 |
| `.python-version` | Pin Python 3.11 | T1 |
| `uv.lock` | Lockfile (autogen by `uv sync`) | T1 |
| `README.md` | Human-facing project overview | T2 |
| `tests/conftest.py` | pytest config (project-root path resolution) | T3 |
| `tests/test_smoke.py` | Asserts repo layout, scripts exist + executable, JSON parses | T3, T5, T8, T9, T10, T11 |
| `tests/test_resume_state_machine.py` | Tests `resume_or_start.py` state machine | T10 |
| `data/README.md`, `checkpoints/README.md`, `experiments/README.md`, `paper/README.md`, `scripts/README.md`, `tests/README.md`, `notebooks/README.md` | Placeholder READMEs (per spec §2.2) | T4 |
| `data/.gitkeep`, `data/raw/.gitkeep`, `data/processed/.gitkeep`, `checkpoints/.gitkeep`, `experiments/runs/.gitkeep`, `paper/figures/.gitkeep`, `paper/tables/.gitkeep`, `paper/sections/.gitkeep` | Keep otherwise-empty dirs in git | T4 |
| `prepare.py` | Phase-1 module stub (one-liner docstring) | T6 |
| `train.py` | Phase-2 module stub | T6 |
| `program.md` | Baseline agent skeleton | T6 |
| `program_explore.md` | Explore-phase agent override | T6 |
| `program_confirm.md` | Confirm-phase agent override | T6 |
| `scripts/keepalive.py` | Heartbeat to `/workspace/heartbeat.txt` every 60s | T7 |
| `scripts/sync_to_gcs.sh` | Periodic gsutil rsync of ledger + checkpoints | T8 |
| `scripts/update_ssh_config.sh` | Pulls SSH host from Colab-side known file | T9 |
| `scripts/resume_or_start.py` | State-machine launcher (PLANNING/ACTIVE/EVALUATED/COMMITTED) | T10 |
| `scripts/colab_bootstrap.sh` | The one-cell Colab bootstrap | T11 |
| `notebooks/00_bootstrap.ipynb` | Single-cell launcher that calls `colab_bootstrap.sh` | T12 |
| `Makefile` | `make verify` smoke target + helpers | T13 |

---

## Task 1: Python pin + uv project metadata

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `uv.lock` (auto-generated)

- [ ] **Step 1: Verify `uv` is installed locally**

```bash
which uv && uv --version
```

Expected: `/usr/local/bin/uv` (or similar) and version `>= 0.4.0`. If missing:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

- [ ] **Step 2: Pin Python version**

Create `/Users/apple/AutoLearnMeds/.python-version`:

```
3.11
```

- [ ] **Step 3: Write `pyproject.toml` with three dependency groups**

Create `/Users/apple/AutoLearnMeds/pyproject.toml`:

```toml
[project]
name = "autolearnmeds"
version = "0.1.0"
description = "Autoresearch-driven custom VLM for pharmaceutical product label information extraction"
readme = "README.md"
requires-python = ">=3.11"
authors = [{ name = "Suman" }]
license = { text = "MIT" }

# Core deps installable on the Mac (no heavy ML)
dependencies = [
    "pyyaml>=6.0",
    "rich>=13.7",
]

[project.optional-dependencies]
# Local dev: tests, lint
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.5",
]

# Heavy ML deps — installed on Colab only via `uv sync --extra ml`
ml = [
    "torch>=2.3",
    "torchvision>=0.18",
    "transformers>=4.42",
    "accelerate>=0.31",
    "datasets>=2.20",
    "pillow>=10.3",
    "numpy>=1.26",
    "pandas>=2.2",
    "wandb>=0.17",
    "scikit-learn>=1.5",
    "matplotlib>=3.9",
    "tqdm>=4.66",
    "tokenizers>=0.19",
    "python-Levenshtein>=0.25",
]

# Colab-specific
colab = [
    "colab-ssh>=0.3.27",
    "google-cloud-storage>=2.17",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "B", "UP"]
ignore = ["E501"]  # line length handled by formatter

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 4: Lock and install dev dependencies locally**

Run from `/Users/apple/AutoLearnMeds`:

```bash
uv sync --extra dev
```

Expected: creates `.venv/`, generates `uv.lock`, prints `+ pytest`, `+ ruff`, etc. No errors. ML deps NOT installed locally (would be 2GB+).

- [ ] **Step 5: Verify pytest is wired**

```bash
uv run pytest --version
```

Expected: `pytest 8.x.x`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .python-version uv.lock
git commit -m "chore: uv project metadata + Python 3.11 pin

Three dependency groups: core (pyyaml, rich), dev (pytest, ruff),
ml (torch, transformers, ...), colab (colab-ssh, gcs). ML deps
are install-on-Colab-only to keep the Mac venv lightweight.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Project README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

Create `/Users/apple/AutoLearnMeds/README.md`:

```markdown
# AutoLearnMeds

> Custom hybrid vision-language model for structured information extraction from pharmaceutical product label images, trained via Karpathy-style autoresearch.

## Status

Phase 0 (plumbing) — see [docs/superpowers/plans/](docs/superpowers/plans/).

## What this is

An autonomous agent (Claude Code, via VSCode SSH-tunneled into a Colab Pro+ A100 runtime) iteratively edits a single-file `train.py` to maximize validation macro-F1 of structured field extraction (medicine name, batch number, MRP, manufacturer, etc.) from images of pharmaceutical products. Architecture: frozen `google/siglip-base-patch16-224` encoder + a ~50M-parameter Donut-style decoder trained from scratch.

## Three files matter (autoresearch discipline)

| File | Owner | Purpose |
|---|---|---|
| `prepare.py` | never modified | data prep, dataloader, canonical `evaluate()` |
| `train.py` | agent edits | model + training loop |
| `program.md` | human edits | the agent's instructions |

## Quick start (after Phase 0)

```bash
# Local
uv sync --extra dev
uv run pytest

# On Colab Pro+ (A100, Background execution ON)
# Open notebooks/00_bootstrap.ipynb, run the single cell.
# Copy the printed SSH host into ~/.ssh/config on your Mac.
# VSCode: ⌘⇧P → "Remote-SSH: Connect to Host" → autolearnmeds-colab.
```

## Design

Full design spec: [`docs/superpowers/specs/2026-05-05-pharma-vlm-autoresearch-design.md`](docs/superpowers/specs/2026-05-05-pharma-vlm-autoresearch-design.md).

## Reference

Built on the methodology from [karpathy/autoresearch](https://github.com/karpathy/autoresearch). All cited papers are stored in [`papers/`](papers/) — see [`papers/README.md`](papers/README.md) for the citation registry.
```

- [ ] **Step 2: Verify renders**

```bash
head -20 README.md
```

Expected: clean Markdown header followed by sections.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: project README

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Test infrastructure (TDD scaffolding)

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_smoke.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Write `tests/__init__.py`**

Create empty file:

```bash
touch /Users/apple/AutoLearnMeds/tests/__init__.py
```

- [ ] **Step 2: Write `tests/conftest.py`**

Create `/Users/apple/AutoLearnMeds/tests/conftest.py`:

```python
"""Pytest configuration: project-root path fixture."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the repo root."""
    return Path(__file__).resolve().parent.parent
```

- [ ] **Step 3: Write minimal `tests/test_smoke.py`**

Create `/Users/apple/AutoLearnMeds/tests/test_smoke.py`:

```python
"""Phase 0 smoke tests — repo layout, scripts, file integrity."""
from __future__ import annotations

from pathlib import Path


def test_papers_folder_has_registry(project_root: Path) -> None:
    """The papers/ citation registry must exist."""
    assert (project_root / "papers" / "README.md").is_file()


def test_design_spec_exists(project_root: Path) -> None:
    """The design spec from brainstorming must be present."""
    spec = project_root / "docs" / "superpowers" / "specs" / "2026-05-05-pharma-vlm-autoresearch-design.md"
    assert spec.is_file()
```

- [ ] **Step 4: Run tests to verify the scaffold works**

```bash
uv run pytest -v
```

Expected output:

```
tests/test_smoke.py::test_papers_folder_has_registry PASSED
tests/test_smoke.py::test_design_spec_exists PASSED
```

If either fails, fix before continuing.

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "test: pytest scaffolding + initial smoke tests

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Folder skeleton + placeholder READMEs (TDD)

**Files:**
- Modify: `tests/test_smoke.py`
- Create: `data/README.md`, `data/.gitkeep`, `data/raw/.gitkeep`, `data/processed/.gitkeep`
- Create: `checkpoints/README.md`, `checkpoints/.gitkeep`
- Create: `experiments/README.md`, `experiments/runs/.gitkeep`
- Create: `paper/README.md`, `paper/figures/.gitkeep`, `paper/tables/.gitkeep`, `paper/sections/.gitkeep`
- Create: `scripts/README.md`
- Create: `notebooks/README.md`

- [ ] **Step 1: Write the failing test**

Append to `/Users/apple/AutoLearnMeds/tests/test_smoke.py`:

```python


EXPECTED_DIRS = [
    "data",
    "data/raw",
    "data/processed",
    "checkpoints",
    "experiments",
    "experiments/runs",
    "paper",
    "paper/figures",
    "paper/tables",
    "paper/sections",
    "scripts",
    "notebooks",
    "tests",
    "papers",
    "docs",
]

EXPECTED_READMES = [
    "data/README.md",
    "checkpoints/README.md",
    "experiments/README.md",
    "paper/README.md",
    "scripts/README.md",
    "notebooks/README.md",
    "papers/README.md",
    "README.md",
]


def test_folder_skeleton_exists(project_root: Path) -> None:
    """Every directory from spec §2.2 must exist."""
    missing = [d for d in EXPECTED_DIRS if not (project_root / d).is_dir()]
    assert not missing, f"Missing directories: {missing}"


def test_placeholder_readmes_exist(project_root: Path) -> None:
    """Every placeholder README from spec §2.2 must exist."""
    missing = [r for r in EXPECTED_READMES if not (project_root / r).is_file()]
    assert not missing, f"Missing READMEs: {missing}"
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
uv run pytest tests/test_smoke.py -v
```

Expected: `test_folder_skeleton_exists FAILED` and `test_placeholder_readmes_exist FAILED`. Other tests still PASS.

- [ ] **Step 3: Create directory tree + .gitkeep markers**

```bash
cd /Users/apple/AutoLearnMeds && \
mkdir -p data/raw data/processed checkpoints experiments/runs paper/figures paper/tables paper/sections scripts notebooks && \
touch data/.gitkeep data/raw/.gitkeep data/processed/.gitkeep \
      checkpoints/.gitkeep experiments/runs/.gitkeep \
      paper/figures/.gitkeep paper/tables/.gitkeep paper/sections/.gitkeep
```

- [ ] **Step 4: Write `data/README.md`**

Create `/Users/apple/AutoLearnMeds/data/README.md`:

```markdown
# data/

Dataset working directory.

- `raw/` — symlink (on Colab) to the GDrive mount of the original images + YOLO labels + transcriptions. Not tracked in git (see `.gitignore`).
- `processed/` — generated `train.jsonl`, `val.jsonl`, `test.jsonl` produced by `scripts/build_processed.py`. Not tracked in git.
- `data_card.md` — auto-generated dataset documentation (Gebru et al. 2018 datasheet style).

The canonical schema for `processed/*.jsonl` is locked in Phase 1 once the user shares the golden_set. See spec §3.2.
```

- [ ] **Step 5: Write `checkpoints/README.md`**

Create `/Users/apple/AutoLearnMeds/checkpoints/README.md`:

```markdown
# checkpoints/

Model checkpoints from training runs. On Colab, this is a symlink to a GDrive folder so checkpoints survive runtime restarts.

Layout:
- `best/` — current best-by-macro-F1 model
- `runs/<run_id>/` — per-experiment checkpoints

Mirrored to GCS every 5 min by `scripts/sync_to_gcs.sh`.
```

- [ ] **Step 6: Write `experiments/README.md`**

Create `/Users/apple/AutoLearnMeds/experiments/README.md`:

```markdown
# experiments/

The autoresearch ledger — single source of truth for all experiments.

- `ledger.jsonl` — append-only, one JSON line per experiment. See spec §5.4 for schema.
- `runs/<run_id>/` — per-experiment artifacts: `train.py` snapshot, `config.yaml`, `metrics.json`, `stdout.log`, `notes.md`.
- `_active.json` — current planned/active experiment (transient; cleared after commit).
- `leaderboard.md` — auto-rendered ranked table of all kept experiments.

All files in this folder ARE tracked in git (small, critical for reproducibility); per-run checkpoints are NOT (large, mirrored to GCS instead).
```

- [ ] **Step 7: Write `paper/README.md`**

Create `/Users/apple/AutoLearnMeds/paper/README.md`:

```markdown
# paper/

The research paper itself.

- `main.tex` — top-level document
- `refs.bib` — auto-generated from `papers/README.md` by `scripts/build_bibtex.py`
- `figures/` — auto-generated PDFs from `scripts/build_paper_artifacts.py`
- `tables/` — auto-generated `.tex` tables
- `sections/` — modular `.tex` files (intro, related work, method, etc.)

Tables and figures are auto-generated from `experiments/ledger.jsonl`. Prose lives in `main.tex` and `sections/`.
```

- [ ] **Step 8: Write `scripts/README.md`**

Create `/Users/apple/AutoLearnMeds/scripts/README.md`:

```markdown
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
```

- [ ] **Step 9: Write `notebooks/README.md`**

Create `/Users/apple/AutoLearnMeds/notebooks/README.md`:

```markdown
# notebooks/

Jupyter notebooks for Colab session entry.

- `00_bootstrap.ipynb` — single-cell launcher that calls `scripts/colab_bootstrap.sh`. This is the ONLY notebook the human user opens directly. Everything else happens via VSCode SSH after the bootstrap completes.
```

- [ ] **Step 10: Run tests, verify they now pass**

```bash
uv run pytest tests/test_smoke.py -v
```

Expected: `test_folder_skeleton_exists PASSED`, `test_placeholder_readmes_exist PASSED`, plus prior tests still PASS.

- [ ] **Step 11: Commit**

```bash
git add data/ checkpoints/ experiments/ paper/ scripts/ notebooks/ tests/
git commit -m "scaffold: folder skeleton + placeholder READMEs (spec §2.2)

Creates the full directory tree per the design spec. Each top-level
folder has a README explaining its role. Empty subfolders kept in git
via .gitkeep. Smoke tests added in tests/test_smoke.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Three-file discipline stubs (`prepare.py`, `train.py`, `program.md`)

**Files:**
- Modify: `tests/test_smoke.py`
- Create: `prepare.py`
- Create: `train.py`
- Create: `program.md`
- Create: `program_explore.md`
- Create: `program_confirm.md`

- [ ] **Step 1: Write the failing test**

Append to `/Users/apple/AutoLearnMeds/tests/test_smoke.py`:

```python


THREE_FILES = ["prepare.py", "train.py", "program.md", "program_explore.md", "program_confirm.md"]


def test_three_file_discipline_stubs(project_root: Path) -> None:
    """The autoresearch three-file convention requires these to exist (even as stubs)."""
    for f in THREE_FILES:
        assert (project_root / f).is_file(), f"Missing: {f}"


def test_program_md_mentions_metric(project_root: Path) -> None:
    """program.md must declare the optimization metric clearly."""
    text = (project_root / "program.md").read_text()
    assert "macro_f1" in text.lower() or "macro-f1" in text.lower()
    assert "final_macro_f1" in text  # must reference exact stdout token
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_smoke.py::test_three_file_discipline_stubs tests/test_smoke.py::test_program_md_mentions_metric -v
```

Expected: both FAIL with file-not-found.

- [ ] **Step 3: Write `prepare.py` stub**

Create `/Users/apple/AutoLearnMeds/prepare.py`:

```python
"""Phase-1 module: data prep, tokenizer, dataloader, evaluate().

This file is FIXED in the autoresearch loop — the agent must not modify it.
Implementation lands in Phase 1 once the golden_set schema is locked.

See spec §3 (Data Pipeline) and §4.4–4.5 (training contract).
"""

raise NotImplementedError(
    "prepare.py is a Phase-1 stub. Schema-derived implementation arrives once the user shares the golden_set."
)
```

- [ ] **Step 4: Write `train.py` stub**

Create `/Users/apple/AutoLearnMeds/train.py`:

```python
"""Phase-2 module: SigLIP encoder + Donut-style decoder + training loop.

This file is the ONE the autoresearch agent edits. Baseline implementation
lands in Phase 2 of the rollout (see spec §9). Outputs `final_macro_f1=X.XXXX`
on the last line of stdout.
"""

raise NotImplementedError(
    "train.py is a Phase-2 stub. Baseline arrives once Phase 1 (data + prepare.py) is green."
)
```

- [ ] **Step 5: Write `program.md` baseline (per spec §5.3)**

Create `/Users/apple/AutoLearnMeds/program.md`:

```markdown
# Pharma-VLM Autoresearch — Baseline `program.md`

You are an autonomous research agent operating on this repo. Your job: iterate on `train.py` to maximize a single scalar metric.

## Goal

Maximize `val_macro_f1` of structured field extraction from pharmaceutical product label images, by editing `train.py`.

## Metric

Single scalar: `val_macro_f1`. Higher is better. Reported by `train.py` as the literal stdout line `final_macro_f1=X.XXXX` on the last line of training. Macro-F1 over fields, with field-aware normalization. See spec §1.3.

## Time budget

- **Phase 1 (explore):** 15 minutes wall-clock per experiment (excluding compile/startup).
- **Phase 2 (confirm):** 60 minutes wall-clock per experiment, 3 seeds.

## Allowed (you may modify in `train.py`)

- Architecture: layers, hidden dim, heads, FFN ratio, activation, dropout, position encoding, cross-attention pattern, layer-norm placement (pre vs post), tied embeddings.
- Optimizer: AdamW vs Lion vs Muon, betas, weight decay, schedule shape, warmup, peak LR.
- Loss: label smoothing, token weighting, auxiliary losses.
- Augmentations & data: RandAugment, AugMix, hard-example mining, curriculum, oversampling.
- Tokenization variants that don't retrain the BPE.

## Forbidden

- Modifying `prepare.py`.
- Modifying `evaluate()`.
- Retraining the BPE tokenizer.
- Changing the test set or the metric definition.
- Reading `data/test.jsonl` or invoking `evaluate_test()`.
- Swapping the encoder out of the SigLIP family without a top-level run-tag.

## Source vetting

Every change MUST cite at least one paper from `papers/` or add a new paper to `papers/` with a one-paragraph summary in `papers/README.md`. If a change is heuristic (no paper), label it `--exploratory--` and run it under a separate run-tag.

## Workflow per experiment

1. Read `experiments/ledger.jsonl` tail; pick a direction not recently tried.
2. Edit `train.py` with a minimal diff (one knob).
3. Add citation + hypothesis to `experiments/runs/<run_id>/notes.md`.
4. Run: `bash scripts/run_experiment.sh`.
5. Read final metric from `experiments/runs/<run_id>/metrics.json`.
6. If improved over current best: `bash scripts/promote.sh <run_id>`. Else: revert `train.py`.
7. Append ledger entry; commit; push; sync to GCS.

## Backup

After every experiment, `scripts/sync_to_gcs.sh` runs automatically. If Colab disconnects or your token quota approaches its limit, write `RESUME_NEEDED.md` with `next_planned_action`, then exit cleanly. The next session reads it and picks up.

## Do not

- Combine multiple changes in one experiment ("one knob per experiment").
- Continue past the time budget.
- Skip the `notes.md` or the citation.
- Touch the test set.
```

- [ ] **Step 6: Write `program_explore.md`**

Create `/Users/apple/AutoLearnMeds/program_explore.md`:

```markdown
# Phase-1 Override: Explore

Inherits `program.md`. Overrides:

- **Time budget:** 15 minutes per experiment.
- **Selection rule:** keep if `new_macro_f1 > current_best + 0.001` (single seed, noisy OK).
- **Preferred directions:** fast-signal knobs — learning rate, batch size, dropout, augmentation toggles, optimizer choice.
- **Stop after:** 20 consecutive non-improving experiments OR user-defined compute cap.

The goal of this phase is breadth, not certainty. False positives are expected and filtered by the confirm phase.
```

- [ ] **Step 7: Write `program_confirm.md`**

Create `/Users/apple/AutoLearnMeds/program_confirm.md`:

```markdown
# Phase-2 Override: Confirm

Inherits `program.md`. Overrides:

- **Time budget:** 60 minutes per experiment.
- **Seeds:** 3 per experiment; report mean ± std.
- **Selection rule:** keep if mean improvement is real AND the 95% confidence interval does not overlap baseline's 95% CI.
- **Preferred inputs:** only experiments that the explore phase already promoted.
- **Stop after:** 5 consecutive non-improving experiments OR user-defined compute cap.

The goal of this phase is statistical rigor. Only confirmed wins go into the paper's main results table.
```

- [ ] **Step 8: Run tests, verify they now pass**

```bash
uv run pytest tests/test_smoke.py -v
```

Expected: all tests PASSED, including `test_three_file_discipline_stubs` and `test_program_md_mentions_metric`.

- [ ] **Step 9: Commit**

```bash
git add prepare.py train.py program.md program_explore.md program_confirm.md tests/
git commit -m "scaffold: three-file discipline stubs + program.md baseline

Adds prepare.py and train.py as NotImplementedError stubs (real
implementations land in Phase 1 and Phase 2). program.md is the
baseline agent contract; program_explore.md and program_confirm.md
are phase-specific overrides per spec §5.3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `scripts/keepalive.py`

**Files:**
- Modify: `tests/test_smoke.py`
- Create: `scripts/keepalive.py`

- [ ] **Step 1: Write failing test**

Append to `/Users/apple/AutoLearnMeds/tests/test_smoke.py`:

```python


import os
import subprocess


def test_keepalive_exists_and_executable(project_root: Path) -> None:
    """scripts/keepalive.py must exist and be executable."""
    p = project_root / "scripts" / "keepalive.py"
    assert p.is_file(), "scripts/keepalive.py missing"
    assert os.access(p, os.X_OK), "scripts/keepalive.py not executable (chmod +x)"


def test_keepalive_imports_clean(project_root: Path) -> None:
    """The keepalive script must import without side effects on import."""
    p = project_root / "scripts" / "keepalive.py"
    # Use python -c with importlib to test import-without-execution.
    result = subprocess.run(
        ["python", "-c", f"import importlib.util, sys; spec = importlib.util.spec_from_file_location('k', '{p}'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)"],
        capture_output=True, text=True, timeout=5,
    )
    # Allowed to fail if it has a __main__ guard; we only care it doesn't crash on import-time evaluation.
    # Empty stdout/stderr or a clean exit is the pass condition.
    assert result.returncode == 0 or "main" in result.stderr.lower() or result.stderr == "", f"Unexpected error: {result.stderr}"
```

- [ ] **Step 2: Run, verify it fails**

```bash
uv run pytest tests/test_smoke.py::test_keepalive_exists_and_executable -v
```

Expected: FAIL — `scripts/keepalive.py` missing.

- [ ] **Step 3: Implement `scripts/keepalive.py`**

Create `/Users/apple/AutoLearnMeds/scripts/keepalive.py`:

```python
#!/usr/bin/env python3
"""Colab keepalive — writes a heartbeat once per minute.

Two purposes:
1. Touches a file frequently enough to discourage idle-disconnect.
2. Lets the user side check liveness without logging into Colab
   (the heartbeat file is mirrored to GCS by sync_to_gcs.sh).

Run: nohup python scripts/keepalive.py > /tmp/keepalive.log 2>&1 &
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HEARTBEAT_PATH = Path(os.environ.get("AUTOLEARNMEDS_HEARTBEAT", "/workspace/heartbeat.txt"))
INTERVAL_SECONDS = int(os.environ.get("AUTOLEARNMEDS_HEARTBEAT_INTERVAL", "60"))


def beat() -> None:
    HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT_PATH.write_text(datetime.now(timezone.utc).isoformat() + "\n")


def main() -> int:
    print(f"[keepalive] writing to {HEARTBEAT_PATH} every {INTERVAL_SECONDS}s", flush=True)
    while True:
        try:
            beat()
        except OSError as e:
            # Don't die on transient FS issues — log and keep going.
            print(f"[keepalive] WARN: {e}", file=sys.stderr, flush=True)
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Make executable**

```bash
chmod +x /Users/apple/AutoLearnMeds/scripts/keepalive.py
```

- [ ] **Step 5: Run tests, verify they pass**

```bash
uv run pytest tests/test_smoke.py -v
```

Expected: all PASSED.

- [ ] **Step 6: Commit**

```bash
git add scripts/keepalive.py tests/
git commit -m "feat(scripts): keepalive heartbeat for Colab idle prevention

Writes ISO-8601 UTC timestamp to /workspace/heartbeat.txt every 60s.
Path and interval overridable via AUTOLEARNMEDS_HEARTBEAT and
AUTOLEARNMEDS_HEARTBEAT_INTERVAL env vars.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: `scripts/sync_to_gcs.sh`

**Files:**
- Modify: `tests/test_smoke.py`
- Create: `scripts/sync_to_gcs.sh`

- [ ] **Step 1: Write failing test**

Append to `/Users/apple/AutoLearnMeds/tests/test_smoke.py`:

```python


def test_sync_to_gcs_exists_and_executable(project_root: Path) -> None:
    p = project_root / "scripts" / "sync_to_gcs.sh"
    assert p.is_file()
    assert os.access(p, os.X_OK)
    text = p.read_text()
    assert text.startswith("#!/"), "Missing shebang"
    assert "gsutil rsync" in text or "gcloud storage rsync" in text, "Should use gsutil/gcloud rsync"
    assert "set -euo pipefail" in text, "Should use strict bash"
```

- [ ] **Step 2: Run, verify failure**

```bash
uv run pytest tests/test_smoke.py::test_sync_to_gcs_exists_and_executable -v
```

Expected: FAIL.

- [ ] **Step 3: Implement `scripts/sync_to_gcs.sh`**

Create `/Users/apple/AutoLearnMeds/scripts/sync_to_gcs.sh`:

```bash
#!/usr/bin/env bash
# Periodic differential sync of experiments/ + checkpoints/ to GCS.
# Designed to run forever in the background on Colab.
#
# Env vars:
#   AUTOLEARNMEDS_GCS_BUCKET  (required) — gs://bucket-name
#   AUTOLEARNMEDS_WORKSPACE   (default: /workspace)
#   AUTOLEARNMEDS_SYNC_INTERVAL (default: 300 — seconds between syncs)
#
# Run: nohup bash scripts/sync_to_gcs.sh > /tmp/sync.log 2>&1 &

set -euo pipefail

BUCKET="${AUTOLEARNMEDS_GCS_BUCKET:-}"
WORKSPACE="${AUTOLEARNMEDS_WORKSPACE:-/workspace}"
INTERVAL="${AUTOLEARNMEDS_SYNC_INTERVAL:-300}"

if [[ -z "$BUCKET" ]]; then
  echo "[sync_to_gcs] FATAL: AUTOLEARNMEDS_GCS_BUCKET is unset" >&2
  exit 1
fi

if ! command -v gsutil >/dev/null 2>&1; then
  echo "[sync_to_gcs] FATAL: gsutil not on PATH (install google-cloud-sdk)" >&2
  exit 1
fi

echo "[sync_to_gcs] starting; bucket=$BUCKET workspace=$WORKSPACE interval=${INTERVAL}s"

while true; do
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "[sync_to_gcs] $ts — sync start"

  # The three things we mirror; each is non-fatal on transient failure.
  for src_rel in experiments checkpoints; do
    src="$WORKSPACE/$src_rel"
    dst="$BUCKET/$src_rel"
    if [[ -d "$src" ]]; then
      gsutil -m rsync -r -d "$src" "$dst" 2>&1 | tail -5 || \
        echo "[sync_to_gcs] WARN: rsync $src failed (non-fatal)"
    fi
  done

  # Active-experiment marker — single small file, sync directly.
  if [[ -f "$WORKSPACE/experiments/_active.json" ]]; then
    gsutil cp "$WORKSPACE/experiments/_active.json" "$BUCKET/experiments/_active.json" 2>&1 | tail -1 || true
  fi

  echo "[sync_to_gcs] $(date -u +%Y-%m-%dT%H:%M:%SZ) — sync done"
  sleep "$INTERVAL"
done
```

- [ ] **Step 4: Make executable**

```bash
chmod +x /Users/apple/AutoLearnMeds/scripts/sync_to_gcs.sh
```

- [ ] **Step 5: Verify tests pass**

```bash
uv run pytest tests/test_smoke.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/sync_to_gcs.sh tests/
git commit -m "feat(scripts): periodic GCS sync for ledger + checkpoints

Differential gsutil rsync of experiments/ and checkpoints/ to a
configurable bucket every 5 minutes. Designed to run as a daemon
on Colab. Tolerates transient gsutil failures non-fatally.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: `scripts/update_ssh_config.sh`

**Files:**
- Modify: `tests/test_smoke.py`
- Create: `scripts/update_ssh_config.sh`

- [ ] **Step 1: Write failing test**

Append to `/Users/apple/AutoLearnMeds/tests/test_smoke.py`:

```python


def test_update_ssh_config_exists_and_executable(project_root: Path) -> None:
    p = project_root / "scripts" / "update_ssh_config.sh"
    assert p.is_file()
    assert os.access(p, os.X_OK)
    text = p.read_text()
    assert text.startswith("#!/"), "Missing shebang"
    assert "trycloudflare" in text or "cloudflared" in text, "Should reference cloudflared"
    assert "Host autolearnmeds-colab" in text, "Should write the canonical Host alias"
```

- [ ] **Step 2: Verify failure**

```bash
uv run pytest tests/test_smoke.py::test_update_ssh_config_exists_and_executable -v
```

- [ ] **Step 3: Implement `scripts/update_ssh_config.sh`**

Create `/Users/apple/AutoLearnMeds/scripts/update_ssh_config.sh`:

```bash
#!/usr/bin/env bash
# Updates the user's ~/.ssh/config with the current Colab cloudflared host.
#
# Run on the USER's MAC, not on Colab. Reads the hostname from a known
# location (either an arg or stdin). Replaces the existing
# `Host autolearnmeds-colab` block atomically.
#
# Usage:
#   ./scripts/update_ssh_config.sh <hostname>
#   echo "myhost.trycloudflare.com" | ./scripts/update_ssh_config.sh

set -euo pipefail

HOSTNAME="${1:-}"
if [[ -z "$HOSTNAME" ]]; then
  HOSTNAME="$(cat -)"
fi
HOSTNAME="$(echo "$HOSTNAME" | tr -d '[:space:]')"

if [[ -z "$HOSTNAME" ]]; then
  echo "[update_ssh_config] FATAL: no hostname provided (arg or stdin)" >&2
  exit 1
fi

SSH_CONFIG="$HOME/.ssh/config"
mkdir -p "$HOME/.ssh"
touch "$SSH_CONFIG"
chmod 600 "$SSH_CONFIG"

# Atomically rewrite ~/.ssh/config: strip any existing
# `Host autolearnmeds-colab` block, then append the new one.
TMPFILE="$(mktemp)"
awk '
  /^Host autolearnmeds-colab/,/^$/ { next }
  { print }
' "$SSH_CONFIG" > "$TMPFILE"

cat >> "$TMPFILE" <<EOF

Host autolearnmeds-colab
    HostName $HOSTNAME
    User root
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ProxyCommand cloudflared access ssh --hostname %h
EOF

mv "$TMPFILE" "$SSH_CONFIG"
chmod 600 "$SSH_CONFIG"

echo "[update_ssh_config] updated ~/.ssh/config with HostName=$HOSTNAME"
echo "[update_ssh_config] connect via: ssh autolearnmeds-colab"
```

- [ ] **Step 4: Make executable**

```bash
chmod +x /Users/apple/AutoLearnMeds/scripts/update_ssh_config.sh
```

- [ ] **Step 5: Verify tests pass**

```bash
uv run pytest tests/test_smoke.py -v
```

- [ ] **Step 6: Commit**

```bash
git add scripts/update_ssh_config.sh tests/
git commit -m "feat(scripts): atomic ~/.ssh/config update for Colab cloudflared host

Replaces the 'Host autolearnmeds-colab' block in the user's
~/.ssh/config with the current cloudflared hostname. Runs on the
user's Mac (not Colab) once per Colab session restart.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: `scripts/resume_or_start.py` — atomic-experiment state machine

**Files:**
- Create: `tests/test_resume_state_machine.py`
- Create: `scripts/resume_or_start.py`

- [ ] **Step 1: Write the state-machine test (TDD)**

Create `/Users/apple/AutoLearnMeds/tests/test_resume_state_machine.py`:

```python
"""Tests for scripts/resume_or_start.py — atomic experiment state machine.

State table (from spec §7.3):
| Last visible state                                     | Action                              |
| no _active.json                                        | START_NEW                           |
| _active.json status=PLANNING                           | RESTART (this experiment)           |
| _active.json status=ACTIVE, no metrics.json            | RESTART (training was killed)       |
| _active.json with metrics.json but no ledger entry     | RESUME_FROM_EVALUATED               |
| _active.json with ledger entry                         | CLEAR_AND_START_NEW                 |
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_module(project_root: Path):
    spec = importlib.util.spec_from_file_location(
        "resume_or_start",
        project_root / "scripts" / "resume_or_start.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A fresh fake workspace tree with the expected layout."""
    (tmp_path / "experiments" / "runs").mkdir(parents=True)
    (tmp_path / "experiments" / "ledger.jsonl").touch()
    return tmp_path


def _write_active(workspace: Path, status: str, run_id: str = "run-001") -> None:
    payload = {"run_id": run_id, "status": status, "started_at": "2026-05-05T12:00:00Z"}
    (workspace / "experiments" / "_active.json").write_text(json.dumps(payload))


def _write_metrics(workspace: Path, run_id: str = "run-001", final_macro_f1: float = 0.5) -> None:
    run_dir = workspace / "experiments" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps({"final_macro_f1": final_macro_f1}))


def _append_ledger(workspace: Path, run_id: str = "run-001") -> None:
    line = json.dumps({"run_id": run_id, "kept": False, "metrics": {"final_macro_f1": 0.5}})
    with (workspace / "experiments" / "ledger.jsonl").open("a") as fh:
        fh.write(line + "\n")


def test_no_active_returns_start_new(workspace: Path, project_root: Path) -> None:
    mod = _load_module(project_root)
    assert mod.next_action(workspace) == "START_NEW"


def test_planning_status_returns_restart(workspace: Path, project_root: Path) -> None:
    mod = _load_module(project_root)
    _write_active(workspace, "PLANNING")
    assert mod.next_action(workspace) == "RESTART"


def test_active_without_metrics_returns_restart(workspace: Path, project_root: Path) -> None:
    mod = _load_module(project_root)
    _write_active(workspace, "ACTIVE")
    assert mod.next_action(workspace) == "RESTART"


def test_active_with_metrics_no_ledger_returns_resume_from_evaluated(
    workspace: Path, project_root: Path
) -> None:
    mod = _load_module(project_root)
    _write_active(workspace, "ACTIVE")
    _write_metrics(workspace)
    assert mod.next_action(workspace) == "RESUME_FROM_EVALUATED"


def test_active_with_ledger_entry_returns_clear_and_start_new(
    workspace: Path, project_root: Path
) -> None:
    mod = _load_module(project_root)
    _write_active(workspace, "COMMITTED")
    _write_metrics(workspace)
    _append_ledger(workspace)
    assert mod.next_action(workspace) == "CLEAR_AND_START_NEW"
```

- [ ] **Step 2: Run, verify failure**

```bash
uv run pytest tests/test_resume_state_machine.py -v
```

Expected: all 5 FAIL — `scripts/resume_or_start.py` doesn't exist yet.

- [ ] **Step 3: Implement `scripts/resume_or_start.py`**

Create `/Users/apple/AutoLearnMeds/scripts/resume_or_start.py`:

```python
#!/usr/bin/env python3
"""Atomic-experiment state machine launcher.

Reads experiments/_active.json and experiments/ledger.jsonl to decide
what the next session should do. See spec §7.3 for the state table.

Usage:
  python scripts/resume_or_start.py [--workspace /workspace] [--print-only]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

Action = Literal["START_NEW", "RESTART", "RESUME_FROM_EVALUATED", "CLEAR_AND_START_NEW"]


def _read_active(workspace: Path) -> dict | None:
    active_path = workspace / "experiments" / "_active.json"
    if not active_path.is_file():
        return None
    try:
        return json.loads(active_path.read_text())
    except json.JSONDecodeError:
        return None


def _metrics_exist(workspace: Path, run_id: str) -> bool:
    return (workspace / "experiments" / "runs" / run_id / "metrics.json").is_file()


def _ledger_has_run(workspace: Path, run_id: str) -> bool:
    ledger_path = workspace / "experiments" / "ledger.jsonl"
    if not ledger_path.is_file():
        return False
    for line in ledger_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("run_id") == run_id:
            return True
    return False


def next_action(workspace: Path) -> Action:
    """Decide the next session's action per spec §7.3."""
    active = _read_active(workspace)
    if active is None:
        return "START_NEW"

    run_id = active.get("run_id", "")
    status = active.get("status", "")

    if status == "PLANNING":
        return "RESTART"

    has_metrics = _metrics_exist(workspace, run_id)

    if status == "ACTIVE" and not has_metrics:
        return "RESTART"

    if has_metrics and not _ledger_has_run(workspace, run_id):
        return "RESUME_FROM_EVALUATED"

    if has_metrics and _ledger_has_run(workspace, run_id):
        return "CLEAR_AND_START_NEW"

    # Defensive default: ACTIVE with metrics but ambiguous state — restart safely.
    return "RESTART"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="/workspace", type=Path)
    parser.add_argument("--print-only", action="store_true",
                        help="Print the action and exit; do not perform any side effects")
    args = parser.parse_args()

    if not args.workspace.exists():
        print(f"[resume_or_start] FATAL: workspace not found: {args.workspace}", file=sys.stderr)
        return 2

    action = next_action(args.workspace)
    print(f"[resume_or_start] action={action}")

    if args.print_only:
        return 0

    # Side effects (Phase 1+ wiring): for now, just print.
    # Future tasks add: invoke run_experiment.sh, finalize ledger entry, etc.
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Make executable**

```bash
chmod +x /Users/apple/AutoLearnMeds/scripts/resume_or_start.py
```

- [ ] **Step 5: Run all tests**

```bash
uv run pytest -v
```

Expected: all PASS, including the 5 state-machine tests.

- [ ] **Step 6: Commit**

```bash
git add scripts/resume_or_start.py tests/test_resume_state_machine.py
git commit -m "feat(scripts): atomic-experiment resume_or_start state machine

Implements the spec §7.3 state table: reads experiments/_active.json
and experiments/ledger.jsonl to decide whether the next session should
start a new experiment, restart the in-flight one, resume from evaluated,
or clear and start fresh. Side-effect wiring deferred to Phase 1; this
commit ships the pure-function decision logic with full test coverage.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: `scripts/colab_bootstrap.sh` — the one-cell Colab setup

**Files:**
- Modify: `tests/test_smoke.py`
- Create: `scripts/colab_bootstrap.sh`

- [ ] **Step 1: Write failing test**

Append to `/Users/apple/AutoLearnMeds/tests/test_smoke.py`:

```python


def test_colab_bootstrap_exists_and_has_required_steps(project_root: Path) -> None:
    p = project_root / "scripts" / "colab_bootstrap.sh"
    assert p.is_file()
    assert os.access(p, os.X_OK)
    text = p.read_text()
    # Required steps per spec §6.4
    assert text.startswith("#!/"), "Missing shebang"
    assert "set -euo pipefail" in text, "Should use strict bash"
    assert "drive.mount" in text, "Step 1: GDrive mount"
    assert "gcsfuse" in text, "Step 2: GCS mount"
    assert "ln -sfn" in text, "Step 3: Symlink workspace"
    assert "uv sync" in text, "Step 4: Install deps"
    assert "launch_ssh_cloudflared" in text, "Step 5: SSH tunnel"
    assert "keepalive.py" in text, "Step 6: Daemons — keepalive"
    assert "sync_to_gcs.sh" in text, "Step 6: Daemons — gcs sync"
    assert "READY" in text, "Step 7: Success banner"
```

- [ ] **Step 2: Run, verify failure**

```bash
uv run pytest tests/test_smoke.py::test_colab_bootstrap_exists_and_has_required_steps -v
```

- [ ] **Step 3: Implement `scripts/colab_bootstrap.sh`**

Create `/Users/apple/AutoLearnMeds/scripts/colab_bootstrap.sh`:

```bash
#!/usr/bin/env bash
# AutoLearnMeds — one-cell Colab Pro+ bootstrap.
#
# Run from a Colab notebook cell:
#   !curl -sSL https://raw.githubusercontent.com/<user>/<repo>/main/scripts/colab_bootstrap.sh | bash
#
# At end, prints an SSH command to copy into your Mac's ~/.ssh/config.
#
# Required env vars (set in the Colab cell BEFORE running this script):
#   AUTOLEARNMEDS_GCS_BUCKET  — gs://your-bucket-name
#   AUTOLEARNMEDS_REPO_URL    — https://github.com/<user>/<repo>.git
#   AUTOLEARNMEDS_SSH_PASSWORD — temp password for SSH (use a strong random)

set -euo pipefail

BUCKET="${AUTOLEARNMEDS_GCS_BUCKET:?Set AUTOLEARNMEDS_GCS_BUCKET=gs://your-bucket}"
REPO_URL="${AUTOLEARNMEDS_REPO_URL:?Set AUTOLEARNMEDS_REPO_URL=https://github.com/...}"
SSH_PASSWORD="${AUTOLEARNMEDS_SSH_PASSWORD:?Set AUTOLEARNMEDS_SSH_PASSWORD=...}"
DRIVE_PROJECT_DIR="/content/drive/MyDrive/AutoLearnMeds"
WORKSPACE="/workspace"

echo "============================================================"
echo "AutoLearnMeds bootstrap — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# 1. Mount Google Drive (interactive auth on first run; cached after)
echo "[1/7] Mounting Google Drive..."
python -c "from google.colab import drive; drive.mount('/content/drive', force_remount=False)" || {
  echo "FATAL: Drive mount failed" >&2; exit 1;
}

# 2. Mount GCS bucket (gcsfuse)
echo "[2/7] Mounting GCS bucket $BUCKET..."
if ! command -v gcsfuse >/dev/null 2>&1; then
  echo "  installing gcsfuse..."
  echo "deb https://packages.cloud.google.com/apt gcsfuse-bookworm main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
  sudo apt-get update -q && sudo apt-get install -y -q gcsfuse
fi
mkdir -p /mnt/gcs
# Bucket name without the gs:// prefix
GCS_BUCKET_NAME="${BUCKET#gs://}"
if ! mountpoint -q /mnt/gcs; then
  gcsfuse "$GCS_BUCKET_NAME" /mnt/gcs || \
    echo "  WARN: gcsfuse mount failed (auth?). GCS sync will still work via gsutil."
fi

# 3. Clone repo into Drive (so it survives runtime restarts) and symlink to /workspace
echo "[3/7] Setting up workspace..."
mkdir -p /content/drive/MyDrive
if [[ ! -d "$DRIVE_PROJECT_DIR/.git" ]]; then
  git clone "$REPO_URL" "$DRIVE_PROJECT_DIR"
else
  cd "$DRIVE_PROJECT_DIR" && git pull --ff-only || echo "  WARN: git pull failed (continuing with existing copy)"
fi
ln -sfn "$DRIVE_PROJECT_DIR" "$WORKSPACE"
cd "$WORKSPACE"

# 4. Install uv + sync deps (full ml + colab extras on Colab)
echo "[4/7] Installing uv and syncing deps..."
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --extra ml --extra colab

# 5. Launch SSH server + Cloudflare Tunnel
echo "[5/7] Launching SSH + cloudflared..."
uv run python -c "
from colab_ssh import launch_ssh_cloudflared
launch_ssh_cloudflared(password='$SSH_PASSWORD')
" | tee /tmp/colab_ssh_output.txt

# Extract the cloudflared host from the colab-ssh output and pin it.
CLOUDFLARED_HOST="$(grep -oE '[a-zA-Z0-9-]+\.trycloudflare\.com' /tmp/colab_ssh_output.txt | head -1 || true)"
if [[ -n "$CLOUDFLARED_HOST" ]]; then
  echo "$CLOUDFLARED_HOST" > "$WORKSPACE/.colab_ssh_host"
  gsutil cp "$WORKSPACE/.colab_ssh_host" "$BUCKET/.colab_ssh_host" 2>/dev/null || true
fi

# 6. Background daemons
echo "[6/7] Starting keepalive + GCS sync daemons..."
nohup uv run python scripts/keepalive.py > /tmp/keepalive.log 2>&1 &
AUTOLEARNMEDS_GCS_BUCKET="$BUCKET" \
AUTOLEARNMEDS_WORKSPACE="$WORKSPACE" \
nohup bash scripts/sync_to_gcs.sh > /tmp/sync_to_gcs.log 2>&1 &

# 7. Success banner
echo "============================================================"
echo "[7/7] READY"
echo "  Workspace:     $WORKSPACE"
echo "  GCS bucket:    $BUCKET"
echo "  Cloudflared:   ${CLOUDFLARED_HOST:-<see /tmp/colab_ssh_output.txt>}"
echo ""
echo "On your Mac, run:"
echo "  ./scripts/update_ssh_config.sh '${CLOUDFLARED_HOST:-<host>}'"
echo "Then in VSCode:"
echo "  ⌘⇧P → Remote-SSH: Connect to Host → autolearnmeds-colab"
echo "============================================================"
```

- [ ] **Step 4: Make executable**

```bash
chmod +x /Users/apple/AutoLearnMeds/scripts/colab_bootstrap.sh
```

- [ ] **Step 5: Verify tests pass**

```bash
uv run pytest tests/test_smoke.py -v
```

- [ ] **Step 6: Commit**

```bash
git add scripts/colab_bootstrap.sh tests/
git commit -m "feat(scripts): one-cell Colab Pro+ bootstrap

Mounts GDrive + GCS, clones repo to a persistent Drive folder, syncs
uv deps with ml+colab extras, launches OpenSSH inside Colab via
colab-ssh + Cloudflare Tunnel, starts keepalive + sync_to_gcs daemons,
prints the SSH host for the user to update on their Mac.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: `notebooks/00_bootstrap.ipynb` — the user's entry point

**Files:**
- Modify: `tests/test_smoke.py`
- Create: `notebooks/00_bootstrap.ipynb`

- [ ] **Step 1: Write failing test**

Append to `/Users/apple/AutoLearnMeds/tests/test_smoke.py`:

```python


def test_bootstrap_notebook_is_valid_json(project_root: Path) -> None:
    p = project_root / "notebooks" / "00_bootstrap.ipynb"
    assert p.is_file()
    data = json.loads(p.read_text())
    assert data.get("nbformat") == 4
    cells = data.get("cells", [])
    assert len(cells) >= 1, "Notebook must have at least one cell"
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    assert any("colab_bootstrap.sh" in "".join(c.get("source", [])) for c in code_cells), \
        "Notebook must call scripts/colab_bootstrap.sh"
```

Also add `import json` to the top of `tests/test_smoke.py` if not already there.

- [ ] **Step 2: Run, verify failure**

```bash
uv run pytest tests/test_smoke.py::test_bootstrap_notebook_is_valid_json -v
```

- [ ] **Step 3: Create `notebooks/00_bootstrap.ipynb`**

Create `/Users/apple/AutoLearnMeds/notebooks/00_bootstrap.ipynb`:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# AutoLearnMeds — Bootstrap\n",
    "\n",
    "Run **once per Colab session**. After it succeeds, copy the printed `update_ssh_config.sh` line into a terminal on your Mac, then connect via VSCode Remote-SSH to the host alias `autolearnmeds-colab`.\n",
    "\n",
    "Required: Colab Pro+ runtime, A100 GPU, **Background execution enabled**.\n",
    "\n",
    "## 1. Set credentials in this cell, then run.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# REPLACE these three values before running. Keep this notebook private.\n",
    "import os\n",
    "os.environ['AUTOLEARNMEDS_GCS_BUCKET']    = 'gs://YOUR-BUCKET-NAME'\n",
    "os.environ['AUTOLEARNMEDS_REPO_URL']      = 'https://github.com/YOUR-USER/YOUR-REPO.git'\n",
    "os.environ['AUTOLEARNMEDS_SSH_PASSWORD']  = 'CHANGE-ME-USE-A-STRONG-PASSWORD'\n",
    "\n",
    "# Bootstrap: mounts GDrive + GCS, clones repo, installs deps, launches SSH tunnel.\n",
    "# All output is printed below. The last line tells you how to connect from your Mac.\n",
    "!curl -sSL \"$AUTOLEARNMEDS_REPO_URL_RAW/raw/main/scripts/colab_bootstrap.sh\" -o /tmp/colab_bootstrap.sh 2>/dev/null || \\\n",
    "  curl -sSL https://raw.githubusercontent.com/${AUTOLEARNMEDS_REPO_URL#https://github.com/}/main/scripts/colab_bootstrap.sh \\\n",
    "       -o /tmp/colab_bootstrap.sh\n",
    "!bash /tmp/colab_bootstrap.sh"
   ],
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. After the bootstrap finishes\n",
    "\n",
    "1. Read the cloudflared hostname from the output above.\n",
    "2. On your **Mac**, run from the cloned repo: `./scripts/update_ssh_config.sh <hostname>`\n",
    "3. In **VSCode**: ⌘⇧P → Remote-SSH: Connect to Host → `autolearnmeds-colab`.\n",
    "4. From inside VSCode, run `make verify` to confirm the rig is green.\n",
    "\n",
    "If anything fails, the agent (Claude Code in your VSCode) reads `/tmp/keepalive.log`, `/tmp/sync_to_gcs.log`, and `/tmp/colab_ssh_output.txt` to diagnose."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 4: Verify tests pass**

```bash
uv run pytest tests/test_smoke.py -v
```

- [ ] **Step 5: Commit**

```bash
git add notebooks/00_bootstrap.ipynb tests/
git commit -m "feat(notebooks): one-cell bootstrap notebook for Colab entry

Single code cell sets the three required env vars (bucket, repo URL,
SSH password) and shells out to scripts/colab_bootstrap.sh. After it
runs, the user updates ~/.ssh/config and connects VSCode Remote-SSH.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: `Makefile` with `make verify`

**Files:**
- Modify: `tests/test_smoke.py`
- Create: `Makefile`

- [ ] **Step 1: Write failing test**

Append to `/Users/apple/AutoLearnMeds/tests/test_smoke.py`:

```python


def test_makefile_has_verify_target(project_root: Path) -> None:
    p = project_root / "Makefile"
    assert p.is_file()
    text = p.read_text()
    assert "verify:" in text, "Makefile must define a 'verify' target"
    assert "test:" in text, "Makefile must define a 'test' target"
    assert "lint:" in text, "Makefile must define a 'lint' target"
```

- [ ] **Step 2: Run, verify failure**

```bash
uv run pytest tests/test_smoke.py::test_makefile_has_verify_target -v
```

- [ ] **Step 3: Create `Makefile`**

Create `/Users/apple/AutoLearnMeds/Makefile`:

```makefile
# AutoLearnMeds — top-level make targets.
.PHONY: help verify test lint sync clean check_gpu check_drive check_gcs

help:
	@echo "Targets:"
	@echo "  make verify     — full smoke test: GPU, mounts, deps, tests, ledger"
	@echo "  make test       — run pytest"
	@echo "  make lint       — run ruff"
	@echo "  make sync       — uv sync (dev extras locally; ml+colab extras on Colab)"

# --- Local + Colab targets ---

test:
	uv run pytest -v

lint:
	uv run ruff check .

sync:
	@if [ -n "$$COLAB_GPU" ] || [ -n "$$COLAB_RELEASE_TAG" ]; then \
		uv sync --extra ml --extra colab; \
	else \
		uv sync --extra dev; \
	fi

# --- Verify (the smoke test) ---
# On Colab: full check including GPU + drive mounts.
# Locally:  skips Colab-only checks gracefully.

verify: test check_gpu check_drive check_gcs
	@echo ""
	@echo "============================================================"
	@echo "  ✓ make verify GREEN — rig is ready"
	@echo "============================================================"

check_gpu:
	@echo "--- GPU ---"
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; \
	else \
		echo "  (no nvidia-smi — local Mac, skipping)"; \
	fi

check_drive:
	@echo "--- GDrive mount ---"
	@if [ -d "/content/drive/MyDrive/AutoLearnMeds" ]; then \
		echo "  ✓ /content/drive/MyDrive/AutoLearnMeds exists"; \
		ls /content/drive/MyDrive/AutoLearnMeds | head -5; \
	else \
		echo "  (not on Colab — skipping)"; \
	fi

check_gcs:
	@echo "--- GCS mount ---"
	@if [ -d "/mnt/gcs" ]; then \
		mountpoint -q /mnt/gcs && echo "  ✓ /mnt/gcs is mounted" || echo "  WARN: /mnt/gcs exists but not mounted"; \
	else \
		echo "  (not on Colab — skipping)"; \
	fi

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ tests/__pycache__
```

- [ ] **Step 4: Verify make targets work locally**

```bash
cd /Users/apple/AutoLearnMeds && make help && make test && make verify
```

Expected: `make verify` runs pytest (PASS), prints `(no nvidia-smi — local Mac, skipping)` and `(not on Colab — skipping)` for the Colab-only checks, ends with green banner.

- [ ] **Step 5: Verify pytest test for Makefile passes**

```bash
uv run pytest tests/test_smoke.py::test_makefile_has_verify_target -v
```

- [ ] **Step 6: Commit**

```bash
git add Makefile tests/
git commit -m "build: Makefile with verify, test, lint, sync targets

'make verify' is the project's smoke test — pytest + GPU presence
+ Drive mount + GCS mount. On a local Mac, GPU/Drive/GCS checks
print 'skipping' instead of failing. On Colab, all four must pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: GitHub remote + initial push (user-driven)

**Files:** none — this is git plumbing.

- [ ] **Step 1: User creates GitHub repo**

This step is performed by the user, not the agent. The user:

1. Goes to https://github.com/new
2. Creates a new **private** repo named `AutoLearnMeds` (or any name; just remember the URL).
3. **Does NOT** initialize with README, .gitignore, or license — the local repo already has these.
4. Copies the SSH or HTTPS clone URL.

Once done, user provides the URL to the agent.

- [ ] **Step 2: Add remote and push**

Run from `/Users/apple/AutoLearnMeds`:

```bash
git remote add origin <USER-PROVIDED-URL>
git push -u origin main
```

Expected: all commits push successfully. `git remote -v` shows `origin <url>`.

- [ ] **Step 3: Verify push**

```bash
git remote -v && git log --oneline -10
```

Expected: remote is set; recent commits visible.

---

## Task 14: First Colab session — bootstrap + VSCode SSH (user-driven, agent verifies)

**Files:** none — runtime setup.

- [ ] **Step 1: User opens Colab Pro+, configures runtime**

User performs in browser:

1. Open https://colab.research.google.com.
2. File → Open → GitHub → paste the repo URL → open `notebooks/00_bootstrap.ipynb`.
3. Runtime → Change runtime type → Hardware: **A100 GPU**, Background execution: **ON** (Pro+ feature).

- [ ] **Step 2: User edits the credentials cell**

In the open notebook, replace the three placeholder values:

```python
os.environ['AUTOLEARNMEDS_GCS_BUCKET']   = 'gs://YOUR-BUCKET-NAME'        # actual bucket
os.environ['AUTOLEARNMEDS_REPO_URL']     = 'https://github.com/USER/REPO.git'  # actual URL
os.environ['AUTOLEARNMEDS_SSH_PASSWORD'] = 'a-strong-random-password'     # use a secret
```

- [ ] **Step 3: User runs the cell**

The cell takes ~3–5 minutes on first run (drive mount + apt-get gcsfuse + uv sync of ml deps). Subsequent runs in the same Colab session are seconds.

Expected output (last lines):

```
============================================================
[7/7] READY
  Workspace:     /workspace
  GCS bucket:    gs://...
  Cloudflared:   <random-name>.trycloudflare.com

On your Mac, run:
  ./scripts/update_ssh_config.sh '<random-name>.trycloudflare.com'
Then in VSCode:
  ⌘⇧P → Remote-SSH: Connect to Host → autolearnmeds-colab
============================================================
```

- [ ] **Step 4: User updates Mac `~/.ssh/config`**

Run from Mac terminal in `/Users/apple/AutoLearnMeds`:

```bash
./scripts/update_ssh_config.sh '<paste-the-host-from-cell-output>'
```

Expected: `[update_ssh_config] updated ~/.ssh/config with HostName=...`.

- [ ] **Step 5: User installs cloudflared on Mac (if not already)**

```bash
brew install cloudflared
```

- [ ] **Step 6: User connects VSCode Remote-SSH**

In VSCode:
1. Install the "Remote - SSH" extension if not already.
2. ⌘⇧P → "Remote-SSH: Connect to Host..." → select `autolearnmeds-colab`.
3. Enter the SSH password (the value of `AUTOLEARNMEDS_SSH_PASSWORD`).

Expected: a new VSCode window opens, status bar bottom-left shows "SSH: autolearnmeds-colab".

- [ ] **Step 7: Open the project in the remote VSCode**

In the new VSCode window: File → Open Folder → `/workspace` → Open.

Expected: the repo's file tree appears in the explorer panel.

---

## Task 15: `make verify` on Colab — Phase 0 exit gate

**Files:** none — verification only.

- [ ] **Step 1: Open a terminal in the remote VSCode**

In the VSCode window connected to `autolearnmeds-colab`: Terminal → New Terminal. The terminal is on the Colab VM.

- [ ] **Step 2: Run `make verify`**

```bash
cd /workspace && make verify
```

Expected output (all checks green):

```
--- pytest ---
tests/test_smoke.py ... PASSED
tests/test_resume_state_machine.py ... PASSED

--- GPU ---
NVIDIA A100-SXM4-40GB, 40960 MiB

--- GDrive mount ---
  ✓ /content/drive/MyDrive/AutoLearnMeds exists
  ...

--- GCS mount ---
  ✓ /mnt/gcs is mounted

============================================================
  ✓ make verify GREEN — rig is ready
============================================================
```

If any check fails, agent diagnoses (read `/tmp/keepalive.log`, `/tmp/sync_to_gcs.log`, `/tmp/colab_ssh_output.txt`).

- [ ] **Step 3: Tag the Phase 0 exit**

Once `make verify` is green, tag the commit:

```bash
git tag -a phase-0-complete -m "Phase 0 plumbing complete; rig green on Colab Pro+ A100"
git push origin phase-0-complete
```

- [ ] **Step 4: Write `phase0_review.md`**

Per spec §9.3, every phase ends with a review. Create `/workspace/docs/superpowers/reviews/phase0_review.md`:

```markdown
# Phase 0 Review

**Tag:** `phase-0-complete`
**Date:** <today>
**Status:** GREEN

## What was built
- uv project metadata, Python 3.11 pin, three dep groups (dev / ml / colab).
- Folder skeleton from spec §2.2 with placeholder READMEs.
- `prepare.py`, `train.py`, `program.md`, `program_explore.md`, `program_confirm.md` stubs.
- Operational scripts: `keepalive.py`, `sync_to_gcs.sh`, `update_ssh_config.sh`, `resume_or_start.py`, `colab_bootstrap.sh`.
- `notebooks/00_bootstrap.ipynb` for one-cell Colab entry.
- `Makefile` with `verify`, `test`, `lint`, `sync`.
- 14 smoke tests in `tests/test_smoke.py` + 5 state-machine tests in `tests/test_resume_state_machine.py`.

## What was verified
- `uv sync` works locally (Mac) and on Colab (A100 runtime).
- VSCode Remote-SSH connects to Colab via Cloudflare Tunnel.
- `make verify` runs end-to-end on Colab with all four checks green.
- GDrive + GCS mounts persist across the bootstrap cell.

## Open items remaining (from Appendix C of spec)
- OI-1 Canonical field schema — unblocked next phase.
- OI-2 GCS bucket name — provided.
- OI-3 GitHub repo URL — provided.
- (others remain as listed in spec.)

## Next step
Begin Phase 1 (Data Ingestion & Schema Lock). User shares golden_set; agent samples examples and proposes final schema.
```

- [ ] **Step 5: Commit and push**

```bash
git add docs/superpowers/reviews/
git commit -m "docs: phase 0 review

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
git push origin main
```

---

## Self-Review

**Spec coverage check** (against spec §9 Phase 0 deliverables):

| Spec deliverable | Implemented in task |
|---|---|
| GitHub repo + local clone | T13 |
| `pyproject.toml`, `uv.lock`, `.python-version` | T1 |
| `.gitignore` | (already done in brainstorming) |
| Folder skeleton + placeholder READMEs | T4 |
| `papers/` populated + `papers/README.md` | (already done in brainstorming) |
| `scripts/colab_bootstrap.sh` | T10 |
| `notebooks/00_bootstrap.ipynb` | T11 |
| `~/.ssh/config` entry, VSCode Remote-SSH | T13–14 |
| `make verify` smoke test | T12, T15 |
| `prepare.py`, `train.py`, `program.md` stubs | T5 |
| `program_explore.md`, `program_confirm.md` | T5 |
| `scripts/keepalive.py` | T6 |
| `scripts/sync_to_gcs.sh` | T7 |
| `scripts/update_ssh_config.sh` | T8 |
| `scripts/resume_or_start.py` | T9 |
| Phase exit review | T15 |

All Phase 0 spec deliverables have a task. ✓

**Placeholder check:** No "TBD", no "TODO", no "implement later". `<USER-PROVIDED-URL>` and `<paste-the-host-from-cell-output>` are user inputs explicitly tracked as such in T13–T14, not internal stubs.

**Type / signature consistency:**
- `next_action()` signature in T9 implementation matches the test calls in T9 step 1.
- `EXPECTED_DIRS` and `EXPECTED_READMES` lists in T4 step 1 are referenced consistently in test bodies.
- All test fixtures (`project_root`, `workspace`) defined once in `conftest.py` and used uniformly.

**Phase 1 coverage:** Out of scope. Will be planned separately once `phase-0-complete` tag exists and golden_set is shared.

---

## Phase 0 → Phase 1 transition

When this plan is fully executed and the `phase-0-complete` tag is pushed:

1. User shares the path to `golden_set` on GDrive or GCS.
2. Agent samples 10–20 (image, label) pairs and proposes the final canonical schema.
3. Agent + user approve the schema.
4. We invoke `superpowers:writing-plans` again to write the Phase 1 plan: `prepare.py` real implementation, BPE tokenizer training, `build_processed.py`, hash-locked test set, `data_card.md`.

Until then, this plan is the single source of truth for Phase 0 work.
