# Phase 4a — Explore-Phase Rails Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the autoresearch ledger + per-experiment housekeeping infrastructure so the agent (Claude Code reading `program.md`) can sustain ~30 explore experiments without manual bookkeeping. After this plan: the agent picks a hypothesis, edits `train.py`, runs `bash scripts/run_experiment.sh && bash scripts/finalize_experiment.sh`, and the system handles ledger append, leaderboard regen, promote-if-kept, and commit.

**Architecture:** `program.md` becomes the agent's full instruction set (vs. the Phase-0 skeleton). A new `scripts/append_ledger.py` reads `metrics.json` + per-run `notes.md` after each run, decides keep-vs-revert based on the current `best_at_time_of_run`, and writes a complete ledger line per spec §5.4. `scripts/leaderboard.py` regenerates `experiments/leaderboard.md` from the ledger. `scripts/promote.sh` copies winning `train.py` to `checkpoints/best/`. `scripts/finalize_experiment.sh` orchestrates the three post-run scripts. `train.py` is updated to write full state-dicts and a `notes.md` skeleton when the agent forgets.

**Tech Stack:** Python (stdlib + json + pathlib), bash. No new pip dependencies. Reuses everything from Phases 0-3.

**Spec:** [`docs/superpowers/specs/2026-05-05-pharma-vlm-autoresearch-design.md`](../specs/2026-05-05-pharma-vlm-autoresearch-design.md), §5 (Autoresearch Harness), §8.2 (notes.md template), §9.4 (Phase 4 exit criteria).

**Phase 0–3 prerequisites met:** repo on Colab + GCS, `prepare.py` v1 frozen with 1853-token tokenizer, `train.py` baseline trains end-to-end (3 seeds reproducible to ±0.008), `disaster_recover.sh` proven, sync_to_gcs daemon mirrors `experiments/` to GCS every 5 min.

**Out of scope for this plan (Phase 4b — the actual sweep):** running the 30 explore experiments. That's open-ended autoresearch work that happens by Claude Code interacting with `program.md` over multiple sessions, not as a TDD task list.

---

## File Structure

| Path | Created/Modified | Responsibility |
|---|---|---|
| `program.md` | **REWRITE from Phase-0 skeleton** | Full agent contract: goal, metric, time budget, allowed/forbidden actions, source-vetting rule, workflow, backup, do-not list. |
| `program_explore.md` | **REWRITE from Phase-0 skeleton** | Phase-1 (explore) override: 15-min budget, single-seed kept-if `>best+ε`, preferred fast-signal directions. |
| `train.py` | Modify | Add full state-dict save + auto-create `notes.md` skeleton if missing. |
| `scripts/append_ledger.py` | Create | Read `metrics.json` + `notes.md` for a `run_id`, decide kept/reverted, append one line to `experiments/ledger.jsonl`. |
| `scripts/leaderboard.py` | Create | Read `experiments/ledger.jsonl`, write a sorted Markdown table to `experiments/leaderboard.md`. |
| `scripts/promote.sh` | Create | Copy winning `train.py` and config to `checkpoints/best/`. |
| `scripts/finalize_experiment.sh` | Create | Orchestrator: append_ledger → leaderboard → promote-if-kept → git add+commit. |
| `tests/test_append_ledger.py` | Create | Lightweight tests for ledger append logic. |
| `tests/test_leaderboard.py` | Create | Lightweight tests for leaderboard rendering. |
| `tests/test_explore_e2e.py` | Create | Colab-marked: end-to-end explore-loop verification. |

**Ledger entry schema** (one JSON object per line in `experiments/ledger.jsonl`, per spec §5.4):

```json
{
  "run_id": "20260505-rope-vs-alibi",
  "phase": "explore",
  "parent_run_id": "baseline-seed42",
  "hypothesis": "ALiBi handles longer label sequences better than RoPE for pharma labels with many fields",
  "citation": ["press_2022_alibi.pdf"],
  "diff_summary": "Replaced RoPE with ALiBi in decoder self-attention",
  "config": {"hidden_dim": 512, "n_decoder_layers": 6, "...": "..."},
  "wall_clock_min": 14.2,
  "gpu": "A100-40GB",
  "metrics": {"final_macro_f1": 0.0721, "trainable_params": 26542592},
  "best_at_time_of_run": 0.0710,
  "kept": true,
  "rationale": "+0.0011 macro_f1 over baseline",
  "notes_path": "experiments/runs/20260505-rope-vs-alibi/notes.md",
  "git_sha": "abc1234",
  "timestamp": "2026-05-05T17:42:11Z"
}
```

---

## Task 1: Finalize `program.md` and `program_explore.md`

**Files:**
- Modify: `program.md` (rewrite from Phase-0 skeleton)
- Modify: `program_explore.md` (rewrite from Phase-0 skeleton)
- Modify: `tests/test_smoke.py` (extend program.md test)

- [ ] **Step 1: Append failing test for the expanded program.md**

Append to `/Users/apple/AutoLearnMeds/tests/test_smoke.py`:

```python


def test_program_md_has_full_contract(project_root: Path) -> None:
    """program.md must include every section the agent needs to operate."""
    text = (project_root / "program.md").read_text()
    required_sections = [
        "## Goal",
        "## Metric",
        "## Time budget",
        "## Allowed",
        "## Forbidden",
        "## Source vetting",
        "## Workflow per experiment",
        "## Notes.md template",
        "## Backup",
        "## Do not",
    ]
    for sec in required_sections:
        assert sec in text, f"missing section: {sec!r}"
    # Must reference the canonical scripts the agent uses
    assert "scripts/run_experiment.sh" in text
    assert "scripts/finalize_experiment.sh" in text
    # Must reference the metric format
    assert "final_macro_f1" in text
    # Must declare the test-set firewall
    assert "test.jsonl" in text
    assert "evaluate_test" in text
```

- [ ] **Step 2: Run, verify failure**

```bash
uv run pytest tests/test_smoke.py::test_program_md_has_full_contract -v
```

Expected: FAIL — Phase-0 program.md doesn't have all those sections.

- [ ] **Step 3: Rewrite `program.md` with the full agent contract**

Use Write tool to overwrite `/Users/apple/AutoLearnMeds/program.md`:

```markdown
# Pharma-VLM Autoresearch — Agent Contract (`program.md`)

You are an autonomous research agent operating on this repo. Your job: iterate on `train.py` to maximize a single scalar metric.

This is the BASELINE contract. Phase-specific overrides live in `program_explore.md` and `program_confirm.md`.

## Goal

Maximize `val_macro_f1` of structured field extraction from pharmaceutical product label images, by editing `train.py`.

## Metric

Single scalar: `val_macro_f1`. Higher is better. Reported by `train.py` as the literal stdout line `final_macro_f1=X.XXXX` on the last line of training. Computed by `prepare.compute_metrics()` over the 10 high-frequency fields.

Phase-2 baseline floor: ~0.071 ± 0.008 across 3 seeds (see `docs/superpowers/reviews/phase2_review.md`). Every change you make must be measured against this floor.

## Time budget

- **Phase explore:** 15 minutes wall-clock per experiment, single seed.
- **Phase confirm:** 60 minutes wall-clock per experiment, 3 seeds (mean ± std reported).

If your edit makes a single experiment exceed the budget, reduce `max_steps` in `train.py` so it fits.

## Allowed (you may modify in `train.py`)

- **Architecture:** layers, hidden_dim, heads, ffn_ratio, activation, dropout, position encoding (RoPE/ALiBi/learned), cross-attention pattern, layer-norm placement (pre vs post), tied embeddings.
- **Optimizer:** AdamW vs Lion vs Muon, betas, weight decay, schedule shape, warmup, peak LR, min LR.
- **Loss:** label smoothing, token weighting, auxiliary losses (e.g., field-presence classification head).
- **Augmentations & data:** RandAugment, AugMix, hard-example mining, curriculum learning, oversampling rare manufacturers, mix-up, paste augmentation.
- **Tokenization-strategy variants** that don't retrain the BPE (e.g., adding `<no_value>` via vocab reserve slots, prompt-prefix conditioning).
- **Decoding strategy:** greedy, beam search, sampling, constrained.
- **Hyperparameters in `experiments/configs/baseline.yaml`** — copy to a new YAML if you want to vary them (don't edit baseline.yaml).

## Forbidden

- Modifying `prepare.py`. (FROZEN — the metric definition lives there.)
- Modifying `prepare.compute_metrics()` or `prepare.evaluate()` indirectly via `format_output`/`parse_output`/`normalize_field_value`.
- Retraining the BPE tokenizer. (Locked at vocab=1853 from Phase 1.)
- Changing the test set or invoking `evaluate_test()`. The agent does not look at `data/processed/test.jsonl`; that's reserved for paper-final numbers.
- Swapping the encoder out of the SigLIP family without a new top-level run-tag (e.g., `dinov2-...`) and a paragraph in your `notes.md` justifying why.
- Skipping `notes.md` or its citation field. Every change must cite a paper from `papers/` or add a new one with a one-paragraph summary.

## Source vetting

Every change MUST cite at least one paper from `papers/` or add a new paper to `papers/` with a one-paragraph summary in `papers/README.md`. If a change is heuristic (no paper), label it `--exploratory--` in your `notes.md` and run it under a separate run-tag prefix.

When you cite a paper, in `notes.md` reference the exact §/page where the technique appears, AND list one or two adjacent ideas from the same paper you did NOT use but COULD repurpose. This is the "creative reading" discipline; it builds a richer paper-prep backlog.

## Workflow per experiment

1. **Read the ledger.** `tail -20 experiments/ledger.jsonl` and skim recent kept/reverted entries. Don't repeat what's been tried (or if you do, run-tag it `--reproduce--`).
2. **Pick a direction.** One knob change. Cite the paper. Predict the direction (sign + rough magnitude on macro_f1) BEFORE running.
3. **Edit `train.py`.** Minimal diff. Comment your change with `# AGENT: <hypothesis>`.
4. **Pre-write `notes.md`** at `experiments/runs/<run_id>/notes.md` (use the template below). Hypothesis + citation + predicted direction are pre-experiment. The Result and Retrospective sections are filled post-run.
5. **Run.** `bash scripts/run_experiment.sh <run_id> --seed 42 --no-wandb`.
6. **Finalize.** `bash scripts/finalize_experiment.sh <run_id>` — this calls append_ledger, regenerates leaderboard, promotes if kept, and commits.
7. **Push.** `git push origin phase-0-plumbing` (or whatever branch is active).

## Notes.md template

Every experiment's notes.md MUST have these sections (enforced by `scripts/check_notes.py`):

```markdown
# Experiment: <run_id>

## Hypothesis
<one sentence. Specific, falsifiable.>

## Change
<minimal diff description. One knob.>

## Citation
- <author year — paper title — papers/<file>.pdf — exact §/page used>

## Adjacent ideas (creative reading)
<1-2 techniques from the cited paper(s) you did NOT use but could repurpose for a future experiment.>

## Predicted direction
<expected sign and rough magnitude on macro_f1, with reasoning. Pre-experiment.>

## Result
- final_macro_f1: <value>
- per-field F1 changes (vs prior best): <list>
- wall_clock_min: <value>
- trainable_params: <value>

## Verdict
- kept: <true/false>
- reason: <if false, why>

## Retrospective
<2-3 sentences. What did we learn? Did the result match the prediction? If not, what does that tell us? What's the next experiment this suggests?>
```

## Backup

After every experiment, `scripts/sync_to_gcs.sh` (daemon) mirrors `experiments/` and `checkpoints/` to GCS within 5 min. If you edit between experiments, `git push` is your version control. If Colab disconnects mid-experiment, `scripts/resume_or_start.py` (state machine) decides whether to restart or resume on the next session. If your token quota approaches its 5-hour limit, write `RESUME_NEEDED.md` at the repo root with `next_planned_action`, then exit cleanly. The next session reads it and picks up.

## Do not

- Combine multiple changes in one experiment ("one knob per experiment"). If you find yourself writing two `# AGENT: ...` comments, split into two experiments.
- Continue past your time budget. If the experiment OOMs, the ledger entry will reflect that; pick a different direction next time.
- Skip the `notes.md` or the citation. The autoresearch ledger's research credibility depends on every entry being defensible.
- Touch the test set. Ever.
- Try to "tune" the metric (e.g., output formatting tricks that game `compute_metrics`). The metric is FROZEN; your job is to improve the model under it.
```

- [ ] **Step 4: Rewrite `program_explore.md`**

Use Write tool to overwrite `/Users/apple/AutoLearnMeds/program_explore.md`:

```markdown
# Phase-explore Override (`program_explore.md`)

Inherits everything from `program.md`. The following overrides apply during the explore phase.

## Time budget

15 minutes wall-clock per experiment, single seed (default seed=42).

## Selection rule

Keep the change if `new_macro_f1 > current_best + 0.001`. Single seed, noisy OK. Confirm phase will filter the false positives.

## Preferred directions (priority order, from Phase 2 review backlog)

1. **Image augmentation** — RandAugment (`papers/cubuk_2020_randaugment.pdf`, M=5-7 for our 564-image train set), AugMix (`papers/hendrycks_2020_augmix.pdf`). Direct hit on the train→val gap.
2. **Label smoothing** — `papers/szegedy_2016_label_smoothing.pdf`, smoothing=0.1. Reduces memorization confidence.
3. **Decoder dropout sweep** — 0.1 → 0.2 → 0.3. Standard regularization knob.
4. **Decoding strategy** — beam search width=4 vs greedy.
5. **Donut "task tokens"** — `papers/kim_2022_donut.pdf`, prefix decoder with `<pack=...>` and `<view=...>` from existing record metadata.
6. **Untie embeddings** — testing if Press 2017 (`papers/press_2017_tied_embeddings.pdf`) regularization argument applies to our small-vocab regime.
7. **RoPE on subset of head_dim** — LLaMA-style; `papers/su_2021_roformer.pdf` ablation.
8. **Pre-LN vs post-LN** — `papers/vaswani_2017_attention.pdf` original used post-LN.
9. **Curriculum by polygon size** — easy fields (large polygon) first, then hard. Free supervision from the data.

## Stop after

20 consecutive non-improving experiments OR `--user-stop`.

## Run-tag prefix

Use `<short-hypothesis>` as the prefix in your run_id, e.g., `randaugm-m5-seed42`, `ls01-seed42`, `dropout02-seed42`. Keeps the leaderboard scannable.
```

- [ ] **Step 5: Run, verify tests pass**

```bash
uv run pytest tests/test_smoke.py -v 2>&1 | tail -3
```

Expected: 64 PASSED (63 prior + 1 new), 18 deselected.

- [ ] **Step 6: Commit**

```bash
git add program.md program_explore.md tests/test_smoke.py
git commit -m "feat(phase4): finalize program.md + program_explore.md (full agent contract)

Replaces the Phase-0 skeleton with the full autoresearch agent contract:
- 10 sections: Goal, Metric, Time budget, Allowed, Forbidden, Source
  vetting, Workflow per experiment, Notes.md template, Backup, Do not.
- Documents the 0.071 baseline floor + ±0.008 reproducibility from
  Phase 2 as the reference point.
- Bakes in the RCA + creative-paper-reading discipline: every notes.md
  must include an 'Adjacent ideas' section listing repurposable
  techniques from cited papers.
- Lists exact scripts the agent calls (run_experiment.sh,
  finalize_experiment.sh).

program_explore.md adds the 9-item priority ordered backlog from the
Phase 2 review, the 15-min budget, the +0.001 keep threshold, and
the 20-consecutive-non-improving stop criterion.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `train.py` — full state-dict save + notes.md auto-create

**Files:**
- Modify: `train.py` (extend train_loop)

The Phase 2 train_loop only saves `best.meta.pt` (just `{"step": ..., "macro_f1": ...}`). Phase 4 needs full weights so the confirm-phase can re-evaluate. Also: if the agent forgets to write `notes.md`, train_loop creates a skeleton (it has the data — config, run dir).

- [ ] **Step 1: Open `train.py` and locate the `train_loop` checkpoint-save block**

The current code in train.py around line 615 (after Phase 2 commits) saves:

```python
if metrics["macro_f1"] > best_macro_f1:
    best_macro_f1 = metrics["macro_f1"]
    ckpt = {"step": step, "macro_f1": best_macro_f1}
    torch.save(ckpt, Path(cfg.checkpoint_dir) / "best.meta.pt")
```

Verify that's still the form by reading the file:

```bash
grep -n "best.meta.pt" /Users/apple/AutoLearnMeds/train.py
```

Expected: one match (the line above).

- [ ] **Step 2: Replace the meta-only save with a full state-dict save**

Use Edit tool. Replace:

```python
            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                ckpt = {"step": step, "macro_f1": best_macro_f1}
                torch.save(ckpt, Path(cfg.checkpoint_dir) / "best.meta.pt")
```

with:

```python
            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                # Full state dict for proj + decoder (encoder is frozen — re-load
                # from HF on resume). Saved as a regular torch dict so confirm-
                # phase / paper-prep can load even from a different machine.
                full_ckpt = {
                    "step": step,
                    "macro_f1": best_macro_f1,
                    "proj_state_dict": model.proj.state_dict(),
                    "decoder_blocks_state_dicts": [
                        {n: p.detach().cpu() for n, p in self_named_params(b)}
                        for b in model.decoder.blocks
                    ],
                    "decoder_embedding_state_dict": model.decoder.embedding.state_dict(),
                    "decoder_final_ln_state_dict": model.decoder.final_ln.state_dict(),
                    "config": dataclasses.asdict(cfg),
                }
                torch.save(full_ckpt, Path(cfg.checkpoint_dir) / "best.pt")
                # Also keep the lightweight meta for backward compatibility
                torch.save(
                    {"step": step, "macro_f1": best_macro_f1},
                    Path(cfg.checkpoint_dir) / "best.meta.pt",
                )
```

- [ ] **Step 3: Add the `self_named_params` helper to `train.py`**

Use Edit tool. Append after the `_block_param_names` function if it exists, or right before `train_loop` if it doesn't. Search:

```bash
grep -n "def train_loop" /Users/apple/AutoLearnMeds/train.py
```

Find the line number, then insert ABOVE it:

```python


def self_named_params(module) -> list[tuple[str, "torch.Tensor"]]:
    """Yield (name, tensor) for every parameter in `module`. Works for both
    nn.Modules (uses .named_parameters()) and our DecoderBlock pseudo-module
    (synthesizes names from _modules + their attributes).
    """
    if hasattr(module, "named_parameters") and callable(module.named_parameters):
        return list(module.named_parameters())
    # DecoderBlock case: walk _modules with stable names per spec.
    out: list[tuple[str, "torch.Tensor"]] = []
    name_map = {
        id(module.ln1): "ln1", id(module.q_proj): "q_proj",
        id(module.k_proj): "k_proj", id(module.v_proj): "v_proj",
        id(module.o_proj): "o_proj", id(module.ln2): "ln2",
        id(module.xq_proj): "xq_proj", id(module.xk_proj): "xk_proj",
        id(module.xv_proj): "xv_proj", id(module.xo_proj): "xo_proj",
        id(module.ln3): "ln3", id(module.fc1): "fc1", id(module.fc2): "fc2",
    }
    for m in module._modules:
        prefix = name_map.get(id(m), m.__class__.__name__.lower())
        for pname, p in m.named_parameters() if hasattr(m, "named_parameters") else []:
            out.append((f"{prefix}.{pname}", p))
    return out
```

- [ ] **Step 4: Verify train.py still imports cleanly**

```bash
uv run pytest tests/test_train_components.py -v 2>&1 | tail -3
```

Expected: 3 PASSED (the lightweight tests that don't need torch).

- [ ] **Step 5: Commit**

```bash
git add train.py
git commit -m "feat(train): save full state-dict (proj + decoder) at best macro_f1

Phase 2's train_loop saved only {step, macro_f1} in best.meta.pt. Phase 4
needs full weights so confirm-phase can re-evaluate from a checkpoint.
We save proj, decoder embedding, decoder final_ln, and per-block param
state dicts (cpu-detached for portability). Encoder is frozen — re-loaded
from HF on resume.

Both files written: best.pt (full), best.meta.pt (light, for tooling
that just wants the score).

self_named_params helper handles both nn.Modules (.named_parameters)
and our DecoderBlock pseudo-module (synthesized names per spec).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `scripts/append_ledger.py` — atomic ledger append

**Files:**
- Create: `scripts/append_ledger.py`
- Create: `tests/test_append_ledger.py`

- [ ] **Step 1: Write the failing test**

Create `/Users/apple/AutoLearnMeds/tests/test_append_ledger.py`:

```python
"""Tests for scripts/append_ledger.py."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_module(project_root: Path):
    spec = importlib.util.spec_from_file_location(
        "append_ledger", project_root / "scripts" / "append_ledger.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_run(workspace: Path, run_id: str, final_macro_f1: float, with_notes: bool = True) -> Path:
    run_dir = workspace / "experiments" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps({
        "run_id": run_id,
        "git_sha": "deadbeef",
        "config": "experiments/configs/baseline.yaml",
        "final_macro_f1": final_macro_f1,
        "wall_clock_seconds": 1800,
        "exit_code": 0,
    }))
    (run_dir / "config.yaml").write_text("seed: 42\nhidden_dim: 512\n")
    (run_dir / "train.py").write_text("# snapshot\n")
    if with_notes:
        (run_dir / "notes.md").write_text(
            "# Experiment: " + run_id + "\n\n"
            "## Hypothesis\nTest hypothesis.\n\n"
            "## Citation\n- press_2022_alibi.pdf §3.1\n\n"
            "## Adjacent ideas\nN/A\n\n"
            "## Predicted direction\n+0.01\n"
        )
    return run_dir


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "experiments" / "runs").mkdir(parents=True)
    (tmp_path / "experiments" / "ledger.jsonl").touch()
    return tmp_path


def test_append_creates_ledger_entry(project_root: Path, workspace: Path) -> None:
    mod = _load_module(project_root)
    _make_run(workspace, "run-001", 0.0710)
    mod.append_ledger(
        run_id="run-001",
        workspace=workspace,
        phase="explore",
        keep_threshold=0.001,
    )
    line = (workspace / "experiments" / "ledger.jsonl").read_text().strip()
    entry = json.loads(line)
    assert entry["run_id"] == "run-001"
    assert entry["phase"] == "explore"
    assert entry["metrics"]["final_macro_f1"] == 0.0710
    assert "kept" in entry
    assert "best_at_time_of_run" in entry
    assert entry["timestamp"].endswith("Z") or "+" in entry["timestamp"]


def test_append_kept_when_above_threshold(project_root: Path, workspace: Path) -> None:
    mod = _load_module(project_root)
    # Seed a prior best at 0.0710
    _make_run(workspace, "run-baseline", 0.0710)
    mod.append_ledger(run_id="run-baseline", workspace=workspace, phase="explore", keep_threshold=0.001)
    # New run beats by 0.005
    _make_run(workspace, "run-better", 0.0760)
    mod.append_ledger(run_id="run-better", workspace=workspace, phase="explore", keep_threshold=0.001)
    lines = [json.loads(l) for l in (workspace / "experiments" / "ledger.jsonl").read_text().splitlines() if l.strip()]
    assert lines[-1]["kept"] is True
    assert lines[-1]["best_at_time_of_run"] == pytest.approx(0.0710)


def test_append_reverted_when_below_threshold(project_root: Path, workspace: Path) -> None:
    mod = _load_module(project_root)
    _make_run(workspace, "run-baseline", 0.0710)
    mod.append_ledger(run_id="run-baseline", workspace=workspace, phase="explore", keep_threshold=0.001)
    _make_run(workspace, "run-worse", 0.0700)
    mod.append_ledger(run_id="run-worse", workspace=workspace, phase="explore", keep_threshold=0.001)
    lines = [json.loads(l) for l in (workspace / "experiments" / "ledger.jsonl").read_text().splitlines() if l.strip()]
    assert lines[-1]["kept"] is False


def test_append_extracts_hypothesis_from_notes(project_root: Path, workspace: Path) -> None:
    mod = _load_module(project_root)
    _make_run(workspace, "run-001", 0.0710)
    mod.append_ledger(run_id="run-001", workspace=workspace, phase="explore", keep_threshold=0.001)
    entry = json.loads((workspace / "experiments" / "ledger.jsonl").read_text().splitlines()[-1])
    assert entry["hypothesis"] == "Test hypothesis."
    assert "press_2022_alibi.pdf" in entry["citation"][0]


def test_append_handles_missing_notes(project_root: Path, workspace: Path) -> None:
    """If notes.md is absent, ledger entry still writes with hypothesis='(missing)' and a warning."""
    mod = _load_module(project_root)
    _make_run(workspace, "run-001", 0.0710, with_notes=False)
    mod.append_ledger(run_id="run-001", workspace=workspace, phase="explore", keep_threshold=0.001)
    entry = json.loads((workspace / "experiments" / "ledger.jsonl").read_text().splitlines()[-1])
    assert entry["hypothesis"].startswith("(missing")
```

- [ ] **Step 2: Run, verify failure**

```bash
uv run pytest tests/test_append_ledger.py -v 2>&1 | tail -10
```

Expected: 5 FAIL.

- [ ] **Step 3: Implement `scripts/append_ledger.py`**

Create `/Users/apple/AutoLearnMeds/scripts/append_ledger.py`:

```python
#!/usr/bin/env python3
"""Append one entry to experiments/ledger.jsonl after run_experiment.sh finishes.

Reads:
  experiments/runs/<run_id>/metrics.json   (mandatory)
  experiments/runs/<run_id>/notes.md       (optional but warned)
  experiments/runs/<run_id>/config.yaml    (snapshot of hyperparameters)

Computes:
  best_at_time_of_run  = max final_macro_f1 of all prior kept entries (or 0)
  kept                 = (this final_macro_f1 > best_at_time_of_run + keep_threshold)
  rationale            = human-readable comparison string
  hypothesis, citation = extracted from notes.md by simple heading lookup

Writes one JSON-line to experiments/ledger.jsonl. Append-only.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path
from typing import Any


def _read_notes_section(text: str, heading: str) -> str:
    """Extract the body under '## <heading>' up to the next '## ' or EOF."""
    pat = re.compile(
        r"^##\s+" + re.escape(heading) + r"\s*\n(.*?)(?=^##\s|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    m = pat.search(text)
    return m.group(1).strip() if m else ""


def _parse_notes(notes_path: Path) -> dict[str, Any]:
    """Return {hypothesis, citation:list[str]} from notes.md, or '(missing)' values."""
    if not notes_path.is_file():
        return {"hypothesis": "(missing notes.md)", "citation": []}
    text = notes_path.read_text()
    hypothesis = _read_notes_section(text, "Hypothesis")
    if not hypothesis:
        hypothesis = "(missing Hypothesis section)"
    cit_block = _read_notes_section(text, "Citation")
    citations = [
        line.lstrip("- ").strip()
        for line in cit_block.splitlines()
        if line.strip().startswith("-")
    ]
    return {"hypothesis": hypothesis, "citation": citations}


def _current_best(ledger_path: Path) -> float:
    """Max final_macro_f1 across all prior `kept: true` entries; 0.0 if empty."""
    if not ledger_path.is_file():
        return 0.0
    best = 0.0
    for line in ledger_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("kept"):
            f1 = (entry.get("metrics") or {}).get("final_macro_f1", 0.0)
            if f1 > best:
                best = f1
    return best


def append_ledger(
    run_id: str,
    workspace: Path | str,
    phase: str = "explore",
    parent_run_id: str = "",
    keep_threshold: float = 0.001,
) -> dict[str, Any]:
    """Build and append a ledger entry. Returns the entry dict."""
    workspace = Path(workspace)
    run_dir = workspace / "experiments" / "runs" / run_id
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics.json not found at {metrics_path}")

    metrics = json.loads(metrics_path.read_text())
    notes_path = run_dir / "notes.md"
    notes = _parse_notes(notes_path)

    ledger_path = workspace / "experiments" / "ledger.jsonl"
    best = _current_best(ledger_path)
    f1 = float(metrics.get("final_macro_f1", -1.0))
    kept = f1 > best + keep_threshold
    if best == 0.0 and f1 > 0:
        rationale = f"first run; sets initial floor at {f1:.4f}"
    elif kept:
        rationale = f"+{f1 - best:.4f} macro_f1 over prior best ({best:.4f})"
    else:
        delta = f1 - best
        rationale = f"{delta:+.4f} macro_f1 vs prior best ({best:.4f}); below keep threshold {keep_threshold}"

    config_yaml_path = run_dir / "config.yaml"
    config_text = config_yaml_path.read_text() if config_yaml_path.is_file() else ""

    entry = {
        "run_id": run_id,
        "phase": phase,
        "parent_run_id": parent_run_id,
        "hypothesis": notes["hypothesis"],
        "citation": notes["citation"],
        "config_path": str(config_yaml_path.relative_to(workspace)) if config_yaml_path.is_file() else "",
        "config_text_first_lines": "\n".join(config_text.splitlines()[:5]),
        "wall_clock_min": round(metrics.get("wall_clock_seconds", 0) / 60.0, 2),
        "metrics": {
            "final_macro_f1": f1,
            "wall_clock_seconds": metrics.get("wall_clock_seconds"),
            "exit_code": metrics.get("exit_code"),
        },
        "best_at_time_of_run": best,
        "kept": kept,
        "rationale": rationale,
        "notes_path": str(notes_path.relative_to(workspace)),
        "git_sha": metrics.get("git_sha", "unknown"),
        "timestamp": _dt.datetime.now(_dt.UTC).isoformat(),
    }

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_id", type=str)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--phase", default="explore", choices=["explore", "confirm"])
    parser.add_argument("--parent-run-id", default="", help="The run this experiment iterates on")
    parser.add_argument("--keep-threshold", type=float, default=0.001)
    args = parser.parse_args()

    if not args.workspace.exists():
        print(f"[append_ledger] FATAL: workspace not found: {args.workspace}", file=sys.stderr)
        return 2

    entry = append_ledger(
        run_id=args.run_id,
        workspace=args.workspace,
        phase=args.phase,
        parent_run_id=args.parent_run_id,
        keep_threshold=args.keep_threshold,
    )
    verdict = "KEPT" if entry["kept"] else "REVERTED"
    print(f"[append_ledger] {verdict} run_id={args.run_id} f1={entry['metrics']['final_macro_f1']:.4f} ({entry['rationale']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Make executable**

```bash
chmod +x /Users/apple/AutoLearnMeds/scripts/append_ledger.py
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_append_ledger.py -v
```

Expected: 5 PASSED.

- [ ] **Step 6: Commit**

```bash
git add scripts/append_ledger.py tests/test_append_ledger.py
git commit -m "feat(phase4): scripts/append_ledger.py — atomic ledger append per spec §5.4

Reads metrics.json + notes.md from a run dir, computes the
best_at_time_of_run from prior kept ledger entries, decides kept based
on a threshold (default 0.001), writes one JSON line to
experiments/ledger.jsonl. Append-only; preserves history.

Hypothesis + citation are pulled from notes.md by markdown heading
lookup. If notes.md is missing, the entry still writes with a
'(missing notes.md)' marker — the agent should never let that happen,
but we don't lose data.

5 lightweight tests cover entry creation, kept/reverted thresholding,
notes parsing, and missing-notes graceful degradation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `scripts/leaderboard.py` — regenerate leaderboard.md

**Files:**
- Create: `scripts/leaderboard.py`
- Create: `tests/test_leaderboard.py`

- [ ] **Step 1: Write the failing test**

Create `/Users/apple/AutoLearnMeds/tests/test_leaderboard.py`:

```python
"""Tests for scripts/leaderboard.py."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_module(project_root: Path):
    spec = importlib.util.spec_from_file_location(
        "leaderboard", project_root / "scripts" / "leaderboard.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_ledger(workspace: Path, entries: list[dict]) -> Path:
    p = workspace / "experiments" / "ledger.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(e) for e in entries))
    return p


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


def test_renders_table_in_descending_order(project_root: Path, workspace: Path) -> None:
    mod = _load_module(project_root)
    _write_ledger(workspace, [
        {"run_id": "a", "phase": "explore", "metrics": {"final_macro_f1": 0.05}, "kept": False, "rationale": "x", "wall_clock_min": 30},
        {"run_id": "b", "phase": "explore", "metrics": {"final_macro_f1": 0.10}, "kept": True, "rationale": "y", "wall_clock_min": 30},
        {"run_id": "c", "phase": "explore", "metrics": {"final_macro_f1": 0.07}, "kept": False, "rationale": "z", "wall_clock_min": 30},
    ])
    out = workspace / "experiments" / "leaderboard.md"
    mod.regenerate(workspace, out)
    text = out.read_text()
    # b (0.10) appears before c (0.07) appears before a (0.05)
    assert text.index("| b ") < text.index("| c ")
    assert text.index("| c ") < text.index("| a ")


def test_marks_kept_with_marker(project_root: Path, workspace: Path) -> None:
    mod = _load_module(project_root)
    _write_ledger(workspace, [
        {"run_id": "kept_run", "phase": "explore", "metrics": {"final_macro_f1": 0.10}, "kept": True, "rationale": "x", "wall_clock_min": 30},
        {"run_id": "rev_run",  "phase": "explore", "metrics": {"final_macro_f1": 0.05}, "kept": False, "rationale": "y", "wall_clock_min": 30},
    ])
    out = workspace / "experiments" / "leaderboard.md"
    mod.regenerate(workspace, out)
    text = out.read_text()
    # Some marker — checking ✓ for kept and ✗ for reverted
    kept_line = [l for l in text.splitlines() if "kept_run" in l][0]
    rev_line = [l for l in text.splitlines() if "rev_run" in l][0]
    assert "✓" in kept_line or "yes" in kept_line.lower()
    assert "✗" in rev_line or "no" in rev_line.lower()


def test_handles_empty_ledger(project_root: Path, workspace: Path) -> None:
    mod = _load_module(project_root)
    (workspace / "experiments").mkdir(exist_ok=True, parents=True)
    (workspace / "experiments" / "ledger.jsonl").write_text("")
    out = workspace / "experiments" / "leaderboard.md"
    mod.regenerate(workspace, out)
    text = out.read_text()
    assert "no experiments yet" in text.lower() or "0 experiments" in text.lower()
```

- [ ] **Step 2: Run, verify failure**

```bash
uv run pytest tests/test_leaderboard.py -v
```

Expected: 3 FAIL.

- [ ] **Step 3: Implement `scripts/leaderboard.py`**

Create `/Users/apple/AutoLearnMeds/scripts/leaderboard.py`:

```python
#!/usr/bin/env python3
"""Regenerate experiments/leaderboard.md from experiments/ledger.jsonl.

Output is a Markdown table sorted by final_macro_f1 descending. Handy at the
top of any explore-loop session for the agent to see where the floor is.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _read_ledger(workspace: Path) -> list[dict]:
    p = workspace / "experiments" / "ledger.jsonl"
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def regenerate(workspace: Path | str, out_path: Path | str | None = None) -> Path:
    workspace = Path(workspace)
    out_path = Path(out_path) if out_path else workspace / "experiments" / "leaderboard.md"
    entries = _read_ledger(workspace)

    lines: list[str] = []
    lines.append("# Experiments Leaderboard")
    lines.append("")
    lines.append("Auto-generated by `scripts/leaderboard.py`. Re-runs after every "
                 "`finalize_experiment.sh`.")
    lines.append("")

    if not entries:
        lines.append("_no experiments yet_")
        out_path.write_text("\n".join(lines) + "\n")
        return out_path

    entries_sorted = sorted(
        entries,
        key=lambda e: (e.get("metrics") or {}).get("final_macro_f1", -1.0),
        reverse=True,
    )

    kept_count = sum(1 for e in entries if e.get("kept"))
    lines.append(f"**{len(entries)} experiments total · {kept_count} kept · "
                 f"current floor: {entries_sorted[0]['metrics']['final_macro_f1']:.4f}**")
    lines.append("")

    lines.append("| rank | run_id | phase | macro_f1 | kept | wall (min) | hypothesis |")
    lines.append("|---|---|---|---|---|---|---|")
    for rank, e in enumerate(entries_sorted, start=1):
        f1 = (e.get("metrics") or {}).get("final_macro_f1", -1.0)
        kept = "✓" if e.get("kept") else "✗"
        wall = e.get("wall_clock_min", "?")
        hyp = (e.get("hypothesis") or "").splitlines()[0][:60]
        lines.append(
            f"| {rank} | {e.get('run_id', '?')} | {e.get('phase', '?')} | "
            f"{f1:.4f} | {kept} | {wall} | {hyp} |"
        )
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path; defaults to <workspace>/experiments/leaderboard.md")
    args = parser.parse_args()
    out = regenerate(args.workspace, args.out)
    print(f"[leaderboard] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Make executable**

```bash
chmod +x /Users/apple/AutoLearnMeds/scripts/leaderboard.py
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_leaderboard.py -v
```

Expected: 3 PASSED.

- [ ] **Step 6: Commit**

```bash
git add scripts/leaderboard.py tests/test_leaderboard.py
git commit -m "feat(phase4): scripts/leaderboard.py — regenerate leaderboard.md from ledger

Reads experiments/ledger.jsonl, writes experiments/leaderboard.md
sorted by final_macro_f1 descending. Each row includes run_id, phase,
score, kept marker (✓/✗), wall clock, and the hypothesis line.

Handles empty ledger ('no experiments yet'). 3 lightweight tests cover
descending-order, kept-marker, and empty-input.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `scripts/promote.sh` — checkpoint a winning experiment

**Files:**
- Create: `scripts/promote.sh`
- Modify: `tests/test_smoke.py`

- [ ] **Step 1: Append failing test**

Append to `/Users/apple/AutoLearnMeds/tests/test_smoke.py`:

```python


def test_promote_script_exists_and_executable(project_root: Path) -> None:
    p = project_root / "scripts" / "promote.sh"
    assert p.is_file()
    assert os.access(p, os.X_OK)
    text = p.read_text()
    assert text.startswith("#!/")
    assert "set -euo pipefail" in text
    assert "checkpoints/best" in text
```

- [ ] **Step 2: Run, verify failure**

```bash
uv run pytest tests/test_smoke.py::test_promote_script_exists_and_executable -v
```

- [ ] **Step 3: Implement `scripts/promote.sh`**

Create `/Users/apple/AutoLearnMeds/scripts/promote.sh`:

```bash
#!/usr/bin/env bash
# Promote a kept experiment's train.py + best.pt to checkpoints/best/.
#
# Usage:
#   bash scripts/promote.sh <run_id>
#
# What it does:
#   1. Snapshots experiments/runs/<run_id>/train.py + config.yaml to
#      checkpoints/best/{train.py, config.yaml}.
#   2. If checkpoints/runs/<run_id>/best.pt exists (saved by train_loop on
#      best macro_f1), copies it to checkpoints/best/best.pt.
#   3. Records the run_id and timestamp in checkpoints/best/manifest.txt.

set -euo pipefail

RUN_ID="${1:?Usage: promote.sh <run_id>}"
WORKSPACE="${AUTOLEARNMEDS_WORKSPACE:-/workspace}"

cd "$WORKSPACE"

RUN_DIR="experiments/runs/${RUN_ID}"
BEST_DIR="checkpoints/best"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "[promote] FATAL: $RUN_DIR not found" >&2
  exit 1
fi

mkdir -p "$BEST_DIR"

cp "$RUN_DIR/train.py" "$BEST_DIR/train.py"
cp "$RUN_DIR/config.yaml" "$BEST_DIR/config.yaml"

# best.pt is saved by train_loop when val macro_f1 improves; might live under
# the per-run checkpoint dir specified in config.yaml. We try the standard
# checkpoints/runs/<run_id>/best.pt first.
SRC_BEST="checkpoints/runs/${RUN_ID}/best.pt"
if [[ -f "$SRC_BEST" ]]; then
  cp "$SRC_BEST" "$BEST_DIR/best.pt"
  echo "[promote] copied $SRC_BEST"
else
  echo "[promote] WARN: $SRC_BEST not found; promoting code only (no weights)"
fi

# Manifest — append-only history of every promotion.
MANIFEST="$BEST_DIR/manifest.txt"
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "$TS  promoted run_id=$RUN_ID" >> "$MANIFEST"

echo "[promote] DONE: $RUN_ID is now the current best"
```

- [ ] **Step 4: Make executable**

```bash
chmod +x /Users/apple/AutoLearnMeds/scripts/promote.sh
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_smoke.py::test_promote_script_exists_and_executable -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/promote.sh tests/test_smoke.py
git commit -m "feat(phase4): scripts/promote.sh — checkpoint a winning experiment

Copies the run's train.py + config.yaml + best.pt (if present) to
checkpoints/best/, and appends a timestamped line to
checkpoints/best/manifest.txt — the append-only promotion history.

Called by finalize_experiment.sh when the ledger entry has kept=true.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `scripts/finalize_experiment.sh` — orchestrate post-run housekeeping

**Files:**
- Create: `scripts/finalize_experiment.sh`
- Modify: `tests/test_smoke.py`

- [ ] **Step 1: Append failing test**

Append to `/Users/apple/AutoLearnMeds/tests/test_smoke.py`:

```python


def test_finalize_experiment_script_exists_and_calls_subscripts(project_root: Path) -> None:
    p = project_root / "scripts" / "finalize_experiment.sh"
    assert p.is_file()
    assert os.access(p, os.X_OK)
    text = p.read_text()
    assert text.startswith("#!/")
    assert "set -euo pipefail" in text
    assert "scripts/append_ledger.py" in text
    assert "scripts/leaderboard.py" in text
    assert "scripts/promote.sh" in text
```

- [ ] **Step 2: Run, verify failure**

```bash
uv run pytest tests/test_smoke.py::test_finalize_experiment_script_exists_and_calls_subscripts -v
```

- [ ] **Step 3: Implement `scripts/finalize_experiment.sh`**

Create `/Users/apple/AutoLearnMeds/scripts/finalize_experiment.sh`:

```bash
#!/usr/bin/env bash
# Post-experiment housekeeping. Runs after scripts/run_experiment.sh.
#
# Usage:
#   bash scripts/finalize_experiment.sh <run_id> [--phase explore|confirm] [--parent <run_id>]
#
# What it does:
#   1. Append a ledger entry via scripts/append_ledger.py.
#   2. Regenerate experiments/leaderboard.md via scripts/leaderboard.py.
#   3. If the ledger entry had kept=true, run scripts/promote.sh.
#   4. git add + commit (the agent / user pushes separately).

set -euo pipefail

RUN_ID="${1:?Usage: finalize_experiment.sh <run_id> [--phase ...] [--parent <id>]}"
shift || true

PHASE="explore"
PARENT=""
while (($#)); do
  case "$1" in
    --phase) PHASE="$2"; shift 2;;
    --parent) PARENT="$2"; shift 2;;
    *) echo "[finalize] unknown arg: $1" >&2; exit 2;;
  esac
done

WORKSPACE="${AUTOLEARNMEDS_WORKSPACE:-/workspace}"
cd "$WORKSPACE"

echo "[finalize] run_id=$RUN_ID phase=$PHASE parent=${PARENT:-none}"

# 1. Append ledger.
PARENT_ARG=""
if [[ -n "$PARENT" ]]; then PARENT_ARG="--parent-run-id $PARENT"; fi
LEDGER_OUT=$(uv run python scripts/append_ledger.py "$RUN_ID" \
    --workspace "$WORKSPACE" --phase "$PHASE" $PARENT_ARG 2>&1)
echo "$LEDGER_OUT"

# 2. Regenerate leaderboard.
uv run python scripts/leaderboard.py --workspace "$WORKSPACE" 2>&1 | tail -1

# 3. Promote if kept (parse the previous output).
if echo "$LEDGER_OUT" | grep -q "^\\[append_ledger\\] KEPT "; then
  echo "[finalize] kept; promoting..."
  bash scripts/promote.sh "$RUN_ID"
else
  echo "[finalize] reverted; not promoting"
fi

# 4. Git commit (the agent / user pushes).
git add experiments/ledger.jsonl experiments/leaderboard.md \
        "experiments/runs/$RUN_ID" \
        checkpoints/best 2>/dev/null || true
if ! git diff --staged --quiet; then
  git commit -m "experiment $RUN_ID ($PHASE): $(echo "$LEDGER_OUT" | head -1 | sed 's/.*\\[append_ledger\\] //')"
  echo "[finalize] committed"
else
  echo "[finalize] nothing staged to commit"
fi

echo "[finalize] DONE"
```

- [ ] **Step 4: Make executable**

```bash
chmod +x /Users/apple/AutoLearnMeds/scripts/finalize_experiment.sh
```

- [ ] **Step 5: Run tests + bash syntax check**

```bash
bash -n /Users/apple/AutoLearnMeds/scripts/finalize_experiment.sh && echo OK
uv run pytest tests/test_smoke.py -v 2>&1 | tail -3
```

Expected: `OK`. Tests PASS (66 PASSED).

- [ ] **Step 6: Commit**

```bash
git add scripts/finalize_experiment.sh tests/test_smoke.py
git commit -m "feat(phase4): scripts/finalize_experiment.sh — post-run orchestrator

After scripts/run_experiment.sh completes, the agent calls finalize:
1. uv run python scripts/append_ledger.py <run_id> ...
2. uv run python scripts/leaderboard.py
3. If kept: bash scripts/promote.sh <run_id>
4. git add + commit (agent pushes separately)

Together with run_experiment.sh, this is the entire per-experiment loop
the agent invokes. The agent's only manual work is editing train.py
and writing notes.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: First explore experiment end-to-end on Colab

**Files:** none — orchestration only.

This task validates the rails by running ONE explore experiment from start to finish on Colab — proving the loop works before the user kicks off the sustained sweep.

- [ ] **Step 1: Push commits**

```bash
cd /Users/apple/AutoLearnMeds && git push origin phase-0-plumbing
```

- [ ] **Step 2: Pull on Colab + verify file structure**

```bash
SSHPASS="$AUTOLEARNMEDS_SSH_PASSWORD" sshpass -e ssh -o ConnectTimeout=20 autolearnmeds-colab "
set -e
cd /workspace
git pull --ff-only origin phase-0-plumbing
ls -la scripts/append_ledger.py scripts/leaderboard.py scripts/promote.sh scripts/finalize_experiment.sh
"
```

Expected: 4 files exist, all executable.

- [ ] **Step 3: Seed the ledger with the 3 baseline entries from Phase 2**

The Phase-2 baseline runs (seeds 42, 43, 44) exist as run dirs but never went through `append_ledger.py`. Backfill them so the leaderboard has historical context. We'll mark them as `phase=baseline`.

```bash
SSHPASS="$AUTOLEARNMEDS_SSH_PASSWORD" sshpass -e ssh -o ConnectTimeout=20 autolearnmeds-colab "
set -e
cd /workspace
# Write minimal notes.md for each baseline run so append_ledger doesn't warn.
for s in 42 43 44; do
  RUN_DIR=experiments/runs/baseline-seed\${s}
  if [[ ! -f \$RUN_DIR/notes.md ]]; then
    cat > \$RUN_DIR/notes.md <<EOF
# Experiment: baseline-seed\${s}

## Hypothesis
Phase 2 baseline run with default config; establishes the macro_f1 floor.

## Citation
- kim_2022_donut.pdf §3.1 — decoder pattern; zhai_2023_siglip.pdf §3 — frozen encoder

## Adjacent ideas
N/A (baseline only; Phase 4 explore experiments ARE the adjacent ideas).

## Predicted direction
N/A — baseline.

## Verdict
- kept: true (seed 42 only — establishes the floor)
EOF
  fi
done
# Append all three to the ledger.
uv run python scripts/append_ledger.py baseline-seed42 --phase baseline
uv run python scripts/append_ledger.py baseline-seed43 --phase baseline
uv run python scripts/append_ledger.py baseline-seed44 --phase baseline
echo ''
echo '=== ledger now ==='
cat experiments/ledger.jsonl
echo ''
echo '=== regenerate leaderboard ==='
uv run python scripts/leaderboard.py
cat experiments/leaderboard.md
"
```

Expected: 3 ledger lines (one per seed); leaderboard shows seed 44 first (highest f1=0.0804), then seed 42 (0.0679), then seed 43 (0.0646).

- [ ] **Step 4: Run a real explore experiment — RandAugment M=5**

The agent's first hypothesis: image augmentation will reduce overfitting. Implement minimally in `train.py` by patching `prepare.preprocess_image` at import time (a hook the agent uses). Edit `train.py` to add a small augmentation wrapper.

Run from local Mac:

```bash
SSHPASS="$AUTOLEARNMEDS_SSH_PASSWORD" sshpass -e ssh -o ConnectTimeout=20 autolearnmeds-colab "
set -e
cd /workspace

# Apply a minimal RandAugment patch to train.py via sed-style edit. (In real
# autoresearch, the agent edits train.py directly; this task just shows the
# end-to-end loop works.)
RUN_ID=randaugment-m5-seed42

# Create the notes.md FIRST (per the workflow rule).
mkdir -p experiments/runs/\$RUN_ID
cat > experiments/runs/\$RUN_ID/notes.md <<EOF
# Experiment: \$RUN_ID

## Hypothesis
Adding RandAugment (M=5, N=2) to image preprocessing will reduce the train→val gap
that produced the 0.07 baseline ceiling.

## Citation
- cubuk_2020_randaugment.pdf §3 — augmentation operations and their composition.
  We use M=5 instead of the paper's M=9 because our 564-image training set is
  much smaller than ImageNet.

## Adjacent ideas (creative reading)
1. RandAugment paper also showed that the optimal M correlates with model
   capacity. Since our decoder is small (26.5M trainable), M=5 may be near-
   optimal; could sweep [3, 5, 7].
2. The paper's 'magnitude warmup' (start at M=0, ramp to target) could
   stabilize early training; not applied here (one knob per experiment).

## Predicted direction
+0.05 to +0.15 macro_f1 over the 0.0710 baseline, with biggest gains on the
fields that vary across photos of the same medicine (batch_number, mfg_date,
expiry_date — they have different values per photo even within one medicine).
EOF

# Apply the augmentation: patch train.py's image preprocessing path.
# In a real edit, the agent would use Edit tool; here we use a Python heredoc
# to patch the model class.
uv run python <<'PY'
from pathlib import Path

src = Path('train.py').read_text()
# Insert RandAugment import + apply at preprocess time.
patch = '''
# AGENT (randaugment-m5-seed42): applying RandAugment M=5 to images at training
# time. Cited: papers/cubuk_2020_randaugment.pdf §3.
def _augment(images):
    \"\"\"Apply RandAugment(M=5, N=2) to a batch of preprocessed image tensors.\"\"\"
    from torchvision.transforms.v2 import RandAugment
    aug = RandAugment(num_ops=2, magnitude=5)
    # Scale 0..1 -> 0..255 uint8 for RandAugment, then back.
    import torch
    img_uint8 = (images.clamp(-1, 1) + 1).mul(127.5).byte()
    augmented = aug(img_uint8).float().div(127.5).sub(1.0)
    return augmented
'''
# Insert right before the PharmaVLM class.
marker = '# === Model'
assert marker in src, 'could not find Model section banner'
src = src.replace(marker, patch + '\n\n' + marker)

# Now have model.forward apply _augment when training.
src = src.replace(
    'def forward(self, images, target_ids):',
    '''def forward(self, images, target_ids):
        # AGENT (randaugment-m5-seed42): augment in training mode only.
        torch_ = _import_torch()
        if self.proj.training:
            images = _augment(images)'''
)
Path('train.py').write_text(src)
print('patched')
PY

# Run the experiment.
echo '=== running randaugment-m5-seed42 ==='
nohup bash scripts/run_experiment.sh \$RUN_ID --config experiments/configs/baseline.yaml --seed 42 --no-wandb > /tmp/\$RUN_ID.log 2>&1 &
echo \"PID=\$!\"
"
```

This launches the experiment in the background. Wait for ~30 min (use Monitor / sleep).

- [ ] **Step 5: After completion, finalize and inspect**

Once `experiments/runs/randaugment-m5-seed42/metrics.json` exists on Colab:

```bash
SSHPASS="$AUTOLEARNMEDS_SSH_PASSWORD" sshpass -e ssh -o ConnectTimeout=20 autolearnmeds-colab "
set -e
cd /workspace
RUN_ID=randaugment-m5-seed42

echo '=== metrics ==='
cat experiments/runs/\$RUN_ID/metrics.json

# Update notes.md with Result + Verdict + Retrospective sections.
F1=\$(grep -oE 'final_macro_f1\":[^,]+' experiments/runs/\$RUN_ID/metrics.json | head -1 | grep -oE '[0-9.]+')
WALL=\$(grep -oE 'wall_clock_seconds\":[^,]+' experiments/runs/\$RUN_ID/metrics.json | head -1 | grep -oE '[0-9]+')
WALL_MIN=\$(echo \"scale=1; \$WALL / 60\" | bc)
cat >> experiments/runs/\$RUN_ID/notes.md <<EOF

## Result
- final_macro_f1: \$F1
- wall_clock_min: \$WALL_MIN
- trainable_params: 26542592 (unchanged; augmentation is data-side only)

## Retrospective
(Filled by agent after reviewing the result against the prediction.)
EOF

echo ''
echo '=== finalize ==='
bash scripts/finalize_experiment.sh \$RUN_ID --phase explore --parent baseline-seed42

echo ''
echo '=== leaderboard ==='
cat experiments/leaderboard.md
"
```

- [ ] **Step 6: If macro_f1 > 0.0710 + 0.001, the rails work.**

Document the result. Whether the experiment is kept or reverted, the lifecycle ran end-to-end:
- run_experiment.sh wrote metrics.json
- append_ledger.py read it, decided kept/reverted, appended ledger
- leaderboard.py regenerated leaderboard.md
- (if kept) promote.sh copied train.py + best.pt to checkpoints/best/
- finalize_experiment.sh did `git add + commit`

Push the result:

```bash
SSHPASS="$AUTOLEARNMEDS_SSH_PASSWORD" sshpass -e ssh -o ConnectTimeout=20 autolearnmeds-colab "
cd /workspace
git push origin phase-0-plumbing 2>&1 | tail -3
"
```

(If git fails on Colab due to credentials, scp the relevant files back to Mac and push from there — the same pattern we used in Phase 1.)

---

## Task 8: Phase 4-rails review + tag

**Files:**
- Create: `docs/superpowers/reviews/phase4a_review.md`

- [ ] **Step 1: Write the review**

Create `/Users/apple/AutoLearnMeds/docs/superpowers/reviews/phase4a_review.md`:

```markdown
# Phase 4a Review — Explore-Phase Rails

**Tag:** `phase-4a-rails-complete`
**Date:** YYYY-MM-DD (fill at commit)
**Status:** GREEN — explore loop runs end-to-end on Colab; one experiment ran via the new rails.

## What was built

- `program.md` — full agent contract (10 sections including the notes.md template + creative-reading discipline). Replaces the Phase-0 skeleton.
- `program_explore.md` — explore-phase override: 15-min budget, +0.001 keep threshold, 9-item priority backlog from Phase 2 review.
- `train.py` — train_loop now saves a full state dict (proj + decoder) at best macro_f1, not just metadata.
- `scripts/append_ledger.py` — atomic ledger append; reads metrics.json + notes.md, computes kept/reverted vs prior best.
- `scripts/leaderboard.py` — regenerates `experiments/leaderboard.md` from ledger.jsonl.
- `scripts/promote.sh` — copies winning train.py + best.pt to checkpoints/best/.
- `scripts/finalize_experiment.sh` — orchestrates the three above + git commit.
- 11 lightweight tests (5 append_ledger, 3 leaderboard, 2 smoke for promote+finalize, 1 program.md).

## What was verified

- All 66+ lightweight pytest tests pass locally.
- All colab-marked tests pass on Colab.
- Phase 2's baseline runs (seed 42/43/44) backfilled into the ledger as `phase=baseline`.
- One real explore experiment (`randaugment-m5-seed42`) ran end-to-end through the rails: ran → metrics.json → append_ledger → leaderboard regen → (if kept) promote → git commit.
- Leaderboard shows the experiment ranked correctly among the baselines.

## Open items

- The agent's iteration loop is now manual: human-driven Claude Code session reads program.md, edits train.py, calls scripts. A self-driving loop (via `loop` skill or `schedule` skill) is a Phase-4b decision.
- `RESUME_NEEDED.md` write-on-quota-approach not yet implemented; the agent doesn't introspect Claude rate limits.
- Crash-mid-experiment recovery (resume from in-flight `_active.json`) still deferred — train.py doesn't write `_active.json`. Wire-up for full resume is Phase-4b nice-to-have.

## Next step

**Phase 4b — actual explore sweep.** Open a Claude Code session pointed at this repo and prompt:
> "Read program.md and program_explore.md. Run the next experiment from the priority backlog. Use the workflow per experiment exactly."

The agent then iterates ~30 times. Each iteration is one ledger entry. Phase 4b's exit criterion: ≥1 confirmed (3-seed) win that beats baseline by ≥0.02 macro_f1 — that triggers Phase 5 (confirm phase).
```

- [ ] **Step 2: Commit, push, tag**

```bash
cd /Users/apple/AutoLearnMeds
git add docs/superpowers/reviews/phase4a_review.md
git commit -m "docs: phase 4a review — explore-phase rails complete

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
git tag -a phase-4a-rails-complete -m "Phase 4a explore-phase rails complete; first experiment ran end-to-end"
git push origin phase-0-plumbing phase-4a-rails-complete
```

---

## Self-Review

**Spec coverage check** (against spec §5 Autoresearch Harness + §9.4 Phase 4):

| Spec requirement | Implemented in task |
|---|---|
| §5.3 program.md skeleton expanded | T1 |
| §5.3 program_explore.md | T1 |
| §5.4 ledger.jsonl with full schema | T3 |
| §8.2 notes.md template | T1 (in program.md) |
| §5.5 keep-threshold (explore: +0.001) | T3 |
| §5.6 stop after K consecutive non-improvements | (deferred — agent-driven, not script-driven) |
| §5.7 git fail-safe per experiment | T6 (finalize commits per run) |
| §9.4 program_explore.md finalized | T1 |
| §9.4 first explore experiment runs cleanly | T7 |
| §9.4 ledger entry per experiment | T3 |
| §9.4 RESUME_NEEDED.md flow tested | (deferred — not blocking; manual user re-run) |

Most §5 / §9.4 requirements covered. Two are deferred to Phase 4b (the actual sweep work) since they're agent-behavior concerns, not script-build concerns.

**Placeholder scan:** the Phase-4a-review template uses `YYYY-MM-DD` — runtime fill. No real placeholders.

**Type consistency:**
- `append_ledger(run_id, workspace, phase, parent_run_id, keep_threshold)` signature matches the call sites in `finalize_experiment.sh` and the test.
- `regenerate(workspace, out_path)` in leaderboard.py matches its CLI invocation.
- The ledger entry schema's keys (`run_id`, `phase`, `metrics`, `kept`, etc.) match between append_ledger.py (writer) and leaderboard.py (reader).

---

## Phase 4a → Phase 4b transition

When this plan is fully executed and tagged `phase-4a-rails-complete`:

1. The full agent contract is documented in `program.md` + `program_explore.md`.
2. Every experiment lifecycle (edit → run → ledger → leaderboard → promote → commit) is one bash invocation chain.
3. The first explore experiment landed cleanly in the ledger.
4. Phase 4b is open-ended: Claude Code reads the contract, runs ~30 experiments, the leaderboard climbs.

There's no Phase 4b plan document because the work isn't scriptable in advance — that's the whole point of autoresearch.
