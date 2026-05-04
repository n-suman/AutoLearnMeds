# Karpathy 2026 — autoresearch

**Cite key:** `karpathy2026autoresearch`
**Type:** GitHub repository + Twitter/X threads (no formal paper)
**Repo:** https://github.com/karpathy/autoresearch
**Author:** Andrej Karpathy
**Year:** 2026 (March)
**Tweets referenced:** linked from the repo README

## What it is

A minimal scaffold for letting an AI agent autonomously run small ML research experiments overnight. The setup:

- Three files matter: `prepare.py` (fixed — data + utilities), `train.py` (agent edits — model + training loop), `program.md` (human edits — agent's instructions).
- The agent reads `program.md`, modifies `train.py`, runs `uv run train.py` for a fixed wall-clock budget (5 minutes on H100), reads a single scalar metric (`val_bpb` — validation bits per byte), and decides whether to keep the change.
- Wins are kept; losses are reverted. Everything is logged.
- The "research org code" is `program.md` itself — the human iterates on it over time, not on the Python files.
- No human edits to `train.py` during the loop. The agent owns the whole training stack: architecture, hyperparameters, optimizer, batch size, augmentation, all of it.

## Quote (from README)

> "The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model."

> "By design, training runs for a fixed 5-minute time budget (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is val_bpb (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared."

## How AutoLearnMeds adapts the methodology

| autoresearch (Karpathy 2026) | AutoLearnMeds (this project) |
|---|---|
| Single-GPU H100, 5-min experiments | Colab Pro+ A100, 15-min explore + 60-min confirm (mixed regime) |
| nanochat — small from-scratch GPT | SigLIP-base (frozen) + small from-scratch Donut-style decoder |
| `val_bpb` (lower is better) | `val_macro_f1` (higher is better) |
| Single agent (Claude Code) edits `train.py` | Same — Claude Code via VSCode SSH to Colab |
| `program.md` is the agent's spec | Same; we add `program_explore.md` and `program_confirm.md` for phased rollout |
| No persistent state design (single-machine, single-run) | Three-way redundant persistence (GDrive + GCS + GitHub), atomic experiment unit, resume-or-start state machine |
| No source-vetting requirement | Source-vetting required: every change cites a paper in `papers/` |
| No paper-generation pipeline | Auto-generated tables/figures from ledger; Phase 6 produces a buildable LaTeX paper |

## Why this is the right reference for our project

The project's design (this design spec) is a direct adaptation of Karpathy's autoresearch methodology to the pharmaceutical-VLM domain. The "autoresearch loop" terminology, the three-file discipline, the single-scalar-metric optimization, the keep-or-revert decision rule — all come from this repository.

## Citation in BibTeX

```bibtex
@misc{karpathy2026autoresearch,
  author       = {Andrej Karpathy},
  title        = {autoresearch: AI agents running research on single-GPU nanochat training automatically},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/karpathy/autoresearch}},
  note         = {Accessed 2026-05-05}
}
```
