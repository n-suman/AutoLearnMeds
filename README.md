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
