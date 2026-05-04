# paper/

The research paper itself.

- `main.tex` — top-level document
- `refs.bib` — auto-generated from `papers/README.md` by `scripts/build_bibtex.py`
- `figures/` — auto-generated PDFs from `scripts/build_paper_artifacts.py`
- `tables/` — auto-generated `.tex` tables
- `sections/` — modular `.tex` files (intro, related work, method, etc.)

Tables and figures are auto-generated from `experiments/ledger.jsonl`. Prose lives in `main.tex` and `sections/`.
