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
