# Experiment: randaug-m5-seed42

## Hypothesis
Adding RandAugment (M=5, N=2) to image preprocessing will reduce the train→val gap
that produced the 0.07 baseline ceiling.

## Citation
- cubuk_2020_randaugment.pdf §3 — augmentation operations and their composition.
  We use M=5 instead of the paper's M=9 because our 564-image training set is
  much smaller than ImageNet.

## Adjacent ideas (creative reading)
1. RandAugment paper showed optimal M correlates with model capacity. Since our
   decoder is small (26.5M trainable), M=5 may be near-optimal; could sweep [3, 5, 7].
2. The paper's 'magnitude warmup' (start M=0, ramp to target) could stabilize early
   training; not applied here (one knob per experiment).

## Predicted direction
+0.05 to +0.15 macro_f1 over the 0.0710 baseline mean, with biggest gains on
fields that vary across photos of the same medicine (batch_number, mfg_date,
expiry_date have different values per photo even within one medicine).

## Result
- final_macro_f1: 1
0.0679
- wall_clock_min: 27.9
- trainable_params: 26542592 (unchanged; augmentation is data-side only)

## Retrospective
RandAugment M=5 with N=2 produced **no improvement** over baseline (0.0679 vs
baseline-seed42's 0.0679 = exactly the same number, suggestive of training-noise
floor rather than a real effect). Hypothesis predicted +0.05 to +0.15; actual
delta is 0.0. Plausible reasons: (a) M=5 too weak for this regime, (b) some
RandAugment ops (color jitter) may corrupt color-coded packaging features
that distinguish medicines, (c) 1000 steps may be too few to see augmentation's
regularization benefit relative to memorization. Next experiment: try larger
M (8-10) OR a per-op subset that excludes color/contrast manipulations.
