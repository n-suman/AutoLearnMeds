"""Coverage-vs-accuracy at varying entropy thresholds.

Input: a JSON list of {field, correct (bool), entropy (float)} per prediction.
Output: PNG showing two curves — coverage and accuracy — as the entropy threshold sweeps.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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
    sys.exit(main())
