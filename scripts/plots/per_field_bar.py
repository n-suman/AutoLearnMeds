#!/usr/bin/env python3
"""Per-field F1 bar chart, one or more tracks side-by-side.

Inputs: one or more per_field_track_*.json files.
Output: a single PNG bar chart with grouped bars per field.

Usage:
    uv run python scripts/plots/per_field_bar.py \\
        --inputs experiments/per_field_track_a.json experiments/per_field_track_b.json \\
        --labels "Track A" "Track B" \\
        --metric macro_f1 \\
        --out paper/figures/F1_per_field_strict.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", type=str, required=True)
    parser.add_argument("--metric", choices=["per_field_f1", "per_field_edit_f1"], default="per_field_f1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    if len(args.inputs) != len(args.labels):
        print("FATAL: --inputs and --labels must have the same count", file=sys.stderr)
        return 2

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Load all inputs.
    data = []
    for p, lab in zip(args.inputs, args.labels):
        d = json.loads(p.read_text())
        data.append((lab, d.get(args.metric, {})))

    # Union of fields across all inputs (sorted alphabetically).
    fields = sorted({k for _, d in data for k in d})

    n_tracks = len(data)
    n_fields = len(fields)
    width = 0.8 / n_tracks  # group width 0.8

    fig, ax = plt.subplots(figsize=(max(8, n_fields * 0.7), 4.5))
    x = np.arange(n_fields)
    for i, (lab, d) in enumerate(data):
        values = [d.get(f, 0.0) for f in fields]
        offset = (i - (n_tracks - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=lab)

    ax.set_xticks(x)
    ax.set_xticklabels(fields, rotation=45, ha="right")
    ax.set_ylabel("Strict F1" if args.metric == "per_field_f1" else "Edit-distance F1")
    ax.set_ylim(0, 1.0)
    ax.legend()
    title = args.title or f"Per-field {args.metric.replace('per_field_', '').replace('_', ' ')} by track"
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"[per_field_bar] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
