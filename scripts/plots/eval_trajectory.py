#!/usr/bin/env python3
"""Eval trajectory plot from a stdout.log.

Parses lines matching `[ val ] step= NNN macro_f1=X.XXXX` (Track A) or
`[track-b] step=NNN val_macro_f1=X.XXXX val_macro_edit_f1=X.XXXX` (Track B)
or `[mae] step=N/M loss=X.XXXX` (MAE) and produces a line plot.

Usage:
    uv run python scripts/plots/eval_trajectory.py \\
        --log experiments/runs/baseline-seed44-rerun/stdout.log \\
        --out paper/figures/F4_track_a_trajectory.png
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def parse_log(text: str) -> dict[str, list[tuple[float, float]]]:
    """Returns {series_name: [(step, value), ...]}."""
    series: dict[str, list[tuple[float, float]]] = {}

    # Track A val: `[ val ] step=  400 macro_f1=0.0374 (n=111)`
    for m in re.finditer(r"\[ val \] step=\s*(\d+)\s+macro_f1=([0-9.]+)", text):
        series.setdefault("val_macro_f1 (Track A strict)", []).append((float(m.group(1)), float(m.group(2))))

    # Track B val: `[track-b] step=400 val_macro_f1=0.0159 val_macro_edit_f1=0.1165`
    for m in re.finditer(r"\[track-b\] step=(\d+) val_macro_f1=([0-9.]+) val_macro_edit_f1=([0-9.]+)", text):
        s = float(m.group(1))
        series.setdefault("val_macro_f1 (Track B strict)", []).append((s, float(m.group(2))))
        series.setdefault("val_macro_edit_f1 (Track B lenient)", []).append((s, float(m.group(3))))

    # Track A train (loss): `[train] step=  400 loss=0.2447 lr=2.32e-04`
    for m in re.finditer(r"\[train\] step=\s*(\d+)\s+loss=([0-9.]+)", text):
        series.setdefault("train_loss (Track A)", []).append((float(m.group(1)), float(m.group(2))))

    # MAE train: `[mae] step=1250/8600 loss=0.4342 lr=1.49e-04 elapsed=3810s`
    for m in re.finditer(r"\[mae\] step=(\d+)/\d+ loss=([0-9.]+)", text):
        series.setdefault("train_loss (MAE)", []).append((float(m.group(1)), float(m.group(2))))

    return series


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--logy", action="store_true", help="log scale on y-axis")
    args = parser.parse_args()

    text = args.log.read_text()
    series = parse_log(text)
    if not series:
        print(f"[eval_trajectory] no series parsed from {args.log}", file=sys.stderr)
        return 2

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Split series into "loss-like" (any range) and "f1-like" (0-1 range) to
    # use twin axes — a train_loss starting at 382 would otherwise dwarf the
    # val_f1 (0-1) curve into a flat line near zero.
    loss_series = {n: pts for n, pts in series.items() if "loss" in n.lower()}
    f1_series = {n: pts for n, pts in series.items() if n not in loss_series}

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Left axis: f1 metrics (0-1).
    f1_handles = []
    for name, points in f1_series.items():
        steps = [s for s, _ in points]
        vals = [v for _, v in points]
        h, = ax.plot(steps, vals, marker="o", linewidth=1.6, label=name, color="tab:blue" if "strict" in name.lower() else "tab:green")
        f1_handles.append(h)
    if f1_series:
        ax.set_ylabel("F1 (0-1 range)")
        ax.set_ylim(0, max(0.5, max(v for pts in f1_series.values() for _, v in pts) * 1.15))

    # Right axis: loss (free range, log-scale optional).
    loss_handles = []
    if loss_series:
        ax2 = ax.twinx()
        for name, points in loss_series.items():
            steps = [s for s, _ in points]
            vals = [v for _, v in points]
            h, = ax2.plot(steps, vals, linewidth=1.2, alpha=0.7, label=name, color="tab:orange")
            loss_handles.append(h)
        ax2.set_ylabel("Training loss (right axis)", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        if args.logy:
            ax2.set_yscale("log")

    ax.set_xlabel("Training step")
    title = args.title or f"Trajectory: {args.log.parent.name}"
    ax.set_title(title)
    handles = f1_handles + loss_handles
    if handles:
        ax.legend(handles, [h.get_label() for h in handles], loc="best", fontsize=9)
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"[eval_trajectory] wrote {args.out} from {args.log} ({sum(len(v) for v in series.values())} points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
