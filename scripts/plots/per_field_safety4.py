"""Bar chart restricted to the 4 safety-critical fields, multi-track.

Usage:
    python scripts/plots/per_field_safety4.py \
        --per-field track_a=experiments/per_field_track_a.json \
        --per-field track_e=experiments/per_field_track_e.json \
        --metric edit \
        --out paper/figures/safety4_cross_track.png
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from prepare import SAFETY_FIELDS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-field", nargs="+", required=True,
                    help="name=path/to/per_field.json (one or more)")
    ap.add_argument("--metric", choices=["strict", "edit"], default="edit")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fields = sorted(SAFETY_FIELDS)
    tracks = {}
    for spec in args.per_field:
        name, path = spec.split("=", 1)
        data = json.loads(Path(path).read_text())
        key = "per_field_edit_f1" if args.metric == "edit" else "per_field_f1"
        per = data.get(key, {})
        tracks[name] = [per.get(f, 0.0) for f in fields]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(fields))
    w = 0.8 / max(1, len(tracks))
    for i, (name, vals) in enumerate(tracks.items()):
        ax.bar(x + i * w, vals, width=w, label=name)
    ax.set_xticks(x + (len(tracks) - 1) * w / 2)
    ax.set_xticklabels(fields, rotation=20)
    ax.set_ylabel(f"F1 ({args.metric})")
    ax.set_title("Safety-4 fields — per-track")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
