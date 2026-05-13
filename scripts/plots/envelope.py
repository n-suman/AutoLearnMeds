"""Running-envelope plot: safety-4 macro_edit_f1 across all runs in the ledger, over time.

Usage:
    python scripts/plots/envelope.py --ledger experiments/ledger.jsonl --out paper/figures/_envelope.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    with open(args.ledger) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            value = r.get("metrics", {}).get("safety4_macro_edit_f1")
            if value is None:
                continue
            rows.append({
                "run_id": r.get("run_id", "?"),
                "value": value,
                "ts": r.get("timestamp", ""),
            })
    rows.sort(key=lambda r: r["ts"])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = list(range(len(rows)))
    ys = [r["value"] for r in rows]
    envelope = [max(ys[:i+1]) for i in range(len(ys))]
    ax.plot(xs, ys, "o-", label="per-run safety-4 macro_edit_f1", alpha=0.6)
    ax.plot(xs, envelope, "s--", color="red", label="running envelope (best so far)")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["run_id"] for r in rows], rotation=30, ha="right")
    ax.set_ylabel("safety-4 macro_edit_f1")
    ax.set_title("Running envelope across runs")
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
