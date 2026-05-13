"""9-panel qualitative grid: 3 each of still-wrong / fixed-by-E / regressed predictions.

Input: a JSON list of {category, image_path, field, gold, pred} (3+ per category).
Output: a 3x3 PNG. If image_path can't be loaded, draws a placeholder rectangle with the text.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


CATEGORIES = ["still_wrong", "fixed_by_e", "regressed"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = json.loads(Path(args.samples).read_text())
    by_cat: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    for r in rows:
        if r["category"] in by_cat:
            by_cat[r["category"]].append(r)

    fig, axes = plt.subplots(3, 3, figsize=(11, 9))
    for col, cat in enumerate(CATEGORIES):
        for row in range(3):
            ax = axes[row, col]
            ax.axis("off")
            entries = by_cat.get(cat, [])
            if row < len(entries):
                e = entries[row]
                try:
                    img = Image.open(e["image_path"]).convert("RGB")
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, f"[image\nmissing]", ha="center", va="center", transform=ax.transAxes)
                title = f"{cat}\n{e['field']}\nGT: {e['gold']}\nPred: {e['pred']}"
                ax.set_title(title, fontsize=8)
    fig.suptitle("Error samples by category (3 each)", fontsize=12)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
