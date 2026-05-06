#!/usr/bin/env python3
"""Render a pharma label image with polygon annotations overlaid.

Reads `gold_standard.jsonl` (each line = one labeled image with `fields` →
{name, polygon: [[x%, y%], ...], text}), looks up the entry by image_file,
and draws each field's polygon on the source image with a per-field color
+ a label box showing the field name and (optionally) the OCR'd text.

Usage:
    uv run --extra dev python scripts/plots/annotate_image.py \\
        --gold /tmp/gold.jsonl \\
        --image-file IMG_5246.jpeg \\
        --images-dir /tmp/sample_images \\
        --out paper/figures/F8_strip_foil_annotated.png \\
        [--show-text]

Polygon coordinates are in [0, 100] (percentage of image width / height).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Distinctive per-field colors (matplotlib-friendly tab/tableau palette + extras).
FIELD_COLORS = {
    "brand_name":   "#e41a1c",  # red
    "generic_name": "#377eb8",  # blue
    "drug_name":    "#984ea3",  # purple
    "strength":     "#ff7f00",  # orange
    "quantity":     "#00ced1",  # cyan
    "company":      "#4daf4a",  # green
    "manufacturer": "#a6d854",  # lime
    "batch_number": "#ffff33",  # yellow
    "mfg_date":     "#ff69b4",  # pink
    "expiry_date":  "#e7298a",  # magenta
    "mrp":          "#a65628",  # brown
    "warnings":     "#666666",  # gray
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gold", type=Path, required=True,
                        help="path to gold_standard.jsonl")
    parser.add_argument("--image-file", type=str, required=True,
                        help="image_file value to look up (e.g. IMG_5246.jpeg)")
    parser.add_argument("--images-dir", type=Path, required=True,
                        help="directory containing the actual image files")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--show-text", action="store_true",
                        help="if set, label each polygon with both field name AND the OCR'd text")
    parser.add_argument("--max-text-chars", type=int, default=20,
                        help="truncate displayed text to this many characters")
    args = parser.parse_args()

    # Load gold entry.
    entry = None
    for line in args.gold.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        if e.get("image_file") == args.image_file:
            entry = e
            break
    if entry is None:
        print(f"FATAL: image_file={args.image_file} not found in {args.gold}", file=sys.stderr)
        return 2

    image_path = args.images_dir / args.image_file
    if not image_path.is_file():
        print(f"FATAL: image not found at {image_path}", file=sys.stderr)
        return 2

    # Heavy imports inside main.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    fig, ax = plt.subplots(figsize=(8, 8 * H / W))
    ax.imshow(img)
    ax.set_axis_off()

    fields = entry.get("fields", {})
    drawn_fields: list[str] = []
    for field_name, field_data in fields.items():
        polygon = field_data.get("polygon")
        if not polygon:
            continue
        # Convert from [0, 100] percentage to pixel coords.
        pixel_coords = [(x * W / 100.0, y * H / 100.0) for x, y in polygon]
        color = FIELD_COLORS.get(field_name, "#000000")
        # Outline-only polygon (thin), no fill — keeps the underlying photo readable.
        poly = Polygon(pixel_coords, closed=True, fill=False, edgecolor=color, linewidth=2.5)
        ax.add_patch(poly)
        # Label at the top-left vertex.
        x_label, y_label = pixel_coords[0]
        label = field_name
        if args.show_text:
            text = field_data.get("text", "").replace("\n", " ").strip()
            if len(text) > args.max_text_chars:
                text = text[: args.max_text_chars - 1] + "…"
            label = f"{field_name}: {text}"
        ax.text(
            x_label,
            y_label - 8,
            label,
            color="white",
            fontsize=9,
            fontweight="bold",
            bbox=dict(facecolor=color, edgecolor="black", boxstyle="round,pad=0.3", alpha=0.9),
            verticalalignment="bottom",
        )
        drawn_fields.append(field_name)

    pack_type = entry.get("pack_type", "unknown")
    medicine = entry.get("medicine_name", "")
    title = f"{args.image_file}  ·  pack_type={pack_type}  ·  {medicine}"
    ax.set_title(title, fontsize=10)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"[annotate_image] wrote {args.out} (pack_type={pack_type}, fields={drawn_fields})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
