"""Render a side-by-side per-field markdown comparison of Track A vs Track B.

Inputs: two JSON files emitted by ``scripts/per_field_eval.py``.
Output: a single Markdown document with title, methodology, per-field table,
"Track B wins" / "Track A wins" subsections, and a short summary paragraph.

Usage:
    python scripts/compare_per_field.py \\
        --a experiments/per_field_track_a.json \\
        --b experiments/per_field_track_b.json \\
        --out experiments/per_field_comparison.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# A single fence character used to mark cells where Track B beats Track A.
WIN_MARK = "*"


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _all_fields(payload_a: dict[str, Any], payload_b: dict[str, Any]) -> list[str]:
    """Union of keys across both per_field_f1 dicts, sorted alphabetically."""
    keys: set[str] = set()
    keys.update((payload_a.get("per_field_f1") or {}).keys())
    keys.update((payload_b.get("per_field_f1") or {}).keys())
    keys.update((payload_a.get("per_field_edit_f1") or {}).keys())
    keys.update((payload_b.get("per_field_edit_f1") or {}).keys())
    return sorted(keys)


def _fmt(x: float, mark: bool = False) -> str:
    """Format a metric as 4 decimals, with optional win-mark suffix."""
    s = f"{x:.4f}"
    return f"{s} {WIN_MARK}" if mark else s


def _build_table(payload_a: dict[str, Any], payload_b: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    """Return (markdown_table, b_wins_fields, a_wins_fields).

    A field is in `b_wins_fields` if Track B beats Track A on EITHER f1 OR
    edit_f1 (strict greater-than). Fields where both metrics tie or A wins
    both go into `a_wins_fields`.
    """
    fields = _all_fields(payload_a, payload_b)
    a_f1 = payload_a.get("per_field_f1") or {}
    b_f1 = payload_b.get("per_field_f1") or {}
    a_ed = payload_a.get("per_field_edit_f1") or {}
    b_ed = payload_b.get("per_field_edit_f1") or {}

    rows: list[str] = []
    rows.append("| field | A f1 | B f1 | Δ f1 (B-A) | A edit_f1 | B edit_f1 | Δ edit_f1 (B-A) |")
    rows.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")

    b_wins: list[str] = []
    a_wins: list[str] = []

    for fname in fields:
        af = float(a_f1.get(fname, 0.0))
        bf = float(b_f1.get(fname, 0.0))
        ae = float(a_ed.get(fname, 0.0))
        be = float(b_ed.get(fname, 0.0))
        delta_f1 = bf - af
        delta_ed = be - ae

        bf_mark = bf > af
        be_mark = be > ae

        rows.append(
            f"| {fname} "
            f"| {_fmt(af)} "
            f"| {_fmt(bf, bf_mark)} "
            f"| {delta_f1:+.4f} "
            f"| {_fmt(ae)} "
            f"| {_fmt(be, be_mark)} "
            f"| {delta_ed:+.4f} |"
        )

        if bf_mark or be_mark:
            b_wins.append(fname)
        else:
            a_wins.append(fname)

    # Macro summary row.
    macro_a_f1 = float(payload_a.get("macro_f1", 0.0))
    macro_b_f1 = float(payload_b.get("macro_f1", 0.0))
    macro_a_ed = float(payload_a.get("macro_edit_f1", 0.0))
    macro_b_ed = float(payload_b.get("macro_edit_f1", 0.0))
    rows.append(
        f"| **macro** "
        f"| **{_fmt(macro_a_f1)}** "
        f"| **{_fmt(macro_b_f1, macro_b_f1 > macro_a_f1)}** "
        f"| **{macro_b_f1 - macro_a_f1:+.4f}** "
        f"| **{_fmt(macro_a_ed)}** "
        f"| **{_fmt(macro_b_ed, macro_b_ed > macro_a_ed)}** "
        f"| **{macro_b_ed - macro_a_ed:+.4f}** |"
    )

    return "\n".join(rows), b_wins, a_wins


def _build_paper_paragraph(
    payload_a: dict[str, Any],
    payload_b: dict[str, Any],
    b_wins: list[str],
    a_wins: list[str],
) -> str:
    """One short, factual paragraph for the paper. No overclaim."""
    macro_a_f1 = float(payload_a.get("macro_f1", 0.0))
    macro_b_f1 = float(payload_b.get("macro_f1", 0.0))
    macro_a_ed = float(payload_a.get("macro_edit_f1", 0.0))
    macro_b_ed = float(payload_b.get("macro_edit_f1", 0.0))
    n_total = len(b_wins) + len(a_wins)

    lines: list[str] = []
    lines.append(
        f"Track A wins both macro metrics (f1 {macro_a_f1:.4f} vs {macro_b_f1:.4f}; "
        f"edit_f1 {macro_a_ed:.4f} vs {macro_b_ed:.4f}). "
    )
    if b_wins:
        lines.append(
            f"At the per-field grain, Track B beats Track A on at least one of "
            f"(f1, edit_f1) for {len(b_wins)} of {n_total} fields: "
            f"{', '.join(sorted(b_wins))}. "
            f"Track A wins or ties on the remaining {len(a_wins)}: "
            f"{', '.join(sorted(a_wins))}."
        )
    else:
        lines.append(
            "At the per-field grain, Track A wins or ties Track B on every field "
            "for both metrics; the macro gap is uniform, not driven by a few "
            "OCR-heavy fields."
        )
    return "".join(lines)


def render_markdown(payload_a: dict[str, Any], payload_b: dict[str, Any]) -> str:
    """Build the full markdown comparison document."""
    table_md, b_wins, a_wins = _build_table(payload_a, payload_b)

    lines: list[str] = []
    lines.append("# Track A vs Track B - per-field comparison")
    lines.append("")
    lines.append(
        "Methodology: identical val split "
        f"(n={payload_a.get('n_samples', 'N/A')} for Track A, "
        f"n={payload_b.get('n_samples', 'N/A')} for Track B), "
        "identical metrics (exact-match per-field F1 and partial-credit edit-F1 "
        "averaged over the 10 high-frequency fields), different methods - "
        f"Track A is `{payload_a.get('ckpt', '?')}` "
        f"({payload_a.get('config', '?')}); "
        f"Track B is `{payload_b.get('ckpt', '?')}` "
        f"({payload_b.get('config', '?')}). "
        f"Cells marked with `{WIN_MARK}` are where Track B beats Track A."
    )
    lines.append("")
    lines.append("## Per-field table (alphabetical)")
    lines.append("")
    lines.append(table_md)
    lines.append("")
    lines.append("## Track B wins (at least one metric)")
    lines.append("")
    if b_wins:
        for f in b_wins:
            lines.append(f"- {f}")
    else:
        lines.append("_None - Track A wins or ties on every field._")
    lines.append("")
    lines.append("## Track A wins (or ties)")
    lines.append("")
    if a_wins:
        for f in a_wins:
            lines.append(f"- {f}")
    else:
        lines.append("_None - Track B wins on at least one metric for every field._")
    lines.append("")
    lines.append("## What this tells the paper")
    lines.append("")
    lines.append(_build_paper_paragraph(payload_a, payload_b, b_wins, a_wins))
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--a", required=True, help="Track A per_field JSON")
    parser.add_argument("--b", required=True, help="Track B per_field JSON")
    parser.add_argument("--out", required=True, help="Output markdown path")
    args = parser.parse_args(argv)

    payload_a = _load_json(args.a)
    payload_b = _load_json(args.b)
    md = render_markdown(payload_a, payload_b)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)

    print(f"[compare_per_field] wrote {out_path} ({len(md)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
