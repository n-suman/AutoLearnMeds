"""One-off: compute safety-4 macro F1 + macro_edit_F1 for every existing per-field JSON.

Reads `experiments/per_field_*.json` (one per track / variant), restricts each to
the four SAFETY_FIELDS, and writes a consolidated baseline file. Track E uses this
as the anchor comparison.

Usage:
    python scripts/backfill_safety4_baseline.py [--out experiments/safety4_baseline.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prepare import SAFETY_FIELDS  # noqa: E402


def backfill_one(per_field_path: Path) -> dict[str, float | dict]:
    """Read a per_field_*.json (shape: {"per_field_f1": {f: x}, "per_field_edit_f1": {f: y}})
    and return the safety-4 subset macros."""
    data = json.loads(per_field_path.read_text())
    pf_strict = data.get("per_field_f1", {})
    pf_edit = data.get("per_field_edit_f1", {})

    safety_strict = {f: pf_strict.get(f, 0.0) for f in SAFETY_FIELDS}
    safety_edit = {f: pf_edit.get(f, 0.0) for f in SAFETY_FIELDS}

    return {
        "source_file": per_field_path.name,
        "safety4_macro_f1": sum(safety_strict.values()) / len(SAFETY_FIELDS),
        "safety4_macro_edit_f1": sum(safety_edit.values()) / len(SAFETY_FIELDS),
        "per_field": {
            f: {"strict": safety_strict[f], "edit": safety_edit[f]}
            for f in SAFETY_FIELDS
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-field-dir", default="experiments", help="Directory containing per_field_*.json")
    parser.add_argument("--out", default="experiments/safety4_baseline.json")
    args = parser.parse_args()

    pf_dir = Path(args.per_field_dir)
    out = Path(args.out)

    results: dict[str, dict] = {}
    for pf_file in sorted(pf_dir.glob("per_field_*.json")):
        key = pf_file.stem.replace("per_field_", "")  # e.g. "track_a"
        results[key] = backfill_one(pf_file)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"Wrote {out} with {len(results)} tracks")
    for key, entry in results.items():
        print(f"  {key:50s}  safety4 macro_edit_f1 = {entry['safety4_macro_edit_f1']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
