#!/usr/bin/env python3
"""Append one entry to experiments/ledger.jsonl after run_experiment.sh finishes.

Reads:
  experiments/runs/<run_id>/metrics.json   (mandatory)
  experiments/runs/<run_id>/notes.md       (optional but warned)
  experiments/runs/<run_id>/config.yaml    (snapshot of hyperparameters)

Computes:
  best_at_time_of_run  = max final_macro_f1 of all prior kept entries (or 0)
  kept                 = (this final_macro_f1 > best_at_time_of_run + keep_threshold)
  rationale            = human-readable comparison string
  hypothesis, citation = extracted from notes.md by simple heading lookup

Writes one JSON-line to experiments/ledger.jsonl. Append-only.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path
from typing import Any


def _read_notes_section(text: str, heading: str) -> str:
    """Extract the body under '## <heading>' up to the next '## ' or EOF."""
    pat = re.compile(
        r"^##\s+" + re.escape(heading) + r"\s*\n(.*?)(?=^##\s|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    m = pat.search(text)
    return m.group(1).strip() if m else ""


def _parse_notes(notes_path: Path) -> dict[str, Any]:
    """Return {hypothesis, citation:list[str]} from notes.md, or '(missing)' values."""
    if not notes_path.is_file():
        return {"hypothesis": "(missing notes.md)", "citation": []}
    text = notes_path.read_text()
    hypothesis = _read_notes_section(text, "Hypothesis")
    if not hypothesis:
        hypothesis = "(missing Hypothesis section)"
    cit_block = _read_notes_section(text, "Citation")
    citations = [
        line.lstrip("- ").strip()
        for line in cit_block.splitlines()
        if line.strip().startswith("-")
    ]
    return {"hypothesis": hypothesis, "citation": citations}


def _current_best(ledger_path: Path) -> float:
    """Max final_macro_f1 across all prior `kept: true` entries; 0.0 if empty."""
    if not ledger_path.is_file():
        return 0.0
    best = 0.0
    for line in ledger_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("kept"):
            f1 = (entry.get("metrics") or {}).get("final_macro_f1", 0.0)
            if f1 > best:
                best = f1
    return best


def append_ledger(
    run_id: str,
    workspace: Path | str,
    phase: str = "explore",
    parent_run_id: str = "",
    keep_threshold: float = 0.001,
) -> dict[str, Any]:
    """Build and append a ledger entry. Returns the entry dict."""
    workspace = Path(workspace)
    run_dir = workspace / "experiments" / "runs" / run_id
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics.json not found at {metrics_path}")

    metrics = json.loads(metrics_path.read_text())
    notes_path = run_dir / "notes.md"
    notes = _parse_notes(notes_path)

    ledger_path = workspace / "experiments" / "ledger.jsonl"
    best = _current_best(ledger_path)
    f1 = float(metrics.get("final_macro_f1", -1.0))
    kept = f1 > best + keep_threshold
    if best == 0.0 and f1 > 0:
        rationale = f"first run; sets initial floor at {f1:.4f}"
    elif kept:
        rationale = f"+{f1 - best:.4f} macro_f1 over prior best ({best:.4f})"
    else:
        delta = f1 - best
        rationale = f"{delta:+.4f} macro_f1 vs prior best ({best:.4f}); below keep threshold {keep_threshold}"

    config_yaml_path = run_dir / "config.yaml"
    config_text = config_yaml_path.read_text() if config_yaml_path.is_file() else ""

    entry = {
        "run_id": run_id,
        "phase": phase,
        "parent_run_id": parent_run_id,
        "hypothesis": notes["hypothesis"],
        "citation": notes["citation"],
        "config_path": str(config_yaml_path.relative_to(workspace)) if config_yaml_path.is_file() else "",
        "config_text_first_lines": "\n".join(config_text.splitlines()[:5]),
        "wall_clock_min": round(metrics.get("wall_clock_seconds", 0) / 60.0, 2),
        "metrics": {
            "final_macro_f1": f1,
            "final_macro_edit_f1": float(metrics.get("final_macro_edit_f1", -1.0)),
            "wall_clock_seconds": metrics.get("wall_clock_seconds"),
            "exit_code": metrics.get("exit_code"),
        },
        "best_at_time_of_run": best,
        "kept": kept,
        "rationale": rationale,
        "notes_path": str(notes_path.relative_to(workspace)),
        "git_sha": metrics.get("git_sha", "unknown"),
        "timestamp": _dt.datetime.now(_dt.UTC).isoformat(),
    }

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_id", type=str)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--phase", default="explore", choices=["explore", "confirm", "baseline"])
    parser.add_argument("--parent-run-id", default="", help="The run this experiment iterates on")
    parser.add_argument("--keep-threshold", type=float, default=0.001)
    args = parser.parse_args()

    if not args.workspace.exists():
        print(f"[append_ledger] FATAL: workspace not found: {args.workspace}", file=sys.stderr)
        return 2

    entry = append_ledger(
        run_id=args.run_id,
        workspace=args.workspace,
        phase=args.phase,
        parent_run_id=args.parent_run_id,
        keep_threshold=args.keep_threshold,
    )
    verdict = "KEPT" if entry["kept"] else "REVERTED"
    print(f"[append_ledger] {verdict} run_id={args.run_id} f1={entry['metrics']['final_macro_f1']:.4f} ({entry['rationale']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
