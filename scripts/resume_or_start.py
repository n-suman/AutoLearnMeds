#!/usr/bin/env python3
"""Atomic-experiment state machine launcher.

Reads experiments/_active.json and experiments/ledger.jsonl to decide
what the next session should do. See spec §7.3 for the state table.

Usage:
  python scripts/resume_or_start.py [--workspace /workspace] [--print-only]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

Action = Literal["START_NEW", "RESTART", "RESUME_FROM_EVALUATED", "CLEAR_AND_START_NEW"]


def _read_active(workspace: Path) -> dict | None:
    active_path = workspace / "experiments" / "_active.json"
    if not active_path.is_file():
        return None
    try:
        return json.loads(active_path.read_text())
    except json.JSONDecodeError:
        return None


def _metrics_exist(workspace: Path, run_id: str) -> bool:
    return (workspace / "experiments" / "runs" / run_id / "metrics.json").is_file()


def _ledger_has_run(workspace: Path, run_id: str) -> bool:
    ledger_path = workspace / "experiments" / "ledger.jsonl"
    if not ledger_path.is_file():
        return False
    for line in ledger_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("run_id") == run_id:
            return True
    return False


def next_action(workspace: Path) -> Action:
    """Decide the next session's action per spec §7.3."""
    active = _read_active(workspace)
    if active is None:
        return "START_NEW"

    run_id = active.get("run_id", "")
    status = active.get("status", "")

    if status == "PLANNING":
        return "RESTART"

    has_metrics = _metrics_exist(workspace, run_id)

    if status == "ACTIVE" and not has_metrics:
        return "RESTART"

    if has_metrics and not _ledger_has_run(workspace, run_id):
        return "RESUME_FROM_EVALUATED"

    if has_metrics and _ledger_has_run(workspace, run_id):
        return "CLEAR_AND_START_NEW"

    # Defensive default: ACTIVE with metrics but ambiguous state — restart safely.
    return "RESTART"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="/workspace", type=Path)
    parser.add_argument("--print-only", action="store_true",
                        help="Print the action and exit; do not perform any side effects")
    args = parser.parse_args()

    if not args.workspace.exists():
        print(f"[resume_or_start] FATAL: workspace not found: {args.workspace}", file=sys.stderr)
        return 2

    action = next_action(args.workspace)
    print(f"[resume_or_start] action={action}")

    if args.print_only:
        return 0

    # Side effects (Phase 1+ wiring): for now, just print.
    # Future tasks add: invoke run_experiment.sh, finalize ledger entry, etc.
    return 0


if __name__ == "__main__":
    sys.exit(main())
