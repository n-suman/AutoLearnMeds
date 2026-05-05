#!/usr/bin/env bash
# Post-experiment housekeeping. Runs after scripts/run_experiment.sh.
#
# Usage:
#   bash scripts/finalize_experiment.sh <run_id> [--phase explore|confirm|baseline] [--parent <run_id>]
#
# What it does:
#   1. Append a ledger entry via scripts/append_ledger.py.
#   2. Regenerate experiments/leaderboard.md via scripts/leaderboard.py.
#   3. If the ledger entry had kept=true, run scripts/promote.sh.
#   4. git add + commit (the agent / user pushes separately).

set -euo pipefail

RUN_ID="${1:?Usage: finalize_experiment.sh <run_id> [--phase ...] [--parent <id>]}"
shift || true

PHASE="explore"
PARENT=""
while (($#)); do
  case "$1" in
    --phase) PHASE="$2"; shift 2;;
    --parent) PARENT="$2"; shift 2;;
    *) echo "[finalize] unknown arg: $1" >&2; exit 2;;
  esac
done

WORKSPACE="${AUTOLEARNMEDS_WORKSPACE:-/workspace}"
cd "$WORKSPACE"

echo "[finalize] run_id=$RUN_ID phase=$PHASE parent=${PARENT:-none}"

# 1. Append ledger.
PARENT_ARG=""
if [[ -n "$PARENT" ]]; then PARENT_ARG="--parent-run-id $PARENT"; fi
LEDGER_OUT=$(uv run python scripts/append_ledger.py "$RUN_ID" \
    --workspace "$WORKSPACE" --phase "$PHASE" $PARENT_ARG 2>&1)
echo "$LEDGER_OUT"

# 2. Regenerate leaderboard.
uv run python scripts/leaderboard.py --workspace "$WORKSPACE" 2>&1 | tail -1

# 3. Promote if kept (parse the previous output).
if echo "$LEDGER_OUT" | grep -q "^\[append_ledger\] KEPT "; then
  echo "[finalize] kept; promoting..."
  bash scripts/promote.sh "$RUN_ID"
else
  echo "[finalize] reverted; not promoting"
fi

# 4. Git commit (the agent / user pushes).
git add experiments/ledger.jsonl experiments/leaderboard.md \
        "experiments/runs/$RUN_ID" \
        checkpoints/best 2>/dev/null || true
if ! git diff --staged --quiet; then
  git commit -m "experiment $RUN_ID ($PHASE): $(echo "$LEDGER_OUT" | head -1 | sed 's/.*\[append_ledger\] //')"
  echo "[finalize] committed"
else
  echo "[finalize] nothing staged to commit"
fi

echo "[finalize] DONE"
