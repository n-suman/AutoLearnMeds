#!/usr/bin/env bash
# Promote a kept experiment's train.py + best.pt to checkpoints/best/.
#
# Usage:
#   bash scripts/promote.sh <run_id>
#
# What it does:
#   1. Snapshots experiments/runs/<run_id>/train.py + config.yaml to
#      checkpoints/best/{train.py, config.yaml}.
#   2. If checkpoints/runs/<run_id>/best.pt exists (saved by train_loop on
#      best macro_f1), copies it to checkpoints/best/best.pt.
#   3. Records the run_id and timestamp in checkpoints/best/manifest.txt.

set -euo pipefail

RUN_ID="${1:?Usage: promote.sh <run_id>}"
WORKSPACE="${AUTOLEARNMEDS_WORKSPACE:-/workspace}"

cd "$WORKSPACE"

RUN_DIR="experiments/runs/${RUN_ID}"
BEST_DIR="checkpoints/best"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "[promote] FATAL: $RUN_DIR not found" >&2
  exit 1
fi

mkdir -p "$BEST_DIR"

cp "$RUN_DIR/train.py" "$BEST_DIR/train.py"
cp "$RUN_DIR/config.yaml" "$BEST_DIR/config.yaml"

# best.pt is saved by train_loop when val macro_f1 improves; might live under
# the per-run checkpoint dir specified in config.yaml. We try the standard
# checkpoints/runs/<run_id>/best.pt first.
SRC_BEST="checkpoints/runs/${RUN_ID}/best.pt"
if [[ -f "$SRC_BEST" ]]; then
  cp "$SRC_BEST" "$BEST_DIR/best.pt"
  echo "[promote] copied $SRC_BEST"
else
  echo "[promote] WARN: $SRC_BEST not found; promoting code only (no weights)"
fi

# Manifest — append-only history of every promotion.
MANIFEST="$BEST_DIR/manifest.txt"
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "$TS  promoted run_id=$RUN_ID" >> "$MANIFEST"

echo "[promote] DONE: $RUN_ID is now the current best"
