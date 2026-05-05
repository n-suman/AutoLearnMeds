#!/usr/bin/env bash
# Canonical autoresearch experiment runner.
#
# Usage:
#   bash scripts/run_experiment.sh [run_id] [--config experiments/configs/baseline.yaml] [--seed N]
#
# What it does:
#   1. Snapshots the current train.py into experiments/runs/<run_id>/train.py
#   2. Records the config used into experiments/runs/<run_id>/config.yaml
#   3. Runs `python train.py --config <cfg> [--seed N]`, capturing stdout/stderr
#      to experiments/runs/<run_id>/stdout.log
#   4. Parses the final `final_macro_f1=X.XXXX` line and writes
#      experiments/runs/<run_id>/metrics.json with {final_macro_f1, run_id,
#      git_sha, exit_code, wall_clock_seconds}.

set -euo pipefail

RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%S)-$(printf '%04x' $RANDOM)}"
shift || true

CONFIG="experiments/configs/baseline.yaml"
SEED_ARG=""
EXTRA_ARGS=()
while (($#)); do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --seed) SEED_ARG="--seed $2"; shift 2;;
    *) EXTRA_ARGS+=("$1"); shift;;
  esac
done

RUN_DIR="experiments/runs/${RUN_ID}"
mkdir -p "$RUN_DIR"

echo "[run_experiment] run_id=$RUN_ID config=$CONFIG ${SEED_ARG}"

cp train.py "$RUN_DIR/train.py"
cp "$CONFIG" "$RUN_DIR/config.yaml"

GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
START_TS=$(date -u +%s)

set +e
uv run python train.py --config "$CONFIG" $SEED_ARG "${EXTRA_ARGS[@]}" \
    > "$RUN_DIR/stdout.log" 2>&1
EXIT_CODE=$?
set -e

END_TS=$(date -u +%s)
WALL_CLOCK=$((END_TS - START_TS))

FINAL_F1="$(grep -oE 'final_macro_f1=[0-9.]+' "$RUN_DIR/stdout.log" | tail -1 | cut -d= -f2 || echo '-1.0')"
[[ -z "$FINAL_F1" ]] && FINAL_F1="-1.0"

cat > "$RUN_DIR/metrics.json" <<EOF
{
  "run_id": "$RUN_ID",
  "git_sha": "$GIT_SHA",
  "config": "$CONFIG",
  "final_macro_f1": $FINAL_F1,
  "wall_clock_seconds": $WALL_CLOCK,
  "exit_code": $EXIT_CODE
}
EOF

echo "[run_experiment] DONE final_macro_f1=$FINAL_F1 wall_clock=${WALL_CLOCK}s exit=$EXIT_CODE"
echo "[run_experiment] artifacts at $RUN_DIR/"
exit $EXIT_CODE
