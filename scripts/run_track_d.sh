#!/usr/bin/env bash
# Phase 8 Track D evaluation launcher.
#
# Usage:
#   bash scripts/run_track_d.sh [run_id] [--config <yaml>] [--seed N]
#
# What it does:
#   1. Snapshots track_d_pipeline.py + the config into experiments/runs/<run_id>/
#   2. Runs `python track_d_pipeline.py --config <cfg> [--seed N]`, capturing
#      stdout/stderr to experiments/runs/<run_id>/stdout.log
#   3. Parses ^final_macro_f1= and ^final_macro_edit_f1= from stdout, writes
#      experiments/runs/<run_id>/metrics.json with track="D" and kind="track_d_eval"
#   4. On clean exit: calls finalize_experiment.sh and rsyncs run dir +
#      per-field json + ledger + leaderboard to GCS.
#
# Eval-only: YOLO weights come from a prior train_yolo.py run via cfg.yolo_weights.

set -euo pipefail

# PATH augmentation BEFORE we use gsutil/python (Colab quirk).
export PATH="$PATH:/tools/google-cloud-sdk/bin:/usr/local/google-cloud-sdk/bin:/opt/google-cloud-sdk/bin:/snap/bin"

# LD_LIBRARY_PATH for CUDA on Colab.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/lib64-nvidia"

RUN_ID="${1:-track_d-seed44}"
shift || true

CONFIG="experiments/configs/track_d.yaml"
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

echo "[run_track_d] run_id=$RUN_ID config=$CONFIG ${SEED_ARG}"

cp track_d_pipeline.py "$RUN_DIR/track_d_pipeline.py"
cp "$CONFIG" "$RUN_DIR/config.yaml"

GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
START_TS=$(date -u +%s)

set +e
PYTHONUNBUFFERED=1 uv run python track_d_pipeline.py --config "$CONFIG" $SEED_ARG "${EXTRA_ARGS[@]}" \
    > "$RUN_DIR/stdout.log" 2>&1
EXIT_CODE=$?
set -e

END_TS=$(date -u +%s)
WALL_CLOCK=$((END_TS - START_TS))

FINAL_F1="$(grep -oE '^final_macro_f1=[0-9.]+' "$RUN_DIR/stdout.log" | tail -1 | cut -d= -f2 || echo '-1.0')"
[[ -z "$FINAL_F1" ]] && FINAL_F1="-1.0"
FINAL_EDIT_F1="$(grep -oE '^final_macro_edit_f1=[0-9.]+' "$RUN_DIR/stdout.log" | tail -1 | cut -d= -f2 || echo '-1.0')"
[[ -z "$FINAL_EDIT_F1" ]] && FINAL_EDIT_F1="-1.0"

cat > "$RUN_DIR/metrics.json" <<EOF
{
  "run_id": "$RUN_ID",
  "track": "D",
  "kind": "track_d_eval",
  "git_sha": "$GIT_SHA",
  "config": "$CONFIG",
  "final_macro_f1": $FINAL_F1,
  "final_macro_edit_f1": $FINAL_EDIT_F1,
  "wall_clock_seconds": $WALL_CLOCK,
  "exit_code": $EXIT_CODE
}
EOF

# --- Auto-finalize: only if eval exited cleanly ---
if [[ "$EXIT_CODE" -eq 0 ]]; then
  echo "[run_track_d] auto-finalizing run $RUN_ID..."
  FINALIZE_PHASE="${FINALIZE_PHASE:-baseline}"
  if bash scripts/finalize_experiment.sh "$RUN_ID" --phase "$FINALIZE_PHASE"; then
    echo "[run_track_d] finalize_experiment.sh DONE (phase=$FINALIZE_PHASE)"
  else
    FCODE=$?
    echo "[run_track_d] WARNING: finalize_experiment.sh failed (exit=$FCODE); manual finalize needed"
  fi

  if command -v gsutil >/dev/null 2>&1; then
    BUCKET="${AUTOLEARNMEDS_GCS_BUCKET:-gs://auto_learn_meds}"
    echo "[run_track_d] explicit GCS push: $BUCKET/experiments/runs/$RUN_ID/ ..."
    gsutil -m rsync -r "$RUN_DIR/" "$BUCKET/experiments/runs/$RUN_ID/" 2>&1 | tail -5 || \
      echo "[run_track_d] WARNING: gsutil rsync of run dir failed; sync daemon will retry"

    PER_FIELD_JSON="experiments/per_field_track_d-seed44.json"
    if [[ -f "$PER_FIELD_JSON" ]]; then
      gsutil -m cp "$PER_FIELD_JSON" "$BUCKET/experiments/" 2>&1 | tail -3 || \
        echo "[run_track_d] WARNING: per-field json cp failed; sync daemon will retry"
    fi

    gsutil -m cp experiments/ledger.jsonl experiments/leaderboard.md \
      "$BUCKET/experiments/" 2>&1 | tail -3 || \
      echo "[run_track_d] WARNING: ledger/leaderboard cp failed; sync daemon will retry"
  else
    echo "[run_track_d] gsutil not on PATH; skipping explicit GCS push (daemon should pick up)"
  fi
fi

echo "[run_track_d] DONE final_macro_f1=$FINAL_F1 final_macro_edit_f1=$FINAL_EDIT_F1 wall_clock=${WALL_CLOCK}s exit=$EXIT_CODE"
echo "[run_track_d] artifacts at $RUN_DIR/"
exit $EXIT_CODE
