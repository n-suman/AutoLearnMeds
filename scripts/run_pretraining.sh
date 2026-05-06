#!/usr/bin/env bash
# MAE pretraining launcher (Phase 7).
#
# Usage:
#   bash scripts/run_pretraining.sh [run_id] [--config <yaml>] [--seed N] [--no-wandb] [--phase pretrain]
#
# Differences from run_experiment.sh:
#   - Output dir is experiments/pretraining/<run_id>/ (NOT runs/).
#   - No --track flag, no ledger entry, no auto-finalize.
#   - Final-line grep target: ^final_pretrain_loss=[0-9.]+
#   - Snapshots pretrain_mae.py + mae_pretrain.yaml.
#   - Mirrors run dir + checkpoints/mae/ to GCS at the end on clean exit.

set -euo pipefail

# PATH augmentation BEFORE we use gsutil/python.
export PATH="$PATH:/tools/google-cloud-sdk/bin:/usr/local/google-cloud-sdk/bin:/opt/google-cloud-sdk/bin:/snap/bin"

RUN_ID="${1:-mae-pretrain-$(date -u +%Y%m%dT%H%M%S)}"
shift || true

CONFIG="experiments/configs/mae_pretrain.yaml"
SEED_ARG=""
EXTRA_ARGS=()
while (($#)); do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --seed) SEED_ARG="--seed $2"; shift 2;;
    *) EXTRA_ARGS+=("$1"); shift;;
  esac
done

RUN_DIR="experiments/pretraining/${RUN_ID}"
mkdir -p "$RUN_DIR"

echo "[run_pretraining] run_id=$RUN_ID config=$CONFIG ${SEED_ARG}"

cp pretrain_mae.py "$RUN_DIR/pretrain_mae.py"
cp "$CONFIG" "$RUN_DIR/config.yaml"

GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
START_TS=$(date -u +%s)

set +e
PYTHONUNBUFFERED=1 uv run python pretrain_mae.py --config "$CONFIG" $SEED_ARG "${EXTRA_ARGS[@]}" \
    > "$RUN_DIR/stdout.log" 2>&1
EXIT_CODE=$?
set -e

END_TS=$(date -u +%s)
WALL_CLOCK=$((END_TS - START_TS))

FINAL_LOSS="$(grep -oE '^final_pretrain_loss=[0-9.]+' "$RUN_DIR/stdout.log" | tail -1 | cut -d= -f2 || echo '-1.0')"
[[ -z "$FINAL_LOSS" ]] && FINAL_LOSS="-1.0"

cat > "$RUN_DIR/metrics.json" <<EOF
{
  "run_id": "$RUN_ID",
  "kind": "mae_pretrain",
  "git_sha": "$GIT_SHA",
  "config": "$CONFIG",
  "final_pretrain_loss": $FINAL_LOSS,
  "wall_clock_seconds": $WALL_CLOCK,
  "exit_code": $EXIT_CODE
}
EOF

# Auto-push to GCS (only on clean exit).
if [[ "$EXIT_CODE" -eq 0 ]]; then
  if command -v gsutil >/dev/null 2>&1; then
    BUCKET="${AUTOLEARNMEDS_GCS_BUCKET:-gs://auto_learn_meds}"
    echo "[run_pretraining] explicit GCS push: $BUCKET/experiments/pretraining/$RUN_ID/ ..."
    gsutil -m rsync -r "$RUN_DIR/" "$BUCKET/experiments/pretraining/$RUN_ID/" 2>&1 | tail -5 || \
      echo "[run_pretraining] WARNING: gsutil rsync of run dir failed; sync daemon will retry"
    # checkpoints/mae/ may have grown; mirror it too. Read checkpoint_dir from the cfg.
    CKPT_LOCAL="$(uv run --extra dev python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['checkpoint_dir'])")"
    if [[ -n "$CKPT_LOCAL" ]] && [[ -d "$CKPT_LOCAL" ]]; then
      echo "[run_pretraining] mirroring $CKPT_LOCAL to GCS..."
      gsutil -m rsync -r "$CKPT_LOCAL/" "$BUCKET/$CKPT_LOCAL/" 2>&1 | tail -5 || \
        echo "[run_pretraining] WARNING: gsutil rsync of checkpoints failed; sync daemon will retry"
    fi
  else
    echo "[run_pretraining] gsutil not on PATH; skipping explicit GCS push (daemon should pick up)"
  fi
fi

echo "[run_pretraining] DONE final_pretrain_loss=$FINAL_LOSS wall_clock=${WALL_CLOCK}s exit=$EXIT_CODE"
echo "[run_pretraining] artifacts at $RUN_DIR/"
exit $EXIT_CODE
