#!/usr/bin/env bash
# pipeline_phase_7.sh — orchestrate Phase 7 + Phase 7b end-to-end on Colab.
#
# Runs sequentially after a successful DAPT MAE pretraining:
#   1. wait for DAPT MAE to land metrics.json
#   2. TAPT (~30 min)
#   3. text-aware MAE (~6 h)
#   4. downstream Track A: baseline_mae_init (~30 min)
#   5. downstream Track A: baseline_mae_text_aware_init (~30 min)
#   6. downstream Track A: baseline_mae_tapt_init (~30 min)
#   7. per-field eval on each of the 4 Track A variants (vanilla + 3 MAE-init)
#   8. final commit + push
#
# Each milestone writes a sentinel file in $STATE_DIR. If the script is
# re-launched (e.g. after a Colab VM reclaim), it skips milestones whose
# sentinel exists and resumes at the next pending one. Resumability AT THE
# SUB-STEP level (mid-MAE-pretraining resume) is provided separately by the
# F8a checkpoint+resume logic in pretrain_mae.py — this orchestrator just
# decides which top-level stage to run next.
#
# Designed to run nohup'd on Colab. Mac-disconnect-safe; final commit pushes
# to GitHub from Colab (git config set up by F6 in colab_bootstrap.sh).
#
# Usage on Colab:
#   cd /workspace
#   nohup bash scripts/pipeline_phase_7.sh > /tmp/pipeline.log 2>&1 &
#
# To start fresh (wipe milestone sentinels):
#   rm -rf experiments/pipeline_state/phase-7
#   bash scripts/pipeline_phase_7.sh

set -uo pipefail   # NB: no -e — we explicitly check exit codes per stage

WORKSPACE="${AUTOLEARNMEDS_WORKSPACE:-/workspace}"
STATE_DIR="$WORKSPACE/experiments/pipeline_state/phase-7"
mkdir -p "$STATE_DIR"

cd "$WORKSPACE"

# Augment PATH so gsutil works in non-interactive shell + LD path for CUDA.
export PATH="$PATH:/tools/google-cloud-sdk/bin:/usr/local/google-cloud-sdk/bin:/opt/google-cloud-sdk/bin:/snap/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/lib64-nvidia"

milestone_done() { test -f "$STATE_DIR/$1.done"; }
mark_done() {
  date -u +%Y-%m-%dT%H:%M:%SZ > "$STATE_DIR/$1.done"
  echo "[pipeline] $(date -u +%Y-%m-%dT%H:%M:%SZ) milestone DONE: $1"
}
log() { echo "[pipeline] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

# Best-effort GCS sync of a path (silent on failure — the sync_to_gcs
# daemon will retry).
push_gcs() {
  local local_path="$1"
  local bucket="${AUTOLEARNMEDS_GCS_BUCKET:-gs://auto_learn_meds}"
  if command -v gsutil >/dev/null 2>&1 && [[ -e "$local_path" ]]; then
    gsutil -m rsync -r "$local_path" "$bucket/$local_path" 2>&1 | tail -3 || true
  fi
}

run_pretraining() {
  local run_id="$1"
  local config="$2"
  log "PRETRAIN start: run_id=$run_id config=$config"
  bash scripts/run_pretraining.sh "$run_id" --config "$config" --seed 44 --no-wandb
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    log "FATAL PRETRAIN: $run_id exited $rc; bailing"
    exit $rc
  fi
}

run_downstream() {
  local run_id="$1"
  local config="$2"
  log "DOWNSTREAM start: run_id=$run_id config=$config"
  FINALIZE_PHASE=baseline bash scripts/run_experiment.sh "$run_id" \
    --track A --seed 44 --no-wandb --config "$config"
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    log "FATAL DOWNSTREAM: $run_id exited $rc; bailing"
    exit $rc
  fi
}

# === Stage 1: wait for DAPT MAE ===
# (This stage exists so the pipeline can be launched even before DAPT
# finishes — it'll poll until metrics.json shows up.)
if ! milestone_done dapt_mae; then
  log "Stage 1/8: waiting for DAPT MAE (mae-pretrain-seed44/metrics.json)..."
  while [[ ! -f experiments/pretraining/mae-pretrain-seed44/metrics.json ]]; do
    sleep 60
  done
  mark_done dapt_mae
fi

# === Stage 2: TAPT ===
if ! milestone_done tapt_mae; then
  run_pretraining "mae-pretrain-tapt-seed44" "experiments/configs/mae_pretrain_tapt.yaml"
  mark_done tapt_mae
fi

# === Stage 3: text-aware MAE ===
if ! milestone_done text_aware_mae; then
  run_pretraining "mae-pretrain-text-aware-seed44" "experiments/configs/mae_pretrain_text_aware.yaml"
  mark_done text_aware_mae
fi

# === Stages 4-6: 3 downstream Track A runs ===
for variant in baseline_mae_init baseline_mae_text_aware_init baseline_mae_tapt_init; do
  if ! milestone_done "downstream_$variant"; then
    run_downstream "${variant}-seed44" "experiments/configs/${variant}.yaml"
    mark_done "downstream_$variant"
  fi
done

# === Stages 7: per-field eval on all 4 Track A variants ===
# Map run_id -> checkpoint dir basename (matches each yaml's checkpoint_dir).
declare -A perfield_runs=(
  ["baseline-seed44-rerun"]="baseline"
  ["baseline_mae_init-seed44"]="baseline_mae_init"
  ["baseline_mae_text_aware_init-seed44"]="baseline_mae_text_aware_init"
  ["baseline_mae_tapt_init-seed44"]="baseline_mae_tapt_init"
)
for run_id in "${!perfield_runs[@]}"; do
  ckpt_dir="${perfield_runs[$run_id]}"
  if ! milestone_done "perfield_$run_id"; then
    log "PERFIELD start: $run_id ckpt=$ckpt_dir"
    uv run python scripts/per_field_eval.py --track A \
      --ckpt "checkpoints/runs/$ckpt_dir/best.pt" \
      --out "experiments/per_field_${run_id}.json" \
      || { log "PERFIELD failed for $run_id; continuing"; continue; }
    mark_done "perfield_$run_id"
  fi
done

# === Stage 8: final commit + push ===
if ! milestone_done final_commit; then
  log "Stage 8/8: final commit + push"
  git add -f experiments/pretraining \
              experiments/runs \
              experiments/per_field_*.json \
              experiments/ledger.jsonl \
              experiments/leaderboard.md \
              experiments/pipeline_state/ 2>&1 | tail -5 || true
  git commit -m "exp(phase-7+7b): full pipeline run results (orchestrated by pipeline_phase_7.sh)" || true
  git push origin phase-0-plumbing 2>&1 | tail -3 || log "git push failed (non-fatal — Mac will pull)"
  push_gcs experiments/pipeline_state/
  mark_done final_commit
fi

log "ALL DONE — Phase 7 + 7b pipeline complete."
log "Summary milestones in $STATE_DIR:"
ls -la "$STATE_DIR" 2>&1 | tail -20
