#!/usr/bin/env bash
# Disaster recovery: rebuild experiments/ + checkpoints/ from GCS after a Colab nuke.
#
# Designed to run after a fresh `colab_bootstrap.sh` (which restores code from
# GitHub, deps from uv, and processed JSONLs from GCS golden_set). This script
# fills the remaining gap: experiment ledger entries, per-run artifacts, and
# checkpoints — all of which the autoresearch loop wrote during prior sessions
# and the sync_to_gcs daemon mirrored to gs://${BUCKET}/{experiments,checkpoints}/.
#
# Usage:
#   bash scripts/disaster_recover.sh
#
# Env vars:
#   AUTOLEARNMEDS_GCS_BUCKET  (required) — gs://bucket
#   AUTOLEARNMEDS_WORKSPACE   (default: /workspace)

set -euo pipefail

BUCKET="${AUTOLEARNMEDS_GCS_BUCKET:?Set AUTOLEARNMEDS_GCS_BUCKET=gs://your-bucket}"
WORKSPACE="${AUTOLEARNMEDS_WORKSPACE:-/workspace}"

# gsutil discovery: Colab has it at /tools/google-cloud-sdk/bin/ but not on PATH.
GSUTIL="$(command -v gsutil 2>/dev/null || true)"
if [[ -z "$GSUTIL" ]]; then
  for candidate in /tools/google-cloud-sdk/bin/gsutil /opt/google-cloud-sdk/bin/gsutil /usr/lib/google-cloud-sdk/bin/gsutil; do
    if [[ -x "$candidate" ]]; then
      GSUTIL="$candidate"
      break
    fi
  done
fi
if [[ -z "$GSUTIL" ]]; then
  echo "[disaster_recover] FATAL: gsutil not found on PATH or in /tools/google-cloud-sdk/bin" >&2
  exit 1
fi

echo "[disaster_recover] starting; bucket=$BUCKET workspace=$WORKSPACE"
echo "[disaster_recover] $(date -u +%Y-%m-%dT%H:%M:%SZ)"

cd "$WORKSPACE"

# experiments/ — restore the autoresearch ledger (run_id dirs + ledger.jsonl + leaderboard.md).
# rsync flags: -r recursive, -u "skip if source is not newer than destination", NO -d
#   - NO -d: never delete files at destination that aren't at source.
#       Reason: a fresh-bootstrap workspace is "empty"; without this guard, -d would
#       delete the entire GCS archive on first sync. Incident 2026-05-06.
#   - -u: only overwrite local with GCS if GCS is newer (by mtime).
#       Reason: a fresh git checkout produces local files with NEW mtimes that may
#       contain CORRECT content not yet in GCS (e.g. ledger.jsonl with appends
#       committed locally + pushed to git but not yet to GCS). Without -u, gsutil
#       overwrites those with stale GCS versions. Incident 2026-05-06 part 2.
echo "[disaster_recover] restoring experiments/..."
mkdir -p experiments
"$GSUTIL" -m rsync -r -u "$BUCKET/experiments" experiments 2>&1 | tail -8 || \
  echo "  WARN: experiments rsync had issues"

# checkpoints/ — same pattern. Big files; -m parallel transfers.
echo "[disaster_recover] restoring checkpoints/..."
mkdir -p checkpoints
"$GSUTIL" -m rsync -r -u "$BUCKET/checkpoints" checkpoints 2>&1 | tail -8 || \
  echo "  WARN: checkpoints rsync had issues"

# Active-experiment marker — tiny, sync directly.
if "$GSUTIL" ls "$BUCKET/experiments/_active.json" >/dev/null 2>&1; then
  echo "[disaster_recover] restoring experiments/_active.json..."
  "$GSUTIL" cp "$BUCKET/experiments/_active.json" experiments/_active.json 2>&1 | tail -1 || true
fi

# Report what we restored.
echo "[disaster_recover] DONE"
echo "  experiment runs restored: $(ls experiments/runs 2>/dev/null | wc -l)"
echo "  checkpoint dirs restored: $(ls checkpoints 2>/dev/null | wc -l)"
