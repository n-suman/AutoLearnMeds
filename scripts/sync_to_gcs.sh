#!/usr/bin/env bash
# Periodic differential sync of experiments/ + checkpoints/ to GCS.
# Designed to run forever in the background on Colab.
#
# Env vars:
#   AUTOLEARNMEDS_GCS_BUCKET  (required) — gs://bucket-name
#   AUTOLEARNMEDS_WORKSPACE   (default: /workspace)
#   AUTOLEARNMEDS_SYNC_INTERVAL (default: 300 — seconds between syncs)
#
# Run: nohup bash scripts/sync_to_gcs.sh > /tmp/sync.log 2>&1 &

set -euo pipefail

BUCKET="${AUTOLEARNMEDS_GCS_BUCKET:-}"
WORKSPACE="${AUTOLEARNMEDS_WORKSPACE:-/workspace}"
INTERVAL="${AUTOLEARNMEDS_SYNC_INTERVAL:-300}"

if [[ -z "$BUCKET" ]]; then
  echo "[sync_to_gcs] FATAL: AUTOLEARNMEDS_GCS_BUCKET is unset" >&2
  exit 1
fi

if ! command -v gsutil >/dev/null 2>&1; then
  echo "[sync_to_gcs] FATAL: gsutil not on PATH (install google-cloud-sdk)" >&2
  exit 1
fi

echo "[sync_to_gcs] starting; bucket=$BUCKET workspace=$WORKSPACE interval=${INTERVAL}s"

while true; do
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "[sync_to_gcs] $ts — sync start"

  # The three things we mirror; each is non-fatal on transient failure.
  for src_rel in experiments checkpoints; do
    src="$WORKSPACE/$src_rel"
    dst="$BUCKET/$src_rel"
    if [[ -d "$src" ]]; then
      gsutil -m rsync -r -d "$src" "$dst" 2>&1 | tail -5 || \
        echo "[sync_to_gcs] WARN: rsync $src failed (non-fatal)"
    fi
  done

  # Active-experiment marker — single small file, sync directly.
  if [[ -f "$WORKSPACE/experiments/_active.json" ]]; then
    gsutil cp "$WORKSPACE/experiments/_active.json" "$BUCKET/experiments/_active.json" 2>&1 | tail -1 || true
  fi

  echo "[sync_to_gcs] $(date -u +%Y-%m-%dT%H:%M:%SZ) — sync done"
  sleep "$INTERVAL"
done
