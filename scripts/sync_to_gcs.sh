#!/usr/bin/env bash
# Periodic differential sync of experiments/ + checkpoints/ to GCS.
# Designed to run forever in the background on Colab.
#
# IMPORTANT: rsync uses NO -d (delete) flag. This is intentional —
# checkpoints + experiments are append-only; we never want a fresh-bootstrap
# session to wipe the historical archive. (See incident 2026-05-06: -d
# silently destroyed Track A's checkpoints/best/best.pt when a new session
# started with an empty workspace.)
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
  # `data` was added 2026-05-13 for Track E pseudo-label artifacts under
  # data/pseudo_labels/ (Qwen2-VL labeling output, ~3h GPU to regenerate).
  # Same no-delete-flag discipline as the rest.
  for src_rel in experiments checkpoints data; do
    src="$WORKSPACE/$src_rel"
    dst="$BUCKET/$src_rel"
    if [[ -d "$src" ]]; then
      gsutil -m rsync -r "$src" "$dst" 2>&1 | tail -5 || \
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
