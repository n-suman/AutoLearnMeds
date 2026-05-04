#!/usr/bin/env bash
# AutoLearnMeds — one-cell Colab Pro+ bootstrap.
#
# Run from a Colab notebook cell:
#   !curl -sSL https://raw.githubusercontent.com/<user>/<repo>/<branch>/scripts/colab_bootstrap.sh | bash
#
# At end, prints an SSH command to copy into your Mac's ~/.ssh/config.
#
# Required env vars (set in the Colab cell BEFORE running this script):
#   AUTOLEARNMEDS_GCS_BUCKET  — gs://your-bucket-name
#   AUTOLEARNMEDS_REPO_URL    — https://github.com/<user>/<repo>.git
#   AUTOLEARNMEDS_SSH_PASSWORD — temp password for SSH (use a strong random)
#
# Optional env vars:
#   AUTOLEARNMEDS_BRANCH                   — repo branch to clone (default: main)
#   AUTOLEARNMEDS_DRIVE_GOLDEN_SET_ID      — Drive folder ID for label JSON
#   AUTOLEARNMEDS_DRIVE_RAW_IMAGES_ID      — Drive folder ID for images
#
# If the two Drive IDs are set, the bootstrap performs a one-time
# (idempotent) sync of those folders into gs://${BUCKET}/raw/* so all
# downstream tasks read from GCS, not from Drive directly.

set -euo pipefail

BUCKET="${AUTOLEARNMEDS_GCS_BUCKET:?Set AUTOLEARNMEDS_GCS_BUCKET=gs://your-bucket}"
REPO_URL="${AUTOLEARNMEDS_REPO_URL:?Set AUTOLEARNMEDS_REPO_URL=https://github.com/...}"
SSH_PASSWORD="${AUTOLEARNMEDS_SSH_PASSWORD:?Set AUTOLEARNMEDS_SSH_PASSWORD=...}"
BRANCH="${AUTOLEARNMEDS_BRANCH:-main}"
DRIVE_GOLDEN_SET_ID="${AUTOLEARNMEDS_DRIVE_GOLDEN_SET_ID:-}"
DRIVE_RAW_IMAGES_ID="${AUTOLEARNMEDS_DRIVE_RAW_IMAGES_ID:-}"
DRIVE_PROJECT_DIR="/content/drive/MyDrive/AutoLearnMeds"
WORKSPACE="/workspace"

echo "============================================================"
echo "AutoLearnMeds bootstrap — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  branch=$BRANCH"
echo "============================================================"

# 1. Verify Google Drive is mounted (notebook cell does the actual mount because
#    google.colab.drive.mount needs the IPython kernel; subprocesses can't.)
echo "[1/8] Verifying Google Drive mount..."
if ! mountpoint -q /content/drive 2>/dev/null && [[ ! -d /content/drive/MyDrive ]]; then
  echo "FATAL: /content/drive is not mounted." >&2
  echo "  This script expects the calling notebook cell to have run:" >&2
  echo "    from google.colab import drive; drive.mount('/content/drive')" >&2
  echo "  (drive.mount needs the IPython kernel, which subprocesses don't have.)" >&2
  exit 1
fi
echo "  OK: /content/drive is mounted"

# 2. Mount GCS bucket (gcsfuse)
echo "[2/8] Mounting GCS bucket $BUCKET..."
if ! command -v gcsfuse >/dev/null 2>&1; then
  echo "  installing gcsfuse..."
  echo "deb https://packages.cloud.google.com/apt gcsfuse-bookworm main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
  sudo apt-get update -q && sudo apt-get install -y -q gcsfuse
fi
mkdir -p /mnt/gcs
GCS_BUCKET_NAME="${BUCKET#gs://}"
if ! mountpoint -q /mnt/gcs; then
  gcsfuse "$GCS_BUCKET_NAME" /mnt/gcs || \
    echo "  WARN: gcsfuse mount failed (auth?). GCS sync will still work via gsutil."
fi

# 3. Clone repo into Drive (so it survives runtime restarts) and symlink to /workspace
echo "[3/8] Setting up workspace (branch=$BRANCH)..."
mkdir -p /content/drive/MyDrive
if [[ ! -d "$DRIVE_PROJECT_DIR/.git" ]]; then
  git clone -b "$BRANCH" "$REPO_URL" "$DRIVE_PROJECT_DIR"
else
  cd "$DRIVE_PROJECT_DIR" && \
    git fetch origin && \
    git checkout "$BRANCH" && \
    git pull --ff-only origin "$BRANCH" || \
    echo "  WARN: git pull failed (continuing with existing copy)"
fi
ln -sfn "$DRIVE_PROJECT_DIR" "$WORKSPACE"
cd "$WORKSPACE"

# 4. Install uv + sync deps (full ml + colab extras on Colab)
echo "[4/8] Installing uv and syncing deps..."
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --extra ml --extra colab

# 5. Drive -> GCS one-time data sync (idempotent; skips if GCS already has data)
echo "[5/8] Drive -> GCS data sync (one-time, idempotent)..."
if [[ -n "$DRIVE_GOLDEN_SET_ID" ]] || [[ -n "$DRIVE_RAW_IMAGES_ID" ]]; then
  AUTOLEARNMEDS_GCS_BUCKET="$BUCKET" \
  AUTOLEARNMEDS_DRIVE_GOLDEN_SET_ID="$DRIVE_GOLDEN_SET_ID" \
  AUTOLEARNMEDS_DRIVE_RAW_IMAGES_ID="$DRIVE_RAW_IMAGES_ID" \
  uv run python scripts/sync_drive_to_gcs.py || \
    echo "  WARN: Drive->GCS sync had issues. Re-run later: uv run python scripts/sync_drive_to_gcs.py [--force]"
else
  echo "  AUTOLEARNMEDS_DRIVE_GOLDEN_SET_ID and AUTOLEARNMEDS_DRIVE_RAW_IMAGES_ID unset; skipping."
fi

# 6. Launch SSH server + Cloudflare Tunnel
echo "[6/8] Launching SSH + cloudflared..."
uv run python -c "
from colab_ssh import launch_ssh_cloudflared
launch_ssh_cloudflared(password='$SSH_PASSWORD')
" | tee /tmp/colab_ssh_output.txt

# Extract the cloudflared host from the colab-ssh output and pin it.
CLOUDFLARED_HOST="$(grep -oE '[a-zA-Z0-9-]+\.trycloudflare\.com' /tmp/colab_ssh_output.txt | head -1 || true)"
if [[ -n "$CLOUDFLARED_HOST" ]]; then
  echo "$CLOUDFLARED_HOST" > "$WORKSPACE/.colab_ssh_host"
  gsutil cp "$WORKSPACE/.colab_ssh_host" "$BUCKET/.colab_ssh_host" 2>/dev/null || true
fi

# 7. Background daemons
echo "[7/8] Starting keepalive + GCS sync daemons..."
nohup uv run python scripts/keepalive.py > /tmp/keepalive.log 2>&1 &
AUTOLEARNMEDS_GCS_BUCKET="$BUCKET" \
AUTOLEARNMEDS_WORKSPACE="$WORKSPACE" \
nohup bash scripts/sync_to_gcs.sh > /tmp/sync_to_gcs.log 2>&1 &

# 8. Success banner
echo "============================================================"
echo "[8/8] READY"
echo "  Workspace:     $WORKSPACE"
echo "  GCS bucket:    $BUCKET"
echo "  Branch:        $BRANCH"
echo "  Cloudflared:   ${CLOUDFLARED_HOST:-<see /tmp/colab_ssh_output.txt>}"
echo ""
echo "On your Mac, run:"
echo "  ./scripts/update_ssh_config.sh '${CLOUDFLARED_HOST:-<host>}'"
echo "Then in VSCode:"
echo "  Cmd-Shift-P -> Remote-SSH: Connect to Host -> autolearnmeds-colab"
echo "============================================================"
