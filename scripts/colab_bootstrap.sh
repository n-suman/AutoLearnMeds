#!/usr/bin/env bash
# AutoLearnMeds — one-cell Colab Pro+ bootstrap.
#
# Run from a Colab notebook cell:
#   !curl -sSL https://raw.githubusercontent.com/<user>/<repo>/main/scripts/colab_bootstrap.sh | bash
#
# At end, prints an SSH command to copy into your Mac's ~/.ssh/config.
#
# Required env vars (set in the Colab cell BEFORE running this script):
#   AUTOLEARNMEDS_GCS_BUCKET  — gs://your-bucket-name
#   AUTOLEARNMEDS_REPO_URL    — https://github.com/<user>/<repo>.git
#   AUTOLEARNMEDS_SSH_PASSWORD — temp password for SSH (use a strong random)

set -euo pipefail

BUCKET="${AUTOLEARNMEDS_GCS_BUCKET:?Set AUTOLEARNMEDS_GCS_BUCKET=gs://your-bucket}"
REPO_URL="${AUTOLEARNMEDS_REPO_URL:?Set AUTOLEARNMEDS_REPO_URL=https://github.com/...}"
SSH_PASSWORD="${AUTOLEARNMEDS_SSH_PASSWORD:?Set AUTOLEARNMEDS_SSH_PASSWORD=...}"
DRIVE_PROJECT_DIR="/content/drive/MyDrive/AutoLearnMeds"
WORKSPACE="/workspace"

echo "============================================================"
echo "AutoLearnMeds bootstrap — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# 1. Mount Google Drive (interactive auth on first run; cached after)
echo "[1/7] Mounting Google Drive..."
python -c "from google.colab import drive; drive.mount('/content/drive', force_remount=False)" || {
  echo "FATAL: Drive mount failed" >&2; exit 1;
}

# 2. Mount GCS bucket (gcsfuse)
echo "[2/7] Mounting GCS bucket $BUCKET..."
if ! command -v gcsfuse >/dev/null 2>&1; then
  echo "  installing gcsfuse..."
  echo "deb https://packages.cloud.google.com/apt gcsfuse-bookworm main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
  sudo apt-get update -q && sudo apt-get install -y -q gcsfuse
fi
mkdir -p /mnt/gcs
# Bucket name without the gs:// prefix
GCS_BUCKET_NAME="${BUCKET#gs://}"
if ! mountpoint -q /mnt/gcs; then
  gcsfuse "$GCS_BUCKET_NAME" /mnt/gcs || \
    echo "  WARN: gcsfuse mount failed (auth?). GCS sync will still work via gsutil."
fi

# 3. Clone repo into Drive (so it survives runtime restarts) and symlink to /workspace
echo "[3/7] Setting up workspace..."
mkdir -p /content/drive/MyDrive
if [[ ! -d "$DRIVE_PROJECT_DIR/.git" ]]; then
  git clone "$REPO_URL" "$DRIVE_PROJECT_DIR"
else
  cd "$DRIVE_PROJECT_DIR" && git pull --ff-only || echo "  WARN: git pull failed (continuing with existing copy)"
fi
ln -sfn "$DRIVE_PROJECT_DIR" "$WORKSPACE"
cd "$WORKSPACE"

# 4. Install uv + sync deps (full ml + colab extras on Colab)
echo "[4/7] Installing uv and syncing deps..."
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --extra ml --extra colab

# 5. Launch SSH server + Cloudflare Tunnel
echo "[5/7] Launching SSH + cloudflared..."
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

# 6. Background daemons
echo "[6/7] Starting keepalive + GCS sync daemons..."
nohup uv run python scripts/keepalive.py > /tmp/keepalive.log 2>&1 &
AUTOLEARNMEDS_GCS_BUCKET="$BUCKET" \
AUTOLEARNMEDS_WORKSPACE="$WORKSPACE" \
nohup bash scripts/sync_to_gcs.sh > /tmp/sync_to_gcs.log 2>&1 &

# 7. Success banner
echo "============================================================"
echo "[7/7] READY"
echo "  Workspace:     $WORKSPACE"
echo "  GCS bucket:    $BUCKET"
echo "  Cloudflared:   ${CLOUDFLARED_HOST:-<see /tmp/colab_ssh_output.txt>}"
echo ""
echo "On your Mac, run:"
echo "  ./scripts/update_ssh_config.sh '${CLOUDFLARED_HOST:-<host>}'"
echo "Then in VSCode:"
echo "  ⌘⇧P → Remote-SSH: Connect to Host → autolearnmeds-colab"
echo "============================================================"
