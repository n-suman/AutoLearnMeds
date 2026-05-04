#!/usr/bin/env bash
# AutoLearnMeds — one-cell Colab Pro+ bootstrap.
#
# Run from a Colab notebook cell (after the cell has done drive.mount and
# auth.authenticate_user — those need the IPython kernel):
#   !curl -sSL https://raw.githubusercontent.com/<user>/<repo>/<branch>/scripts/colab_bootstrap.sh | bash
#
# At end, prints an SSH command to copy into your Mac's ~/.ssh/config.
#
# Required env vars:
#   AUTOLEARNMEDS_GCS_BUCKET    gs://your-bucket-name
#   AUTOLEARNMEDS_REPO_URL      https://github.com/<user>/<repo>.git
#   AUTOLEARNMEDS_SSH_PASSWORD  temp password for SSH (use a strong random)
#
# Optional:
#   AUTOLEARNMEDS_BRANCH        repo branch to clone (default: main)
#
# Note: data is uploaded directly to gs://${BUCKET}/raw/* by the user
# (no Drive→GCS sync inside this bootstrap).

set -euo pipefail

BUCKET="${AUTOLEARNMEDS_GCS_BUCKET:?Set AUTOLEARNMEDS_GCS_BUCKET=gs://your-bucket}"
REPO_URL="${AUTOLEARNMEDS_REPO_URL:?Set AUTOLEARNMEDS_REPO_URL=https://github.com/...}"
SSH_PASSWORD="${AUTOLEARNMEDS_SSH_PASSWORD:?Set AUTOLEARNMEDS_SSH_PASSWORD=...}"
BRANCH="${AUTOLEARNMEDS_BRANCH:-main}"
DRIVE_PROJECT_DIR="/content/drive/MyDrive/AutoLearnMeds"
WORKSPACE="/workspace"

echo "============================================================"
echo "AutoLearnMeds bootstrap — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  branch=$BRANCH"
echo "============================================================"

# 1. Verify Google Drive is mounted (notebook cell does the actual mount because
#    google.colab.drive.mount needs the IPython kernel; subprocesses can't.)
echo "[1/7] Verifying Google Drive mount..."
if ! mountpoint -q /content/drive 2>/dev/null && [[ ! -d /content/drive/MyDrive ]]; then
  echo "FATAL: /content/drive is not mounted." >&2
  echo "  This script expects the calling notebook cell to have run:" >&2
  echo "    from google.colab import drive; drive.mount('/content/drive')" >&2
  echo "  (drive.mount needs the IPython kernel, which subprocesses don't have.)" >&2
  exit 1
fi
echo "  OK: /content/drive is mounted"

# 2. Mount GCS bucket (gcsfuse)
echo "[2/7] Mounting GCS bucket $BUCKET..."
if ! command -v gcsfuse >/dev/null 2>&1; then
  echo "  installing gcsfuse..."
  echo "deb https://packages.cloud.google.com/apt gcsfuse-bookworm main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
  # apt update can fail on pre-installed Colab launchpad PPAs (deadsnakes,
  # graphics-drivers, ubuntugis). Those aren't needed by us. Tolerate partial
  # failures and use short HTTP timeouts so flaky PPAs fail fast (~15s vs ~100s).
  sudo apt-get update -q \
       -o Acquire::http::Timeout=15 \
       -o Acquire::https::Timeout=15 \
       -o APT::Update::Error-Mode=any || \
    echo "  WARN: apt-get update had partial failures (likely unrelated PPAs); continuing"
  # Try apt install; fall back to a direct .deb if the gcsfuse source itself was down.
  if ! sudo apt-get install -y -q gcsfuse; then
    echo "  apt install gcsfuse failed; downloading .deb directly..."
    GCSFUSE_VER="2.7.0"
    curl -fsSL "https://github.com/GoogleCloudPlatform/gcsfuse/releases/download/v${GCSFUSE_VER}/gcsfuse_${GCSFUSE_VER}_amd64.deb" -o /tmp/gcsfuse.deb
    sudo dpkg -i /tmp/gcsfuse.deb || sudo apt-get install -f -y
  fi
fi
mkdir -p /mnt/gcs
GCS_BUCKET_NAME="${BUCKET#gs://}"
if ! mountpoint -q /mnt/gcs; then
  gcsfuse "$GCS_BUCKET_NAME" /mnt/gcs || \
    echo "  WARN: gcsfuse mount failed (auth?). GCS sync will still work via gsutil."
fi

# 3. Clone repo into Drive (so it survives runtime restarts) and symlink to /workspace
echo "[3/7] Setting up workspace (branch=$BRANCH)..."
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
echo "[4/7] Installing uv and syncing deps..."
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --extra ml --extra colab

# 5. Launch SSH server + Cloudflare Tunnel
# IMPORTANT: colab-ssh imports `apt`, the python-apt module that ships with
# Debian/Ubuntu's system Python. Our project venv (Python 3.11 created by uv)
# does NOT have it. Run from a system Python that does.
echo "[5/7] Launching SSH + cloudflared..."
SYSTEM_PYTHON=""
for candidate in \
    /usr/local/bin/python3.12 /usr/local/bin/python3.11 /usr/local/bin/python3.10 \
    /usr/bin/python3.12 /usr/bin/python3.11 /usr/bin/python3.10 \
    /usr/bin/python3
do
  if [[ -x "$candidate" ]] && "$candidate" -c "import apt" 2>/dev/null; then
    SYSTEM_PYTHON="$candidate"
    break
  fi
done

if [[ -z "$SYSTEM_PYTHON" ]]; then
  echo "FATAL: no system Python with the 'apt' module found (needed by colab-ssh)" >&2
  echo "  Tried: /usr/local/bin/python3.{10,11,12} /usr/bin/python3.{10,11,12} /usr/bin/python3" >&2
  exit 1
fi
echo "  using system Python: $SYSTEM_PYTHON"

# Install colab-ssh into the system Python's user site (no venv overlay).
"$SYSTEM_PYTHON" -m pip install -q --user colab-ssh

"$SYSTEM_PYTHON" -c "
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
echo "  Branch:        $BRANCH"
echo "  Cloudflared:   ${CLOUDFLARED_HOST:-<see /tmp/colab_ssh_output.txt>}"
echo ""
echo "On your Mac, run:"
echo "  ./scripts/update_ssh_config.sh '${CLOUDFLARED_HOST:-<host>}'"
echo "Then in VSCode:"
echo "  Cmd-Shift-P -> Remote-SSH: Connect to Host -> autolearnmeds-colab"
echo "============================================================"
