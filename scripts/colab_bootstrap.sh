#!/usr/bin/env bash
# AutoLearnMeds — one-cell Colab Pro+ bootstrap.
#
# Run from a Colab notebook cell (after the cell has done auth.authenticate_user
# — that needs the IPython kernel):
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
#   AUTOLEARNMEDS_PROJECT_DIR   where to clone (default: /content/AutoLearnMeds
#                               on local SSD; do NOT put on Drive — uv sync writes
#                               thousands of files which trip Drive's API quota)
#
# Persistence model:
# - Code: GitHub (committed + pushed; re-cloned on each session — fast).
# - Data: gs://${BUCKET}/raw/* (uploaded by user directly to GCS).
# - Experiment ledger + checkpoints: GCS (synced live by sync_to_gcs.sh).
# - Drive is NOT in the working path — too quota-prone for ML workloads.

set -euo pipefail

BUCKET="${AUTOLEARNMEDS_GCS_BUCKET:?Set AUTOLEARNMEDS_GCS_BUCKET=gs://your-bucket}"
REPO_URL="${AUTOLEARNMEDS_REPO_URL:?Set AUTOLEARNMEDS_REPO_URL=https://github.com/...}"
SSH_PASSWORD="${AUTOLEARNMEDS_SSH_PASSWORD:?Set AUTOLEARNMEDS_SSH_PASSWORD=...}"
BRANCH="${AUTOLEARNMEDS_BRANCH:-main}"
PROJECT_DIR="${AUTOLEARNMEDS_PROJECT_DIR:-/content/AutoLearnMeds}"
WORKSPACE="/workspace"

echo "============================================================"
echo "AutoLearnMeds bootstrap — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  branch=$BRANCH"
echo "  project_dir=$PROJECT_DIR"
echo "============================================================"

# 1. Verify GCP auth (notebook cell does auth.authenticate_user, which sets ADC).
#    drive.mount is intentionally NOT used — putting the project on Drive triggers
#    Google Drive's per-user API rate limit when uv sync writes ~thousands of
#    small dependency files.
echo "[1/7] Verifying GCP auth (gsutil ls of bucket)..."
if ! gsutil ls "$BUCKET" >/dev/null 2>&1; then
  echo "FATAL: cannot 'gsutil ls $BUCKET'." >&2
  echo "  This script expects the calling notebook cell to have run:" >&2
  echo "    from google.colab import auth; auth.authenticate_user()" >&2
  echo "  (auth.authenticate_user needs the IPython kernel; subprocesses can't.)" >&2
  exit 1
fi
echo "  OK: gsutil reaches $BUCKET"

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

# 3. Clone repo to LOCAL SSD (not Drive) and symlink to /workspace.
#    Local SSD = fast random IO, no Drive API quota. Re-cloning every session
#    is cheap (under 5s for our small repo).
echo "[3/7] Setting up workspace (branch=$BRANCH)..."
if [[ ! -d "$PROJECT_DIR/.git" ]]; then
  git clone -b "$BRANCH" "$REPO_URL" "$PROJECT_DIR"
else
  cd "$PROJECT_DIR" && \
    git fetch origin && \
    git checkout "$BRANCH" && \
    git pull --ff-only origin "$BRANCH" || \
    echo "  WARN: git pull failed (continuing with existing copy)"
fi
ln -sfn "$PROJECT_DIR" "$WORKSPACE"
cd "$WORKSPACE"

# 4. Install uv + sync deps (full ml + colab extras on Colab)
echo "[4/7] Installing uv and syncing deps..."
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --extra ml --extra colab

# Make NVIDIA libraries discoverable to non-interactive shells (e.g., the SSH
# session that VSCode connects through). Colab's interactive shell sets
# LD_LIBRARY_PATH to include /usr/lib64-nvidia, but SSH sessions don't inherit
# that, so `nvidia-smi` and any CUDA-using process can't find libnvidia-ml.so.
# Add the path to the system linker config so all shells see it.
if [[ -d /usr/lib64-nvidia ]] && [[ ! -f /etc/ld.so.conf.d/nvidia.conf ]]; then
  echo "  registering /usr/lib64-nvidia with ldconfig (so SSH sees nvidia-smi)..."
  echo "/usr/lib64-nvidia" | sudo tee /etc/ld.so.conf.d/nvidia.conf >/dev/null
  sudo ldconfig 2>/dev/null || true
fi

# 5. Launch SSH server + Cloudflare Tunnel — pure shell (no colab-ssh).
# colab-ssh got killed by something (OOM? sandbox?) on the user's last run,
# and its only value over plain bash is convenience. Bypass it: install
# cloudflared, configure sshd ourselves, run cloudflared tunnel, scrape URL.
echo "[5/7] Launching SSH + cloudflared..."

# Install openssh-server if not present
if ! command -v sshd >/dev/null 2>&1 && [[ ! -x /usr/sbin/sshd ]]; then
  echo "  installing openssh-server..."
  sudo apt-get install -y -q openssh-server || \
    echo "  WARN: openssh-server install failed; sshd may already be present"
fi

# Configure sshd to accept root login with password (cloudflared tunnels SSH).
echo "  configuring sshd (root login + password auth)..."
echo "root:$SSH_PASSWORD" | sudo chpasswd
sudo sed -i \
  -e 's/^#\?PermitRootLogin.*/PermitRootLogin yes/' \
  -e 's/^#\?PasswordAuthentication.*/PasswordAuthentication yes/' \
  /etc/ssh/sshd_config

# Restart sshd. service may not exist; fall back to direct invocation.
sudo service ssh restart 2>/dev/null || sudo /usr/sbin/sshd -D &
sleep 1

# Install cloudflared binary if not already present.
if ! command -v cloudflared >/dev/null 2>&1; then
  echo "  installing cloudflared (Cloudflare Tunnel client)..."
  sudo curl -fsSL -o /usr/local/bin/cloudflared \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  sudo chmod +x /usr/local/bin/cloudflared
fi

# Launch the tunnel in background. cloudflared writes its public hostname
# (somefoo.trycloudflare.com) to its own stderr; we tail for it.
echo "  launching cloudflared tunnel..."
: > /tmp/cloudflared.log
nohup cloudflared tunnel --url tcp://localhost:22 \
                         --no-autoupdate \
                         --logfile /tmp/cloudflared.log \
                         > /tmp/cloudflared.stdout.log 2>&1 &
CLOUDFLARED_PID=$!

# Wait up to 30s for the tunnel to come up and emit the trycloudflare hostname.
# Use cat | grep so we don't pick up grep's "filename:" prefix when matching
# across multiple files; -h would also work but `cat | grep` is unambiguous.
CLOUDFLARED_HOST=""
for _ in $(seq 1 15); do
  sleep 2
  CLOUDFLARED_HOST="$(cat /tmp/cloudflared.log /tmp/cloudflared.stdout.log 2>/dev/null \
                      | grep -oE '[a-zA-Z0-9-]+\.trycloudflare\.com' \
                      | head -1 || true)"
  [[ -n "$CLOUDFLARED_HOST" ]] && break
done

if [[ -z "$CLOUDFLARED_HOST" ]]; then
  echo "FATAL: cloudflared tunnel did not establish in 30s." >&2
  echo "  Tail of /tmp/cloudflared.log:" >&2
  tail -30 /tmp/cloudflared.log /tmp/cloudflared.stdout.log >&2 || true
  exit 1
fi

# Persist the host alongside the legacy /tmp/colab_ssh_output.txt path used
# by other tooling (and grepped by older docs).
echo "ssh root@$CLOUDFLARED_HOST" | tee /tmp/colab_ssh_output.txt
echo "$CLOUDFLARED_HOST" > "$WORKSPACE/.colab_ssh_host"
gsutil cp "$WORKSPACE/.colab_ssh_host" "$BUCKET/.colab_ssh_host" 2>/dev/null || true
echo "  cloudflared tunnel ESTABLISHED: $CLOUDFLARED_HOST (pid $CLOUDFLARED_PID)"

# 6. Background daemons
echo "[6/7] Starting keepalive + GCS sync daemons..."
nohup uv run python scripts/keepalive.py > /tmp/keepalive.log 2>&1 &
AUTOLEARNMEDS_GCS_BUCKET="$BUCKET" \
AUTOLEARNMEDS_WORKSPACE="$WORKSPACE" \
nohup bash scripts/sync_to_gcs.sh > /tmp/sync_to_gcs.log 2>&1 &

# 7. Success banner
echo "============================================================"
echo "[7/7] READY"
echo "  Workspace:     $WORKSPACE  ->  $PROJECT_DIR"
echo "  GCS bucket:    $BUCKET"
echo "  Branch:        $BRANCH"
echo "  Cloudflared:   ${CLOUDFLARED_HOST:-<see /tmp/colab_ssh_output.txt>}"
echo ""
echo "On your Mac, run:"
echo "  ./scripts/update_ssh_config.sh '${CLOUDFLARED_HOST:-<host>}'"
echo "Then in VSCode:"
echo "  Cmd-Shift-P -> Remote-SSH: Connect to Host -> autolearnmeds-colab"
echo "============================================================"
