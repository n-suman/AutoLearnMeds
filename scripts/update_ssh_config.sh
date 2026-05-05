#!/usr/bin/env bash
# Updates the user's ~/.ssh/config with the current Colab cloudflared host.
#
# Run on the USER's MAC, not on Colab. Reads the hostname from a known
# location (either an arg or stdin). Replaces the existing
# `Host autolearnmeds-colab` block atomically.
#
# Usage:
#   ./scripts/update_ssh_config.sh <hostname>
#   echo "myhost.trycloudflare.com" | ./scripts/update_ssh_config.sh

set -euo pipefail

HOSTNAME="${1:-}"
if [[ -z "$HOSTNAME" ]]; then
  HOSTNAME="$(cat -)"
fi
HOSTNAME="$(echo "$HOSTNAME" | tr -d '[:space:]')"

if [[ -z "$HOSTNAME" ]]; then
  echo "[update_ssh_config] FATAL: no hostname provided (arg or stdin)" >&2
  exit 1
fi

SSH_CONFIG="$HOME/.ssh/config"
mkdir -p "$HOME/.ssh"
touch "$SSH_CONFIG"
chmod 600 "$SSH_CONFIG"

# Atomically rewrite ~/.ssh/config: strip any existing
# `Host autolearnmeds-colab` block, then append the new one.
TMPFILE="$(mktemp)"
awk '
  /^Host autolearnmeds-colab/,/^$/ { next }
  { print }
' "$SSH_CONFIG" > "$TMPFILE"

cat >> "$TMPFILE" <<EOF

Host autolearnmeds-colab
    HostName $HOSTNAME
    User root
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ProxyCommand cloudflared access ssh --hostname %h
    ServerAliveInterval 60
    ServerAliveCountMax 10
EOF

mv "$TMPFILE" "$SSH_CONFIG"
chmod 600 "$SSH_CONFIG"

echo "[update_ssh_config] updated ~/.ssh/config with HostName=$HOSTNAME"
echo "[update_ssh_config] connect via: ssh autolearnmeds-colab"
