#!/usr/bin/env python3
"""Colab keepalive — writes a heartbeat once per minute.

Two purposes:
1. Touches a file frequently enough to discourage idle-disconnect.
2. Lets the user side check liveness without logging into Colab
   (the heartbeat file is mirrored to GCS by sync_to_gcs.sh).

Run: nohup python scripts/keepalive.py > /tmp/keepalive.log 2>&1 &
"""
from __future__ import annotations

import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

HEARTBEAT_PATH = Path(os.environ.get("AUTOLEARNMEDS_HEARTBEAT", "/workspace/heartbeat.txt"))
INTERVAL_SECONDS = int(os.environ.get("AUTOLEARNMEDS_HEARTBEAT_INTERVAL", "60"))


def beat() -> None:
    HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT_PATH.write_text(datetime.now(UTC).isoformat() + "\n")


def main() -> int:
    print(f"[keepalive] writing to {HEARTBEAT_PATH} every {INTERVAL_SECONDS}s", flush=True)
    while True:
        try:
            beat()
        except OSError as e:
            # Don't die on transient FS issues — log and keep going.
            print(f"[keepalive] WARN: {e}", file=sys.stderr, flush=True)
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
