# AutoLearnMeds — top-level make targets.
.PHONY: help verify test lint sync clean check_gpu check_drive check_gcs

help:
	@echo "Targets:"
	@echo "  make verify     — full smoke test: GPU, mounts, deps, tests, ledger"
	@echo "  make test       — run pytest"
	@echo "  make lint       — run ruff"
	@echo "  make sync       — uv sync (dev extras locally; ml+colab extras on Colab)"

# --- Local + Colab targets ---

test:
	uv run pytest -v

lint:
	uv run ruff check .

sync:
	@if [ -n "$$COLAB_GPU" ] || [ -n "$$COLAB_RELEASE_TAG" ]; then \
		uv sync --extra ml --extra colab; \
	else \
		uv sync --extra dev; \
	fi

# --- Verify (the smoke test) ---
# On Colab: full check including GPU + drive mounts.
# Locally:  skips Colab-only checks gracefully.

verify: test check_gpu check_drive check_gcs
	@echo ""
	@echo "============================================================"
	@echo "  ✓ make verify GREEN — rig is ready"
	@echo "============================================================"

check_gpu:
	@echo "--- GPU ---"
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; \
	else \
		echo "  (no nvidia-smi — local Mac, skipping)"; \
	fi

check_drive:
	@echo "--- GDrive mount ---"
	@if [ -d "/content/drive/MyDrive/AutoLearnMeds" ]; then \
		echo "  ✓ /content/drive/MyDrive/AutoLearnMeds exists"; \
		ls /content/drive/MyDrive/AutoLearnMeds | head -5; \
	else \
		echo "  (not on Colab — skipping)"; \
	fi

check_gcs:
	@echo "--- GCS mount ---"
	@if [ -d "/mnt/gcs" ]; then \
		mountpoint -q /mnt/gcs && echo "  ✓ /mnt/gcs is mounted" || echo "  WARN: /mnt/gcs exists but not mounted"; \
	else \
		echo "  (not on Colab — skipping)"; \
	fi

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ tests/__pycache__
