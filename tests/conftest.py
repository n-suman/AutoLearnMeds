"""Pytest configuration: project-root path fixture."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the repo root."""
    return Path(__file__).resolve().parent.parent
