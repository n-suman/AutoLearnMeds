"""Pytest configuration: project-root path fixture."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the repo root."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def synthetic_jsonl_dir(project_root: Path) -> Path:
    """Path to tests/fixtures/synthetic_jsonl/ (gold_standard.jsonl + splits.json)."""
    p = project_root / "tests" / "fixtures" / "synthetic_jsonl"
    assert (p / "gold_standard.jsonl").is_file()
    assert (p / "splits.json").is_file()
    return p


@pytest.fixture(scope="session")
def synthetic_images_dir(project_root: Path) -> Path:
    """Path to tests/fixtures/synthetic_images/ (5 small JPEGs)."""
    p = project_root / "tests" / "fixtures" / "synthetic_images"
    assert p.is_dir()
    return p


@pytest.fixture
def tiny_processed_dir(tmp_path: Path) -> Path:
    """Empty dir for write-test JSONL outputs."""
    d = tmp_path / "processed"
    d.mkdir()
    return d
