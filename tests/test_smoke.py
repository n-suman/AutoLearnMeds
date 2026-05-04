"""Phase 0 smoke tests — repo layout, scripts, file integrity."""
from __future__ import annotations

from pathlib import Path


def test_papers_folder_has_registry(project_root: Path) -> None:
    """The papers/ citation registry must exist."""
    assert (project_root / "papers" / "README.md").is_file()


def test_design_spec_exists(project_root: Path) -> None:
    """The design spec from brainstorming must be present."""
    spec = project_root / "docs" / "superpowers" / "specs" / "2026-05-05-pharma-vlm-autoresearch-design.md"
    assert spec.is_file()
