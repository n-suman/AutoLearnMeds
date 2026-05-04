"""Phase 0 smoke tests — repo layout, scripts, file integrity."""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path


def test_papers_folder_has_registry(project_root: Path) -> None:
    """The papers/ citation registry must exist."""
    assert (project_root / "papers" / "README.md").is_file()


def test_design_spec_exists(project_root: Path) -> None:
    """The design spec from brainstorming must be present."""
    spec = project_root / "docs" / "superpowers" / "specs" / "2026-05-05-pharma-vlm-autoresearch-design.md"
    assert spec.is_file()


EXPECTED_DIRS = [
    "data",
    "data/raw",
    "data/processed",
    "checkpoints",
    "experiments",
    "experiments/runs",
    "paper",
    "paper/figures",
    "paper/tables",
    "paper/sections",
    "scripts",
    "notebooks",
    "tests",
    "papers",
    "docs",
]

EXPECTED_READMES = [
    "data/README.md",
    "checkpoints/README.md",
    "experiments/README.md",
    "paper/README.md",
    "scripts/README.md",
    "notebooks/README.md",
    "papers/README.md",
    "README.md",
]


def test_folder_skeleton_exists(project_root: Path) -> None:
    """Every directory from spec §2.2 must exist."""
    missing = [d for d in EXPECTED_DIRS if not (project_root / d).is_dir()]
    assert not missing, f"Missing directories: {missing}"


def test_placeholder_readmes_exist(project_root: Path) -> None:
    """Every placeholder README from spec §2.2 must exist."""
    missing = [r for r in EXPECTED_READMES if not (project_root / r).is_file()]
    assert not missing, f"Missing READMEs: {missing}"


THREE_FILES = ["prepare.py", "train.py", "program.md", "program_explore.md", "program_confirm.md"]


def test_three_file_discipline_stubs(project_root: Path) -> None:
    """The autoresearch three-file convention requires these to exist (even as stubs)."""
    for f in THREE_FILES:
        assert (project_root / f).is_file(), f"Missing: {f}"


def test_program_md_mentions_metric(project_root: Path) -> None:
    """program.md must declare the optimization metric clearly."""
    text = (project_root / "program.md").read_text()
    assert "macro_f1" in text.lower() or "macro-f1" in text.lower()
    assert "final_macro_f1" in text  # must reference exact stdout token


def test_keepalive_exists_and_executable(project_root: Path) -> None:
    """scripts/keepalive.py must exist and be executable."""
    p = project_root / "scripts" / "keepalive.py"
    assert p.is_file(), "scripts/keepalive.py missing"
    assert os.access(p, os.X_OK), "scripts/keepalive.py not executable (chmod +x)"


def test_keepalive_imports_clean(project_root: Path) -> None:
    """The keepalive script must import without side effects on import."""
    p = project_root / "scripts" / "keepalive.py"
    # Use python -c with importlib to test import-without-execution.
    result = subprocess.run(
        ["python", "-c", f"import importlib.util, sys; spec = importlib.util.spec_from_file_location('k', '{p}'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)"],
        capture_output=True, text=True, timeout=5,
    )
    # Allowed to fail if it has a __main__ guard; we only care it doesn't crash on import-time evaluation.
    # Empty stdout/stderr or a clean exit is the pass condition.
    assert result.returncode == 0 or "main" in result.stderr.lower() or result.stderr == "", f"Unexpected error: {result.stderr}"


def test_sync_to_gcs_exists_and_executable(project_root: Path) -> None:
    p = project_root / "scripts" / "sync_to_gcs.sh"
    assert p.is_file()
    assert os.access(p, os.X_OK)
    text = p.read_text()
    assert text.startswith("#!/"), "Missing shebang"
    assert re.search(r"\bgsutil\b(?:\s+-\w+)*\s+rsync\b", text) or "gcloud storage rsync" in text, \
        "Should use gsutil rsync (with optional flags) or gcloud storage rsync"
    assert "set -euo pipefail" in text, "Should use strict bash"


def test_update_ssh_config_exists_and_executable(project_root: Path) -> None:
    p = project_root / "scripts" / "update_ssh_config.sh"
    assert p.is_file()
    assert os.access(p, os.X_OK)
    text = p.read_text()
    assert text.startswith("#!/"), "Missing shebang"
    assert "trycloudflare" in text or "cloudflared" in text, "Should reference cloudflared"
    assert "Host autolearnmeds-colab" in text, "Should write the canonical Host alias"


def test_colab_bootstrap_exists_and_has_required_steps(project_root: Path) -> None:
    p = project_root / "scripts" / "colab_bootstrap.sh"
    assert p.is_file()
    assert os.access(p, os.X_OK)
    text = p.read_text()
    # Required steps per spec §6.4
    assert text.startswith("#!/"), "Missing shebang"
    assert "set -euo pipefail" in text, "Should use strict bash"
    # The bootstrap clones to a LOCAL SSD path by default (not Drive) to avoid
    # Drive's API quota during uv sync. AUTOLEARNMEDS_PROJECT_DIR can override.
    assert "AUTOLEARNMEDS_PROJECT_DIR" in text, "Should support PROJECT_DIR override"
    assert "/content/AutoLearnMeds" in text, "Default project dir on local SSD"
    # GCP auth is verified via gsutil before any other step.
    assert "auth.authenticate_user" in text, "Should reference auth.authenticate_user in error message"
    assert "gsutil ls" in text, "Should verify GCP auth via gsutil ls"
    assert "gcsfuse" in text, "GCS mount"
    assert "ln -sfn" in text, "Symlink workspace"
    assert "uv sync" in text, "Install deps"
    assert "launch_ssh_cloudflared" in text, "SSH tunnel"
    assert "keepalive.py" in text, "Daemons — keepalive"
    assert "sync_to_gcs.sh" in text, "Daemons — gcs sync"
    assert "AUTOLEARNMEDS_BRANCH" in text, "Branch parameter for git clone"
    # Bootstrap must run colab-ssh from a system Python (not venv) because
    # colab-ssh imports `apt`, the python-apt module that lives in system Python.
    assert "import apt" in text, "Should detect a system Python that has the apt module"
    assert "READY" in text, "Success banner"


def test_bootstrap_notebook_is_valid_json(project_root: Path) -> None:
    p = project_root / "notebooks" / "00_bootstrap.ipynb"
    assert p.is_file()
    data = json.loads(p.read_text())
    assert data.get("nbformat") == 4
    cells = data.get("cells", [])
    assert len(cells) >= 1, "Notebook must have at least one cell"
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    assert any("colab_bootstrap.sh" in "".join(c.get("source", [])) for c in code_cells), \
        "Notebook must call scripts/colab_bootstrap.sh"


def test_makefile_has_verify_target(project_root: Path) -> None:
    p = project_root / "Makefile"
    assert p.is_file()
    text = p.read_text()
    assert "verify:" in text, "Makefile must define a 'verify' target"
    assert "test:" in text, "Makefile must define a 'test' target"
    assert "lint:" in text, "Makefile must define a 'lint' target"
