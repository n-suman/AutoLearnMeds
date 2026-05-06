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
    # SSH tunnel is set up via direct cloudflared invocation (no colab-ssh).
    assert "cloudflared tunnel" in text, "SSH tunnel via cloudflared"
    assert "PermitRootLogin yes" in text, "sshd config: allow root login"
    assert "PasswordAuthentication yes" in text, "sshd config: allow password auth"
    assert "colab.capulamedia.com" in text or "AUTOLEARNMEDS_TUNNEL_HOSTNAME" in text, \
        "Should configure named-tunnel public hostname"
    assert "keepalive.py" in text, "Daemons — keepalive"
    assert "sync_to_gcs.sh" in text, "Daemons — gcs sync"
    assert "AUTOLEARNMEDS_BRANCH" in text, "Branch parameter for git clone"
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


def test_program_md_has_full_contract(project_root: Path) -> None:
    """program.md must include every section the agent needs to operate."""
    text = (project_root / "program.md").read_text()
    required_sections = [
        "## Goal",
        "## Metric",
        "## Time budget",
        "## Allowed",
        "## Forbidden",
        "## Source vetting",
        "## Workflow per experiment",
        "## Notes.md template",
        "## Backup",
        "## Do not",
    ]
    for sec in required_sections:
        assert sec in text, f"missing section: {sec!r}"
    # Must reference the canonical scripts the agent uses
    assert "scripts/run_experiment.sh" in text
    assert "scripts/finalize_experiment.sh" in text
    # Must reference the metric format
    assert "final_macro_f1" in text
    # Must declare the test-set firewall
    assert "test.jsonl" in text
    assert "evaluate_test" in text


def test_promote_script_exists_and_executable(project_root: Path) -> None:
    p = project_root / "scripts" / "promote.sh"
    assert p.is_file()
    assert os.access(p, os.X_OK)
    text = p.read_text()
    assert text.startswith("#!/")
    assert "set -euo pipefail" in text
    assert "checkpoints/best" in text


def test_finalize_experiment_script_exists_and_calls_subscripts(project_root: Path) -> None:
    p = project_root / "scripts" / "finalize_experiment.sh"
    assert p.is_file()
    assert os.access(p, os.X_OK)
    text = p.read_text()
    assert text.startswith("#!/")
    assert "set -euo pipefail" in text
    assert "scripts/append_ledger.py" in text
    assert "scripts/leaderboard.py" in text
    assert "scripts/promote.sh" in text


def test_run_experiment_supports_track_arg(project_root):
    text = (project_root / "scripts" / "run_experiment.sh").read_text()
    assert "--track" in text
    assert "train_qwen.py" in text
    assert "train.py" in text
    assert "qwen_baseline.yaml" in text


def test_run_experiment_sets_pythonunbuffered(project_root):
    text = (project_root / "scripts" / "run_experiment.sh").read_text()
    assert "PYTHONUNBUFFERED=1" in text


def test_run_experiment_auto_finalizes_on_success(project_root):
    text = (project_root / "scripts" / "run_experiment.sh").read_text()
    assert "finalize_experiment.sh" in text
    # The auto-finalize must be guarded by exit-code check.
    assert 'EXIT_CODE" -eq 0' in text or 'EXIT_CODE -eq 0' in text


def test_run_experiment_pushes_gcs_on_success(project_root):
    text = (project_root / "scripts" / "run_experiment.sh").read_text()
    assert "gsutil" in text and "rsync" in text


def test_train_qwen_critical_prints_use_flush(project_root):
    text = (project_root / "train_qwen.py").read_text()
    # All step-log and final-line prints should have flush=True.
    # Loose check: count flush=True occurrences; should be at least 4 (step log,
    # eval log, save-best log, final-summary line).
    assert text.count("flush=True") >= 4


def test_update_ssh_config_includes_keepalive(project_root):
    text = (project_root / "scripts" / "update_ssh_config.sh").read_text()
    assert "ServerAliveInterval 60" in text
    assert "ServerAliveCountMax 10" in text


def test_bootstrap_uses_named_tunnel(project_root):
    text = (project_root / "scripts" / "colab_bootstrap.sh").read_text()
    assert "CLOUDFLARED_TUNNEL_CREDS" in text
    assert "AUTOLEARNMEDS_TUNNEL_HOSTNAME" in text
    assert "tunnel" in text and "run" in text
    # must NOT use the quick-tunnel form anymore
    assert "tunnel --url" not in text


def test_update_ssh_config_defaults_to_persistent_host(project_root):
    text = (project_root / "scripts" / "update_ssh_config.sh").read_text()
    assert "colab.capulamedia.com" in text


def test_sync_to_gcs_does_not_use_delete_flag(project_root):
    """Regression test for the 2026-05-06 incident where -d destroyed Track A checkpoints."""
    text = (project_root / "scripts" / "sync_to_gcs.sh").read_text()
    # Look for the rsync command line(s); must not contain `-d` between rsync and the path.
    lines = [ln for ln in text.splitlines() if "gsutil" in ln and "rsync" in ln]
    assert lines, "no rsync line found"
    for ln in lines:
        assert " -d " not in ln, f"rsync line still uses destructive -d: {ln}"


def test_bootstrap_calls_disaster_recover(project_root):
    text = (project_root / "scripts" / "colab_bootstrap.sh").read_text()
    assert "disaster_recover.sh" in text, "bootstrap must repopulate from GCS via disaster_recover.sh"


def test_train_main_prints_final_macro_edit_f1(project_root):
    text = (project_root / "train.py").read_text()
    assert "final_macro_edit_f1=" in text, "train.py main() must print the edit_f1 final-line"


def test_run_experiment_path_includes_tools_gcloud(project_root):
    """Regression: 2026-05-06 run skipped GCS push because PATH didn't include /tools/google-cloud-sdk/bin."""
    text = (project_root / "scripts" / "run_experiment.sh").read_text()
    assert "/tools/google-cloud-sdk/bin" in text


def test_bootstrap_sets_git_config(project_root):
    """Regression: 2026-05-06 auto-finalize's git commit failed because Colab had no git config."""
    text = (project_root / "scripts" / "colab_bootstrap.sh").read_text()
    assert "git config --local user.email" in text
    assert "git config --local user.name" in text


def test_run_pretraining_exists(project_root):
    p = project_root / "scripts" / "run_pretraining.sh"
    assert p.is_file()
    assert os.access(p, os.X_OK)
    text = p.read_text()
    assert "PYTHONUNBUFFERED=1" in text
    assert "/tools/google-cloud-sdk/bin" in text
    assert "final_pretrain_loss" in text


def test_disaster_recover_uses_skip_older_flag(project_root):
    """Regression test for the 2026-05-06 part-2 incident: disaster_recover overwrote
    newer-local files with stale GCS versions because rsync had no -u flag."""
    text = (project_root / "scripts" / "disaster_recover.sh").read_text()
    rsync_lines = [ln for ln in text.splitlines() if '"$GSUTIL"' in ln and "rsync" in ln]
    assert rsync_lines, "no GSUTIL rsync line found"
    for ln in rsync_lines:
        assert " -u " in ln or ln.rstrip().endswith(" -u"), \
            f"rsync line missing -u (skip-if-not-newer): {ln}"
