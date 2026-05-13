"""Smoke tests for the Track E plot scripts."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_per_field_safety4_plot(project_root: Path, tmp_path: Path) -> None:
    track_a = tmp_path / "per_field_track_a.json"
    track_e = tmp_path / "per_field_track_e.json"
    track_a.write_text(json.dumps({
        "per_field_f1": {"batch_number": 0.0, "mfg_date": 0.05, "expiry_date": 0.05, "mrp": 0.10},
        "per_field_edit_f1": {"batch_number": 0.18, "mfg_date": 0.22, "expiry_date": 0.29, "mrp": 0.35},
    }))
    track_e.write_text(json.dumps({
        "per_field_f1": {"batch_number": 0.30, "mfg_date": 0.40, "expiry_date": 0.45, "mrp": 0.50},
        "per_field_edit_f1": {"batch_number": 0.55, "mfg_date": 0.60, "expiry_date": 0.65, "mrp": 0.70},
    }))
    out = tmp_path / "per_field_safety4.png"
    result = subprocess.run(
        ["python", "scripts/plots/per_field_safety4.py",
         "--per-field", f"track_a={track_a}", f"track_e={track_e}",
         "--metric", "edit",
         "--out", str(out)],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists() and out.stat().st_size > 1000


def test_envelope_plot(project_root: Path, tmp_path: Path) -> None:
    """Envelope across multiple runs reads ledger.jsonl entries and produces a PNG."""
    import json
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("\n".join(json.dumps(r) for r in [
        {"run_id": "track_a_baseline", "metrics": {"safety4_macro_edit_f1": 0.10}, "timestamp": "2026-05-05T14:00:00Z"},
        {"run_id": "track_e_gold_only", "metrics": {"safety4_macro_edit_f1": 0.18}, "timestamp": "2026-05-13T14:00:00Z"},
        {"run_id": "track_e_full", "metrics": {"safety4_macro_edit_f1": 0.25}, "timestamp": "2026-05-14T14:00:00Z"},
    ]) + "\n")
    out = tmp_path / "envelope.png"
    result = subprocess.run(
        ["python", "scripts/plots/envelope.py", "--ledger", str(ledger), "--out", str(out)],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists() and out.stat().st_size > 1000
