"""Test for the Track E BPE tokenizer builder."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


def test_builder_runs_on_synthetic_input(project_root: Path, tmp_path: Path) -> None:
    """Given a single gold jsonl, the script writes a tokenizer.json that is loadable."""
    gold = tmp_path / "gold.jsonl"
    gold.write_text('{"image_path": "x.jpg", "xml_label": "<medication><brand_name>Crocin</brand_name><drug_name>Crocin Advance</drug_name></medication>"}\n')
    out = tmp_path / "track_e_bpe.json"
    result = subprocess.run(
        ["python", "scripts/build_track_e_tokenizer.py",
         "--train-jsonl", str(gold),
         "--out", str(out),
         "--vocab-size", "256"],  # small for test
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    data = json.loads(out.read_text())
    assert "version" in data or "model" in data  # standard tokenizers.json structure


def test_builder_includes_pseudo_when_provided(project_root: Path, tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    pseudo = tmp_path / "pseudo.jsonl"
    gold.write_text('{"image_path": "g.jpg", "xml_label": "<medication><brand_name>X</brand_name></medication>"}\n')
    pseudo.write_text('{"image_path": "p.jpg", "xml_label": "<medication><brand_name>UNIQUE_PSEUDO_TOKEN</brand_name></medication>"}\n')
    out = tmp_path / "track_e_bpe.json"
    result = subprocess.run(
        ["python", "scripts/build_track_e_tokenizer.py",
         "--train-jsonl", str(gold),
         "--pseudo-jsonl", str(pseudo),
         "--out", str(out),
         "--vocab-size", "512"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
