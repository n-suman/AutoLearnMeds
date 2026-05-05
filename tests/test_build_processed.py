"""Tests for scripts/build_processed.py — raw -> processed JSONL conversion."""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


def _load_module(project_root: Path):
    spec = importlib.util.spec_from_file_location(
        "build_processed", project_root / "scripts" / "build_processed.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_module_loads(project_root: Path) -> None:
    mod = _load_module(project_root)
    assert hasattr(mod, "build_processed")
    assert hasattr(mod, "compute_image_hash")


def test_compute_image_hash_is_deterministic(
    project_root: Path, synthetic_images_dir: Path
) -> None:
    mod = _load_module(project_root)
    img = synthetic_images_dir / "IMG_synth_001.jpeg"
    h1 = mod.compute_image_hash(img)
    h2 = mod.compute_image_hash(img)
    assert h1 == h2
    assert h1.startswith("sha256:")
    assert len(h1) == len("sha256:") + 64


def test_compute_image_hash_distinct_for_distinct_images(
    project_root: Path, synthetic_images_dir: Path
) -> None:
    mod = _load_module(project_root)
    h1 = mod.compute_image_hash(synthetic_images_dir / "IMG_synth_001.jpeg")
    h2 = mod.compute_image_hash(synthetic_images_dir / "IMG_synth_002.jpeg")
    assert h1 != h2


def test_build_processed_creates_three_jsonl_files(
    project_root: Path,
    synthetic_jsonl_dir: Path,
    synthetic_images_dir: Path,
    tiny_processed_dir: Path,
) -> None:
    mod = _load_module(project_root)
    mod.build_processed(
        gold_standard_path=synthetic_jsonl_dir / "gold_standard.jsonl",
        splits_path=synthetic_jsonl_dir / "splits.json",
        images_root=synthetic_images_dir,
        out_dir=tiny_processed_dir,
        image_path_prefix="raw/raw_images",
    )
    assert (tiny_processed_dir / "train.jsonl").is_file()
    assert (tiny_processed_dir / "val.jsonl").is_file()
    assert (tiny_processed_dir / "test.jsonl").is_file()


def test_build_processed_split_sizes(
    project_root: Path,
    synthetic_jsonl_dir: Path,
    synthetic_images_dir: Path,
    tiny_processed_dir: Path,
) -> None:
    mod = _load_module(project_root)
    mod.build_processed(
        gold_standard_path=synthetic_jsonl_dir / "gold_standard.jsonl",
        splits_path=synthetic_jsonl_dir / "splits.json",
        images_root=synthetic_images_dir,
        out_dir=tiny_processed_dir,
        image_path_prefix="raw/raw_images",
    )
    train = (tiny_processed_dir / "train.jsonl").read_text().splitlines()
    val = (tiny_processed_dir / "val.jsonl").read_text().splitlines()
    test = (tiny_processed_dir / "test.jsonl").read_text().splitlines()
    assert len(train) == 3
    assert len(val) == 1
    assert len(test) == 1


def test_build_processed_record_schema(
    project_root: Path,
    synthetic_jsonl_dir: Path,
    synthetic_images_dir: Path,
    tiny_processed_dir: Path,
) -> None:
    mod = _load_module(project_root)
    mod.build_processed(
        gold_standard_path=synthetic_jsonl_dir / "gold_standard.jsonl",
        splits_path=synthetic_jsonl_dir / "splits.json",
        images_root=synthetic_images_dir,
        out_dir=tiny_processed_dir,
        image_path_prefix="raw/raw_images",
    )
    line0 = (tiny_processed_dir / "train.jsonl").read_text().splitlines()[0]
    rec = json.loads(line0)
    # Required added fields
    assert "image_id" in rec
    assert "image_file" in rec
    assert "image_path" in rec
    assert "image_hash" in rec
    assert "split" in rec
    assert rec["image_hash"].startswith("sha256:")
    assert rec["image_path"].startswith("raw/raw_images/")
    assert rec["image_id"].startswith("IMG_synth_")
    # Original fields preserved
    assert "fields" in rec
    assert "medicine_name" in rec
    assert rec["split"] == "train"


def test_build_processed_idempotent(
    project_root: Path,
    synthetic_jsonl_dir: Path,
    synthetic_images_dir: Path,
    tiny_processed_dir: Path,
) -> None:
    mod = _load_module(project_root)
    args = dict(
        gold_standard_path=synthetic_jsonl_dir / "gold_standard.jsonl",
        splits_path=synthetic_jsonl_dir / "splits.json",
        images_root=synthetic_images_dir,
        out_dir=tiny_processed_dir,
        image_path_prefix="raw/raw_images",
    )
    mod.build_processed(**args)
    train_v1 = (tiny_processed_dir / "train.jsonl").read_text()
    mod.build_processed(**args)
    train_v2 = (tiny_processed_dir / "train.jsonl").read_text()
    assert train_v1 == train_v2


def test_build_processed_cli_invocation(
    project_root: Path,
    synthetic_jsonl_dir: Path,
    synthetic_images_dir: Path,
    tiny_processed_dir: Path,
) -> None:
    """The script must be runnable as a CLI."""
    result = subprocess.run(
        [
            "python",
            str(project_root / "scripts" / "build_processed.py"),
            "--gold-standard", str(synthetic_jsonl_dir / "gold_standard.jsonl"),
            "--splits", str(synthetic_jsonl_dir / "splits.json"),
            "--images-root", str(synthetic_images_dir),
            "--out-dir", str(tiny_processed_dir),
            "--image-path-prefix", "raw/raw_images",
        ],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert (tiny_processed_dir / "train.jsonl").is_file()
