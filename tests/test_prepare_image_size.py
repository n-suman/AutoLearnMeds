"""Tests for image_size parameter on PharmaLabelDataset + preprocess_image.

Marked @pytest.mark.colab — requires torch, transformers, Pillow. Skipped
in local Mac dev runs (default pytest config); run on Colab via
`pytest -m colab` from /workspace.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.colab


@pytest.fixture
def prepare_mod():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if "prepare" in sys.modules:
        importlib.reload(sys.modules["prepare"])
    import prepare
    return prepare


def test_preprocess_image_default_size(prepare_mod, synthetic_images_dir: Path) -> None:
    img_path = next(synthetic_images_dir.glob("*.jpeg"))
    arr = prepare_mod.preprocess_image(img_path)
    # Tensor shape is (C, H, W) — both H and W should be 224 (default IMAGE_SIZE)
    assert arr.shape[-1] == 224 and arr.shape[-2] == 224


def test_preprocess_image_custom_size(prepare_mod, synthetic_images_dir: Path) -> None:
    img_path = next(synthetic_images_dir.glob("*.jpeg"))
    arr = prepare_mod.preprocess_image(img_path, image_size=384)
    assert arr.shape[-1] == 384 and arr.shape[-2] == 384


def test_dataset_default_size_is_backwards_compatible(
    prepare_mod, synthetic_jsonl_dir: Path, synthetic_images_dir: Path, tiny_processed_dir: Path, project_root: Path
) -> None:
    """Old code path (no image_size kwarg) still returns 224x224 tensors."""
    import sys
    sys.path.insert(0, str(project_root / "scripts"))
    import build_processed
    build_processed.build_processed(
        gold_standard_path=synthetic_jsonl_dir / "gold_standard.jsonl",
        splits_path=synthetic_jsonl_dir / "splits.json",
        images_root=synthetic_images_dir,
        out_dir=tiny_processed_dir,
        image_path_prefix="raw/raw_images",
    )
    ds = prepare_mod.PharmaLabelDataset(
        jsonl_path=tiny_processed_dir / "train.jsonl",
        images_root=synthetic_images_dir,
        path_strip_prefix="raw/raw_images/",
    )
    item = ds[0]
    # Just verify the spatial dimensions match the default (backwards-compatible)
    assert item["image"].shape[-1] == 224 and item["image"].shape[-2] == 224


def test_dataset_custom_size(
    prepare_mod, synthetic_jsonl_dir: Path, synthetic_images_dir: Path, tiny_processed_dir: Path, project_root: Path
) -> None:
    """PharmaLabelDataset with image_size=384 returns 384x384 tensors."""
    import sys
    sys.path.insert(0, str(project_root / "scripts"))
    import build_processed
    build_processed.build_processed(
        gold_standard_path=synthetic_jsonl_dir / "gold_standard.jsonl",
        splits_path=synthetic_jsonl_dir / "splits.json",
        images_root=synthetic_images_dir,
        out_dir=tiny_processed_dir,
        image_path_prefix="raw/raw_images",
    )
    ds = prepare_mod.PharmaLabelDataset(
        jsonl_path=tiny_processed_dir / "train.jsonl",
        images_root=synthetic_images_dir,
        path_strip_prefix="raw/raw_images/",
        image_size=384,
    )
    item = ds[0]
    assert item["image"].shape[-1] == 384 and item["image"].shape[-2] == 384
