"""Tests for prepare.py image preprocessing + Dataset + DataLoader.

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


def test_preprocess_image_returns_tensor(
    prepare_mod, synthetic_images_dir: Path
) -> None:
    import torch  # type: ignore[import-not-found]
    img = synthetic_images_dir / "IMG_synth_001.jpeg"
    t = prepare_mod.preprocess_image(img)
    assert isinstance(t, torch.Tensor)
    assert t.shape == (3, prepare_mod.IMAGE_SIZE, prepare_mod.IMAGE_SIZE)


def test_preprocess_image_letterbox_aspect(
    prepare_mod, tmp_path: Path
) -> None:
    """Non-square images should be letterboxed to IMAGE_SIZE x IMAGE_SIZE."""
    from PIL import Image

    img_path = tmp_path / "wide.jpeg"
    Image.new("RGB", (256, 64), color=(127, 127, 127)).save(img_path, "JPEG")
    t = prepare_mod.preprocess_image(img_path)
    assert t.shape == (3, prepare_mod.IMAGE_SIZE, prepare_mod.IMAGE_SIZE)


def test_pharma_label_dataset_loads(
    prepare_mod,
    synthetic_jsonl_dir: Path,
    synthetic_images_dir: Path,
    tiny_processed_dir: Path,
    project_root: Path,
) -> None:
    # Build processed JSONL from synthetic fixtures
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
    assert len(ds) == 3
    item = ds[0]
    assert "image" in item
    assert "target_text" in item
    assert "image_id" in item
    assert item["image"].shape == (3, prepare_mod.IMAGE_SIZE, prepare_mod.IMAGE_SIZE)
    assert isinstance(item["target_text"], str)


def test_get_dataloader_iterates(
    prepare_mod,
    synthetic_jsonl_dir: Path,
    synthetic_images_dir: Path,
    tiny_processed_dir: Path,
    project_root: Path,
) -> None:
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
    dl = prepare_mod.get_dataloader(
        jsonl_path=tiny_processed_dir / "train.jsonl",
        images_root=synthetic_images_dir,
        path_strip_prefix="raw/raw_images/",
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(dl))
    assert batch["image"].shape[0] == 2  # batch dim
    assert len(batch["target_text"]) == 2
    assert len(batch["image_id"]) == 2
