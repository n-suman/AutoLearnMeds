"""Lightweight (non-Colab) unit tests for train_yolo.py pure-logic helpers."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture
def train_yolo_mod():
    """Import train_yolo.py from the project root, fresh per test.
    Lazy ml imports inside train_yolo.py mean this works in the torch-less Mac venv.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if "train_yolo" in sys.modules:
        importlib.reload(sys.modules["train_yolo"])
    import train_yolo
    return train_yolo


def test_yolo_config_loads_baseline_yaml(train_yolo_mod, project_root: Path) -> None:
    cfg = train_yolo_mod.YoloConfig.from_yaml(
        project_root / "experiments" / "configs" / "yolo_baseline.yaml"
    )
    assert cfg.model == "yolov12s.pt"
    assert cfg.imgsz == 640
    assert cfg.epochs == 100
    assert cfg.hsv_s == 0.0


def test_train_yolo_module_imports_without_torch() -> None:
    """train_yolo.py must import in a torch-less venv (lazy ml imports).

    Confirms torch and ultralytics are not in the module's top-level namespace.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if "train_yolo" in sys.modules:
        del sys.modules["train_yolo"]
    import train_yolo
    assert not hasattr(train_yolo, "torch")
    assert not hasattr(train_yolo, "ultralytics")
