"""Lightweight tests for scripts/build_yolo_dataset.py polygon->bbox + writer."""
from __future__ import annotations
import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture
def builder_mod():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if "scripts.build_yolo_dataset" in sys.modules:
        importlib.reload(sys.modules["scripts.build_yolo_dataset"])
    # The script lives at scripts/build_yolo_dataset.py — import it as a module.
    import scripts.build_yolo_dataset as mod  # noqa: F401
    return sys.modules["scripts.build_yolo_dataset"]


def test_polygon_to_bbox_simple_square(builder_mod):
    """A square polygon at percentages (10, 10) - (30, 30) should give bbox (20%, 20%, 20%, 20%)."""
    poly = [[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]]
    cx, cy, w, h = builder_mod.polygon_to_yolo_bbox(poly)
    assert abs(cx - 0.20) < 1e-6
    assert abs(cy - 0.20) < 1e-6
    assert abs(w - 0.20) < 1e-6
    assert abs(h - 0.20) < 1e-6


def test_polygon_to_bbox_rectangle(builder_mod):
    """A rectangle 20-80 wide, 40-60 tall should give cx=0.5, cy=0.5, w=0.6, h=0.2."""
    poly = [[20.0, 40.0], [80.0, 40.0], [80.0, 60.0], [20.0, 60.0]]
    cx, cy, w, h = builder_mod.polygon_to_yolo_bbox(poly)
    assert abs(cx - 0.50) < 1e-6
    assert abs(w - 0.60) < 1e-6
    assert abs(h - 0.20) < 1e-6


def test_class_id_map_is_complete(builder_mod):
    expected = ["brand_name", "drug_name", "generic_name", "strength", "quantity",
                "company", "manufacturer", "batch_number", "mfg_date", "expiry_date",
                "mrp", "warnings"]
    assert builder_mod.FIELD_NAMES == expected
    for i, name in enumerate(expected):
        assert builder_mod.FIELD_TO_CLASS_ID[name] == i
