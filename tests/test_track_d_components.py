"""Lightweight (non-Colab) unit tests for track_d_pipeline.py pure-logic helpers."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture
def track_d_mod():
    """Import track_d_pipeline.py from the project root, fresh per test.
    Lazy ml imports inside track_d_pipeline.py mean this works in the torch-less
    Mac venv.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if "track_d_pipeline" in sys.modules:
        importlib.reload(sys.modules["track_d_pipeline"])
    import track_d_pipeline
    return track_d_pipeline


def test_track_d_config_loads_yaml(track_d_mod, project_root: Path) -> None:
    cfg = track_d_mod.TrackDConfig.from_yaml(
        project_root / "experiments" / "configs" / "track_d.yaml"
    )
    assert cfg.slice_size == 1024
    assert cfg.overlap == 0.35
    assert cfg.ocr_critical_classes == (
        "batch_number", "mfg_date", "expiry_date", "mrp",
    )


def test_track_d_module_imports_without_torch() -> None:
    """track_d_pipeline.py must import in a torch-less venv (lazy ml imports).

    Confirms torch / ultralytics / sahi / transformers / PIL are NOT in the
    module's top-level namespace.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if "track_d_pipeline" in sys.modules:
        del sys.modules["track_d_pipeline"]
    import track_d_pipeline
    assert not hasattr(track_d_pipeline, "torch")
    assert not hasattr(track_d_pipeline, "ultralytics")
    assert not hasattr(track_d_pipeline, "sahi")
    assert not hasattr(track_d_pipeline, "transformers")
    assert not hasattr(track_d_pipeline, "PIL")


def test_compose_xml_orders_fields(track_d_mod) -> None:
    """_compose_xml emits fields in canonical FIELD_NAMES order, omitting empty/unknown."""
    # Pass dict in random key order, with one unknown field and one empty value.
    field_values = {
        "expiry_date": "12/2026",
        "brand_name": "AcmePharm",
        "unknown_field": "ignored",
        "mrp": "",                # empty - should be omitted
        "batch_number": "B12345",
        "drug_name": "Ibuprofen",
    }
    xml = track_d_mod.TrackDPipeline._compose_xml(field_values)

    # Fields appear in FIELD_NAMES order: brand_name, drug_name, ..., batch_number,
    # ..., expiry_date, ...
    expected = (
        "<s>"
        "<brand_name>AcmePharm</brand_name>"
        "<drug_name>Ibuprofen</drug_name>"
        "<batch_number>B12345</batch_number>"
        "<expiry_date>12/2026</expiry_date>"
        "</s>"
    )
    assert xml == expected
    # Unknown field should NOT appear.
    assert "unknown_field" not in xml
    # Empty mrp value should NOT appear.
    assert "<mrp>" not in xml
