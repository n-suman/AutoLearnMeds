"""Tests for pseudo-label normalization + filtering (Track E)."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def prepare_mod():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if "prepare" in sys.modules:
        importlib.reload(sys.modules["prepare"])
    import prepare
    return prepare


def test_normalize_strips_inr_currency_prefix(prepare_mod) -> None:
    row = {"image_path": "x.jpg", "xml_label": "<medication><mrp>₹85.00</mrp></medication>"}
    out = prepare_mod.normalize_pseudo_xml(row)
    assert "₹" not in out["xml_label"]
    assert "85.00" in out["xml_label"]


def test_normalize_strips_rs_prefix(prepare_mod) -> None:
    row = {"image_path": "x.jpg", "xml_label": "<medication><mrp>Rs. 85.00</mrp></medication>"}
    out = prepare_mod.normalize_pseudo_xml(row)
    assert "Rs." not in out["xml_label"]
    assert "85.00" in out["xml_label"]


def test_normalize_date_separators(prepare_mod) -> None:
    row = {"image_path": "x.jpg", "xml_label": "<medication><expiry_date>10-2025</expiry_date></medication>"}
    out = prepare_mod.normalize_pseudo_xml(row)
    assert "10/2025" in out["xml_label"]


def test_normalize_collapses_whitespace(prepare_mod) -> None:
    row = {"image_path": "x.jpg", "xml_label": "<medication><brand_name>  Crocin   Advance  </brand_name></medication>"}
    out = prepare_mod.normalize_pseudo_xml(row)
    assert "Crocin Advance" in out["xml_label"]


def test_normalize_preserves_other_fields(prepare_mod) -> None:
    row = {"image_path": "x.jpg", "xml_label": "<medication><batch_number>BC4521A</batch_number></medication>", "extra": "keep"}
    out = prepare_mod.normalize_pseudo_xml(row)
    assert out["image_path"] == "x.jpg"
    assert out["extra"] == "keep"
    assert "BC4521A" in out["xml_label"]


def test_filter_drops_val_leak(prepare_mod) -> None:
    rows = [
        {"image_path": "raw_images/IMG_001.JPG", "xml_label": "<medication><brand_name>X</brand_name></medication>"},
        {"image_path": "raw_images/IMG_002.JPG", "xml_label": "<medication><brand_name>Y</brand_name></medication>"},
    ]
    val_paths = {"raw_images/IMG_001.JPG"}
    test_paths: set[str] = set()
    kept, dropped = prepare_mod.filter_pseudo_rows(rows, val_paths=val_paths, test_paths=test_paths)
    assert len(kept) == 1
    assert kept[0]["image_path"] == "raw_images/IMG_002.JPG"
    assert dropped["val_leak"] == 1


def test_filter_drops_all_empty(prepare_mod) -> None:
    """Rows with >=8 of 12 empty fields look like non-pharma images and are dropped."""
    # All 12 fields empty
    xml_empty = "<medication>" + "".join(f"<{f}></{f}>" for f in prepare_mod.FIELD_ORDER) + "</medication>"
    rows = [
        {"image_path": "raw_images/IMG_003.JPG", "xml_label": xml_empty},
        {"image_path": "raw_images/IMG_004.JPG", "xml_label": "<medication><brand_name>Real</brand_name><drug_name>Real</drug_name><generic_name>R</generic_name><strength>5mg</strength><quantity>10</quantity><company>X</company><manufacturer>X</manufacturer><batch_number>B</batch_number><mfg_date>11/2023</mfg_date><expiry_date>10/2025</expiry_date><mrp>85</mrp><warnings>w</warnings></medication>"},
    ]
    kept, dropped = prepare_mod.filter_pseudo_rows(rows, val_paths=set(), test_paths=set(), max_empty_fields=8)
    assert len(kept) == 1
    assert kept[0]["image_path"] == "raw_images/IMG_004.JPG"
    assert dropped["all_empty"] == 1


def test_filter_drops_parse_failures(prepare_mod) -> None:
    rows = [
        {"image_path": "raw_images/IMG_005.JPG", "xml_label": "not valid xml at all"},
        {"image_path": "raw_images/IMG_006.JPG", "xml_label": "<medication><brand_name>X</brand_name></medication>"},
    ]
    kept, dropped = prepare_mod.filter_pseudo_rows(rows, val_paths=set(), test_paths=set())
    assert len(kept) == 1
    assert kept[0]["image_path"] == "raw_images/IMG_006.JPG"
    assert dropped["parse_fail"] == 1


def test_filter_returns_dropped_counts(prepare_mod) -> None:
    kept, dropped = prepare_mod.filter_pseudo_rows([], val_paths=set(), test_paths=set())
    assert kept == []
    assert dropped == {"val_leak": 0, "test_leak": 0, "all_empty": 0, "parse_fail": 0}
