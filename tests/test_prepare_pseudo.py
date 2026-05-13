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
