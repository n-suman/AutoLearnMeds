"""Tests for prepare.py normalization + F1 metric."""
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


# --- normalize_field_value ---

def test_normalize_lowercases(prepare_mod) -> None:
    assert prepare_mod.normalize_field_value("Pantocid DSR") == "pantocid dsr"


def test_normalize_strips(prepare_mod) -> None:
    assert prepare_mod.normalize_field_value("  Pantocid  ") == "pantocid"


def test_normalize_collapses_whitespace(prepare_mod) -> None:
    assert prepare_mod.normalize_field_value("Pantocid    DSR\t\nfoo") == "pantocid dsr foo"


def test_normalize_handles_none_and_empty(prepare_mod) -> None:
    assert prepare_mod.normalize_field_value(None) == ""
    assert prepare_mod.normalize_field_value("") == ""


# --- compute_field_f1 ---

def test_field_f1_perfect(prepare_mod) -> None:
    """All matches → F1 = 1.0."""
    preds = [
        {"brand_name": "Pantocid"},
        {"brand_name": "Synthacid"},
    ]
    truths = [
        {"brand_name": "Pantocid"},
        {"brand_name": "Synthacid"},
    ]
    f1 = prepare_mod.compute_field_f1(preds, truths, "brand_name")
    assert f1 == 1.0


def test_field_f1_all_wrong(prepare_mod) -> None:
    """Zero matches → F1 = 0.0."""
    preds = [{"brand_name": "X"}, {"brand_name": "Y"}]
    truths = [{"brand_name": "Pantocid"}, {"brand_name": "Synthacid"}]
    f1 = prepare_mod.compute_field_f1(preds, truths, "brand_name")
    assert f1 == 0.0


def test_field_f1_normalization_applied(prepare_mod) -> None:
    """Casing differences should not penalize."""
    preds = [{"brand_name": "PANTOCID"}]
    truths = [{"brand_name": "pantocid"}]
    f1 = prepare_mod.compute_field_f1(preds, truths, "brand_name")
    assert f1 == 1.0


def test_field_f1_partial_credit(prepare_mod) -> None:
    """4 records, 2 with matching brand_name, 2 with mismatched.
    TP=2, FP=2 (predicted but wrong), FN=2 (truth but wrong predicted).
    Precision=2/4=0.5, Recall=2/4=0.5, F1=0.5.
    """
    preds = [
        {"brand_name": "A"},
        {"brand_name": "X"},  # wrong
        {"brand_name": "B"},
        {"brand_name": "Y"},  # wrong
    ]
    truths = [
        {"brand_name": "A"},
        {"brand_name": "B"},
        {"brand_name": "B"},
        {"brand_name": "C"},
    ]
    f1 = prepare_mod.compute_field_f1(preds, truths, "brand_name")
    assert f1 == pytest.approx(0.5, abs=1e-6)


def test_field_f1_handles_missing_field_in_pred(prepare_mod) -> None:
    """If pred is missing a field but truth has it, that's a false negative."""
    preds = [{}]  # no brand_name predicted
    truths = [{"brand_name": "A"}]
    f1 = prepare_mod.compute_field_f1(preds, truths, "brand_name")
    assert f1 == 0.0


def test_field_f1_handles_missing_field_in_truth(prepare_mod) -> None:
    """If truth doesn't have a field and pred predicts something, that's a FP."""
    preds = [{"brand_name": "A"}]
    truths = [{}]
    f1 = prepare_mod.compute_field_f1(preds, truths, "brand_name")
    assert f1 == 0.0


def test_field_f1_both_missing_is_zero_not_one(prepare_mod) -> None:
    """If neither has the field, no TP/FP/FN — F1 is undefined; return 0.0 by convention."""
    preds = [{}]
    truths = [{}]
    f1 = prepare_mod.compute_field_f1(preds, truths, "brand_name")
    assert f1 == 0.0


# --- compute_metrics ---

def test_compute_metrics_returns_macro_and_per_field(prepare_mod) -> None:
    preds = [
        {"brand_name": "A", "mrp": "99"},
        {"brand_name": "B"},
    ]
    truths = [
        {"brand_name": "A", "mrp": "99"},
        {"brand_name": "B", "warnings": "X"},
    ]
    m = prepare_mod.compute_metrics(preds, truths)
    assert "macro_f1" in m
    assert "per_field_f1" in m
    assert isinstance(m["per_field_f1"], dict)
    # brand_name perfect (2/2) -> 1.0
    assert m["per_field_f1"]["brand_name"] == pytest.approx(1.0)


def test_compute_metrics_macro_averages_high_frequency_only(prepare_mod) -> None:
    """compute_metrics's macro_f1 averages only HIGH_FREQUENCY_FIELDS (10 fields)."""
    # Make every high-freq field perfect (1.0). manufacturer/quantity should NOT be averaged in.
    preds = [{f: "x" for f in prepare_mod.HIGH_FREQUENCY_FIELDS}] * 5
    truths = [{f: "x" for f in prepare_mod.HIGH_FREQUENCY_FIELDS}] * 5
    m = prepare_mod.compute_metrics(preds, truths)
    assert m["macro_f1"] == pytest.approx(1.0)


def test_compute_metrics_macro_excludes_quantity_and_manufacturer(prepare_mod) -> None:
    """Even if quantity/manufacturer are perfect, they don't influence macro_f1
    (they aren't in HIGH_FREQUENCY_FIELDS)."""
    # All high-freq are wrong (F1=0), but quantity is perfect.
    preds = [{"quantity": "x"}] * 3
    truths = [{"quantity": "x"}] * 3
    m = prepare_mod.compute_metrics(preds, truths)
    assert m["macro_f1"] == 0.0
    assert m["per_field_f1"].get("quantity") == 1.0
