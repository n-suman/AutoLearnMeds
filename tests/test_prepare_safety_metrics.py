"""Tests for safety-4 subset metric + SAFETY_FIELDS constant."""
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


def test_safety_fields_constant_exists(prepare_mod) -> None:
    assert prepare_mod.SAFETY_FIELDS == frozenset(["batch_number", "mfg_date", "expiry_date", "mrp"])


def test_compute_subset_metrics_macro_strict(prepare_mod) -> None:
    """Macro F1 over a subset of fields is the mean of per-field strict F1 for those fields."""
    preds = [{"batch_number": "B1", "mfg_date": "11/2023", "mrp": "85.00", "expiry_date": "10/2025", "brand_name": "Wrong"}]
    truths = [{"batch_number": "B1", "mfg_date": "11/2023", "mrp": "85.00", "expiry_date": "10/2025", "brand_name": "Right"}]
    result = prepare_mod.compute_subset_metrics(preds, truths, fields=prepare_mod.SAFETY_FIELDS)
    assert result["macro_f1"] == pytest.approx(1.0)  # all 4 safety fields exact match
    assert result["macro_edit_f1"] == pytest.approx(1.0)
    assert set(result["per_field_f1"].keys()) == prepare_mod.SAFETY_FIELDS
    assert "brand_name" not in result["per_field_f1"]  # not in subset


def test_compute_subset_metrics_partial(prepare_mod) -> None:
    """When 2 of 4 safety fields match strictly, macro_f1 should be 0.5.

    The two mismatched fields use close-but-not-exact values so edit distance
    awards partial credit, lifting macro_edit_f1 above the strict macro_f1.
    """
    preds = [{"batch_number": "B1", "mfg_date": "11/2023", "mrp": "85.5", "expiry_date": "10/202"}]
    truths = [{"batch_number": "B1", "mfg_date": "11/2023", "mrp": "85.00", "expiry_date": "10/2025"}]
    result = prepare_mod.compute_subset_metrics(preds, truths, fields=prepare_mod.SAFETY_FIELDS)
    assert result["macro_f1"] == pytest.approx(0.5)
    assert result["macro_edit_f1"] > result["macro_f1"]   # partial credit lifts above strict
    assert result["macro_edit_f1"] < 1.0                  # but doesn't reach perfect


def test_compute_subset_metrics_returns_n_examples(prepare_mod) -> None:
    """The return dict must include n_examples (contract with compute_metrics)."""
    preds = [{"batch_number": "X"}, {"batch_number": "Y"}, {"batch_number": "Z"}]
    truths = [{"batch_number": "X"}, {"batch_number": "Y"}, {"batch_number": "Z"}]
    result = prepare_mod.compute_subset_metrics(preds, truths, fields=prepare_mod.SAFETY_FIELDS)
    assert result["n_examples"] == 3


def test_compute_subset_metrics_rejects_unknown_field(prepare_mod) -> None:
    """Unknown field in subset raises ValueError (not silent assert)."""
    preds = [{}]
    truths = [{}]
    with pytest.raises(ValueError, match="unknown fields"):
        prepare_mod.compute_subset_metrics(preds, truths, fields=frozenset(["not_a_real_field"]))


def test_compute_safety4_metrics_convenience(prepare_mod) -> None:
    """Thin wrapper that calls compute_subset_metrics with SAFETY_FIELDS."""
    preds = [{"batch_number": "X", "mfg_date": "Y", "mrp": "Z", "expiry_date": "W"}]
    truths = [{"batch_number": "X", "mfg_date": "Y", "mrp": "Z", "expiry_date": "W"}]
    result = prepare_mod.compute_safety4_metrics(preds, truths)
    assert result["macro_f1"] == pytest.approx(1.0)
