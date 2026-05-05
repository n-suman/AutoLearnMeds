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


# --- _levenshtein (edit distance) ---

def test_levenshtein_identical(prepare_mod) -> None:
    assert prepare_mod._levenshtein("hello", "hello") == 0


def test_levenshtein_one_substitution(prepare_mod) -> None:
    assert prepare_mod._levenshtein("kitten", "sitten") == 1


def test_levenshtein_classic_kitten_sitting(prepare_mod) -> None:
    # k → s (sub), e → i (sub), append g (insert) = 3
    assert prepare_mod._levenshtein("kitten", "sitting") == 3


def test_levenshtein_empty_string(prepare_mod) -> None:
    assert prepare_mod._levenshtein("", "abc") == 3
    assert prepare_mod._levenshtein("abc", "") == 3
    assert prepare_mod._levenshtein("", "") == 0


# --- normalized_edit_score ---

def test_edit_score_exact_match_is_one(prepare_mod) -> None:
    assert prepare_mod.normalized_edit_score("Pantocid DSR", "Pantocid DSR") == 1.0


def test_edit_score_normalization_applied(prepare_mod) -> None:
    """Casing + whitespace differences should yield 1.0 after normalize."""
    assert prepare_mod.normalized_edit_score("PANTOCID  DSR", "pantocid dsr") == 1.0


def test_edit_score_partial_credit(prepare_mod) -> None:
    """'B.No.: 123' vs 'B.No.:123' (one space removed) — high but not 1.0."""
    score = prepare_mod.normalized_edit_score("B.No.: 123", "B.No.:123")
    assert 0.85 <= score < 1.0  # one char different out of ~10


def test_edit_score_complete_mismatch(prepare_mod) -> None:
    """Totally different strings of equal length give exactly 0.0."""
    score = prepare_mod.normalized_edit_score("aaaaa", "bbbbb")
    assert score == 0.0


def test_edit_score_handles_none_and_empty(prepare_mod) -> None:
    assert prepare_mod.normalized_edit_score(None, None) == 1.0  # both empty after normalize
    assert prepare_mod.normalized_edit_score("", "") == 1.0
    assert prepare_mod.normalized_edit_score("foo", "") == 0.0
    assert prepare_mod.normalized_edit_score("", "foo") == 0.0


# --- compute_field_edit_f1 ---

def test_field_edit_f1_perfect(prepare_mod) -> None:
    """All-exact predictions → 1.0."""
    preds = [{"brand_name": "Pantocid"}, {"brand_name": "Synthacid"}]
    truths = [{"brand_name": "Pantocid"}, {"brand_name": "Synthacid"}]
    assert prepare_mod.compute_field_edit_f1(preds, truths, "brand_name") == 1.0


def test_field_edit_f1_lenient_vs_strict(prepare_mod) -> None:
    """A near-miss that exact-F1 scores 0 but edit-F1 scores >0.8."""
    preds = [{"brand_name": "B.No.: 123"}]
    truths = [{"brand_name": "B.No.:123"}]
    strict = prepare_mod.compute_field_f1(preds, truths, "brand_name")
    lenient = prepare_mod.compute_field_edit_f1(preds, truths, "brand_name")
    assert strict == 0.0
    assert lenient > 0.8


def test_field_edit_f1_skips_no_truth(prepare_mod) -> None:
    """Records where truth lacks the field don't contribute to the average."""
    preds = [{"brand_name": "X"}, {"brand_name": "Y"}]
    truths = [{"brand_name": "X"}, {}]  # second has no truth
    score = prepare_mod.compute_field_edit_f1(preds, truths, "brand_name")
    # First: exact match → 1.0. Second: skipped. Mean = 1.0.
    assert score == 1.0


def test_field_edit_f1_missing_pred_is_zero(prepare_mod) -> None:
    """Truth has the field but pred doesn't → score 0 for that record."""
    preds = [{}]
    truths = [{"brand_name": "X"}]
    score = prepare_mod.compute_field_edit_f1(preds, truths, "brand_name")
    assert score == 0.0


# --- compute_metrics now returns BOTH metric families ---

def test_compute_metrics_returns_edit_metrics_too(prepare_mod) -> None:
    """compute_metrics should return both strict and lenient macro/per-field."""
    preds = [{"brand_name": "B.No.: 123"}]
    truths = [{"brand_name": "B.No.:123"}]
    m = prepare_mod.compute_metrics(preds, truths)
    assert "macro_f1" in m
    assert "macro_edit_f1" in m
    assert "per_field_f1" in m
    assert "per_field_edit_f1" in m
    # Strict says 0; lenient says ~0.9
    assert m["macro_f1"] == 0.0
    assert m["macro_edit_f1"] > 0.05  # at least one field of 10 contributes ~0.9


def test_compute_metrics_macro_edit_excludes_low_freq_fields(prepare_mod) -> None:
    """macro_edit_f1, like macro_f1, averages only HIGH_FREQUENCY_FIELDS."""
    # Perfect quantity + manufacturer (NOT in HIGH_FREQUENCY_FIELDS); empty for others.
    preds = [{"quantity": "x", "manufacturer": "y"}] * 3
    truths = [{"quantity": "x", "manufacturer": "y"}] * 3
    m = prepare_mod.compute_metrics(preds, truths)
    # macro_edit_f1 averages over the 10 high-freq fields, none of which are present.
    # Per-field edit_f1 for high-freq fields with no truth = 0.0 (empty scores list).
    assert m["macro_edit_f1"] == 0.0
    # But the per-field dict still records quantity's perfect score.
    assert m["per_field_edit_f1"]["quantity"] == 1.0
