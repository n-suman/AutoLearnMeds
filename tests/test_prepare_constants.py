"""Tests for prepare.py constants — field universe + special tokens."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def prepare_mod():
    """Import prepare from the project root, fresh per test."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if "prepare" in sys.modules:
        importlib.reload(sys.modules["prepare"])
    import prepare
    return prepare


def test_field_order_is_canonical_12(prepare_mod) -> None:
    """FIELD_ORDER lists exactly 12 OCR fields in canonical decoder order."""
    expected = [
        "brand_name", "drug_name", "generic_name",
        "strength", "quantity",
        "company", "manufacturer",
        "batch_number", "mfg_date", "expiry_date", "mrp",
        "warnings",
    ]
    assert prepare_mod.FIELD_ORDER == expected
    assert len(prepare_mod.FIELD_ORDER) == 12


def test_high_frequency_fields_is_10(prepare_mod) -> None:
    """HIGH_FREQUENCY_FIELDS includes only fields with ≥5% presence (everything except quantity, manufacturer)."""
    high = prepare_mod.HIGH_FREQUENCY_FIELDS
    assert len(high) == 10
    assert "quantity" not in high
    assert "manufacturer" not in high
    # The 10 retained fields:
    for f in [
        "brand_name", "drug_name", "generic_name", "strength",
        "company", "batch_number", "mfg_date", "expiry_date", "mrp", "warnings",
    ]:
        assert f in high


def test_special_tokens_cover_all_fields(prepare_mod) -> None:
    """For every field there's an open and close special token."""
    tokens = prepare_mod.SPECIAL_TOKENS
    assert prepare_mod.BOS_TOKEN in tokens
    assert prepare_mod.EOS_TOKEN in tokens
    for field in prepare_mod.FIELD_ORDER:
        assert f"<{field}>" in tokens
        assert f"</{field}>" in tokens


def test_image_size(prepare_mod) -> None:
    """SigLIP-base native resolution."""
    assert prepare_mod.IMAGE_SIZE == 224


def test_module_imports_without_torch(prepare_mod) -> None:
    """prepare.py must be importable in a torch-less venv (lazy imports)."""
    # If we got here, the import worked. Belt-and-suspenders: confirm torch
    # isn't in the module's top-level namespace.
    assert not hasattr(prepare_mod, "torch")
