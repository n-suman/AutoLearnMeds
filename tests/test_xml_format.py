"""Tests for prepare.py XML format — format_output / parse_output round-trip."""
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


def test_format_output_emits_only_present_fields(prepare_mod) -> None:
    record = {
        "fields": {
            "brand_name": {"text": "Pantocid DSR"},
            "batch_number": {"text": "B.No.:GTF3406A"},
        }
    }
    out = prepare_mod.format_output(record)
    assert "<brand_name>Pantocid DSR</brand_name>" in out
    assert "<batch_number>B.No.:GTF3406A</batch_number>" in out
    # Absent fields not emitted
    assert "<generic_name>" not in out
    assert "<mrp>" not in out
    # Wrapped in BOS/EOS
    assert out.startswith(prepare_mod.BOS_TOKEN)
    assert out.endswith(prepare_mod.EOS_TOKEN)


def test_format_output_canonical_field_order(prepare_mod) -> None:
    """Fields must appear in FIELD_ORDER, regardless of dict insertion order."""
    record = {
        "fields": {
            "warnings": {"text": "X"},
            "brand_name": {"text": "Y"},
            "mrp": {"text": "Z"},
        }
    }
    out = prepare_mod.format_output(record)
    # brand_name precedes mrp precedes warnings in FIELD_ORDER
    pos_brand = out.index("<brand_name>")
    pos_mrp = out.index("<mrp>")
    pos_warn = out.index("<warnings>")
    assert pos_brand < pos_mrp < pos_warn


def test_parse_output_extracts_fields(prepare_mod) -> None:
    text = (
        prepare_mod.BOS_TOKEN
        + "<brand_name>Pantocid DSR</brand_name>"
        + "<mrp>M.R.P.Rs.: 245.00</mrp>"
        + prepare_mod.EOS_TOKEN
    )
    parsed = prepare_mod.parse_output(text)
    assert parsed == {"brand_name": "Pantocid DSR", "mrp": "M.R.P.Rs.: 245.00"}


def test_round_trip(prepare_mod) -> None:
    """format_output -> parse_output recovers the original {field: text}."""
    record = {
        "fields": {
            "brand_name": {"text": "Pantocid DSR"},
            "generic_name": {"text": "Pantoprazole Gastro-Resistant and Domperidone"},
            "batch_number": {"text": "B.No.:GTF3406A"},
        }
    }
    out = prepare_mod.format_output(record)
    parsed = prepare_mod.parse_output(out)
    assert parsed == {
        "brand_name": "Pantocid DSR",
        "generic_name": "Pantoprazole Gastro-Resistant and Domperidone",
        "batch_number": "B.No.:GTF3406A",
    }


def test_parse_output_handles_missing_eos(prepare_mod) -> None:
    """A truncated decode (no </s>) should still extract whatever fields are complete."""
    text = (
        prepare_mod.BOS_TOKEN
        + "<brand_name>Pantocid DSR</brand_name>"
        + "<mrp>M.R.P.Rs.: 245.00</"  # truncated mid-tag
    )
    parsed = prepare_mod.parse_output(text)
    assert parsed.get("brand_name") == "Pantocid DSR"
    # mrp is incomplete — should NOT be present (or present with empty/partial)
    # The exact behavior is up to the implementation, but it shouldn't crash.


def test_parse_output_ignores_unknown_tags(prepare_mod) -> None:
    """Tags not in FIELD_ORDER are ignored."""
    text = (
        prepare_mod.BOS_TOKEN
        + "<brand_name>X</brand_name>"
        + "<not_a_field>Y</not_a_field>"
        + prepare_mod.EOS_TOKEN
    )
    parsed = prepare_mod.parse_output(text)
    assert "brand_name" in parsed
    assert "not_a_field" not in parsed
