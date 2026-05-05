"""Tests for prepare.py tokenizer — BPE-8192 train + load."""
from __future__ import annotations

import importlib
import json
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


@pytest.fixture
def tiny_field_corpus() -> list[str]:
    """A tiny but realistic corpus of field texts to train BPE on."""
    return [
        "Pantocid DSR",
        "Pantoprazole Gastro-Resistant and Domperidone",
        "Synthacid",
        "Paracetamol",
        "Synthbol",
        "M.R.P.Rs.: 99.50",
        "B.No.:GTF3406A",
        "Exp. Dt.: 09/2026",
        "Mfg. Dt.: 11/2024",
        "NEON LABORATORIES LIMITED",
        "ABC LABS PVT LTD",
        "SCHEDULE H PRESCRIPTION DRUG-CAUTION",
        "FOR EXTERNAL USE ONLY",
        "500MG / 2 ML",
        "10MG / 5 ML",
    ] * 10  # repeat for enough merges


def test_train_tokenizer_writes_file(
    prepare_mod, tiny_field_corpus: list[str], tmp_path: Path
) -> None:
    out = tmp_path / "tokenizer.json"
    prepare_mod.train_tokenizer(
        corpus=tiny_field_corpus,
        out_path=out,
        vocab_size=512,  # small for fast testing
    )
    assert out.is_file()
    # tokenizer.json should be valid JSON with a vocab and merges.
    data = json.loads(out.read_text())
    assert "model" in data
    assert "added_tokens" in data


def test_get_tokenizer_loads_from_file(
    prepare_mod, tiny_field_corpus: list[str], tmp_path: Path
) -> None:
    out = tmp_path / "tokenizer.json"
    prepare_mod.train_tokenizer(corpus=tiny_field_corpus, out_path=out, vocab_size=512)
    tok = prepare_mod.get_tokenizer(out)
    # Encode a known string; decode should round-trip (modulo whitespace BPE adds).
    enc = tok.encode("Pantocid")
    assert len(enc.ids) > 0


def test_special_tokens_are_in_vocab(
    prepare_mod, tiny_field_corpus: list[str], tmp_path: Path
) -> None:
    """All SPECIAL_TOKENS must be added as added_tokens during training."""
    out = tmp_path / "tokenizer.json"
    prepare_mod.train_tokenizer(corpus=tiny_field_corpus, out_path=out, vocab_size=512)
    tok = prepare_mod.get_tokenizer(out)
    for st in prepare_mod.SPECIAL_TOKENS:
        token_id = tok.token_to_id(st)
        assert token_id is not None, f"Special token {st!r} missing from tokenizer"


def test_tokenizer_round_trip_preserves_special_tokens(
    prepare_mod, tiny_field_corpus: list[str], tmp_path: Path
) -> None:
    """An XML output sequence encodes and decodes losslessly (no special-token mangling)."""
    out = tmp_path / "tokenizer.json"
    prepare_mod.train_tokenizer(corpus=tiny_field_corpus, out_path=out, vocab_size=512)
    tok = prepare_mod.get_tokenizer(out)
    seq = (
        prepare_mod.BOS_TOKEN
        + "<brand_name>Pantocid</brand_name>"
        + prepare_mod.EOS_TOKEN
    )
    enc = tok.encode(seq)
    decoded = tok.decode(enc.ids, skip_special_tokens=False)
    # The decoded sequence should still contain all special tokens we put in.
    assert "<s>" in decoded
    assert "</s>" in decoded
    assert "<brand_name>" in decoded
    assert "</brand_name>" in decoded
    assert "Pantocid" in decoded
