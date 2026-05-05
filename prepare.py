"""Phase-1 module: data prep, tokenizer, dataloader, evaluate().

This file is FIXED in the autoresearch loop — the agent must not modify it.
Defines: FIELD_ORDER, HIGH_FREQUENCY_FIELDS, SPECIAL_TOKENS, format_output,
parse_output, train_tokenizer, get_tokenizer, normalize_field_value,
compute_field_f1, compute_metrics, preprocess_image, PharmaLabelDataset,
get_dataloader, evaluate, evaluate_test (gated).

torch / transformers / Pillow / tokenizers are imported lazily inside the
functions that need them, so this module imports cleanly in a torch-less
venv (used for lightweight tests).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# === Constants ===

# Canonical decoder output order — all 12 OCR fields. Order chosen to keep
# semantically related fields adjacent (drug names, then dosing, then origin,
# then dates+price, then warnings). Locked: changing this invalidates trained
# tokenizer + checkpoints.
FIELD_ORDER: list[str] = [
    "brand_name", "drug_name", "generic_name",
    "strength", "quantity",
    "company", "manufacturer",
    "batch_number", "mfg_date", "expiry_date", "mrp",
    "warnings",
]

ALL_FIELDS: frozenset[str] = frozenset(FIELD_ORDER)

# High-frequency fields (≥5% presence in the 837-record golden_set).
# Excludes 'quantity' (0.7%) and 'manufacturer' (0.6%) — too few samples to
# give a reliable per-field F1. The primary macro-F1 averages over these 10.
HIGH_FREQUENCY_FIELDS: frozenset[str] = frozenset([
    "brand_name", "drug_name", "generic_name", "strength",
    "company", "batch_number", "mfg_date", "expiry_date", "mrp", "warnings",
])

BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

# Open and close tags for every field, plus the BOS/EOS sentinels.
SPECIAL_TOKENS: list[str] = (
    [BOS_TOKEN, EOS_TOKEN]
    + [f"<{f}>" for f in FIELD_ORDER]
    + [f"</{f}>" for f in FIELD_ORDER]
)

# SigLIP-base-patch16-224 native resolution. Locked.
IMAGE_SIZE = 224

# Tokenizer vocabulary size (BPE). Locked.
TOKENIZER_VOCAB_SIZE = 8192


# === Output format ===

def format_output(record: dict) -> str:
    """Render a record's `fields` dict as the canonical Donut-style XML output.

    Only present fields are emitted, in FIELD_ORDER. The text inside each tag
    is the raw OCR text (no normalization).
    """
    fields = record.get("fields") or {}
    parts: list[str] = [BOS_TOKEN]
    for fname in FIELD_ORDER:
        if fname not in fields:
            continue
        text = (fields[fname] or {}).get("text")
        if text is None:
            continue
        parts.append(f"<{fname}>{text}</{fname}>")
    parts.append(EOS_TOKEN)
    return "".join(parts)


# Compiled once at import time: matches a single complete <fname>value</fname>
# for any fname in FIELD_ORDER. Capture group 1 = field name, group 2 = value.
_FIELD_TAG_RE = re.compile(
    r"<(" + "|".join(re.escape(f) for f in FIELD_ORDER) + r")>(.*?)</\1>",
    re.DOTALL,
)


def parse_output(text: str) -> dict[str, str]:
    """Inverse of format_output: extract {field_name: text} from XML output.

    Tolerates missing BOS/EOS, ignores unknown tags, and skips truncated tags
    (where the closing </name> is missing).
    """
    return {m.group(1): m.group(2) for m in _FIELD_TAG_RE.finditer(text)}
