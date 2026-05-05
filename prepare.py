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


# === Tokenizer ===

def train_tokenizer(
    corpus: list[str],
    out_path: Path | str,
    vocab_size: int = TOKENIZER_VOCAB_SIZE,
) -> None:
    """Train a BPE tokenizer on `corpus` and save to `out_path` (tokenizer.json).

    All SPECIAL_TOKENS are registered as added_tokens (single-id, not splittable
    by BPE), so they survive encode/decode unchanged.

    `corpus` is a list of strings — typically the field text values from
    train.jsonl (NOT the XML-formatted output, since SPECIAL_TOKENS handle the
    structural part).
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<pad>", *SPECIAL_TOKENS],
        show_progress=False,
    )
    tok.train_from_iterator(corpus, trainer=trainer)
    tok.save(str(out_path))


def get_tokenizer(path: Path | str):
    """Load a saved tokenizer.json from `path`. Returns a tokenizers.Tokenizer."""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(Path(path)))


# === Normalization + F1 ===

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_field_value(s: str | None) -> str:
    """Light normalization: lowercase + strip + collapse internal whitespace.

    Phase-1 deliberately does NOT strip prefixes like "B.No.:" or parse dates;
    those are Phase-7 sweep candidates.
    """
    if not s:
        return ""
    return _WHITESPACE_RE.sub(" ", s.strip()).lower()


def compute_field_f1(
    predictions: list[dict[str, str]],
    truths: list[dict[str, str]],
    field: str,
) -> float:
    """F1 for a single field across N records.

    A predicted text for `field` matches truth iff their normalized values
    are equal AND truth contains the field. Records where neither has the
    field contribute nothing (no TP/FP/FN).

    Returns 0.0 when precision+recall == 0.
    """
    if len(predictions) != len(truths):
        raise ValueError(f"len mismatch: {len(predictions)} vs {len(truths)}")

    tp = fp = fn = 0
    for pred, truth in zip(predictions, truths):
        p = pred.get(field)
        t = truth.get(field)
        if p is None and t is None:
            continue
        if p is not None and t is None:
            fp += 1
            continue
        if p is None and t is not None:
            fn += 1
            continue
        # Both present.
        if normalize_field_value(p) == normalize_field_value(t):
            tp += 1
        else:
            fp += 1
            fn += 1

    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_metrics(
    predictions: list[dict[str, str]],
    truths: list[dict[str, str]],
) -> dict[str, Any]:
    """Compute macro_f1 (over HIGH_FREQUENCY_FIELDS) + per-field F1 (over all 12).

    Returns:
        {
            "macro_f1": float in [0, 1],
            "per_field_f1": {field: float, ...} for ALL 12 fields,
            "n_examples": int,
        }
    """
    per_field = {f: compute_field_f1(predictions, truths, f) for f in FIELD_ORDER}
    high_freq = [v for f, v in per_field.items() if f in HIGH_FREQUENCY_FIELDS]
    macro = sum(high_freq) / len(high_freq) if high_freq else 0.0
    return {
        "macro_f1": macro,
        "per_field_f1": per_field,
        "n_examples": len(predictions),
    }


# === Image preprocessing ===

# Cached after first call so we don't redownload SigLIP processor each batch.
_PROCESSOR_CACHE: dict[str, Any] = {}


def _get_siglip_processor():
    """Lazy-load and cache the SigLIP IMAGE processor only.

    Use AutoImageProcessor (not AutoProcessor) so we don't drag in the SigLIP
    tokenizer — that requires SentencePiece, and we have our own BPE for the
    decoder anyway. Only the visual transform (resize/normalize) is needed.
    """
    if "processor" not in _PROCESSOR_CACHE:
        from transformers import AutoImageProcessor

        _PROCESSOR_CACHE["processor"] = AutoImageProcessor.from_pretrained(
            "google/siglip-base-patch16-224"
        )
    return _PROCESSOR_CACHE["processor"]


def preprocess_image(image_path: Path | str):
    """Resize + letterbox + normalize an image. Returns a (3, IMAGE_SIZE, IMAGE_SIZE) torch tensor.

    Letterboxing preserves aspect ratio with mid-gray padding so non-square
    medicine boxes don't get squashed.
    """
    from PIL import Image

    img = Image.open(str(image_path)).convert("RGB")
    w, h = img.size
    scale = IMAGE_SIZE / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(127, 127, 127))
    canvas.paste(img, ((IMAGE_SIZE - nw) // 2, (IMAGE_SIZE - nh) // 2))

    proc = _get_siglip_processor()
    out = proc(images=canvas, return_tensors="pt")
    return out["pixel_values"].squeeze(0)  # (3, H, W)


# === Dataset + DataLoader ===

class PharmaLabelDataset:
    """Iterates a processed JSONL file. Yields {image, target_text, image_id} dicts.

    `image` is the preprocessed tensor.
    `target_text` is format_output(record) — the canonical XML decoder target.
    `image_id` is the record's image_id (for evaluation cross-reference).
    """

    def __init__(
        self,
        jsonl_path: Path | str,
        images_root: Path | str,
        path_strip_prefix: str = "raw/raw_images/",
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.images_root = Path(images_root)
        self.path_strip_prefix = path_strip_prefix
        self.records: list[dict] = [
            json.loads(line)
            for line in self.jsonl_path.read_text().splitlines()
            if line.strip()
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        rel = rec["image_path"]
        if rel.startswith(self.path_strip_prefix):
            rel = rel[len(self.path_strip_prefix):]
        img_path = self.images_root / rel
        return {
            "image": preprocess_image(img_path),
            "target_text": format_output(rec),
            "image_id": rec["image_id"],
            "fields_truth": {f: (rec.get("fields") or {}).get(f, {}).get("text")
                              for f in FIELD_ORDER
                              if (rec.get("fields") or {}).get(f, {}).get("text") is not None},
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Default-style collate that stacks tensors and lists everything else."""
    import torch  # type: ignore[import-not-found]

    out: dict[str, Any] = {}
    out["image"] = torch.stack([b["image"] for b in batch])
    out["target_text"] = [b["target_text"] for b in batch]
    out["image_id"] = [b["image_id"] for b in batch]
    out["fields_truth"] = [b["fields_truth"] for b in batch]
    return out


def get_dataloader(
    jsonl_path: Path | str,
    images_root: Path | str,
    path_strip_prefix: str = "raw/raw_images/",
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
):
    """PyTorch DataLoader over PharmaLabelDataset."""
    from torch.utils.data import DataLoader  # type: ignore[import-not-found]

    ds = PharmaLabelDataset(
        jsonl_path=jsonl_path,
        images_root=images_root,
        path_strip_prefix=path_strip_prefix,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate,
    )


# === Public eval API ===

def evaluate(
    model,
    jsonl_path: Path | str,
    images_root: Path | str,
    path_strip_prefix: str = "raw/raw_images/",
    batch_size: int = 16,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    """Run `model.predict_text(batch_images, max_new_tokens) -> list[str]`
    over the given JSONL split, parse the predicted XML, compute
    {macro_f1, per_field_f1, n_examples}.

    `model` must implement `predict_text(batch_images, max_new_tokens)` returning
    a list of strings (one per image). The model itself is opaque — train.py
    decides how to wrap its decoder behind this method.
    """
    dl = get_dataloader(
        jsonl_path=jsonl_path,
        images_root=images_root,
        path_strip_prefix=path_strip_prefix,
        batch_size=batch_size,
        shuffle=False,
    )
    predictions: list[dict[str, str]] = []
    truths: list[dict[str, str]] = []
    for batch in dl:
        outputs = model.predict_text(batch["image"], max_new_tokens=max_new_tokens)
        for out_text, fields_truth in zip(outputs, batch["fields_truth"]):
            predictions.append(parse_output(out_text))
            truths.append(fields_truth)
    return compute_metrics(predictions, truths)


_TEST_FIREWALL_CONSENT = "i_understand_this_consumes_the_test_set"


def evaluate_test(
    model,
    jsonl_path: Path | str,
    images_root: Path | str,
    path_strip_prefix: str = "raw/raw_images/",
    batch_size: int = 16,
    max_new_tokens: int = 256,
    explicit_consent: str = "",
    audit_log: Path | str | None = None,
) -> dict[str, Any]:
    """Gated wrapper around evaluate() for the held-out TEST split.

    Refuses to run unless `explicit_consent == "i_understand_this_consumes_the_test_set"`.
    Every successful call appends a timestamped line to `audit_log`
    (default: ./experiments/evaluate_test_audit.log).
    """
    import datetime as _dt

    if explicit_consent != _TEST_FIREWALL_CONSENT:
        raise RuntimeError(
            "evaluate_test requires explicit_consent="
            f"{_TEST_FIREWALL_CONSENT!r}. Test-set evals must be rare and intentional."
        )

    audit_path = Path(audit_log) if audit_log else Path("experiments/evaluate_test_audit.log")
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = evaluate(
        model=model,
        jsonl_path=jsonl_path,
        images_root=images_root,
        path_strip_prefix=path_strip_prefix,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    line = (
        f"{_dt.datetime.now(_dt.UTC).isoformat()} evaluate_test invoked "
        f"jsonl={jsonl_path} macro_f1={metrics['macro_f1']:.4f} "
        f"n={metrics['n_examples']}\n"
    )
    with audit_path.open("a") as fh:
        fh.write(line)
    return metrics
