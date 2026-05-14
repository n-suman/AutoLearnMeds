"""Inference dump + post-processing normalization analysis for Track E.

Two phases:

Phase A (requires GPU): load a Track E checkpoint, run inference on val,
write per-image predictions to a JSONL file. Each row contains:
    {image_id, image_path, gold_xml, pred_xml, gold_fields, pred_fields_raw}

Phase B (CPU only): load that JSONL, apply different normalization rule sets,
recompute macro/per-field F1 for each rule set, write a comparison JSON.

The point: the baseline `prepare.normalize_field_value` does lowercase + strip +
whitespace-collapse. Strict F1 ~ 0 on safety-4 fields with non-zero lenient
F1 means content is near-correct but format-mismatched in other ways
(currency prefixes, date separators, decimal style, etc). Field-specific
normalization at inference time may lift strict F1 without retraining.

Usage:
    # Phase A (run once on Colab):
    python scripts/score_with_normalization.py \\
        --mode dump \\
        --config experiments/configs/track_e_highres_gold_only.yaml \\
        --ckpt checkpoints/runs/track_e_highres_gold_only-seed44/best.pt \\
        --predictions-out experiments/track_e_v1_predictions.jsonl

    # Phase B (iterate locally with different rule sets):
    python scripts/score_with_normalization.py \\
        --mode score \\
        --predictions-in experiments/track_e_v1_predictions.jsonl \\
        --rules-out experiments/track_e_v1_norm_analysis.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


# ============================================================================
# Phase A: dump predictions
# ============================================================================

def dump_predictions(args: argparse.Namespace) -> int:
    """Load checkpoint, run inference on val, write predictions JSONL."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import yaml
    import torch

    from train import Config, PharmaVLM, self_named_params
    import prepare

    cfg_raw = yaml.safe_load(open(args.config).read())
    cfg = Config(**cfg_raw)
    print(f"config: encoder={cfg.encoder_model} image_size={cfg.image_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = prepare.get_tokenizer(cfg.tokenizer_path)

    model = PharmaVLM(
        encoder_model=cfg.encoder_model,
        encoder_init_path=cfg.encoder_init_path,
        vocab_size=tokenizer.get_vocab_size(),
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_decoder_layers,
        n_heads=cfg.n_decoder_heads,
        ffn_ratio=cfg.ffn_ratio,
        dropout=cfg.dropout,
        tied_embeddings=cfg.tied_embeddings,
        tokenizer=tokenizer,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    print(f"ckpt step={ckpt['step']} macro_f1={ckpt['macro_f1']:.4f}")
    model.proj.load_state_dict(ckpt["proj_state_dict"])
    model.decoder.embedding.load_state_dict(ckpt["decoder_embedding_state_dict"])
    model.decoder.final_ln.load_state_dict(ckpt["decoder_final_ln_state_dict"])
    for i, block_sd in enumerate(ckpt["decoder_blocks_state_dicts"]):
        blk = model.decoder.blocks[i]
        name_to_param = dict(self_named_params(blk))
        for name, tensor in block_sd.items():
            if name in name_to_param:
                name_to_param[name].data.copy_(tensor.to(device))
    model.train(False)

    dl = prepare.get_dataloader(
        jsonl_path=cfg.val_jsonl,
        images_root=cfg.images_root,
        path_strip_prefix=cfg.path_strip_prefix,
        batch_size=cfg.batch_size,
        shuffle=False,
        image_size=cfg.image_size if cfg.image_size != 224 else None,
    )

    out_path = Path(args.predictions_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with torch.no_grad(), open(out_path, "w") as f:
        for batch in dl:
            outputs = model.predict_text(batch["image"], max_new_tokens=cfg.max_target_length)
            for image_id, pred_xml, fields_truth in zip(
                batch["image_id"], outputs, batch["fields_truth"]
            ):
                pred_fields_raw = prepare.parse_output(pred_xml)
                f.write(json.dumps({
                    "image_id": image_id,
                    "pred_xml": pred_xml,
                    "pred_fields_raw": pred_fields_raw,
                    "gold_fields": fields_truth,
                }) + "\n")
                n_written += 1
    print(f"wrote {n_written} predictions → {out_path}")
    return 0


# ============================================================================
# Phase B: normalization rules + scoring
# ============================================================================

# All normalizers operate on field VALUE strings and return new strings.
# The baseline (lowercase + strip + ws-collapse) is applied via
# prepare.normalize_field_value upstream; these rules go BEYOND that.


_INR_PREFIX_RE = re.compile(r"^(rs\.?|₹)\s*[.:]?\s*", re.IGNORECASE)
_DATE_DASH_RE = re.compile(r"^(\d{1,2})\s*[-./\s]\s*(\d{2,4})$")
_DATE_MONTH_RE = re.compile(
    r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[.\s]*(\d{2,4})$",
    re.IGNORECASE,
)
# Field-label prefixes that appear in MANY gold values for the dataset.
# Gold examples we found in val.jsonl:
#   "B. No. KP", "Batch No. : SHAP-021", "B.No:KP1307495"
#   "M.D : MAY 2024", "Mfg.Date :04/2025", "MFG.DATE 11/2024"
#   "E.D : APR 2026", "Exp.Date:09/2026", "Expiry Date DEC.2027"
#   "M.R.P ₹ : 26.70", "Maximum retail price RS.243.07", "M.R.P.₹.16.12"
# These prefixes appear before the actual value. Strip them on both
# prediction AND gold so the comparison is content-only.
_BATCH_PREFIX_RE = re.compile(
    r"^(b\s*\.?\s*no\.?\s*:?\s*|batch\s*no\.?\s*:?\s*|batch\s*number\.?\s*:?\s*|lot\s*no\.?\s*:?\s*)",
    re.IGNORECASE,
)
_DATE_PREFIX_RE = re.compile(
    r"^(m\s*\.?\s*d\s*\.?\s*:?\s*|"
    r"mfg\s*\.?\s*date\s*\.?\s*:?\s*|"
    r"mfd\s*\.?\s*date\s*\.?\s*:?\s*|"
    r"manufacturing\s*date\s*:?\s*|"
    r"e\s*\.?\s*d\s*\.?\s*:?\s*|"
    r"exp\s*\.?\s*date\s*\.?\s*:?\s*|"
    r"expiry\s*date\s*\.?\s*:?\s*|"
    r"use\s*by\s*:?\s*)",
    re.IGNORECASE,
)
_MRP_PREFIX_RE = re.compile(
    r"^(m\s*\.?\s*r\s*\.?\s*p\s*\.?\s*[₹rs.]*\s*[:.]?\s*|"
    r"maximum\s+retail\s+price\s*[₹rs.]*\s*[:.]?\s*|"
    r"mrp\s*[₹rs.]*\s*[:.]?\s*)",
    re.IGNORECASE,
)
_MONTH_TO_NUM = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


def norm_strip_all_ws(s: str) -> str:
    """Strip ALL whitespace (not just collapse)."""
    return re.sub(r"\s+", "", s)


def norm_mrp(s: str) -> str:
    """MRP: strip field-label + Rs./₹ prefixes, force numeric integer.

    "M.R.P ₹ : 26.70" → "26.70"
    "Maximum retail price RS.243.07" → "243.07"
    "M.R.P.RS. 113.73" → "113.73"
    "85.00" → "85"
    "85.50" → "85.50"
    """
    s = _MRP_PREFIX_RE.sub("", s).strip()
    s = _INR_PREFIX_RE.sub("", s).strip()
    s = re.sub(r"[₹rs.\s]+$", "", s, flags=re.IGNORECASE).strip()
    try:
        v = float(s)
        if v == int(v):
            return str(int(v))
        return str(v)
    except (ValueError, TypeError):
        return s


def norm_date(s: str) -> str:
    """Date: strip field-label prefix, then normalize to MM/YYYY.

    "M.D : MAY 2024" → "05/2024"
    "Mfg.Date :04/2025" → "04/2025"
    "MFG.DATE 11/2024" → "11/2024"
    "Exp.Date:09/2026" → "09/2026"
    "APR 2026" → "04/2026"
    """
    s = _DATE_PREFIX_RE.sub("", s).strip()
    m = _DATE_DASH_RE.match(s)
    if m:
        mm = m.group(1).zfill(2)
        yy = m.group(2)
        if len(yy) == 2:
            yy = "20" + yy
        return f"{mm}/{yy}"
    m = _DATE_MONTH_RE.match(s)
    if m:
        mm = _MONTH_TO_NUM[m.group(1).lower()]
        yy = m.group(2)
        if len(yy) == 2:
            yy = "20" + yy
        return f"{mm}/{yy}"
    return s


def norm_batch_number(s: str) -> str:
    """Batch number: strip field-label prefix + all whitespace."""
    s = _BATCH_PREFIX_RE.sub("", s).strip()
    return re.sub(r"\s+", "", s)


# Rule sets to try. Each is a dict mapping field name → normalizer function.
# Special "_" key applies to ALL fields.
RULE_SETS = {
    "baseline": {},  # No additional normalization beyond prepare.normalize_field_value
    "strip_all_ws_global": {"_": norm_strip_all_ws},
    "currency_dates": {
        "mrp": norm_mrp,
        "expiry_date": norm_date,
        "mfg_date": norm_date,
    },
    "currency_dates_batch": {
        "mrp": norm_mrp,
        "expiry_date": norm_date,
        "mfg_date": norm_date,
        "batch_number": norm_batch_number,
    },
    "all_safety4": {
        "mrp": norm_mrp,
        "expiry_date": norm_date,
        "mfg_date": norm_date,
        "batch_number": norm_batch_number,
        "_": norm_strip_all_ws,
    },
}


def apply_rules(value: str, field: str, rules: dict) -> str:
    """Apply field-specific then global normalization."""
    if field in rules:
        value = rules[field](value)
    if "_" in rules:
        value = rules["_"](value)
    return value


def score_predictions(predictions_path: Path, rules_out: Path) -> int:
    """Phase B: load predictions JSONL, score under each rule set, write analysis."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import prepare

    rows = []
    with open(predictions_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"loaded {len(rows)} predictions")

    SAFETY = {"batch_number", "mfg_date", "expiry_date", "mrp"}
    results = {}

    for rule_name, rules in RULE_SETS.items():
        # Build predicted + truth field dicts under this rule set.
        # IMPORTANT: gold values also contain field-label prefixes (e.g. "M.R.P ₹ : 26.70"),
        # so the rules must be applied symmetrically to BOTH pred and gold or the
        # comparison is unfair.
        preds, truths = [], []
        for r in rows:
            pred_norm = {
                f: apply_rules(v, f, rules)
                for f, v in r.get("pred_fields_raw", {}).items()
            }
            preds.append(pred_norm)
            # gold_fields may be either {f: "value"} or {f: {"text": "value"}}.
            # Flatten to {f: "value"} first, then normalize symmetrically.
            gold_raw = r.get("gold_fields", {})
            gold_flat = {}
            for f, v in gold_raw.items():
                if isinstance(v, dict) and "text" in v:
                    gold_flat[f] = v["text"]
                elif isinstance(v, str):
                    gold_flat[f] = v
            gold_norm = {
                f: apply_rules(v, f, rules)
                for f, v in gold_flat.items()
            }
            truths.append(gold_norm)

        metrics = prepare.compute_metrics(preds, truths)
        s4_strict = sum(metrics["per_field_f1"].get(f, 0.0) for f in SAFETY) / 4
        s4_edit = sum(metrics["per_field_edit_f1"].get(f, 0.0) for f in SAFETY) / 4

        results[rule_name] = {
            "macro_f1": metrics["macro_f1"],
            "macro_edit_f1": metrics["macro_edit_f1"],
            "safety4_macro_f1": s4_strict,
            "safety4_macro_edit_f1": s4_edit,
            "per_field_f1": metrics["per_field_f1"],
            "per_field_edit_f1": metrics["per_field_edit_f1"],
            "rules_applied": list(rules.keys()),
        }
        print(
            f"  {rule_name:25s}  macro={metrics['macro_f1']:.4f}/{metrics['macro_edit_f1']:.4f}  "
            f"safety4={s4_strict:.4f}/{s4_edit:.4f}"
        )

    rules_out.parent.mkdir(parents=True, exist_ok=True)
    rules_out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {rules_out}")
    return 0


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dump", "score"], required=True)
    # dump-mode args
    ap.add_argument("--config")
    ap.add_argument("--ckpt")
    ap.add_argument("--predictions-out")
    # score-mode args
    ap.add_argument("--predictions-in")
    ap.add_argument("--rules-out")
    args = ap.parse_args()

    if args.mode == "dump":
        if not (args.config and args.ckpt and args.predictions_out):
            raise SystemExit("--mode dump requires --config, --ckpt, --predictions-out")
        return dump_predictions(args)

    if args.mode == "score":
        if not (args.predictions_in and args.rules_out):
            raise SystemExit("--mode score requires --predictions-in, --rules-out")
        return score_predictions(Path(args.predictions_in), Path(args.rules_out))

    raise SystemExit(f"unknown mode {args.mode}")


if __name__ == "__main__":
    sys.exit(main())
