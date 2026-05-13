#!/usr/bin/env python3
"""Pseudo-label pharma label images with Qwen2-VL on Colab.

Codex-side smoke tests import prompt/parse helpers only. The heavy model path is
lazy-loaded inside `load_qwen_model`, intended for the user's Colab A100.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import re
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


CANONICAL_FIELDS = [
    "brand_name",
    "generic_name",
    "drug_name",
    "strength",
    "quantity",
    "company",
    "manufacturer",
    "batch_number",
    "mfg_date",
    "expiry_date",
    "mrp",
    "warnings",
]

FIELD_ALIASES = {
    "manufacturing_date": "mfg_date",
    "manufacture_date": "mfg_date",
    "mfg": "mfg_date",
    "exp_date": "expiry_date",
    "expiration_date": "expiry_date",
    "batch": "batch_number",
    "lot": "batch_number",
}

SAFETY_CRITICAL_FIELDS = ["batch_number", "expiry_date", "mrp", "mfg_date"]


PROMPT_TEMPLATE = """You are reading a photograph of an Indian pharmaceutical product package.

Extract ONLY text that is visible in the image. Do not infer or invent missing values.

Return two blocks and nothing else:

1. A valid XML block with this exact schema and field order:

<medication>
  <brand_name></brand_name>
  <generic_name></generic_name>
  <drug_name></drug_name>
  <strength></strength>
  <quantity></quantity>
  <company></company>
  <manufacturer></manufacturer>
  <batch_number></batch_number>
  <mfg_date></mfg_date>
  <expiry_date></expiry_date>
  <mrp></mrp>
  <warnings></warnings>
</medication>

2. A JSON confidence block with one low/medium/high value per field:

```json
{{"brand_name":"low","generic_name":"low","drug_name":"low","strength":"low","quantity":"low","company":"low","manufacturer":"low","batch_number":"low","mfg_date":"low","expiry_date":"low","mrp":"low","warnings":"low"}}
```

Field semantics:
- brand_name: marketed product name, often largest text on the pack.
- generic_name: pharmacological / INN name, e.g. Paracetamol.
- drug_name: full marketed name including variant; when unclear, can equal brand_name.
- strength: dose per unit including units, e.g. 500 mg or 10 mg/5 ml.
- quantity: pack contents, e.g. 10 tablets, 60 ml, 1 vial.
- company: top-level pharma company / brand owner.
- manufacturer: Mfd by / manufactured by entity, often with address.
- batch_number: alphanumeric code labeled B.No., Batch, or Lot.
- mfg_date: manufacturing date. If unclear, emit MM/YYYY when possible.
- expiry_date: expiry / use-by date. If unclear, emit MM/YYYY when possible.
- mrp: Indian MRP. Emit a bare number in INR; strip Rs., INR, and rupee symbols.
- warnings: caution/warning/storage text. Join multiple warning lines with ". ".

If a field is not visible, leave that XML tag empty and set confidence to low."""


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def build_prompt(few_shot_examples: list[dict[str, Any]] | None = None) -> str:
    """Build the extraction prompt.

    The script currently keeps few-shot examples as text-only optional context;
    Colab inference uses a single image per request to keep memory predictable.
    """
    prompt = PROMPT_TEMPLATE
    examples = few_shot_examples or []
    if examples:
        prompt += "\n\nGold examples:\n"
        for i, ex in enumerate(examples, start=1):
            prompt += f"\nExample {i} expected XML:\n{ex.get('xml_label', '').strip()}\n"
    return prompt


def normalize_field_name(name: str) -> str:
    clean = name.strip().lower().replace("-", "_")
    return FIELD_ALIASES.get(clean, clean)


def normalize_xml_fields(xml: str) -> str:
    """Normalize known non-canonical tags into project-canonical tags."""
    for old, new in FIELD_ALIASES.items():
        xml = re.sub(fr"<\s*{old}\s*>", f"<{new}>", xml, flags=re.IGNORECASE)
        xml = re.sub(fr"<\s*/\s*{old}\s*>", f"</{new}>", xml, flags=re.IGNORECASE)
    return xml


def empty_confidence() -> dict[str, str]:
    return {field: "low" for field in CANONICAL_FIELDS}


def extract_xml(response: str) -> str:
    match = re.search(r"<medication\b[^>]*>.*?</medication>", response, flags=re.DOTALL | re.IGNORECASE)
    source = match.group(0) if match else response
    source = normalize_xml_fields(source)

    fields: dict[str, str] = {}
    for field in CANONICAL_FIELDS:
        tag = re.escape(field)
        tag_match = re.search(fr"<{tag}>(.*?)</{tag}>", source, flags=re.DOTALL | re.IGNORECASE)
        if tag_match:
            fields[field] = tag_match.group(1).strip()
    if match:
        return compose_medication_xml(fields)

    for field in FIELD_ALIASES:
        tag = re.escape(field)
        tag_match = re.search(fr"<{tag}>(.*?)</{tag}>", response, flags=re.DOTALL | re.IGNORECASE)
        if tag_match:
            fields[normalize_field_name(field)] = tag_match.group(1).strip()
    return compose_medication_xml(fields)


def extract_confidence(response: str) -> dict[str, str]:
    confidences = empty_confidence()
    candidates = re.findall(r"```json\s*(\{.*?\})\s*```", response, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(re.findall(r"(\{[^{}]*(?:low|medium|high)[^{}]*\})", response, flags=re.DOTALL | re.IGNORECASE))
    for candidate in candidates:
        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                for key, value in parsed.items():
                    field = normalize_field_name(str(key))
                    if field in confidences:
                        val = str(value).strip().lower()
                        confidences[field] = val if val in {"low", "medium", "high"} else "low"
                return confidences
    return confidences


def parse_response(response: str) -> dict[str, Any]:
    """Parse a Qwen response without raising on malformed output."""
    try:
        xml = extract_xml(response)
        confidence = extract_confidence(response)
        return {"xml_label": xml, "per_field_confidence": confidence, "error": None}
    except Exception as exc:  # defensive: malformed model text should not stop a run
        return {
            "xml_label": compose_medication_xml({}),
            "per_field_confidence": empty_confidence(),
            "error": f"{type(exc).__name__}: {exc}",
        }


def escape_xml_text(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def compose_medication_xml(values: dict[str, str]) -> str:
    parts = ["<medication>"]
    for field in CANONICAL_FIELDS:
        parts.append(f"<{field}>{escape_xml_text(values.get(field, '') or '')}</{field}>")
    parts.append("</medication>")
    return "".join(parts)


def record_to_truth(record: dict[str, Any]) -> dict[str, str]:
    """Convert a gold row to prepare.compute_metrics truth dict shape."""
    from prepare import parse_output

    if "xml_label" in record and record["xml_label"]:
        return parse_output(normalize_xml_fields(str(record["xml_label"])))
    fields = record.get("fields") or {}
    return {
        normalize_field_name(field): str(payload.get("text", ""))
        for field, payload in fields.items()
        if isinstance(payload, dict) and payload.get("text") is not None
    }


def image_path_from_record(record: dict[str, Any]) -> str:
    for key in ("image_path", "image_file", "path"):
        if record.get(key):
            return str(record[key])
    raise ValueError(f"record has no image path field: keys={sorted(record)}")


def resolve_image_path(path: str, images_root: str) -> str:
    if path.startswith("gs://") or Path(path).is_absolute():
        return path
    if path.startswith("raw/raw_images/"):
        return f"gs://auto_learn_meds/{path}"
    if path.startswith("raw_images/"):
        return f"gs://auto_learn_meds/raw/{path}"
    if images_root.startswith("gs://"):
        return f"{images_root.rstrip('/')}/{Path(path).name}"
    return str(Path(images_root) / Path(path).name)


@contextlib.contextmanager
def local_image_path(image_path: str):
    """Yield a local path, copying gs:// images to a temp file when needed."""
    if not image_path.startswith("gs://"):
        yield image_path
        return

    suffix = Path(image_path).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        subprocess.run(["gsutil", "cp", image_path, tmp.name], check=True)
        yield tmp.name


@dataclasses.dataclass
class QwenRuntime:
    model: Any
    processor: Any
    device: str
    dtype: Any


def load_qwen_model(model_name: str, min_pixels: int, max_pixels: int) -> QwenRuntime:
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return QwenRuntime(model=model, processor=processor, device="cuda", dtype=torch.bfloat16)


def generate_one(runtime: QwenRuntime, image_path: str, prompt: str, max_new_tokens: int) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    with local_image_path(image_path) as local_path:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": local_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = runtime.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = runtime.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(runtime.model.device)

        with torch.no_grad():
            generated_ids = runtime.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        decoded = runtime.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0] if decoded else ""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()


def done_paths(progress_path: Path) -> set[str]:
    done: set[str] = set()
    if not progress_path.exists():
        return done
    with progress_path.open() as f:
        for line in f:
            with contextlib.suppress(json.JSONDecodeError):
                row = json.loads(line)
                if row.get("status") == "done" and row.get("image_path"):
                    done.add(str(row["image_path"]))
    return done


def progress_path_for(out_jsonl: Path) -> Path:
    if out_jsonl.name.endswith(".jsonl"):
        return out_jsonl.with_name(out_jsonl.name.removesuffix(".jsonl") + ".progress.jsonl")
    return out_jsonl.with_suffix(out_jsonl.suffix + ".progress.jsonl")


def calibration_path_for(out_jsonl: Path) -> Path:
    if out_jsonl.name.endswith(".jsonl"):
        return out_jsonl.with_name(out_jsonl.name.removesuffix(".jsonl") + ".calibration.json")
    return out_jsonl.with_suffix(out_jsonl.suffix + ".calibration.json")


def pack_type_breakdown(predictions: list[dict[str, str]], truths: list[dict[str, str]], records: list[dict[str, Any]]) -> dict[str, Any]:
    from prepare import compute_metrics

    out: dict[str, Any] = {}
    for pack_type in sorted({str(r.get("pack_type", "unknown")) for r in records}):
        idxs = [i for i, r in enumerate(records) if str(r.get("pack_type", "unknown")) == pack_type]
        if not idxs:
            continue
        metrics = compute_metrics([predictions[i] for i in idxs], [truths[i] for i in idxs])
        out[pack_type] = {
            "n": len(idxs),
            "macro_f1": metrics["macro_f1"],
            "macro_edit_f1": metrics.get("macro_edit_f1", 0.0),
        }
    return out


def safety4_edit_f1(metrics: dict[str, Any]) -> float:
    per_field = metrics.get("per_field_edit_f1", {})
    values = [float(per_field.get(field, 0.0)) for field in SAFETY_CRITICAL_FIELDS]
    return sum(values) / len(values)


def calibrate(args: argparse.Namespace) -> int:
    from prepare import compute_metrics, parse_output

    prompt = build_prompt([])
    runtime = load_qwen_model(args.model, args.min_pixels, args.max_pixels)
    records = read_jsonl(args.gold_jsonl)
    predictions: list[dict[str, str]] = []
    truths: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []

    for record in records:
        image_path = resolve_image_path(image_path_from_record(record), args.images_root)
        started = time.time()
        parsed: dict[str, Any]
        try:
            response = generate_one(runtime, image_path, prompt, args.max_new_tokens)
            parsed = parse_response(response)
        except Exception as exc:
            parsed = {
                "xml_label": compose_medication_xml({}),
                "per_field_confidence": empty_confidence(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        latency = time.time() - started
        pred = parse_output(parsed["xml_label"])
        truth = record_to_truth(record)
        predictions.append(pred)
        truths.append(truth)
        rows.append(
            {
                "image_path": image_path,
                "xml_label": parsed["xml_label"],
                "per_field_confidence": parsed["per_field_confidence"],
                "model_version": args.model,
                "prompt_hash": prompt_hash(prompt),
                "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
                "latency_sec": round(latency, 3),
                "error": parsed.get("error"),
            }
        )

    metrics = compute_metrics(predictions, truths)
    safety4 = safety4_edit_f1(metrics)
    report = {
        "model_version": args.model,
        "n": len(records),
        "prompt_hash": prompt_hash(prompt),
        "macro_f1": metrics["macro_f1"],
        "macro_edit_f1": metrics.get("macro_edit_f1", 0.0),
        "safety4_macro_edit_f1": safety4,
        "per_field_f1": metrics.get("per_field_f1", {}),
        "per_field_edit_f1": metrics.get("per_field_edit_f1", {}),
        "per_pack_type": pack_type_breakdown(predictions, truths, records),
        "stop_reason": (
            "safety4_macro_edit_f1_below_0.30"
            if safety4 < args.safety4_threshold
            else "safety4_macro_edit_f1_passed"
        ),
        "predictions": rows,
    }
    cal_path = calibration_path_for(args.out_jsonl)
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    cal_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {cal_path}")
    print(f"macro_f1={report['macro_f1']:.4f}")
    print(f"macro_edit_f1={report['macro_edit_f1']:.4f}")
    print(f"safety4_macro_edit_f1={safety4:.4f}")
    if safety4 < args.safety4_threshold:
        print("STOP: safety-critical edit F1 below threshold")
        return 2
    return 0


def label(args: argparse.Namespace) -> int:
    from prepare import parse_output

    prompt = build_prompt([])
    runtime = load_qwen_model(args.model, args.min_pixels, args.max_pixels)
    image_paths = [
        line.strip()
        for line in args.unlabeled_list.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    progress_path = progress_path_for(args.out_jsonl)
    completed = done_paths(progress_path) if args.resume else set()

    for raw_path in image_paths:
        image_path = resolve_image_path(raw_path, args.images_root)
        if args.resume and image_path in completed:
            continue
        started = time.time()
        try:
            response = generate_one(runtime, image_path, prompt, args.max_new_tokens)
            parsed = parse_response(response)
            status = "done" if parsed.get("error") is None else "errored"
            output_row = {
                "image_path": image_path,
                "xml_label": parsed["xml_label"],
                "per_field_confidence": parsed["per_field_confidence"],
                "model_version": args.model,
                "prompt_hash": prompt_hash(prompt),
                "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
                "latency_sec": round(time.time() - started, 3),
            }
            if status == "done":
                # Ensures malformed XML still goes through the same parser path now.
                parse_output(output_row["xml_label"])
                append_jsonl(args.out_jsonl, output_row)
            error = parsed.get("error")
        except Exception as exc:
            status = "errored"
            error = f"{type(exc).__name__}: {exc}"

        append_jsonl(
            progress_path,
            {
                "image_path": image_path,
                "status": status,
                "latency_sec": round(time.time() - started, 3),
                "error": error,
            },
        )
        print(f"{status}: {image_path}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["calibrate", "label"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--gold-jsonl", type=Path)
    parser.add_argument("--unlabeled-list", type=Path)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--images-root", default="gs://auto_learn_meds/raw/raw_images")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-pixels", type=int, default=1024 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1600 * 28 * 28)
    parser.add_argument("--safety4-threshold", type=float, default=0.30)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size != 1:
        raise SystemExit("Only --batch-size 1 is currently supported for predictable A100 memory use.")
    if args.mode == "calibrate" and not args.gold_jsonl:
        raise SystemExit("--gold-jsonl is required in calibrate mode")
    if args.mode == "label" and not args.unlabeled_list:
        raise SystemExit("--unlabeled-list is required in label mode")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "calibrate":
        raise SystemExit(calibrate(args))
    raise SystemExit(label(args))


if __name__ == "__main__":
    main()
