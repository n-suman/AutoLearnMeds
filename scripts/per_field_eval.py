"""Per-field evaluation for one checkpoint (Track A or Track B).

CLI tool for paper-grade Table 2: takes a single checkpoint, runs
``prepare.evaluate(...)`` on the val split, and writes the full per-field
metric dict to a JSON file. The companion ``compare_per_field.py`` script
then diffs two such JSONs side-by-side.

Module-level imports are kept lightweight (argparse / json / pathlib / yaml)
so this file can be imported in the torch-less Mac dev venv (used by tests).
torch / transformers / peft are imported lazily inside the per-track build
helpers - same convention as ``train_qwen.py``.

Usage:
    # Track A
    uv run python scripts/per_field_eval.py --track A \\
        --ckpt /workspace/checkpoints/runs/baseline/best.pt \\
        --out experiments/per_field_track_a.json

    # Track B
    uv run python scripts/per_field_eval.py --track B \\
        --lora /workspace/checkpoints/runs/qwen_baseline/best_lora \\
        --out experiments/per_field_track_b.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


# === Config helpers ===

def _load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML config and return its top-level dict."""
    return yaml.safe_load(Path(path).read_text())


def _resolve_io(cfg_dict: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Build the I/O kwargs for prepare.evaluate from CLI args + config defaults.

    Precedence: explicit CLI flag > config file value > hard-coded default.
    """
    return {
        "val_jsonl": args.val_jsonl or cfg_dict.get("val_jsonl", "data/processed/val.jsonl"),
        "images_root": args.images_root or cfg_dict.get("images_root", "/mnt/gcs/raw/raw_images"),
        "path_strip_prefix": (
            args.path_strip_prefix
            or cfg_dict.get("path_strip_prefix", "raw/raw_images/")
        ),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
    }


# === Track A: build PharmaVLM and load best.pt ===

def build_track_a(ckpt_path: str | Path, config_path: str | Path):
    """Construct PharmaVLM, load best.pt state, return (model, device).

    Mirrors how train.py:main() instantiates the model, then loads each piece
    of state from the full checkpoint dict that train.py:train_loop() saves
    at every new-best step. The encoder is FROZEN and re-loaded fresh from
    HF; only proj + decoder weights are restored from the checkpoint.
    """
    import torch  # noqa: F401  (lazy)

    # Add repo root to path so train + prepare can be imported.
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    import prepare
    import train

    cfg = train.Config.from_yaml(config_path)
    tokenizer = prepare.get_tokenizer(cfg.tokenizer_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = train.PharmaVLM(
        encoder_model=cfg.encoder_model,
        vocab_size=tokenizer.get_vocab_size(),
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_decoder_layers,
        n_heads=cfg.n_decoder_heads,
        ffn_ratio=cfg.ffn_ratio,
        dropout=cfg.dropout,
        tied_embeddings=cfg.tied_embeddings,
        tokenizer=tokenizer,
    ).to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    print(
        f"[per_field_eval] loaded ckpt step={ckpt.get('step')} "
        f"saved_macro_f1={ckpt.get('macro_f1')}"
    )

    # proj
    model.proj.load_state_dict(ckpt["proj_state_dict"])

    # decoder embedding + final LN
    model.decoder.embedding.load_state_dict(ckpt["decoder_embedding_state_dict"])
    model.decoder.final_ln.load_state_dict(ckpt["decoder_final_ln_state_dict"])

    # decoder blocks. train.py:self_named_params yields keys like
    # "ln1.weight", "q_proj.weight", "fc1.bias" - flat; one dict per block.
    block_attr_names = (
        "ln1", "q_proj", "k_proj", "v_proj", "o_proj",
        "ln2", "xq_proj", "xk_proj", "xv_proj", "xo_proj",
        "ln3", "fc1", "fc2",
    )
    for block, block_state in zip(model.decoder.blocks, ckpt["decoder_blocks_state_dicts"]):
        # Group keys by their attribute prefix (e.g. all "ln1.*" -> ln1).
        for attr in block_attr_names:
            sub_state = {
                k[len(attr) + 1:]: v
                for k, v in block_state.items()
                if k.startswith(attr + ".")
            }
            if sub_state:
                getattr(block, attr).load_state_dict(sub_state)

    model.train(False)
    return model, device


# === Track B: build base Qwen, load saved LoRA adapter, wrap ===

def build_track_b(lora_path: str | Path, config_path: str | Path):
    """Construct Qwen2-VL with 4-bit quant, attach SAVED LoRA adapters, wrap.

    Diverges from train_qwen.load_qwen_with_lora: that fn attaches FRESH
    LoRA via get_peft_model(); here we use PeftModel.from_pretrained to load
    the trained adapter weights from disk. Returns a QwenWrapper that
    plugs into prepare.evaluate.
    """
    import torch  # noqa: F401  (lazy)
    from peft import PeftModel
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2VLForConditionalGeneration,
    )

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    import train_qwen

    cfg = train_qwen.QwenConfig.from_yaml(config_path)

    bnb_config = None
    if cfg.load_in_4bit:
        compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
        )

    base = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(cfg.model_name)

    model = PeftModel.from_pretrained(base, str(lora_path))
    model.eval()
    print(f"[per_field_eval] loaded LoRA adapters from {lora_path}")

    wrapper = train_qwen.QwenWrapper(model, processor, cfg)
    return wrapper, cfg


# === Main ===

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--track", required=True, choices=["A", "B"])
    p.add_argument("--ckpt", default=None,
                   help="Path to best.pt (Track A only)")
    p.add_argument("--lora", default=None,
                   help="Path to LoRA adapter dir (Track B only)")
    p.add_argument("--config", default=None,
                   help="YAML config path (defaults per track)")
    p.add_argument("--val-jsonl", default=None)
    p.add_argument("--images-root", default=None)
    p.add_argument("--path-strip-prefix", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--out", required=True, help="Output JSON path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.track == "A":
        if not args.ckpt:
            print("ERROR: --ckpt is required for --track A", file=sys.stderr)
            return 2
        config_path = args.config or "experiments/configs/baseline.yaml"
        cfg_dict = _load_yaml(config_path)
        io = _resolve_io(cfg_dict, args)
        model, _device = build_track_a(args.ckpt, config_path)
        ckpt_or_lora = str(args.ckpt)
    else:
        if not args.lora:
            print("ERROR: --lora is required for --track B", file=sys.stderr)
            return 2
        config_path = args.config or "experiments/configs/qwen_baseline.yaml"
        cfg_dict = _load_yaml(config_path)
        io = _resolve_io(cfg_dict, args)
        model, _qcfg = build_track_b(args.lora, config_path)
        ckpt_or_lora = str(args.lora)

    # Lazy: only import prepare here so module-level remains light.
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    import prepare

    metrics = prepare.evaluate(
        model=model,
        jsonl_path=io["val_jsonl"],
        images_root=io["images_root"],
        path_strip_prefix=io["path_strip_prefix"],
        batch_size=io["batch_size"],
        max_new_tokens=io["max_new_tokens"],
    )

    out_payload: dict[str, Any] = {
        "track": args.track,
        "ckpt": ckpt_or_lora,
        "config": str(config_path),
        "val_jsonl": io["val_jsonl"],
        "n_samples": metrics.get("n_examples", 0),
        "macro_f1": metrics.get("macro_f1", 0.0),
        "macro_edit_f1": metrics.get("macro_edit_f1", 0.0),
        "per_field_f1": metrics.get("per_field_f1", {}),
        "per_field_edit_f1": metrics.get("per_field_edit_f1", {}),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2))

    print(
        f"[per_field_eval] track={args.track} "
        f"macro_f1={out_payload['macro_f1']:.4f} "
        f"macro_edit_f1={out_payload['macro_edit_f1']:.4f} "
        f"wrote={out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
