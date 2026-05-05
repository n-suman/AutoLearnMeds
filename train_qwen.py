"""Track B trainer skeleton: Qwen2-VL-2B-Instruct + 4-bit quantization + LoRA.

Sibling of ``train.py`` (Track A: SigLIP+Donut). Same shape, same single-file
agent-editable convention, same final-metric stdout protocol so the existing
autoresearch harness (``scripts/run_experiment.sh``) parses runs end-to-end.

Heavy ML imports (torch / transformers / peft / bitsandbytes) are deliberately
deferred to inside the functions that use them, so this module imports cleanly
in the Mac dev venv (no ml extras) for unit tests and config-load smoke tests.
Tasks T4-T6 fill in the actual model wiring, training loop, and evaluation;
this file currently emits placeholder ``final_macro_*`` lines so the harness
contract holds even before the real trainer lands.
"""
from __future__ import annotations

# === Imports (lightweight only — heavy ml imports go inside functions) ===
import argparse
import dataclasses
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import yaml


# === Args dataclass ===

@dataclasses.dataclass
class QwenConfig:
    """Hyperparameters for one Track B (Qwen2-VL+LoRA) training run.

    Loaded from a YAML in experiments/configs/. Mirrors the schema of
    experiments/configs/qwen_baseline.yaml exactly. Field order matches
    that YAML's logical grouping: identity, quantization, LoRA, training,
    I/O, logging.
    """
    track: str
    model_name: str
    load_in_4bit: bool
    bnb_4bit_compute_dtype: str
    bnb_4bit_quant_type: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: tuple[str, ...]
    seed: int
    batch_size: int
    grad_accum_steps: int
    peak_lr: float
    min_lr: float
    warmup_steps: int
    max_steps: int
    weight_decay: float
    adam_betas: tuple[float, float]
    grad_clip: float
    eval_every: int
    log_every: int
    precision: str
    max_new_tokens_eval: int
    train_jsonl: str
    val_jsonl: str
    images_root: str
    path_strip_prefix: str
    checkpoint_dir: str
    wandb_project: str
    wandb_mode: str

    @classmethod
    def from_yaml(cls, path: str | Path) -> "QwenConfig":
        data = yaml.safe_load(Path(path).read_text())
        if isinstance(data.get("lora_target_modules"), list):
            data["lora_target_modules"] = tuple(data["lora_target_modules"])
        if isinstance(data.get("adam_betas"), list):
            data["adam_betas"] = tuple(data["adam_betas"])
        return cls(**data)


# === Seeding ===

def set_seed(seed: int) -> None:
    """Seed Python, numpy, torch (if installed) for reproducibility.

    Mirrors train.py:set_seed — numpy and torch imports are lazy so this
    function is safe to call from a torch-less venv.
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_qwen_with_lora(cfg: "QwenConfig") -> dict[str, Any]:
    """Load Qwen2-VL with optional 4-bit quantization, attach LoRA adapters.

    Returns {'model': PeftModel, 'processor': AutoProcessor}. The model has only
    LoRA adapter parameters trainable; the base Qwen2-VL weights are frozen
    (and 4-bit-quantized if cfg.load_in_4bit).
    """
    import torch
    from transformers import (
        AutoProcessor,
        Qwen2VLForConditionalGeneration,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb_config = None
    if cfg.load_in_4bit:
        compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
        )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(cfg.model_name)
    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.lora_target_modules),
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[track-b] LoRA attached: trainable={trainable:,} / total={total:,}")
    return {"model": model, "processor": processor}


# === Main (placeholder — T4-T6 fill in model + training loop + eval) ===

def main() -> int:
    parser = argparse.ArgumentParser(description="Track B (Qwen2-VL+LoRA) trainer.")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/qwen_baseline.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for cfg.seed.",
    )
    args = parser.parse_args()

    cfg = QwenConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg = dataclasses.replace(cfg, seed=args.seed)
    set_seed(cfg.seed)

    print(f"[track-b] track={cfg.track} model={cfg.model_name} seed={cfg.seed}")

    # Placeholder final metrics — T6 replaces these with the real eval values.
    print("final_macro_f1=0.0000")
    print("final_macro_edit_f1=0.0000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
