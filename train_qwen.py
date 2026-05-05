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


PHARMA_SYSTEM_PROMPT = (
    "You are a pharmaceutical-label structured-information extractor. "
    "Given an image of a medicine product label, output the structured fields as XML "
    "in the form: <s><brand_name>...</brand_name><generic_name>...</generic_name>...</s>. "
    "Emit only fields you can read confidently. Field names: brand_name, drug_name, "
    "generic_name, strength, quantity, company, manufacturer, batch_number, "
    "mfg_date, expiry_date, mrp, warnings."
)


def build_qwen_chat(image_pil) -> list[dict[str, Any]]:
    """Build a 2-turn chat for Qwen2-VL: system prompt + user image."""
    return [
        {"role": "system", "content": [{"type": "text", "text": PHARMA_SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": image_pil}]},
    ]


class QwenWrapper:
    """Adapter so QwenVL plugs into prepare.evaluate alongside Track A's PharmaVLM.

    Exposes predict_text(images: torch.Tensor, max_new_tokens: int) -> list[str], the
    same contract as PharmaVLM. Internally: reverses SigLIP normalization to PIL, runs
    Qwen's processor + chat template, generates with do_sample=False, decodes the new
    tokens (excluding prompt), returns the strings.
    """

    def __init__(self, model, processor, cfg: "QwenConfig") -> None:
        self.model = model
        self.processor = processor
        self.cfg = cfg

    def parameters(self):
        return self.model.parameters()

    def to(self, device):
        # bnb 4-bit handles its own device placement via device_map="auto".
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def predict_text(self, images, max_new_tokens: int = 256) -> list[str]:
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        if images.dtype != torch.float32:
            images = images.float()

        outputs: list[str] = []
        for i in range(images.shape[0]):
            # Reverse SigLIP normalization: [-1, 1] → [0, 255] uint8 PIL.
            arr = (
                images[i]
                .clamp(-1, 1)
                .add(1)
                .mul(127.5)
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            img_pil = Image.fromarray(arr)

            messages = build_qwen_chat(img_pil)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            self.train(False)
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
            generated_trim = generated[:, inputs["input_ids"].size(1):]
            decoded = self.processor.batch_decode(
                generated_trim,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            outputs.append(decoded[0] if decoded else "")
        return outputs


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
