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


# === Dataset ===

def build_train_dataset(cfg: "QwenConfig", processor):
    """Wrap Track A's PharmaLabelDataset for Track B's per-step iteration.

    The Qwen training loop iterates this dataset index-by-index and tokenizes
    per-step (no DataLoader collation — Qwen's variable-length pixel_values
    makes batching awkward; gradient accumulation is used instead). The
    `processor` arg is unused here (kept in signature so the loop can pass
    it through if a future refactor needs it).
    """
    import prepare
    return prepare.PharmaLabelDataset(
        jsonl_path=cfg.train_jsonl,
        images_root=cfg.images_root,
        path_strip_prefix=cfg.path_strip_prefix,
    )


# === Per-step tokenization ===

def _build_qwen_sft_inputs(processor, image_pil, target_text: str):
    """Build a single-example SFT inputs dict for Qwen2-VL.

    Constructs a 3-turn chat (system / user-with-image / assistant-with-target),
    applies Qwen's chat template, runs the processor (with vision-info), then
    masks all non-assistant tokens to -100 in `labels` so the loss is computed
    only over the assistant span.
    """
    from qwen_vl_utils import process_vision_info

    messages = [
        {"role": "system", "content": [{"type": "text", "text": PHARMA_SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": image_pil}]},
        {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
    ]
    full_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Determine prompt-only token length so we mask the prompt out of labels.
    prompt_messages = messages[:2]
    prompt_text = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_image_inputs, prompt_video_inputs = process_vision_info(prompt_messages)
    prompt_inputs = processor(
        text=[prompt_text],
        images=prompt_image_inputs,
        videos=prompt_video_inputs,
        padding=True,
        return_tensors="pt",
    )
    prompt_len = int(prompt_inputs["input_ids"].shape[1])

    labels = inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    inputs["labels"] = labels
    return inputs


# === Training loop ===

def train_qwen_loop(bundle: dict, cfg: "QwenConfig", wandb_run) -> dict:
    """Hand-rolled single-process training loop with gradient accumulation.

    Iterates the train dataset index-by-index, tokenizes per step, accumulates
    gradients over `cfg.grad_accum_steps` micro-steps, then does a single optim
    step under cosine-with-warmup LR schedule. Evaluates every `cfg.eval_every`
    steps via `prepare.evaluate(QwenWrapper(...), ...)` and saves best LoRA
    adapters to `cfg.checkpoint_dir/best_lora`.

    Returns the best-by-macro-f1 metrics dict, regardless of whether the final
    step happened to be the best.
    """
    import math
    import time

    import torch
    from PIL import Image

    model = bundle["model"]
    processor = bundle["processor"]

    train_ds = build_train_dataset(cfg, processor)
    print(f"[track-b] train dataset size = {len(train_ds)}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        trainable_params,
        lr=cfg.peak_lr,
        betas=tuple(cfg.adam_betas),
        weight_decay=cfg.weight_decay,
    )

    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return cfg.peak_lr * (step + 1) / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return cfg.min_lr + 0.5 * (cfg.peak_lr - cfg.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )

    best_f1 = 0.0
    best_metrics = {"macro_f1": 0.0, "macro_edit_f1": 0.0}
    start = time.time()

    model.train(True)
    rng_indices = list(range(len(train_ds)))
    random.shuffle(rng_indices)
    cursor = 0
    step = 0
    loss_value = 0.0

    while step < cfg.max_steps:
        optim.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum_steps):
            if cursor >= len(rng_indices):
                random.shuffle(rng_indices)
                cursor = 0
            idx = rng_indices[cursor]
            cursor += 1

            sample = train_ds[idx]
            image_tensor = sample["image"]
            target_text = sample["target_text"]

            arr = (
                image_tensor.clamp(-1, 1)
                .add(1)
                .mul(127.5)
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            img_pil = Image.fromarray(arr)

            inputs = _build_qwen_sft_inputs(processor, img_pil, target_text)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(**inputs)
            loss = out.loss / cfg.grad_accum_steps
            loss.backward()
            loss_value = loss.item()

        torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
        cur_lr = lr_at(step)
        for g in optim.param_groups:
            g["lr"] = cur_lr
        optim.step()
        step += 1

        if step % cfg.log_every == 0:
            eff_loss = loss_value * cfg.grad_accum_steps
            print(
                f"[track-b] step={step} loss={eff_loss:.4f} lr={cur_lr:.2e}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {"train/loss": eff_loss, "train/lr": cur_lr, "step": step}
                )

        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            model.train(False)
            wrapper = QwenWrapper(model, processor, cfg)
            import prepare
            metrics = prepare.evaluate(
                wrapper,
                cfg.val_jsonl,
                cfg.images_root,
                cfg.path_strip_prefix,
                max_new_tokens=cfg.max_new_tokens_eval,
            )
            macro = metrics["macro_f1"]
            macro_edit = metrics.get("macro_edit_f1", 0.0)
            print(
                f"[track-b] step={step} val_macro_f1={macro:.4f} "
                f"val_macro_edit_f1={macro_edit:.4f}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "val/macro_f1": macro,
                        "val/macro_edit_f1": macro_edit,
                        "step": step,
                    }
                )
            if macro > best_f1:
                best_f1 = macro
                best_metrics = {"macro_f1": macro, "macro_edit_f1": macro_edit}
                ckpt_dir = Path(cfg.checkpoint_dir) / "best_lora"
                ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                print(
                    f"[track-b] saved best LoRA adapters to {ckpt_dir} "
                    f"(macro_f1={macro:.4f})"
                )
            model.train(True)

    wall = time.time() - start
    print(
        f"[track-b] DONE wall={wall:.1f}s "
        f"best_macro_f1={best_metrics['macro_f1']:.4f} "
        f"best_macro_edit_f1={best_metrics['macro_edit_f1']:.4f}"
    )
    return best_metrics


# === Main ===

def main(argv: list[str] | None = None) -> int:
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
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    args = parser.parse_args(argv)

    cfg = QwenConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg = dataclasses.replace(cfg, seed=args.seed)
    set_seed(cfg.seed)

    print(f"[track-b] track={cfg.track} model={cfg.model_name} seed={cfg.seed}")

    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                config=dataclasses.asdict(cfg),
                mode=cfg.wandb_mode,
            )
        except Exception as e:
            print(f"[track-b] WARN: wandb init failed ({e}); continuing without it")

    bundle = load_qwen_with_lora(cfg)
    metrics = train_qwen_loop(bundle, cfg, wandb_run)

    if wandb_run is not None:
        wandb_run.finish()

    print(f"final_macro_f1={metrics['macro_f1']:.4f}")
    print(f"final_macro_edit_f1={metrics['macro_edit_f1']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
