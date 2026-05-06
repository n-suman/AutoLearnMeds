# Track B — Qwen2-VL + LoRA Fine-Tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a parallel "Track B" — fine-tune `Qwen/Qwen2-VL-2B-Instruct` via LoRA on the same `data/processed/{train,val,test}.jsonl`, evaluate against the same `prepare.evaluate()` (so results compare directly to Track A's custom hybrid), add Track B's runs to the unified `experiments/ledger.jsonl`. The paper's contribution shifts to *comparative methodology under small-data pharma constraints*.

**Architecture:** New `train_qwen.py` (sibling to `train.py`, NOT a replacement) loads Qwen2-VL via HuggingFace `transformers`, attaches LoRA adapters via `peft`, fine-tunes on (image, structured-XML-output) pairs, exposes `predict_text(images, max_new_tokens) -> list[str]` matching Track A's interface. `scripts/run_experiment.sh` learns `--track A|B` dispatch.

**Tech Stack:** PyTorch, Transformers (>=4.46 for Qwen2-VL), `peft` (LoRA), `bitsandbytes` (4-bit quantization), accelerate. Qwen2-VL-2B at 4-bit + LoRA fits comfortably on A100 40GB.

**Note on `.train(False)`:** PyTorch's `Module.eval()` and `.train(False)` are semantically identical. We use `.train(False)` throughout for symmetry with `.train(True)`.

**Phase 4a prerequisites met:** `prepare.py` has `macro_edit_f1` alongside `macro_f1`; ledger pipeline operational; leaderboard renders both metrics.

**Out of scope:** Mistral / Phi / PaliGemma comparisons. One large-VLM baseline first.

---

## File Structure

| Path | Created/Modified | Responsibility |
|---|---|---|
| `train_qwen.py` | Create | Single-file Track B trainer. Sibling to `train.py`. |
| `experiments/configs/qwen_baseline.yaml` | Create | Locked Track B baseline config. |
| `pyproject.toml` | Modify | Add `peft`, `bitsandbytes`, `qwen-vl-utils` to ml extras; bump `transformers>=4.46`. |
| `scripts/run_experiment.sh` | Modify | Accept `--track A|B`, dispatch to `train.py` vs `train_qwen.py`, record `track` in `metrics.json`. |
| `tests/test_train_qwen_components.py` | Create | Lightweight: yaml load + lazy-imports invariant. |
| `tests/test_train_qwen_smoke.py` | Create | Colab-marked: load model+LoRA, predict_text shape, single forward. |
| `docs/superpowers/reviews/track_b_review.md` | Create | End-of-track review. |

**`train_qwen.py` internal sections:** Imports, QwenConfig dataclass, set_seed, load_qwen_with_lora, build_qwen_chat (system prompt + chat template), QwenWrapper (predict_text façade), build_train_dataset (Qwen-tokenized SFT pairs), train_qwen_loop, Main.

---

## Task 1: Add deps to pyproject.toml

- [ ] **Step 1: Edit `pyproject.toml`** — bump `transformers>=4.46` and add to ml extras:

```toml
ml = [
    "torch>=2.3", "torchvision>=0.18",
    "transformers>=4.46",  # bumped from 4.42 — Qwen2-VL minimum
    "accelerate>=0.31", "datasets>=2.20", "pillow>=10.3",
    "numpy>=1.26", "pandas>=2.2", "wandb>=0.17",
    "scikit-learn>=1.5", "matplotlib>=3.9", "tqdm>=4.66",
    "tokenizers>=0.19", "python-Levenshtein>=0.25",
    "peft>=0.13",                    # LoRA
    "bitsandbytes>=0.44",            # 4-bit quantization
    "qwen-vl-utils>=0.0.10",         # Qwen2-VL helpers
]
```

- [ ] **Step 2: `uv sync --extra dev`** locally (no ml install — those land on Colab).

- [ ] **Step 3: Commit** `chore(deps): peft + bitsandbytes for Track B (Qwen2-VL+LoRA)`.

---

## Task 2: `experiments/configs/qwen_baseline.yaml`

Create the locked Track B baseline:

```yaml
track: B
model_name: "Qwen/Qwen2-VL-2B-Instruct"

# Quantization
load_in_4bit: true
bnb_4bit_compute_dtype: "bfloat16"
bnb_4bit_quant_type: "nf4"

# LoRA
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training
seed: 42
batch_size: 4
grad_accum_steps: 4   # effective batch 16
peak_lr: 2.0e-4       # higher than Track A — LoRA converges faster
min_lr: 2.0e-5
warmup_steps: 50
max_steps: 500        # half of Track A
weight_decay: 0.01    # lower — LoRA is inherently regularizing
adam_betas: [0.9, 0.999]
grad_clip: 1.0
eval_every: 100
log_every: 10
precision: "bf16"
max_new_tokens_eval: 256

# I/O (shared with Track A)
train_jsonl: "data/processed/train.jsonl"
val_jsonl: "data/processed/val.jsonl"
images_root: "/mnt/gcs/raw/raw_images"
path_strip_prefix: "raw/raw_images/"
checkpoint_dir: "checkpoints/runs/qwen_baseline"
wandb_project: "autolearnmeds"
wandb_mode: "online"
```

Commit: `feat(track-b): qwen_baseline.yaml — locked Track B config`.

---

## Task 3: `train_qwen.py` skeleton + 2 lightweight tests

`tests/test_train_qwen_components.py` — fixture loads `train_qwen` like Track A's pattern. Tests:
- `test_qwen_config_loads_baseline_yaml`: asserts `track == "B"`, `model_name == "Qwen/Qwen2-VL-2B-Instruct"`, `lora_rank == 8`.
- `test_qwen_module_imports_without_torch`: asserts `not hasattr(train_qwen_mod, "torch")` and `not hasattr(train_qwen_mod, "transformers")`.

`train_qwen.py` skeleton has:
- `QwenConfig` dataclass mirroring the YAML fields, `from_yaml` converts list→tuple for `adam_betas` and `lora_target_modules`.
- `set_seed` (same semantics as `train.py`).
- `main()` that loads config, calls set_seed, prints the literal `final_macro_f1=0.0000\nfinal_macro_edit_f1=0.0000` lines so the autoresearch harness parses end-to-end.

Commit: `feat(track-b): train_qwen.py skeleton + QwenConfig dataclass + 2 lightweight tests`.

---

## Task 4: Qwen2-VL loader + LoRA attachment

Append to `train_qwen.py`:

```python
def load_qwen_with_lora(cfg: QwenConfig) -> dict[str, Any]:
    """Load Qwen2-VL with optional 4-bit quantization, attach LoRA adapters."""
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
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.lora_target_modules),
        lora_dropout=cfg.lora_dropout, bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[track-b] LoRA attached: trainable={trainable:,} / total={total:,}")
    return {"model": model, "processor": processor}
```

Test (Colab-marked, in `tests/test_train_qwen_smoke.py`): `test_load_qwen_with_lora_returns_model` asserts the bundle has `model` + `processor`, and trainable param count is between 100k and total/100 (LoRA-expected range).

Commit: `feat(track-b): load_qwen_with_lora — 4-bit Qwen2-VL + LoRA r=8 + 1 colab-test`.

---

## Task 5: `QwenWrapper.predict_text` — match `prepare.evaluate` interface

Append to `train_qwen.py`:

```python
PHARMA_SYSTEM_PROMPT = """You are a pharmaceutical-label structured-information extractor. \
Given an image of a medicine product label, output the structured fields as XML in the form: \
<s><brand_name>...</brand_name><generic_name>...</generic_name>...</s>. \
Emit only fields you can read confidently. Field names: \
brand_name, drug_name, generic_name, strength, quantity, company, manufacturer, \
batch_number, mfg_date, expiry_date, mrp, warnings."""


def build_qwen_chat(image_pil) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": PHARMA_SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": image_pil}]},
    ]


class QwenWrapper:
    """Adapter so QwenVL plugs into prepare.evaluate alongside Track A's PharmaVLM."""

    def __init__(self, model, processor, cfg: QwenConfig) -> None:
        self.model = model
        self.processor = processor
        self.cfg = cfg

    def parameters(self):
        return self.model.parameters()

    def to(self, device):
        return self  # bnb 4-bit handles its own device placement

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def predict_text(self, images, max_new_tokens: int = 256) -> list[str]:
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        # Reverse SigLIP normalization (Track A's preprocessor) → PIL for Qwen.
        if images.dtype != torch.float32:
            images = images.float()
        outputs: list[str] = []
        for i in range(images.shape[0]):
            arr = images[i].clamp(-1, 1).add(1).mul(127.5).byte().permute(1, 2, 0).cpu().numpy()
            img_pil = Image.fromarray(arr)
            messages = build_qwen_chat(img_pil)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(self.model.device)
            self.train(False)
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
            generated_trim = generated[:, inputs["input_ids"].size(1):]
            decoded = self.processor.batch_decode(
                generated_trim, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            outputs.append(decoded[0] if decoded else "")
        return outputs
```

Test (Colab-marked): `test_qwen_predict_text_returns_strings` — asserts `predict_text(2-image-batch, 16)` returns `list[str]` of len 2.

Commit: `feat(track-b): QwenWrapper.predict_text — pluggable into prepare.evaluate`.

---

## Task 6: Training loop + main with grad accumulation

`build_train_dataset(cfg, processor)` wraps `prepare.PharmaLabelDataset`: for each (image, target_text), reconstructs PIL from SigLIP-normalized tensor, builds 3-turn chat (system, user with image, assistant with target_text), tokenizes via Qwen processor, masks all non-assistant tokens to `-100` in labels.

`train_qwen_loop(bundle, cfg, wandb_run)`:
- AdamW over `[p for p in model.parameters() if p.requires_grad]` (LoRA adapters only — ~5M params).
- One example per step + grad accumulation = effective batch `cfg.batch_size * cfg.grad_accum_steps`.
- Hand-rolled training (DataLoader collation is awkward for Qwen's variable-length pixel_values).
- bf16 autocast.
- Every `eval_every` steps: `model.train(False)` + `prepare.evaluate(QwenWrapper(...), ...)` → log macro_f1 + macro_edit_f1.
- Save best LoRA adapters via `model.save_pretrained(...)` (small, ~5M params).
- Final eval after `max_steps`; print both `final_macro_f1=X.XXXX` and `final_macro_edit_f1=X.XXXX` on stdout.

`main()` wires: argparse → QwenConfig.from_yaml → set_seed → load_qwen_with_lora → wandb init → train_qwen_loop → wandb.finish → final-metric prints.

Commit: `feat(track-b): training loop + main with grad accumulation`.

---

## Task 7: `run_experiment.sh` `--track` dispatch

Add to the arg-parse `while` loop:
```bash
--track) TRACK="$2"; shift 2;;
```

After arg-parse, dispatch:
```bash
TRACK="${TRACK:-A}"
case "$TRACK" in
  A) TRAINER="train.py"; DEFAULT_CONFIG="experiments/configs/baseline.yaml" ;;
  B) TRAINER="train_qwen.py"; DEFAULT_CONFIG="experiments/configs/qwen_baseline.yaml" ;;
  *) echo "[run_experiment] FATAL: unknown track $TRACK" >&2; exit 2 ;;
esac
if [[ "$CONFIG" == "experiments/configs/baseline.yaml" && "$TRACK" == "B" ]]; then
  CONFIG="$DEFAULT_CONFIG"
fi
```

Replace `python train.py` → `python "$TRAINER"`. Add `"track": "$TRACK"` to the `metrics.json` heredoc.

New test in `test_smoke.py`:
```python
def test_run_experiment_supports_track_arg(project_root: Path) -> None:
    text = (project_root / "scripts" / "run_experiment.sh").read_text()
    assert "--track" in text
    assert "train_qwen.py" in text
    assert "train.py" in text
```

Commit: `feat(track-b): run_experiment.sh accepts --track A|B`.

---

## Task 8: Track B baseline run on Colab

Push commits, re-bootstrap Colab if needed, then on Colab:

```bash
RUN_ID=qwen-baseline-seed42
mkdir -p experiments/runs/$RUN_ID
# Pre-write notes.md (hypothesis: +0.20 to +0.40 macro_f1; +0.40 to +0.70 edit_f1)
# Citations: Qwen2-VL technical report (add to papers/), Hu 2021 LoRA (add to papers/)
# Adjacent ideas: r=4/8/16 sweep; Qwen2-VL "Naive Dynamic Resolution" at native res
nohup bash scripts/run_experiment.sh $RUN_ID --track B --seed 42 --no-wandb \
  > /tmp/$RUN_ID.log 2>&1 &
```

~30-60 min including model download (~5GB) + LoRA train + eval. Monitor with poll-until-metrics.json. After completion, append Result + Retrospective to notes.md, run `bash scripts/finalize_experiment.sh $RUN_ID --phase baseline`.

scp artifacts back to Mac, commit + push from Mac (Colab git lacks creds).

---

## Task 9: Track B review + tag

Create `docs/superpowers/reviews/track_b_review.md` documenting:
- What was built (the 7 changes above).
- Track B baseline result (numbers TBD until run lands).
- Head-to-head with Track A:

| Approach | trainable params | macro_f1 | macro_edit_f1 | wall (min) |
|---|---|---|---|---|
| Track A: SigLIP+Donut | 26.5M | 0.071 | TBD | 30 |
| Track B: Qwen2-VL+LoRA r=8 | ~5M | TBD | TBD | 30-60 |

- The comparative paper framing (decision tree: when to use which approach under what data-size constraint).

Tag `track-b-baseline-complete`, push.

---

## Self-Review

| Spec / framing requirement | Implemented in task |
|---|---|
| Parallel Track B (Qwen2-VL + LoRA) | T3-T6 |
| Shared `prepare.evaluate` interface | T5 (QwenWrapper.predict_text) |
| Unified ledger across tracks | T7 (track field in metrics.json) |
| Same metrics on both tracks | done in upstream commit `ed729f2` (macro_edit_f1 added) |
| One launcher for both tracks | T7 (--track A|B) |
| Track B baseline on real data | T8 |
| Track B review | T9 |

No placeholders. Type consistency: `QwenWrapper.predict_text(images, max_new_tokens) -> list[str]` matches Track A's `PharmaVLM.predict_text` so `prepare.evaluate` accepts both unchanged.

Phase 4b-Track-B sweep (the open-ended autoresearch on Track B) follows after this plan tags `track-b-baseline-complete`.
