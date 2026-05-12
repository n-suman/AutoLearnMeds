# Brief 003: Pseudo-label unlabeled pharma images with Qwen2-VL-7B on Colab GPU

**Owner:** Codex (writes script + light test); user runs the script on Colab.
**Created:** 2026-05-13 by Claude
**Supersedes:** Brief 001 (which required an OpenAI API key the user doesn't have)
**Estimated effort:** M (script writing + a small smoke test on Codex's side; bulk inference happens later on Colab)
**Estimated cost:** $0 in API spend; ~10–20 Colab Pro+ compute units (user has ~1,600 available; negligible)

## Why this brief exists

Brief 001 wanted GPT-4o to generate pseudo-XML labels for ~2,275 unlabeled pharma images. The user has no standalone OpenAI API key — only a Codex $20 subscription which is for agentic coding, not bulk inference. We pivot: do the labeling with an **open-source vision-language model** running on the user's Colab A100, where compute is already paid for. Quality will be lower than GPT-4o; the calibration step (run on the 111 gold val images first) decides whether the pseudo-labels are usable at all.

## Context (self-contained — Codex hasn't seen the conversation)

The `AutoLearnMeds` project is in deployment-push mode. Five tracks of comparative experiments are done (Track A is best at 0.074 strict / 0.32 lenient F1 on 12-field XML extraction from pharma product photos). The bottleneck is data: only 564 labeled training images. We want to expand to ~2,839 (564 gold + ~2,275 pseudo) so the upcoming high-resolution SigLIP-large training run has enough signal.

The unlabeled bucket: `gs://auto_learn_meds/raw/raw_images/` (~3,061 total minus 786 labeled). The labeled manifest reconstructs from `gs://auto_learn_meds/raw/golden_set/gold_standard.jsonl` + `splits.json`.

The 12-field XML schema and field semantics are documented in brief 001 — re-read that brief's "## The 12-field XML schema" + "Prompting tips" sections; they apply unchanged here.

## Goal

Produce `data/pseudo_labels/round_001.jsonl` (~2,275 rows) using an open-source VLM running on Colab, with a calibration step on the 111-image val set first to decide if quality is usable.

## Model choice

Use **`Qwen/Qwen2-VL-7B-Instruct`** (HuggingFace) as the labeling oracle. Rationale:

- Strong OCR + structured-output capabilities, especially on small text.
- 7B params fits comfortably on a Colab A100 (40 GB) in bf16 with native-resolution input.
- Apache 2.0 license — no usage restrictions for our research/deployment context.
- Already familiar to this codebase: Track B fine-tuned a 2B variant via LoRA. The 7B base is just a bigger sibling, no surprises.

**Backups if Qwen2-VL-7B underperforms:**

- `OpenGVLab/InternVL2-8B` — comparable size, sometimes stronger on dense text.
- `meta-llama/Llama-3.2-11B-Vision-Instruct` — larger but slower; gated, user has access.
- (Last resort) `Qwen/Qwen2-VL-2B-Instruct` — same family, half the size, runs faster but with quality loss.

Pick Qwen2-VL-7B as default. Document the choice and any switching rationale in the status file.

## Prerequisites Codex needs

1. `python3 --version` ≥ 3.11.
2. `gsutil` installed; `gcloud auth list` shows `suman.nandamury@gmail.com` is active.
3. Read access to `gs://auto_learn_meds/raw/` confirmed via `gsutil ls`.
4. Repo on branch `phase-0-plumbing`, working tree clean (or only `uv.lock` + the Malepati PDF dirty, which is fine).
5. **No GPU is needed on Codex's machine.** The script writes runs on Colab. Codex's job is to write the script and run a tiny CPU-side smoke test against a single sample image to confirm the inference path executes; the bulk labeling happens on the user's Colab.
6. No paid API keys required.

## Step-by-step actions for Codex

### Step 1 — Write the labeling script

Path: `scripts/codex/pseudo_label_qwen2vl.py`

Required CLI signature:

```bash
python scripts/codex/pseudo_label_qwen2vl.py \
    --mode {calibrate, label} \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --gold-jsonl raw/val_reconstructed.jsonl  \   # for calibrate mode
    --unlabeled-list /path/to/unlabeled_paths.txt \  # for label mode
    --out-jsonl data/pseudo_labels/round_001.jsonl \
    --batch-size 1 \
    --max-new-tokens 512 \
    --resume                                          # idempotent
```

Behavior:

- `--mode calibrate`: load val_reconstructed.jsonl, for each row, run the VLM with the project's 12-field-XML prompt, parse output to XML, compute `prepare.compute_metrics(predictions, truths)` against gold targets. Write `data/pseudo_labels/round_001.calibration.json` with macro_f1, macro_edit_f1, per-field breakdown, and per-pack-type breakdown if pack-type is in the val manifest. STOP after calibration if `macro_edit_f1` on safety-critical 4 (batch_number, expiry_date, mrp, manufacturing_date) is **< 0.30**.
- `--mode label`: process the unlabeled list. Resume-aware via a sidecar `data/pseudo_labels/round_001.progress.jsonl` that holds `{image_path, status: done|errored, latency_sec, error: str|null}` per processed image. On startup, skip image_paths already marked `done`. Write atomically — flush JSONL line by line, not in batches.
- Output row format:
  ```json
  {
    "image_path": "gs://auto_learn_meds/raw/raw_images/IMG_5246.JPG",
    "xml_label": "<medication>...</medication>",
    "per_field_confidence": {"brand_name": "high", ...},
    "model_version": "Qwen/Qwen2-VL-7B-Instruct",
    "prompt_hash": "<sha256>",
    "generated_at": "2026-05-13T20:00:00Z",
    "latency_sec": 3.4
  }
  ```
- Use Qwen2-VL's native chat-template via `transformers.AutoProcessor`. Run in bf16. Resize to a side length of 1024 (Qwen2-VL handles native resolution via `min_pixels` / `max_pixels`; configure those to favor higher resolution for OCR).
- Per-field confidence: ask the model to emit a final JSON block AFTER the XML, listing each field as low/medium/high. Parse that block separately from the XML.
- Robustness: wrap each call in try/except; on parse failure, write an `errored` row to progress and continue.

### Step 2 — CPU smoke test (on Codex's machine)

Verify the script imports and the prompt-template path works without actually loading the 14 GB model:

```bash
python -c "
from scripts.codex.pseudo_label_qwen2vl import build_prompt, parse_response
prompt = build_prompt(few_shot_examples=[])
print(prompt[:500])
response = '<medication><brand_name>Crocin</brand_name>...</medication>\n```json\n{\"brand_name\": \"high\"}\n```'
parsed = parse_response(response)
print(parsed)
"
```

This confirms the prompt-builder + response-parser are well-formed; full model loading happens on Colab.

### Step 3 — Document the Colab runbook

Write `scripts/codex/qwen2vl_colab_runbook.md` with the exact commands the user runs on Colab:

```bash
# 1. Pull the latest code from the branch
cd /content/AutoLearnMeds && git pull origin phase-0-plumbing

# 2. Calibrate on val
python scripts/codex/pseudo_label_qwen2vl.py \
    --mode calibrate \
    --gold-jsonl raw/val_reconstructed.jsonl \
    --out-jsonl data/pseudo_labels/round_001.jsonl

# 3. Inspect calibration result
cat data/pseudo_labels/round_001.calibration.json

# 4. If lenient F1 safety-4 >= 0.30, build the unlabeled list
python -c "
import json
from pathlib import Path
gold = set()
with open('raw/golden_set/gold_standard.jsonl') as f:
    for line in f:
        gold.add(json.loads(line)['image_path'])
# list all images in raw_images/, subtract gold
import subprocess
out = subprocess.check_output(['gsutil', 'ls', 'gs://auto_learn_meds/raw/raw_images/']).decode().splitlines()
unlabeled = [p for p in out if p not in gold]
Path('data/pseudo_labels/unlabeled_paths.txt').write_text('\n'.join(unlabeled))
print(f'{len(unlabeled)} unlabeled images')
"

# 5. Run labeling on the unlabeled set
python scripts/codex/pseudo_label_qwen2vl.py \
    --mode label \
    --unlabeled-list data/pseudo_labels/unlabeled_paths.txt \
    --out-jsonl data/pseudo_labels/round_001.jsonl \
    --resume
```

The user runs these in the Colab notebook (`notebooks/00_bootstrap.ipynb` or a new cell). Codex's job is just to write the script + runbook; the user reports back numbers.

### Step 4 — Commit

```bash
git add scripts/codex/pseudo_label_qwen2vl.py scripts/codex/qwen2vl_colab_runbook.md
git commit -m "feat(codex): Qwen2-VL-7B pseudo-labeling pipeline (brief 003)"
```

### Step 5 — Write status

In `status/003-pseudo-labeling-qwen2vl-colab.md`, record:
- Confirmation that the script's CPU smoke test passes.
- Decision rationale for any model swap (if Qwen2-VL-7B unavailable, what's the fallback).
- The runbook hand-off note: "Ready for user to run on Colab. Calibration first."

Codex stops here. The user runs the calibration on Colab and reports the F1. Based on that, Claude decides whether to launch the full labeling pass and what sample-weight to give the pseudo-labels in Track E training.

## Acceptance criteria

1. ✅ `scripts/codex/pseudo_label_qwen2vl.py` exists, imports cleanly, has the CLI documented above.
2. ✅ CPU smoke test in Step 2 prints non-empty output and the parser handles malformed responses without crashing.
3. ✅ `scripts/codex/qwen2vl_colab_runbook.md` exists and is copy-pasteable.
4. ✅ `status/003-...md` documents script choices + hand-off.
5. ✅ Script is **resume-aware** — if interrupted, re-running with `--resume` continues from the last processed row.

## Anti-goals — explicitly DO NOT

- ❌ Do not try to install or run the 14 GB Qwen2-VL-7B model on Codex's machine. Smoke-test the script paths only.
- ❌ Do not call any paid API (OpenAI, Anthropic, Google AI Studio). The whole point of this brief is zero API cost.
- ❌ Do not modify `train*.py`, `prepare.py`, `experiments/ledger.jsonl`, or `experiments/configs/*`. Those are Claude's.
- ❌ Do not commit anything to `data/pseudo_labels/round_001.jsonl` from Codex's side — the actual labels come from the Colab run.
- ❌ Do not push the branch. The user pushes manually.

## Where to write

| File | Purpose |
|---|---|
| `scripts/codex/pseudo_label_qwen2vl.py` | The reusable labeling script |
| `scripts/codex/qwen2vl_colab_runbook.md` | Copy-paste Colab runbook |
| `status/003-pseudo-labeling-qwen2vl-colab.md` | Status + hand-off note |

## Coordination

- Brief 002 (GCS inventory) is independent and can run concurrently.
- Brief 001 is `SUPERSEDED` — leave its status file alone.
- The user runs the actual labeling on Colab once Codex hands off; Claude then picks up the calibration result to decide on Track E training.
