# Phase 8 — Track D: YOLOv12 + SAHI + per-region OCR Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build **Track D** — a modular detection-then-extract pipeline that detects field-bearing regions in pharma label photos using YOLOv12 with Slicing-Aided Hyper Inference (SAHI), runs per-region text recognition with TrOCR, and assembles the results into structured XML evaluatable through the same `prepare.evaluate()` harness as Tracks A/B/C. Replicates the modular pipeline of Malepati 2026 [malepati2026sahi] and extends it to full structured-extraction (their paper ends at detection + classification; we add the OCR head and the XML composition).

**Architecture:**

```
Image (3024×4032 phone photo)
    ↓
SAHI tiling (1024-px tiles, 0.35 overlap, base 1280)
    ↓
YOLOv12-s detection per tile
    ↓  (per-class confidence threshold; targeted SAHI on OCR-critical classes per Malepati 2026)
NMS-merge across tiles → per-image detections (bbox, class, confidence)
    ↓
For each detection: crop bbox region from full-res image
    ↓
TrOCR-base-printed per region → field value text
    ↓
Aggregate (field_name, value) pairs → XML
    ↓
prepare.evaluate(track_d_predict_text, val_jsonl, ...)
    ↓
macro_f1, macro_edit_f1, per_field_f1, per_field_edit_f1
```

**Tech stack:** `ultralytics` (YOLOv12 training+inference), `sahi` (sliced inference), `transformers` (TrOCR), `torch`, `Pillow`. Inference-time only — TrOCR is used pretrained without fine-tuning in the v1 (we add fine-tuning as a follow-up if needed).

---

## Hypothesis

Per Malepati 2026's published result on the same 837-image dataset: YOLOv12+SAHI lifts macro AP@0.5 on the four OCR-critical classes (batch_number, MRP, mfg_date, expiry_date) from 0.035 to 0.609 — a 17× gain. That establishes detection accuracy on small-text fields. With a competent per-region OCR head (TrOCR-base-printed pretrained on printed-document text), Track D should:

- Win on OCR-critical fields where Tracks A and B currently score 0.0 strict and ≤0.2 lenient.
- Underperform Tracks A/C on named-entity fields (brand_name, drug_name, generic_name) where end-to-end methods can leverage decoder context across the full image.

**Predicted ranges (best.pt eval, n=111 val):**

| Track | macro_f1 | macro_edit_f1 | OCR-critical avg | Named-entity avg |
|---|---:|---:|---:|---:|
| Track A vanilla (current best) | 0.0741 | 0.3225 | low | high |
| **Track D (predicted)** | 0.10–0.15 | 0.40–0.55 | **high** | medium |

**Null result threshold**: if Track D ties Track A within ±0.005 strict, count as null. The result is then "modular and end-to-end are statistically tied in this regime; modular adds latency without compensating gain." That's also publishable as the paper's "decision tree" empirical justification.

**Where Track D's expected wins are:**
- batch_number: A/B currently 0.0/0.0 strict, edit_f1 0.18/0.11. Track D w/ SAHI should hit 0.05+ strict and 0.4+ edit.
- MRP: A 0.0 strict, edit_f1 0.22. Track D should hit similar to mfg_date.
- expiry_date: A 0.0 strict, edit_f1 0.27. Same.
- mfg_date: A 0.0 strict, edit_f1 0.29. Same.

---

## File Structure

| Path | Created/Modified | Responsibility |
|---|---|---|
| `pyproject.toml` | Modify | Add `ultralytics>=8.3`, `sahi>=0.11` to `ml` extras. |
| `scripts/build_yolo_dataset.py` | Create | Convert `data/processed/{train,val,test}.jsonl` polygon annotations → YOLO-format bbox txt files in `data/yolo/{images,labels}/{train,val,test}/`. |
| `experiments/configs/yolo_baseline.yaml` | Create | Locked YOLOv12 training config (epochs, imgsz, batch, augmentation, model). |
| `train_yolo.py` | Create | Single-file YOLOv12 trainer (sibling of train.py / train_qwen.py / pretrain_mae.py). Uses `ultralytics` API. Prints `final_yolo_map50=X.XXXX` on stdout for autoresearch harness. |
| `track_d_pipeline.py` | Create | The Track D inference pipeline: SAHI tiling + YOLOv12 detect + per-region TrOCR + XML composition. Exposes a `TrackDWrapper` class with `predict_text(images, max_new_tokens) → list[str]` to plug into `prepare.evaluate`. |
| `experiments/configs/track_d.yaml` | Create | Track D inference config (YOLO weights path, SAHI params, TrOCR model name, per-class confidence thresholds). |
| `scripts/run_track_d.sh` | Create | Track D inference + eval launcher. Snapshot config, run track_d_pipeline through prepare.evaluate, emit metrics.json. Mirror run_experiment.sh shape but no training (eval-only). |
| `tests/test_track_d_components.py` | Create | Lightweight: yaml load + module imports + lazy-import discipline. |
| `tests/test_track_d_smoke.py` | Create | Colab-marked: load YOLO weights + TrOCR model + run on a single test image, verify output XML shape. |
| `experiments/runs/track_d-seed44/` | Output | Run dir: config.yaml + stdout.log + metrics.json + notes.md + per-image predictions JSON. |
| `experiments/per_field_track_d-seed44.json` | Output | Per-field eval results, same shape as other tracks. |
| `docs/superpowers/reviews/phase_8_review.md` | Create | End-of-phase review with the 5-track final comparison and the comparative paper conclusion. |

---

## Task 1: pyproject.toml deps + smoke import test

- [ ] **Step 1: Edit `pyproject.toml`** — add to `ml` extras after `qwen-vl-utils`:

```toml
    "ultralytics>=8.3",         # YOLOv12 training + inference
    "sahi>=0.11",               # Slicing-Aided Hyper Inference
```

- [ ] **Step 2: `uv sync --extra dev`** locally to confirm TOML parses.

- [ ] **Step 3: Commit** `chore(deps): ultralytics + sahi for Phase 8 Track D`.

---

## Task 2: Convert polygon annotations to YOLO bbox format

YOLOv12 expects each image to have a sibling `.txt` file with one line per bounding box: `class_id center_x center_y width height` (all normalized to [0,1]).

Our annotations live in `gold_standard.jsonl` with `polygon` coordinates in [0, 100] percentage. We need to:

1. Read `gold_standard.jsonl` + the train/val/test split jsons.
2. For each image, for each field's polygon, compute the axis-aligned bounding box (min_x, min_y, max_x, max_y) of the polygon, then convert to YOLO format (center_x, center_y, width, height all normalized to [0,1]).
3. Map field names to integer class IDs (consistent across splits): brand_name=0, drug_name=1, generic_name=2, ..., warnings=11.
4. Write the output to `data/yolo/labels/{train,val,test}/IMG_NNNN.txt`.
5. Symlink the corresponding image to `data/yolo/images/{train,val,test}/IMG_NNNN.jpeg`.
6. Generate the YOLO data.yaml: `path: data/yolo`, `train: images/train`, `val: images/val`, `test: images/test`, `names: [brand_name, drug_name, ...]`.

Lightweight tests:
- `test_polygon_to_bbox_correctness` — synthetic polygon, check bbox math.
- `test_yolo_dataset_built_correctly` — assert n_train + n_val + n_test image counts match jsonl, label files exist, classes are 0–11.

Commit: `feat(phase-8): build_yolo_dataset.py — polygon → YOLO bbox conversion + data.yaml`.

---

## Task 3: train_yolo.py + experiments/configs/yolo_baseline.yaml

YAML (`experiments/configs/yolo_baseline.yaml`):

```yaml
phase: 8
run_kind: yolo_train
model: "yolov12s.pt"   # ultralytics auto-downloads if not local
data: "data/yolo/data.yaml"
epochs: 100
imgsz: 640
batch: 16
optimizer: "AdamW"
lr0: 1.0e-3
seed: 44
device: 0
patience: 20            # early-stopping patience
amp: true               # automatic mixed precision (bf16)
augment: true
hsv_h: 0.015
hsv_s: 0.0              # NO color jitter (per Malepati 2026 reasoning + our text-aware MAE finding)
hsv_v: 0.0
fliplr: 0.5
project: "checkpoints/runs"
name: "yolo-baseline-seed44"
```

`train_yolo.py` is a thin wrapper around `ultralytics.YOLO`:

```python
"""Phase 8 — YOLOv12 training for Track D detection.

Single-file trainer (sibling of train.py / train_qwen.py / pretrain_mae.py).
Wraps ultralytics.YOLO. Prints final_yolo_map50=<float> on stdout for the
autoresearch harness.
"""
from __future__ import annotations
import argparse, dataclasses, sys, time
from pathlib import Path
import yaml

@dataclasses.dataclass
class YoloConfig:
    phase: int
    run_kind: str
    model: str
    data: str
    epochs: int
    imgsz: int
    batch: int
    optimizer: str
    lr0: float
    seed: int
    device: int
    patience: int
    amp: bool
    augment: bool
    hsv_h: float
    hsv_s: float
    hsv_v: float
    fliplr: float
    project: str
    name: str

    @classmethod
    def from_yaml(cls, path):
        return cls(**yaml.safe_load(Path(path).read_text()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/yolo_baseline.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = YoloConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg = dataclasses.replace(cfg, seed=args.seed)

    from ultralytics import YOLO
    model = YOLO(cfg.model)
    results = model.train(
        data=cfg.data,
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        optimizer=cfg.optimizer,
        lr0=cfg.lr0,
        seed=cfg.seed,
        device=cfg.device,
        patience=cfg.patience,
        amp=cfg.amp,
        augment=cfg.augment,
        hsv_h=cfg.hsv_h,
        hsv_s=cfg.hsv_s,
        hsv_v=cfg.hsv_v,
        fliplr=cfg.fliplr,
        project=cfg.project,
        name=cfg.name,
    )
    metrics = model.val(data=cfg.data)
    map50 = float(metrics.box.map50)
    print(f"final_yolo_map50={map50:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Lightweight tests: yaml-loads-config, module-imports-without-torch (ultralytics is heavy; verify lazy import).

Commit: `feat(phase-8): train_yolo.py + yolo_baseline.yaml`.

---

## Task 4: track_d_pipeline.py — SAHI + YOLO + TrOCR

The full inference pipeline. Single file, ~250 lines.

**Class structure:**

```python
class TrackDPipeline:
    def __init__(self, cfg: TrackDConfig):
        # Lazy load YOLO + TrOCR + SAHI
        ...

    def predict_text(self, images: torch.Tensor, max_new_tokens: int = 256) -> list[str]:
        """Same signature as PharmaVLM.predict_text and QwenWrapper.predict_text.
        Plugs into prepare.evaluate(...) directly.
        """
        outputs = []
        for img_tensor in images:
            # Reverse SigLIP normalization to PIL (same as Track A/B do)
            pil = self._tensor_to_pil(img_tensor)
            xml = self._predict_one(pil)
            outputs.append(xml)
        return outputs

    def _predict_one(self, pil_image: Image.Image) -> str:
        # 1. SAHI tiled detection
        detections = self._sahi_detect(pil_image)
        # detections = [(class_id, bbox, confidence), ...]
        
        # 2. NMS-merge across tiles
        # (SAHI does this internally; just collect results)
        
        # 3. Per-region OCR
        field_values = {}
        for class_id, bbox, conf in detections:
            field_name = CLASS_NAMES[class_id]
            crop = pil_image.crop(bbox)
            text = self._trocr_predict(crop)
            # If multiple detections for same field, pick highest confidence
            if field_name not in field_values or conf > field_values[field_name][1]:
                field_values[field_name] = (text, conf)
        
        # 4. Compose XML
        return self._compose_xml(field_values)

    def _sahi_detect(self, pil_image):
        from sahi.predict import get_sliced_prediction
        result = get_sliced_prediction(
            pil_image,
            self.yolo_detection_model,  # SAHI wrapper around YOLO
            slice_height=self.cfg.slice_size,
            slice_width=self.cfg.slice_size,
            overlap_height_ratio=self.cfg.overlap,
            overlap_width_ratio=self.cfg.overlap,
            postprocess_type="NMS",
        )
        return [(p.category.id, p.bbox.to_xyxy(), p.score.value) 
                for p in result.object_prediction_list]

    def _trocr_predict(self, pil_crop):
        # TrOCR-base-printed via transformers
        pixel_values = self.trocr_processor(pil_crop, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            generated_ids = self.trocr_model.generate(pixel_values, max_length=64)
        return self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _compose_xml(self, field_values):
        parts = ["<s>"]
        for field, (text, _conf) in field_values.items():
            parts.append(f"<{field}>{text}</{field}>")
        parts.append("</s>")
        return "".join(parts)
```

(One missing detail: TrOCR-base-printed expects 384×384 input or similar; the processor handles resize. Cropped regions from full-res 4032-px images can be very large or very narrow — we may need to resize-with-pad before TrOCR. Default behavior should handle it; verify.)

YAML (`experiments/configs/track_d.yaml`):

```yaml
phase: 8
run_kind: track_d_eval
yolo_weights: "checkpoints/runs/yolo-baseline-seed44/weights/best.pt"
trocr_model: "microsoft/trocr-base-printed"

# SAHI params (per Malepati 2026 recipe)
slice_size: 1024
overlap: 0.35
inference_imgsz: 1280
per_class_confidence: 0.15      # for the four OCR-critical classes
default_confidence: 0.25

# I/O (shared with Tracks A/B/C)
val_jsonl: "data/processed/val.jsonl"
test_jsonl: "data/processed/test.jsonl"
images_root: "/mnt/gcs/raw/raw_images"
path_strip_prefix: "raw/raw_images/"
checkpoint_dir: "checkpoints/runs/track_d-seed44"
```

Tests:
- Lightweight: yaml load, lazy imports, _compose_xml deterministic for fixed dict.
- Colab smoke: load weights + TrOCR + single end-to-end on test image.

Commit: `feat(phase-8): track_d_pipeline.py + track_d.yaml + tests`.

---

## Task 5: scripts/run_track_d.sh + integration with prepare.evaluate

Mirrors `run_experiment.sh` shape but evaluation-only (no training). Pseudo:

```bash
#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${1:-track_d-$(date -u +%s)}"
shift || true
CONFIG="experiments/configs/track_d.yaml"
SEED_ARG=""
while (($#)); do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --seed) SEED_ARG="--seed $2"; shift 2;;
    *) shift;;
  esac
done

RUN_DIR="experiments/runs/${RUN_ID}"
mkdir -p "$RUN_DIR"
cp track_d_pipeline.py "$RUN_DIR/"
cp "$CONFIG" "$RUN_DIR/config.yaml"

START_TS=$(date -u +%s)
PYTHONUNBUFFERED=1 uv run python -c "
from track_d_pipeline import TrackDPipeline, TrackDConfig
import prepare
import json

cfg = TrackDConfig.from_yaml('$CONFIG')
pipeline = TrackDPipeline(cfg)
metrics = prepare.evaluate(pipeline, cfg.val_jsonl, cfg.images_root, cfg.path_strip_prefix, batch_size=4, max_new_tokens=256)

print(f'final_macro_f1={metrics[\"macro_f1\"]:.4f}', flush=True)
print(f'final_macro_edit_f1={metrics[\"macro_edit_f1\"]:.4f}', flush=True)

import json
json.dump(metrics, open('$RUN_DIR/per_field_track_d.json', 'w'), indent=2)
" > "$RUN_DIR/stdout.log" 2>&1
EXIT_CODE=$?

END_TS=$(date -u +%s)
WALL_CLOCK=$((END_TS - START_TS))

FINAL_F1=$(grep -oE '^final_macro_f1=[0-9.]+' "$RUN_DIR/stdout.log" | tail -1 | cut -d= -f2)
FINAL_EDIT_F1=$(grep -oE '^final_macro_edit_f1=[0-9.]+' "$RUN_DIR/stdout.log" | tail -1 | cut -d= -f2)

cat > "$RUN_DIR/metrics.json" <<EOF
{
  "run_id": "$RUN_ID",
  "track": "D",
  "config": "$CONFIG",
  "final_macro_f1": $FINAL_F1,
  "final_macro_edit_f1": $FINAL_EDIT_F1,
  "wall_clock_seconds": $WALL_CLOCK,
  "exit_code": $EXIT_CODE
}
EOF

# Auto-finalize (mirrors run_experiment.sh F1 patch)
if [[ $EXIT_CODE -eq 0 ]]; then
  bash scripts/finalize_experiment.sh "$RUN_ID" --phase baseline || true
fi
```

Commit: `feat(phase-8): scripts/run_track_d.sh — Track D eval launcher`.

---

## Task 6: Run YOLO training on Colab (~6h on A100)

Pre-write `experiments/runs/yolo-baseline-seed44/notes.md` (TODO: hypothesis paragraph + Malepati 2026 citation).

```bash
cd /workspace
export LD_LIBRARY_PATH=/usr/lib64-nvidia
nohup bash scripts/run_yolo_train.sh yolo-baseline-seed44 --seed 44 \
  > /tmp/yolo-train.log 2>&1 &
```

Wait for completion (~6h on A100). YOLO weights land at `checkpoints/runs/yolo-baseline-seed44/weights/best.pt`. Auto-push to GCS via existing sync_to_gcs daemon.

---

## Task 7: Run Track D inference + eval

```bash
cd /workspace
nohup bash scripts/run_track_d.sh track_d-seed44 --seed 44 \
  > /tmp/track-d.log 2>&1 &
```

Per-image inference: SAHI tiling + YOLO detection + TrOCR per region. ~30s per image × 111 val = ~55 min. Auto-finalize writes ledger entry + per_field json.

---

## Task 8: Per-field comparison + 5-track figures

`uv run python scripts/per_field_eval.py` extension for Track D (or use the per_field_track_d.json directly written by Task 7).

Generate F2c (5-track per-field bar charts) using existing `scripts/plots/per_field_bar.py`:

```bash
uv run python scripts/plots/per_field_bar.py \
  --inputs experiments/per_field_track_a.json \
           experiments/per_field_baseline_mae_init-seed44.json \
           experiments/per_field_baseline_mae_text_aware_init-seed44.json \
           experiments/per_field_baseline_mae_tapt_init-seed44.json \
           experiments/per_field_track_d-seed44.json \
  --labels "A: Vanilla SigLIP" "C-DAPT" "C-text-aware" "C-TAPT" "D: YOLO+SAHI+TrOCR" \
  --metric per_field_f1 --out paper/figures/F2c_5track_strict.png
```

Commit: `exp(phase-8): Track D run + 5-track per-field comparison + F2c figure`.

---

## Task 9: Update paper main.md + Phase 8 review

Update `paper/main.md`:
- Section IV-A Table 1: fill in Track D row with numbers.
- Section IV-E (decision tree): final concrete recommendation.
- Section V conclusion: write.

Create `docs/superpowers/reviews/phase_8_review.md`:
- What was built (Tasks 1-8).
- Track D headline numbers.
- Per-field win analysis (which fields each track wins on).
- Decision tree by data-size regime (final).
- Tag suggestion: `phase-8-track-d-complete`.

Commit: `docs(phase-8): track-d results + paper section IV/V updates`.

---

## Self-Review

| Spec / framing requirement | Implemented in task |
|---|---|
| YOLOv12 detection (Malepati 2026 base) | T2-T3 |
| SAHI tiled inference (Malepati 2026 17× lift recipe) | T4 |
| Per-region OCR (TrOCR pretrained) | T4 |
| End-to-end pipeline plugged into prepare.evaluate | T4-T5 |
| Same val split, same metric harness as A/B/C | T5 |
| 5-track comparison figure | T8 |
| Paper Table 1 + Section V finalization | T9 |

Total compute budget: ~6h YOLO training + ~1h Track D inference + ~1h analysis = **~8h A100 time** (~43 compute units on Pro+; user has 1634+ left).

Code time: ~6 hours of subagent work.

Citations:
- **Malepati, Nandamury, Manjunath, Rajan, Prabhune (2026)** [malepati2026sahi] — same dataset, YOLOv12+SAHI baseline.
- **Akyon, Altinuc, Temizel (2022)** [akyon2022sahi] — SAHI primary reference.
- **Tian, Ye, Doermann (2025)** [tian2025yolov12] — YOLOv12 architecture.
- **Li et al. (2021)** "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models" [arXiv:2109.10282] — per-region text recognizer (need to add to papers/README.md).

---

## Risks and mitigations

1. **TrOCR may be too slow at 30s/image × 111 val = ~55 min.** If too slow on the live tunnel, run async + GCS push for resilience.
2. **TrOCR base-printed may not generalize to pharma label fonts.** v1 uses pretrained; if macro_edit_f1 is much lower than expected, fine-tune TrOCR on cropped pharma label regions in v2 (Phase 8b).
3. **YOLO bbox-from-polygon conversion might lose precision.** Polygons are tight around text; axis-aligned bboxes are looser. Acceptable approximation; Malepati 2026 uses the same conversion implicitly.
4. **Per-class confidence thresholds** matter for SAHI false-positive control. Malepati 2026 uses 0.15 for OCR-critical classes, 0.25 for others. We follow that.
5. **Multiple detections per field per image** — pick the highest-confidence one, or aggregate values. v1 uses highest-confidence; if lenient F1 is hurt, try first-detection or top-2 concatenation in v2.
