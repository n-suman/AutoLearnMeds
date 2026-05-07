"""Phase 8 - Track D: YOLOv12 + SAHI + TrOCR pipeline.

Modular detection-then-extract: YOLOv12 (with Slicing-Aided Hyper Inference)
detects field-bearing regions in pharma label photos; TrOCR-base-printed reads
each cropped region; results compose into the canonical XML.

Sibling tracks (A, B, C) all expose the same predict_text(images, max_new_tokens)
-> list[str] interface; TrackDPipeline does too, so it slots into the shared
prepare.evaluate harness with no changes.

Heavy ML imports (torch / ultralytics / sahi / transformers / PIL) are
deliberately deferred to inside the methods that use them, so this module
imports cleanly in the Mac dev venv (no ml extras) for unit tests and
config-load smoke tests.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml


# === Class names (must match the order in the YOLO data.yaml) ===
FIELD_NAMES = [
    "brand_name", "drug_name", "generic_name", "strength", "quantity",
    "company", "manufacturer", "batch_number", "mfg_date", "expiry_date",
    "mrp", "warnings",
]


@dataclasses.dataclass
class TrackDConfig:
    """Inference + eval config for Track D."""
    phase: int
    run_kind: str
    yolo_weights: str
    trocr_model: str
    slice_size: int
    overlap: float
    inference_imgsz: int
    ocr_critical_classes: tuple[str, ...]
    per_class_confidence: float
    default_confidence: float
    val_jsonl: str
    test_jsonl: str
    images_root: str
    path_strip_prefix: str
    checkpoint_dir: str

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrackDConfig":
        data = yaml.safe_load(Path(path).read_text())
        if isinstance(data.get("ocr_critical_classes"), list):
            data["ocr_critical_classes"] = tuple(data["ocr_critical_classes"])
        return cls(**data)


class TrackDPipeline:
    """Wraps YOLOv12 + SAHI + TrOCR.

    Exposes predict_text(images, max_new_tokens) -> list[str] - same interface
    as PharmaVLM.predict_text and QwenWrapper.predict_text, so it plugs into
    prepare-evaluate(...) without harness changes.
    """

    def __init__(self, cfg: TrackDConfig) -> None:
        self.cfg = cfg
        self._device = None
        self._yolo = None
        self._sahi_detection_model = None
        self._trocr_processor = None
        self._trocr_model = None

    def _lazy_init(self) -> None:
        """Lazy-load all ML models on first call."""
        if self._yolo is not None:
            return
        import torch
        from ultralytics import YOLO
        from sahi import AutoDetectionModel
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Raw YOLO model (used for .parameters() and as a sanity-loadable handle).
        self._yolo = YOLO(self.cfg.yolo_weights)

        # SAHI auto-detection model wraps YOLO with tiling.
        # ultralytics YOLOv12 uses the yolov8 backbone API in SAHI.
        self._sahi_detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=self.cfg.yolo_weights,
            confidence_threshold=self.cfg.default_confidence,
            device=str(self._device),
        )

        # TrOCR-base-printed (pretrained, no fine-tune in v1).
        self._trocr_processor = TrOCRProcessor.from_pretrained(self.cfg.trocr_model)
        self._trocr_model = VisionEncoderDecoderModel.from_pretrained(self.cfg.trocr_model).to(self._device)
        self._trocr_model.eval()

    # === Wrapper methods so the eval harness doesn't choke on the model interface. ===

    def parameters(self):
        """Return YOLO params; eval harness only inspects this for device/dtype."""
        self._lazy_init()
        return self._yolo.parameters() if hasattr(self._yolo, "parameters") else iter([])

    def to(self, device):
        # No-op: SAHI/YOLO/TrOCR all already on the right device after _lazy_init.
        return self

    def train(self, mode: bool = True):
        # No-op: Track D is inference-only.
        return self

    def predict_text(self, images, max_new_tokens: int = 256) -> list[str]:
        """Per-image: SAHI detect -> TrOCR per region -> compose XML.

        Args:
            images: torch.Tensor of shape [B, 3, H, W], SigLIP-normalized to [-1, 1].
            max_new_tokens: unused (TrOCR uses its own max_length=64).

        Returns:
            List of XML strings, one per image, in the canonical Track-A format.
        """
        import torch
        from PIL import Image

        self._lazy_init()
        if images.dtype != torch.float32:
            images = images.float()

        outputs: list[str] = []
        for i in range(images.shape[0]):
            # Reverse SigLIP normalization: [-1, 1] -> [0, 255] uint8 PIL.
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
            pil = Image.fromarray(arr)
            outputs.append(self._predict_one(pil))
        return outputs

    def _predict_one(self, pil_image) -> str:
        """SAHI tiled detection -> per-field highest-confidence OCR -> XML."""
        from sahi.predict import get_sliced_prediction

        result = get_sliced_prediction(
            pil_image,
            self._sahi_detection_model,
            slice_height=self.cfg.slice_size,
            slice_width=self.cfg.slice_size,
            overlap_height_ratio=self.cfg.overlap,
            overlap_width_ratio=self.cfg.overlap,
            postprocess_type="NMS",
            verbose=0,
        )

        # Per-field: keep the highest-confidence detection that passes its threshold.
        # SAHI returns object_prediction_list; each element exposes .category.id,
        # .bbox.minx/miny/maxx/maxy, .score.value.
        field_values: dict[str, tuple[str, float]] = {}
        for pred in result.object_prediction_list:
            class_id = pred.category.id
            if class_id < 0 or class_id >= len(FIELD_NAMES):
                continue
            field_name = FIELD_NAMES[class_id]
            confidence = pred.score.value

            # Per-class confidence threshold (Malepati 2026 recipe).
            threshold = (
                self.cfg.per_class_confidence
                if field_name in self.cfg.ocr_critical_classes
                else self.cfg.default_confidence
            )
            if confidence < threshold:
                continue

            bbox = (pred.bbox.minx, pred.bbox.miny, pred.bbox.maxx, pred.bbox.maxy)
            crop = pil_image.crop(bbox)
            text = self._trocr_predict(crop)

            if field_name not in field_values or confidence > field_values[field_name][1]:
                field_values[field_name] = (text, confidence)

        return self._compose_xml({k: v[0] for k, v in field_values.items()})

    def _trocr_predict(self, pil_crop) -> str:
        """Run TrOCR-base-printed on a single cropped region."""
        import torch
        # Skip empty/tiny crops that cause TrOCR errors.
        if pil_crop.size[0] < 4 or pil_crop.size[1] < 4:
            return ""
        pixel_values = self._trocr_processor(pil_crop, return_tensors="pt").pixel_values.to(self._device)
        with torch.no_grad():
            generated_ids = self._trocr_model.generate(pixel_values, max_length=64)
        decoded = self._trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded[0] if decoded else ""

    @staticmethod
    def _compose_xml(field_values: dict[str, str]) -> str:
        """Compose the canonical Track-A-style XML output from a field-text dict.

        Fields are emitted in FIELD_NAMES order so the output is deterministic.
        Unknown or empty-value fields are omitted.
        """
        parts = ["<s>"]
        for name in FIELD_NAMES:
            if name in field_values and field_values[name].strip():
                parts.append(f"<{name}>{field_values[name]}</{name}>")
        parts.append("</s>")
        return "".join(parts)


def main(argv: list[str] | None = None) -> int:
    """CLI for standalone eval (typically called via run_track_d.sh)."""
    parser = argparse.ArgumentParser(description="Phase 8 - Track D inference + eval.")
    parser.add_argument("--config", default="experiments/configs/track_d.yaml")
    parser.add_argument("--seed", type=int, default=None, help="For ledger consistency; unused.")
    args = parser.parse_args(argv)

    cfg = TrackDConfig.from_yaml(args.config)
    print(f"[track-d] config={args.config} yolo_weights={cfg.yolo_weights} trocr={cfg.trocr_model}", flush=True)

    pipeline = TrackDPipeline(cfg)

    # Run eval through the same harness as Tracks A/B/C.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import prepare

    start = time.time()
    metrics = prepare.evaluate(
        pipeline,
        cfg.val_jsonl,
        cfg.images_root,
        cfg.path_strip_prefix,
        batch_size=4,
        max_new_tokens=256,
    )
    wall = time.time() - start
    print(f"[track-d] eval done wall={wall:.1f}s", flush=True)
    print(f"final_macro_f1={metrics['macro_f1']:.4f}", flush=True)
    print(f"final_macro_edit_f1={metrics['macro_edit_f1']:.4f}", flush=True)

    out = Path(cfg.checkpoint_dir).parent.parent / "per_field_track_d-seed44.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"[track-d] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
