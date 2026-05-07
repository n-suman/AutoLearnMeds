"""Phase 8 — YOLOv12 training for Track D detection.

Single-file trainer (sibling of train.py / train_qwen.py / pretrain_mae.py).
Wraps ultralytics.YOLO. Prints final_yolo_map50=<float> and
final_yolo_map5095=<float> on stdout for the autoresearch harness.

Heavy ML imports (ultralytics / torch) are deferred to inside main() so this
module imports cleanly in the Mac dev venv (no ml extras) for unit tests and
config-load smoke tests.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import yaml


@dataclasses.dataclass
class YoloConfig:
    """Hyperparameters for one YOLOv12 training run.

    Loaded from a YAML in experiments/configs/. Mirrors the schema of
    experiments/configs/yolo_baseline.yaml exactly.
    """
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
    def from_yaml(cls, path: str | Path) -> "YoloConfig":
        return cls(**yaml.safe_load(Path(path).read_text()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 8 — YOLOv12 trainer (Track D detection).")
    parser.add_argument(
        "--config",
        default="experiments/configs/yolo_baseline.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional override for cfg.seed.")
    args = parser.parse_args(argv)

    cfg = YoloConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg = dataclasses.replace(cfg, seed=args.seed)

    print(f"[track-d-yolo] config={args.config} model={cfg.model} seed={cfg.seed}", flush=True)

    # Lazy import: ultralytics pulls torch transitively, so keep it inside main().
    from ultralytics import YOLO

    start = time.time()
    model = YOLO(cfg.model)
    model.train(
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
    map5095 = float(metrics.box.map)
    wall = time.time() - start
    print(f"[track-d-yolo] DONE wall={wall:.1f}s map50={map50:.4f} map50-95={map5095:.4f}", flush=True)
    print(f"final_yolo_map50={map50:.4f}", flush=True)
    print(f"final_yolo_map5095={map5095:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
