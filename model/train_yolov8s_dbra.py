#!/usr/bin/env python3
"""Train YOLOv8s-DBRA v2 (with Relative Position Encoding).

This script trains the enhanced YOLOv8s-DBRA model with relative position encoding
on the UAV-Benchmark-M dataset.

Improvements over v1:
- Added relative position bias (Swin Transformer style) to DBRA modules
- Better spatial awareness for attention mechanism
- Expected to improve mAP performance
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path(__file__).resolve().parent / "ultralytics/cfg/datasets/UAV-Benchmark-M.yaml"
MODEL_CFG = Path(__file__).resolve().parent / "ultralytics/cfg/models/v8/yolov8-dbra.yaml"


def main() -> None:
    """Train YOLOv8s-DBRA v2 model with relative position encoding."""
    data_path = DATA_YAML.expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    model_path = MODEL_CFG.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model YAML not found: {model_path}")

    print("=" * 60)
    print("YOLOv8s-DBRA v2 Training (with Relative Position Encoding)")
    print("=" * 60)
    print(f"Model config: {model_path}")
    print(f"Dataset: {data_path}")
    print("=" * 60)

    model = YOLO(str(model_path))
    
    # Training with mostly default parameters
    model.train(
        data=str(data_path),
        epochs=100,
        patience=20,
        batch=16,
        imgsz=640,
        workers=8,
        device="4",
        amp=False,
        project="runs/train",
        name="uav-benchmark-m-yolov8-dbra",
        plots=True,
        save=True,
        seed=42,
    )


if __name__ == "__main__":
    main()
