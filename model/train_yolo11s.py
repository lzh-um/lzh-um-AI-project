#!/usr/bin/env python3
"""Minimal YOLO11s training script matching the Ultralytics docs example.

This script follows Ultralytics official best practices for optimal performance.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path(__file__).resolve().parent / "ultralytics/cfg/datasets/UAV-Benchmark-M.yaml"


def main() -> None:
    """Train YOLO11s model with optimized settings."""
    data_path = DATA_YAML.expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    # Load pretrained model (recommended for better performance)
    model = YOLO("yolo11s.pt")
    
    # Training with performance optimizations
    model.train(
        # Data settings
        data=str(data_path),
        
        # Training duration
        epochs=30,
        patience=10,  # Early stopping if no improvement for 50 epochs
        
        # Batch and image settings
        batch=32,  # Reduced for 1280 image size to avoid OOM
        imgsz=640,
        rect=True,  # Rectangular training for faster processing with large images
        
        # Performance optimizations
        cache="disk",  # Cache images on disk to save RAM (server RAM is full)
        workers=8,  # Adjust based on CPU cores (typically num_cores - 2)
        amp=False,  # Mixed precision training for faster computation
        
        # Learning rate settings
        lr0=0.01,
        lrf=0.01,  # Final learning rate = lr0 * lrf
        cos_lr=True,  # Cosine learning rate scheduler
        
        # Augmentation settings
        close_mosaic=10,  # Disable mosaic augmentation in last 10 epochs
        
        # Device and output
        device="0",  # use "cpu" or "0,1" if needed
        project="runs/train",
        name="uav-benchmark-m-yolo11s",
        exist_ok=False,
        
        # Visualization and monitoring
        plots=True,  # Generate training plots
        save=True,
        save_period=-1,  # Save checkpoint every N epochs (-1 = only save last and best)
        
        # Reproducibility
        deterministic=True,
        seed=0,
    )


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
