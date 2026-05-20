#!/usr/bin/env python3
"""Extract validation results from all trained models and save to a summary CSV."""

import pandas as pd
from pathlib import Path

# Define the base directory
BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "runs/train"

# Model directories
MODEL_DIRS = [
    "uav-benchmark-m-rtdetr-l",
    "uav-benchmark-m-yolo11s",
    "uav-benchmark-m-yolov8-dbra",
    "uav-benchmark-m-yolov8n5",
    "uav-benchmark-m-yolov8s6",
]

# Model display names
MODEL_NAMES = {
    "uav-benchmark-m-rtdetr-l": "RT-DETR-L",
    "uav-benchmark-m-yolo11s": "YOLO11s",
    "uav-benchmark-m-yolov8-dbra": "YOLOv8s-DBRA",
    "uav-benchmark-m-yolov8n5": "YOLOv8n",
    "uav-benchmark-m-yolov8s6": "YOLOv8s",
}


def main():
    results = []
    
    for model_dir in MODEL_DIRS:
        csv_path = TRAIN_DIR / model_dir / "results.csv"
        
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping...")
            continue
        
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Remove empty rows
        df = df.dropna(how='all')
        
        # Clean column names (strip whitespace)
        df.columns = [c.strip() for c in df.columns]

        # Select row based on model
        if model_dir == "uav-benchmark-m-yolov8-dbra":
            # For YOLOv8-DBRA, user specifically requested epoch 20
            # Note: epoch column values might be integers or strings
            target_epoch = 20
            row = df[df['epoch'] == target_epoch]
            if not row.empty:
                last_row = row.iloc[0]
            else:
                print(f"Warning: Epoch {target_epoch} not found for {model_name}, using last epoch.")
                last_row = df.iloc[-1]
        else:
            # Default: Get the last row (final epoch results)
            last_row = df.iloc[-1]
        
        # Extract validation metrics
        model_name = MODEL_NAMES.get(model_dir, model_dir)
        
        result = {
            "Model": model_name,
            "Epochs": int(last_row.get("epoch", 0)),
            "Precision": round(last_row.get("metrics/precision(B)", 0), 4),
            "Recall": round(last_row.get("metrics/recall(B)", 0), 4),
            "mAP50": round(last_row.get("metrics/mAP50(B)", 0), 4),
            "mAP50-95": round(last_row.get("metrics/mAP50-95(B)", 0), 4),
        }
        
        # Add validation losses if available
        if "val/box_loss" in last_row:
            result["Val_Box_Loss"] = round(last_row["val/box_loss"], 4)
        if "val/cls_loss" in last_row:
            result["Val_Cls_Loss"] = round(last_row["val/cls_loss"], 4)
        
        results.append(result)
        print(f"Extracted: {model_name} - mAP50: {result['mAP50']:.4f}, mAP50-95: {result['mAP50-95']:.4f}")
    
    # Create DataFrame and save
    results_df = pd.DataFrame(results)
    
    # Sort by mAP50-95 descending
    results_df = results_df.sort_values("mAP50-95", ascending=False)
    
    # Save to CSV
    output_path = BASE_DIR / "validation_results_summary.csv"
    results_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*70}")
    print("Validation Results Summary")
    print('='*70)
    print(results_df.to_string(index=False))
    print('='*70)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
