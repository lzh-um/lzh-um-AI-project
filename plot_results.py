
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results(csv_path, output_path):
    # Read the results CSV
    # The file has a header with spaces, we need to strip them
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    epochs = df['epoch']
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot mAP
    ax1.plot(epochs, df['metrics/mAP50(B)'], label='mAP50', linewidth=2)
    ax1.plot(epochs, df['metrics/mAP50-95(B)'], label='mAP50-95', linewidth=2)
    ax1.set_title('mAP Metrics over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('mAP')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Losses
    ax2.plot(epochs, df['train/box_loss'], label='Train Box Loss', linestyle='--')
    ax2.plot(epochs, df['val/box_loss'], label='Val Box Loss')
    ax2.plot(epochs, df['train/cls_loss'], label='Train Cls Loss', linestyle='--')
    ax2.plot(epochs, df['val/cls_loss'], label='Val Cls Loss')
    ax2.set_title('Training and Validation Losses')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    csv_file = "/mnt/gemlab_data_3/User_database/xiejiefeng/PT/1600/runs/train/uav-benchmark-m-yolov8-dbra-v25/results.csv"
    output_png = "/home/sun/.gemini/antigravity/brain/4f2af80c-cbc9-4e5c-ada0-63c24f4e2c58/training_results.png"
    plot_results(csv_file, output_png)
