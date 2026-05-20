#!/usr/bin/env python3
"""Summarize tracking results from multiple experiments."""

import pandas as pd
from pathlib import Path

# Define base directory
BASE_DIR = Path(__file__).resolve().parent

# Find all tracking result directories
TRACKING_DIRS = list(BASE_DIR.glob("mot_results_*"))

def main():
    summary_data = []

    for tracking_dir in TRACKING_DIRS:
        if not tracking_dir.is_dir():
            continue
        
        result_csv = tracking_dir / "evaluation_results.csv"
        if not result_csv.exists():
            print(f"Warning: {result_csv} not found, skipping...")
            continue
            
        try:
            # Read the CSV
            df = pd.read_csv(result_csv)
            
            # Find the OVERALL row
            overall_row = df[df['Sequence'] == 'OVERALL']
            
            if overall_row.empty:
                print(f"Warning: 'OVERALL' row not found in {result_csv}")
                continue
                
            # Extract metrics
            row = overall_row.iloc[0]
            
            # Clean up model name from directory name
            model_name = tracking_dir.name.replace("mot_results_", "")
            
            summary_data.append({
                "Model": model_name,
                "MOTA": row.get('MOTA', 'N/A'),
                "MOTP": row.get('MOTP', 'N/A'),
                "IDF1": row.get('IDF1', 'N/A'),
                "IDs": row.get('IDs', 'N/A'),
                "FM": row.get('FM', 'N/A'),
                "Precision": row.get('Precision', 'N/A'),
                "Recall": row.get('Recall', 'N/A')
            })
            
        except Exception as e:
            print(f"Error processing {result_csv}: {e}")

    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by MOTA if possible (need to convert string percentage to float for sorting)
    try:
        summary_df['MOTA_val'] = summary_df['MOTA'].str.rstrip('%').astype(float)
        summary_df = summary_df.sort_values('MOTA_val', ascending=False)
        summary_df = summary_df.drop(columns=['MOTA_val'])
    except Exception:
        pass # If conversion fails, just keep original order

    # Save to CSV
    output_path = BASE_DIR / "tracking_results_summary.csv"
    summary_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print("Tracking Results Summary")
    print('='*80)
    print(summary_df.to_string(index=False))
    print('='*80)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
