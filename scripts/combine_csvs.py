#!/usr/bin/env python
"""
Combine all CSV result files into a single master CSV file.
This script reads all 21 CSV files and consolidates them into one master file
with a 'source_table' column indicating where each data came from.
"""

import os
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CSV directory containing all result tables
CSV_DIR = Path("paper/figures/csv")
OUTPUT_FILE = Path("paper/figures/csv/MASTER_RESULTS_COMBINED.csv")

# List of all CSV files to combine
CSV_FILES = [
    "table1_baseline_comparison.csv",
    "table5_feature_ablation.csv",
    "table6_multi_baseline_comparison.csv",
    "table6b_multi_baseline_hard_test.csv",
    "table7_variant_recall.csv",
    "table8_temporal_stability.csv",
    "table8b_shap_stability.csv",
    "table_roc_pr_auc.csv",
    "table_entropy_stats.csv",
    "table_family_prevalence.csv",
    "table_fn_analysis.csv",
    "table_adversarial_results.csv",
    "adversarial_full_results.csv",
    "generalisation_results.csv",
    "hard_test_comparison.csv",
    "three_model_comparison.csv",
    "table_external_benign_validation.csv",
    "table_leave_one_family_out.csv",
    "table_real_only_ablation.csv",
    "table_synthetic_train_real_test.csv",
    "table_synthetic_real_feature_alignment.csv",
]


def combine_csvs():
    """Combine all CSV files into a single master CSV."""
    
    print("="*70)
    print("COMBINING ALL CSV RESULT FILES")
    print("="*70)
    
    combined_data = []
    
    for csv_file in CSV_FILES:
        csv_path = CSV_DIR / csv_file
        
        if not csv_path.exists():
            print(f"  ⚠️  Missing: {csv_file}")
            continue
        
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Add source table column
            df['source_table'] = csv_file.replace('.csv', '')
            
            # Add to combined list
            combined_data.append(df)
            
            print(f"  ✓ {csv_file:<50} ({len(df):>5} rows)")
        
        except Exception as e:
            print(f"  ✗ {csv_file:<50} ERROR: {str(e)}")
    
    if not combined_data:
        print("\n  ERROR: No CSV files found to combine!")
        return False
    
    # Combine all DataFrames
    print(f"\nCombining {len(combined_data)} CSV files...")
    
    # Use pd.concat with ignore_index to handle varying columns
    master_df = pd.concat(combined_data, ignore_index=True, sort=False)
    
    # Reorder columns: source_table first, then others
    cols = master_df.columns.tolist()
    cols.remove('source_table')
    master_df = master_df[['source_table'] + cols]
    
    # Save combined CSV
    master_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✓ COMBINED CSV CREATED")
    print(f"  Location: {OUTPUT_FILE}")
    print(f"  Total rows: {len(master_df)}")
    print(f"  Total columns: {len(master_df.columns)}")
    print(f"  File size: {OUTPUT_FILE.stat().st_size / 1024:.2f} KB")
    
    # Print summary statistics
    print(f"\n  Source tables included:")
    for source in sorted(master_df['source_table'].unique()):
        count = len(master_df[master_df['source_table'] == source])
        print(f"    - {source:<45} {count:>5} rows")
    
    print("\n" + "="*70)
    return True


if __name__ == "__main__":
    success = combine_csvs()
    sys.exit(0 if success else 1)
