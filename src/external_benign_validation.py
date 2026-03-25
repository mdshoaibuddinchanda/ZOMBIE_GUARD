"""
External Benign Validation — Test detector on independent benign ZIP corpus.

This script provides credibility by testing the trained LightGBM detector
on publicly-available benign ZIPs sourced from GitHub, PyPI, and other
open repositories — ensuring generalization beyond the training distribution.

Output:
  CSV: paper/figures/csv/table_external_benign_validation.csv
    - Columns: source, file_path, predictions (0=benign, 1=malicious), false_positives
"""

import os
import sys
import joblib
import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extractor import extract_features
from src.classifier import FEATURE_COLS


def load_model(model_path):
    """Load trained LightGBM model."""
    return joblib.load(model_path)


def test_external_benign_corpus(model_path, benign_zip_dir, output_csv=None):
    """
    Test detector on external benign ZIP corpus.
    
    Args:
        model_path (str): Path to trained LightGBM model
        benign_zip_dir (str): Directory containing independent benign ZIPs
        output_csv (str): Output CSV path (default: paper/figures/csv/table_external_benign_validation.csv)
    
    Returns:
        dict: Summary stats (false_positives, true_negatives, fp_rate, sample_count)
    """
    
    if output_csv is None:
        output_csv = "paper/figures/csv/table_external_benign_validation.csv"
    
    print("\n" + "="*70)
    print("External Benign Validation — Independent ZIP Corpus")
    print("="*70)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = load_model(model_path)
    print(f"  ✓ Model loaded (LightGBM, num_leaves={model.num_leaves})")
    
    # Find all ZIP files
    benign_zip_dir = Path(benign_zip_dir)
    if not benign_zip_dir.exists():
        print(f"\n Error: Directory not found: {benign_zip_dir}")
        print("\nNote: This directory should contain independent benign ZIPs")
        print("  from public sources (GitHub projects, PyPI, etc.)")
        print("\n  To add benign corpus:")
        print("    1. Clone small Python projects from GitHub")
        print("    2. Extract their .zip archives into data/external_benign/")
        print("    3. Re-run this script")
        return None
    
    # Collect all ZIP files
    zip_files = list(benign_zip_dir.rglob("*.zip"))
    
    if not zip_files:
        print(f"\n  No ZIP files found in {benign_zip_dir}")
        print("   Skipping validation (directory is empty or no .zip files)")
        return None
    
    print(f"\n  ✓ Found {len(zip_files)} benign ZIP files")
    
    # Test each file
    results = []
    false_positives = 0
    true_negatives = 0
    errors = 0
    
    print(f"\nTesting {len(zip_files)} external benign files...")
    
    for idx, zip_path in enumerate(zip_files, 1):
        try:
            # Extract features
            features_dict = extract_features(str(zip_path))
            
            # Convert to DataFrame row
            features_df = pd.DataFrame([features_dict])
            
            # Filter to required columns only
            if not all(col in features_df.columns for col in FEATURE_COLS):
                raise ValueError(f"Missing columns. Expected {FEATURE_COLS}, got {list(features_df.columns)}")
            
            features_df = features_df[FEATURE_COLS]
            
            # Predict
            prediction = model.predict(features_df)[0]
            prediction_proba = model.predict_proba(features_df)[0]
            
            # Record result
            results.append({
                'file_path': str(zip_path.relative_to(benign_zip_dir)),
                'file_size_kb': zip_path.stat().st_size / 1024,
                'prediction': int(prediction),
                'prob_benign': prediction_proba[0],
                'prob_malicious': prediction_proba[1],
                'is_false_positive': 1 if prediction == 1 else 0
            })
            
            if prediction == 1:
                false_positives += 1
            else:
                true_negatives += 1
            
            # Progress
            if idx % 10 == 0:
                print(f"  [{idx}/{len(zip_files)}] Processed... (FP so far: {false_positives})")
        
        except Exception as e:
            errors += 1
            results.append({
                'file_path': str(zip_path.relative_to(benign_zip_dir)),
                'file_size_kb': None,
                'prediction': None,
                'prob_benign': None,
                'prob_malicious': None,
                'is_false_positive': None,
                'error': str(e)
            })
    
    # Compute stats
    valid_results = len(zip_files) - errors
    fp_rate = (false_positives / valid_results * 100) if valid_results > 0 else 0
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    
    print(f"\n  ✓ Processed {valid_results}/{len(zip_files)} files")
    print(f"  ✓ Errors: {errors}")
    
    # Summary
    print("\n" + "-"*70)
    print("EXTERNAL VALIDATION RESULTS")
    print("-"*70)
    print(f"Total benign files tested:     {valid_results}")
    print(f"True negatives (correctly benign):  {true_negatives}")
    print(f"False positives (incorrectly flagged): {false_positives}")
    print(f"False positive rate:           {fp_rate:.2f}%")
    print(f"\n  Saved: {output_csv}")
    
    if false_positives == 0:
        print(f"\n PASS: No false positives on {valid_results} independent benign samples")
    else:
        print(f"\n  WARNING: {false_positives} false positives on {valid_results} samples")
        print("\n  False positive files:")
        fp_files = results_df[results_df['is_false_positive'] == 1]
        for _, row in fp_files.iterrows():
            print(f"    - {row['file_path']} (prob_mal={row['prob_malicious']:.4f})")
    
    return {
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'total_tested': valid_results,
        'fp_rate': fp_rate,
        'errors': errors,
        'output_csv': output_csv
    }


if __name__ == '__main__':
    
    # Setup paths
    repo_root = Path(__file__).parent.parent
    model_path = repo_root / "models" / "lgbm_model.pkl"
    external_benign_dir = repo_root / "data" / "external_benign"
    
    # Run validation
    results = test_external_benign_corpus(
        model_path=str(model_path),
        benign_zip_dir=str(external_benign_dir),
        output_csv=str(repo_root / "paper" / "figures" / "csv" / "table_external_benign_validation.csv")
    )
    
    if results is None:
        print("\n  Validation skipped (no external benign corpus found)")
        print("\nTo conduct external validation:")
        print("  1. Create directory: data/external_benign/")
        print("  2. Add ~50-100 public benign ZIPs (GitHub projects, PyPI pkgs, etc.)")
        print("  3. Re-run: python src/external_benign_validation.py")
        sys.exit(0)
    
    # Exit code reflects pass/fail
    sys.exit(0 if results['false_positives'] == 0 else 1)
