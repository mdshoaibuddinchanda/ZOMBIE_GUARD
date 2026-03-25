#!/usr/bin/env python
"""
Combine all PDF figure files into a single master PDF.
This script reads all 16 PDF files and merges them into one master PDF.
Requires: PyPDF2 or pypdf
"""

import os
import sys
from pathlib import Path

# Try to import pdf merger (different versions of the library)
try:
    from PyPDF2 import PdfMerger
except ImportError:
    try:
        from pypdf import PdfMerger
    except ImportError:
        print("ERROR: PyPDF2 or pypdf not installed.")
        print("Install with: uv pip install PyPDF2")
        sys.exit(1)

# PDF directory containing all figure PDFs
PDF_DIR = Path("paper/figures/pdf")
OUTPUT_FILE = Path("paper/figures/MASTER_ALL_FIGURES_COMBINED.pdf")

# List of all PDF files in order
PDF_FILES = [
    "fig1_zip_header_mismatch.pdf",
    "fig2_attack_taxonomy.pdf",
    "fig3_shap_importance.pdf",
    "fig4_generalisation_chart.pdf",
    "fig5_multi_baseline_chart.pdf",
    "fig5b_multi_baseline_hard_chart.pdf",
    "fig6_variant_recall_chart.pdf",
    "fig7_temporal_stability_chart.pdf",
    "fig8_roc_curve.pdf",
    "fig9_pr_curve.pdf",
    "fig10_entropy_distribution.pdf",
    "fig11_family_prevalence.pdf",
    "fig12_adversarial_results.pdf",
    "fig_synthetic_real_feature_space_pca.pdf",
    "table3_prevalence_breakdown.pdf",
    "table3a_targeted_prevalence.pdf",
]


def combine_pdfs():
    """Combine all PDF files into a single master PDF."""
    
    print("="*70)
    print("COMBINING ALL PDF FIGURES")
    print("="*70)
    
    # Check if PDF directory exists
    if not PDF_DIR.exists():
        print(f"ERROR: PDF directory not found: {PDF_DIR}")
        return False
    
    # Collect valid PDF files
    valid_pdfs = []
    
    for pdf_file in PDF_FILES:
        pdf_path = PDF_DIR / pdf_file
        
        if not pdf_path.exists():
            print(f"  ⚠️  Missing: {pdf_file}")
            continue
        
        if not pdf_path.is_file():
            print(f"  ⚠️  Not a file: {pdf_file}")
            continue
        
        valid_pdfs.append((pdf_file, pdf_path))
        print(f"  ✓ Found: {pdf_file}")
    
    if not valid_pdfs:
        print("\n  ERROR: No PDF files found to combine!")
        return False
    
    print(f"\nMerging {len(valid_pdfs)} PDF files...")
    
    try:
        # Create PDF merger
        merger = PdfMerger()
        
        # Add each PDF
        for pdf_name, pdf_path in valid_pdfs:
            merger.append(str(pdf_path))
            print(f"  ✓ Added: {pdf_name}")
        
        # Create output directory if it doesn't exist
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Write combined PDF
        merger.write(str(OUTPUT_FILE))
        merger.close()
        
        # Get file size
        file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
        
        print(f"\n✓ COMBINED PDF CREATED")
        print(f"  Location: {OUTPUT_FILE}")
        print(f"  Total pages: {len(valid_pdfs)}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print("\n" + "="*70)
        
        return True
    
    except Exception as e:
        print(f"\nERROR during PDF merge: {str(e)}")
        print("Make sure PyPDF2 is installed: uv pip install PyPDF2")
        return False


if __name__ == "__main__":
    success = combine_pdfs()
    sys.exit(0 if success else 1)
