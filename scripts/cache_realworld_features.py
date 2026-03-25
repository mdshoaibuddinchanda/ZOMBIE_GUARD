#!/usr/bin/env python
"""Pre-compute and cache real-world features for faster validation runs."""

import logging
import os
import sys
from pathlib import Path

# Setup logging before anything else
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.realworld_features import load_realworld_features, REAL_CACHE_PATH

def main() -> None:
    """Build and cache real-world feature matrix."""
    logger.info("=" * 70)
    logger.info("PRE-COMPUTING REAL-WORLD FEATURE CACHE")
    logger.info("=" * 70)
    
    if REAL_CACHE_PATH.exists():
        logger.info(f"✓ Cache already exists at {REAL_CACHE_PATH}")
        logger.info("  Run with --refresh to rebuild.")
        return
    
    logger.info(f"Building cache from 1,366 real-world ZIP samples...")
    logger.info(f"Cache will be saved to: {REAL_CACHE_PATH}")
    logger.info("")
    
    # Load features (will extract and cache automatically)
    logger.info("Starting feature extraction (this may take 5-15 minutes)...")
    df = load_realworld_features(refresh=False)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"✓ COMPLETE: Cached {len(df)} samples with {len(df.columns)} features")
    logger.info(f"  Malicious: {(df['label']==1).sum()}")
    logger.info(f"  Benign:    {(df['label']==0).sum()}")
    logger.info(f"  Families:  {df['family'].nunique()}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
