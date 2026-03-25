"""Utilities for loading and caching real-world ZombieGuard feature matrices."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.classifier import FEATURE_COLS
from src.extractor import extract_features

REAL_LABELS_PATH = Path("data/realworld_labels.csv")
REAL_DIR = Path("data/real_world_validation")
REAL_CACHE_PATH = Path("data/processed/realworld_features.csv")

_BOOL_COLS = [
    "method_mismatch",
    "declared_vs_entropy_flag",
    "lf_crc_valid",
    "any_crc_mismatch",
    "is_encrypted",
]


def _normalize_family(signal_value: str) -> str:
    """Map raw signal string into a stable family label."""
    signal = str(signal_value).strip().lower()
    if not signal or signal == "nan":
        return "Unknown"
    mapping = {
        "gootkit": "Gootloader",
        "gootloader": "Gootloader",
    }
    return mapping.get(signal, signal.capitalize())


def _extract_all_real_features() -> pd.DataFrame:
    """Extract features from all real-world ZIP samples listed in labels CSV."""
    labels_df = pd.read_csv(REAL_LABELS_PATH)
    rows: list[dict[str, object]] = []

    for idx, row in enumerate(labels_df.itertuples(index=False), start=1):
        filename = str(row.filename)
        file_path = REAL_DIR / filename
        if not file_path.exists():
            continue

        features = extract_features(str(file_path))
        features["filename"] = filename
        features["label"] = int(float(str(row.label)))
        features["signal"] = str(row.signal)
        features["family"] = _normalize_family(features["signal"])
        rows.append(features)

        if idx % 200 == 0:
            print(f"  Extracted {idx}/{len(labels_df)} samples")

    if not rows:
        raise RuntimeError("No real-world features were extracted.")

    real_df = pd.DataFrame(rows)
    for col in _BOOL_COLS:
        if col in real_df.columns:
            real_df[col] = real_df[col].astype(int)

    for col in FEATURE_COLS:
        if col not in real_df.columns:
            real_df[col] = 0

    return real_df


def load_realworld_features(refresh: bool = False) -> pd.DataFrame:
    """Load cached real-world features or build cache from ZIP files."""
    if REAL_CACHE_PATH.exists() and not refresh:
        cached = pd.read_csv(REAL_CACHE_PATH)
        for col in FEATURE_COLS:
            if col not in cached.columns:
                cached[col] = 0
        return cached

    REAL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    real_df = _extract_all_real_features()
    real_df.to_csv(REAL_CACHE_PATH, index=False)
    print(f"Saved real-world feature cache: {REAL_CACHE_PATH}")
    return real_df
