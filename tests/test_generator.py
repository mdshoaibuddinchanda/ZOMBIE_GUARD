from __future__ import annotations

import random
from pathlib import Path

from data.scripts.generate_zombie_samples import (
    add_decoy_entries_raw,
    variant_extra_field_noise,
    variant_gootloader_concat,
)
from zombieguard.extractor import extract_features


def test_concatenated_variant_stays_valid_after_decoy_normalization(
    tmp_path: Path,
) -> None:
    random.seed(42)
    archive = tmp_path / "concatenated.zip"
    variant_gootloader_concat(archive, chain_len=3)
    archive.write_bytes(add_decoy_entries_raw(archive.read_bytes(), count=3))

    features = extract_features(archive)

    assert features["parse_status"] == "ok"
    assert features["eocd_count"] == 3


def test_unknown_extra_field_variant_is_well_formed(tmp_path: Path) -> None:
    random.seed(42)
    archive = tmp_path / "extra-field.zip"
    variant_extra_field_noise(archive)

    features = extract_features(archive)

    assert features["parse_status"] == "ok"
    assert features["method_mismatch"] is True
