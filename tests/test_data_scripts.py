from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from data.scripts import build_dataset


def _zip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("fixture.txt", "safe deterministic fixture")


def test_builder_refuses_to_overwrite_with_only_one_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    positive = tmp_path / "positive"
    benign = tmp_path / "benign"
    benign.mkdir()
    _zip(positive / "zombie_A_fixture_0000.zip")
    features = tmp_path / "features.csv"
    labels = tmp_path / "labels.csv"
    features.write_text("existing-features\n", encoding="utf-8")
    labels.write_text("existing-labels\n", encoding="utf-8")
    monkeypatch.setattr(build_dataset, "POSITIVE_DIR", positive)
    monkeypatch.setattr(build_dataset, "BENIGN_DIR", benign)
    monkeypatch.setattr(build_dataset, "OUTPUT_FEATURES", features)
    monkeypatch.setattr(build_dataset, "OUTPUT_LABELS", labels)

    with pytest.raises(RuntimeError, match="without both positive and benign"):
        build_dataset.main()

    assert features.read_text(encoding="utf-8") == "existing-features\n"
    assert labels.read_text(encoding="utf-8") == "existing-labels\n"
