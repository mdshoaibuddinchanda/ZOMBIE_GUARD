from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from zombieguard.classifier import FEATURE_COLS, save_model
from zombieguard.detector import _as_json, _as_sarif, _discover, main, scan_file


class FixedScoreModel:
    def __init__(self, score: float) -> None:
        self.score = score

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray([[1.0 - self.score, self.score] for _ in range(len(frame))])


def _normal_zip(path: Path) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("document.txt", "safe fixture content " * 200)
    return path


def _set_encrypted_flag(path: Path) -> Path:
    content = bytearray(path.read_bytes())
    central_offset = content.find(b"PK\x01\x02")
    assert central_offset >= 0
    local_flags = struct.unpack_from("<H", content, 6)[0]
    central_flags = struct.unpack_from("<H", content, central_offset + 8)[0]
    struct.pack_into("<H", content, 6, local_flags | 0x0001)
    struct.pack_into("<H", content, central_offset + 8, central_flags | 0x0001)
    path.write_bytes(content)
    return path


def _tiny_model(path: Path) -> Path:
    rows = []
    labels = []
    for index in range(24):
        label = index % 2
        rows.append({column: float(label) for column in FEATURE_COLS})
        labels.append(label)
    frame = pd.DataFrame(rows, columns=FEATURE_COLS)
    model = LGBMClassifier(
        n_estimators=8,
        num_leaves=4,
        min_child_samples=1,
        random_state=42,
        verbose=-1,
    )
    model.fit(frame, labels)
    save_model(model, path, metadata={"test_fixture": True})
    return path


def test_scan_file_clean_result_has_hash(tmp_path: Path) -> None:
    archive = _normal_zip(tmp_path / "normal.zip")
    result = scan_file(
        archive,
        model=FixedScoreModel(0.1),
        model_path="test-model.txt",
        threshold=0.5,
    )

    assert result["status"] == "clean"
    assert result["verdict"] == "CLEAN"
    assert len(result["file"]["sha256"]) == 64
    assert result["duration_ms"] >= 0


def test_scan_file_detected_is_machine_readable(tmp_path: Path) -> None:
    archive = _normal_zip(tmp_path / "suspicious.zip")
    result = scan_file(
        archive,
        model=FixedScoreModel(0.9),
        model_path="test-model.txt",
        threshold=0.5,
        include_features=True,
    )

    assert result["status"] == "detected"
    assert result["score"] == 0.9
    assert result["features"]["parse_status"] == "ok"


def test_missing_input_is_unscannable(tmp_path: Path) -> None:
    result = scan_file(
        tmp_path / "missing.zip",
        model=FixedScoreModel(0.1),
        model_path="test-model.txt",
        threshold=0.5,
    )

    assert result["status"] == "unscannable"
    assert result["reason_codes"] == ["PARSER_NOT_FOUND"]
    assert result["score"] is None


def test_json_and_sarif_reports(tmp_path: Path) -> None:
    archive = _normal_zip(tmp_path / "report.zip")
    result = scan_file(
        archive,
        model=FixedScoreModel(0.9),
        model_path="test-model.txt",
        threshold=0.5,
    )

    json_report = json.loads(_as_json([result]))
    sarif_report = json.loads(_as_sarif([result]))
    assert json_report["summary"]["detected"] == 1
    assert sarif_report["version"] == "2.1.0"
    assert sarif_report["runs"][0]["results"][0]["ruleId"] == "ZG001"


def test_cli_json_smoke_uses_verified_native_model(
    tmp_path: Path, capsys
) -> None:
    archive = _normal_zip(tmp_path / "cli.zip")
    model_path = _tiny_model(tmp_path / "model.txt")

    exit_code = main(
        [
            "scan",
            str(archive),
            "--model",
            str(model_path),
            "--format",
            "json",
            "--exit-zero",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["summary"]["total"] == 1
    assert report["results"][0]["status"] in {"clean", "detected"}


def test_directory_discovery_is_zip_only(tmp_path: Path) -> None:
    _normal_zip(tmp_path / "archive.zip")
    _normal_zip(tmp_path / "application.jar")

    assert [path.name for path in _discover([str(tmp_path)], recursive=False)] == [
        "archive.zip"
    ]


def test_empty_directory_is_an_input_error(tmp_path: Path, capsys) -> None:
    model_path = _tiny_model(tmp_path / "model.txt")
    empty = tmp_path / "empty"
    empty.mkdir()

    exit_code = main(["scan", str(empty), "--model", str(model_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "no ZIP files" in captured.err
    assert captured.out == ""


def test_cli_writes_sarif_and_returns_detected_exit(tmp_path: Path) -> None:
    archive = _normal_zip(tmp_path / "input.zip")
    model_path = _tiny_model(tmp_path / "model.txt")
    output = tmp_path / "reports" / "result.sarif"

    exit_code = main(
        [
            "scan",
            str(archive),
            "--model",
            str(model_path),
            "--threshold",
            "0.000001",
            "--format",
            "sarif",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report["runs"][0]["results"][0]["ruleId"] == "ZG001"


def test_cli_rejects_missing_model(tmp_path: Path, capsys) -> None:
    archive = _normal_zip(tmp_path / "input.zip")

    exit_code = main(
        ["scan", str(archive), "--model", str(tmp_path / "missing-model.txt")]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "model error" in captured.err


def test_scan_size_limit_is_explicit(tmp_path: Path) -> None:
    archive = _normal_zip(tmp_path / "oversized.zip")

    result = scan_file(
        archive,
        model=FixedScoreModel(0.1),
        model_path="test-model.txt",
        threshold=0.5,
        max_input_size_bytes=1,
    )

    assert result["status"] == "unscannable"
    assert result["reason_codes"] == ["PARSER_TOO_LARGE"]


def test_unscannable_result_is_included_in_sarif(tmp_path: Path) -> None:
    result = scan_file(
        tmp_path / "missing.zip",
        model=FixedScoreModel(0.1),
        model_path="test-model.txt",
        threshold=0.5,
    )

    report = json.loads(_as_sarif([result]))
    finding = report["runs"][0]["results"][0]
    assert finding["ruleId"] == "ZG002"
    assert finding["level"] == "warning"


def test_encrypted_archive_is_never_sent_to_inference(tmp_path: Path) -> None:
    archive = _set_encrypted_flag(_normal_zip(tmp_path / "encrypted.zip"))

    result = scan_file(
        archive,
        model=FixedScoreModel(0.0),
        model_path="test-model.txt",
        threshold=0.5,
    )

    assert result["status"] == "unscannable"
    assert result["verdict"] == "UNSCANNABLE"
    assert result["score"] is None
    assert result["reason_codes"] == ["ENCRYPTED_ARCHIVE_UNSUPPORTED"]


def test_cli_rejects_non_finite_size_limit(capsys) -> None:
    exit_code = main(["scan", "anything.zip", "--max-size-mb", "nan"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "finite and positive" in captured.err
