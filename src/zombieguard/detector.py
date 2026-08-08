"""Command-line interface for the ZombieGuard defensive ZIP scanner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from zombieguard import __version__
from zombieguard.classifier import (
    DEFAULT_THRESHOLD,
    MODEL_PATH,
    ModelIntegrityError,
    load_model,
    predict,
)
from zombieguard.extractor import DEFAULT_MAX_INPUT_SIZE_BYTES, extract_features

SUPPORTED_SUFFIXES = {".zip"}
SCHEMA_VERSION = "1.0"
EXIT_CLEAN = 0
EXIT_DETECTED = 1
EXIT_ERROR = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError:
        size = None
    return {"path": str(path), "name": path.name, "size_bytes": size}


def scan_file(
    path: str | Path,
    *,
    model: Any,
    model_path: str | Path,
    threshold: float,
    max_input_size_bytes: int = DEFAULT_MAX_INPUT_SIZE_BYTES,
    include_features: bool = False,
) -> dict[str, Any]:
    """Scan one archive and return a stable, machine-readable result."""

    started = time.perf_counter()
    archive = Path(path).expanduser().resolve()
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scanner": {"name": "ZombieGuard", "version": __version__},
        "file": _file_record(archive),
        "status": "unscannable",
        "verdict": "UNSCANNABLE",
        "score": None,
        "threshold": threshold,
        "reason_codes": [],
        "error": None,
    }

    features = extract_features(
        archive, max_input_size_bytes=max_input_size_bytes
    )
    parse_status = str(features.get("parse_status", "malformed"))
    if parse_status != "ok":
        if features.get("is_encrypted"):
            result["reason_codes"] = ["ENCRYPTED_ARCHIVE_UNSUPPORTED"]
        elif features.get("lf_unknown_method"):
            result["reason_codes"] = ["COMPRESSION_METHOD_UNSUPPORTED"]
        else:
            result["reason_codes"] = [f"PARSER_{parse_status.upper()}"]
        result["error"] = features.get("parse_error") or "archive could not be parsed"
    else:
        try:
            result["file"]["sha256"] = _sha256(archive)
            prediction = predict(model, features, threshold=threshold)
            detected = bool(prediction["label"])
            result.update(
                {
                    "status": "detected" if detected else "clean",
                    "verdict": "ZIP EVASION DETECTED" if detected else "CLEAN",
                    "score": prediction["score"],
                    "reason_codes": prediction["reason_codes"],
                    "target": prediction["target"],
                    "model": {"path": str(Path(model_path))},
                }
            )
            if include_features:
                result["features"] = {
                    key: value
                    for key, value in features.items()
                    if key not in {"parse_error"}
                }
        except (OSError, TypeError, ValueError) as exc:
            result["reason_codes"] = ["INFERENCE_ERROR"]
            result["error"] = str(exc)

    result["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return result


def _discover(paths: Iterable[str], recursive: bool) -> list[Path]:
    discovered: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_file():
            discovered.append(path)
            continue
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            discovered.extend(
                candidate
                for candidate in iterator
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES
            )
            continue
        # Preserve missing paths so they become explicit UNSCANNABLE results.
        discovered.append(path)
    return sorted(set(discovered), key=lambda item: str(item).lower())


def _summary(results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(results),
        "clean": sum(result["status"] == "clean" for result in results),
        "detected": sum(result["status"] == "detected" for result in results),
        "unscannable": sum(
            result["status"] == "unscannable" for result in results
        ),
    }


def _as_json(results: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "summary": _summary(results),
            "results": results,
        },
        indent=2,
        sort_keys=True,
    )


def _as_sarif(results: list[dict[str, Any]]) -> str:
    sarif_results: list[dict[str, Any]] = []
    for result in results:
        if result["status"] == "clean":
            continue
        rule_id = (
            "ZG001" if result["status"] == "detected" else "ZG002"
        )
        message = (
            "ZIP structural-evasion indicators detected"
            if rule_id == "ZG001"
            else f"Archive could not be scanned: {result.get('error')}"
        )
        sarif_results.append(
            {
                "ruleId": rule_id,
                "level": "error" if rule_id == "ZG001" else "warning",
                "message": {"text": message},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": result["file"]["path"]}
                        }
                    }
                ],
                "properties": {
                    "score": result.get("score"),
                    "reasonCodes": result.get("reason_codes", []),
                    "sha256": result["file"].get("sha256"),
                },
            }
        )
    payload = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "ZombieGuard",
                        "version": __version__,
                        "informationUri": "https://github.com/mdshoaibuddinchanda/zombieguard",
                        "rules": [
                            {
                                "id": "ZG001",
                                "shortDescription": {
                                    "text": "ZIP structural-evasion indicators"
                                },
                            },
                            {
                                "id": "ZG002",
                                "shortDescription": {
                                    "text": "Archive was not safely scannable"
                                },
                            },
                        ],
                    }
                },
                "results": sarif_results,
            }
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _as_text(results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for result in results:
        file_name = result["file"]["path"]
        if result["status"] == "detected" or result["status"] == "clean":
            details = f"score={result['score']:.4f}"
        else:
            details = str(result.get("error") or "unknown parser error")
        reasons = ",".join(result.get("reason_codes", [])) or "none"
        lines.append(
            f"{result['verdict']:<22} {file_name} ({details}; reasons={reasons})"
        )
    summary = _summary(results)
    lines.append(
        "Summary: "
        f"{summary['total']} scanned, {summary['detected']} detected, "
        f"{summary['clean']} clean, {summary['unscannable']} unscannable"
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zombieguard",
        description="Detect structural ZIP metadata evasion defensively.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    scan = subcommands.add_parser("scan", help="scan one file or a directory")
    scan.add_argument("paths", nargs="+", help="archive files or directories")
    scan.add_argument("--recursive", "-r", action="store_true")
    scan.add_argument("--model", type=Path, default=MODEL_PATH)
    scan.add_argument("--threshold", type=float, default=None)
    scan.add_argument(
        "--max-size-mb",
        type=float,
        default=DEFAULT_MAX_INPUT_SIZE_BYTES / (1024 * 1024),
    )
    scan.add_argument("--format", choices=("text", "json", "sarif"), default="text")
    scan.add_argument("--output", type=Path, help="write output to a file")
    scan.add_argument("--include-features", action="store_true")
    scan.add_argument(
        "--exit-zero",
        action="store_true",
        help="always return zero after a completed scan (reporting mode)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command != "scan":
        return EXIT_ERROR
    if not math.isfinite(args.max_size_mb) or args.max_size_mb <= 0:
        print(
            "ZombieGuard input error: --max-size-mb must be finite and positive",
            file=sys.stderr,
        )
        return EXIT_ERROR

    try:
        model = load_model(args.model)
        threshold = (
            float(args.threshold)
            if args.threshold is not None
            else float(getattr(model, "_zombieguard_threshold", DEFAULT_THRESHOLD))
        )
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be strictly between 0 and 1")
    except (FileNotFoundError, OSError, ValueError, ModelIntegrityError) as exc:
        print(f"ZombieGuard model error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    paths = _discover(args.paths, args.recursive)
    if not paths:
        print(
            "ZombieGuard input error: no ZIP files were found in the supplied paths",
            file=sys.stderr,
        )
        return EXIT_ERROR
    results = [
        scan_file(
            path,
            model=model,
            model_path=args.model,
            threshold=threshold,
            max_input_size_bytes=int(args.max_size_mb * 1024 * 1024),
            include_features=args.include_features,
        )
        for path in paths
    ]

    if args.format == "json":
        rendered = _as_json(results)
    elif args.format == "sarif":
        rendered = _as_sarif(results)
    else:
        rendered = _as_text(results)

    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"ZombieGuard output error: {exc}", file=sys.stderr)
            return EXIT_ERROR
    else:
        print(rendered)

    if args.exit_zero:
        return EXIT_CLEAN
    summary = _summary(results)
    if summary["unscannable"]:
        return EXIT_ERROR
    if summary["detected"]:
        return EXIT_DETECTED
    return EXIT_CLEAN


if __name__ == "__main__":
    raise SystemExit(main())
