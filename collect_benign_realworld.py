"""
collect_benign_realworld.py
===========================
ZombieGuard - Benign False Positive Corpus Experiment

Purpose:
Download real-world benign ZIP archives from public software releases and
run ZombieGuard on all files to measure false positive rate.

Outputs:
- data/benign_realworld/          downloaded ZIP files
- data/benign_fp_results.csv      per-file detector results
- data/benign_fp_summary.txt      paper-ready summary text
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("tqdm not found - progress bars disabled. Run: uv pip install tqdm")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benign_fp")


PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "benign_realworld"
RESULTS_CSV = PROJECT_ROOT / "data" / "benign_fp_results.csv"
SUMMARY_TXT = PROJECT_ROOT / "data" / "benign_fp_summary.txt"
PROGRESS_JSON = PROJECT_ROOT / "data" / "benign_fp_progress.json"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

DETECTION_THRESHOLD = 0.5
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_DELAY = 5
MAX_FILE_SIZE_MB = 500


BENIGN_SOURCES = [
    (
        "https://www.python.org/ftp/python/3.12.3/python-3.12.3-embed-amd64.zip",
        "python312_embed_amd64",
        "python_runtime",
    ),
    (
        "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip",
        "python311_embed_amd64",
        "python_runtime",
    ),
    (
        "https://www.python.org/ftp/python/3.10.14/python-3.10.14-embed-amd64.zip",
        "python310_embed_amd64",
        "python_runtime",
    ),
    (
        "https://www.python.org/ftp/python/3.12.3/python-3.12.3-embed-win32.zip",
        "python312_embed_win32",
        "python_runtime",
    ),
    (
        "https://archive.apache.org/dist/maven/maven-3/3.9.6/binaries/apache-maven-3.9.6-bin.zip",
        "maven_396_bin",
        "apache_maven",
    ),
    (
        "https://archive.apache.org/dist/maven/maven-3/3.8.8/binaries/apache-maven-3.8.8-bin.zip",
        "maven_388_bin",
        "apache_maven",
    ),
    (
        "https://archive.apache.org/dist/ant/binaries/apache-ant-1.10.14-bin.zip",
        "ant_11014_bin",
        "apache_ant",
    ),
    (
        "https://archive.apache.org/dist/ant/binaries/apache-ant-1.10.13-bin.zip",
        "ant_11013_bin",
        "apache_ant",
    ),
    (
        "https://services.gradle.org/distributions/gradle-8.7-bin.zip",
        "gradle_87_bin",
        "gradle",
    ),
    (
        "https://services.gradle.org/distributions/gradle-8.6-bin.zip",
        "gradle_86_bin",
        "gradle",
    ),
    (
        "https://services.gradle.org/distributions/gradle-7.6.4-bin.zip",
        "gradle_764_bin",
        "gradle",
    ),
    (
        "https://services.gradle.org/distributions/gradle-8.7-src.zip",
        "gradle_87_src",
        "gradle",
    ),
    (
        "https://github.com/Kitware/CMake/releases/download/v3.29.3/cmake-3.29.3-windows-x86_64.zip",
        "cmake_3293_win64",
        "cmake",
    ),
    (
        "https://github.com/Kitware/CMake/releases/download/v3.28.6/cmake-3.28.6-windows-x86_64.zip",
        "cmake_3286_win64",
        "cmake",
    ),
    (
        "https://github.com/JetBrains/kotlin/releases/download/v1.9.24/kotlin-compiler-1.9.24.zip",
        "kotlin_1924_compiler",
        "kotlin",
    ),
    (
        "https://github.com/JetBrains/kotlin/releases/download/v2.0.0/kotlin-compiler-2.0.0.zip",
        "kotlin_200_compiler",
        "kotlin",
    ),
    (
        "https://github.com/SeleniumHQ/selenium/releases/download/selenium-4.21.0/selenium-java-4.21.0.zip",
        "selenium_java_4210",
        "selenium",
    ),
    (
        "https://github.com/OpenAPITools/openapi-generator/archive/refs/tags/v7.5.0.zip",
        "openapi_generator_750_src",
        "openapi",
    ),
    (
        "https://archive.apache.org/dist/commons/lang/binaries/commons-lang3-3.14.0-bin.zip",
        "commons_lang3_3140_bin",
        "apache_commons",
    ),
    (
        "https://archive.apache.org/dist/commons/collections/binaries/commons-collections4-4.4-bin.zip",
        "commons_collections4_44_bin",
        "apache_commons",
    ),
    (
        "https://archive.apache.org/dist/commons/io/binaries/commons-io-2.16.1-bin.zip",
        "commons_io_2161_bin",
        "apache_commons",
    ),
    (
        "https://github.com/spring-projects/spring-framework/archive/refs/tags/v6.1.8.zip",
        "spring_framework_618_src",
        "spring",
    ),
    (
        "https://github.com/gohugoio/hugo/releases/download/v0.126.1/hugo_0.126.1_windows-amd64.zip",
        "hugo_01261_win64",
        "hugo",
    ),
    (
        "https://releases.hashicorp.com/terraform/1.8.4/terraform_1.8.4_windows_amd64.zip",
        "terraform_184_win64",
        "terraform",
    ),
    (
        "https://releases.hashicorp.com/terraform/1.8.4/terraform_1.8.4_linux_amd64.zip",
        "terraform_184_linux64",
        "terraform",
    ),
    (
        "https://releases.hashicorp.com/terraform/1.7.5/terraform_1.7.5_windows_amd64.zip",
        "terraform_175_win64",
        "terraform",
    ),
    (
        "https://releases.hashicorp.com/terraform/1.6.6/terraform_1.6.6_windows_amd64.zip",
        "terraform_166_win64",
        "terraform",
    ),
    (
        "https://releases.hashicorp.com/packer/1.10.3/packer_1.10.3_windows_amd64.zip",
        "packer_1103_win64",
        "packer",
    ),
    (
        "https://releases.hashicorp.com/vault/1.16.2/vault_1.16.2_windows_amd64.zip",
        "vault_1162_win64",
        "vault",
    ),
    (
        "https://releases.hashicorp.com/consul/1.18.2/consul_1.18.2_windows_amd64.zip",
        "consul_1182_win64",
        "consul",
    ),
    (
        "https://download.sysinternals.com/files/SysinternalsSuite.zip",
        "sysinternals_suite",
        "sysinternals",
    ),
    (
        "https://download.sysinternals.com/files/ProcessMonitor.zip",
        "sysinternals_procmon",
        "sysinternals",
    ),
    (
        "https://download.sysinternals.com/files/ProcessExplorer.zip",
        "sysinternals_procexp",
        "sysinternals",
    ),
    (
        "https://download.sysinternals.com/files/Autoruns.zip",
        "sysinternals_autoruns",
        "sysinternals",
    ),
    (
        "https://download.sysinternals.com/files/TCPView.zip",
        "sysinternals_tcpview",
        "sysinternals",
    ),
    (
        "https://download.sysinternals.com/files/Bginfo.zip",
        "sysinternals_bginfo",
        "sysinternals",
    ),
    (
        "https://2.na.dl.wireshark.org/win64/WiresharkPortable64_4.2.5.zip",
        "wireshark_portable_425",
        "wireshark",
    ),
    (
        "https://nmap.org/dist/nmap-7.95-win32.zip",
        "nmap_795_win32",
        "nmap",
    ),
    (
        "https://downloads.sourceforge.net/project/nsis/NSIS%203/3.10/nsis-3.10.zip",
        "nsis_310",
        "nsis",
    ),
    (
        "https://github.com/nicowillis/7-zip/archive/refs/heads/main.zip",
        "sevenzip_src_main",
        "sevenzip_src",
    ),
    (
        "https://github.com/mikefarah/yq/releases/download/v4.44.1/yq_windows_amd64.zip",
        "yq_4441_win64",
        "yq",
    ),
    (
        "https://github.com/jqlang/jq/releases/download/jq-1.7.1/jq-windows-amd64.zip",
        "jq_171_win64",
        "jq",
    ),
    (
        "https://github.com/astral-sh/ruff/releases/download/v0.4.4/ruff-x86_64-pc-windows-msvc.zip",
        "ruff_044_win64",
        "ruff",
    ),
    (
        "https://github.com/astral-sh/uv/releases/download/0.1.45/uv-x86_64-pc-windows-msvc.zip",
        "uv_0145_win64",
        "uv",
    ),
    (
        "https://github.com/mermaid-js/mermaid/archive/refs/tags/v10.9.1.zip",
        "mermaid_10091_src",
        "mermaid",
    ),
    (
        "https://github.com/zaproxy/zaproxy/releases/download/v2.15.0/ZAP_2.15.0_Core.zip",
        "owasp_zap_2150_core",
        "owasp_zap",
    ),
    (
        "https://github.com/volatilityfoundation/volatility3/archive/refs/tags/v2.7.0.zip",
        "volatility3_270_src",
        "forensics",
    ),
    (
        "https://github.com/rapid7/metasploit-framework/archive/refs/tags/6.4.15.zip",
        "metasploit_6415_src",
        "security_tools",
    ),
    (
        "https://github.com/SigmaHQ/sigma/archive/refs/tags/r2024-05-15.zip",
        "sigma_rules_20240515",
        "security_tools",
    ),
    (
        "https://github.com/VirusTotal/yara/archive/refs/tags/v4.5.1.zip",
        "yara_451_src",
        "security_tools",
    ),
    (
        "https://github.com/snort3/snort3/archive/refs/tags/3.3.3.0.zip",
        "snort3_3330_src",
        "security_tools",
    ),
    (
        "https://update.code.visualstudio.com/1.89.1/win32-x64-archive/stable",
        "vscode_18901_win64",
        "vscode",
    ),
    (
        "https://update.code.visualstudio.com/1.88.1/win32-x64-archive/stable",
        "vscode_18801_win64",
        "vscode",
    ),
    (
        "https://www.eclipse.org/downloads/download.php?file=/eclipse/downloads/drops4/R-4.31-202402290520/eclipse-SDK-4.31-win32-x86_64.zip&mirror_id=1&r=1",
        "eclipse_431_sdk_win64",
        "eclipse",
    ),
    (
        "https://download.jetbrains.com/idea/ideaIC-2024.1.2.win.zip",
        "intellij_241_win_zip",
        "jetbrains",
    ),
    (
        "https://download.jetbrains.com/python/pycharm-community-2024.1.2.win.zip",
        "pycharm_241_win_zip",
        "jetbrains",
    ),
    (
        "https://archive.apache.org/dist/tomcat/tomcat-10/v10.1.24/bin/apache-tomcat-10.1.24.zip",
        "tomcat_10124_bin",
        "apache_tomcat",
    ),
    (
        "https://archive.apache.org/dist/tomcat/tomcat-9/v9.0.89/bin/apache-tomcat-9.0.89.zip",
        "tomcat_9089_bin",
        "apache_tomcat",
    ),
    (
        "https://binaries.sonarsource.com/Distribution/sonarqube/sonarqube-10.5.1.90531.zip",
        "sonarqube_10051_community",
        "sonarqube",
    ),
    (
        "https://github.com/python/cpython/archive/refs/tags/v3.12.3.zip",
        "cpython_3123_src",
        "python_source",
    ),
    (
        "https://github.com/pallets/flask/archive/refs/tags/3.0.3.zip",
        "flask_303_src",
        "python_source",
    ),
    (
        "https://github.com/django/django/archive/refs/tags/5.0.6.zip",
        "django_506_src",
        "python_source",
    ),
    (
        "https://github.com/numpy/numpy/archive/refs/tags/v1.26.4.zip",
        "numpy_1264_src",
        "python_source",
    ),
    (
        "https://github.com/scikit-learn/scikit-learn/archive/refs/tags/1.5.0.zip",
        "sklearn_150_src",
        "python_source",
    ),
    (
        "https://github.com/dmlc/xgboost/archive/refs/tags/v2.0.3.zip",
        "xgboost_203_src",
        "python_source",
    ),
    (
        "https://github.com/facebook/react/archive/refs/tags/v18.3.1.zip",
        "react_1831_src",
        "javascript_source",
    ),
    (
        "https://github.com/vuejs/vue/archive/refs/tags/v2.7.16.zip",
        "vue_2716_src",
        "javascript_source",
    ),
    (
        "https://github.com/twbs/bootstrap/archive/refs/tags/v5.3.3.zip",
        "bootstrap_533_src",
        "javascript_source",
    ),
    (
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
        "ffmpeg_latest_win64_gpl",
        "binary_high_entropy",
    ),
    (
        "https://slproweb.com/download/Win64OpenSSL-3_3_0.zip",
        "openssl_330_win64",
        "binary_high_entropy",
    ),
    (
        "https://go.dev/dl/go1.22.3.windows-amd64.zip",
        "go_1223_win64",
        "go_runtime",
    ),
    (
        "https://go.dev/dl/go1.21.11.windows-amd64.zip",
        "go_12111_win64",
        "go_runtime",
    ),
    (
        "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.3%2B9/OpenJDK21U-jdk_x64_windows_hotspot_21.0.3_9.zip",
        "temurin_jdk21_win64",
        "java_runtime",
    ),
    (
        "https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.11%2B9/OpenJDK17U-jdk_x64_windows_hotspot_17.0.11_9.zip",
        "temurin_jdk17_win64",
        "java_runtime",
    ),
    (
        "https://github.com/winpython/winpython/releases/download/7.0.20240203/Winpython64-3.12.1.0dot.zip",
        "winpython_3121_portable",
        "python_portable",
    ),
    (
        "https://github.com/mitre-attack/attack-navigator/archive/refs/tags/v4.9.5.zip",
        "attack_navigator_495_src",
        "security_tools",
    ),
    (
        "https://github.com/gchq/CyberChef/releases/download/v10.18.9/CyberChef_v10.18.9.zip",
        "cyberchef_v10189",
        "security_tools",
    ),
]


def sha256_of_file(path: Path) -> str:
    """Compute SHA-256 for integrity logging/debug."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest_path: Path, name: str) -> bool:
    """Download one ZIP with retries, size checks, and magic-byte validation."""
    if dest_path.exists():
        size_mb = dest_path.stat().st_size / 1_048_576
        log.info(f"  SKIP (already exists, {size_mb:.1f} MB): {name}")
        return True

    for attempt in range(1, MAX_RETRIES + 1):
        tmp_path = dest_path.with_suffix(".tmp")
        try:
            log.info(f"  Downloading [{attempt}/{MAX_RETRIES}]: {name}")
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; ZombieGuard-Research/1.0; "
                    "+https://github.com/mdshoaibuddinchanda/zombieguard)"
                )
            }

            with requests.get(
                url,
                stream=True,
                timeout=REQUEST_TIMEOUT,
                headers=headers,
                allow_redirects=True,
            ) as resp:
                resp.raise_for_status()

                content_length = resp.headers.get("Content-Length")
                if content_length:
                    size_mb = int(content_length) / 1_048_576
                    if size_mb > MAX_FILE_SIZE_MB:
                        log.warning(
                            f"  SKIP (too large: {size_mb:.0f} MB > "
                            f"{MAX_FILE_SIZE_MB} MB): {name}"
                        )
                        return False

                total_written = 0
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if not chunk:
                            continue
                        f.write(chunk)
                        total_written += len(chunk)
                        if total_written > MAX_FILE_SIZE_MB * 1_048_576:
                            f.close()
                            tmp_path.unlink(missing_ok=True)
                            log.warning(
                                f"  SKIP (exceeded size during download): {name}"
                            )
                            return False

                with open(tmp_path, "rb") as f:
                    magic = f.read(4)
                if magic[:2] != b"PK":
                    log.warning(
                        f"  SKIP (not ZIP by magic bytes={magic.hex()}): {name}"
                    )
                    tmp_path.unlink(missing_ok=True)
                    return False

                tmp_path.rename(dest_path)
                size_mb = dest_path.stat().st_size / 1_048_576
                log.info(f"  OK ({size_mb:.1f} MB): {name}")
                return True

        except requests.exceptions.HTTPError as exc:
            log.warning(f"  HTTP error for {name}: {exc}")
        except requests.exceptions.ConnectionError as exc:
            log.warning(f"  Connection error for {name}: {exc}")
        except requests.exceptions.Timeout:
            log.warning(f"  Timeout for {name}")
        except Exception as exc:
            log.error(f"  Unexpected error for {name}: {exc}")
            break
        finally:
            tmp_path.unlink(missing_ok=True)

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    log.error(f"  FAILED after {MAX_RETRIES} attempts: {name}")
    return False


def load_zombieguard_runner():
    """Load ZombieGuard extractor/model and return a per-file inference closure."""
    try:
        from extractor import extract_features
    except ImportError as exc:
        raise ImportError(
            "Cannot import extractor. Run from project root with PYTHONPATH=.\n"
            f"Original error: {exc}"
        )

    try:
        from classifier import FEATURE_COLS, load_model
    except ImportError as exc:
        raise ImportError(
            "Cannot import classifier. Run from project root with PYTHONPATH=.\n"
            f"Original error: {exc}"
        )

    model_path = MODELS_DIR / "xgboost_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train first: python src/classifier.py"
        )

    model = load_model(str(model_path))
    log.info(f"Model loaded: {model_path}")

    import pandas as pd

    def run_detection(zip_path: Path) -> dict[str, Any]:
        try:
            features = extract_features(str(zip_path))
            if features is None:
                return {
                    "status": "error",
                    "error": "extract_features returned None",
                    "score": None,
                    "flagged": None,
                }

            row = {}
            for col in FEATURE_COLS:
                val = features.get(col, 0)
                row[col] = int(val) if isinstance(val, bool) else val

            x = pd.DataFrame([row])
            score = float(model.predict_proba(x)[0][1])
            flagged = score >= DETECTION_THRESHOLD

            return {
                "status": "ok",
                "score": score,
                "flagged": bool(flagged),
                **features,
            }
        except Exception as exc:
            return {
                "status": "error",
                "error": str(exc),
                "score": None,
                "flagged": None,
            }

    return run_detection


def download_phase() -> list[Path]:
    """Download all source ZIPs with resume support."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    progress: dict[str, str] = {}
    if PROGRESS_JSON.exists():
        with open(PROGRESS_JSON, "r", encoding="utf-8") as f:
            progress = json.load(f)

    log.info("=" * 60)
    log.info("PHASE 1: DOWNLOADING BENIGN ZIP CORPUS")
    log.info(f"Target dir: {DOWNLOAD_DIR}")
    log.info(f"Source count: {len(BENIGN_SOURCES)}")
    log.info("=" * 60)

    downloaded = 0
    skipped = 0
    failed = 0

    iterator = BENIGN_SOURCES
    if HAS_TQDM:
        iterator = tqdm(BENIGN_SOURCES, desc="Downloading")

    for url, name, category in iterator:
        del category
        if progress.get(name) == "done":
            skipped += 1
            continue

        dest = DOWNLOAD_DIR / f"{name}.zip"
        ok = download_file(url, dest, name)
        progress[name] = "done" if ok else "failed"

        if ok:
            downloaded += 1
        else:
            failed += 1

        with open(PROGRESS_JSON, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)

    log.info(
        f"Download complete: {downloaded} new, {skipped} resumed, {failed} failed"
    )
    return sorted(DOWNLOAD_DIR.glob("*.zip"))


def detection_phase(zip_files: list[Path]) -> list[dict[str, Any]]:
    """Run ZombieGuard over all downloaded benign ZIP files."""
    log.info("=" * 60)
    log.info("PHASE 2: RUNNING ZOMBIEGUARD DETECTOR")
    log.info(f"Files to scan: {len(zip_files)}")
    log.info(f"Threshold: {DETECTION_THRESHOLD}")
    log.info("=" * 60)

    run_detection = load_zombieguard_runner()
    results: list[dict[str, Any]] = []

    iterator: Any = zip_files
    if HAS_TQDM:
        iterator = tqdm(zip_files, desc="Detecting")

    for zip_path in iterator:
        stem = zip_path.stem
        category = "unknown"
        for _url, name, cat in BENIGN_SOURCES:
            if name == stem:
                category = cat
                break

        result = run_detection(zip_path)
        result["file_name"] = stem
        try:
            result["file_path"] = zip_path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            result["file_path"] = str(zip_path)
        result["category"] = category
        result["file_size_mb"] = zip_path.stat().st_size / 1_048_576

        if result["status"] == "ok" and result.get("flagged"):
            log.warning(
                f"  FALSE POSITIVE: {stem} "
                f"(score={result['score']:.4f}, category={category})"
            )
        elif result["status"] == "error":
            log.warning(f"  ERROR on {stem}: {result.get('error', 'unknown')}")

        results.append(result)

    ok_total = sum(1 for r in results if r["status"] == "ok")
    fp_total = sum(1 for r in results if r["status"] == "ok" and r["flagged"])
    err_total = sum(1 for r in results if r["status"] == "error")

    log.info("Detection complete:")
    log.info(f"  Total scanned : {len(results)}")
    log.info(f"  Parsed        : {ok_total}")
    log.info(f"  Parse errors  : {err_total}")
    log.info(f"  False positives: {fp_total}")
    log.info(f"  FP rate       : {fp_total / max(ok_total, 1) * 100:.2f}%")

    return results


def save_results(results: list[dict[str, Any]]) -> None:
    """Write detailed CSV and concise paper-ready summary."""
    log.info("=" * 60)
    log.info("PHASE 3: SAVING RESULTS")
    log.info("=" * 60)

    all_keys: set[str] = set()
    for row in results:
        all_keys.update(row.keys())

    priority_cols = [
        "file_name",
        "category",
        "file_size_mb",
        "status",
        "flagged",
        "score",
        "method_mismatch",
        "data_entropy_shannon",
        "data_entropy_renyi",
        "eocd_count",
        "any_crc_mismatch",
        "is_encrypted",
        "lf_compression_method",
        "cd_compression_method",
        "declared_vs_entropy_flag",
        "lf_unknown_method",
        "suspicious_entry_count",
        "suspicious_entry_ratio",
        "error",
        "file_path",
    ]
    fieldnames = priority_cols + sorted(all_keys - set(priority_cols))

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    log.info(f"CSV saved: {RESULTS_CSV}")

    ok_rows = [r for r in results if r.get("status") == "ok"]
    fp_rows = [r for r in ok_rows if r.get("flagged")]
    err_rows = [r for r in results if r.get("status") == "error"]

    categories: dict[str, dict[str, int]] = {}
    for row in ok_rows:
        cat = str(row.get("category", "unknown"))
        if cat not in categories:
            categories[cat] = {"total": 0, "fp": 0}
        categories[cat]["total"] += 1
        if row.get("flagged"):
            categories[cat]["fp"] += 1

    scores = [float(r["score"]) for r in ok_rows if r.get("score") is not None]
    if scores:
        scores_sorted = sorted(scores)
        p95_idx = max(0, int(0.95 * (len(scores_sorted) - 1)))
        mean_score = sum(scores) / len(scores)
        max_score = max(scores)
        p95_score = scores_sorted[p95_idx]
    else:
        mean_score = 0.0
        max_score = 0.0
        p95_score = 0.0

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fp_rate = len(fp_rows) / max(len(ok_rows), 1) * 100.0

    lines = [
        "=" * 60,
        "ZOMBIEGUARD BENIGN FP CORPUS - RESULTS SUMMARY",
        f"Generated: {now}",
        "=" * 60,
        "",
        "HEADLINE NUMBERS (for paper Section 6.1)",
        "-" * 40,
        f"Total ZIP files scanned    : {len(results)}",
        f"Successfully parsed        : {len(ok_rows)}",
        f"Parse errors               : {len(err_rows)}",
        f"FALSE POSITIVES            : {len(fp_rows)}",
        f"FP rate                    : {fp_rate:.2f}%",
        "",
        "SCORE STATISTICS (parseable files)",
        "-" * 40,
        f"Mean malicious score       : {mean_score:.4f}",
        f"Maximum score              : {max_score:.4f}",
        f"95th percentile score      : {p95_score:.4f}",
        f"Detection threshold        : {DETECTION_THRESHOLD}",
        "",
        "BREAKDOWN BY SOURCE CATEGORY",
        "-" * 40,
    ]

    for cat, counts in sorted(categories.items()):
        marker = " *** FP>0 ***" if counts["fp"] > 0 else ""
        lines.append(
            f"  {cat:<28} n={counts['total']:>4}  FP={counts['fp']}{marker}"
        )

    if fp_rows:
        lines.extend(["", "FALSE POSITIVE DETAILS", "-" * 40])
        for row in fp_rows:
            lines.append(
                f"  {row['file_name']}\n"
                f"    Category : {row.get('category', 'unknown')}\n"
                f"    Score    : {float(row.get('score', 0.0)):.4f}\n"
                f"    method_mismatch      : {row.get('method_mismatch', 'N/A')}\n"
                f"    data_entropy_shannon : {row.get('data_entropy_shannon', 'N/A')}\n"
                f"    eocd_count           : {row.get('eocd_count', 'N/A')}\n"
                f"    any_crc_mismatch     : {row.get('any_crc_mismatch', 'N/A')}"
            )
    else:
        lines.extend([
            "",
            "FALSE POSITIVE DETAILS",
            "-" * 40,
            "  None. Zero false positives across all parsed files.",
        ])

    if err_rows:
        lines.extend(["", "PARSE ERROR DETAILS", "-" * 40])
        for row in err_rows[:20]:
            lines.append(f"  {row.get('file_name', 'unknown')}: {row.get('error', 'unknown')}")

    lines.extend([
        "",
        "PAPER TEXT (paste into Section 6.1)",
        "-" * 40,
        (
            f"We additionally evaluated ZombieGuard on {len(ok_rows)} real-world benign ZIP "
            f"archives collected from publicly released software packages. ZombieGuard produced "
            f"{len(fp_rows)} false positives across all {len(ok_rows)} files "
            f"(FP rate {fp_rate:.2f}%)."
        ),
        "",
        "=" * 60,
    ])

    summary_text = "\n".join(lines)
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(summary_text)

    log.info(f"Summary saved: {SUMMARY_TXT}")
    print("\n" + summary_text)


def main() -> None:
    log.info("ZombieGuard - Benign FP Corpus Experiment")
    log.info(f"Project root: {PROJECT_ROOT}")
    log.info(f"Download dir: {DOWNLOAD_DIR}")

    zip_files = download_phase()
    if not zip_files:
        log.error("No ZIP files available. Check network and source availability.")
        raise SystemExit(1)

    log.info(f"{len(zip_files)} ZIP files ready for detection")

    results = detection_phase(zip_files)
    save_results(results)

    log.info("Experiment complete")
    log.info(f"Results CSV: {RESULTS_CSV}")
    log.info(f"Summary TXT: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
