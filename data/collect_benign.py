import os
import random
import time
import io
import zipfile
from typing import List, Tuple

import requests

OUTPUT_DIR = "data/raw/benign"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_WHEELS_PER_PACKAGE = 3

PACKAGE_ALIASES = {
    "barcode": "python-barcode",
    "dateutil": "python-dateutil",
    "gputil": "GPUtil",
    "httpretty": "HTTPretty",
    "platform": "platformdirs",
}

OFFICE_DOCS = [
    "https://github.com/python-openxml/python-docx/raw/master/tests/fixtures/having-images.docx",
    "https://github.com/python-openxml/python-docx/raw/master/tests/fixtures/plain-text.docx",
    "https://github.com/scanny/python-pptx/raw/master/tests/test_files/presentation.pptx",
    "https://github.com/SheetJS/test_files/raw/master/xlsx/issue.2295.xlsm",
]

# PyPI packages - their wheels are ZIP files
PYPI_PACKAGES = [
    "requests",
    "numpy",
    "pandas",
    "flask",
    "django",
    "fastapi",
    "pydantic",
    "click",
    "rich",
    "httpx",
    "pytest",
    "black",
    "mypy",
    "pylint",
    "setuptools",
    "pip",
    "wheel",
    "tqdm",
    "pillow",
    "scipy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "xgboost",
    "transformers",
    "tokenizers",
    "datasets",
    "huggingface-hub",
    "torch",
    "torchvision",
    "tensorflow",
    "keras",
    "sqlalchemy",
    "alembic",
    "psycopg2",
    "pymongo",
    "redis",
    "celery",
    "boto3",
    "paramiko",
    "cryptography",
    "certifi",
    "charset-normalizer",
    "urllib3",
    "aiohttp",
    "pyyaml",
    "toml",
    "tomli",
    "six",
    "packaging",
    "attrs",
    "typing-extensions",
    "decorator",
    "wrapt",
    "more-itertools",
    "pluggy",
    "iniconfig",
    "colorama",
    "idna",
    "sniffio",
    "anyio",
    "h11",
    "httpcore",
    "starlette",
    "uvicorn",
    "gunicorn",
    "werkzeug",
    "jinja2",
    "markupsafe",
    "itsdangerous",
    "wtforms",
    "marshmallow",
    "pydantic-settings",
    "annotated-types",
    "python-dotenv",
    "python-multipart",
    "email-validator",
    "httptools",
    "websockets",
    "watchfiles",
    "python-jose",
    "passlib",
    "bcrypt",
    "pyotp",
    "qrcode",
    "barcode",
    "fpdf2",
    "reportlab",
    "openpyxl",
    "xlrd",
    "xlwt",
    "python-docx",
    "python-pptx",
    "tabulate",
    "prettytable",
    "rich-argparse",
    "typer",
    "fire",
    "loguru",
    "structlog",
    "sentry-sdk",
    "rollbar",
    "bugsnag",
    "psutil",
    "py-cpuinfo",
    "gputil",
    "distro",
    "platform",
    "arrow",
    "pendulum",
    "dateutil",
    "pytz",
    "babel",
    "humanize",
    "inflect",
    "unidecode",
    "ftfy",
    "chardet",
    "lxml",
    "beautifulsoup4",
    "html5lib",
    "cssselect",
    "pyquery",
    "scrapy",
    "mechanize",
    "playwright",
    "selenium",
    "pyppeteer",
    "pillow",
    "imageio",
    "scikit-image",
    "opencv-python-headless",
    "pydub",
    "librosa",
    "soundfile",
    "mutagen",
    "tinytag",
    "networkx",
    "igraph",
    "graphviz",
    "pyvis",
    "plotly",
    "bokeh",
    "altair",
    "streamlit",
    "gradio",
    "panel",
    "nltk",
    "spacy",
    "gensim",
    "textblob",
    "langdetect",
    "sympy",
    "statsmodels",
    "pingouin",
    "pymc",
    "arviz",
    "sqlmodel",
    "tortoise-orm",
    "databases",
    "aiosqlite",
    "motor",
    "minio",
    "google-cloud-storage",
    "azure-storage-blob",
    "s3fs",
    "parameterized",
    "hypothesis",
    "faker",
    "factory-boy",
    "responses",
    "freezegun",
    "time-machine",
    "vcrpy",
    "respx",
    "httpretty",
    "nox",
    "tox",
    "coverage",
    "codecov",
    "coveralls",
    "sphinx",
    "mkdocs",
    "pdoc",
    "pydoc-markdown",
    "autodoc",
    "bumpversion",
    "twine",
    "flit",
    "hatch",
    "pdm",
    "pre-commit",
    "commitizen",
    "semantic-release",
    "changelog",
    "towncrier",
    "pynput",
    "keyboard",
    "mouse",
    "pyautogui",
    "pygetwindow",
    "schedule",
    "apscheduler",
    "rq",
    "dramatiq",
    "huey",
    "kombu",
    "pika",
    "aio-pika",
    "nats-py",
    "confluent-kafka",
    "grpcio",
    "protobuf",
    "thrift",
    "avro-python3",
    "fastavro",
    "pyarrow",
    "fastparquet",
    "h5py",
    "zarr",
    "netcdf4",
    "dask",
    "ray",
    "joblib",
    "multiprocess",
    "pathos",
    "numba",
    "cython",
    "cffi",
    "ctypes",
    "pybind11",
    "mypy-extensions",
    "types-requests",
    "types-pyyaml",
    "types-redis",
]


def _normalized_package_name(package_name: str) -> str:
    return PACKAGE_ALIASES.get(package_name, package_name)


def get_pypi_wheel_urls(package_name: str, max_count: int = 3) -> List[Tuple[str, str]]:
    """Get up to max_count wheel URLs as (version, url) from latest releases first."""
    normalized = _normalized_package_name(package_name)
    url = f"https://pypi.org/pypi/{normalized}/json"
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
        releases = data.get("releases", {})

        versions = sorted(
            releases.keys(),
            key=lambda value: tuple(int(part) if part.isdigit() else 0 for part in value.replace("-", ".").split(".")),
            reverse=True,
        )

        found = []
        for version in versions:
            files = releases.get(version, [])
            wheel = next((item for item in files if item.get("packagetype") == "bdist_wheel"), None)
            if wheel and wheel.get("url"):
                found.append((version, wheel["url"]))
                if len(found) >= max_count:
                    break
        return found
    except Exception as exc:
        print(f"Failed to get wheel URLs for {package_name}: {exc}")
    return []


def download_file(url: str, output_path: str) -> bool:
    try:
        resp = requests.get(url, timeout=60, stream=True)
        if resp.status_code == 200:
            with open(output_path, "wb") as file:
                for chunk in resp.iter_content(chunk_size=8192):
                    file.write(chunk)
            return True
    except Exception as exc:
        print(f"Download failed: {exc}")
    return False


def download_office_docs() -> tuple[int, int]:
    downloaded = 0
    failed = 0
    print("\nDownloading Office-document-like benign samples...")

    for i, url in enumerate(OFFICE_DOCS):
        ext = ".docx" if i % 2 == 0 else ".xlsx"
        output_path = os.path.join(OUTPUT_DIR, f"office_sample_{i:03d}{ext}.zip")
        if os.path.exists(output_path):
            print(f"  Exists: office_sample_{i:03d}{ext}")
            downloaded += 1
            continue

        print(f"  Downloading office sample {i + 1}/{len(OFFICE_DOCS)}")
        if download_file(url, output_path):
            downloaded += 1
            print(f"  Saved office sample ({downloaded} total)")
        else:
            failed += 1
    return downloaded, failed


def create_small_benign_zips(output_dir: str, count: int = 400):
    """
    Create small benign ZIPs matching the size profile of synthetic
    malicious samples. Content is clean text - no malicious structure.
    """
    os.makedirs(output_dir, exist_ok=True)

    words = [
        "report",
        "invoice",
        "meeting",
        "summary",
        "update",
        "notes",
        "draft",
        "review",
        "final",
        "version",
    ]
    generated = 0

    for i in range(count):
        # Random small benign content - similar size to malicious samples
        size = random.randint(256, 8192)
        content = (" ".join(random.choices(words, k=size // 10))).encode()

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            filename = f"document_{i:04d}.txt"
            zf.writestr(filename, content)

        out_path = os.path.join(output_dir, f"small_benign_{i:04d}.zip")
        with open(out_path, "wb") as file:
            file.write(buffer.getvalue())
        generated += 1

    print(f"Created {generated} small benign ZIPs")
    return generated


if __name__ == "__main__":
    downloaded = 0
    failed = 0

    print("Downloading PyPI wheels as benign ZIP samples...")
    for package in PYPI_PACKAGES:
        wheel_urls = get_pypi_wheel_urls(package, max_count=MAX_WHEELS_PER_PACKAGE)
        if not wheel_urls:
            print(f"  No wheel found for: {package}")
            failed += 1
            continue

        for version, url in wheel_urls:
            output_path = os.path.join(OUTPUT_DIR, f"pypi_{package}_{version}.zip")
            if os.path.exists(output_path):
                print(f"  Exists: {package}=={version}")
                downloaded += 1
                continue

            print(f"  Downloading: {package}=={version}")
            success = download_file(url, output_path)
            if success:
                downloaded += 1
                print(f"  Saved ({downloaded} total)")
            else:
                failed += 1

            time.sleep(0.2)

    office_downloaded, office_failed = download_office_docs()
    downloaded += office_downloaded
    failed += office_failed

    create_small_benign_zips(OUTPUT_DIR, count=400)

    print(f"\nDownloaded: {downloaded} benign samples")
    print(f"Failed: {failed}")
    print(f"Total benign samples: {len(os.listdir(OUTPUT_DIR))}")