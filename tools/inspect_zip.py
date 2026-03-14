import argparse
from pathlib import Path


def inspect_zip_header(path: Path) -> None:
    with path.open("rb") as file:
        header = file.read(30)

    print(f"File: {path}")
    print(f"Read bytes: {len(header)}")
    print("Hex dump (first 30 bytes):")

    for offset in range(0, len(header), 16):
        chunk = header[offset : offset + 16]
        hex_bytes = " ".join(f"{byte:02X}" for byte in chunk)
        print(f"{offset:04X}: {hex_bytes}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the first 30 bytes of a ZIP file local header."
    )
    parser.add_argument("zip_path", type=Path, help="Path to .zip file")
    args = parser.parse_args()

    inspect_zip_header(args.zip_path)


if __name__ == "__main__":
    main()
