"""Download HaluMem dataset from Hugging Face.

Source dataset:
  https://huggingface.co/datasets/IAAR-Shanghai/HaluMem

The dataset provides JSONL files directly in the Hugging Face repository.

Output files:
  - HaluMem-Medium.jsonl   (~32 MB,  20 users, ~160K tokens/user)
  - HaluMem-Long.jsonl     (~102 MB, 20 users, ~1M tokens/user)

Usage:
    python prepare_halumem.py                        # download medium (used by default)
    python prepare_halumem.py --variant medium long  # download all variants
    python prepare_halumem.py --force                # overwrite existing files

License: CC BY-NC-ND 4.0
"""

import argparse
import json
import os
import sys
import urllib.request

REPO_BASE = "https://huggingface.co/datasets/IAAR-Shanghai/HaluMem/resolve/main"

VARIANT_MAP = {
    "medium": "HaluMem-Medium.jsonl",
    "long":   "HaluMem-Long.jsonl",
}

ALL_VARIANTS = list(VARIANT_MAP.keys())
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def validate_jsonl(path: str) -> int:
    rows = 0
    try:
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise RuntimeError(f"line {line_no} is not a JSON object")
                rows += 1
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{path} is not valid JSONL: {exc}") from exc
    if rows == 0:
        raise RuntimeError(f"{path} contains no JSONL records")
    return rows


def download_url(url: str, output_path: str) -> None:
    tmp_path = f"{output_path}.tmp"
    try:
        urllib.request.urlretrieve(url, tmp_path)
        validate_jsonl(tmp_path)
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def download_variant(variant: str, force: bool = False) -> None:
    filename = VARIANT_MAP[variant]
    output_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(output_path) and not force:
        rows = validate_jsonl(output_path)
        print(f"  [{variant}] {filename} already exists and has {rows} JSONL records, skipping (use --force to overwrite)")
        return

    url = f"{REPO_BASE}/{filename}"
    print(f"  [{variant}] Downloading {filename} ...")

    download_url(url, output_path)
    rows = validate_jsonl(output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  [{variant}] Saved {filename} ({rows} JSONL records, {size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download HaluMem dataset from Hugging Face."
    )
    parser.add_argument(
        "--variant",
        nargs="+",
        choices=ALL_VARIANTS,
        default=["medium"],
        help="Variant(s) to download (default: medium). Evaluation uses medium by default.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Variants to download: {', '.join(args.variant)}")
    print()

    try:
        for variant in args.variant:
            download_variant(variant, force=args.force)
            print()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
