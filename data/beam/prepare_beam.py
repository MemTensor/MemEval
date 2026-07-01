"""Download BEAM datasets from Hugging Face and convert them to JSON Lines.

Source datasets (Parquet format):
  - https://huggingface.co/datasets/Mohammadta/BEAM       (100K / 500K / 1M)
  - https://huggingface.co/datasets/Mohammadta/BEAM-10M   (10M)

Output files (JSON Lines, one conversation per line):
  - beam_100k.json      (~14 MB,  20 conversations)
  - beam_500k.json      (~86 MB,  35 conversations)
  - beam_1m.json        (~172 MB, 35 conversations)
  - beam_10m_10m.json   (~979 MB, 10 conversations)

Usage:
    pip install datasets
    python prepare_beam.py                   # download 100k, matching runner default
    python prepare_beam.py --scale 100k      # download a single scale
    python prepare_beam.py --scale 100k 500k # download selected scales
    python prepare_beam.py --scale all       # download all scales

License: CC BY-SA 4.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

SPLIT_MAP = {
    "100k": {"repo": "Mohammadta/BEAM", "split": "100K", "output": "beam_100k.json"},
    "500k": {"repo": "Mohammadta/BEAM", "split": "500K", "output": "beam_500k.json"},
    "1m":   {"repo": "Mohammadta/BEAM", "split": "1M",   "output": "beam_1m.json"},
    "10m":  {"repo": "Mohammadta/BEAM-10M", "split": "10M", "output": "beam_10m_10m.json"},
}

ALL_SCALES = list(SPLIT_MAP.keys())
SCALE_CHOICES = [*ALL_SCALES, "all"]


def row_to_dict(row):
    """Recursively convert a datasets Row to a plain JSON-serializable dict."""
    if isinstance(row, dict):
        return {k: row_to_dict(v) for k, v in row.items()}
    if isinstance(row, list):
        return [row_to_dict(v) for v in row]
    return row


def download_scale(scale: str, output_dir: str, force: bool = False) -> None:
    cfg = SPLIT_MAP[scale]
    output_path = os.path.join(output_dir, cfg["output"])

    if os.path.exists(output_path) and not force:
        print(f"  [{scale}] {cfg['output']} already exists, skipping (use --force to overwrite)")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package is required. Install it with:")
        print("  pip install datasets")
        sys.exit(1)

    print(f"  [{scale}] Downloading {cfg['repo']} split={cfg['split']} ...")
    ds = load_dataset(cfg["repo"], split=cfg["split"])

    print(f"  [{scale}] Converting {len(ds)} conversations to JSON Lines ...")
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            obj = row_to_dict(row)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  [{scale}] Saved {cfg['output']} ({len(ds)} conversations, {size_mb:.1f} MB)")


def resolve_scales(values: list[str]) -> list[str]:
    if "all" in values:
        return ALL_SCALES
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download BEAM datasets from Hugging Face and convert them to JSON Lines."
    )
    parser.add_argument(
        "--scale",
        nargs="+",
        choices=SCALE_CHOICES,
        default=["100k"],
        help="Scale(s) to download (default: 100k). Use --scale all for every scale.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(os.path.abspath(__file__))
    scales = resolve_scales(args.scale)

    print(f"Output directory: {output_dir}")
    print(f"Scales to download: {', '.join(scales)}")
    print()

    for scale in scales:
        download_scale(scale, output_dir, force=args.force)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
