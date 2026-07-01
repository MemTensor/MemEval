"""Download PersonaMem-v2 from Hugging Face for OmniMemEval.

Dataset: https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2

OmniMemEval needs ``benchmark/text/benchmark.csv`` plus every JSON path listed in the
``chat_history_32k_link`` column (~200 unique files under ``data/chat_history_32k/``).
Those files are downloaded with ``hf_hub_download``, which respects ``HF_ENDPOINT``
(e.g. ``https://hf-mirror.com``) and avoids brittle bulk tree sync when
``huggingface.co`` is unreachable.

Optional extras (``--include``) use ``snapshot_download`` for whole subtrees.

Usage::

    pip install huggingface_hub
    export HF_ENDPOINT=https://hf-mirror.com   # if needed
    export HF_HUB_DOWNLOAD_TIMEOUT=300         # optional; large JSON defaults below
    python prepare_personamem.py
    python prepare_personamem.py --include 128k raw-data
    python prepare_personamem.py --force
    python prepare_personamem.py --verify-chat-32k

License: MIT
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ID = "bowen-upenn/PersonaMem-v2"
BENCHMARK_CSV = "benchmark/text/benchmark.csv"
REQUIRED_BENCHMARK_COLUMNS = {
    "persona_id",
    "chat_history_32k_link",
    "user_query",
    "pref_type",
    "topic_query",
    "correct_answer",
    "incorrect_answers",
}

OPTIONAL_PATTERNS = {
    "all-csv": ["benchmark/text/*.csv", "column_descriptions.md"],
    "raw-data": ["data/raw_data/*.json"],
    "128k": ["data/chat_history_128k/*.json"],
    "irrelevant": ["combined_irrelevant_data.json"],
}


def _require_hf_hub():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download  # noqa: F401

        return hf_hub_download, snapshot_download
    except ImportError:
        print("Error: install huggingface_hub:  pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)


def verify_chat_history_json_dir(dirpath: str) -> tuple[int, list[str]]:
    """Return (bad_count, sample error lines)."""
    import glob

    bad: list[str] = []
    if not os.path.isdir(dirpath):
        return 0, []
    for path in sorted(glob.glob(os.path.join(dirpath, "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                head = f.read(8192)
            if not head.strip():
                bad.append(f"{path}: empty")
                continue
            if head.lstrip().startswith("<"):
                bad.append(f"{path}: not JSON (HTML or error page)")
                continue
            with open(path, encoding="utf-8") as f:
                json.load(f)
        except json.JSONDecodeError as e:
            bad.append(f"{path}: {e}")
    return len(bad), bad[:20]


def validate_benchmark_csv(path: str) -> int:
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise RuntimeError("missing CSV header")
            missing = sorted(REQUIRED_BENCHMARK_COLUMNS.difference(reader.fieldnames))
            if missing:
                raise RuntimeError(f"missing required column(s): {', '.join(missing)}")
            rows = sum(1 for _ in reader)
    except OSError as exc:
        raise RuntimeError(f"cannot read {path}: {exc}") from exc
    if rows == 0:
        raise RuntimeError(f"{path} contains no benchmark rows")
    return rows


def _purge_invalid_chat_json(dirpath: str) -> int:
    """Remove *.json that are empty, HTML, or invalid JSON."""
    import glob

    removed = 0
    for path in glob.glob(os.path.join(dirpath, "*.json")):
        try:
            with open(path, encoding="utf-8") as fh:
                head = fh.read(4096)
            if not head.strip() or head.lstrip().startswith("<"):
                raise ValueError("bad")
            with open(path, encoding="utf-8") as fh:
                json.load(fh)
        except (json.JSONDecodeError, ValueError, OSError):
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
    return removed


def _unique_benchmark_chat_32k_paths() -> list[str]:
    benchmark_csv = os.path.join(OUTPUT_DIR, BENCHMARK_CSV)
    validate_benchmark_csv(benchmark_csv)
    seen: list[str] = []
    found: set[str] = set()
    with open(benchmark_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            link = row["chat_history_32k_link"].strip().replace("\\", "/")
            if link and link not in found:
                found.add(link)
                seen.append(link)
    return seen


def _file_is_valid_json(path: str) -> bool:
    try:
        with open(path, encoding="utf-8") as fh:
            head = fh.read(4096)
        if not head.strip() or head.lstrip().startswith("<"):
            return False
        with open(path, encoding="utf-8") as fh:
            json.load(fh)
        return True
    except (json.JSONDecodeError, OSError):
        return False


def _hf_download_one(
    hf_hub_download,
    filename: str,
    *,
    force: bool,
    max_attempts: int = 5,
) -> None:
    from huggingface_hub.errors import RemoteEntryNotFoundError

    for attempt in range(1, max_attempts + 1):
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=filename,
                local_dir=OUTPUT_DIR,
                force_download=force,
            )
            return
        except RemoteEntryNotFoundError:
            raise
        except Exception as exc:
            if attempt >= max_attempts:
                raise
            wait = min(2**attempt, 60)
            print(f"  retry {attempt}/{max_attempts} {filename}: {exc!r} (sleep {wait}s)", file=sys.stderr)
            time.sleep(wait)


def download_eval_bundle(*, force: bool) -> None:
    """Download benchmark CSV + every chat_history_32k JSON referenced by it."""
    hf_hub_download, _ = _require_hf_hub()
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")

    print("Downloading benchmark CSV ...")
    _hf_download_one(hf_hub_download, BENCHMARK_CSV, force=force)
    rows = validate_benchmark_csv(os.path.join(OUTPUT_DIR, BENCHMARK_CSV))
    print(f"  [OK] {BENCHMARK_CSV} has {rows} benchmark rows")

    chat_dir = os.path.join(OUTPUT_DIR, "data", "chat_history_32k")
    os.makedirs(chat_dir, exist_ok=True)
    n_rm = _purge_invalid_chat_json(chat_dir)
    if n_rm:
        print(f"  removed {n_rm} invalid JSON file(s) under data/chat_history_32k/")

    rels = _unique_benchmark_chat_32k_paths()
    print(f"Downloading {len(rels)} chat_history_32k JSON files (from benchmark links) ...")
    for i, rel in enumerate(rels, 1):
        dest = os.path.join(OUTPUT_DIR, rel)
        if not force and os.path.isfile(dest) and _file_is_valid_json(dest):
            continue
        _hf_download_one(hf_hub_download, rel, force=True)
        if i % 50 == 0 or i == len(rels):
            print(f"  ... {i}/{len(rels)}")

    dirpath = os.path.join(OUTPUT_DIR, "data", "chat_history_32k")
    n_bad, samples = verify_chat_history_json_dir(dirpath)
    if n_bad:
        print(f"Error: {n_bad} invalid JSON file(s) under data/chat_history_32k/", file=sys.stderr)
        for line in samples:
            print(f"  {line}", file=sys.stderr)
        sys.exit(1)
    print("  [OK] all chat_history_32k JSON files parse as JSON")


def download_optional_includes(include: list[str], *, force: bool) -> None:
    _, snapshot_download = _require_hf_hub()
    patterns: list[str] = []
    for key in include:
        patterns.extend(OPTIONAL_PATTERNS[key])
    print(f"Downloading optional paths via snapshot_download: {patterns}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=OUTPUT_DIR,
        allow_patterns=patterns,
        force_download=force,
    )


def print_summary() -> None:
    benchmark_csv = os.path.join(OUTPUT_DIR, BENCHMARK_CSV)
    if os.path.isfile(benchmark_csv):
        mb = os.path.getsize(benchmark_csv) / (1024 * 1024)
        try:
            rows = validate_benchmark_csv(benchmark_csv)
            print(f"  [OK] {BENCHMARK_CSV} ({rows} rows, {mb:.1f} MB)")
        except RuntimeError as exc:
            print(f"  [INVALID] {BENCHMARK_CSV}: {exc}")
    else:
        print(f"  [MISSING] {BENCHMARK_CSV}")

    for dirname in ("chat_history_32k", "chat_history_128k"):
        dirpath = os.path.join(OUTPUT_DIR, "data", dirname)
        if os.path.isdir(dirpath):
            n = len([f for f in os.listdir(dirpath) if f.endswith(".json")])
            print(f"  [OK] data/{dirname}/ ({n} json files)")

    raw_dir = os.path.join(OUTPUT_DIR, "data", "raw_data")
    if os.path.isdir(raw_dir):
        n = len([f for f in os.listdir(raw_dir) if f.endswith(".json")])
        print(f"  [OK] data/raw_data/ ({n} json files)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PersonaMem-v2 for OmniMemEval.")
    parser.add_argument(
        "--include",
        nargs="+",
        choices=list(OPTIONAL_PATTERNS.keys()),
        default=None,
        help="Optional dataset subtrees (uses snapshot_download)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files when re-downloading")
    parser.add_argument(
        "--verify-chat-32k",
        action="store_true",
        help="Only verify data/chat_history_32k/*.json then exit",
    )
    args = parser.parse_args()

    if args.verify_chat_32k:
        dirpath = os.path.join(OUTPUT_DIR, "data", "chat_history_32k")
        if not os.path.isdir(dirpath):
            print(f"Verification failed: {dirpath} does not exist")
            sys.exit(1)
        n_bad, samples = verify_chat_history_json_dir(dirpath)
        if n_bad:
            print(f"Verification failed: {n_bad} bad file(s)")
            for line in samples:
                print(f"  {line}")
            sys.exit(1)
        print(f"OK: all JSON valid under {dirpath}")
        return

    print(f"PersonaMem-v2 -> {OUTPUT_DIR}")
    print(f"  repo={REPO_ID}  HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '(unset)')}")
    download_eval_bundle(force=args.force)
    if args.include:
        download_optional_includes(args.include, force=args.force)
    print()
    print("Summary:")
    print_summary()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
