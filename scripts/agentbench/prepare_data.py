from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data" / "agentbench"


DATASET_MARKERS = (
    "BrowseComp-Plus",
    "Reasoning & Problem Decomposition",
    "gdpval",
    "livecode",
    "swebench",
)


def _replace_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _resolve_dataset_dir(source_root: Path) -> Path:
    candidates = (source_root, source_root / "data")
    for candidate in candidates:
        if candidate.is_dir() and any((candidate / marker).exists() for marker in DATASET_MARKERS):
            return candidate
    raise FileNotFoundError(
        f"AgentBench dataset directory not found under {source_root}. "
        "Download EverMind-AI/EvoAgentBench into data/agentbench first."
    )


def ensure_data_link(source_root: Path, *, force: bool = False) -> Path:
    source_data = _resolve_dataset_dir(source_root)

    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "agentbench"

    if source_data.resolve() == target.resolve():
        return target

    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source_data.resolve():
            return target
        if not force:
            raise FileExistsError(
                f"{target} already exists and does not point to {source_data}. "
                "Use --force to replace it."
            )
        _replace_path(target)

    target.symlink_to(source_data, target_is_directory=True)
    return target


def ensure_livecodebench_copy(source_root: Path, *, force: bool = False) -> Path:
    source_repo = source_root / "LiveCodeBench"
    if not source_repo.is_dir():
        raise FileNotFoundError(f"LiveCodeBench repo not found: {source_repo}")

    target = ROOT / "LiveCodeBench"
    if target.exists() or target.is_symlink():
        if not force:
            return target
        _replace_path(target)

    ignore = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".pytest_cache")
    shutil.copytree(source_repo, target, ignore=ignore)
    return target


def ensure_legacy_omnimath_link(source_root: Path, *, force: bool = False) -> Path:
    source_dir = _resolve_dataset_dir(source_root) / "Reasoning & Problem Decomposition"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"OmniMath source directory not found: {source_dir}")

    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "omnimath"

    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source_dir.resolve():
            return target
        if not force:
            return target
        _replace_path(target)

    target.symlink_to(source_dir, target_is_directory=True)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OmniMemEval AgentBench data/code assets.")
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="Path to downloaded EverMind-AI/EvoAgentBench data. Defaults to data/agentbench.",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing prepared paths.")
    parser.add_argument(
        "--skip-livecodebench",
        action="store_true",
        help="Only prepare data links; do not copy LiveCodeBench.",
    )
    parser.add_argument(
        "--skip-legacy-omnimath",
        action="store_true",
        help="Do not create the backward-compatible data/omnimath link.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"AgentBench source directory not found: {source_root}")

    data_link = ensure_data_link(source_root, force=args.force)
    print(f"data/agentbench -> {data_link.resolve()}")

    if not args.skip_legacy_omnimath:
        omnimath_link = ensure_legacy_omnimath_link(source_root, force=args.force)
        print(f"data/omnimath -> {omnimath_link.resolve()}")

    if not args.skip_livecodebench:
        livecodebench = ensure_livecodebench_copy(source_root, force=args.force)
        print(f"LiveCodeBench copied/prepared at {livecodebench}")


if __name__ == "__main__":
    main()
