#!/usr/bin/env python3
"""Delete OmniMemEval ingestion data for a version.

This tool uses the same user-id schemes as the ingestion scripts. It is
destructive by default only when ``--yes`` is passed; use ``--dry-run`` to
inspect the target ids without calling backend delete APIs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from contextlib import suppress
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = SCRIPT_DIR
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)

sys.path.insert(0, SCRIPTS_DIR)

from dotenv import load_dotenv

from client_factory import SUPPORTED_LIBS, create_client

PM_BENCHMARK_CSV = os.path.join(
    PROJECT_DIR,
    "data",
    "personamem_v2",
    "benchmark",
    "text",
    "benchmark.csv",
)
PM_CHAT_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "personamem_v2",
    "data",
    "chat_history_32k",
)
HALUMEM_MEDIUM = os.path.join(PROJECT_DIR, "data", "halumem", "HaluMem-Medium.jsonl")
HALUMEM_LONG = os.path.join(PROJECT_DIR, "data", "halumem", "HaluMem-Long.jsonl")
LME_JSON = os.path.join(
    PROJECT_DIR,
    "data",
    "longmemeval",
    "longmemeval_s_cleaned.json",
)
LOCOMO_JSON = os.path.join(PROJECT_DIR, "data", "locomo", "locomo10.json")
BEAM_SCALE_FILES = {
    "100k": "beam_100k.json",
    "500k": "beam_500k.json",
    "1m": "beam_1m.json",
    "10m": "beam_10m_10m.json",
}


def _delete_user_generic(
    client: Any,
    lib_name: str,
    user_id: str,
    dry_run: bool,
) -> None:
    if dry_run:
        print(f"  would delete user: {user_id}")
        return
    try:
        if getattr(client, "delete_all", None) and "mem0" in lib_name:
            client.delete_all(user_id)
            print(f"  deleted user: {user_id}")
            return
        if getattr(client, "delete", None):
            client.delete(user_id)
            print(f"  deleted user: {user_id}")
            return
        if getattr(client, "delete_user", None):
            client.delete_user(user_id)
            print(f"  deleted user: {user_id}")
            return
        print(f"  ⚠ no delete method available for {user_id!r}")
    except Exception as exc:
        print(f"  ⚠ delete failed for {user_id!r}: {exc}")


def _zep_use_group() -> bool:
    return os.environ.get("ZEP_USE_GROUP", "true").lower() in ("true", "1", "yes")


def _everos_use_group() -> bool:
    return os.environ.get("EVEROS_USE_GROUP", "true").lower() in ("true", "1", "yes")


def cleanup_locomo(client: Any, lib_name: str, version: str, dry_run: bool) -> None:
    if not os.path.isfile(LOCOMO_JSON):
        print(f"  ⚠ missing {LOCOMO_JSON}", file=sys.stderr)
        return

    with open(LOCOMO_JSON) as f:
        locomo_df = json.load(f)
    n = len(locomo_df)

    if lib_name == "zep" and _zep_use_group():
        for conv_idx in range(n):
            graph_id = f"locomo_exp_group_{conv_idx}_{version}"
            if dry_run:
                print(f"  would delete zep graph: {graph_id}")
            elif hasattr(client, "sdk_graph_delete"):
                with suppress(Exception):
                    client.sdk_graph_delete(graph_id)
                print(f"  zep graph delete: {graph_id}")
        return

    if lib_name == "everos" and _everos_use_group():
        for conv_idx in range(n):
            group_id = f"locomo_exp_user_{conv_idx}_speaker_a_{version}"
            if dry_run:
                print(f"  would delete everos group: {group_id}")
            elif getattr(client, "delete_group", None):
                with suppress(Exception):
                    client.delete_group(group_id)
                print(f"  everos delete_group: {group_id}")
        return

    for conv_idx in range(n):
        a = f"locomo_exp_user_{conv_idx}_speaker_a_{version}"
        b = f"locomo_exp_user_{conv_idx}_speaker_b_{version}"
        if lib_name == "supermemory":
            _delete_user_generic(client, lib_name, a, dry_run=dry_run)
            continue
        _delete_user_generic(client, lib_name, a, dry_run=dry_run)
        _delete_user_generic(client, lib_name, b, dry_run=dry_run)


def _pm_persona_ids() -> list[int]:
    if not os.path.isfile(PM_BENCHMARK_CSV):
        print(f"  ⚠ missing {PM_BENCHMARK_CSV}", file=sys.stderr)
        return []

    personas: dict[int, None] = {}
    with open(PM_BENCHMARK_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            personas[int(row["persona_id"])] = None
    index: set[int] = set()
    for filepath in glob.glob(os.path.join(PM_CHAT_DIR, "*.json")):
        base = os.path.basename(filepath)
        parts = base.split("_persona")
        if len(parts) == 2:
            index.add(int(parts[1].replace(".json", "")))
    if not index:
        print(f"  ⚠ missing PersonaMem v2 chat histories under {PM_CHAT_DIR}", file=sys.stderr)
    return sorted(pid for pid in personas if pid in index)


def cleanup_pm(client: Any, lib_name: str, version: str, dry_run: bool) -> None:
    for pid in _pm_persona_ids():
        uid = f"pm_exper_user_{pid}_{version}"
        if lib_name == "zep":
            if dry_run:
                print(f"  would delete zep user: {uid}")
            else:
                with suppress(Exception):
                    client.delete_user(uid)
                print(f"  zep delete_user: {uid}")
            continue
        _delete_user_generic(client, lib_name, uid, dry_run=dry_run)


def cleanup_hm(
    client: Any,
    lib_name: str,
    version: str,
    halumem_path: str,
    dry_run: bool,
) -> None:
    users = []
    with open(halumem_path) as f:
        for line in f:
            line = line.strip()
            if line:
                users.append(json.loads(line))
    for u in users:
        uid = f"hm_exp_user_{version}_{u['uuid']}"
        _delete_user_generic(client, lib_name, uid, dry_run=dry_run)


def cleanup_lme(client: Any, lib_name: str, version: str, dry_run: bool) -> None:
    if not os.path.isfile(LME_JSON):
        print(f"  ⚠ missing {LME_JSON}", file=sys.stderr)
        return

    with open(LME_JSON) as f:
        lme_df = json.load(f)
    n = len(lme_df) if isinstance(lme_df, list) else len(lme_df)
    for conv_idx in range(n):
        uid = f"lme_exper_user_{version}_{conv_idx}"
        _delete_user_generic(client, lib_name, uid, dry_run=dry_run)


def _load_beam_conv_ids(scales: list[str]) -> list[str]:
    ids: list[str] = []
    data_dir = os.path.join(PROJECT_DIR, "data", "beam")
    for scale in scales:
        fn = BEAM_SCALE_FILES.get(scale)
        if not fn:
            print(f"  ⚠ unknown beam scale {scale!r}, skip", file=sys.stderr)
            continue
        path = os.path.join(data_dir, fn)
        if not os.path.isfile(path):
            print(f"  ⚠ beam file missing: {path}", file=sys.stderr)
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                conv = json.loads(line)
                ids.append(str(conv["conversation_id"]))
    return ids


def cleanup_beam(
    client: Any,
    lib_name: str,
    version: str,
    beam_scales: list[str],
    dry_run: bool,
) -> None:
    for conv_id in _load_beam_conv_ids(beam_scales):
        uid = f"beam_exp_user_{version}_{conv_id}"
        _delete_user_generic(client, lib_name, uid, dry_run=dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delete OmniMemEval benchmark data for a given version.",
    )
    parser.add_argument("--lib", required=True, choices=SUPPORTED_LIBS)
    parser.add_argument(
        "--env",
        required=True,
        help="Dotenv path (shell sets via OMNIMEMEVAL_ENV_FILE)",
    )
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print target ids without deleting backend data.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm destructive deletion. Required unless --dry-run is set.",
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated: locomo,lme,beam,pmv2,hm or all",
    )
    parser.add_argument(
        "--beam-scale",
        action="append",
        dest="beam_scales",
        metavar="SCALE",
        help=(
            "100k, 500k, 1m, 10m (repeatable). "
            "Default if beam is selected: 100k"
        ),
    )
    parser.add_argument(
        "--halumem-variant",
        choices=("medium", "long"),
        default="medium",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.yes:
        print(
            "Error: cleanup is destructive. Re-run with --dry-run to inspect "
            "targets, or pass --yes to confirm deletion.",
            file=sys.stderr,
        )
        return 2

    env_path = args.env
    if not os.path.isfile(env_path):
        env_path = os.path.join(PROJECT_DIR, args.env)
    if not os.path.isfile(env_path):
        print(f"Error: env file not found: {args.env}", file=sys.stderr)
        return 1

    load_dotenv(env_path, override=True)
    os.environ["OMNIMEMEVAL_ENV_FILE"] = os.path.abspath(env_path)

    raw = args.datasets.strip().lower()
    if raw == "all":
        ds = {"locomo", "lme", "beam", "pmv2", "hm"}
    else:
        ds = {x.strip() for x in raw.split(",") if x.strip()}
        valid = {"locomo", "lme", "beam", "pmv2", "hm"}
        bad = ds - valid
        if bad:
            print(f"Error: unknown dataset(s): {bad}", file=sys.stderr)
            return 1

    beam_scales = args.beam_scales or (["100k"] if "beam" in ds else [])

    mode = "DRY RUN" if args.dry_run else "DELETE"
    print(f"Cleanup mode: {mode}")
    print(f"Cleanup: lib={args.lib} version={args.version} env={env_path}")
    print(f"Datasets: {', '.join(sorted(ds))}")
    if "beam" in ds:
        print(f"BEAM scales: {beam_scales}")

    client = None if args.dry_run else create_client(args.lib)

    if "locomo" in ds:
        print("\n[LoCoMo]")
        cleanup_locomo(client, args.lib, args.version, args.dry_run)
    if "pmv2" in ds:
        print("\n[PersonaMem v2]")
        cleanup_pm(client, args.lib, args.version, args.dry_run)
    if "hm" in ds:
        print("\n[HaluMem]")
        hm_path = HALUMEM_LONG if args.halumem_variant == "long" else HALUMEM_MEDIUM
        if not os.path.isfile(hm_path):
            print(f"  ⚠ missing {hm_path}", file=sys.stderr)
        else:
            cleanup_hm(client, args.lib, args.version, hm_path, args.dry_run)
    if "lme" in ds:
        print("\n[LongMemEval]")
        cleanup_lme(client, args.lib, args.version, args.dry_run)
    if "beam" in ds:
        print("\n[BEAM]")
        cleanup_beam(client, args.lib, args.version, beam_scales, args.dry_run)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
