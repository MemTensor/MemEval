#!/usr/bin/env python3
"""HaluMem streaming add-search-delete pipeline.

Each HaluMem user is treated as one independent evaluation unit:

1. add all sessions for one user;
2. search all benchmark questions for that user;
3. save the normal HaluMem search-result JSON shape;
4. delete the user from the memory service.

The output is intentionally compatible with hm_responses.py, hm_eval.py,
hm_metric.py, and hm_report.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client_factory import DEFAULT_LIB, SUPPORTED_LIBS, create_client  # noqa: E402
from halumem.hm_common import (  # noqa: E402
    STATUS_SKIPPED,
    build_search_entry,
    error_payload,
    iter_questions,
    load_halumem_data,
    status_counts,
    user_id_for,
)
from halumem.hm_ingestion import ingest_session  # noqa: E402
from halumem.hm_search import process_user as search_user  # noqa: E402
from utils.checkpoint import atomic_json_dump  # noqa: E402
from utils.duration_stats import update_unit_duration_list  # noqa: E402
from utils.env import load_env  # noqa: E402
from utils.ingest_helpers import AddCallTimer  # noqa: E402
from utils.response_options import parse_bool  # noqa: E402
from utils.streaming import (  # noqa: E402
    LongCallLogger,
    configure_single_user_streaming,
    delete_user_data,
    load_marker_set as load_added_sessions,
    load_marker_set as load_completed,
    log_event,
    mark_marker as mark_added_session,
    mark_marker as mark_completed,
    prepare_user_after_delete,
)
from utils.time import parse_halumem_time  # noqa: E402


def _results_dir(frame: str, version: str) -> Path:
    return Path("results") / "halumem" / f"{frame}-{version}"


def _tmp_path(frame: str, version: str, user_uuid: str) -> Path:
    return _results_dir(frame, version) / "tmp" / f"{frame}_hm_search_results_{user_uuid}.json"


def _combined_path(frame: str, version: str) -> Path:
    return _results_dir(frame, version) / f"{frame}_hm_search_results.json"


def _completed_path(frame: str, version: str) -> Path:
    return _results_dir(frame, version) / "streaming_completed.txt"


def _added_sessions_path(frame: str, version: str, user_uuid: str) -> Path:
    return _results_dir(frame, version) / "tmp" / f"{frame}_hm_added_sessions_{user_uuid}.txt"


def _events_path(frame: str, version: str) -> Path:
    return _results_dir(frame, version) / "streaming_events.jsonl"


def _stats_path(frame: str, version: str) -> Path:
    return _results_dir(frame, version) / f"{frame}_hm_ingestion_stats.json"


def session_checkpoint_id(user_uuid: str, session_idx: int) -> str:
    return f"{user_uuid}_{session_idx}"


def write_combined_results(frame: str, version: str, completed: set[str]) -> None:
    tmp_dir = _results_dir(frame, version) / "tmp"
    combined: dict[str, list] = {}
    pattern = str(tmp_dir / f"{frame}_hm_search_results_*.json")

    def _user_uuid(path: str) -> str:
        match = re.search(r"_hm_search_results_(.+)\.json$", path)
        return match.group(1) if match else ""

    for path in sorted(glob.glob(pattern), key=_user_uuid):
        if _user_uuid(path) not in completed:
            continue
        with open(path) as f:
            data = json.load(f)
        for user_id, entries in data.items():
            combined[user_id] = entries

    out = _combined_path(frame, version)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(combined, out, indent=4)


def _completed_search_records(frame: str, version: str, completed: set[str]) -> list[dict]:
    records: list[dict] = []
    tmp_dir = _results_dir(frame, version) / "tmp"
    pattern = str(tmp_dir / f"{frame}_hm_search_results_*.json")

    def _user_uuid(path: str) -> str:
        match = re.search(r"_hm_search_results_(.+)\.json$", path)
        return match.group(1) if match else ""

    for path in sorted(glob.glob(pattern), key=_user_uuid):
        if _user_uuid(path) not in completed:
            continue
        with open(path) as f:
            data = json.load(f)
        for entries in data.values():
            if isinstance(entries, list):
                records.extend(entry for entry in entries if isinstance(entry, dict))
    return records


def write_search_status(
    frame: str,
    version: str,
    completed: set[str],
    *,
    allow_empty_search: bool,
    skip_failed_search: bool,
    skip_failed_streaming: bool,
    failed_users: list[dict],
    skipped_records: list[dict],
) -> None:
    records = _completed_search_records(frame, version, completed)
    atomic_json_dump(
        {
            "stage": "search",
            "mode": "halumem_streaming",
            "allow_empty_search": allow_empty_search,
            "skip_failed_search": skip_failed_search,
            "skip_failed_streaming": skip_failed_streaming,
            "status_counts": status_counts(records),
            "failed_users": failed_users,
            "skipped_records": skipped_records,
        },
        _results_dir(frame, version) / f"{frame}_hm_search_status.json",
        indent=2,
    )


def load_stats(frame: str, version: str) -> dict:
    path = _stats_path(frame, version)
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return {
        "mode": "halumem_streaming",
        "user_durations_ms": {},
        "add_call_durations_by_unit": {},
        "add_call_durations_ms": [],
        "search_call_durations_ms": [],
        "final_delete_statuses": {},
        "session_counts": {},
    }


def save_stats(frame: str, version: str, stats: dict) -> None:
    path = _stats_path(frame, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(stats, path, indent=2)


def add_user_sessions(
    user_obj: dict,
    *,
    frame: str,
    version: str,
    client,
    added_sessions_path: Path,
    added_sessions: set[str],
) -> list[float]:
    user_uuid = str(user_obj["uuid"])
    user_id = user_id_for(version, user_uuid)
    timer = AddCallTimer(client)
    skipped_sessions = 0
    sessions = user_obj.get("sessions", [])

    for session_idx, session in enumerate(sessions):
        checkpoint_id = session_checkpoint_id(user_uuid, session_idx)
        if checkpoint_id in added_sessions:
            skipped_sessions += 1
            print(
                f"[{frame}] HaluMem user {user_uuid}: session {session_idx + 1}/"
                f"{len(sessions)} already added, skipping",
                flush=True,
            )
            continue

        session_id = f"{user_id}_hm_session_{session_idx}"
        end_time_str = session.get("end_time", session.get("start_time", ""))
        date_obj = parse_halumem_time(end_time_str)
        dialogue = session.get("dialogue", [])
        label = (
            f"{frame} HaluMem add user={user_uuid} "
            f"session={session_idx + 1}/{len(sessions)} messages={len(dialogue)}"
        )
        with LongCallLogger(label):
            ingest_session(dialogue, date_obj, user_id, session_id, frame, client)
        mark_added_session(added_sessions_path, added_sessions, checkpoint_id)

    if skipped_sessions:
        print(
            f"[{frame}] HaluMem user {user_uuid}: "
            f"{skipped_sessions}/{len(sessions)} sessions resumed from checkpoint",
            flush=True,
        )
    return timer.durations_ms


def search_user_questions(
    user_obj: dict,
    *,
    frame: str,
    version: str,
    top_k: int,
    allow_empty_search: bool,
    skip_failed_search: bool,
) -> tuple[list[dict], list[dict]]:
    search_results, blocking_records = search_user(
        user_obj,
        frame,
        version,
        top_k,
        allow_empty_search=allow_empty_search,
        skip_failed_search=skip_failed_search,
    )
    records: list[dict] = []
    for entries in search_results.values():
        if isinstance(entries, list):
            records.extend(entry for entry in entries if isinstance(entry, dict))
    if blocking_records:
        raise RuntimeError(
            f"HaluMem user {user_obj['uuid']}: "
            f"{len(blocking_records)} search records have disallowed status"
        )
    return records, blocking_records


def process_user(
    user_obj: dict,
    *,
    frame: str,
    version: str,
    top_k: int,
    allow_empty_search: bool,
    skip_failed_search: bool,
    wait_after_ingest: float,
    completed: set[str],
    stats: dict,
    restart_unit: bool,
) -> None:
    user_uuid = str(user_obj["uuid"])
    user_id = user_id_for(version, user_uuid)
    started = time.time()
    print("\n" + "=" * 80)
    print(f"HALUMEM STREAM user {user_uuid}: add, search, delete")
    print("=" * 80)

    client = create_client(frame)
    added_path = _added_sessions_path(frame, version, user_uuid)
    added_sessions = load_added_sessions(added_path)
    should_start_fresh = restart_unit or not added_sessions

    if should_start_fresh:
        delete_user_data(frame, client, user_id)
        prepare_user_after_delete(frame, client, user_id)
        if added_path.exists():
            added_path.unlink()
        added_sessions = set()
        tmp_path = _tmp_path(frame, version, user_uuid)
        if tmp_path.exists():
            tmp_path.unlink()
    else:
        print(f"Resuming HaluMem user {user_uuid}: {len(added_sessions)} sessions recorded")

    add_durations = add_user_sessions(
        user_obj,
        frame=frame,
        version=version,
        client=client,
        added_sessions_path=added_path,
        added_sessions=added_sessions,
    )

    if wait_after_ingest > 0:
        print(f"Waiting {wait_after_ingest}s after ingest")
        time.sleep(wait_after_ingest)

    search_records, _ = search_user_questions(
        user_obj,
        frame=frame,
        version=version,
        top_k=top_k,
        allow_empty_search=allow_empty_search,
        skip_failed_search=skip_failed_search,
    )
    search_durations = [
        float(record.get("search_duration_ms") or 0.0)
        for record in search_records
    ]

    final_delete_ok, final_delete_error = delete_user_data(frame, client, user_id)
    final_delete_status = "ok" if final_delete_ok else "error_skipped"
    if final_delete_ok and added_path.exists():
        added_path.unlink()
    if not final_delete_ok:
        log_event(
            _events_path(frame, version),
            "final_delete_error_skipped",
            user_uuid=user_uuid,
            user_id=user_id,
            error=final_delete_error,
        )

    mark_completed(_completed_path(frame, version), completed, user_uuid)
    elapsed_ms = round((time.time() - started) * 1000, 1)

    stats.setdefault("user_durations_ms", {})[user_uuid] = elapsed_ms
    update_unit_duration_list(
        stats,
        user_uuid,
        add_durations,
        map_key="add_call_durations_by_unit",
        flat_key="add_call_durations_ms",
    )
    stats.setdefault("search_call_durations_ms", []).extend(
        round(v, 2) for v in search_durations
    )
    stats.setdefault("final_delete_statuses", {})[user_uuid] = final_delete_status
    stats.setdefault("session_counts", {})[user_uuid] = len(user_obj.get("sessions", []))

    log_event(
        _events_path(frame, version),
        "completed",
        user_uuid=user_uuid,
        user_id=user_id,
        sessions=len(user_obj.get("sessions", [])),
        questions=len(search_records),
        final_delete_status=final_delete_status,
    )
    write_combined_results(frame, version, completed)
    save_stats(frame, version, stats)


def mark_streaming_failure_skipped(
    user_obj: dict,
    frame: str,
    version: str,
    completed: set[str],
    exc: BaseException,
) -> dict:
    user_uuid = str(user_obj["uuid"])
    user_id = user_id_for(version, user_uuid)
    entries = [
        build_search_entry(
            meta,
            context="",
            duration_ms=0.0,
            status=STATUS_SKIPPED,
            error=error_payload("streaming", exc),
        )
        for meta in iter_questions(user_obj, version)
    ]
    tmp_path = _tmp_path(frame, version, user_uuid)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump({user_id: entries}, tmp_path, indent=4)
    mark_completed(_completed_path(frame, version), completed, user_uuid)
    return {
        "user_uuid": user_uuid,
        "user_id": user_id,
        "error": error_payload("streaming", exc),
    }


def main() -> int:
    parser = argparse.ArgumentParser("HaluMem streaming add-search-delete")
    parser.add_argument("--lib", choices=SUPPORTED_LIBS, default=DEFAULT_LIB)
    parser.add_argument("--env", help="Dotenv file to load")
    parser.add_argument("--version", default="default")
    parser.add_argument("--variant", choices=["medium", "long"], default="medium")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--allow-empty-search",
        "--allow_empty_search",
        type=parse_bool,
        default=True,
        help="Allow successful searches with no raw memories. Default: 1.",
    )
    parser.add_argument(
        "--skip-failed-search",
        "--skip_failed_search",
        type=parse_bool,
        default=False,
        help="Mark failed search calls as skipped instead of failing the unit. Default: 0.",
    )
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int)
    parser.add_argument("--wait-after-ingest", type=float, default=0.0)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore streaming_completed.txt and process selected users anyway.",
    )
    parser.add_argument(
        "--restart-unit",
        action="store_true",
        help="Delete each selected user and discard session/search checkpoints.",
    )
    parser.add_argument(
        "--skip-failed-streaming",
        action="store_true",
        help="Mark failed streaming users as skipped and continue.",
    )
    args = parser.parse_args()

    if args.env:
        env_path = Path(args.env)
        if not env_path.is_file():
            env_path = Path.cwd() / args.env
        if not env_path.is_file():
            raise SystemExit(f"Env file not found: {args.env}")
        os.environ["OMNIMEMEVAL_ENV_FILE"] = str(env_path.resolve())

    load_env()
    configure_single_user_streaming(args.lib)

    results_dir = _results_dir(args.lib, args.version)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "tmp").mkdir(parents=True, exist_ok=True)

    users = load_halumem_data(args.variant)
    total = len(users)
    end_idx = args.end_idx if args.end_idx is not None else total - 1
    if args.start_idx < 0 or end_idx >= total or end_idx < args.start_idx:
        raise SystemExit(
            f"Invalid range start={args.start_idx}, end={end_idx}, total={total}"
        )

    completed_path = _completed_path(args.lib, args.version)
    existing_completed = load_completed(completed_path)
    completed = set() if args.no_resume else set(existing_completed)
    stats = load_stats(args.lib, args.version)

    print("\n" + "=" * 80)
    print("HALUMEM STREAMING")
    print("=" * 80)
    print(f"lib={args.lib}")
    print(f"version={args.version}")
    print(f"variant={args.variant}")
    print(f"range={args.start_idx}-{end_idx}")
    print(f"top_k={args.top_k}")
    print(f"allow_empty_search={args.allow_empty_search}")
    print(f"skip_failed_search={args.skip_failed_search}")
    print(f"wait_after_ingest={args.wait_after_ingest}")
    print(f"already_completed={len(completed)}")
    print("=" * 80)

    failed_users: list[dict] = []
    skipped_records: list[dict] = []
    for row_pos in range(args.start_idx, end_idx + 1):
        user_obj = users[row_pos]
        user_uuid = str(user_obj["uuid"])
        if user_uuid in completed and not args.no_resume:
            print(f"Skipping HaluMem user {user_uuid}: already completed")
            continue
        try:
            process_user(
                user_obj,
                frame=args.lib,
                version=args.version,
                top_k=args.top_k,
                allow_empty_search=args.allow_empty_search,
                skip_failed_search=args.skip_failed_search,
                wait_after_ingest=args.wait_after_ingest,
                completed=completed,
                stats=stats,
                restart_unit=(
                    args.restart_unit
                    or (args.no_resume and user_uuid in existing_completed)
                ),
            )
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            raise
        except Exception as exc:
            log_event(
                _events_path(args.lib, args.version),
                "failed",
                user_uuid=user_uuid,
                error=str(exc),
            )
            print(f"ERROR HaluMem user {user_uuid}: {type(exc).__name__}: {exc}")
            failure = {
                "user_uuid": user_uuid,
                "user_id": user_id_for(args.version, user_uuid),
                "error": error_payload("streaming", exc),
            }
            if args.skip_failed_streaming:
                skipped_records.append(
                    mark_streaming_failure_skipped(
                        user_obj,
                        args.lib,
                        args.version,
                        completed,
                        exc,
                    )
                )
                write_combined_results(args.lib, args.version, completed)
                save_stats(args.lib, args.version, stats)
                continue
            failed_users.append(failure)
            continue

    write_combined_results(args.lib, args.version, completed)
    save_stats(args.lib, args.version, stats)
    write_search_status(
        args.lib,
        args.version,
        completed,
        allow_empty_search=args.allow_empty_search,
        skip_failed_search=args.skip_failed_search,
        skip_failed_streaming=args.skip_failed_streaming,
        failed_users=failed_users,
        skipped_records=skipped_records,
    )
    if failed_users:
        print(f"\nHaluMem streaming failed for {len(failed_users)} user(s)")
        return 1
    print("\nHaluMem streaming complete")
    print(f"Combined search results: {_combined_path(args.lib, args.version)}")
    print(f"Completed users: {len(completed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
