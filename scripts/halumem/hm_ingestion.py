import argparse
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from utils.env import load_env
from utils.progress import track
from utils.time import parse_halumem_time, to_iso
from utils.checkpoint import atomic_json_dump, fsync_write_line
from utils.ingest_helpers import inject_time, session_id_kwargs, AddCallTimer
from client_factory import SUPPORTED_LIBS, DEFAULT_LIB
from halumem.hm_common import load_halumem_data

_HM_RETAIN_MISSION = (
    "Extract ALL factual claims from this conversation, paying special attention to: "
    "1) TEMPORAL facts — dates, times, durations, sequences of events; "
    "2) Personal details — names, preferences, habits, relationships; "
    "3) Plans and intentions — future plans, goals, commitments with timeframes. "
    "Preserve exact dates and times whenever mentioned."
)


def ingest_session(session, date, user_id, session_id, frame, client):
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in session]

    time_kw = inject_time(messages, date, frame)
    sess_kw = session_id_kwargs(frame, session_id)

    if frame == "letta":
        client.add(messages, user_id, **sess_kw, **time_kw)
    elif frame == "hindsight":
        date_display = date.strftime("%Y-%m-%d %H:%M:%S")
        context_str = (
            f"Session {session_id} - you are the assistant in this "
            f"conversation - happened on {date_display} UTC."
        )
        client.add(
            [], user_id,
            raw_content=json.dumps(session),
            context=context_str,
            retain_mission=_HM_RETAIN_MISSION,
            **sess_kw, **time_kw,
        )
    elif frame == "supermemory":
        client.add(messages, user_id, session_id=session_id, **time_kw)
    else:
        client.add(messages, user_id, **sess_kw, **time_kw)

    print(f"[{frame}] Session {session_id}: Ingested {len(messages)} messages at {to_iso(date)}")


def ingest_user(user_obj, version, frame, success_records, f, record_lock, clear=False):
    import time as _time
    user_start = _time.time()

    user_uuid = user_obj["uuid"]
    sessions = user_obj["sessions"]
    user_id = f"hm_exp_user_{version}_{user_uuid}"

    print("\n" + "=" * 80)
    print(f"🔄 [INGESTING USER {user_uuid}]".center(80))
    print("=" * 80)

    from client_factory import create_client

    client = create_client(frame)
    timer = AddCallTimer(client)
    try:
        if frame == "zep":
            client.delete_user(user_id)
            client.add_user(user_id)
        elif "mem0" in frame:
            client.delete_all(user_id=user_id)
        elif clear and hasattr(client, "delete"):
            client.delete(user_id)
        elif clear and hasattr(client, "delete_user"):
            client.delete_user(user_id)
    except Exception as exc:
        print(f"  ⚠ Cleanup failed for {user_id}, continuing ingestion: {exc}")

    failed_sessions = []
    for idx, session in enumerate(sessions):
        record_key = f"{user_uuid}_{idx}"
        if record_key not in success_records:
            session_id = user_id + "_hm_session_" + str(idx)
            end_time_str = session.get("end_time", session.get("start_time", ""))
            date_obj = parse_halumem_time(end_time_str)

            dialogue = session.get("dialogue", [])
            try:
                ingest_session(dialogue, date_obj, user_id, session_id, frame, client)
                with record_lock:
                    fsync_write_line(f, record_key)
                    success_records.add(record_key)
            except Exception as e:
                import traceback
                print(f"❌ Error ingesting session {record_key}: {type(e).__name__}: {e}")
                traceback.print_exc()
                failed_sessions.append(record_key)
        else:
            print(f"✅ Session {record_key} already ingested")

    if failed_sessions:
        raise RuntimeError(
            f"User {user_uuid}: {len(failed_sessions)}/{len(sessions)} "
            f"sessions failed: {failed_sessions}"
        )

    print("=" * 80)
    return round((_time.time() - user_start) * 1000, 1), timer.durations_ms


def main(frame, version, variant="medium", num_workers=2, clear=False):
    load_env()

    print("\n" + "=" * 80)
    print(f"🚀 HALUMEM INGESTION - {frame.upper()} V-{version} ({variant})".center(80))
    print("=" * 80)
    if clear:
        print("🧹 --clear enabled: will delete existing memories before ingestion")

    users = load_halumem_data(variant)

    print(f"📚 Loaded HaluMem-{variant.capitalize()} dataset ({len(users)} users)")
    print("-" * 80)

    start_time = datetime.now()
    os.makedirs(f"results/halumem/{frame}-{version}/", exist_ok=True)
    success_records = set()
    record_file = f"results/halumem/{frame}-{version}/success_records.txt"
    if clear and os.path.exists(record_file):
        os.remove(record_file)
        print("🧹 Cleared progress records")
    if os.path.exists(record_file):
        with open(record_file) as f:
            for i in f.readlines():
                success_records.add(i.strip())

    user_durations = {}
    all_add_call_durations = []
    failed_users = []
    record_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=num_workers) as executor, open(record_file, "a+") as f:
        future_to_uuid = {}
        for user_obj in users:
            future = executor.submit(
                ingest_user,
                user_obj,
                version,
                frame,
                success_records,
                f,
                record_lock,
                clear,
            )
            future_to_uuid[future] = user_obj["uuid"]

        for future in track(
            as_completed(future_to_uuid), total=len(future_to_uuid), description="Ingesting users",
        ):
            uuid = future_to_uuid[future]
            try:
                dur_ms, add_call_ms = future.result()
                user_durations[str(uuid)] = dur_ms
                all_add_call_durations.extend(add_call_ms)
            except Exception as e:
                import traceback
                print(f"❌ Error processing user {uuid}: {type(e).__name__}: {e}")
                traceback.print_exc()
                failed_users.append(uuid)

    stats_path = os.path.join(
        f"results/halumem/{frame}-{version}",
        f"{frame}_hm_ingestion_stats.json",
    )
    atomic_json_dump(
        {
            "user_durations_ms": user_durations,
            "add_call_durations_ms": [round(d, 2) for d in all_add_call_durations],
        },
        stats_path,
        indent=2,
    )
    print(f"Ingestion stats saved to {stats_path}")

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    elapsed_time_str = str(elapsed_time).split(".")[0]

    if failed_users:
        print("\n" + "=" * 80)
        print(f"❌ INGESTION FAILED: {len(failed_users)}/{len(users)} users had errors".center(80))
        print("=" * 80)
        print(f"⏱️  Total time: {elapsed_time_str}")
        print(
            "💡 Fix errors and re-run — successfully ingested sessions are "
            "saved in success_records.txt"
        )
        print("=" * 80 + "\n")
        raise SystemExit(1)

    print("\n" + "=" * 80)
    print("✅ INGESTION COMPLETE".center(80))
    print("=" * 80)
    print(f"⏱️  Total time taken to ingest {len(users)} users: {elapsed_time_str}")
    print(f"🔄 Framework: {frame} | Version: {version} | Workers: {num_workers}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HaluMem Ingestion Script")
    parser.add_argument(
        "--lib",
        type=str,
        choices=SUPPORTED_LIBS,
        default=DEFAULT_LIB,
    )
    parser.add_argument(
        "--version", type=str, default="default", help="Version of the evaluation framework."
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="Number of parallel ingestion workers."
    )
    parser.add_argument(
        "--clear", action="store_true", help="Clear existing memories before ingestion"
    )
    parser.add_argument(
        "--variant", type=str, default="medium", choices=["medium", "long"],
        help="HaluMem dataset variant (medium or long)"
    )

    args = parser.parse_args()
    main(
        frame=args.lib,
        version=args.version,
        variant=args.variant,
        num_workers=args.workers,
        clear=args.clear,
    )
