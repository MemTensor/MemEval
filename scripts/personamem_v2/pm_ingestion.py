import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from utils.env import load_env
from utils.progress import track
from utils.checkpoint import atomic_json_dump, fsync_write_line
from utils.ingest_helpers import inject_time, session_id_kwargs, AddCallTimer
from utils.streaming import LongCallLogger
from client_factory import SUPPORTED_LIBS, DEFAULT_LIB
from client_factory import iter_batches

BENCHMARK_CSV = "data/personamem_v2/benchmark/text/benchmark.csv"
CHAT_HISTORY_DIR = "data/personamem_v2/data/chat_history_32k"

_PM_RETAIN_MISSION = (
    "Extract ALL factual claims from this conversation, paying special attention to: "
    "1) Personal preferences — likes, dislikes, favorites, habits, routines; "
    "2) Personal details — names, relationships, occupations, locations, hobbies; "
    "3) Opinions and values — views on topics, beliefs, priorities; "
    "4) Plans and intentions — future plans, goals, commitments; "
    "5) Past experiences — things that happened, places visited, activities done. "
    "Extract negative statements ('I have never...', 'I don't like...') as separate facts."
)


def build_chat_history_index(chat_history_dir):
    index = {}
    for filepath in glob.glob(os.path.join(chat_history_dir, "*.json")):
        basename = os.path.basename(filepath)
        parts = basename.split("_persona")
        if len(parts) == 2:
            pid = int(parts[1].replace(".json", ""))
            index[pid] = filepath
    return index


def load_benchmark_personas(csv_path):
    personas = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["persona_id"])
            if pid not in personas:
                personas[pid] = {
                    "persona_id": pid,
                    "chat_history_32k_link": row["chat_history_32k_link"],
                    "question_count": 0,
                }
            personas[pid]["question_count"] += 1
    return personas


def ingest_session(session, user_id, session_id, frame, client):
    messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in session
        if msg["role"] != "system" and msg.get("content", "").strip()
    ]
    if not messages:
        return 0
    char_count = sum(len(str(msg.get("content", ""))) for msg in messages)

    # PersonaMem v2 has no time data — pass dt=None
    time_kw = inject_time(messages, None, frame)
    sess_kw = session_id_kwargs(frame, session_id)

    label = (
        f"{frame} PersonaMem add user={user_id} session={session_id} "
        f"messages={len(messages)} chars={char_count}"
    )
    with LongCallLogger(label):
        if frame == "letta":
            client.add(messages, user_id, **sess_kw, **time_kw)
        elif frame == "hindsight":
            context_str = f"Chat session {session_id} with persona {user_id}."
            client.add(
                [], user_id,
                raw_content=json.dumps(session),
                context=context_str,
                retain_mission=_PM_RETAIN_MISSION,
                **sess_kw, **time_kw,
            )
        elif frame == "supermemory":
            client.add(messages, user_id, session_id=session_id, **time_kw)
        else:
            client.add(messages, user_id, **sess_kw, **time_kw)

    return len(messages)


def ingest_persona(persona_id, chat_history_path, version, frame, success_records, f, clear=False):
    import time as _time

    if str(persona_id) in success_records:
        print(f"  Persona {persona_id} already ingested, skipping...")
        return persona_id, 0, []

    persona_start = _time.time()
    user_id = f"pm_exper_user_{persona_id}_{version}"

    print(f"\n{'=' * 80}")
    print(f"  INGESTING PERSONA {persona_id}".center(80))
    print(f"{'=' * 80}")

    with open(chat_history_path, encoding="utf-8") as fh:
        data = json.load(fh)

    chat_history = data["chat_history"]
    print(f"  User ID: {user_id}")
    print(f"  Total messages: {len(chat_history)}")

    from client_factory import create_client

    client = create_client(frame)
    timer = AddCallTimer(client)
    try:
        if frame == "zep":
            client.delete_user(user_id)
            client.add_user(user_id=user_id)
        elif "mem0" in frame:
            client.delete_all(user_id=user_id)
        elif frame == "supermemory" and clear:
            client.delete(user_id)
        elif clear and hasattr(client, "delete"):
            client.delete(user_id)
        elif clear and hasattr(client, "delete_user"):
            client.delete_user(user_id)
    except Exception as exc:
        print(f"  ⚠ Cleanup failed for {user_id}, continuing ingestion: {exc}")

    try:
        total_ingested = 0
        for chunk_idx, chunk in enumerate(iter_batches(chat_history)):
            session_id = f"pm_persona_{persona_id}_chunk_{chunk_idx}"
            count = ingest_session(chunk, user_id, session_id, frame, client)
            total_ingested += count

        print(f"  Ingested {total_ingested} messages for persona {persona_id}")
        print(f"{'=' * 80}")

        fsync_write_line(f, str(persona_id))
        dur_ms = round((_time.time() - persona_start) * 1000, 1)
        return persona_id, dur_ms, timer.durations_ms
    except Exception as e:
        print(f"  Error ingesting persona {persona_id}: {e}")
        raise


def main(frame, version, num_workers=2, clear=False, allow_missing_data=False):
    load_env()

    os.makedirs(f"results/pmv2/{frame}-{version}/", exist_ok=True)
    record_file = f"results/pmv2/{frame}-{version}/success_records.txt"

    if clear and os.path.exists(record_file):
        os.remove(record_file)
        print("  Cleared progress records")

    print("\n" + "=" * 80)
    print(f"  PERSONAMEM V2 INGESTION - {frame.upper()} v{version}".center(80))
    print("=" * 80)

    chat_history_index = build_chat_history_index(CHAT_HISTORY_DIR)
    print(f"  Found {len(chat_history_index)} chat history files in {CHAT_HISTORY_DIR}")

    benchmark_personas = load_benchmark_personas(BENCHMARK_CSV)
    print(f"  Found {len(benchmark_personas)} unique personas in benchmark")

    available_personas = {
        pid: info
        for pid, info in benchmark_personas.items()
        if pid in chat_history_index
    }
    missing_count = len(benchmark_personas) - len(available_personas)
    print(f"  Available (with chat history): {len(available_personas)}")
    if missing_count > 0:
        print(f"  Missing chat history files: {missing_count} personas (will be skipped)")
        if not allow_missing_data:
            print("  Use --allow-missing-data to explicitly evaluate the available subset.")
            raise SystemExit(1)
    print("-" * 80)

    success_records = set()
    if os.path.exists(record_file):
        with open(record_file) as f:
            success_records = {line.strip() for line in f}
        print(
            f"  Found {len(success_records)} completed personas, "
            f"{len(available_personas) - len(success_records)} remaining"
        )

    pending_personas = [
        (pid, chat_history_index[pid])
        for pid in sorted(available_personas.keys())
        if str(pid) not in success_records
    ]

    if not pending_personas:
        print("  All available personas have been processed!")
        return

    print(f"  Processing {len(pending_personas)} personas...")

    start_time = datetime.now()

    with ThreadPoolExecutor(max_workers=num_workers) as executor, open(record_file, "a") as f:
        futures = []
        for pid, chat_path in pending_personas:
            future = executor.submit(
                ingest_persona,
                persona_id=pid,
                chat_history_path=chat_path,
                version=version,
                frame=frame,
                success_records=success_records,
                f=f,
                clear=clear,
            )
            futures.append(future)

        completed_count = 0
        failed_personas = []
        user_durations = {}
        all_add_call_durations = []
        for future in track(
            as_completed(futures), total=len(futures), description="Ingesting personas"
        ):
            try:
                pid, dur_ms, add_call_ms = future.result()
                completed_count += 1
                if dur_ms > 0:
                    user_durations[str(pid)] = dur_ms
                all_add_call_durations.extend(add_call_ms)
            except Exception as exc:
                import traceback
                print(f"\n❌ Persona ingestion failed: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                failed_personas.append(str(exc))

    stats_path = os.path.join(f"results/pmv2/{frame}-{version}", f"{frame}_pm_ingestion_stats.json")
    atomic_json_dump(
        {
            "user_durations_ms": user_durations,
            "add_call_durations_ms": [round(d, 2) for d in all_add_call_durations],
            "failed_personas": failed_personas,
        },
        stats_path,
        indent=2,
    )
    print(f"Ingestion stats saved to {stats_path}")

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    elapsed_time_str = str(elapsed_time).split(".")[0]

    if failed_personas:
        print("\n" + "=" * 80)
        print(f"❌ INGESTION FAILED: {len(failed_personas)}/{len(pending_personas)} personas had errors".center(80))
        print("=" * 80)
        print(f"⏱️  Total time: {elapsed_time_str}")
        print("💡 Fix errors and re-run — successfully ingested personas are saved in success_records.txt")
        print("=" * 80 + "\n")
        raise SystemExit(1)

    print("\n" + "=" * 80)
    print("✅ INGESTION COMPLETE".center(80))
    print("=" * 80)
    print(f"⏱️  Total time: {elapsed_time_str}")
    print(f"🔄 Framework: {frame} | Version: {version} | Workers: {num_workers}")
    print(f"📊 Processed: {len(success_records) + completed_count}/{len(available_personas)} personas")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PersonaMem v2 Ingestion Script")
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
    parser.add_argument("--clear", action="store_true", help="Clear progress and start fresh")
    parser.add_argument(
        "--allow-missing-data",
        action="store_true",
        help="Allow missing PersonaMem chat history files and evaluate the available subset.",
    )
    args = parser.parse_args()

    main(
        frame=args.lib,
        version=args.version,
        num_workers=args.workers,
        clear=args.clear,
        allow_missing_data=args.allow_missing_data,
    )
