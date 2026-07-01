import argparse
import ast
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)

sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, SCRIPT_DIR)

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from beam_common import (
    STATUS_FAILED,
    STATUS_SKIPPED,
    build_question_meta,
    build_search_entry,
    classify_search_status,
    error_payload,
    get_search_entries,
    record_status,
    search_allowed_statuses,
    status_counts,
    user_id_for,
)
from utils.checkpoint import atomic_json_dump
from utils.env import load_env
from utils.progress import track
from utils.response_options import parse_bool
from utils.search_helpers import dispatch_search, unpack_search_result
from client_factory import SUPPORTED_LIBS, DEFAULT_LIB

SCALE_FILE_MAP = {
    "100k": "beam_100k.json",
    "500k": "beam_500k.json",
    "1m": "beam_1m.json",
    "10m": "beam_10m_10m.json",
}


def load_beam_data(scale="all"):
    data_dir = os.path.join(PROJECT_DIR, "data", "beam")
    all_conversations = []

    if scale == "all":
        scales = list(SCALE_FILE_MAP.keys())
    else:
        scales = [scale]

    for s in scales:
        filepath = os.path.join(data_dir, SCALE_FILE_MAP[s])
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                conv = json.loads(line)
                conv["_scale"] = s
                all_conversations.append(conv)

    return all_conversations


def parse_probing_questions(conv):
    pq = conv.get("probing_questions", {})
    if isinstance(pq, str):
        pq = ast.literal_eval(pq)
    return pq


def load_existing_results(frame, version, conv_id):
    result_path = (
        f"results/beam/{frame}-{version}/tmp/{frame}_beam_search_results_{conv_id}.json"
    )
    if os.path.exists(result_path):
        try:
            with open(result_path) as f:
                return json.load(f), True
        except Exception as e:
            print(f"❌ Error loading existing results for conv {conv_id}: {e}")
    return {}, False


def _valid_existing_entries(existing_results, expected, user_id, allowed_statuses):
    expected_by_key = {str(meta["key"]): meta for meta in expected}
    valid = {}
    for entry in get_search_entries(existing_results, user_id):
        key = str(entry.get("key") or "")
        meta = expected_by_key.get(key)
        if meta is None:
            continue
        if entry.get("question") != meta["question"].get("question", ""):
            continue
        if entry.get("question_idx") != meta["question_idx"]:
            continue
        if "status" not in entry:
            continue
        if record_status(entry) not in allowed_statuses:
            continue
        valid[key] = entry
    return valid


def _ordered_entries(expected, entries_by_key):
    entries = []
    for meta in expected:
        entry = entries_by_key.get(str(meta["key"]))
        if entry is not None:
            entries.append(entry)
    return entries


def process_conversation(
    conv,
    frame,
    version,
    top_k=20,
    *,
    allow_empty_search=True,
    skip_failed_search=False,
):
    conv_id = str(conv["conversation_id"])
    user_id = user_id_for(version, conv_id)
    question_meta = build_question_meta(
        conv,
        version=version,
        parse_probing_questions=parse_probing_questions,
    )

    if not question_meta:
        return {}, []

    allowed_statuses = search_allowed_statuses(
        allow_empty_search=allow_empty_search,
        allow_skipped=skip_failed_search,
    )

    existing_results, exists = load_existing_results(frame, version, conv_id)
    entries_by_key = (
        _valid_existing_entries(
            existing_results,
            question_meta,
            user_id,
            allowed_statuses,
        )
        if exists
        else {}
    )
    if entries_by_key and len(entries_by_key) >= len(question_meta):
        print(f"♻️  Using existing results for conversation {conv_id}")
        return {user_id: _ordered_entries(question_meta, entries_by_key)}, []
    if entries_by_key:
        print(
            f"♻️  Resuming conversation {conv_id}: "
            f"{len(entries_by_key)}/{len(question_meta)} questions done"
        )

    from client_factory import create_client

    client = create_client(frame)

    tmp_dir = f"results/beam/{frame}-{version}/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = f"{tmp_dir}/{frame}_beam_search_results_{conv_id}.json"

    for meta in question_meta:
        key = str(meta["key"])
        if key in entries_by_key:
            continue

        question_text = meta["question"].get("question", "")

        print(f"  🔎 [{conv_id}] Q: {question_text[:80]}...")

        try:
            result = dispatch_search(frame, client, question_text, user_id, top_k)
            context, duration_ms, reflect_answer, raw_context = unpack_search_result(result)
        except Exception as e:
            print(f"  ❌ Search failed for conv {conv_id}: {e}")
            status = STATUS_SKIPPED if skip_failed_search else STATUS_FAILED
            entry = build_search_entry(
                meta,
                context="",
                duration_ms=0.0,
                status=status,
                error=error_payload("search", e),
            )
        else:
            context = context or ""
            status = classify_search_status(
                context,
                reflect_answer,
                raw_context=raw_context,
            )
            entry = build_search_entry(
                meta,
                context=context,
                duration_ms=duration_ms,
                status=status,
                reflect_answer=reflect_answer,
            )

        entries_by_key[key] = entry
        atomic_json_dump(
            {user_id: _ordered_entries(question_meta, entries_by_key)},
            tmp_path,
            indent=4,
        )

    print(f"💾 Search results for conversation {conv_id} saved")

    ordered = _ordered_entries(question_meta, entries_by_key)
    blocking_records = [
        entry for entry in ordered if record_status(entry) not in allowed_statuses
    ]
    return {user_id: ordered}, blocking_records


def main(
    frame,
    version,
    top_k=20,
    num_workers=2,
    scale="all",
    *,
    allow_empty_search=True,
    skip_failed_search=False,
):
    load_env()

    if frame == "everos":
        os.environ["EVEROS_USE_GROUP"] = "false"

    print(f"\n{'=' * 80}")
    print(f"🔍 BEAM SEARCH - {frame.upper()} v{version} (scale={scale})".center(80))
    print(f"{'=' * 80}")

    conversations = load_beam_data(scale)
    num_conversations = len(conversations)
    print(f"📚 Loaded {num_conversations} BEAM conversations")
    print(f"⚙️  Search parameters: top_k={top_k}, workers={num_workers}")
    print(
        "⚙️  Failure controls: "
        f"allow_empty_search={allow_empty_search}, "
        f"skip_failed_search={skip_failed_search}"
    )
    print(f"{'-' * 80}")

    os.makedirs(f"results/beam/{frame}-{version}/", exist_ok=True)
    all_search_results = defaultdict(list)
    start_time = datetime.now()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_conv = {
            executor.submit(
                process_conversation,
                conv,
                frame,
                version,
                top_k,
                allow_empty_search=allow_empty_search,
                skip_failed_search=skip_failed_search,
            ): conv
            for conv in conversations
        }

        failed_conversations = []
        all_status_records = []
        for future in track(
            as_completed(future_to_conv),
            total=num_conversations,
            description="Searching conversations",
        ):
            try:
                search_results, blocking_records = future.result()
                for uid, results in search_results.items():
                    all_search_results[uid].extend(results)
                    all_status_records.extend(results)
                if blocking_records:
                    conv = future_to_conv[future]
                    cid = str(conv.get("conversation_id", "?"))
                    failed_conversations.append({
                        "conv_id": cid,
                        "user_id": user_id_for(version, cid),
                        "failures": blocking_records,
                    })
            except Exception as e:
                conv = future_to_conv[future]
                cid = str(conv.get("conversation_id", "?"))
                print(f"❌ Error processing conversation {cid}: {e}")
                failed_conversations.append({
                    "conv_id": cid,
                    "user_id": user_id_for(version, cid),
                    "error": error_payload("search", e),
                })

    end_time = datetime.now()
    elapsed = str(end_time - start_time).split(".")[0]

    output_path = f"results/beam/{frame}-{version}/{frame}_beam_search_results.json"
    atomic_json_dump(dict(all_search_results), output_path, indent=4)
    atomic_json_dump(
        {
            "stage": "search",
            "allow_empty_search": allow_empty_search,
            "skip_failed_search": skip_failed_search,
            "status_counts": status_counts(all_status_records),
            "failed_users": failed_conversations,
        },
        f"results/beam/{frame}-{version}/{frame}_beam_search_status.json",
        indent=2,
    )

    if failed_conversations:
        print(f"\n{'=' * 80}")
        print(f"❌ SEARCH FAILED: {len(failed_conversations)}/{num_conversations} conversations had errors".center(80))
        print(f"{'=' * 80}")
        print(f"⏱️  Total time: {elapsed}")
        print(f"🔄 Framework: {frame} | Version: {version} | Workers: {num_workers}")
        print(f"{'=' * 80}\n")
        raise SystemExit(1)

    print(f"\n{'=' * 80}")
    print("✅ SEARCH COMPLETE".center(80))
    print(f"{'=' * 80}")
    print(f"⏱️  Total time: {elapsed}")
    print(f"🔄 Framework: {frame} | Version: {version} | Workers: {num_workers}")
    print(f"📁 Results saved to: {output_path}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BEAM Search Script")
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
        "--top-k", type=int, default=20, help="Number of top results to retrieve."
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="Number of parallel workers."
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="all",
        choices=["100k", "500k", "1m", "10m", "all"],
        help="BEAM evaluation scale.",
    )
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
        help="Mark failed search calls as skipped instead of failing the step. Default: 0.",
    )

    args = parser.parse_args()
    main(
        frame=args.lib,
        version=args.version,
        top_k=args.top_k,
        num_workers=args.workers,
        scale=args.scale,
        allow_empty_search=args.allow_empty_search,
        skip_failed_search=args.skip_failed_search,
    )
