import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from client_factory import DEFAULT_LIB, SUPPORTED_LIBS
from halumem.hm_common import (
    STATUS_FAILED,
    STATUS_SKIPPED,
    build_search_entry,
    classify_search_status,
    error_payload,
    get_search_entries,
    iter_questions,
    load_halumem_data,
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


def load_existing_results(frame, version, user_uuid):
    result_path = (
        f"results/halumem/{frame}-{version}/tmp/"
        f"{frame}_hm_search_results_{user_uuid}.json"
    )
    if os.path.exists(result_path):
        try:
            with open(result_path) as f:
                return json.load(f), True
        except Exception as e:
            print(f"❌ Error loading existing results for user {user_uuid}: {e}")
    return {}, False


def _valid_existing_entries(
    existing_results,
    user_obj,
    version,
    allowed_statuses,
) -> dict[str, dict]:
    user_id = user_id_for(version, str(user_obj["uuid"]))
    expected = {str(meta["key"]): meta for meta in iter_questions(user_obj, version)}
    valid = {}
    for entry in get_search_entries(existing_results, user_id):
        key = str(entry.get("key") or "")
        meta = expected.get(key)
        if meta is None:
            continue
        if entry.get("question") != meta["question"].get("question"):
            continue
        if entry.get("session_idx") != meta["session_idx"]:
            continue
        if entry.get("question_idx") != meta["question_idx"]:
            continue
        if "status" not in entry:
            continue
        if record_status(entry) not in allowed_statuses:
            continue
        valid[key] = entry
    return valid


def _ordered_entries(user_obj, version, entries_by_key):
    entries = []
    for meta in iter_questions(user_obj, version):
        entry = entries_by_key.get(str(meta["key"]))
        if entry is not None:
            entries.append(entry)
    return entries


def process_user(
    user_obj,
    frame,
    version,
    top_k=20,
    *,
    allow_empty_search=True,
    skip_failed_search=False,
):
    user_uuid = str(user_obj["uuid"])
    user_id = user_id_for(version, user_uuid)
    question_meta = list(iter_questions(user_obj, version))
    total_expected = len(question_meta)
    allowed_statuses = search_allowed_statuses(
        allow_empty_search=allow_empty_search,
        allow_skipped=skip_failed_search,
    )

    existing_results, exists = load_existing_results(frame, version, user_uuid)
    entries_by_key = (
        _valid_existing_entries(
            existing_results,
            user_obj,
            version,
            allowed_statuses,
        )
        if exists
        else {}
    )
    if entries_by_key and len(entries_by_key) >= total_expected:
        print(f"♻️  Using existing results for user {user_uuid}")
        return {user_id: _ordered_entries(user_obj, version, entries_by_key)}, []
    if entries_by_key:
        print(
            f"♻️  Resuming user {user_uuid}: "
            f"{len(entries_by_key)}/{total_expected} questions done"
        )

    from client_factory import create_client

    client = create_client(frame)

    tmp_dir = f"results/halumem/{frame}-{version}/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = f"{tmp_dir}/{frame}_hm_search_results_{user_uuid}.json"

    for flat_idx, meta in enumerate(question_meta, start=1):
        key = str(meta["key"])
        if key in entries_by_key:
            continue

        question = meta["question"]
        question_text = question["question"]
        print(f"  🔎 [{user_uuid}] Q{flat_idx}: {question_text[:80]}")

        try:
            result = dispatch_search(
                frame,
                client,
                question_text,
                user_id,
                top_k,
            )
            context, duration_ms, reflect_answer, raw_context = unpack_search_result(result)
        except Exception as e:
            print(f"  ❌ Search failed for Q{flat_idx}: {e}")
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
            {user_id: _ordered_entries(user_obj, version, entries_by_key)},
            tmp_path,
            indent=4,
        )

    ordered = _ordered_entries(user_obj, version, entries_by_key)
    print(f"💾 Search results for user {user_uuid} saved ({len(ordered)} questions)")

    blocking_records = [
        entry for entry in ordered if record_status(entry) not in allowed_statuses
    ]
    return {user_id: ordered}, blocking_records


def main(
    frame,
    version,
    variant="medium",
    top_k=20,
    num_workers=2,
    *,
    allow_empty_search=True,
    skip_failed_search=False,
):
    load_env()

    print("\n" + "=" * 80)
    print(f"🔍 HALUMEM SEARCH - {frame.upper()} v{version} ({variant})".center(80))
    print("=" * 80)

    users = load_halumem_data(variant)
    total_questions = sum(len(list(iter_questions(user, version))) for user in users)
    print(
        f"📚 Loaded HaluMem-{variant.capitalize()} dataset "
        f"({len(users)} users, {total_questions} questions)"
    )
    print(f"⚙️  Search parameters: top_k={top_k}, workers={num_workers}")
    print(
        "⚙️  Failure controls: "
        f"allow_empty_search={allow_empty_search}, "
        f"skip_failed_search={skip_failed_search}"
    )
    print("-" * 80)

    all_search_results = {}
    all_status_records = []
    failed_users = []
    start_time = datetime.now()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_uuid = {
            executor.submit(
                process_user,
                user_obj,
                frame,
                version,
                top_k,
                allow_empty_search=allow_empty_search,
                skip_failed_search=skip_failed_search,
            ): user_obj["uuid"]
            for user_obj in users
        }

        for future in track(
            as_completed(future_to_uuid),
            total=len(future_to_uuid),
            description="Searching users",
        ):
            uuid = str(future_to_uuid[future])
            try:
                search_results, blocking_records = future.result()
                for user_id, results in search_results.items():
                    all_search_results[user_id] = results
                    all_status_records.extend(results)
                if blocking_records:
                    failed_users.append({
                        "user_uuid": uuid,
                        "user_id": user_id_for(version, uuid),
                        "failures": blocking_records,
                    })
            except Exception as e:
                print(f"❌ Error searching user {uuid}: {e}")
                failed_users.append({
                    "user_uuid": uuid,
                    "user_id": user_id_for(version, uuid),
                    "error": error_payload("search", e),
                })

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    elapsed_time_str = str(elapsed_time).split(".")[0]

    results_dir = f"results/halumem/{frame}-{version}"
    os.makedirs(results_dir, exist_ok=True)
    output_path = f"{results_dir}/{frame}_hm_search_results.json"
    atomic_json_dump(dict(all_search_results), output_path, indent=4)
    atomic_json_dump(
        {
            "stage": "search",
            "allow_empty_search": allow_empty_search,
            "skip_failed_search": skip_failed_search,
            "status_counts": status_counts(all_status_records),
            "failed_users": failed_users,
        },
        f"{results_dir}/{frame}_hm_search_status.json",
        indent=2,
    )

    if failed_users:
        print("\n" + "=" * 80)
        failure_message = (
            f"❌ SEARCH FAILED: {len(failed_users)}/{len(users)} users had errors"
        )
        print(failure_message.center(80))
        print("=" * 80)
        print(f"⏱️  Total time: {elapsed_time_str}")
        print(f"🔄 Framework: {frame} | Version: {version} | Workers: {num_workers}")
        print("=" * 80 + "\n")
        raise SystemExit(1)

    print("\n" + "=" * 80)
    print("✅ SEARCH COMPLETE".center(80))
    print("=" * 80)
    print(f"⏱️  Total time: {elapsed_time_str}")
    print(f"🔄 Framework: {frame} | Version: {version} | Workers: {num_workers}")
    print(f"📁 Results saved to: {output_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HaluMem Search Script")
    parser.add_argument(
        "--lib",
        type=str,
        choices=SUPPORTED_LIBS,
        default=DEFAULT_LIB,
    )
    parser.add_argument(
        "--version",
        type=str,
        default="default",
        help="Version of the evaluation framework.",
    )
    parser.add_argument(
        "--top-k",
        "--top-k",
        type=int,
        default=20,
        help="Number of top results to retrieve from the search.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="medium",
        choices=["medium", "long"],
        help="HaluMem dataset variant (medium or long)",
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
        help=(
            "Explicitly mark failed search calls as skipped instead of failing "
            "the step. Default: 0."
        ),
    )

    args = parser.parse_args()
    main(
        frame=args.lib,
        version=args.version,
        variant=args.variant,
        top_k=args.top_k,
        num_workers=args.workers,
        allow_empty_search=args.allow_empty_search,
        skip_failed_search=args.skip_failed_search,
    )
