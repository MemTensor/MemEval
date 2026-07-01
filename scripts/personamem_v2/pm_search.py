import argparse
import csv
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from personamem_v2.pm_common import (
    STATUS_FAILED,
    STATUS_SKIPPED,
    build_search_entry,
    classify_search_status,
    error_payload,
    get_single_search_entry,
    memory_user_id_for,
    record_status,
    result_key_for,
    search_allowed_statuses,
    status_counts,
    validate_single_search_result,
)
from utils.checkpoint import atomic_json_dump
from utils.env import load_env
from utils.progress import track
from utils.response_options import parse_bool
from utils.search_helpers import dispatch_search, unpack_search_result
from client_factory import SUPPORTED_LIBS, DEFAULT_LIB

BENCHMARK_CSV = "data/personamem_v2/benchmark/text/benchmark.csv"
CHAT_HISTORY_DIR = "data/personamem_v2/data/chat_history_32k"


def build_chat_history_index(chat_history_dir):
    index = {}
    for filepath in glob.glob(os.path.join(chat_history_dir, "*.json")):
        basename = os.path.basename(filepath)
        parts = basename.split("_persona")
        if len(parts) == 2:
            pid = int(parts[1].replace(".json", ""))
            index[pid] = filepath
    return index


def build_options(correct_answer, incorrect_answers_str, row_idx):
    random.seed(row_idx)
    incorrect_answers = json.loads(incorrect_answers_str)
    all_answers = [("correct", correct_answer)] + [
        ("incorrect", ans) for ans in incorrect_answers
    ]
    random.shuffle(all_answers)

    option_labels = ["a", "b", "c", "d"]
    correct_label = None
    options_lines = []

    for i, (tag, ans) in enumerate(all_answers):
        label = option_labels[i]
        options_lines.append(f"({label}) {ans}")
        if tag == "correct":
            correct_label = f"({label})"

    return "\n".join(options_lines), correct_label


def parse_user_query(user_query_str):
    import ast
    try:
        parsed = ast.literal_eval(user_query_str)
        if isinstance(parsed, dict):
            return parsed.get("content", user_query_str)
        return user_query_str
    except (ValueError, SyntaxError):
        return user_query_str


def load_benchmark_rows(csv_path, chat_history_index):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            pid = int(row["persona_id"])
            if pid not in chat_history_index:
                continue
            rows.append((idx, row))
    return rows


def load_benchmark_persona_ids(csv_path):
    persona_ids = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            persona_ids.add(int(row["persona_id"]))
    return persona_ids


# ── Main processing ──────────────────────────────────────────────────────────


def process_user(
    row_data,
    row_idx,
    frame,
    version,
    top_k=20,
    *,
    allow_empty_search=True,
    skip_failed_search=False,
    allow_missing_data=False,
):
    persona_id = int(row_data["persona_id"])
    question = parse_user_query(row_data["user_query"])
    pref_type = row_data["pref_type"]
    topic = row_data["topic_query"]
    correct_answer = row_data["correct_answer"]

    all_options, golden_answer = build_options(
        correct_answer, row_data["incorrect_answers"], row_idx
    )

    memory_user_id = memory_user_id_for(version, persona_id)
    result_key = result_key_for(version, row_idx)

    search_results = defaultdict(list)
    print(f"\n  [{row_idx + 1}] Processing persona {persona_id}")
    print(f"  Question: {question[:80]}...")
    print(f"  Type: {pref_type}")

    allowed_statuses = search_allowed_statuses(
        allow_empty_search=allow_empty_search,
        allow_skipped=skip_failed_search,
    )
    existing_results, exists = load_existing_results(frame, version, row_idx)
    if exists:
        ok, issues = validate_single_search_result(
            existing_results,
            result_key=result_key,
            question=question,
            row_idx=row_idx,
            allowed_statuses=allowed_statuses,
            require_status=True,
        )
        if not ok:
            print(
                f"  Existing result for row {row_idx} is incomplete; "
                f"will retry ({'; '.join(issues)})"
            )
        else:
            entry = get_single_search_entry(existing_results, result_key)
            blocking = []
            if entry and record_status(entry) not in allowed_statuses:
                blocking.append(entry)
            print(f"  Using existing results for row {row_idx}")
            return existing_results, blocking

    from client_factory import create_client

    client = create_client(frame)

    try:
        result = dispatch_search(frame, client, question, memory_user_id, top_k)

        context, duration_ms, reflect_answer, raw_context = unpack_search_result(result)
    except Exception as e:
        print(f"  ❌ Search failed for persona {persona_id}, row {row_idx}: {e}")
        status = STATUS_SKIPPED if skip_failed_search else STATUS_FAILED
        entry = build_search_entry(
            result_key=result_key,
            persona_id=persona_id,
            row_idx=row_idx,
            question=question,
            category=pref_type,
            all_options=all_options,
            topic=topic,
            golden_answer=golden_answer,
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
            result_key=result_key,
            persona_id=persona_id,
            row_idx=row_idx,
            question=question,
            category=pref_type,
            all_options=all_options,
            topic=topic,
            golden_answer=golden_answer,
            context=context,
            duration_ms=duration_ms,
            status=status,
            reflect_answer=reflect_answer,
        )
    search_results[result_key].append(entry)

    os.makedirs(f"results/pmv2/{frame}-{version}/tmp", exist_ok=True)
    atomic_json_dump(
        dict(search_results),
        f"results/pmv2/{frame}-{version}/tmp/{frame}_pm_search_results_{row_idx}.json",
        indent=4,
    )
    print(
        f"  Search results for row {row_idx} saved "
        f"(duration: {entry.get('search_duration_ms', 0):.0f}ms)"
    )

    blocking_records = []
    if record_status(entry) not in allowed_statuses:
        blocking_records.append(entry)
    return search_results, blocking_records


def load_existing_results(frame, version, row_idx):
    result_path = f"results/pmv2/{frame}-{version}/tmp/{frame}_pm_search_results_{row_idx}.json"
    if os.path.exists(result_path):
        try:
            with open(result_path) as f:
                return json.load(f), True
        except Exception as e:
            print(f"  Error loading existing results for row {row_idx}: {e}")
    return {}, False


def main(
    frame,
    version,
    top_k=20,
    num_workers=2,
    *,
    allow_empty_search=True,
    skip_failed_search=False,
    allow_missing_data=False,
):
    load_env()

    print("\n" + "=" * 80)
    print(f"  PERSONAMEM V2 SEARCH - {frame.upper()} v{version}".center(80))
    print("=" * 80)

    chat_history_index = build_chat_history_index(CHAT_HISTORY_DIR)
    benchmark_personas = load_benchmark_persona_ids(BENCHMARK_CSV)
    missing_personas = sorted(set(benchmark_personas) - set(chat_history_index))
    if missing_personas and not allow_missing_data:
        print(
            f"❌ Missing PersonaMem chat history files for {len(missing_personas)} "
            "benchmark personas."
        )
        print("Use --allow-missing-data 1 to explicitly evaluate the available subset.")
        raise SystemExit(1)
    benchmark_rows = load_benchmark_rows(BENCHMARK_CSV, chat_history_index)
    total_rows = len(benchmark_rows)

    print(f"  Loaded {total_rows} questions with available chat histories")
    print(f"  Search parameters: top_k={top_k}, workers={num_workers}")
    print(
        "  Failure controls: "
        f"allow_empty_search={allow_empty_search}, "
        f"skip_failed_search={skip_failed_search}"
    )
    print("-" * 80)

    all_search_results = defaultdict(list)
    start_time = datetime.now()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(
                process_user,
                row_data=row_data,
                row_idx=row_idx,
                version=version,
                frame=frame,
                top_k=top_k,
                allow_empty_search=allow_empty_search,
                skip_failed_search=skip_failed_search,
            ): row_idx
            for row_idx, row_data in benchmark_rows
        }

        failed_rows = []
        all_status_records = []
        for future in track(
            as_completed(future_to_idx), total=len(future_to_idx), description="Searching questions"
        ):
            idx = future_to_idx[future]
            try:
                search_results, blocking_records = future.result()
                for user_id, results in search_results.items():
                    all_search_results[user_id].extend(results)
                    all_status_records.extend(results)
                if blocking_records:
                    failed_rows.append({
                        "row_idx": idx,
                        "failures": blocking_records,
                    })
            except Exception as exc:
                print(f"\n❌ Row {idx} generated an exception: {exc}")
                failed_rows.append({
                    "row_idx": idx,
                    "error": error_payload("search", exc),
                })

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    elapsed_time_str = str(elapsed_time).split(".")[0]

    results_dir = f"results/pmv2/{frame}-{version}"
    os.makedirs(results_dir, exist_ok=True)
    output_path = f"{results_dir}/{frame}_pm_search_results.json"
    atomic_json_dump(dict(all_search_results), output_path, indent=4)
    atomic_json_dump(
        {
            "stage": "search",
            "allow_empty_search": allow_empty_search,
            "skip_failed_search": skip_failed_search,
            "status_counts": status_counts(all_status_records),
            "failed_users": failed_rows,
        },
        f"{results_dir}/{frame}_pm_search_status.json",
        indent=2,
    )

    if failed_rows:
        print("\n" + "=" * 80)
        print(f"❌ SEARCH FAILED: {len(failed_rows)}/{total_rows} rows had errors".center(80))
        print("=" * 80)
        print(f"  Total time: {elapsed_time_str}")
        print(f"  Framework: {frame} | Version: {version} | Workers: {num_workers}")
        print("=" * 80 + "\n")
        raise SystemExit(1)

    print("\n" + "=" * 80)
    print("✅ SEARCH COMPLETE".center(80))
    print("=" * 80)
    print(f"  Total time: {elapsed_time_str}")
    print(f"  Framework: {frame} | Version: {version} | Workers: {num_workers}")

    print(f"  Results saved to: {output_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PersonaMem v2 Search Script")
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
        "--top-k", type=int, default=20, help="Number of top results to retrieve from the search."
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="Number of parallel search workers."
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
    parser.add_argument(
        "--allow-missing-data",
        "--allow_missing_data",
        type=parse_bool,
        default=False,
        help="Allow missing PersonaMem chat histories and evaluate the available subset. Default: 0.",
    )

    args = parser.parse_args()

    main(
        frame=args.lib,
        version=args.version,
        top_k=args.top_k,
        num_workers=args.workers,
        allow_empty_search=args.allow_empty_search,
        skip_failed_search=args.skip_failed_search,
        allow_missing_data=args.allow_missing_data,
    )
