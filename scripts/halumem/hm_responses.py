import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from time import time

from client_factory import DEFAULT_LIB, SUPPORTED_LIBS
from halumem.hm_common import (
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    error_payload,
    record_status,
    response_complete,
    search_entry_map,
    skipped_response_record,
    status_counts,
)
from utils.checkpoint import atomic_json_dump
from utils.env import load_env
from utils.progress import create_progress
from utils.prompts import HM_ANSWER_PROMPT
from utils.response_options import add_save_model_input_arg, parse_bool


async def hm_response(llm_client, model_name, context, question, frame=None):
    prompt = HM_ANSWER_PROMPT.format(
        question=question,
        context=context,
    )
    messages = [{"role": "user", "content": prompt}]
    response = await llm_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
    )
    result = response.choices[0].message.content or ""
    return result, messages


async def process_qa(
    user_id,
    search_result,
    llm_client,
    model_name,
    semaphore,
    frame=None,
    save_model_input=False,
):
    async with semaphore:
        start = time()
        question = search_result.get("question")
        context = search_result.get("search_context", "")
        reflect_answer = search_result.get("reflect_answer")

        if reflect_answer:
            answer = reflect_answer
            model_input = None
        else:
            answer, model_input = await hm_response(
                llm_client,
                model_name,
                context,
                question,
                frame=frame,
            )

        response_duration_ms = (time() - start) * 1000

        print("\n" + "-" * 80)
        print(f"🤖 Processed: {user_id}")
        print(f"⏱️  Duration: {response_duration_ms:.2f} ms")
        print(f"❓ Question: {question[:120]}")
        print(
            f"💬 Answer: {answer[:150]}..."
            if len(answer) > 150
            else f"💬 Answer: {answer}"
        )
        print("-" * 80)

        response_record = {
            "key": search_result.get("key"),
            "user_id": user_id,
            "category": search_result.get("category"),
            "difficulty": search_result.get("difficulty"),
            "question": question,
            "answer": answer,
            "golden_answer": search_result.get("golden_answer"),
            "evidence": search_result.get("evidence", ""),
            "response_duration_ms": response_duration_ms,
            "search_duration_ms": search_result.get("search_duration_ms"),
            "session_idx": search_result.get("session_idx"),
            "question_idx": search_result.get("question_idx"),
            "status": STATUS_SUCCESS,
        }
        if save_model_input:
            response_record["model_input"] = model_input
        return response_record


async def _with_key(key, coro):
    try:
        return key, await coro, None
    except Exception as exc:
        return key, None, exc


async def main(
    frame,
    version,
    llm_workers=10,
    save_model_input=False,
    *,
    skip_failed_answer=False,
):
    print("\n" + "=" * 80)
    print(f"🚀 HALUMEM RESPONSE GENERATION - {frame.upper()} v{version}".center(80))
    print("=" * 80)

    load_env()
    from utils.llm_client import create_async_openai_client

    oai_client, answer_model = create_async_openai_client("ANSWER")
    print(f"🔌 [ANSWER] model={answer_model}")

    results_dir = f"results/halumem/{frame}-{version}"
    search_path = f"{results_dir}/{frame}_hm_search_results.json"
    response_path = f"{results_dir}/{frame}_hm_responses.json"

    print(f"📂 Loading search results from: {search_path}")
    with open(search_path) as file:
        hm_search_results = json.load(file)

    search_entries = search_entry_map(hm_search_results)
    total_questions = len(search_entries)
    print(f"📊 Found {len(hm_search_results)} users, {total_questions} questions to process")
    print(f"⚙️  Using {llm_workers} LLM worker threads")
    print(f"⚙️  Failure controls: skip_failed_answer={skip_failed_answer}")
    print("-" * 80)

    hm_responses = {}
    if os.path.exists(response_path):
        try:
            with open(response_path) as f:
                hm_responses = json.load(f)
            print(f"♻️  Loaded {len(hm_responses)} existing results for checkpoint/resume")
        except Exception:
            hm_responses = {}

    start_time = time()
    semaphore = asyncio.Semaphore(llm_workers)

    tasks = []
    skipped_existing = 0
    failed_keys = []
    skipped_records = []
    for key, (user_id, search_result) in search_entries.items():
        if key in hm_responses:
            ok, issues = response_complete(hm_responses.get(key), search_result)
            if ok:
                skipped_existing += 1
                continue
            print(
                f"♻️  Reprocessing {key}; existing response incomplete "
                f"({'; '.join(issues)})"
            )
            hm_responses.pop(key, None)

        status = record_status(search_result)
        if status == STATUS_SKIPPED:
            skipped = skipped_response_record(
                search_entry=search_result,
                reason="search was explicitly skipped",
                error=search_result.get("error"),
            )
            hm_responses[key] = skipped
            skipped_records.append(skipped)
            atomic_json_dump(hm_responses, response_path, indent=4)
            continue
        if status == STATUS_FAILED:
            failed_keys.append({
                "key": key,
                "user_id": user_id,
                "error": search_result.get("error")
                or error_payload("answer", "search result is failed"),
            })
            continue

        tasks.append(
            _with_key(
                key,
                process_qa(
                    user_id,
                    search_result,
                    oai_client,
                    answer_model,
                    semaphore,
                    frame=frame,
                    save_model_input=save_model_input,
                ),
            )
        )

    if skipped_existing > 0:
        print(f"♻️  Skipping {skipped_existing} already-processed questions")

    os.makedirs(results_dir, exist_ok=True)
    with create_progress() as progress:
        task_id = progress.add_task("Generating responses", total=len(tasks))
        for coro in asyncio.as_completed(tasks):
            key, result, exc = await coro
            if exc:
                print(f"❌ Error processing {key}: {exc}")
                _, search_result = search_entries.get(key, ("", {}))
                failure = {
                    "key": key,
                    "error": error_payload("answer", exc),
                }
                if skip_failed_answer:
                    skipped = skipped_response_record(
                        search_entry=search_result,
                        reason="answer_failed",
                        error=failure["error"],
                    )
                    hm_responses[key] = skipped
                    skipped_records.append(skipped)
                    atomic_json_dump(hm_responses, response_path, indent=4)
                else:
                    failed_keys.append(failure)
            else:
                hm_responses[key] = result
                atomic_json_dump(hm_responses, response_path, indent=4)
            progress.advance(task_id)

    end_time = time()
    elapsed_sec = int(end_time - start_time)

    atomic_json_dump(hm_responses, response_path, indent=4)
    print(f"📁 Responses saved to: {response_path}")

    from utils.token_tracker import get_tracker

    get_tracker().save(f"{results_dir}/token_usage_answer.json")
    atomic_json_dump(
        {
            "stage": "answer",
            "skip_failed_answer": skip_failed_answer,
            "status_counts": status_counts(hm_responses.values()),
            "failed_keys": failed_keys,
            "skipped_records": skipped_records,
        },
        f"{results_dir}/{frame}_hm_response_status.json",
        indent=2,
    )

    if failed_keys:
        print(
            f"\n❌ RESPONSE GENERATION FAILED: {len(failed_keys)}/"
            f"{total_questions} questions had errors"
        )
        raise SystemExit(1)

    print("\n" + "=" * 80)
    print("✅ RESPONSE GENERATION COMPLETE".center(80))
    print("=" * 80)
    print(f"⏱️ Total time: {elapsed_sec // 60}m {elapsed_sec % 60}s")
    print(f"📊 Processed: {len(hm_responses)} questions")
    print(f"🔄 Framework: {frame} | Version: {version}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HaluMem Response Generation Script")
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
        "--llm-workers",
        "--llm_workers",
        type=int,
        default=10,
        help="Max concurrent LLM API calls.",
    )
    add_save_model_input_arg(parser)
    parser.add_argument(
        "--skip-failed-answer",
        "--skip_failed_answer",
        type=parse_bool,
        default=False,
        help=(
            "Explicitly skip failed answer-generation calls instead of failing "
            "the step. Default: 0."
        ),
    )

    args = parser.parse_args()
    asyncio.run(
        main(
            frame=args.lib,
            version=args.version,
            llm_workers=args.llm_workers,
            save_model_input=args.save_model_input,
            skip_failed_answer=args.skip_failed_answer,
        )
    )
