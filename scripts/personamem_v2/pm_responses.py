import argparse
import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from time import time

from personamem_v2.pm_common import (
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    error_payload,
    record_status,
    response_complete,
    skipped_response_record,
    status_counts,
)
from utils.checkpoint import atomic_json_dump
from utils.progress import create_progress
from utils.prompts import PM_ANSWER_PROMPT
from utils.response_options import add_save_model_input_arg, parse_bool
from client_factory import SUPPORTED_LIBS, DEFAULT_LIB


def extract_choice_answer(predicted_answer, correct_answer):
    def _extract_only_options(text):
        text = text.lower()
        in_parens = re.findall(r"\(([a-d])\)", text)
        if in_parens:
            return set(in_parens)
        else:
            return set(re.findall(r"\b([a-d])\b", text))

    correct = correct_answer.lower().strip("() ")

    full_response = predicted_answer
    predicted_answer = predicted_answer.strip()

    if "<final_answer>" in predicted_answer:
        predicted_answer = predicted_answer.split("<final_answer>")[-1].strip()
    if predicted_answer.endswith("</final_answer>"):
        predicted_answer = predicted_answer[: -len("</final_answer>")].strip()

    pred_options = _extract_only_options(predicted_answer)

    if pred_options == {correct}:
        return True, predicted_answer

    response_options = _extract_only_options(full_response)
    if response_options == {correct}:
        return True, predicted_answer

    return False, predicted_answer


async def pm_response(llm_client, model_name, context, question, options, frame=None):
    prompt = PM_ANSWER_PROMPT.format(
        question=question,
        context=context,
        options=options,
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
    num_runs,
    llm_client,
    model_name,
    semaphore,
    frame=None,
    save_model_input=False,
):
    async with semaphore:
        search_result = search_result[0]
        question = search_result.get("question")
        context = search_result.get("search_context", "")
        options = search_result.get("all_options", [])
        reflect_answer = search_result.get("reflect_answer")

        run_results = []
        model_input = None

        for idx in range(num_runs):
            start = time()
            if reflect_answer:
                raw_answer = reflect_answer
            else:
                raw_answer, model_input = await pm_response(llm_client, model_name, context, question, options, frame=frame)
            is_correct, answer = extract_choice_answer(raw_answer, search_result.get("golden_answer", ""))
            response_duration_ms = (time() - start) * 1000

            run_results.append(
                {
                    "run_id": idx + 1,
                    "answer": answer,
                    "is_correct": is_correct,
                    "response_duration_ms": response_duration_ms,
                }
            )

        response_duration_ms = sum(result["response_duration_ms"] for result in run_results) / num_runs

        print("\n" + "-" * 80)
        print(f"🤖 Processed User: {user_id}")
        print(f"⏱️  Duration: {response_duration_ms:.2f} ms")
        print(f"❓ Question: {question}")
        print(f"💡 Golden Answer: {search_result.get('golden_answer', 'N/A')}")
        for idx, result in enumerate(run_results, start=1):
            print(f"\n🔄 Run {idx}/{num_runs}:")
            print(
                f"💬 Run Answer: {result['answer'][:150]}..."
                if len(result["answer"]) > 150
                else f"💬 Run Answer: {result['answer']}"
            )
            print(f"✅ Run Is Correct: {result['is_correct']}")
            print(f"⏱️  Run Duration: {result['response_duration_ms']:.2f} ms")
        print("-" * 80)

        response_record = {
            "key": search_result.get("key"),
            "user_id": user_id,
            "category": search_result.get("category"),
            "question": question,
            "results": run_results,
            "golden_answer": search_result.get("golden_answer"),
            "all_options": search_result.get("all_options", []),
            "response_duration_ms": response_duration_ms,
            "search_duration_ms": search_result.get("search_duration_ms"),
            "topic": search_result.get("topic"),
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
    num_runs=1,
    llm_workers=10,
    save_model_input=False,
    *,
    skip_failed_answer=False,
):
    print("\n" + "=" * 80)
    print(f"🚀 PERSONAMEM RESPONSE GENERATION - {frame.upper()} v{version}".center(80))
    print("=" * 80)

    from utils.env import load_env
    load_env()
    from utils.llm_client import create_async_openai_client

    oai_client, answer_model = create_async_openai_client("ANSWER")
    print(f"🔌 [ANSWER] model={answer_model}")

    search_path = f"results/pmv2/{frame}-{version}/{frame}_pm_search_results.json"
    response_path = f"results/pmv2/{frame}-{version}/{frame}_pm_responses.json"

    print(f"📂 Loading search results from: {search_path}")
    with open(search_path) as file:
        pm_search_results = json.load(file)
    print(f"📊 Found {len(pm_search_results)} users to process")
    print(f"⚙️  Using {llm_workers} LLM worker threads")
    print("-" * 80)

    pm_responses = {}
    if os.path.exists(response_path):
        try:
            with open(response_path) as f:
                pm_responses = json.load(f)
            print(f"♻️  Loaded {len(pm_responses)} existing results for checkpoint/resume")
        except Exception:
            pm_responses = {}

    search_entry_by_key = {}
    pending_keys = []
    skipped_records = []
    failed_users = []
    for user_id, search_results in pm_search_results.items():
        search_entry = search_results[0] if search_results else {}
        search_entry_by_key[user_id] = search_entry
        existing = pm_responses.get(user_id)
        ok, issues = response_complete(existing, search_entry, num_runs)
        if ok:
            continue
        if existing is not None:
            print(
                f"♻️  Reprocessing {user_id}; existing response incomplete "
                f"({'; '.join(issues)})"
            )
            pm_responses.pop(user_id, None)

        status = record_status(search_entry)
        if status == STATUS_SKIPPED:
            skipped = skipped_response_record(
                search_entry=search_entry,
                reason="search was explicitly skipped",
                error=search_entry.get("error"),
            )
            pm_responses[user_id] = skipped
            skipped_records.append(skipped)
            atomic_json_dump(pm_responses, response_path, indent=4)
            continue
        if status == STATUS_FAILED:
            failed_users.append({
                "user_id": user_id,
                "error": search_entry.get("error")
                or error_payload("answer", "search result is failed"),
            })
            continue
        pending_keys.append(user_id)

    pending_count = len(pending_keys)
    if pending_count < len(pm_search_results):
        print(f"♻️  Skipping {len(pm_search_results) - pending_count} already-processed users")
    print(f"📊 Processing {pending_count} remaining users")

    start_time = time()
    semaphore = asyncio.Semaphore(llm_workers)

    tasks = []
    for user_id in pending_keys:
        search_results = pm_search_results[user_id]
        tasks.append(
            _with_key(
                user_id,
                process_qa(
                    user_id,
                    search_results,
                    num_runs,
                    oai_client,
                    answer_model,
                    semaphore,
                    frame=frame,
                    save_model_input=save_model_input,
                ),
            )
        )

    with create_progress() as progress:
        task_id = progress.add_task("Generating responses", total=len(tasks))
        for coro in asyncio.as_completed(tasks):
            user_id, result, exc = await coro
            if exc:
                print(f"❌ Error processing user {user_id}: {exc}")
                failure = {
                    "user_id": user_id,
                    "error": error_payload("answer", exc),
                }
                if skip_failed_answer:
                    skipped = skipped_response_record(
                        search_entry=search_entry_by_key.get(user_id, {}),
                        reason="answer_failed",
                        error=failure["error"],
                    )
                    pm_responses[user_id] = skipped
                    skipped_records.append(skipped)
                    atomic_json_dump(pm_responses, response_path, indent=4)
                else:
                    failed_users.append(failure)
            else:
                pm_responses[user_id] = result
                atomic_json_dump(pm_responses, response_path, indent=4)
            progress.advance(task_id)

    end_time = time()
    elapsed_time = end_time - start_time
    elapsed_sec = int(elapsed_time)

    atomic_json_dump(pm_responses, response_path, indent=4)
    print(f"📁 Responses saved to: {response_path}")

    from utils.token_tracker import get_tracker

    results_dir = f"results/pmv2/{frame}-{version}"
    get_tracker().save(f"{results_dir}/token_usage_answer.json")
    atomic_json_dump(
        {
            "stage": "answer",
            "skip_failed_answer": skip_failed_answer,
            "status_counts": status_counts(pm_responses.values()),
            "failed_users": failed_users,
            "skipped_records": skipped_records,
        },
        f"{results_dir}/{frame}_pm_response_status.json",
        indent=2,
    )

    if failed_users:
        print(f"\n❌ RESPONSE GENERATION FAILED: {len(failed_users)}/{len(pm_search_results)} users had errors")
        raise SystemExit(1)

    print("\n" + "=" * 80)
    print("✅ RESPONSE GENERATION COMPLETE".center(80))
    print("=" * 80)
    print(f"⏱️  Total time: {elapsed_sec // 60}m {elapsed_sec % 60}s")
    print(f"📊 Processed: {len(pm_responses)} users")
    print(f"🔄 Framework: {frame} | Version: {version}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PersonaMem Response Generation Script")
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
        "--num_runs", type=int, default=1, help="Number of answer generation runs per question."
    )
    parser.add_argument(
        "--llm-workers", "--llm_workers", type=int, default=10, help="Max concurrent LLM API calls."
    )
    add_save_model_input_arg(parser)
    parser.add_argument(
        "--skip-failed-answer",
        "--skip_failed_answer",
        type=parse_bool,
        default=False,
        help="Mark failed answer calls as skipped instead of failing the step. Default: 0.",
    )

    args = parser.parse_args()
    asyncio.run(
        main(
            frame=args.lib,
            version=args.version,
            num_runs=args.num_runs,
            llm_workers=args.llm_workers,
            save_model_input=args.save_model_input,
            skip_failed_answer=args.skip_failed_answer,
        )
    )
