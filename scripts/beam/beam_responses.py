import argparse
import asyncio
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, SCRIPT_DIR)

from time import time

from beam_common import (
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
from utils.env import load_env
from utils.progress import create_progress
from utils.prompts import BEAM_ANSWER_PROMPT
from utils.response_options import add_save_model_input_arg, parse_bool
from client_factory import SUPPORTED_LIBS, DEFAULT_LIB



async def beam_response(llm_client, model_name, context: str, question: str, frame=None):
    prompt = BEAM_ANSWER_PROMPT.format(context=context, question=question)
    messages = [{"role": "user", "content": prompt}]
    response = await llm_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
    )
    result = response.choices[0].message.content or ""
    return result, messages


async def process_question(q_data, oai_client, model_name, semaphore, frame=None, save_model_input=False):
    async with semaphore:
        start = time()
        question = q_data.get("question", "")
        golden_answer = q_data.get("golden_answer", "")
        rubric = q_data.get("rubric", "")
        dimension = q_data.get("dimension", "")
        scale = q_data.get("scale", "")
        difficulty = q_data.get("difficulty", "")
        context = q_data.get("search_context", "")
        reflect_answer = q_data.get("reflect_answer")

        if reflect_answer:
            answer = reflect_answer
            model_input = None
        else:
            answer, model_input = await beam_response(oai_client, model_name, context, question, frame=frame)
        response_duration_ms = (time() - start) * 1000

        response_record = {
            "key": q_data.get("key"),
            "conv_id": q_data.get("conv_id"),
            "question_idx": q_data.get("question_idx"),
            "question": question,
            "answer": answer,
            "golden_answer": golden_answer,
            "rubric": rubric,
            "dimension": dimension,
            "scale": scale,
            "difficulty": difficulty,
            "response_duration_ms": response_duration_ms,
            "search_duration_ms": q_data.get("search_duration_ms", 0),
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
    version="default",
    llm_workers=10,
    save_model_input=False,
    *,
    skip_failed_answer=False,
):
    search_path = f"results/beam/{frame}-{version}/{frame}_beam_search_results.json"
    response_path = f"results/beam/{frame}-{version}/{frame}_beam_responses.json"

    load_env()
    from utils.llm_client import create_async_openai_client

    oai_client, answer_model = create_async_openai_client("ANSWER")
    print(f"[ANSWER] model={answer_model}")

    with open(search_path) as file:
        search_results = json.load(file)

    semaphore = asyncio.Semaphore(llm_workers)

    all_responses = {}
    total_questions = 0
    failed_users = []
    skipped_records = []

    if os.path.exists(response_path):
        try:
            with open(response_path) as f:
                all_responses = json.load(f)
            print(f"♻️  Loaded {len(all_responses)} existing users for checkpoint/resume")
        except Exception:
            all_responses = {}

    user_ids = sorted(search_results.keys())
    for uid_idx, user_id in enumerate(user_ids):
        questions = search_results[user_id]
        existing = all_responses.get(user_id, [])
        existing_by_key = {
            str(record.get("key")): record
            for record in existing
            if isinstance(record, dict) and record.get("key")
        }
        valid_by_key = {}
        for q in questions:
            key = str(q.get("key") or "")
            existing_record = existing_by_key.get(key)
            ok, issues = response_complete(existing_record, q)
            if ok:
                valid_by_key[key] = existing_record
            elif existing_record is not None:
                print(
                    f"♻️  Reprocessing {key}; existing response incomplete "
                    f"({'; '.join(issues)})"
                )

        pending = []
        responses_by_key = dict(valid_by_key)
        for q in questions:
            key = str(q.get("key") or "")
            if key in responses_by_key:
                continue
            status = record_status(q)
            if status == STATUS_SKIPPED:
                skipped = skipped_response_record(
                    search_entry=q,
                    reason="search was explicitly skipped",
                    error=q.get("error"),
                )
                responses_by_key[key] = skipped
                skipped_records.append({"user_id": user_id, **skipped})
                continue
            if status == STATUS_FAILED:
                failed_users.append({
                    "user_id": user_id,
                    "key": key,
                    "error": q.get("error")
                    or error_payload("answer", "search result is failed"),
                })
                continue
            pending.append(q)

        if valid_by_key and not pending:
            print(f"♻️  Skipping {user_id} (already processed)")
            all_responses[user_id] = [
                responses_by_key[str(q.get("key"))]
                for q in questions
                if str(q.get("key")) in responses_by_key
            ]
            total_questions += len(all_responses[user_id])
            continue

        pending_by_key = {str(q.get("key") or ""): q for q in pending}
        tasks = [
            _with_key(
                str(q.get("key") or ""),
                process_question(
                    q,
                    oai_client,
                    answer_model,
                    semaphore,
                    frame=frame,
                    save_model_input=save_model_input,
                ),
            )
            for q in pending
        ]

        pbar_desc = f"[{uid_idx + 1}/{len(user_ids)}] {user_id}"
        with create_progress() as progress:
            task_id = progress.add_task(pbar_desc, total=len(tasks))
            for coro in asyncio.as_completed(tasks):
                key, result, exc = await coro
                if exc:
                    print(f"❌ Error generating response for {user_id}/{key}: {exc}")
                    failure = {
                        "user_id": user_id,
                        "key": key,
                        "error": error_payload("answer", exc),
                    }
                    if skip_failed_answer:
                        skipped = skipped_response_record(
                            search_entry=pending_by_key.get(key, {}),
                            reason="answer_failed",
                            error=failure["error"],
                        )
                        responses_by_key[key] = skipped
                        skipped_records.append({"user_id": user_id, **skipped})
                    else:
                        failed_users.append(failure)
                else:
                    responses_by_key[str(result.get("key"))] = result
                all_responses[user_id] = [
                    responses_by_key[str(q.get("key"))]
                    for q in questions
                    if str(q.get("key")) in responses_by_key
                ]
                atomic_json_dump(all_responses, response_path, indent=2)
                progress.advance(task_id)

        all_responses[user_id] = [
            responses_by_key[str(q.get("key"))]
            for q in questions
            if str(q.get("key")) in responses_by_key
        ]
        total_questions += len(all_responses[user_id])

        atomic_json_dump(all_responses, response_path, indent=2)

    print(f"Total: {total_questions} questions across {len(user_ids)} users")

    atomic_json_dump(all_responses, response_path, indent=2)
    print(f"Save response results to {response_path}")

    from utils.token_tracker import get_tracker

    results_dir = f"results/beam/{frame}-{version}"
    get_tracker().save(f"{results_dir}/token_usage_answer.json")
    all_response_records = [
        record for records in all_responses.values() for record in records
    ]
    atomic_json_dump(
        {
            "stage": "answer",
            "skip_failed_answer": skip_failed_answer,
            "status_counts": status_counts(all_response_records),
            "failed_users": failed_users,
            "skipped_records": skipped_records,
        },
        f"{results_dir}/{frame}_beam_response_status.json",
        indent=2,
    )

    if failed_users:
        print(f"\n❌ RESPONSE GENERATION FAILED: {len(failed_users)} users had errors")
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BEAM Response Generation Script")
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
        help="Version identifier for loading results.",
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
            args.lib,
            args.version,
            llm_workers=args.llm_workers,
            save_model_input=args.save_model_input,
            skip_failed_answer=args.skip_failed_answer,
        )
    )
