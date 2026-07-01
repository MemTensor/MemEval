import argparse
import asyncio
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import transformers

from client_factory import DEFAULT_LIB, SUPPORTED_LIBS
from halumem.hm_common import (
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    error_payload,
    grade_complete,
    record_status,
    status_counts,
)
from utils.checkpoint import atomic_json_dump
from utils.env import load_env
from utils.nlp_metrics import (
    LLMGrade,
    calculate_nlp_metrics,
    extract_label_json,
    init_nlp,
)
from utils.progress import create_progress
from utils.prompts import HM_JUDGE_PROMPT, JUDGE_SYSTEM_PROMPT
from utils.response_options import parse_bool


logging.basicConfig(level=logging.CRITICAL)
transformers.logging.set_verbosity_error()


async def hm_grader(
    llm_client,
    eval_model_name,
    question,
    golden_answer,
    response,
    semaphore: asyncio.Semaphore,
):
    judge_prompt = HM_JUDGE_PROMPT.format(
        question=question,
        golden_answer=golden_answer,
        response=response,
    )
    async with semaphore:
        api_response = await llm_client.chat.completions.create(
            model=eval_model_name,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0,
        )
    message_content = api_response.choices[0].message.content or ""
    label_json = extract_label_json(text=message_content)
    if label_json is None:
        raise ValueError(
            f"could not extract judge label from response: {message_content[:200]}"
        )
    label = json.loads(label_json)["label"]
    parsed = LLMGrade(llm_judgment=label, llm_reasoning="")
    return parsed.llm_judgment.strip().lower() == "correct"


def convert_numpy_types(obj):
    if isinstance(obj, np.number):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj


async def process_qa(
    key,
    response_data,
    llm_client,
    eval_model_name,
    num_runs,
    llm_semaphore: asyncio.Semaphore,
    nlp_options,
):
    question = response_data.get("question")
    golden_answer = response_data.get("golden_answer", "")
    context = response_data.get("search_context", "")
    response = response_data.get("answer", "")

    grading_tasks = [
        hm_grader(
            llm_client,
            eval_model_name,
            question,
            golden_answer,
            response,
            llm_semaphore,
        )
        for _ in range(num_runs)
    ]
    judgments = await asyncio.gather(*grading_tasks, return_exceptions=True)
    errors = [judgment for judgment in judgments if isinstance(judgment, Exception)]
    if errors:
        raise RuntimeError(f"judge failed: {errors[0]}") from errors[0]
    judgments_dict = {f"judgment_{i + 1}": j for i, j in enumerate(judgments)}

    nlp_metrics = calculate_nlp_metrics(golden_answer, response, context, nlp_options)

    print(
        f"  ⚖️  [{key}] "
        + ", ".join(
            f"run{i + 1}: {'✓' if j else '✗'}" for i, j in enumerate(judgments)
        )
    )

    return {
        "key": key,
        "user_id": response_data.get("user_id"),
        "category": response_data.get("category"),
        "difficulty": response_data.get("difficulty"),
        "question": question,
        "answer": response,
        "golden_answer": golden_answer,
        "evidence": response_data.get("evidence", ""),
        "llm_judgments": judgments_dict,
        "nlp_metrics": nlp_metrics,
        "response_duration_ms": response_data.get("response_duration_ms", 0.0),
        "search_duration_ms": response_data.get("search_duration_ms", 0.0),
        "total_duration_ms": response_data.get("response_duration_ms", 0.0)
        + response_data.get("search_duration_ms", 0.0),
        "status": STATUS_SUCCESS,
    }


async def _with_key(key, coro):
    try:
        return key, await coro, None
    except Exception as exc:
        return key, None, exc


def skipped_grade_record(key, response_data, *, reason, error=None):
    response_duration = response_data.get("response_duration_ms") or 0.0
    search_duration = response_data.get("search_duration_ms") or 0.0
    record = {
        "key": key,
        "user_id": response_data.get("user_id"),
        "category": response_data.get("category"),
        "difficulty": response_data.get("difficulty"),
        "question": response_data.get("question"),
        "answer": response_data.get("answer", ""),
        "golden_answer": response_data.get("golden_answer"),
        "evidence": response_data.get("evidence", ""),
        "response_duration_ms": response_duration,
        "search_duration_ms": search_duration,
        "total_duration_ms": response_duration + search_duration,
        "status": STATUS_SKIPPED,
        "skip_reason": reason,
    }
    if error is not None:
        record["error"] = error
    return record


async def main(
    frame,
    version="default",
    nlp_options=None,
    num_runs=1,
    llm_workers=10,
    *,
    skip_failed_judge=False,
):
    init_nlp()
    print(f"Starting HaluMem evaluation for {frame} version {version}...")

    load_env()
    from utils.llm_client import create_async_openai_client

    oai_client, eval_model = create_async_openai_client("EVAL")
    print(f"[EVAL] model={eval_model}")

    results_dir = f"results/halumem/{frame}-{version}"
    response_path = f"{results_dir}/{frame}_hm_responses.json"
    search_path = f"{results_dir}/{frame}_hm_search_results.json"
    judged_path = f"{results_dir}/{frame}_hm_judged.json"

    with open(response_path) as file:
        hm_responses = json.load(file)

    if os.path.exists(search_path):
        with open(search_path) as f:
            hm_search_data = json.load(f)
        for _, entries in hm_search_data.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                key = str(entry.get("key") or "")
                if key in hm_responses:
                    hm_responses[key].setdefault(
                        "search_context",
                        entry.get("search_context", ""),
                    )
        print(f"📂 Loaded search contexts from: {search_path}")

    print(f"Found {len(hm_responses)} questions to evaluate")

    hm_eval_results = {}
    if os.path.exists(judged_path):
        try:
            with open(judged_path) as f:
                hm_eval_results = json.load(f)
            print(f"♻️  Loaded {len(hm_eval_results)} existing results for checkpoint/resume")
        except Exception:
            hm_eval_results = {}

    tasks = []
    skipped_records = []
    failed_keys = []
    already_done = 0
    llm_semaphore = asyncio.Semaphore(llm_workers)

    for key, response_data in hm_responses.items():
        if record_status(response_data) == STATUS_SKIPPED:
            hm_eval_results[key] = skipped_grade_record(
                key,
                response_data,
                reason=response_data.get("skip_reason", "response was skipped"),
                error=response_data.get("error"),
            )
            skipped_records.append({
                "key": key,
                "question": response_data.get("question"),
                "reason": response_data.get("skip_reason", "response was skipped"),
                "error": response_data.get("error"),
            })
            continue

        if key in hm_eval_results:
            ok, issues = grade_complete(
                hm_eval_results.get(key),
                response_data,
                num_runs,
                allow_skipped_grade=skip_failed_judge,
            )
            if ok:
                already_done += 1
                continue
            print(
                f"♻️  Reprocessing {key}; existing grade incomplete "
                f"({'; '.join(issues)})"
            )
            hm_eval_results.pop(key, None)

        tasks.append(
            _with_key(
                key,
                process_qa(
                    key,
                    response_data,
                    oai_client,
                    eval_model,
                    num_runs,
                    llm_semaphore,
                    nlp_options,
                ),
            )
        )

    if already_done:
        print(f"♻️  Skipping {already_done} already-evaluated questions")

    with create_progress() as progress:
        task_id = progress.add_task("Evaluating questions", total=len(tasks))
        for coro in asyncio.as_completed(tasks):
            key, result, exc = await coro
            if exc:
                response_data = hm_responses.get(key, {})
                failure = {
                    "key": key,
                    "error": error_payload("eval", exc),
                }
                if skip_failed_judge:
                    hm_eval_results[key] = skipped_grade_record(
                        key,
                        response_data,
                        reason="eval_failed",
                        error=failure["error"],
                    )
                    skipped_records.append(failure)
                    atomic_json_dump(
                        convert_numpy_types(hm_eval_results),
                        judged_path,
                        indent=4,
                    )
                else:
                    failed_keys.append(failure)
                print(f"[ERROR] Processing {key} failed: {exc}")
            else:
                hm_eval_results[result["key"]] = result
                atomic_json_dump(
                    convert_numpy_types(hm_eval_results),
                    judged_path,
                    indent=4,
                )
            progress.advance(task_id)

    all_judgment_keys = set()
    for r in hm_eval_results.values():
        if record_status(r) != STATUS_SUCCESS:
            continue
        all_judgment_keys.update(r.get("llm_judgments", {}).keys())

    run_scores = []
    for k in sorted(all_judgment_keys):
        correct = sum(
            1
            for r in hm_eval_results.values()
            if record_status(r) == STATUS_SUCCESS
            and r.get("llm_judgments", {}).get(k)
        )
        total = sum(
            1
            for r in hm_eval_results.values()
            if record_status(r) == STATUS_SUCCESS
            and k in r.get("llm_judgments", {})
        )
        if total > 0:
            run_scores.append(correct / total)

    print("\n" + "=" * 80)
    print("📊 EVALUATION SUMMARY".center(80))
    print("=" * 80)

    if run_scores:
        print(f"📋 Evaluated: {len(hm_eval_results)} questions across {num_runs} runs")
        print(f"🎯 LLM-as-a-Judge Mean Accuracy: {np.mean(run_scores):.4f}")
        print(f"🔍 Standard Deviation: {np.std(run_scores):.4f}")
        run_scores_formatted = [f"{round(s, 4):.4f}" for s in run_scores]
        print(f"🔢 Individual run scores: [{', '.join(run_scores_formatted)}]")
    else:
        print("⚠️  No responses were evaluated.")

    print("-" * 80)

    hm_eval_results = convert_numpy_types(hm_eval_results)
    atomic_json_dump(hm_eval_results, judged_path, indent=4)
    print(f"📁 Results saved to: {judged_path}")

    atomic_json_dump(
        {
            "stage": "eval",
            "skip_failed_judge": skip_failed_judge,
            "status_counts": status_counts(hm_eval_results.values()),
            "failed_keys": failed_keys,
            "skipped_records": skipped_records,
        },
        f"{results_dir}/{frame}_hm_eval_status.json",
        indent=2,
    )

    if failed_keys:
        print(f"\n❌ EVALUATION FAILED: {len(failed_keys)} questions had errors")
        raise SystemExit(1)

    print("✅ Evaluation completed successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HaluMem LLM-as-Judge Evaluation")
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
        "--options",
        type=str,
        nargs="+",
        default=["lexical"],
        choices=["lexical", "semantic"],
        help="NLP options to use for evaluation.",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of runs for LLM-as-a-Judge evaluation.",
    )
    parser.add_argument(
        "--llm-workers",
        "--llm_workers",
        type=int,
        default=10,
        help="Max concurrent LLM API calls.",
    )
    parser.add_argument(
        "--skip-failed-judge",
        "--skip_failed_judge",
        type=parse_bool,
        default=False,
        help="Explicitly skip failed judge calls instead of failing the step. Default: 0.",
    )

    args = parser.parse_args()
    asyncio.run(
        main(
            frame=args.lib,
            version=args.version,
            nlp_options=args.options,
            num_runs=args.num_runs,
            llm_workers=args.llm_workers,
            skip_failed_judge=args.skip_failed_judge,
        )
    )
