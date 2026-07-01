import argparse
import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beam.beam_common import (
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    error_payload,
    grade_complete,
    record_status,
    status_counts,
)
from utils.checkpoint import atomic_json_dump
from utils.env import load_env
from utils.progress import create_progress
from utils.prompts import BEAM_RUBRIC_ITEM_JUDGE_PROMPT, BEAM_EVENT_ORDERING_JUDGE_PROMPT
from utils.response_options import parse_bool
from client_factory import SUPPORTED_LIBS, DEFAULT_LIB


def _extract_item_score(text: str) -> tuple[float, str]:
    try:
        m = re.search(r'\{[^{}]*"score"\s*:\s*([0-9.]+)[^{}]*\}', text)
        if m:
            parsed = json.loads(m.group(0))
            raw = float(parsed["score"])
            reason = parsed.get("reason", "")
            if raw >= 0.75:
                return 1.0, reason
            elif raw >= 0.25:
                return 0.5, reason
            return 0.0, reason
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        pass

    m = re.search(r'"?score"?\s*[:=]\s*([0-9.]+)', text)
    if m:
        raw = float(m.group(1))
        if raw >= 0.75:
            return 1.0, text[:200]
        elif raw >= 0.25:
            return 0.5, text[:200]
        return 0.0, text[:200]

    return 0.0, f"Failed to parse: {text[:200]}"


async def _judge_rubric_item(llm_client, eval_model, question, rubric_item, response):
    prompt = BEAM_RUBRIC_ITEM_JUDGE_PROMPT.format(
        question=question,
        rubric_item=rubric_item,
        response=response,
    )
    api_response = await llm_client.chat.completions.create(
        model=eval_model,
        messages=[
            {"role": "system", "content": "You are a precise and fair evaluation judge."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    content = api_response.choices[0].message.content or ""
    return _extract_item_score(content)


def _kendall_tau_b(ref_order: list[int], pred_order: list[int]) -> float:
    """Compute Kendall tau-b between two rankings. Returns value in [-1, 1]."""
    n = len(ref_order)
    if n < 2:
        return 1.0

    concordant = 0
    discordant = 0
    ties_ref = 0
    ties_pred = 0
    for i in range(n):
        for j in range(i + 1, n):
            ref_diff = ref_order[i] - ref_order[j]
            pred_diff = pred_order[i] - pred_order[j]
            if ref_diff == 0 and pred_diff == 0:
                ties_ref += 1
                ties_pred += 1
            elif ref_diff == 0:
                ties_ref += 1
            elif pred_diff == 0:
                ties_pred += 1
            elif (ref_diff > 0) == (pred_diff > 0):
                concordant += 1
            else:
                discordant += 1

    n_pairs = n * (n - 1) // 2
    denom = ((n_pairs - ties_ref) * (n_pairs - ties_pred)) ** 0.5
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


async def _judge_event_ordering(llm_client, eval_model, question, rubric_items, response):
    """Use LLM to detect event positions in the response, then compute Kendall tau-b."""
    ref_listing = "\n".join(f"{i+1}. {item}" for i, item in enumerate(rubric_items))
    prompt = BEAM_EVENT_ORDERING_JUDGE_PROMPT.format(
        question=question,
        reference_ordering=ref_listing,
        response=response,
    )
    api_response = await llm_client.chat.completions.create(
        model=eval_model,
        messages=[
            {"role": "system", "content": "You are a precise and fair evaluation judge."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    content = api_response.choices[0].message.content or ""
    try:
        m = re.search(r'\{[^{}]*"positions"\s*:\s*\[.*?\]\s*\}', content, re.DOTALL)
        if not m:
            return 0.0, f"Failed to parse ordering: {content[:300]}", []

        parsed = json.loads(m.group(0))
        positions_list = parsed.get("positions", [])

        detected = []
        for entry in positions_list:
            pos = entry.get("position", -1)
            detected.append(pos)

        if len(detected) != len(rubric_items):
            detected = detected[:len(rubric_items)]
            while len(detected) < len(rubric_items):
                detected.append(-1)

        found = [(i, p) for i, p in enumerate(detected) if p > 0]
        coverage = len(found) / len(rubric_items)

        if len(found) < 2:
            score = 0.0 if coverage == 0 else coverage * 0.5
            return score, f"Too few events detected (coverage={coverage:.2f})", detected

        ref_ranks = [i for i, _ in found]
        pred_ranks = [p for _, p in found]
        tau_b = _kendall_tau_b(ref_ranks, pred_ranks)
        tau_b_norm = max(0.0, (tau_b + 1.0) / 2.0)

        score = tau_b_norm * coverage

        return score, f"tau_b={tau_b:.4f}, coverage={coverage:.2f}, score={score:.4f}", detected

    except (json.JSONDecodeError, TypeError, ValueError, KeyError) as e:
        return 0.0, f"Failed to parse ordering: {e}", []


async def process_response(response_data, oai_client, eval_model, num_runs, semaphore):
    async with semaphore:
        question = response_data.get("question", "")
        answer = response_data.get("answer", "")
        golden_answer = response_data.get("golden_answer", "")
        rubric_items = response_data.get("rubric", [])
        dimension = response_data.get("dimension", "")
        scale = response_data.get("scale", "")
        difficulty = response_data.get("difficulty", "")

        if not rubric_items:
            rubric_items = [f"LLM response should contain: {golden_answer}"]

        response_dur = response_data.get("response_duration_ms", 0)
        search_dur = response_data.get("search_duration_ms", 0)

        if dimension == "event_ordering":
            tasks = [
                _judge_event_ordering(oai_client, eval_model, question, rubric_items, answer)
                for _ in range(num_runs)
            ]
            results = await asyncio.gather(*tasks)
            run_scores = [r[0] for r in results]
            run_reasons = [r[1] for r in results]
            run_positions = [r[2] for r in results]
            nugget_score = sum(run_scores) / len(run_scores)

            return {
                "key": response_data.get("key"),
                "conv_id": response_data.get("conv_id"),
                "question_idx": response_data.get("question_idx"),
                "question": question,
                "answer": answer,
                "golden_answer": golden_answer,
                "rubric": rubric_items,
                "dimension": dimension,
                "scale": scale,
                "difficulty": difficulty,
                "nugget_score": nugget_score,
                "scoring_method": "kendall_tau_b",
                "run_scores": run_scores,
                "run_reasons": run_reasons,
                "run_positions": run_positions,
                "response_duration_ms": response_dur,
                "search_duration_ms": search_dur,
                "total_duration_ms": response_dur + search_dur,
                "status": STATUS_SUCCESS,
            }

        item_details = []
        all_item_scores = []

        for item in rubric_items:
            tasks = [
                _judge_rubric_item(oai_client, eval_model, question, item, answer)
                for _ in range(num_runs)
            ]
            results = await asyncio.gather(*tasks)
            run_scores = [r[0] for r in results]
            run_reasons = [r[1] for r in results]
            item_avg = sum(run_scores) / len(run_scores)
            all_item_scores.append(item_avg)
            item_details.append({
                "rubric_item": item,
                "item_score": item_avg,
                "run_scores": run_scores,
                "run_reasons": run_reasons,
            })

        nugget_score = sum(all_item_scores) / len(all_item_scores)

        return {
            "key": response_data.get("key"),
            "conv_id": response_data.get("conv_id"),
            "question_idx": response_data.get("question_idx"),
            "question": question,
            "answer": answer,
            "golden_answer": golden_answer,
            "rubric": rubric_items,
            "dimension": dimension,
            "scale": scale,
            "difficulty": difficulty,
            "nugget_score": nugget_score,
            "scoring_method": "per_rubric_item",
            "rubric_item_scores": item_details,
            "response_duration_ms": response_dur,
            "search_duration_ms": search_dur,
            "total_duration_ms": response_dur + search_dur,
            "status": STATUS_SUCCESS,
        }


async def _with_key(key, coro):
    try:
        return key, await coro, None
    except Exception as exc:
        return key, None, exc


def skipped_grade_record(response_data, *, reason, error=None):
    response_duration = response_data.get("response_duration_ms") or 0.0
    search_duration = response_data.get("search_duration_ms") or 0.0
    record = {
        "key": response_data.get("key"),
        "conv_id": response_data.get("conv_id"),
        "question_idx": response_data.get("question_idx"),
        "question": response_data.get("question"),
        "answer": response_data.get("answer", ""),
        "golden_answer": response_data.get("golden_answer"),
        "rubric": response_data.get("rubric", []),
        "dimension": response_data.get("dimension", ""),
        "scale": response_data.get("scale", ""),
        "difficulty": response_data.get("difficulty", ""),
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
    llm_workers=10,
    num_runs=1,
    *,
    skip_failed_judge=False,
):
    print(
        f"\n=== Starting BEAM evaluation for {frame} (version: {version}) ==="
    )
    print(f"Using {llm_workers} max concurrent LLM API calls, {num_runs} judge runs per rubric item")

    results_dir = f"results/beam/{frame}-{version}"
    response_path = f"{results_dir}/{frame}_beam_responses.json"
    judged_path = f"{results_dir}/{frame}_beam_judged.json"

    os.makedirs(results_dir, exist_ok=True)

    load_env()
    from utils.llm_client import create_async_openai_client

    oai_client, eval_model = create_async_openai_client("EVAL")
    print(f"[EVAL] model={eval_model}")

    with open(response_path) as file:
        beam_responses = json.load(file)

    semaphore = asyncio.Semaphore(llm_workers)
    all_grades = {}

    if os.path.exists(judged_path):
        try:
            with open(judged_path) as f:
                all_grades = json.load(f)
            print(f"Loaded {len(all_grades)} existing users for checkpoint/resume")
        except Exception:
            all_grades = {}

    total_count = sum(len(v) for v in beam_responses.values())
    print(f"Found {total_count} total responses to evaluate")

    failed_users = []
    skipped_records = []
    for uid_idx, (user_id, responses) in enumerate(sorted(beam_responses.items())):
        if not responses:
            print(f"No responses found for {user_id}")
            continue

        existing = all_grades.get(user_id, [])
        existing_by_key = {
            str(record.get("key")): record
            for record in existing
            if isinstance(record, dict) and record.get("key")
        }
        grades_by_key = {}
        pending = []
        for response in responses:
            key = str(response.get("key") or "")
            if record_status(response) == STATUS_SKIPPED:
                grades_by_key[key] = skipped_grade_record(
                    response,
                    reason=response.get("skip_reason", "response was skipped"),
                    error=response.get("error"),
                )
                skipped_records.append({
                    "user_id": user_id,
                    "key": key,
                    "question": response.get("question"),
                    "reason": response.get("skip_reason", "response was skipped"),
                    "error": response.get("error"),
                })
                continue
            existing_grade = existing_by_key.get(key)
            ok, issues = grade_complete(
                existing_grade,
                response,
                num_runs,
                allow_skipped_grade=skip_failed_judge,
            )
            if ok:
                grades_by_key[key] = existing_grade
                continue
            if existing_grade is not None:
                print(
                    f"Reprocessing {key}; existing grade incomplete "
                    f"({'; '.join(issues)})"
                )
            pending.append(response)

        if grades_by_key and not pending:
            print(f"Skipping {user_id} (already evaluated)")
            all_grades[user_id] = [
                grades_by_key[str(r.get("key"))]
                for r in responses
                if str(r.get("key")) in grades_by_key
            ]
            continue

        tasks = [
            _with_key(
                str(r.get("key") or ""),
                process_response(r, oai_client, eval_model, num_runs, semaphore),
            )
            for r in pending
        ]

        pbar_desc = f"[{uid_idx + 1}/{len(beam_responses)}] {user_id}"
        with create_progress() as progress:
            task_id = progress.add_task(pbar_desc, total=len(tasks))
            for coro in asyncio.as_completed(tasks):
                key, result, exc = await coro
                if exc:
                    response = next(
                        (
                            item for item in pending
                            if str(item.get("key") or "") == key
                        ),
                        {},
                    )
                    failure = {
                        "user_id": user_id,
                        "key": key,
                        "error": error_payload("eval", exc),
                    }
                    if skip_failed_judge:
                        grades_by_key[key] = skipped_grade_record(
                            response,
                            reason="eval_failed",
                            error=failure["error"],
                        )
                        skipped_records.append(failure)
                    else:
                        failed_users.append(failure)
                    print(f"❌ Error evaluating response for {user_id}/{key}: {exc}")
                else:
                    grades_by_key[str(result.get("key"))] = result
                all_grades[user_id] = [
                    grades_by_key[str(r.get("key"))]
                    for r in responses
                    if str(r.get("key")) in grades_by_key
                ]
                atomic_json_dump(all_grades, judged_path, indent=2)
                progress.advance(task_id)

        all_grades[user_id] = [
            grades_by_key[str(r.get("key"))]
            for r in responses
            if str(r.get("key")) in grades_by_key
        ]

        atomic_json_dump(all_grades, judged_path, indent=2)

    evaluated = sum(len(v) for v in all_grades.values())
    if evaluated > 0:
        all_scores = [
            r["nugget_score"]
            for responses in all_grades.values()
            for r in responses
            if record_status(r) == STATUS_SUCCESS
        ]
        if all_scores:
            mean_score = sum(all_scores) / len(all_scores)
            print(f"\nNugget Score (mean): {mean_score:.4f}")
            print(f"Total questions evaluated: {len(all_scores)}")
        else:
            print("No successful responses were evaluated")
    else:
        print("No responses were evaluated")

    atomic_json_dump(all_grades, judged_path, indent=2)
    print(f"Saved evaluation results to {judged_path}")

    from utils.token_tracker import get_tracker
    get_tracker().save(f"{results_dir}/token_usage_eval.json")
    all_grade_records = [record for records in all_grades.values() for record in records]
    atomic_json_dump(
        {
            "stage": "eval",
            "skip_failed_judge": skip_failed_judge,
            "status_counts": status_counts(all_grade_records),
            "failed_users": failed_users,
            "skipped_records": skipped_records,
        },
        f"{results_dir}/{frame}_beam_eval_status.json",
        indent=2,
    )

    if failed_users:
        print(f"\n❌ EVALUATION FAILED: {len(failed_users)} users had errors")
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BEAM Evaluation Script (Nugget Score)")
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
    parser.add_argument(
        "--num_runs", type=int, default=1, help="Number of runs per rubric item for LLM-as-a-Judge evaluation."
    )
    parser.add_argument(
        "--skip-failed-judge",
        "--skip_failed_judge",
        type=parse_bool,
        default=False,
        help="Record failed judge calls and skip them instead of failing the step. Default: 0.",
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            args.lib,
            args.version,
            args.llm_workers,
            args.num_runs,
            skip_failed_judge=args.skip_failed_judge,
        )
    )
