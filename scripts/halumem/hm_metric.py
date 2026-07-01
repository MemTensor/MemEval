import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from client_factory import SUPPORTED_LIBS, DEFAULT_LIB
from utils.checkpoint import atomic_json_dump
from utils.duration_stats import add_duration_values
from halumem.hm_common import record_status

QUESTION_TYPES = [
    "Memory Boundary",
    "Memory Conflict",
    "Basic Fact Recall",
    "Generalization & Application",
    "Multi-hop Inference",
    "Dynamic Update",
]


def save_to_excel(results, output_path):
    combined_data = []

    overall_row = {"category": "overall"}
    overall_row["llm_judge_score"] = results["metrics"]["llm_judge_score"]
    overall_row["llm_judge_std"] = results["metrics"]["llm_judge_std"]
    for metric, value in results["metrics"]["lexical"].items():
        overall_row[metric] = value if value is not None else "-"
    for metric, value in results["metrics"]["semantic"].items():
        overall_row[metric] = value if value is not None else "-"
    overall_row["context_tokens"] = results["metrics"]["context_tokens"]
    for metric, value in results["metrics"]["duration"].items():
        overall_row[metric] = value
    combined_data.append(overall_row)

    for _, scores in results["category_scores"].items():
        category_row = {"category": scores["category_name"]}
        category_row["llm_judge_score"] = scores["llm_judge_score"]
        category_row["llm_judge_std"] = scores["llm_judge_std"]
        for metric, value in scores["lexical"].items():
            category_row[metric] = value if value is not None else "-"
        for metric, value in scores["semantic"].items():
            category_row[metric] = value if value is not None else "-"
        category_row["context_tokens"] = scores["context_tokens"]
        for metric, value in scores["duration"].items():
            category_row[metric] = value
        combined_data.append(category_row)

    if "difficulty_scores" in results:
        for _, scores in results["difficulty_scores"].items():
            diff_row = {"category": f"[difficulty] {scores['difficulty_name']}"}
            diff_row["llm_judge_score"] = scores["llm_judge_score"]
            diff_row["llm_judge_std"] = scores["llm_judge_std"]
            for metric, value in scores["lexical"].items():
                diff_row[metric] = value if value is not None else "-"
            for metric, value in scores["semantic"].items():
                diff_row[metric] = value if value is not None else "-"
            diff_row["context_tokens"] = scores["context_tokens"]
            for metric, value in scores["duration"].items():
                diff_row[metric] = value
            combined_data.append(diff_row)

    pd.DataFrame(combined_data).to_excel(output_path, sheet_name="Metrics", index=False)
    print(f"Excel file saved to: {output_path}")


def _aggregate_group(items, all_judgment_keys, metrics_template):
    metrics = {
        "lexical": {m: [] for m in metrics_template["lexical"]},
        "semantic": {m: [] for m in metrics_template["semantic"]},
        "context_tokens": [],
        "duration": {m: [] for m in metrics_template["duration"]},
    }
    judgment_run_scores = {k: [] for k in all_judgment_keys}
    total = 0

    for q in items:
        total += 1
        if "llm_judgments" in q:
            for k, v in q["llm_judgments"].items():
                score = 1 if v else 0
                judgment_run_scores[k].append(score)
        nlp = q.get("nlp_metrics", {})
        for m in metrics["lexical"]:
            v = nlp.get("lexical", {}).get(m)
            if v is not None:
                metrics["lexical"][m].append(v)
        for m in metrics["semantic"]:
            v = nlp.get("semantic", {}).get(m)
            if v is not None:
                metrics["semantic"][m].append(v)
        ct = nlp.get("context_tokens")
        if ct is not None:
            metrics["context_tokens"].append(ct)
        for m in metrics["duration"]:
            v = q.get(m)
            if v is not None:
                metrics["duration"][m].append(v)

    judgment_avgs = [np.mean(s) for s in judgment_run_scores.values() if s]
    llm_judge_score = np.mean(judgment_avgs) if judgment_avgs else 0.0
    llm_judge_std = np.std(judgment_avgs) if len(judgment_avgs) > 1 else 0.0

    result = {
        "total": total,
        "llm_judge_score": llm_judge_score,
        "llm_judge_std": llm_judge_std,
        "lexical": {},
        "semantic": {},
        "context_tokens": np.mean(metrics["context_tokens"]) if metrics["context_tokens"] else 0.0,
        "duration": {},
    }
    for group in ["lexical", "semantic"]:
        for m in metrics[group]:
            vals = metrics[group][m]
            result[group][m] = float(np.mean(vals)) if vals else None
    for m in list(metrics["duration"].keys()):
        vals = metrics["duration"][m]
        if vals:
            result["duration"][m] = np.mean(vals)
            result["duration"][f"{m}_p50"] = np.percentile(vals, 50)
            result["duration"][f"{m}_p95"] = np.percentile(vals, 95)
        else:
            result["duration"][m] = 0.0
            result["duration"][f"{m}_p50"] = 0.0
            result["duration"][f"{m}_p95"] = 0.0

    return result


def convert_numpy_types(obj):
    if isinstance(obj, np.number):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj


def load_pipeline_status(results_dir, lib):
    status_files = {
        "search": f"{lib}_hm_search_status.json",
        "answer": f"{lib}_hm_response_status.json",
        "eval": f"{lib}_hm_eval_status.json",
    }
    status = {}
    for stage, filename in status_files.items():
        path = os.path.join(results_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as exc:
            status[stage] = {"load_error": str(exc)}
            continue
        status[stage] = {
            "status_counts": data.get("status_counts", {}),
            "failed_units": len(data.get("failed_users", [])),
            "failed_users": len(data.get("failed_users", [])),
            "failed_records": len(data.get("failed_keys", [])),
            "failed_keys": len(data.get("failed_keys", [])),
            "skipped_records": len(data.get("skipped_records", [])),
        }
    return status


def calculate_scores(data, grade_path, output_path):
    all_items = [
        item for item in data.values()
        if record_status(item) == "success"
    ]

    all_judgment_keys = set()
    for q in all_items:
        if "llm_judgments" in q:
            all_judgment_keys.update(q["llm_judgments"].keys())

    metrics_template = {
        "lexical": [
            "f1", "rouge1_f", "rouge2_f", "rougeL_f",
            "bleu1", "bleu2", "bleu3", "bleu4", "meteor",
        ],
        "semantic": ["bert_f1", "similarity"],
        "duration": ["search_duration_ms"],
    }

    overall = _aggregate_group(all_items, all_judgment_keys, metrics_template)
    overall_metrics = {
        "llm_judge_score": overall["llm_judge_score"],
        "llm_judge_std": overall["llm_judge_std"],
        "lexical": overall["lexical"],
        "semantic": overall["semantic"],
        "context_tokens": overall["context_tokens"],
        "duration": overall["duration"],
    }

    by_category = {qt: [] for qt in QUESTION_TYPES}
    for q in all_items:
        cat = q.get("category", "unknown")
        by_category.setdefault(cat, []).append(q)

    category_scores = {}
    for cat, items in by_category.items():
        agg = _aggregate_group(items, all_judgment_keys, metrics_template)
        agg["category_name"] = cat
        category_scores[cat] = agg

    by_difficulty = {}
    for q in all_items:
        diff = q.get("difficulty", "unknown")
        by_difficulty.setdefault(diff, []).append(q)

    difficulty_scores = {}
    for diff, items in by_difficulty.items():
        agg = _aggregate_group(items, all_judgment_keys, metrics_template)
        agg["difficulty_name"] = str(diff)
        difficulty_scores[diff] = agg

    by_user = {}
    for q in all_items:
        uid = q.get("user_id", "unknown")
        by_user.setdefault(uid, []).append(q)

    user_scores = {}
    for uid, items in by_user.items():
        agg = _aggregate_group(items, all_judgment_keys, metrics_template)
        user_scores[uid] = agg

    results = {
        "metrics": overall_metrics,
        "category_scores": category_scores,
        "difficulty_scores": difficulty_scores,
        "user_scores": user_scores,
    }

    results = convert_numpy_types(results)

    # Load ingestion stats for add_duration_ms
    ingestion_stats_path = os.path.join(os.path.dirname(grade_path),
        f"{os.path.basename(grade_path).split('_hm_')[0]}_hm_ingestion_stats.json")
    if os.path.exists(ingestion_stats_path):
        with open(ingestion_stats_path) as sf:
            ingestion_stats = json.load(sf)
        add_values = add_duration_values(ingestion_stats)
        if add_values:
            results["metrics"]["duration"]["add_duration_ms"] = float(np.mean(add_values))
            results["metrics"]["duration"]["add_duration_ms_p50"] = float(
                np.percentile(add_values, 50)
            )
            results["metrics"]["duration"]["add_duration_ms_p95"] = float(
                np.percentile(add_values, 95)
            )

    results_dir = os.path.dirname(grade_path)
    lib_name = os.path.basename(grade_path).split("_hm_")[0]
    results["pipeline_status"] = load_pipeline_status(results_dir, lib_name)
    results = convert_numpy_types(results)
    atomic_json_dump(results, grade_path, indent=4)
    save_to_excel(results, output_path)

    print("\n=== Metric Calculation Complete ===")
    total = sum(s["total"] for s in results["category_scores"].values())
    print(
        f"LLM-as-a-Judge score: {results['metrics']['llm_judge_score']:.4f}"
        f" +/- {results['metrics']['llm_judge_std']:.4f}"
    )
    print(f"Total questions evaluated: {total}")
    print("\n=== Duration Metrics ===")
    for m in ["search_duration_ms", "add_duration_ms"]:
        dur = results["metrics"].get("duration", {})
        if m in dur:
            print(f"{m} (avg): {dur[m]:.2f} ms")
            print(f"{m} (P50): {dur.get(f'{m}_p50', 0):.2f} ms")
            print(f"{m} (P95): {dur.get(f'{m}_p95', 0):.2f} ms")
    print(f"\nResults written to {grade_path}")
    print(f"Excel report saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("HaluMem Metric Calculation Script")
    parser.add_argument(
        "--lib",
        type=str,
        choices=SUPPORTED_LIBS,
        default=DEFAULT_LIB,
    )
    parser.add_argument(
        "--version", type=str, default="default", help="Version of the evaluation framework."
    )
    args = parser.parse_args()
    lib, version = args.lib, args.version

    judged_path = f"results/halumem/{lib}-{version}/{lib}_hm_judged.json"
    grade_path = f"results/halumem/{lib}-{version}/{lib}_hm_grades.json"
    output_path = f"results/halumem/{lib}-{version}/{lib}_hm_results.xlsx"

    try:
        with open(judged_path) as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"❌ Input file not found: {judged_path}")
        raise SystemExit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {judged_path}: {e}")
        raise SystemExit(1)
    calculate_scores(data, grade_path, output_path)
