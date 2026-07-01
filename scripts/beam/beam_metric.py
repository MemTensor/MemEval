import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from beam.beam_common import record_status
from client_factory import SUPPORTED_LIBS, DEFAULT_LIB
from utils.checkpoint import atomic_json_dump
from utils.duration_stats import add_duration_values


def load_pipeline_status(results_dir, lib):
    status_files = {
        "search": f"{lib}_beam_search_status.json",
        "answer": f"{lib}_beam_response_status.json",
        "eval": f"{lib}_beam_eval_status.json",
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
        failed = data.get("failed_users", []) or data.get("failed_records", [])
        status[stage] = {
            "status_counts": data.get("status_counts", {}),
            "failed_units": len(failed),
            "skipped_records": len(data.get("skipped_records", [])),
        }
    return status

def calculate_scores(data):
    overall_scores = []
    scale_scores = {}
    dimension_scores = {}
    difficulty_scores = {}

    duration_metrics = {
        "search_duration_ms": [],
    }
    scale_duration = {}
    dimension_duration = {}

    for _user_id, questions in data.items():
        for q in questions:
            if record_status(q) != "success":
                continue
            score = q.get("nugget_score", 0.0)
            scale = q.get("scale", "unknown")
            dimension = q.get("dimension", "unknown")
            difficulty = q.get("difficulty", "unknown")

            overall_scores.append(score)

            scale_scores.setdefault(scale, []).append(score)
            dimension_scores.setdefault(dimension, []).append(score)
            if difficulty:
                difficulty_scores.setdefault(str(difficulty), []).append(score)

            for metric in duration_metrics:
                v = q.get(metric, 0)
                if v:
                    duration_metrics[metric].append(v)
                    scale_duration.setdefault(scale, {}).setdefault(metric, []).append(v)
                    dimension_duration.setdefault(dimension, {}).setdefault(metric, []).append(v)

    overall_mean = float(np.mean(overall_scores)) if overall_scores else 0.0
    overall_std = float(np.std(overall_scores)) if overall_scores else 0.0

    per_scale = {}
    for scale, scores in sorted(scale_scores.items()):
        per_scale[scale] = {
            "nugget_score_mean": float(np.mean(scores)),
            "nugget_score_std": float(np.std(scores)),
            "count": len(scores),
            "duration": {},
        }
        for metric in duration_metrics:
            values = scale_duration.get(scale, {}).get(metric, [])
            if values:
                per_scale[scale]["duration"][metric] = float(np.mean(values))
                per_scale[scale]["duration"][f"{metric}_p50"] = float(np.percentile(values, 50))
                per_scale[scale]["duration"][f"{metric}_p95"] = float(np.percentile(values, 95))

    per_dimension = {}
    for dim, scores in sorted(dimension_scores.items()):
        per_dimension[dim] = {
            "nugget_score_mean": float(np.mean(scores)),
            "nugget_score_std": float(np.std(scores)),
            "count": len(scores),
            "duration": {},
        }
        for metric in duration_metrics:
            values = dimension_duration.get(dim, {}).get(metric, [])
            if values:
                per_dimension[dim]["duration"][metric] = float(np.mean(values))
                per_dimension[dim]["duration"][f"{metric}_p50"] = float(
                    np.percentile(values, 50)
                )
                per_dimension[dim]["duration"][f"{metric}_p95"] = float(
                    np.percentile(values, 95)
                )

    per_difficulty = {}
    for diff, scores in sorted(difficulty_scores.items()):
        per_difficulty[diff] = {
            "nugget_score_mean": float(np.mean(scores)),
            "nugget_score_std": float(np.std(scores)),
            "count": len(scores),
        }

    overall_duration = {}
    for metric, values in duration_metrics.items():
        if values:
            overall_duration[metric] = float(np.mean(values))
            overall_duration[f"{metric}_p50"] = float(np.percentile(values, 50))
            overall_duration[f"{metric}_p95"] = float(np.percentile(values, 95))

    return {
        "overall": {
            "nugget_score_mean": overall_mean,
            "nugget_score_std": overall_std,
            "total_questions": len(overall_scores),
            "duration": overall_duration,
        },
        "per_scale": per_scale,
        "per_dimension": per_dimension,
        "per_difficulty": per_difficulty,
    }


def save_to_excel(results, output_path):
    rows = []

    overall = results["overall"]
    rows.append({
        "group_type": "overall",
        "group_name": "overall",
        "nugget_score_mean": overall["nugget_score_mean"],
        "nugget_score_std": overall["nugget_score_std"],
        "count": overall["total_questions"],
        **{k: v for k, v in overall.get("duration", {}).items()},
    })

    for scale, metrics in results["per_scale"].items():
        rows.append({
            "group_type": "scale",
            "group_name": scale,
            "nugget_score_mean": metrics["nugget_score_mean"],
            "nugget_score_std": metrics["nugget_score_std"],
            "count": metrics["count"],
            **{k: v for k, v in metrics.get("duration", {}).items()},
        })

    for dim, metrics in results["per_dimension"].items():
        rows.append({
            "group_type": "dimension",
            "group_name": dim,
            "nugget_score_mean": metrics["nugget_score_mean"],
            "nugget_score_std": metrics["nugget_score_std"],
            "count": metrics["count"],
            **{k: v for k, v in metrics.get("duration", {}).items()},
        })

    for diff, metrics in results["per_difficulty"].items():
        rows.append({
            "group_type": "difficulty",
            "group_name": diff,
            "nugget_score_mean": metrics["nugget_score_mean"],
            "nugget_score_std": metrics["nugget_score_std"],
            "count": metrics["count"],
        })

    df = pd.DataFrame(rows)
    with pd.ExcelWriter(output_path) as writer:
        df.to_excel(writer, sheet_name="BEAM Metrics", index=False)

    print(f"Excel file saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="BEAM Metric Calculation")
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

    args = parser.parse_args()
    lib = args.lib
    version = args.version

    judged_path = f"results/beam/{lib}-{version}/{lib}_beam_judged.json"
    grade_path = f"results/beam/{lib}-{version}/{lib}_beam_grades.json"

    try:
        with open(judged_path) as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"❌ Input file not found: {judged_path}")
        raise SystemExit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {judged_path}: {e}")
        raise SystemExit(1)

    results = calculate_scores(data)

    # Load ingestion stats for add_duration_ms
    ingestion_stats_path = f"results/beam/{lib}-{version}/{lib}_beam_ingestion_stats.json"
    if os.path.exists(ingestion_stats_path):
        with open(ingestion_stats_path) as sf:
            ingestion_stats = json.load(sf)
        add_values = add_duration_values(ingestion_stats)
        if add_values:
            results["overall"]["duration"]["add_duration_ms"] = float(np.mean(add_values))
            results["overall"]["duration"]["add_duration_ms_p50"] = float(np.percentile(add_values, 50))
            results["overall"]["duration"]["add_duration_ms_p95"] = float(np.percentile(add_values, 95))

    results["pipeline_status"] = load_pipeline_status(
        f"results/beam/{lib}-{version}",
        lib,
    )

    atomic_json_dump(results, grade_path, indent=4)

    excel_path = f"results/beam/{lib}-{version}/{lib}_beam_results.xlsx"
    save_to_excel(results, excel_path)

    print("\n=== BEAM Metric Calculation Complete ===")
    print(
        f"Overall Nugget Score: {results['overall']['nugget_score_mean']:.4f} "
        f"± {results['overall']['nugget_score_std']:.4f}"
    )
    print(f"Total questions: {results['overall']['total_questions']}")

    print("\n=== Per-Scale Scores ===")
    for scale, metrics in results["per_scale"].items():
        print(
            f"  {scale}: {metrics['nugget_score_mean']:.4f} "
            f"± {metrics['nugget_score_std']:.4f} ({metrics['count']} questions)"
        )

    print("\n=== Per-Dimension Scores ===")
    for dim, metrics in results["per_dimension"].items():
        print(
            f"  {dim}: {metrics['nugget_score_mean']:.4f} "
            f"± {metrics['nugget_score_std']:.4f} ({metrics['count']} questions)"
        )

    if results["per_difficulty"]:
        print("\n=== Per-Difficulty Scores ===")
        for diff, metrics in results["per_difficulty"].items():
            print(
                f"  {diff}: {metrics['nugget_score_mean']:.4f} "
                f"± {metrics['nugget_score_std']:.4f} ({metrics['count']} questions)"
            )

    print("\n=== Duration Metrics ===")
    for metric in ["search_duration_ms", "add_duration_ms"]:
        dur = results["overall"].get("duration", {})
        if metric in dur:
            print(f"{metric} (avg): {dur[metric]:.2f} ms")
            print(f"{metric} (P50): {dur.get(f'{metric}_p50', 0):.2f} ms")
            print(f"{metric} (P95): {dur.get(f'{metric}_p95', 0):.2f} ms")

    print(f"\nResults saved to {grade_path}")
    print(f"Excel report saved to {excel_path}")


if __name__ == "__main__":
    main()
