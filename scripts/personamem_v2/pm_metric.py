import argparse
import sys
import os
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from personamem_v2.pm_common import record_status
from client_factory import SUPPORTED_LIBS, DEFAULT_LIB
from utils.checkpoint import atomic_json_dump
from utils.duration_stats import add_duration_values


def load_pipeline_status(results_dir, lib):
    status_files = {
        "search": f"{lib}_pm_search_status.json",
        "answer": f"{lib}_pm_response_status.json",
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


def convert_numpy_types(obj):
    if isinstance(obj, np.number):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj


def save_to_excel(results, output_path):
    """Save results to Excel file"""
    combined_data = []

    # Add overall statistics row
    overall_row = {
        "category": "overall",
        "accuracy": results["metrics"]["accuracy"],
        "accuracy_std": results["metrics"]["accuracy_std"],
        "total_questions": results["metrics"]["total_questions"],
        "total_runs": results["metrics"]["total_runs"],
    }

    # Add response duration metrics
    for metric, value in results["metrics"]["response_duration"].items():
        overall_row[f"response_{metric}"] = value

    # Add search duration metrics (if exists)
    if "search_duration" in results["metrics"] and results["metrics"]["search_duration"]:
        for metric, value in results["metrics"]["search_duration"].items():
            overall_row[f"search_{metric}"] = value

    combined_data.append(overall_row)

    # Add category statistics rows
    for category, scores in results["category_scores"].items():
        category_row = {
            "category": category,
            "accuracy": scores["accuracy"],
            "accuracy_std": scores["accuracy_std"],
            "total_questions": scores["total_questions"],
            "total_runs": scores["total_runs"],
        }

        # Add response duration metrics
        for metric, value in scores["response_duration"].items():
            category_row[f"response_{metric}"] = value

        # Add search duration metrics (if exists)
        if scores.get("search_duration"):
            for metric, value in scores["search_duration"].items():
                category_row[f"search_{metric}"] = value

        combined_data.append(category_row)

    # Save to Excel
    df = pd.DataFrame(combined_data)
    df.to_excel(output_path, sheet_name="PersonaMem_Metrics", index=False)
    print(f"📊 Excel file saved to: {output_path}")


def calculate_scores(data, grade_path, output_path):
    """Calculate PersonaMem evaluation metrics"""

    # Initialize statistics variables
    category_scores = {}
    user_metrics = {}

    # Overall metrics - collect accuracy for each run
    all_response_durations = []
    all_search_durations = []
    total_questions = 0

    # For calculating accuracy across multiple runs
    num_runs = None  # Will be determined from first user's data
    run_accuracies = []  # List to store accuracy for each run across all users

    # Category-wise statistics
    category_response_durations = {}
    category_search_durations = {}
    category_run_accuracies = {}  # Store accuracy for each run by category

    print(f"📋 Processing response data for {len(data)} users...")

    # First pass: determine number of runs and initialize run accuracy arrays
    for _user_id, user_data in data.items():
        if record_status(user_data) != "success":
            continue
        results = user_data.get("results", [])
        if not results:
            continue

        if num_runs is None:
            num_runs = len(results)
            run_accuracies = [[] for _ in range(num_runs)]
            print(f"📊 Detected {num_runs} runs per user")
        break

    if num_runs is None:
        print("⚠️  No successful PersonaMem responses found; writing zero-score metrics")
        lib_key = os.path.basename(grade_path).split("_pm_")[0]
        results = {
            "metrics": {
                "accuracy": 0.0,
                "accuracy_std": 0.0,
                "total_questions": 0,
                "total_runs": 0,
                "response_duration": {},
                "search_duration": {},
                "add_duration": {},
            },
            "category_scores": {},
            "user_scores": {},
            "pipeline_status": load_pipeline_status(os.path.dirname(grade_path), lib_key),
        }
        atomic_json_dump(results, grade_path, indent=4, ensure_ascii=False)
        save_to_excel(results, output_path)
        print_summary(results)
        return results

    # Iterate through all user data
    for user_id, user_data in data.items():
        if record_status(user_data) != "success":
            continue
        results = user_data.get("results", [])
        if not results:
            print(f"⚠️  Skipping user {user_id}: no results data")
            continue
        if len(results) != num_runs:
            raise ValueError(
                f"Inconsistent PersonaMem run count for {user_id}: "
                f"expected {num_runs}, got {len(results)}. Re-run "
                "pm_responses.py with a consistent --num-runs value or clear "
                "mixed response checkpoints before metric calculation."
            )

        category = user_data.get("category", "unknown")

        # Initialize category if not exists
        if category not in category_scores:
            category_scores[category] = {
                "category_name": category,
                "total_questions": 0,
                "total_runs": 0,
                "accuracy": 0.0,
                "accuracy_std": 0.0,
                "response_duration": {},
                "search_duration": {},
            }
            category_response_durations[category] = []
            category_search_durations[category] = []
            category_run_accuracies[category] = [[] for _ in range(num_runs)]

        # Process each run for this user
        user_response_durations = []
        for run_idx, result in enumerate(results):
            is_correct = result.get("is_correct", False)

            # Collect accuracy for each run (1 if correct, 0 if not)
            if run_idx < num_runs:
                run_accuracies[run_idx].append(1.0 if is_correct else 0.0)
                category_run_accuracies[category][run_idx].append(1.0 if is_correct else 0.0)

            # Collect response duration
            response_duration = result.get("response_duration_ms", 0)
            if response_duration > 0:
                user_response_durations.append(response_duration)
                all_response_durations.append(response_duration)
                category_response_durations[category].append(response_duration)

        # Get search duration (usually same for all runs)
        search_duration = user_data.get("search_duration_ms", 0)
        if search_duration > 0:
            all_search_durations.append(search_duration)
            category_search_durations[category].append(search_duration)

        # Calculate user-level accuracy (average across runs)
        user_correct_count = sum(1 for result in results if result.get("is_correct", False))
        user_accuracy = user_correct_count / len(results) if results else 0.0

        # Store user-level metrics
        user_metrics[user_id] = {
            "user_id": user_id,
            "category": category,
            "question": user_data.get("question", ""),
            "accuracy": user_accuracy,
            "total_runs": len(results),
            "correct_runs": user_correct_count,
            "avg_response_duration_ms": np.mean(user_response_durations)
            if user_response_durations
            else 0.0,
            "search_duration_ms": search_duration,
            "golden_answer": user_data.get("golden_answer", ""),
            "topic": user_data.get("topic", ""),
        }

        # Count statistics
        total_questions += 1
        category_scores[category]["total_questions"] += 1
        category_scores[category]["total_runs"] += len(results)

    # Calculate overall accuracy and std across runs
    overall_run_accuracies = [np.mean(run_acc) for run_acc in run_accuracies if run_acc]
    overall_accuracy = np.mean(overall_run_accuracies) if overall_run_accuracies else 0.0
    overall_accuracy_std = (
        np.std(overall_run_accuracies) if len(overall_run_accuracies) > 1 else 0.0
    )

    # Calculate response duration statistics
    response_duration_stats = {}
    if all_response_durations:
        response_duration_stats = {
            "mean": np.mean(all_response_durations),
            "median": np.median(all_response_durations),
            "p50": np.percentile(all_response_durations, 50),
            "p95": np.percentile(all_response_durations, 95),
            "std": np.std(all_response_durations),
            "min": np.min(all_response_durations),
            "max": np.max(all_response_durations),
        }

    # Calculate search duration statistics
    search_duration_stats = {}
    if all_search_durations:
        search_duration_stats = {
            "mean": np.mean(all_search_durations),
            "median": np.median(all_search_durations),
            "p50": np.percentile(all_search_durations, 50),
            "p95": np.percentile(all_search_durations, 95),
            "std": np.std(all_search_durations),
            "min": np.min(all_search_durations),
            "max": np.max(all_search_durations),
        }

    # Calculate category-wise metrics
    for category in category_scores:
        # Calculate accuracy mean and std across runs for this category
        cat_run_accuracies = [
            np.mean(run_acc) for run_acc in category_run_accuracies[category] if run_acc
        ]
        category_scores[category]["accuracy"] = (
            np.mean(cat_run_accuracies) if cat_run_accuracies else 0.0
        )
        category_scores[category]["accuracy_std"] = (
            np.std(cat_run_accuracies) if len(cat_run_accuracies) > 1 else 0.0
        )

        # Response duration statistics for this category
        if category_response_durations[category]:
            durations = category_response_durations[category]
            category_scores[category]["response_duration"] = {
                "mean": np.mean(durations),
                "median": np.median(durations),
                "p50": np.percentile(durations, 50),
                "p95": np.percentile(durations, 95),
                "std": np.std(durations),
                "min": np.min(durations),
                "max": np.max(durations),
            }
        else:
            category_scores[category]["response_duration"] = {
                "mean": 0.0,
                "median": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        # Search duration statistics for this category
        if category_search_durations[category]:
            durations = category_search_durations[category]
            category_scores[category]["search_duration"] = {
                "mean": np.mean(durations),
                "median": np.median(durations),
                "p50": np.percentile(durations, 50),
                "p95": np.percentile(durations, 95),
                "std": np.std(durations),
                "min": np.min(durations),
                "max": np.max(durations),
            }
        else:
            category_scores[category]["search_duration"] = {
                "mean": 0.0,
                "median": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

    # Load ingestion stats for add_duration
    add_duration_stats = {}
    ingestion_stats_path = os.path.join(os.path.dirname(grade_path),
        f"{os.path.basename(grade_path).split('_pm_')[0]}_pm_ingestion_stats.json")
    if os.path.exists(ingestion_stats_path):
        with open(ingestion_stats_path) as sf:
            ingestion_stats = json.load(sf)
        add_values = add_duration_values(ingestion_stats)
        if add_values:
            add_duration_stats = {
                "mean": float(np.mean(add_values)),
                "median": float(np.median(add_values)),
                "p50": float(np.percentile(add_values, 50)),
                "p95": float(np.percentile(add_values, 95)),
                "std": float(np.std(add_values)),
                "min": float(np.min(add_values)),
                "max": float(np.max(add_values)),
            }

    lib_key = os.path.basename(grade_path).split("_pm_")[0]

    # Build final results
    results = {
        "metrics": {
            "accuracy": overall_accuracy,
            "accuracy_std": overall_accuracy_std,
            "total_questions": total_questions,
            "total_runs": total_questions * num_runs if num_runs else 0,
            "response_duration": response_duration_stats,
            "search_duration": search_duration_stats,
            "add_duration": add_duration_stats,
        },
        "category_scores": category_scores,
        "user_scores": user_metrics,
        "pipeline_status": load_pipeline_status(os.path.dirname(grade_path), lib_key),
    }

    # Convert numpy types and save results to JSON file
    results = convert_numpy_types(results)
    atomic_json_dump(results, grade_path, indent=4, ensure_ascii=False)

    # Save to Excel
    save_to_excel(results, output_path)

    # Print summary
    print_summary(results)

    return results


def print_summary(results):
    """Print evaluation results summary"""
    print("\n" + "=" * 80)
    print("📊 PERSONAMEM EVALUATION SUMMARY".center(80))
    print("=" * 80)

    # Overall accuracy
    accuracy = results["metrics"]["accuracy"]
    accuracy_std = results["metrics"]["accuracy_std"]
    total_questions = results["metrics"]["total_questions"]
    total_runs = results["metrics"]["total_runs"]

    print(f"🎯 Overall Accuracy: {accuracy:.4f} ± {accuracy_std:.4f}")
    print(f"📋 Total Questions: {total_questions}")
    print(f"🔄 Total Runs: {total_runs}")

    print("-" * 80)

    # Search duration statistics
    if results["metrics"]["search_duration"]:
        sd = results["metrics"]["search_duration"]
        print("Search Duration Stats (ms):")
        print(f"   Mean: {sd['mean']:.2f}")
        print(f"   P50: {sd['p50']:.2f}")
        print(f"   P95: {sd['p95']:.2f}")

    # Add duration statistics
    if results["metrics"].get("add_duration"):
        ad = results["metrics"]["add_duration"]
        print("Add Duration Stats (ms):")
        print(f"   Mean: {ad['mean']:.2f}")
        print(f"   P50: {ad['p50']:.2f}")
        print(f"   P95: {ad['p95']:.2f}")

    print("-" * 80)

    # Category-wise accuracy
    print("📂 Category-wise Accuracy:")
    for category, scores in results["category_scores"].items():
        acc = scores["accuracy"]
        acc_std = scores["accuracy_std"]
        total_cat = scores["total_questions"]
        total_runs_cat = scores["total_runs"]
        print(
            f"   {category:<35}: {acc:.4f} ± {acc_std:.4f} ({total_cat} questions, {total_runs_cat} runs)"
        )

    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PersonaMem evaluation metrics calculation script")
    parser.add_argument(
        "--lib",
        type=str,
        choices=SUPPORTED_LIBS,
        default=DEFAULT_LIB,
        help="Memory library to evaluate",
    )
    parser.add_argument(
        "--version", type=str, default="default", help="Evaluation framework version"
    )

    args = parser.parse_args()
    lib, version = args.lib, args.version

    # Define file paths
    responses_path = f"results/pmv2/{lib}-{version}/{lib}_pm_responses.json"
    grade_path = f"results/pmv2/{lib}-{version}/{lib}_pm_grades.json"
    output_path = f"results/pmv2/{lib}-{version}/{lib}_pm_results.xlsx"

    print(f"📂 Loading response data from: {responses_path}")

    try:
        with open(responses_path, encoding="utf-8") as file:
            data = json.load(file)

        # Calculate metrics
        results = calculate_scores(data, grade_path, output_path)

        print(f"📁 Results saved to: {grade_path}")
        print(f"📊 Excel report saved to: {output_path}")

    except FileNotFoundError:
        print(f"❌ Error: File not found {responses_path}")
        print("Please make sure to run pm_responses.py first to generate response data")
        raise SystemExit(1)
    except Exception as e:
        print(f"❌ Error occurred during processing: {e}")
        raise SystemExit(1)
