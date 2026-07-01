"""PersonaMem v2 experiment report — delegates shared logic to report_base."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from utils.report_base import BenchmarkReport, render_latency, report_main


class PersonaMemReport(BenchmarkReport):
    benchmark_name = "PersonaMem v2"
    results_prefix = "pmv2"
    grades_suffix = "pm_grades"
    default_script = "run_pmv2_eval.sh"
    config_params = (
        "WORKERS", "LLM_WORKERS", "TOPK", "NUM_RUNS",
        "SAVE_MODEL_INPUT", "CLEAR", "WAIT_AFTER_INGEST", "STREAMING",
        "START_IDX", "END_IDX", "ALLOW_EMPTY_SEARCH", "SKIP_FAILED_SEARCH",
        "SKIP_FAILED_ANSWER", "SKIP_FAILED_STREAMING", "ALLOW_MISSING_DATA",
    )
    dingtalk_metric_name = "Accuracy"

    def render_scores(self, lines, grades):
        metrics = grades.get("metrics", {})
        category_scores = grades.get("category_scores", {})

        lines.append("## Evaluation Results")
        lines.append("")

        lines.append("### Overall Scores")
        lines.append("")
        lines.append("| Metric | Score |")
        lines.append("|--------|-------|")
        accuracy = metrics.get("accuracy", 0)
        accuracy_std = metrics.get("accuracy_std", 0)
        lines.append(f"| Precision (Accuracy) | {accuracy:.4f} ± {accuracy_std:.4f} |")
        lines.append(f"| Questions | {metrics.get('total_questions', 0)} |")
        lines.append(f"| Total Runs | {metrics.get('total_runs', 0)} |")
        lines.append("")

        if category_scores:
            lines.append("### Category Breakdown")
            lines.append("")
            lines.append("| Category | Precision | Questions |")
            lines.append("|----------|-----------|-----------|")
            for cat in sorted(category_scores.keys()):
                scores = category_scores[cat]
                acc = scores.get("accuracy", 0)
                acc_std = scores.get("accuracy_std", 0)
                total = scores.get("total_questions", 0)
                lines.append(f"| {cat} | {acc:.4f} ± {acc_std:.4f} | {total} |")
            lines.append("")

        pipeline_status = grades.get("pipeline_status", {})
        if pipeline_status:
            lines.append("### Pipeline Status")
            lines.append("")
            lines.append("| Stage | Failed Units | Skipped Records | Status Counts |")
            lines.append("|-------|--------------|-----------------|---------------|")
            for stage in ("search", "answer"):
                data = pipeline_status.get(stage)
                if not data:
                    continue
                counts = data.get("status_counts") or {}
                count_text = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "-"
                lines.append(
                    f"| {stage} | {data.get('failed_units', 0)} | "
                    f"{data.get('skipped_records', 0)} | {count_text} |"
                )
            lines.append("")

        render_latency(lines, metrics)

    def extract_dingtalk_data(self, grades):
        metrics = grades.get("metrics", {})
        cat_raw = grades.get("category_scores", {})
        cats = []
        for cat in sorted(cat_raw.keys()):
            scores = cat_raw[cat]
            cats.append({
                "name": cat,
                "score": scores.get("accuracy", 0),
                "count": scores.get("total_questions", 0),
            })
        return metrics.get("accuracy", 0), metrics.get("accuracy_std", 0), cats


if __name__ == "__main__":
    report_main(PersonaMemReport())
