"""BEAM experiment report — delegates shared logic to report_base."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from utils.report_base import BenchmarkReport, render_latency, report_main


class BEAMReport(BenchmarkReport):
    benchmark_name = "BEAM"
    results_prefix = "beam"
    grades_suffix = "beam_grades"
    default_script = "run_beam_eval.sh"
    config_params = (
        "WORKERS", "LLM_WORKERS", "TOPK", "SCALE", "NUM_RUNS",
        "SAVE_MODEL_INPUT", "CLEAR", "WAIT_AFTER_INGEST", "STREAMING",
        "START_IDX", "END_IDX", "ALLOW_EMPTY_SEARCH", "SKIP_FAILED_SEARCH",
        "SKIP_FAILED_ANSWER", "SKIP_FAILED_JUDGE", "SKIP_FAILED_STREAMING",
    )
    dingtalk_metric_name = "Nugget Score"

    def render_scores(self, lines, grades):
        overall = grades.get("overall", {})
        per_scale = grades.get("per_scale", {})
        per_dimension = grades.get("per_dimension", {})
        per_difficulty = grades.get("per_difficulty", {})

        lines.append("## Evaluation Results")
        lines.append("")

        lines.append("### Overall Scores")
        lines.append("")
        lines.append("| Metric | Score |")
        lines.append("|--------|-------|")
        lines.append(f"| Nugget Score | {overall.get('nugget_score_mean', 0):.4f} ± {overall.get('nugget_score_std', 0):.4f} |")
        lines.append(f"| Questions | {overall.get('total_questions', 0)} |")
        lines.append("")

        if per_scale:
            lines.append("### By Conversation Scale")
            lines.append("")
            lines.append("| Scale | Nugget Score | Questions |")
            lines.append("|-------|-------------|-----------|")
            for scale in sorted(per_scale.keys()):
                m = per_scale[scale]
                lines.append(f"| {scale} | {m['nugget_score_mean']:.4f} ± {m['nugget_score_std']:.4f} | {m['count']} |")
            lines.append("")

        if per_dimension:
            lines.append("### By Question Dimension")
            lines.append("")
            lines.append("| Dimension | Nugget Score | Questions |")
            lines.append("|-----------|-------------|-----------|")
            for dim in sorted(per_dimension.keys()):
                m = per_dimension[dim]
                lines.append(f"| {dim} | {m['nugget_score_mean']:.4f} ± {m['nugget_score_std']:.4f} | {m['count']} |")
            lines.append("")

        if per_difficulty:
            lines.append("### By Difficulty")
            lines.append("")
            lines.append("| Difficulty | Nugget Score | Questions |")
            lines.append("|------------|-------------|-----------|")
            for diff in sorted(per_difficulty.keys()):
                m = per_difficulty[diff]
                lines.append(f"| {diff} | {m['nugget_score_mean']:.4f} ± {m['nugget_score_std']:.4f} | {m['count']} |")
            lines.append("")

        pipeline_status = grades.get("pipeline_status", {})
        if pipeline_status:
            lines.append("### Pipeline Status")
            lines.append("")
            lines.append("| Stage | Failed Units | Skipped Records | Status Counts |")
            lines.append("|-------|--------------|-----------------|---------------|")
            for stage in ("search", "answer", "eval"):
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

        render_latency(lines, overall.get("duration", {}))

    def extract_dingtalk_data(self, grades):
        overall = grades.get("overall", {})
        per_scale = grades.get("per_scale", {})
        cats = []
        for scale in sorted(per_scale.keys()):
            m = per_scale[scale]
            cats.append({
                "name": scale,
                "score": m.get("nugget_score_mean", 0),
                "count": m.get("count", 0),
            })
        return overall.get("nugget_score_mean", 0), overall.get("nugget_score_std", 0), cats


if __name__ == "__main__":
    report_main(BEAMReport())
