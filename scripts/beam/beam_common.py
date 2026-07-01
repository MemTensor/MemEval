"""Shared BEAM pipeline bookkeeping helpers.

The helpers here add stable record keys, status labels, and checkpoint
validation. They do not alter search calls, answer prompts, judge prompts, or
metric formulas.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from utils.pipeline_status import (
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    STATUS_SUCCESS_EMPTY,
    classify_search_status,
    error_payload,
    record_status,
    search_allowed_statuses,
    status_counts,
)


def user_id_for(version: str, conv_id: str | int) -> str:
    return f"beam_exp_user_{version}_{conv_id}"


def question_key(version: str, conv_id: str | int, question_idx: int) -> str:
    return f"{user_id_for(version, conv_id)}__q{question_idx}"


def golden_answer_for(question: Mapping[str, Any]) -> Any:
    return (
        question.get("answer")
        or question.get("ideal_response")
        or question.get("ideal_answer")
        or question.get("ideal_summary")
        or question.get("expected_compliance", "")
    )


def build_question_meta(
    conv: Mapping[str, Any],
    *,
    version: str,
    parse_probing_questions,
) -> list[dict[str, Any]]:
    conv_id = str(conv["conversation_id"])
    scale = conv.get("_scale", "unknown")
    metas = []
    q_idx = 0
    for dimension, q_list in parse_probing_questions(conv).items():
        for question in q_list:
            q = dict(question)
            metas.append({
                "key": question_key(version, conv_id, q_idx),
                "conv_id": conv_id,
                "user_id": user_id_for(version, conv_id),
                "question_idx": q_idx,
                "dimension": dimension,
                "scale": scale,
                "question": q,
            })
            q_idx += 1
    return metas


def build_search_entry(
    meta: Mapping[str, Any],
    *,
    context: str,
    duration_ms: float,
    status: str,
    reflect_answer: str | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    question = meta["question"]
    entry = {
        "key": meta["key"],
        "conv_id": meta["conv_id"],
        "question_idx": meta["question_idx"],
        "question": question.get("question", ""),
        "golden_answer": golden_answer_for(question),
        "rubric": question.get("rubric", ""),
        "difficulty": question.get("difficulty", ""),
        "dimension": meta["dimension"],
        "scale": meta["scale"],
        "search_context": context,
        "search_duration_ms": duration_ms,
        "status": status,
    }
    if reflect_answer is not None:
        entry["reflect_answer"] = reflect_answer
    if error is not None:
        entry["error"] = error
    return entry


def get_search_entries(
    search_results: Mapping[str, Any],
    user_id: str,
) -> list[dict[str, Any]]:
    entries = search_results.get(user_id)
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def search_entry_map(
    search_results: Mapping[str, Any],
) -> dict[str, tuple[str, dict[str, Any]]]:
    mapped: dict[str, tuple[str, dict[str, Any]]] = {}
    for user_id, entries in search_results.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and entry.get("key"):
                mapped[str(entry["key"])] = (str(user_id), entry)
    return mapped


def validate_search_results(
    search_results: Mapping[str, Any],
    expected: list[Mapping[str, Any]],
    *,
    user_id: str,
    allowed_statuses: set[str],
    require_status: bool = True,
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    entries = get_search_entries(search_results, user_id)
    by_key = {str(entry.get("key")): entry for entry in entries if entry.get("key")}
    if len(by_key) != len(expected):
        issues.append(f"entry count mismatch: {len(by_key)}/{len(expected)}")

    for meta in expected:
        key = str(meta["key"])
        entry = by_key.get(key)
        if entry is None:
            issues.append(f"missing {key}")
            continue
        if entry.get("question") != meta["question"].get("question", ""):
            issues.append(f"question mismatch: {key}")
        if entry.get("question_idx") != meta["question_idx"]:
            issues.append(f"question_idx mismatch: {key}")
        if require_status and "status" not in entry:
            issues.append(f"missing status: {key}")
        if record_status(entry) not in allowed_statuses:
            issues.append(f"disallowed status {record_status(entry)}: {key}")
    return not issues, issues


def response_complete(
    response: Mapping[str, Any] | None,
    search_entry: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    if not isinstance(response, Mapping):
        return False, ["missing response"]
    issues: list[str] = []
    for key in ("key", "question"):
        if response.get(key) != search_entry.get(key):
            issues.append(f"{key} mismatch")
    if record_status(response) == STATUS_SKIPPED:
        return not issues, issues
    for key in ("answer", "golden_answer", "response_duration_ms", "search_duration_ms"):
        if key not in response:
            issues.append(f"missing {key}")
    return not issues, issues


def grade_complete(
    grade: Mapping[str, Any] | None,
    response: Mapping[str, Any],
    num_runs: int,
    *,
    allow_skipped_grade: bool = False,
) -> tuple[bool, list[str]]:
    if not isinstance(grade, Mapping):
        return False, ["missing grade"]
    issues: list[str] = []
    for key in ("key", "question"):
        if grade.get(key) != response.get(key):
            issues.append(f"{key} mismatch")
    if record_status(response) == STATUS_SKIPPED:
        return not issues, issues
    if record_status(grade) == STATUS_SKIPPED:
        if allow_skipped_grade:
            return not issues, issues
        issues.append("existing grade is skipped")
        return False, issues
    if "nugget_score" not in grade:
        issues.append("missing nugget_score")

    if grade.get("scoring_method") == "kendall_tau_b":
        if len(grade.get("run_scores", [])) < num_runs:
            issues.append("missing judge runs")
    else:
        item_scores = grade.get("rubric_item_scores")
        if not isinstance(item_scores, list) or not item_scores:
            issues.append("missing rubric item scores")
        else:
            incomplete = [
                item for item in item_scores
                if len(item.get("run_scores", [])) < num_runs
            ]
            if incomplete:
                issues.append(f"missing judge runs: {len(incomplete)}")
    return not issues, issues


def skipped_response_record(
    *,
    search_entry: Mapping[str, Any],
    reason: str,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "key": search_entry.get("key"),
        "conv_id": search_entry.get("conv_id"),
        "question_idx": search_entry.get("question_idx"),
        "question": search_entry.get("question"),
        "answer": "",
        "golden_answer": search_entry.get("golden_answer"),
        "rubric": search_entry.get("rubric", ""),
        "dimension": search_entry.get("dimension", ""),
        "scale": search_entry.get("scale", ""),
        "difficulty": search_entry.get("difficulty", ""),
        "response_duration_ms": 0.0,
        "search_duration_ms": search_entry.get("search_duration_ms", 0.0),
        "status": STATUS_SKIPPED,
        "skip_reason": reason,
    }
    if error is not None:
        record["error"] = error
    return record


__all__ = [
    "STATUS_FAILED",
    "STATUS_SKIPPED",
    "STATUS_SUCCESS",
    "STATUS_SUCCESS_EMPTY",
    "build_question_meta",
    "build_search_entry",
    "classify_search_status",
    "error_payload",
    "get_search_entries",
    "grade_complete",
    "record_status",
    "response_complete",
    "search_allowed_statuses",
    "search_entry_map",
    "skipped_response_record",
    "status_counts",
    "user_id_for",
    "validate_search_results",
]
