"""Shared PersonaMem v2 pipeline bookkeeping helpers.

This module centralizes stable record keys, status labels, and checkpoint
validation without changing retrieval, answer extraction, or metric formulas.
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


def memory_user_id_for(version: str, persona_id: int | str) -> str:
    return f"pm_exper_user_{persona_id}_{version}"


def result_key_for(version: str, row_idx: int | str) -> str:
    return f"pm_exper_user_{row_idx}_{version}"


def build_search_entry(
    *,
    result_key: str,
    persona_id: int,
    row_idx: int,
    question: str,
    category: str,
    all_options: str,
    topic: str,
    golden_answer: str,
    context: str,
    duration_ms: float,
    status: str,
    reflect_answer: str | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry = {
        "key": result_key,
        "user_id": result_key,
        "question": question,
        "category": category,
        "persona_id": persona_id,
        "row_idx": row_idx,
        "all_options": all_options,
        "topic": topic,
        "golden_answer": golden_answer,
        "search_context": context,
        "search_duration_ms": duration_ms,
        "status": status,
    }
    if reflect_answer is not None:
        entry["reflect_answer"] = reflect_answer
    if error is not None:
        entry["error"] = error
    return entry


def get_single_search_entry(
    search_results: Mapping[str, Any],
    result_key: str,
) -> dict[str, Any] | None:
    entries = search_results.get(result_key)
    if not isinstance(entries, list) or len(entries) != 1:
        return None
    entry = entries[0]
    if not isinstance(entry, dict):
        return None
    return entry


def validate_single_search_result(
    search_results: Mapping[str, Any],
    *,
    result_key: str,
    question: str,
    row_idx: int,
    allowed_statuses: set[str],
    require_status: bool = True,
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    entry = get_single_search_entry(search_results, result_key)
    if entry is None:
        return False, ["expected exactly one search entry"]
    if entry.get("key") != result_key:
        issues.append("key mismatch")
    if entry.get("question") != question:
        issues.append("question mismatch")
    if entry.get("row_idx") != row_idx:
        issues.append("row_idx mismatch")
    if require_status and "status" not in entry:
        issues.append("missing status")
    if record_status(entry) not in allowed_statuses:
        issues.append(f"disallowed status: {record_status(entry)}")
    return not issues, issues


def response_complete(
    response: Mapping[str, Any] | None,
    search_entry: Mapping[str, Any],
    num_runs: int,
) -> tuple[bool, list[str]]:
    if not isinstance(response, Mapping):
        return False, ["missing response"]
    issues: list[str] = []
    for key in ("key", "question"):
        if response.get(key) != search_entry.get(key):
            issues.append(f"{key} mismatch")
    if record_status(response) == STATUS_SKIPPED:
        return not issues, issues
    results = response.get("results")
    if not isinstance(results, list):
        issues.append("missing results")
    elif len(results) < num_runs:
        issues.append(f"missing answer runs: {len(results)}/{num_runs}")
    for key in ("golden_answer", "response_duration_ms", "search_duration_ms"):
        if key not in response:
            issues.append(f"missing {key}")
    return not issues, issues


def skipped_response_record(
    *,
    search_entry: Mapping[str, Any],
    reason: str,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "key": search_entry.get("key"),
        "user_id": search_entry.get("user_id"),
        "category": search_entry.get("category"),
        "question": search_entry.get("question"),
        "results": [],
        "golden_answer": search_entry.get("golden_answer"),
        "all_options": search_entry.get("all_options", []),
        "response_duration_ms": 0.0,
        "search_duration_ms": search_entry.get("search_duration_ms", 0.0),
        "topic": search_entry.get("topic"),
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
    "build_search_entry",
    "classify_search_status",
    "error_payload",
    "get_single_search_entry",
    "memory_user_id_for",
    "record_status",
    "response_complete",
    "result_key_for",
    "search_allowed_statuses",
    "skipped_response_record",
    "status_counts",
    "validate_single_search_result",
]
