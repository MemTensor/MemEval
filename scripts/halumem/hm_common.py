"""Shared HaluMem pipeline bookkeeping helpers.

This module centralizes status labels, stable question keys, result-shape
construction, and checkpoint validation. It does not change search calls,
answer prompts, judge prompts, or metric formulas.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

STATUS_SUCCESS = "success"
STATUS_SUCCESS_EMPTY = "success_empty"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

VARIANT_FILES = {
    "medium": "data/halumem/HaluMem-Medium.jsonl",
    "long": "data/halumem/HaluMem-Long.jsonl",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_halumem_data(variant: str) -> list[dict[str, Any]]:
    path = VARIANT_FILES.get(variant)
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"HaluMem data file not found: {path}")
    users = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                users.append(json.loads(line))
    return users


def user_id_for(version: str, user_uuid: str) -> str:
    return f"hm_exp_user_{version}_{user_uuid}"


def question_key(user_id: str, session_idx: int, question_idx: int) -> str:
    return f"{user_id}__s{session_idx}__q{question_idx}"


def iter_questions(
    user_obj: Mapping[str, Any],
    version: str,
) -> Iterable[dict[str, Any]]:
    user_uuid = str(user_obj["uuid"])
    user_id = user_id_for(version, user_uuid)
    for session_idx, session in enumerate(user_obj["sessions"]):
        questions = session.get("questions", [])
        for question_idx, question in enumerate(questions):
            yield {
                "user_uuid": user_uuid,
                "user_id": user_id,
                "key": question_key(user_id, session_idx, question_idx),
                "session_idx": session_idx,
                "question_idx": question_idx,
                "question": question,
            }


def record_status(record: Mapping[str, Any] | None) -> str:
    if not record:
        return STATUS_FAILED
    return str(record.get("status") or STATUS_SUCCESS)


def error_payload(stage: str, exc: BaseException | str) -> dict[str, str]:
    if isinstance(exc, BaseException):
        err_type = type(exc).__name__
        message = str(exc)
    else:
        err_type = "PipelineError"
        message = str(exc)
    return {
        "stage": stage,
        "type": err_type,
        "message": message,
        "timestamp": utc_now_iso(),
    }


def classify_search_status(
    context: str,
    reflect_answer: str | None = None,
    *,
    raw_context: str | None = None,
) -> str:
    if reflect_answer is not None and str(reflect_answer).strip():
        return STATUS_SUCCESS
    search_payload = context if raw_context is None else raw_context
    if str(search_payload or "").strip():
        return STATUS_SUCCESS
    return STATUS_SUCCESS_EMPTY


def search_allowed_statuses(
    *,
    allow_empty_search: bool,
    allow_skipped: bool,
) -> set[str]:
    statuses = {STATUS_SUCCESS}
    if allow_empty_search:
        statuses.add(STATUS_SUCCESS_EMPTY)
    if allow_skipped:
        statuses.add(STATUS_SKIPPED)
    return statuses


def status_counts(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        status = record_status(record)
        counts[status] = counts.get(status, 0) + 1
    return counts


def build_search_entry(
    question_meta: Mapping[str, Any],
    *,
    context: str,
    duration_ms: float,
    status: str,
    reflect_answer: str | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    question = question_meta["question"]
    entry = {
        "key": question_meta["key"],
        "user_id": question_meta["user_id"],
        "question": question["question"],
        "golden_answer": question["answer"],
        "category": question.get("question_type", "unknown"),
        "difficulty": question.get("difficulty", "unknown"),
        "evidence": question.get("evidence", ""),
        "session_idx": question_meta["session_idx"],
        "question_idx": question_meta["question_idx"],
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


def validate_user_search_results(
    search_results: Mapping[str, Any],
    user_obj: Mapping[str, Any],
    version: str,
    allowed_statuses: set[str],
    *,
    require_status: bool = True,
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    expected = list(iter_questions(user_obj, version))
    user_id = user_id_for(version, str(user_obj["uuid"]))
    entries = get_search_entries(search_results, user_id)
    by_key = {str(entry.get("key")): entry for entry in entries if entry.get("key")}

    if len(by_key) != len(expected):
        issues.append(f"entry count mismatch: {len(by_key)}/{len(expected)}")
    for meta in expected:
        entry = by_key.get(str(meta["key"]))
        if entry is None:
            issues.append(f"missing {meta['key']}")
            continue
        if entry.get("question") != meta["question"].get("question"):
            issues.append(f"question mismatch: {meta['key']}")
        if entry.get("session_idx") != meta["session_idx"]:
            issues.append(f"session_idx mismatch: {meta['key']}")
        if entry.get("question_idx") != meta["question_idx"]:
            issues.append(f"question_idx mismatch: {meta['key']}")
        if require_status and "status" not in entry:
            issues.append(f"missing status: {meta['key']}")
        if record_status(entry) not in allowed_statuses:
            issues.append(f"disallowed status {record_status(entry)}: {meta['key']}")
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
    for key in (
        "answer",
        "golden_answer",
        "response_duration_ms",
        "search_duration_ms",
    ):
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

    judgments = grade.get("llm_judgments")
    if not isinstance(judgments, Mapping):
        issues.append("missing judgments")
    else:
        missing = [
            f"judgment_{idx}"
            for idx in range(1, num_runs + 1)
            if f"judgment_{idx}" not in judgments
        ]
        if missing:
            issues.append(f"missing judgment runs: {len(missing)}")
    return not issues, issues


def skipped_response_record(
    *,
    search_entry: Mapping[str, Any],
    reason: str,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "key": search_entry.get("key"),
        "user_id": search_entry.get("user_id"),
        "category": search_entry.get("category"),
        "difficulty": search_entry.get("difficulty"),
        "question": search_entry.get("question"),
        "answer": "",
        "golden_answer": search_entry.get("golden_answer"),
        "evidence": search_entry.get("evidence", ""),
        "response_duration_ms": 0.0,
        "search_duration_ms": search_entry.get("search_duration_ms", 0.0),
        "session_idx": search_entry.get("session_idx"),
        "question_idx": search_entry.get("question_idx"),
        "status": STATUS_SKIPPED,
        "skip_reason": reason,
        "error": error,
    }
