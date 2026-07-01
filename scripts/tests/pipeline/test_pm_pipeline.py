import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from personamem_v2.pm_common import (
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    STATUS_SUCCESS_EMPTY,
    build_search_entry,
    classify_search_status,
    error_payload,
    memory_user_id_for,
    response_complete,
    result_key_for,
    search_allowed_statuses,
    skipped_response_record,
    status_counts,
    validate_single_search_result,
)


def _entry(status=STATUS_SUCCESS_EMPTY):
    return build_search_entry(
        result_key=result_key_for("v1", 7),
        persona_id=3,
        row_idx=7,
        question="Which option matches my preference?",
        category="preference",
        all_options="(a) tea\n(b) coffee",
        topic="drinks",
        golden_answer="(a)",
        context="",
        duration_ms=0.0,
        status=status,
    )


class TestPmPipelineStatus(unittest.TestCase):
    def test_stable_ids(self):
        self.assertEqual(memory_user_id_for("v1", 3), "pm_exper_user_3_v1")
        self.assertEqual(result_key_for("v1", 7), "pm_exper_user_7_v1")

    def test_search_status_distinguishes_success_empty_from_failure(self):
        self.assertEqual(classify_search_status("", None), STATUS_SUCCESS_EMPTY)
        self.assertEqual(classify_search_status("context", None), STATUS_SUCCESS)
        self.assertEqual(classify_search_status("", "direct answer"), STATUS_SUCCESS)
        self.assertEqual(
            classify_search_status("Conversation memories:\n\n", raw_context=""),
            STATUS_SUCCESS_EMPTY,
        )
        self.assertEqual(
            classify_search_status("Conversation memories:\n\nmemory", raw_context="memory"),
            STATUS_SUCCESS,
        )

    def test_search_result_validation_requires_allowed_status(self):
        entry = _entry()
        result = {entry["key"]: [entry]}

        ok, issues = validate_single_search_result(
            result,
            result_key=entry["key"],
            question=entry["question"],
            row_idx=entry["row_idx"],
            allowed_statuses=search_allowed_statuses(
                allow_empty_search=True,
                allow_skipped=False,
            ),
        )
        self.assertTrue(ok, issues)

        ok, issues = validate_single_search_result(
            result,
            result_key=entry["key"],
            question=entry["question"],
            row_idx=entry["row_idx"],
            allowed_statuses=search_allowed_statuses(
                allow_empty_search=False,
                allow_skipped=False,
            ),
        )
        self.assertFalse(ok)
        self.assertIn("disallowed status", issues[0])

    def test_skipped_response_is_complete_without_model_answer(self):
        entry = _entry(status=STATUS_SKIPPED)
        entry["error"] = error_payload("search", "rate limit")
        response = skipped_response_record(
            search_entry=entry,
            reason="search was explicitly skipped",
            error=entry["error"],
        )

        ok, issues = response_complete(response, entry, num_runs=2)
        self.assertTrue(ok, issues)
        self.assertEqual(response["status"], STATUS_SKIPPED)

    def test_response_resume_checks_expected_answer_runs(self):
        entry = _entry(status=STATUS_SUCCESS)
        response = {
            "key": entry["key"],
            "question": entry["question"],
            "golden_answer": entry["golden_answer"],
            "response_duration_ms": 1.0,
            "search_duration_ms": 2.0,
            "results": [{"is_correct": True}, {"is_correct": False}],
            "status": STATUS_SUCCESS,
        }
        ok, issues = response_complete(response, entry, num_runs=2)
        self.assertTrue(ok, issues)

        incomplete = {**response, "results": [{"is_correct": True}]}
        ok, issues = response_complete(incomplete, entry, num_runs=2)
        self.assertFalse(ok)
        self.assertIn("missing answer runs", issues[0])

    def test_status_counts_defaults_legacy_records_to_success(self):
        records = [
            {"status": STATUS_FAILED},
            {"status": STATUS_SKIPPED},
            {"question": "legacy success record"},
        ]

        self.assertEqual(
            status_counts(records),
            {
                STATUS_FAILED: 1,
                STATUS_SKIPPED: 1,
                STATUS_SUCCESS: 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
