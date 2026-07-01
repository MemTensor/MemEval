import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from halumem.hm_common import (
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    STATUS_SUCCESS_EMPTY,
    build_search_entry,
    classify_search_status,
    error_payload,
    grade_complete,
    iter_questions,
    question_key,
    response_complete,
    search_allowed_statuses,
    skipped_response_record,
    status_counts,
    user_id_for,
    validate_user_search_results,
)


def _user():
    return {
        "uuid": "user-1",
        "sessions": [
            {
                "questions": [
                    {
                        "question": "What city do I live in?",
                        "answer": "Shanghai",
                        "question_type": "Basic Fact Recall",
                        "difficulty": "easy",
                        "evidence": "The user moved to Shanghai.",
                    }
                ]
            }
        ],
    }


class TestHmPipelineStatus(unittest.TestCase):
    def test_question_key_is_stable_per_session_and_question(self):
        user_id = user_id_for("v1", "user-1")
        self.assertEqual(user_id, "hm_exp_user_v1_user-1")
        self.assertEqual(
            question_key(user_id, 2, 3),
            "hm_exp_user_v1_user-1__s2__q3",
        )

        metas = list(iter_questions(_user(), "v1"))
        self.assertEqual(len(metas), 1)
        self.assertEqual(metas[0]["key"], "hm_exp_user_v1_user-1__s0__q0")

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
        meta = next(iter_questions(_user(), "v1"))
        entry = build_search_entry(
            meta,
            context="",
            duration_ms=0.0,
            status=STATUS_SUCCESS_EMPTY,
        )
        search_result = {meta["user_id"]: [entry]}

        ok, issues = validate_user_search_results(
            search_result,
            _user(),
            "v1",
            search_allowed_statuses(
                allow_empty_search=True,
                allow_skipped=False,
            ),
        )
        self.assertTrue(ok, issues)

        ok, issues = validate_user_search_results(
            search_result,
            _user(),
            "v1",
            search_allowed_statuses(
                allow_empty_search=False,
                allow_skipped=False,
            ),
        )
        self.assertFalse(ok)
        self.assertTrue(any("disallowed status" in issue for issue in issues))

    def test_skipped_response_is_complete_without_model_answer(self):
        meta = next(iter_questions(_user(), "v1"))
        entry = build_search_entry(
            meta,
            context="",
            duration_ms=0.0,
            status=STATUS_SKIPPED,
            error=error_payload("search", "rate limit"),
        )

        response = skipped_response_record(
            search_entry=entry,
            reason="search was explicitly skipped",
            error=entry["error"],
        )

        ok, issues = response_complete(response, entry)
        self.assertTrue(ok, issues)
        self.assertEqual(response["status"], STATUS_SKIPPED)

    def test_grade_resume_checks_expected_judge_runs(self):
        response = {
            "key": "hm_exp_user_v1_user-1__s0__q0",
            "question": "What city do I live in?",
            "status": STATUS_SUCCESS,
        }
        complete_grade = {
            "key": response["key"],
            "question": response["question"],
            "llm_judgments": {"judgment_1": True, "judgment_2": False},
        }
        ok, issues = grade_complete(complete_grade, response, num_runs=2)
        self.assertTrue(ok, issues)

        incomplete_grade = {
            "key": response["key"],
            "question": response["question"],
            "llm_judgments": {"judgment_1": True},
        }
        ok, issues = grade_complete(incomplete_grade, response, num_runs=2)
        self.assertFalse(ok)
        self.assertIn("missing judgment runs", issues[0])

        skipped_grade = {
            "key": response["key"],
            "question": response["question"],
            "status": STATUS_SKIPPED,
            "skip_reason": "eval_failed",
        }
        ok, issues = grade_complete(
            skipped_grade,
            response,
            num_runs=2,
            allow_skipped_grade=True,
        )
        self.assertTrue(ok, issues)

        ok, issues = grade_complete(skipped_grade, response, num_runs=2)
        self.assertFalse(ok)
        self.assertIn("existing grade is skipped", issues[0])

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
