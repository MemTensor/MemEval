import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from beam.beam_common import (
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    STATUS_SUCCESS_EMPTY,
    build_question_meta,
    build_search_entry,
    classify_search_status,
    error_payload,
    grade_complete,
    response_complete,
    search_allowed_statuses,
    skipped_response_record,
    status_counts,
    user_id_for,
    validate_search_results,
)
from beam.beam_search import parse_probing_questions


def _conv():
    return {
        "conversation_id": "conv-1",
        "_scale": "100k",
        "probing_questions": {
            "fact_recall": [
                {
                    "question": "What framework did I use?",
                    "answer": "OmniMemEval",
                    "rubric": ["Mentions OmniMemEval"],
                    "difficulty": "easy",
                }
            ]
        },
    }


class TestBeamPipelineStatus(unittest.TestCase):
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
        meta = build_question_meta(
            _conv(),
            version="v1",
            parse_probing_questions=parse_probing_questions,
        )[0]
        entry = build_search_entry(
            meta,
            context="",
            duration_ms=0.0,
            status=STATUS_SUCCESS_EMPTY,
        )
        result = {meta["user_id"]: [entry]}

        ok, issues = validate_search_results(
            result,
            [meta],
            user_id=meta["user_id"],
            allowed_statuses=search_allowed_statuses(
                allow_empty_search=True,
                allow_skipped=False,
            ),
        )
        self.assertTrue(ok, issues)

        ok, issues = validate_search_results(
            result,
            [meta],
            user_id=meta["user_id"],
            allowed_statuses=search_allowed_statuses(
                allow_empty_search=False,
                allow_skipped=False,
            ),
        )
        self.assertFalse(ok)
        self.assertTrue(any("disallowed status" in issue for issue in issues))

    def test_skipped_response_is_complete_without_model_answer(self):
        meta = build_question_meta(
            _conv(),
            version="v1",
            parse_probing_questions=parse_probing_questions,
        )[0]
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
        user_id = user_id_for("v1", "conv-1")
        response = {
            "key": f"{user_id}__q0",
            "question": "What framework did I use?",
            "status": STATUS_SUCCESS,
        }
        complete_grade = {
            "key": response["key"],
            "question": response["question"],
            "nugget_score": 1.0,
            "scoring_method": "per_rubric_item",
            "rubric_item_scores": [
                {"run_scores": [1.0, 0.5]},
            ],
        }
        ok, issues = grade_complete(complete_grade, response, num_runs=2)
        self.assertTrue(ok, issues)

        incomplete_grade = {
            **complete_grade,
            "rubric_item_scores": [{"run_scores": [1.0]}],
        }
        ok, issues = grade_complete(incomplete_grade, response, num_runs=2)
        self.assertFalse(ok)
        self.assertTrue(any("missing judge runs" in issue for issue in issues))

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
