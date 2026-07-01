import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from halumem.hm_streaming import (  # noqa: E402
    add_user_sessions,
    load_added_sessions,
    mark_streaming_failure_skipped,
    session_checkpoint_id,
    write_combined_results,
)
from utils.streaming import load_marker_set  # noqa: E402


def _user(uuid="user-1"):
    return {
        "uuid": uuid,
        "sessions": [
            {
                "end_time": "May 20, 2023, 02:21:00",
                "dialogue": [
                    {"role": "user", "content": "I moved to Shanghai."},
                    {"role": "assistant", "content": "Noted."},
                ],
                "questions": [
                    {
                        "question": "Where do I live?",
                        "answer": "Shanghai",
                        "question_type": "Basic Fact Recall",
                        "difficulty": "easy",
                        "evidence": "I moved to Shanghai.",
                    }
                ],
            },
            {
                "start_time": "May 21, 2023, 03:24:00",
                "dialogue": [
                    {"role": "user", "content": "I commute by train."},
                ],
            },
        ],
    }


class TestHmStreaming(unittest.TestCase):
    def test_add_user_sessions_skips_recorded_sessions(self):
        class FakeClient:
            def __init__(self):
                self.calls = []

            def add(self, messages, user_id, **kwargs):
                self.calls.append((messages, user_id, kwargs))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "added.txt"
            path.write_text(session_checkpoint_id("user-1", 0) + "\n")
            added = load_added_sessions(path)
            client = FakeClient()

            durations = add_user_sessions(
                _user(),
                frame="memos",
                version="v1",
                client=client,
                added_sessions_path=path,
                added_sessions=added,
            )

            self.assertEqual(len(client.calls), 1)
            messages, user_id, _ = client.calls[0]
            self.assertEqual(user_id, "hm_exp_user_v1_user-1")
            self.assertEqual(messages[0]["content"], "I commute by train.")
            self.assertEqual(len(durations), 1)
            self.assertEqual(
                load_marker_set(path),
                {
                    session_checkpoint_id("user-1", 0),
                    session_checkpoint_id("user-1", 1),
                },
            )

    def test_mark_streaming_failure_skipped_writes_hm_search_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                completed = set()
                skipped = mark_streaming_failure_skipped(
                    _user(),
                    "memos",
                    "v1",
                    completed,
                    RuntimeError("add failed"),
                )

                self.assertEqual(skipped["user_uuid"], "user-1")
                self.assertEqual(completed, {"user-1"})
                path = Path("results/halumem/memos-v1/tmp/memos_hm_search_results_user-1.json")
                with path.open() as f:
                    data = json.load(f)

                entries = data["hm_exp_user_v1_user-1"]
                self.assertEqual(len(entries), 1)
                self.assertEqual(entries[0]["status"], "skipped")
                self.assertEqual(entries[0]["key"], "hm_exp_user_v1_user-1__s0__q0")
                self.assertEqual(entries[0]["error"]["stage"], "streaming")
            finally:
                os.chdir(old_cwd)

    def test_write_combined_results_uses_only_completed_users(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                tmp_dir = Path("results/halumem/memos-v1/tmp")
                tmp_dir.mkdir(parents=True)
                with (tmp_dir / "memos_hm_search_results_user-1.json").open("w") as f:
                    json.dump({"hm_exp_user_v1_user-1": [{"key": "k1"}]}, f)
                with (tmp_dir / "memos_hm_search_results_user-2.json").open("w") as f:
                    json.dump({"hm_exp_user_v1_user-2": [{"key": "k2"}]}, f)

                write_combined_results("memos", "v1", {"user-2"})

                with Path("results/halumem/memos-v1/memos_hm_search_results.json").open() as f:
                    combined = json.load(f)

                self.assertEqual(list(combined), ["hm_exp_user_v1_user-2"])
                self.assertEqual(combined["hm_exp_user_v1_user-2"][0]["key"], "k2")
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
