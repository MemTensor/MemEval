import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from agentbench.memos_feedback import _normalize_trace_ref


def test_normalize_trace_ref_accepts_snake_and_camel_case_ids():
    assert _normalize_trace_ref({"episode_id": "ep1", "trace_id": "tr1"}) == {
        "episode_id": "ep1",
        "trace_id": "tr1",
    }
    assert _normalize_trace_ref({"episodeId": "ep2", "traceId": "tr2"}) == {
        "episode_id": "ep2",
        "trace_id": "tr2",
    }
