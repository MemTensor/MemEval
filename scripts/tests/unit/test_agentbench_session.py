import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from agentbench.agents.openclaw import OpenClawAgentAdapter
from agentbench.session import sanitize_session_id


def test_sanitize_session_id_removes_colons():
    assert sanitize_session_id("omni:test/task 1") == "omni-test-task-1"


def test_openclaw_gateway_compatible_session_id():
    agent = OpenClawAgentAdapter({"command": "openclaw"})
    session = agent.build_session_spec(
        phase="test",
        domain="reasoning",
        split="test",
        task={"name": "omni_1"},
        trial=1,
    )

    assert ":" not in session.cli_session_id
    assert session.openclaw_session_key == f"agent:main:explicit:{session.cli_session_id}"
    assert session.openclaw_gateway_session_id == (
        f"openclaw::main::agent:main:explicit:{session.cli_session_id}"
    )

