import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from agentbench.agents.base import AgentAdapter
from agentbench.run_agent_eval import parse_args
from agentbench.session import SessionSpec


class _RetryAgent(AgentAdapter):
    def _build_cli_cmd(self, prompt: str, session: SessionSpec, timeout: int) -> list[str]:
        return ["true"]


def test_runner_defaults_to_two_retries(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_agent_eval.py"])

    args = parse_args()

    assert args.max_retries == 2


def test_retry_infra_error_even_when_agent_completed():
    result = {
        "agent_result": {
            "completion_status": "completed",
            "stderr": "HTTP 503 upstream connect error",
        },
        "verifier_result": {"reward": 0.0},
    }

    assert _RetryAgent().should_retry(result) == "infra_error"


def test_does_not_retry_completed_wrong_answer():
    result = {
        "agent_result": {"completion_status": "completed"},
        "verifier_result": {"reward": 0.0},
    }

    assert _RetryAgent().should_retry(result) is None


def test_retries_timeout_and_exception():
    assert _RetryAgent().should_retry({
        "agent_result": {"completion_status": "timeout"},
        "verifier_result": {"reward": 0.0},
    }) == "timeout"
    assert _RetryAgent().should_retry({
        "agent_result": {"completion_status": "completed"},
        "verifier_result": {"reward": 0.0},
        "exception_info": {"type": "RuntimeError", "message": "docker failed"},
    }) == "exception"
