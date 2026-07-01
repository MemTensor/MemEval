from __future__ import annotations

import json
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from agentbench.session import SessionSpec, make_base_session_id
from agentbench.summary import classify_failure


class AgentAdapter(ABC):
    name = "base"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.run_dir: Path | None = None

    def prepare_run(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    def finalize_run(self) -> None:
        pass

    def build_session_spec(
        self,
        *,
        phase: str,
        domain: str,
        split: str,
        task: dict,
        trial: int,
    ) -> SessionSpec:
        task_name = str(task["name"])
        cli_session_id = make_base_session_id(
            phase=phase,
            domain=domain,
            split=split,
            task_name=task_name,
            trial=trial,
        )
        semantic_session_id = (
            f"omnimemeval:{phase}:{domain}:{split}:{task_name}:trial:{trial}"
        )
        source_ref = f"omnimemeval::{phase}::{domain}::{split}::{task_name}"
        return SessionSpec(
            cli_session_id=cli_session_id,
            semantic_session_id=semantic_session_id,
            source_ref=source_ref,
            metadata={
                "benchmark": "omnimemeval-agentbench",
                "phase": phase,
                "domain": domain,
                "split": split,
                "task": task_name,
                "trial": trial,
                "agent": self.name,
            },
        )

    def prepare_task(self, task: dict, env_info: dict, session: SessionSpec) -> None:
        pass

    def cleanup_task(self) -> None:
        pass

    @abstractmethod
    def _build_cli_cmd(self, prompt: str, session: SessionSpec, timeout: int) -> list[str]:
        ...

    def _get_subprocess_env(self, session: SessionSpec) -> dict[str, str] | None:
        return None

    def call(self, prompt: str, session: SessionSpec, timeout: int = 3600) -> dict:
        cmd = self._build_cli_cmd(prompt, session, timeout)
        start = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 60,
                env=self._get_subprocess_env(session),
            )
            elapsed = time.time() - start
            data = {
                "response": result.stdout,
                "completion_status": "completed" if result.returncode == 0 else "error",
                "elapsed_sec": round(elapsed, 1),
                "method": "cli",
                "returncode": result.returncode,
                "stderr": (result.stderr or "")[:2000] or None,
            }
            data.update(self._parse_extra(result))
            return data
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            data = {
                "response": "",
                "completion_status": "timeout",
                "elapsed_sec": round(elapsed, 1),
                "method": "cli",
                "error": f"subprocess timed out after {timeout + 60}s",
            }
            data.update(self._parse_timeout_extra(session))
            return data

    def _parse_extra(self, result: subprocess.CompletedProcess) -> dict:
        return {}

    def _parse_timeout_extra(self, session: SessionSpec) -> dict:
        return {}

    def _session_dir(self) -> Path:
        return Path.home() / f".{self.name}" / "sessions"

    def _session_file(self, session: SessionSpec) -> Path:
        return self._session_dir() / f"{session.cli_session_id}.jsonl"

    def collect_session(self, session: SessionSpec, trial_dir: Path) -> dict:
        session_jsonl = self._session_file(session)
        stats = {
            "turns": 0,
            "input": 0,
            "output": 0,
            "total": 0,
            "last_stop_reason": None,
        }
        if not session_jsonl.exists():
            return stats

        shutil.copy2(session_jsonl, trial_dir / "session.jsonl")
        try:
            with session_jsonl.open() as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._parse_session_entry(entry, stats)
        except OSError:
            pass
        return stats

    def _parse_session_entry(self, entry: dict, stats: dict) -> None:
        if entry.get("role") == "assistant":
            stats["turns"] += 1

    def should_retry(self, result: dict) -> str | None:
        reward = result.get("verifier_result", {}).get("reward", 0)
        if reward and reward > 0:
            return None
        if result.get("exception_info"):
            return "exception"
        failure_class = classify_failure(result, float(reward or 0))
        if failure_class == "infra_error":
            return "infra_error"
        status = result.get("agent_result", {}).get("completion_status")
        if status == "timeout":
            return "timeout"
        if status == "completed":
            return None
        if failure_class in {"no_patch", "patch_apply_failed", "resolved_false", "verifier_error", "failed"}:
            return None
        return "agent_error"
