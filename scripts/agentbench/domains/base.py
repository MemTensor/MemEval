from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class DomainAdapter(ABC):
    name = "base"

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def initialize(self, args) -> None:
        pass

    def finalize(self) -> None:
        pass

    @abstractmethod
    def load_tasks(self, args) -> list[dict]:
        ...

    def pre_task_trials(self, task: dict) -> None:
        pass

    def post_task_trials(self, task: dict) -> None:
        pass

    def setup(self, task: dict, agent_name: str, trial: int) -> dict:
        return {}

    @abstractmethod
    def build_prompt(self, task: dict, env_info: dict, phase: str) -> str:
        ...

    @abstractmethod
    def verify(
        self,
        task: dict,
        env_info: dict,
        trial_dir: Path,
        agent_result: dict | None = None,
    ) -> dict:
        ...

    def cleanup(self, task: dict, env_info: dict) -> None:
        pass

    def record_agent_outcome(
        self,
        task: dict,
        env_info: dict,
        trial_dir: Path,
        agent_result: dict,
        verifier_result: dict,
    ) -> None:
        pass

    def get_agent_timeout(self, task: dict, env_info: dict) -> int:
        return int(self.config.get("agent_timeout", 3600))

    def pass_threshold(self) -> float:
        return float(self.config.get("pass_threshold", 0.0))

    def aggregate_metrics(self, results: list[dict]) -> dict:
        return {}
