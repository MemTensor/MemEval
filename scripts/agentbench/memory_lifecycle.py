from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentbench.config import write_json


class CommandMemoryLifecycle:
    """Command-backed lifecycle hooks for agent memory plugins.

    The runner owns the benchmark protocol; plugin-specific behavior lives in
    YAML as shell commands.  Templates use ``@name@`` tokens so embedded shell,
    JSON, and Python snippets can still use braces freely.
    """

    def __init__(
        self,
        *,
        config: dict[str, Any],
        project_dir: Path,
        run_dir: Path,
        run_id: str,
        version: str,
    ) -> None:
        self.config = config
        self.project_dir = project_dir
        self.run_dir = run_dir
        self.run_id = run_id
        self.version = version
        self.run_date = datetime.now().strftime("%F")
        self.plugin = str(config.get("plugin") or config.get("name") or "memory")
        self.backup_dir = self._path(config.get("backup_dir") or "~/memory_backup")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = run_dir / "memory_lifecycle.log"
        self.manifest_file = run_dir / "memory_lifecycle.json"
        self._events: list[dict[str, Any]] = []

    def validate(self, domain: str) -> None:
        self._run_stage("validate", domain)

    def set_mode(self, mode: str, domain: str) -> None:
        modes = self.config.get("modes") or {}
        if isinstance(modes, dict) and mode in modes:
            self._run_commands(f"set_mode:{mode}", modes[mode], domain)
            return
        self._run_stage(f"set_mode_{mode}", domain)

    def clear(self, domain: str) -> None:
        self._run_stage("clear", domain)

    def wait_settle(self, domain: str) -> None:
        if self._has_stage("wait_settle"):
            self._run_stage("wait_settle", domain)
            return
        seconds = int(self.config.get("settle_seconds", 0) or 0)
        if seconds > 0:
            self._record("wait_settle", domain, {"seconds": seconds, "method": "sleep"})
            time.sleep(seconds)

    def backup(self, domain: str) -> Path:
        backup_file = self.backup_file(domain)
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        self._run_stage("backup", domain, backup_file=backup_file)
        if not backup_file.exists() and self.config.get("require_backup_file", True):
            raise RuntimeError(f"Backup command did not create expected file: {backup_file}")
        self._record("backup", domain, {"backup_file": str(backup_file)})
        return backup_file

    def restore(self, domain: str, backup_file: str | os.PathLike[str]) -> None:
        backup = Path(backup_file).expanduser()
        if not backup.exists():
            raise FileNotFoundError(f"Backup file not found for {domain}: {backup}")
        self._run_stage("restore", domain, backup_file=backup)
        self._record("restore", domain, {"backup_file": str(backup)})

    def finalize(self, domain: str | None = None) -> None:
        self._run_stage("finalize", domain or "")
        write_json(self.manifest_file, {
            "plugin": self.plugin,
            "run_id": self.run_id,
            "version": self.version,
            "backup_dir": str(self.backup_dir),
            "events": self._events,
        })

    def backup_file(self, domain: str) -> Path:
        template = str(
            self.config.get("backup_file_template")
            or "@backup_dir@/@plugin@-@domain@-@run_date@-@run_id@.tar.gz"
        )
        rendered = self._render(template, domain, backup_file="")
        return self._path(rendered)

    def _has_stage(self, stage: str) -> bool:
        commands = self.config.get("commands") or {}
        return isinstance(commands, dict) and stage in commands and commands[stage] not in (None, "")

    def _run_stage(
        self,
        stage: str,
        domain: str,
        *,
        backup_file: str | os.PathLike[str] | None = None,
    ) -> None:
        commands = self.config.get("commands") or {}
        if not isinstance(commands, dict):
            return
        if stage not in commands:
            return
        self._run_commands(stage, commands[stage], domain, backup_file=backup_file)

    def _run_commands(
        self,
        stage: str,
        commands: Any,
        domain: str,
        *,
        backup_file: str | os.PathLike[str] | None = None,
    ) -> None:
        if commands in (None, "", []):
            return
        if isinstance(commands, str):
            command_list = [commands]
        elif isinstance(commands, list):
            command_list = [str(item) for item in commands if str(item).strip()]
        else:
            raise TypeError(f"Lifecycle stage {stage!r} must be a string or list of strings")

        backup = Path(backup_file).expanduser() if backup_file else self.backup_file(domain)
        for index, command in enumerate(command_list, start=1):
            rendered = self._render(command, domain, backup_file=str(backup))
            self._run_command(stage, domain, rendered, index=index)

    def _run_command(self, stage: str, domain: str, command: str, *, index: int) -> None:
        started = datetime.now(timezone.utc)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a", encoding="utf-8") as log:
            log.write(f"\n[{started.isoformat()}] stage={stage} domain={domain} command#{index}\n")
            log.write(command.rstrip() + "\n")
            log.flush()
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.project_dir,
                env=self._env(domain),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        ended = datetime.now(timezone.utc)
        event = {
            "stage": stage,
            "domain": domain,
            "command_index": index,
            "started_at": started.isoformat(),
            "ended_at": ended.isoformat(),
            "returncode": result.returncode,
        }
        self._events.append(event)
        if result.returncode != 0:
            raise RuntimeError(
                f"Memory lifecycle stage={stage} domain={domain} command#{index} "
                f"failed with returncode={result.returncode}. See {self.log_file}"
            )

    def _env(self, domain: str) -> dict[str, str]:
        env = os.environ.copy()
        env.update({
            "OMNIMEMEVAL_MEMORY_PLUGIN": self.plugin,
            "OMNIMEMEVAL_MEMORY_DOMAIN": domain,
            "OMNIMEMEVAL_RUN_ID": self.run_id,
            "OMNIMEMEVAL_VERSION": self.version,
            "OMNIMEMEVAL_BACKUP_DIR": str(self.backup_dir),
            "OMNIMEMEVAL_PROJECT_DIR": str(self.project_dir),
            "OMNIMEMEVAL_RUN_DIR": str(self.run_dir),
        })
        for key, value in (self.config.get("env") or {}).items():
            env[str(key)] = self._render(str(value), domain, backup_file="")
        return env

    def _render(
        self,
        value: str,
        domain: str,
        *,
        backup_file: str | os.PathLike[str],
    ) -> str:
        replacements = {
            "plugin": self.plugin,
            "domain": domain,
            "run_id": self.run_id,
            "run_date": self.run_date,
            "version": self.version,
            "backup_dir": str(getattr(self, "backup_dir", "")),
            "backup_file": str(backup_file),
            "project_dir": str(self.project_dir),
            "run_dir": str(self.run_dir),
            "home": str(Path.home()),
            "openclaw_home": os.environ.get("OPENCLAW_HOME", str(Path.home() / ".openclaw")),
        }
        rendered = value
        for key, replacement in replacements.items():
            rendered = rendered.replace(f"@{key}@", replacement)
        return rendered

    def _path(self, value: str | os.PathLike[str]) -> Path:
        text = str(value)
        rendered = self._render(text, "", backup_file="")
        path = Path(rendered).expanduser()
        if not path.is_absolute():
            path = self.project_dir / path
        return path

    def _record(self, stage: str, domain: str, data: dict[str, Any]) -> None:
        payload = {
            "stage": stage,
            "domain": domain,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        payload.update(data)
        self._events.append(payload)
