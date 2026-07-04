from __future__ import annotations

import math
import os
import shutil
import signal
import subprocess
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Sequence

from ..exceptions import TransportLaunchError
from ..exceptions import TransportNotAvailableError
from ..exceptions import TransportResumeError
from ..exceptions import TransportTimeoutError
from ..types import ParsedTurn

Runner = Callable[..., subprocess.CompletedProcess[str]]
BinaryResolver = Callable[[str], str | None]

RESERVED_HARNESS_OPTIONS = {
    "worktree_agent_slug",
    "worktree_path",
    "branch",
    "model",
    "auto",
    "timeout_seconds",
}


@dataclass(frozen=True, slots=True)
class ParsedOutput:
    session_id: str
    message_text: str
    parsed_turn: ParsedTurn
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TransportResult:
    session_id: str
    command: list[str]
    exit_code: int
    raw_stdout: str
    raw_stderr: str
    message_text: str
    parsed_turn: ParsedTurn
    usage: dict[str, Any] = field(default_factory=dict)


class Transport(ABC):
    harness: str

    def __init__(
        self,
        *,
        cwd: Path,
        model: str | None = None,
        harness_options: Mapping[str, Any] | None = None,
        runner: Runner = subprocess.run,
        binary_resolver: BinaryResolver = shutil.which,
    ) -> None:
        self.cwd = Path(cwd)
        self.model = model
        self.harness_options = dict(harness_options or {})
        self.timeout_seconds = self._parse_timeout_seconds(
            self.harness_options.get("timeout_seconds")
        )
        self._runner = runner
        self._binary_resolver = binary_resolver

    def healthcheck(self) -> bool:
        return self._binary_resolver(self.harness) is not None

    def launch(self, prompt: str, *, allowed_roles: set[str]) -> TransportResult:
        self._ensure_available()
        command, session_id = self._build_launch_command(prompt)
        process = self._run_command(command)
        if process.returncode != 0:
            raise TransportLaunchError(self._process_error_message(process))
        parsed = self.parse_output(
            process.stdout,
            allowed_roles=allowed_roles,
            session_id=session_id,
            is_resume=False,
        )
        return TransportResult(
            session_id=parsed.session_id,
            command=command,
            exit_code=process.returncode,
            raw_stdout=process.stdout,
            raw_stderr=process.stderr,
            message_text=parsed.message_text,
            parsed_turn=parsed.parsed_turn,
            usage=parsed.usage,
        )

    def resume(self, session_id: str, prompt: str, *, allowed_roles: set[str]) -> TransportResult:
        self._ensure_available()
        command = self._build_resume_command(session_id, prompt)
        process = self._run_command(command)
        if process.returncode != 0:
            raise TransportResumeError(self._process_error_message(process))
        parsed = self.parse_output(
            process.stdout,
            allowed_roles=allowed_roles,
            session_id=session_id,
            is_resume=True,
        )
        return TransportResult(
            session_id=parsed.session_id,
            command=command,
            exit_code=process.returncode,
            raw_stdout=process.stdout,
            raw_stderr=process.stderr,
            message_text=parsed.message_text,
            parsed_turn=parsed.parsed_turn,
            usage=parsed.usage,
        )

    @abstractmethod
    def parse_output(
        self,
        raw_stdout: str,
        *,
        allowed_roles: set[str],
        session_id: str | None,
        is_resume: bool,
    ) -> ParsedOutput:
        raise NotImplementedError

    @abstractmethod
    def _build_launch_command(self, prompt: str) -> tuple[list[str], str | None]:
        raise NotImplementedError

    @abstractmethod
    def _build_resume_command(self, session_id: str, prompt: str) -> list[str]:
        raise NotImplementedError

    def _ensure_available(self) -> None:
        if not self.healthcheck():
            raise TransportNotAvailableError(f"{self.harness} is not installed")

    def _run_command(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        kwargs: dict[str, Any] = {
            "cwd": self.cwd,
            "stdin": subprocess.DEVNULL,
            "text": True,
            "capture_output": True,
            "check": False,
        }
        if self.timeout_seconds is not None:
            kwargs["timeout"] = self.timeout_seconds
        try:
            if self._runner is subprocess.run:
                return self._run_subprocess_with_group(command)
            return self._runner(command, **kwargs)
        except subprocess.TimeoutExpired as exc:
            timeout_text = (
                f"{self.timeout_seconds:g}s" if self.timeout_seconds is not None else "unknown"
            )
            raise TransportTimeoutError(
                f"{self.harness} command timed out after {timeout_text}"
            ) from exc

    def _run_subprocess_with_group(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        process = subprocess.Popen(
            command,
            cwd=self.cwd,
            stdin=subprocess.DEVNULL,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            self._kill_process_group(process)
            process.communicate()
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

    @staticmethod
    def _kill_process_group(process: subprocess.Popen[str]) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except OSError:
            process.kill()

    def _process_error_message(self, process: subprocess.CompletedProcess[str]) -> str:
        parts: list[str] = []
        if process.stdout.strip():
            parts.append("stdout:\n" + process.stdout.strip()[:4000])
        if process.stderr.strip():
            parts.append("stderr:\n" + process.stderr.strip()[:4000])
        return "\n".join(parts) if parts else self.harness

    @staticmethod
    def _parse_timeout_seconds(value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            timeout = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("timeout_seconds must be a positive number") from exc
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_seconds must be a positive number")
        return timeout

    def _cli_option_args(self, extra_reserved: Sequence[str] = ()) -> list[str]:
        args: list[str] = []
        reserved = RESERVED_HARNESS_OPTIONS.union(extra_reserved)
        for key in sorted(self.harness_options):
            if key in reserved:
                continue
            value = self.harness_options[key]
            if isinstance(value, bool):
                if value:
                    args.append(key)
                continue
            if isinstance(value, (list, tuple)):
                for item in value:
                    args.extend([key, str(item)])
                continue
            args.extend([key, str(value)])
        return args
