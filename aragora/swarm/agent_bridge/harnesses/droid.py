from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

from ..exceptions import MissingSessionIdentityError
from ..exceptions import TransportOutputParseError
from ..footer import extract_footer
from .base import BinaryResolver
from .base import ParsedOutput
from .base import Runner
from .base import Transport


class DroidTransport(Transport):
    harness = "droid"

    def __init__(
        self,
        *,
        cwd: Path,
        model: str | None = None,
        harness_options: dict[str, Any] | None = None,
        runner: Runner | None = None,
        binary_resolver: BinaryResolver | None = None,
        home: Path | None = None,
    ) -> None:
        super().__init__(
            cwd=cwd,
            model=model,
            harness_options=harness_options,
            runner=runner or subprocess.run,
            binary_resolver=binary_resolver or shutil.which,
        )
        self.home = home or Path.home()

    def _build_launch_command(self, prompt: str) -> tuple[list[str], str | None]:
        auto_mode = str(self.harness_options.get("auto", "low"))
        command = [self.harness, "exec", "--auto", auto_mode, "--output-format", "json"]
        if self.model:
            command.extend(["--model", self.model])
        command.extend(["--cwd", str(self.cwd)])
        command.extend(self._cli_option_args(extra_reserved=("auto",)))
        command.append(prompt)
        return command, None

    def _build_resume_command(self, session_id: str, prompt: str) -> list[str]:
        auto_mode = str(self.harness_options.get("auto", "low"))
        command = [
            self.harness,
            "exec",
            "--auto",
            auto_mode,
            "--output-format",
            "json",
            "-s",
            session_id,
        ]
        if self.model:
            command.extend(["--model", self.model])
        command.extend(["--cwd", str(self.cwd)])
        command.extend(self._cli_option_args(extra_reserved=("auto",)))
        command.append(prompt)
        return command

    def parse_output(
        self,
        raw_stdout: str,
        *,
        allowed_roles: set[str],
        session_id: str | None,
        is_resume: bool,
    ) -> ParsedOutput:
        del is_resume
        try:
            payload = json.loads(raw_stdout)
        except json.JSONDecodeError as exc:
            raise TransportOutputParseError("droid output is not valid json") from exc
        if not isinstance(payload, dict):
            raise TransportOutputParseError("droid output must be a json object")

        message_text = payload.get("result")
        if not isinstance(message_text, str) or not message_text:
            raise TransportOutputParseError("droid output missing result")

        discovered_session_id = payload.get("session_id")
        if not isinstance(discovered_session_id, str) or not discovered_session_id:
            discovered_session_id = session_id or self._discover_session_id_from_filesystem()
        if not isinstance(discovered_session_id, str) or not discovered_session_id:
            raise MissingSessionIdentityError("droid output missing session_id")

        usage = payload.get("usage")
        parsed_turn = extract_footer(message_text, allowed_roles=allowed_roles)
        return ParsedOutput(
            session_id=discovered_session_id,
            message_text=message_text,
            parsed_turn=parsed_turn,
            usage=dict(usage) if isinstance(usage, dict) else {},
        )

    def _discover_session_id_from_filesystem(self) -> str:
        session_dir = self.home / ".factory" / "sessions" / self._mangled_cwd()
        if not session_dir.exists():
            raise MissingSessionIdentityError("droid session directory missing")
        candidates = [
            path
            for path in session_dir.glob("*.settings.json")
            if path.name != "last.settings.json"
        ]
        if not candidates:
            raise MissingSessionIdentityError(
                "droid filesystem fallback found no concrete session files"
            )
        newest = max(candidates, key=lambda path: path.stat().st_mtime)
        return newest.name.removesuffix(".settings.json")

    def _mangled_cwd(self) -> str:
        return str(self.cwd.resolve()).replace("/", "-")
