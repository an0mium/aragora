from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Callable
from typing import Any

from ..exceptions import TransportLaunchError
from ..exceptions import TransportOutputParseError
from ..footer import extract_footer
from .base import BinaryResolver
from .base import ParsedOutput
from .base import Runner
from .base import Transport


class ClaudeTransport(Transport):
    harness = "claude"

    def __init__(
        self,
        *,
        cwd: Path,
        model: str | None = None,
        harness_options: dict[str, Any] | None = None,
        runner: Runner | None = None,
        binary_resolver: BinaryResolver | None = None,
        uuid_factory: Callable[[], uuid.UUID] = uuid.uuid4,
    ) -> None:
        super().__init__(
            cwd=cwd,
            model=model,
            harness_options=harness_options,
            runner=runner or subprocess.run,
            binary_resolver=binary_resolver or shutil.which,
        )
        self._uuid_factory = uuid_factory

    def _build_launch_command(self, prompt: str) -> tuple[list[str], str | None]:
        session_id = str(self._uuid_factory())
        parsed_uuid = uuid.UUID(session_id)
        if parsed_uuid.version != 4:
            raise TransportLaunchError("claude session id must be UUIDv4")
        command = [self.harness, "-p", "--session-id", session_id]
        if self.model:
            command.extend(["--model", self.model])
        command.extend(self._cli_option_args())
        command.append(prompt)
        return command, session_id

    def _build_resume_command(self, session_id: str, prompt: str) -> list[str]:
        command = [self.harness, "-p", "--resume", session_id]
        if self.model:
            command.extend(["--model", self.model])
        command.extend(self._cli_option_args())
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
        if session_id is None:
            raise TransportOutputParseError("claude session identity is required")
        message_text = raw_stdout.strip()
        if not message_text:
            raise TransportOutputParseError("claude emitted empty stdout")
        parsed_turn = extract_footer(message_text, allowed_roles=allowed_roles)
        return ParsedOutput(
            session_id=session_id,
            message_text=message_text,
            parsed_turn=parsed_turn,
            usage={},
        )
