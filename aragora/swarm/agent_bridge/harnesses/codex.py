from __future__ import annotations

import json
import re
from typing import Any

from ..exceptions import MissingSessionIdentityError
from ..exceptions import TransportOutputParseError
from ..footer import extract_footer
from .base import ParsedOutput
from .base import Transport

SESSION_ID_PATTERN = re.compile(r"^session id:\s*(?P<session_id>\S+)\s*$", re.IGNORECASE)


class CodexTransport(Transport):
    harness = "codex"

    def _build_launch_command(self, prompt: str) -> tuple[list[str], str | None]:
        command = [self.harness, "exec", "--json"]
        if self.model:
            command.extend(["--model", self.model])
        command.extend(self._cli_option_args())
        command.append(prompt)
        return command, None

    def _build_resume_command(self, session_id: str, prompt: str) -> list[str]:
        command = [self.harness, "exec", "resume", "--json"]
        if self.model:
            command.extend(["--model", self.model])
        command.extend(self._cli_option_args())
        command.extend([session_id, prompt])
        return command

    def parse_output(
        self,
        raw_stdout: str,
        *,
        allowed_roles: set[str],
        session_id: str | None,
        is_resume: bool,
    ) -> ParsedOutput:
        raw_lines = [line for line in raw_stdout.splitlines() if line.strip()]
        parsed_lines: list[dict[str, Any]] = []
        parse_failed = False
        for line in raw_lines:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                parse_failed = True
                break
            if not isinstance(payload, dict):
                raise TransportOutputParseError("codex jsonl row must be an object")
            parsed_lines.append(payload)

        if parse_failed or not parsed_lines:
            if is_resume:
                raise TransportOutputParseError("codex resume output must be valid jsonl")
            return self._parse_plaintext_fallback(raw_stdout, allowed_roles=allowed_roles)

        return self._parse_jsonl(
            parsed_lines,
            allowed_roles=allowed_roles,
            session_id=session_id,
            is_resume=is_resume,
        )

    def _parse_jsonl(
        self,
        payloads: list[dict[str, Any]],
        *,
        allowed_roles: set[str],
        session_id: str | None,
        is_resume: bool,
    ) -> ParsedOutput:
        discovered_session_id: str | None = None
        agent_messages: list[str] = []
        usage: dict[str, Any] = {}
        for payload in payloads:
            event_type = payload.get("type")
            if event_type == "thread.started":
                thread_id = payload.get("thread_id")
                if not isinstance(thread_id, str) or not thread_id:
                    raise MissingSessionIdentityError("codex thread.started omitted thread_id")
                if is_resume and session_id is not None and thread_id != session_id:
                    raise TransportOutputParseError(
                        "codex resume emitted mismatched thread.started"
                    )
                discovered_session_id = thread_id
                continue
            if event_type == "item.completed":
                item = payload.get("item")
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                    agent_messages.append(item["text"])
                continue
            if event_type == "turn.completed" and isinstance(payload.get("usage"), dict):
                usage = dict(payload["usage"])

        if not is_resume and discovered_session_id is None:
            raise MissingSessionIdentityError("codex start output missing thread.started")
        if is_resume and discovered_session_id is None:
            discovered_session_id = session_id
        if discovered_session_id is None:
            raise MissingSessionIdentityError("codex output missing session identity")
        if not agent_messages:
            raise TransportOutputParseError("codex output missing agent_message items")

        message_text = "\n\n".join(agent_messages)
        parsed_turn = extract_footer(message_text, allowed_roles=allowed_roles)
        return ParsedOutput(
            session_id=discovered_session_id,
            message_text=message_text,
            parsed_turn=parsed_turn,
            usage=usage,
        )

    def _parse_plaintext_fallback(
        self,
        raw_stdout: str,
        *,
        allowed_roles: set[str],
    ) -> ParsedOutput:
        lines = raw_stdout.splitlines()
        session_id: str | None = None
        body_lines: list[str] = []
        for line in lines:
            match = SESSION_ID_PATTERN.match(line.strip())
            if match:
                session_id = match.group("session_id")
                continue
            body_lines.append(line)
        if session_id is None:
            raise MissingSessionIdentityError("codex plaintext output missing session id header")
        message_text = "\n".join(body_lines).strip()
        if not message_text:
            raise TransportOutputParseError("codex plaintext output missing assistant message")
        parsed_turn = extract_footer(message_text, allowed_roles=allowed_roles)
        return ParsedOutput(
            session_id=session_id,
            message_text=message_text,
            parsed_turn=parsed_turn,
            usage={},
        )
