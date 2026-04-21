from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import BridgeFooter
from .types import BridgeRun
from .types import ParsedTurn
from .types import SCHEMA_VERSION
from .types import SessionRegistry
from .types import TurnRecord


def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


class BridgeStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def runs_root(self) -> Path:
        path = self.root / ".aragora" / "agent_bridge" / "runs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run_dir(self, run_id: str) -> Path:
        path = self.runs_root() / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "run.json"

    def sessions_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "sessions.json"

    def events_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "events.jsonl"

    def turns_dir(self, run_id: str) -> Path:
        path = self.run_dir(run_id) / "turns"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_run(self, run: BridgeRun) -> None:
        self._write_atomic(self.run_path(run.run_id), _json_text(run.to_dict()))

    def load_run(self, run_id: str) -> BridgeRun:
        payload = json.loads(self.run_path(run_id).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("run payload must be a mapping")
        return BridgeRun.from_dict(payload)

    def save_sessions(self, run_id: str, registry: SessionRegistry) -> None:
        self._write_atomic(self.sessions_path(run_id), _json_text(registry.to_dict()))

    def load_sessions(self, run_id: str) -> SessionRegistry:
        payload = json.loads(self.sessions_path(run_id).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("sessions payload must be a mapping")
        return SessionRegistry.from_dict(payload)

    def append_event(self, run_id: str, record: TurnRecord) -> bool:
        path = self.events_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict) and payload.get("event_id") == record.event_id:
                    return False
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        return True

    def load_events(self, run_id: str) -> list[TurnRecord]:
        path = self.events_path(run_id)
        if not path.exists():
            return []
        records: list[TurnRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(TurnRecord.from_dict(payload))
        return records

    def write_turn_transcript(
        self,
        run_id: str,
        *,
        turn_index: int,
        role: str,
        harness: str,
        model: str,
        session_id: str | None,
        started_at: str,
        completed_at: str,
        exit_code: int,
        prompt: str,
        raw_stdout: str,
        raw_stderr: str,
        parsed_turn: ParsedTurn,
        repair_attempts: list[dict[str, Any]] | None = None,
    ) -> Path:
        path = self.turns_dir(run_id) / f"{turn_index:03d}-{harness}-{role}.md"
        content = self._render_turn_transcript(
            run_id=run_id,
            turn_index=turn_index,
            role=role,
            harness=harness,
            model=model,
            session_id=session_id,
            started_at=started_at,
            completed_at=completed_at,
            exit_code=exit_code,
            prompt=prompt,
            raw_stdout=raw_stdout,
            raw_stderr=raw_stderr,
            parsed_turn=parsed_turn,
            repair_attempts=repair_attempts or [],
        )
        self._write_atomic(path, content)
        return path

    def relative_path(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.root))

    def _render_turn_transcript(
        self,
        *,
        run_id: str,
        turn_index: int,
        role: str,
        harness: str,
        model: str,
        session_id: str | None,
        started_at: str,
        completed_at: str,
        exit_code: int,
        prompt: str,
        raw_stdout: str,
        raw_stderr: str,
        parsed_turn: ParsedTurn,
        repair_attempts: list[dict[str, Any]],
    ) -> str:
        front_matter = [
            "---",
            f"schema_version: {SCHEMA_VERSION}",
            f"run_id: {run_id}",
            f"turn_index: {turn_index}",
            f"role: {role}",
            f"harness: {harness}",
            f"model: {model}",
            f"session_id: {session_id or ''}",
            f"started_at: {started_at}",
            f"completed_at: {completed_at}",
            f"exit_code: {exit_code}",
            f"parse_status: {parsed_turn.parse_status}",
            f"repair_attempts: {len(repair_attempts)}",
            "---",
            "",
        ]
        sections = [
            "## Prompt",
            prompt.rstrip(),
            "",
            "## Raw Stdout",
            self._fenced(raw_stdout),
            "## Raw Stderr",
            self._fenced(raw_stderr),
            "## Parsed Message",
            parsed_turn.body_without_footer.rstrip(),
            "",
            "## Footer",
            self._render_footer(parsed_turn.footer, parsed_turn.parse_errors),
        ]

        for index, attempt in enumerate(repair_attempts, start=1):
            repair_turn = attempt["parsed_turn"]
            if not isinstance(repair_turn, ParsedTurn):
                raise TypeError("repair attempt parsed_turn must be a ParsedTurn")
            sections.extend(
                [
                    "",
                    f"## Repair Attempt {index}",
                    "",
                    "### Prompt",
                    str(attempt["prompt"]).rstrip(),
                    "",
                    "### Raw Stdout",
                    self._fenced(str(attempt["raw_stdout"])),
                    "### Raw Stderr",
                    self._fenced(str(attempt["raw_stderr"])),
                    "### Parsed Message",
                    repair_turn.body_without_footer.rstrip(),
                    "",
                    "### Footer",
                    self._render_footer(repair_turn.footer, repair_turn.parse_errors),
                ]
            )

        return "\n".join(front_matter + sections).rstrip() + "\n"

    def _render_footer(
        self,
        footer: BridgeFooter | None,
        parse_errors: list[str],
    ) -> str:
        if footer is None:
            errors = ", ".join(parse_errors) if parse_errors else "missing"
            return f"<none>\nparse_errors: [{errors}]"
        return "\n".join(
            [
                f"summary: {footer.summary}",
                f"next_actor: {footer.next_actor if footer.next_actor is not None else 'null'}",
                f"needs_human: {'true' if footer.needs_human else 'false'}",
                f"done: {'true' if footer.done else 'false'}",
                f"artifacts: {json.dumps(footer.artifacts)}",
                f"tests_run: {json.dumps(footer.tests_run)}",
            ]
        )

    def _fenced(self, text: str) -> str:
        return "```text\n" + text.rstrip("\n") + "\n```"

    def _write_atomic(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(content, encoding="utf-8")
        self._replace_file(temp_path, path)

    def _replace_file(self, source: Path, target: Path) -> None:
        source.replace(target)
