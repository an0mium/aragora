from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from typing import Any
from typing import Literal
from typing import TypeAlias

SCHEMA_VERSION = 1

ParseStatus: TypeAlias = Literal["ok", "missing", "malformed"]
RunStatus: TypeAlias = Literal["running", "awaiting_human", "completed", "failed"]
SessionStatus: TypeAlias = Literal["not_started", "active", "completed", "failed"]
EventType: TypeAlias = Literal[
    "run_started",
    "run_failed",
    "run_completed",
    "turn.started",
    "turn.result",
    "turn.completed",
    "turn.repair_requested",
    "footer_ok",
    "footer_malformed",
    "footer_missing",
]


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class Participant:
    role: str
    harness: str
    model: str

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "harness": self.harness,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Participant":
        role = payload.get("role")
        harness = payload.get("harness")
        model = payload.get("model")
        if not isinstance(role, str) or not isinstance(harness, str) or not isinstance(model, str):
            raise TypeError("participant role, harness, and model must be strings")
        return cls(role=role, harness=harness, model=model)


@dataclass(slots=True)
class BridgeRun:
    run_id: str
    task: str
    created_at: str
    updated_at: str
    status: RunStatus
    completed_at: str | None
    last_turn_index: int
    next_actor: str | None
    repair_budget_per_turn: int
    footer_mode: str
    worktree_cleanup_mode: str
    participants: list[Participant]
    worktree_path: str
    worktree_agent_slug: str
    last_event_id: str | None = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "task": self.task,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "last_turn_index": self.last_turn_index,
            "next_actor": self.next_actor,
            "repair_budget_per_turn": self.repair_budget_per_turn,
            "footer_mode": self.footer_mode,
            "worktree_cleanup_mode": self.worktree_cleanup_mode,
            "participants": [participant.to_dict() for participant in self.participants],
            "worktree_path": self.worktree_path,
            "worktree_agent_slug": self.worktree_agent_slug,
            "last_event_id": self.last_event_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BridgeRun":
        participants = payload.get("participants")
        if not isinstance(participants, list):
            raise TypeError("participants must be a list")
        task = payload.get("task")
        if not isinstance(task, str):
            raise TypeError("task must be a string")
        next_actor = payload.get("next_actor")
        if next_actor is not None and not isinstance(next_actor, str):
            raise TypeError("next_actor must be a string or null")
        completed_at = payload.get("completed_at")
        if completed_at is not None and not isinstance(completed_at, str):
            raise TypeError("completed_at must be a string or null")
        return cls(
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
            run_id=str(payload["run_id"]),
            task=task,
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
            status=payload["status"],
            completed_at=completed_at,
            last_turn_index=int(payload.get("last_turn_index", 0)),
            next_actor=next_actor,
            repair_budget_per_turn=int(payload.get("repair_budget_per_turn", 1)),
            footer_mode=str(payload["footer_mode"]),
            worktree_cleanup_mode=str(payload.get("worktree_cleanup_mode", "operator_triggered")),
            participants=[
                Participant.from_dict(item) for item in participants if isinstance(item, dict)
            ],
            worktree_path=str(payload["worktree_path"]),
            worktree_agent_slug=str(payload["worktree_agent_slug"]),
            last_event_id=(
                str(payload["last_event_id"]) if payload.get("last_event_id") is not None else None
            ),
        )


@dataclass(slots=True)
class BridgeSession:
    role: str
    harness: str
    model: str
    session_id: str | None
    worktree_agent_slug: str | None
    worktree_path: str | None
    branch: str | None
    session_status: SessionStatus
    started_at: str | None
    last_turn_index: int
    last_completed_at: str | None
    harness_options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": self.role,
            "harness": self.harness,
            "model": self.model,
            "session_id": self.session_id,
            "worktree_agent_slug": self.worktree_agent_slug,
            "worktree_path": self.worktree_path,
            "branch": self.branch,
            "session_status": self.session_status,
            "started_at": self.started_at,
            "last_turn_index": self.last_turn_index,
            "last_completed_at": self.last_completed_at,
        }
        if self.harness_options:
            payload["harness_options"] = dict(self.harness_options)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BridgeSession":
        harness_options = payload.get("harness_options", {})
        if not isinstance(harness_options, dict):
            raise TypeError("harness_options must be a mapping")
        session_status = payload.get("session_status", "not_started")
        if session_status not in {"not_started", "active", "completed", "failed"}:
            raise TypeError("session_status is invalid")
        role = payload.get("role")
        harness = payload.get("harness")
        model = payload.get("model", "")
        if not isinstance(role, str) or not isinstance(harness, str) or not isinstance(model, str):
            raise TypeError("role, harness, and model must be strings")
        session_id = payload.get("session_id")
        started_at = payload.get("started_at")
        last_completed_at = payload.get("last_completed_at")
        worktree_agent_slug = payload.get("worktree_agent_slug")
        worktree_path = payload.get("worktree_path")
        branch = payload.get("branch")
        for value, name in (
            (session_id, "session_id"),
            (started_at, "started_at"),
            (last_completed_at, "last_completed_at"),
            (worktree_agent_slug, "worktree_agent_slug"),
            (worktree_path, "worktree_path"),
            (branch, "branch"),
        ):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string or null")
        return cls(
            role=role,
            harness=harness,
            model=model,
            session_id=session_id,
            worktree_agent_slug=worktree_agent_slug,
            worktree_path=worktree_path,
            branch=branch,
            session_status=session_status,
            started_at=started_at,
            last_turn_index=int(payload.get("last_turn_index", 0)),
            last_completed_at=last_completed_at,
            harness_options=dict(harness_options),
        )


@dataclass(slots=True)
class SessionRegistry:
    run_id: str
    updated_at: str
    sessions: dict[str, BridgeSession]
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "updated_at": self.updated_at,
            "sessions": {role: session.to_dict() for role, session in self.sessions.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SessionRegistry":
        raw_sessions = payload.get("sessions", {})
        if not isinstance(raw_sessions, dict):
            raise TypeError("sessions must be a mapping")
        return cls(
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
            run_id=str(payload["run_id"]),
            updated_at=str(payload["updated_at"]),
            sessions={
                str(role): BridgeSession.from_dict(session)
                for role, session in raw_sessions.items()
                if isinstance(session, dict)
            },
        )


@dataclass(slots=True)
class BridgeFooter:
    summary: str
    next_actor: str | None
    needs_human: bool
    done: bool
    artifacts: list[str]
    tests_run: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "next_actor": self.next_actor,
            "needs_human": self.needs_human,
            "done": self.done,
            "artifacts": list(self.artifacts),
            "tests_run": list(self.tests_run),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BridgeFooter":
        summary = payload.get("summary")
        next_actor = payload.get("next_actor")
        needs_human = payload.get("needs_human")
        done = payload.get("done")
        artifacts = payload.get("artifacts")
        tests_run = payload.get("tests_run")
        if not isinstance(summary, str):
            raise TypeError("summary must be a string")
        if next_actor is not None and not isinstance(next_actor, str):
            raise TypeError("next_actor must be a string or null")
        if not isinstance(needs_human, bool):
            raise TypeError("needs_human must be a bool")
        if not isinstance(done, bool):
            raise TypeError("done must be a bool")
        if not isinstance(artifacts, list) or not all(isinstance(item, str) for item in artifacts):
            raise TypeError("artifacts must be a list[str]")
        if not isinstance(tests_run, list) or not all(isinstance(item, str) for item in tests_run):
            raise TypeError("tests_run must be a list[str]")
        return cls(
            summary=summary,
            next_actor=next_actor,
            needs_human=needs_human,
            done=done,
            artifacts=list(artifacts),
            tests_run=list(tests_run),
        )


@dataclass(slots=True)
class ParsedTurn:
    footer: BridgeFooter | None
    body_without_footer: str
    parse_status: ParseStatus
    footer_raw: str | None = None
    parse_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "footer": self.footer.to_dict() if self.footer is not None else None,
            "body_without_footer": self.body_without_footer,
            "parse_status": self.parse_status,
            "footer_raw": self.footer_raw,
            "parse_errors": list(self.parse_errors),
        }


@dataclass(slots=True)
class TurnRecord:
    event_id: str
    run_id: str
    turn_index: int
    event_type: EventType
    role: str
    harness: str
    session_id: str | None
    ts: str
    payload: dict[str, Any]
    parse_status: ParseStatus | None = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "run_id": self.run_id,
            "ts": self.ts,
            "event_type": self.event_type,
            "turn_index": self.turn_index,
            "role": self.role,
            "harness": self.harness,
            "session_id": self.session_id,
            "parse_status": self.parse_status,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TurnRecord":
        data = payload.get("payload", {})
        if not isinstance(data, dict):
            raise TypeError("payload must be a mapping")
        session_id = payload.get("session_id")
        if session_id is not None and not isinstance(session_id, str):
            raise TypeError("session_id must be a string or null")
        return cls(
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
            event_id=str(payload["event_id"]),
            run_id=str(payload["run_id"]),
            turn_index=int(payload["turn_index"]),
            event_type=payload["event_type"],
            role=str(payload["role"]),
            harness=str(payload["harness"]),
            session_id=session_id,
            parse_status=payload.get("parse_status"),
            ts=str(payload["ts"]),
            payload=dict(data),
        )
