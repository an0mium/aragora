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
EventType: TypeAlias = Literal[
    "turn_started",
    "turn_completed",
    "footer_ok",
    "footer_malformed",
    "footer_missing",
    "footer_repair_requested",
    "run_started",
    "run_failed",
    "run_completed",
]


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class BridgeRun:
    run_id: str
    created_at: str
    updated_at: str
    status: RunStatus
    active_role: str | None
    footer_mode: str
    participants: list[str]
    worktree_path: str
    worktree_agent_slug: str
    turn_count: int = 0
    last_event_id: str | None = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "active_role": self.active_role,
            "footer_mode": self.footer_mode,
            "participants": list(self.participants),
            "worktree_path": self.worktree_path,
            "worktree_agent_slug": self.worktree_agent_slug,
            "turn_count": self.turn_count,
            "last_event_id": self.last_event_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BridgeRun":
        participants = payload.get("participants", [])
        if not isinstance(participants, list):
            raise TypeError("participants must be a list")
        return cls(
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
            run_id=str(payload["run_id"]),
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
            status=payload["status"],
            active_role=payload.get("active_role"),
            footer_mode=str(payload["footer_mode"]),
            participants=[str(item) for item in participants],
            worktree_path=str(payload["worktree_path"]),
            worktree_agent_slug=str(payload["worktree_agent_slug"]),
            turn_count=int(payload.get("turn_count", 0)),
            last_event_id=(
                str(payload["last_event_id"]) if payload.get("last_event_id") is not None else None
            ),
        )


@dataclass(slots=True)
class BridgeSession:
    harness: str
    session_id: str | None
    created_at: str
    last_turn_at: str | None
    harness_options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "harness": self.harness,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_turn_at": self.last_turn_at,
            "harness_options": dict(self.harness_options),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BridgeSession":
        options = payload.get("harness_options", {})
        if not isinstance(options, dict):
            raise TypeError("harness_options must be a mapping")
        return cls(
            harness=str(payload["harness"]),
            session_id=str(payload["session_id"]) if payload.get("session_id") else None,
            created_at=str(payload["created_at"]),
            last_turn_at=str(payload["last_turn_at"]) if payload.get("last_turn_at") else None,
            harness_options=dict(options),
        )


@dataclass(slots=True)
class SessionRegistry:
    roles: dict[str, BridgeSession]
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "roles": {role: session.to_dict() for role, session in self.roles.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SessionRegistry":
        raw_roles = payload.get("roles", {})
        if not isinstance(raw_roles, dict):
            raise TypeError("roles must be a mapping")
        return cls(
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
            roles={
                str(role): BridgeSession.from_dict(session)
                for role, session in raw_roles.items()
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
    turn_index: int
    type: EventType
    role: str
    at: str
    payload: dict[str, Any]
    parse_status: ParseStatus | None = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "turn_index": self.turn_index,
            "type": self.type,
            "role": self.role,
            "parse_status": self.parse_status,
            "at": self.at,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TurnRecord":
        data = payload.get("payload", {})
        if not isinstance(data, dict):
            raise TypeError("payload must be a mapping")
        return cls(
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
            event_id=str(payload["event_id"]),
            turn_index=int(payload["turn_index"]),
            type=payload["type"],
            role=str(payload["role"]),
            parse_status=payload.get("parse_status"),
            at=str(payload["at"]),
            payload=dict(data),
        )
