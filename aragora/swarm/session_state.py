from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

_SESSION_ID_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return (
            value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
        )
    text = _optional_text(value)
    if not text:
        return _utcnow()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return _utcnow()


def _coerce_issue_number(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _sanitize_session_id(session_id: str) -> str:
    cleaned = _SESSION_ID_PATTERN.sub("-", str(session_id).strip()).strip("-")
    if not cleaned:
        raise ValueError("session_id must contain at least one filename-safe character")
    return cleaned


@dataclass(slots=True)
class SessionState:
    """Durable local session metadata for future retry/repair orchestration."""

    session_id: str
    status: str = "created"
    issue_number: int | None = None
    target_agent: str | None = None
    runner_type: str | None = None
    worktree_path: str | None = None
    branch_name: str | None = None
    pr_url: str | None = None
    resume_hint: str | None = None
    retry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)

    def __post_init__(self) -> None:
        self.session_id = _sanitize_session_id(self.session_id)
        self.status = str(self.status or "created").strip() or "created"
        self.issue_number = _coerce_issue_number(self.issue_number)
        self.target_agent = _optional_text(self.target_agent)
        self.runner_type = _optional_text(self.runner_type)
        self.worktree_path = _optional_text(self.worktree_path)
        self.branch_name = _optional_text(self.branch_name)
        self.pr_url = _optional_text(self.pr_url)
        self.resume_hint = _optional_text(self.resume_hint)
        self.retry_count = max(0, int(self.retry_count or 0))
        self.created_at = _coerce_datetime(self.created_at)
        self.updated_at = _coerce_datetime(self.updated_at)
        self.metadata = dict(self.metadata or {})

    def touch(self) -> None:
        self.updated_at = _utcnow()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.status,
            "issue_number": self.issue_number,
            "target_agent": self.target_agent,
            "runner_type": self.runner_type,
            "worktree_path": self.worktree_path,
            "branch_name": self.branch_name,
            "pr_url": self.pr_url,
            "resume_hint": self.resume_hint,
            "retry_count": self.retry_count,
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "SessionState":
        data = dict(payload or {})
        return cls(
            session_id=str(data.get("session_id", "")).strip(),
            status=data.get("status", "created"),
            issue_number=data.get("issue_number"),
            target_agent=data.get("target_agent"),
            runner_type=data.get("runner_type"),
            worktree_path=data.get("worktree_path"),
            branch_name=data.get("branch_name"),
            pr_url=data.get("pr_url"),
            resume_hint=data.get("resume_hint"),
            retry_count=data.get("retry_count", 0),
            metadata=dict(data.get("metadata") or {}),
            created_at=_coerce_datetime(data.get("created_at")),
            updated_at=_coerce_datetime(data.get("updated_at")),
        )


class SessionStateStore:
    """Local JSON store for persisted swarm session state."""

    def __init__(self, *, state_dir: Path | None = None) -> None:
        self._state_dir = (
            Path(state_dir).resolve()
            if state_dir is not None
            else Path.home() / ".aragora" / "sessions"
        )
        self._state_dir.mkdir(parents=True, exist_ok=True)

    @property
    def state_dir(self) -> Path:
        return self._state_dir

    def path_for(self, session_id: str) -> Path:
        return self._state_dir / f"{_sanitize_session_id(session_id)}.json"

    def save(self, state: SessionState) -> Path:
        state.touch()
        destination = self.path_for(state.session_id)
        tmp_path = destination.with_suffix(".json.tmp")
        payload = json.dumps(state.to_dict(), indent=2, sort_keys=False) + "\n"
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(destination)
        return destination

    def load(self, session_id: str) -> SessionState | None:
        path = self.path_for(session_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("session state payload must be a JSON object")
        return SessionState.from_dict(payload)

    def list_sessions(self, *, issue_number: int | None = None) -> list[SessionState]:
        items: list[SessionState] = []
        issue_filter = _coerce_issue_number(issue_number)
        for path in sorted(self._state_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    continue
                state = SessionState.from_dict(payload)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
            if issue_filter is not None and state.issue_number != issue_filter:
                continue
            items.append(state)
        items.sort(
            key=lambda state: (state.updated_at, state.created_at, state.session_id),
            reverse=True,
        )
        return items

    def cleanup_old(
        self,
        *,
        older_than: datetime | None = None,
        max_age: timedelta | None = None,
        now: datetime | None = None,
    ) -> list[Path]:
        reference = _coerce_datetime(now) if now is not None else _utcnow()
        cutoff = (
            _coerce_datetime(older_than)
            if older_than is not None
            else reference - (max_age or timedelta(days=7))
        )
        removed: list[Path] = []
        for state in self.list_sessions():
            if state.updated_at >= cutoff:
                continue
            path = self.path_for(state.session_id)
            if path.exists():
                path.unlink()
                removed.append(path)
        return removed
