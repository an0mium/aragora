from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

_SESSION_ID_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
_IMPORT_ERROR_RE = re.compile(r"\b(module(notfounderror)?|importerror|cannot import name)\b", re.I)
_DEPENDENCY_MISSING_RE = re.compile(
    r"(command not found|executable file not found|not installed|missing dependency|no such file or directory)",
    re.I,
)
_TIMEOUT_RE = re.compile(r"\b(time[ -]?out|timed out)\b", re.I)
_SCOPE_TOO_BROAD_RE = re.compile(
    r"(scope too broad|too broad|outside file scope|file scope spans|task sanitizer|quarantined)",
    re.I,
)
_TEST_FAILURE_RE = re.compile(r"(assertionerror|\bfailed\b|\btraceback\b|\bpytest\b)", re.I)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _coerce_phase(value: Any) -> str:
    return str(value or "").strip() or "explore"


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


def _coerce_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    items: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            items.append(dict(item))
    return items


def _coerce_path_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _coerce_dict(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return dict(value)


@dataclass(slots=True)
class SessionState:
    """Durable local session metadata for future retry/repair orchestration."""

    session_id: str
    phase: str = "explore"
    status: str = "created"
    issue_number: int | None = None
    target_agent: str | None = None
    runner_type: str | None = None
    worktree_path: str | None = None
    branch_name: str | None = None
    pr_url: str | None = None
    resume_hint: str | None = None
    blocker_evidence: str | None = None
    retry_count: int = 0
    attempts: list[dict[str, Any]] = field(default_factory=list)
    repair_journal: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)

    def __post_init__(self) -> None:
        self.session_id = _sanitize_session_id(self.session_id)
        self.phase = _coerce_phase(self.phase)
        self.status = str(self.status or "created").strip() or "created"
        self.issue_number = _coerce_issue_number(self.issue_number)
        self.target_agent = _optional_text(self.target_agent)
        self.runner_type = _optional_text(self.runner_type)
        self.worktree_path = _optional_text(self.worktree_path)
        self.branch_name = _optional_text(self.branch_name)
        self.pr_url = _optional_text(self.pr_url)
        self.resume_hint = _optional_text(self.resume_hint)
        self.blocker_evidence = _optional_text(self.blocker_evidence)
        self.retry_count = max(0, int(self.retry_count or 0))
        self.attempts = _coerce_dict_list(self.attempts)
        self.repair_journal = _coerce_dict_list(self.repair_journal)
        self.created_at = _coerce_datetime(self.created_at)
        self.updated_at = _coerce_datetime(self.updated_at)
        self.metadata = dict(self.metadata or {})

    def touch(self) -> None:
        self.updated_at = _utcnow()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "phase": self.phase,
            "status": self.status,
            "issue_number": self.issue_number,
            "target_agent": self.target_agent,
            "runner_type": self.runner_type,
            "worktree_path": self.worktree_path,
            "branch_name": self.branch_name,
            "pr_url": self.pr_url,
            "resume_hint": self.resume_hint,
            "blocker_evidence": self.blocker_evidence,
            "retry_count": self.retry_count,
            "attempts": [dict(item) for item in self.attempts],
            "repair_journal": [dict(item) for item in self.repair_journal],
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "SessionState":
        data = dict(payload or {})
        return cls(
            session_id=str(data.get("session_id", "")).strip(),
            phase=data.get("phase", "explore"),
            status=data.get("status", "created"),
            issue_number=data.get("issue_number"),
            target_agent=data.get("target_agent"),
            runner_type=data.get("runner_type"),
            worktree_path=data.get("worktree_path"),
            branch_name=data.get("branch_name"),
            pr_url=data.get("pr_url"),
            resume_hint=data.get("resume_hint"),
            blocker_evidence=data.get("blocker_evidence"),
            retry_count=data.get("retry_count", 0),
            attempts=_coerce_dict_list(data.get("attempts")),
            repair_journal=_coerce_dict_list(data.get("repair_journal")),
            metadata=dict(data.get("metadata") or {}),
            created_at=_coerce_datetime(data.get("created_at")),
            updated_at=_coerce_datetime(data.get("updated_at")),
        )

    def record_attempt(
        self,
        exit_code: int | None,
        changed_files: list[str] | tuple[str, ...] | set[str] | None,
        test_output: str | None,
        worker_outcome: str | None,
        *,
        failure_reason: str | None = None,
        failing_verification: Mapping[str, Any] | None = None,
        stdout_tail: str | None = None,
        stderr_tail: str | None = None,
    ) -> dict[str, Any]:
        attempt = {
            "at": _utcnow().isoformat(),
            "exit_code": int(exit_code) if exit_code is not None else None,
            "changed_files": _coerce_path_list(changed_files),
            "test_output": _optional_text(test_output),
            "worker_outcome": _optional_text(worker_outcome),
            "failure_reason": _optional_text(failure_reason),
            "failing_verification": _coerce_dict(failing_verification),
            "stdout_tail": _optional_text(stdout_tail),
            "stderr_tail": _optional_text(stderr_tail),
        }
        self.attempts.append(attempt)
        self.retry_count = max(self.retry_count, len(self.attempts))
        self.touch()
        return dict(attempt)

    def last_attempt(self) -> dict[str, Any] | None:
        if not self.attempts:
            return None
        return dict(self.attempts[-1])

    def should_resume(self) -> bool:
        last = self.last_attempt()
        if last is not None:
            if last.get("changed_files"):
                return True
            if _optional_text(last.get("test_output")):
                return True
            if _optional_text(last.get("worker_outcome")) not in {None, "completed"}:
                return True
        return bool(self.repair_journal) or self.phase not in {"explore", "plan"}

    def resume_context(self) -> str:
        if not self.should_resume():
            return ""

        lines = [f"Resume from phase: {self.phase}"]
        if self.resume_hint:
            lines.append(f"Resume hint: {self.resume_hint}")

        if self.attempts:
            lines.append("Prior attempts:")
            for index, attempt in enumerate(
                self.attempts[-3:], start=max(1, len(self.attempts) - 2)
            ):
                summary: list[str] = []
                exit_code = attempt.get("exit_code")
                if exit_code is not None:
                    summary.append(f"exit={exit_code}")
                worker_outcome = _optional_text(attempt.get("worker_outcome"))
                if worker_outcome:
                    summary.append(worker_outcome)
                failure_reason = _optional_text(attempt.get("failure_reason"))
                if failure_reason:
                    summary.append(f"reason={failure_reason}")
                changed = _coerce_path_list(attempt.get("changed_files"))
                if changed:
                    summary.append(f"changed={', '.join(changed[:5])}")
                lines.append(f"- Attempt {index}: {', '.join(summary) if summary else 'recorded'}")
                failing = _coerce_dict(attempt.get("failing_verification"))
                if failing:
                    command = _optional_text(failing.get("command"))
                    if command:
                        lines.append(
                            f"  Failing verification: {command} (exit {failing.get('exit_code')})"
                        )
                    stderr_tail = _optional_text(failing.get("stderr_tail"))
                    if stderr_tail:
                        lines.append(f"  Verification stderr: {stderr_tail[-240:]}")
                test_output = _optional_text(attempt.get("test_output"))
                if test_output:
                    lines.append(f"  Evidence: {test_output[-240:]}")
                stderr_tail = _optional_text(attempt.get("stderr_tail"))
                if stderr_tail:
                    lines.append(f"  stderr: {stderr_tail[-240:]}")
                stdout_tail = _optional_text(attempt.get("stdout_tail"))
                if stdout_tail:
                    lines.append(f"  stdout: {stdout_tail[-240:]}")

        if self.repair_journal:
            lines.append(f"Repair journal entries available: {len(self.repair_journal)}")
            for entry in self.repair_journal[-2:]:
                at = _optional_text(entry.get("at"))
                worker_outcome = _optional_text(entry.get("worker_outcome"))
                failure_reason = _optional_text(entry.get("failure_reason"))
                header_parts = [part for part in [worker_outcome, failure_reason] if part]
                header = f"- Repair {at}: " if at else "- Repair: "
                header += ", ".join(header_parts) if header_parts else "recorded"
                lines.append(header)
                failing = _coerce_dict(entry.get("failing_verification"))
                if failing:
                    command = _optional_text(failing.get("command"))
                    if command:
                        lines.append(
                            f"  Failing verification: {command} (exit {failing.get('exit_code')})"
                        )
                    stderr_tail = _optional_text(failing.get("stderr_tail"))
                    if stderr_tail:
                        lines.append(f"  Verification stderr: {stderr_tail[-240:]}")
                blocker = _optional_text(entry.get("blocker_evidence"))
                if blocker:
                    lines.append(f"  Blocker evidence: {blocker[-240:]}")
                changed = _coerce_path_list(entry.get("changed_paths"))
                if changed:
                    lines.append(f"  Changed: {', '.join(changed[:5])}")

        return "\n".join(lines).strip()

    def set_blocker(self, evidence: str) -> None:
        self.blocker_evidence = _optional_text(evidence)
        self.touch()

    def clear_blocker(self) -> None:
        self.blocker_evidence = None
        self.touch()

    def sync_repair_journal(self, entries: Any, *, max_entries: int = 3) -> None:
        if not isinstance(entries, list):
            return
        normalized = [dict(entry) for entry in entries if isinstance(entry, Mapping)]
        if not normalized:
            return
        self.repair_journal = normalized[-max_entries:]
        self.touch()


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


def _attempt_evidence(state: SessionState) -> list[tuple[dict[str, Any], str]]:
    evidence: list[tuple[dict[str, Any], str]] = []
    for attempt in reversed(state.attempts):
        text_parts = [
            str(attempt.get("worker_outcome") or "").strip(),
            str(attempt.get("test_output") or "").strip(),
        ]
        combined = "\n".join(part for part in text_parts if part).strip()
        evidence.append((attempt, combined))
    if state.blocker_evidence:
        evidence.append(({}, state.blocker_evidence))
    return evidence


def _suggested_action(blocker_type: str) -> str:
    actions = {
        "test_failure": "Run the failing validation locally, fix the assertion or regression, and retry.",
        "import_error": "Repair the import/module path, then rerun the targeted validation command.",
        "timeout": "Narrow the validation scope or add progress checkpoints before retrying.",
        "scope_too_broad": "Split the task into a smaller bounded write scope before another retry.",
        "dependency_missing": "Install or provision the missing dependency or command before retrying.",
    }
    return actions[blocker_type]


def classify_session_blocker(state: SessionState) -> dict[str, str]:
    evidence_rows = _attempt_evidence(state)
    for attempt, text in evidence_rows:
        normalized = text.strip()
        exit_code = attempt.get("exit_code")
        if exit_code in {-1, 124, 137} or _TIMEOUT_RE.search(normalized):
            blocker_type = "timeout"
        elif _IMPORT_ERROR_RE.search(normalized):
            blocker_type = "import_error"
        elif _DEPENDENCY_MISSING_RE.search(normalized):
            blocker_type = "dependency_missing"
        elif _SCOPE_TOO_BROAD_RE.search(normalized):
            blocker_type = "scope_too_broad"
        elif _TEST_FAILURE_RE.search(normalized):
            blocker_type = "test_failure"
        else:
            continue
        return {
            "blocker_type": blocker_type,
            "evidence": normalized or "session contains blocker evidence",
            "suggested_action": _suggested_action(blocker_type),
        }

    fallback = state.blocker_evidence or "Latest retry failed without structured blocker evidence."
    return {
        "blocker_type": "test_failure",
        "evidence": fallback,
        "suggested_action": _suggested_action("test_failure"),
    }


def load_resume_context_for_issue(issue_number: int | None) -> str:
    """Load resume context from prior session state for an issue (BC-02).

    Returns the resume_context string if a prior session exists with
    resumable state, otherwise returns empty string.
    """
    if not issue_number:
        return ""
    store = SessionStateStore()
    sessions = store.list_sessions(issue_number=issue_number)
    if not sessions:
        return ""
    latest = max(sessions, key=lambda s: s.updated_at)
    if latest.should_resume():
        context = latest.resume_context()
        if context and len(context.strip()) > 20:
            return context.strip()
    return ""
