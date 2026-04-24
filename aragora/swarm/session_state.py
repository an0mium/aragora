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


def _normalize_repo_slug(value: Any) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    normalized = text.strip().strip("/").lower()
    return normalized or None


def _coerce_exit_code(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = _optional_text(value)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _text_list(value: Any, *, limit: int = 25) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    items: list[str] = []
    seen: set[str] = set()
    for raw in value:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        items.append(text)
        if len(items) >= limit:
            break
    return items


def _coerce_failing_verification(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    command = _optional_text(value.get("command"))
    exit_code = _coerce_exit_code(value.get("exit_code"))
    stderr_tail = _optional_text(value.get("stderr_tail"))
    stdout_tail = _optional_text(value.get("stdout_tail"))
    result: dict[str, Any] = {}
    if command is not None:
        result["command"] = command
    if exit_code is not None:
        result["exit_code"] = exit_code
    if stderr_tail is not None:
        result["stderr_tail"] = stderr_tail
    if stdout_tail is not None:
        result["stdout_tail"] = stdout_tail
    return result or None


def _coerce_attempt_entry(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    entry: dict[str, Any] = {}
    at = _optional_text(value.get("at"))
    if at is not None:
        entry["at"] = at
    status = _optional_text(value.get("status"))
    if status is not None:
        entry["status"] = status
    worker_outcome = _optional_text(value.get("worker_outcome"))
    if worker_outcome is not None:
        entry["worker_outcome"] = worker_outcome
    failure_reason = _optional_text(value.get("failure_reason"))
    if failure_reason is not None:
        entry["failure_reason"] = failure_reason
    exit_code = _coerce_exit_code(value.get("exit_code"))
    if exit_code is not None:
        entry["exit_code"] = exit_code
    changed_paths = _text_list(value.get("changed_paths"))
    if not changed_paths:
        changed_paths = _coerce_path_list(value.get("changed_files"))
    if changed_paths:
        entry["changed_paths"] = list(changed_paths)
        entry["changed_files"] = list(changed_paths)
    test_output = _optional_text(value.get("test_output"))
    if test_output is not None:
        entry["test_output"] = test_output
    stderr_tail = _optional_text(value.get("stderr_tail"))
    if stderr_tail is not None:
        entry["stderr_tail"] = stderr_tail
    stdout_tail = _optional_text(value.get("stdout_tail"))
    if stdout_tail is not None:
        entry["stdout_tail"] = stdout_tail
    branch_name = _optional_text(value.get("branch_name"))
    if branch_name is not None:
        entry["branch_name"] = branch_name
    pr_url = _optional_text(value.get("pr_url"))
    if pr_url is not None:
        entry["pr_url"] = pr_url
    failing_verification = _coerce_failing_verification(value.get("failing_verification"))
    if failing_verification is not None:
        entry["failing_verification"] = failing_verification
    return entry or None


def _attempt_changed_paths(attempt: Mapping[str, Any]) -> list[str]:
    paths = _text_list(attempt.get("changed_paths"))
    if paths:
        return paths
    return _coerce_path_list(attempt.get("changed_files"))


def _session_repo_slug(state: "SessionState") -> str | None:
    for key in ("repo_slug", "boss_repo", "repo", "repo_full_name"):
        slug = _normalize_repo_slug(state.metadata.get(key))
        if slug is not None:
            return slug
    return None


def _default_issue_session_id(issue_number: int, repo_slug: str | None = None) -> str:
    if repo_slug:
        return f"issue-{repo_slug}-{issue_number}"
    return f"issue-{issue_number}"


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
        self.attempts = [
            entry
            for entry in (_coerce_attempt_entry(item) for item in self.attempts)
            if entry is not None
        ]
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
            attempts=list(data.get("attempts") or []),
            repair_journal=_coerce_dict_list(data.get("repair_journal")),
            metadata=dict(data.get("metadata") or {}),
            created_at=_coerce_datetime(data.get("created_at")),
            updated_at=_coerce_datetime(data.get("updated_at")),
        )

    def record_attempt(
        self,
        exit_code: int | None = None,
        changed_files: list[str] | tuple[str, ...] | set[str] | None = None,
        test_output: str | None = None,
        worker_outcome: str | None = None,
        *,
        failure_reason: str | None = None,
        failing_verification: Mapping[str, Any] | None = None,
        stdout_tail: str | None = None,
        stderr_tail: str | None = None,
        status: str | None = None,
        outcome: str | None = None,
        target_agent: str | None = None,
        runner_type: str | None = None,
        worktree_path: str | None = None,
        branch_name: str | None = None,
        pr_url: str | None = None,
        resume_hint: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        advanced_mode = any(
            value is not None
            for value in (
                status,
                outcome,
                target_agent,
                runner_type,
                worktree_path,
                branch_name,
                pr_url,
                resume_hint,
                metadata,
            )
        )
        changed_paths = _text_list(changed_files or ())

        if advanced_mode:
            self.status = str(status or self.status or "created").strip() or "created"
            self.target_agent = _optional_text(target_agent) or self.target_agent
            self.runner_type = _optional_text(runner_type) or self.runner_type
            self.worktree_path = _optional_text(worktree_path) or self.worktree_path
            self.branch_name = _optional_text(branch_name) or self.branch_name
            self.pr_url = _optional_text(pr_url) or self.pr_url
            resolved_resume_hint = _optional_text(resume_hint)
            if resolved_resume_hint is not None:
                self.resume_hint = resolved_resume_hint

            payload = dict(metadata or {})
            if failure_reason is not None and "failure_reason" not in payload:
                payload["failure_reason"] = failure_reason
            if stdout_tail is not None and "stdout_tail" not in payload:
                payload["stdout_tail"] = stdout_tail
            if stderr_tail is not None and "stderr_tail" not in payload:
                payload["stderr_tail"] = stderr_tail
            if failing_verification is not None and "failing_verification" not in payload:
                payload["failing_verification"] = dict(failing_verification)
            failure_reason = _optional_text(payload.get("failure_reason")) or resolved_resume_hint
            attempt: dict[str, Any] = {
                "at": _utcnow().isoformat(),
                "status": self.status,
            }
            outcome_text = _optional_text(outcome) or _optional_text(worker_outcome)
            if outcome_text is not None:
                attempt["worker_outcome"] = outcome_text
            normalized_exit_code = _coerce_exit_code(exit_code)
            if normalized_exit_code is not None:
                attempt["exit_code"] = normalized_exit_code
            if changed_paths:
                attempt["changed_paths"] = list(changed_paths)
                attempt["changed_files"] = list(changed_paths)
            output_text = _optional_text(test_output)
            if output_text is not None:
                attempt["test_output"] = output_text
            if failure_reason is not None:
                attempt["failure_reason"] = failure_reason

            resolved_branch = _optional_text(payload.get("branch_name")) or _optional_text(
                branch_name
            )
            if resolved_branch is not None:
                attempt["branch_name"] = resolved_branch
            resolved_pr_url = _optional_text(payload.get("pr_url")) or _optional_text(pr_url)
            if resolved_pr_url is not None:
                attempt["pr_url"] = resolved_pr_url
            for key in ("stderr_tail", "stdout_tail"):
                text = _optional_text(payload.get(key))
                if text is not None:
                    attempt[key] = text
            failing_verification = _coerce_failing_verification(payload.get("failing_verification"))
            if failing_verification is not None:
                attempt["failing_verification"] = failing_verification

            self.attempts.append(attempt)
            self.retry_count = max(self.retry_count + 1, len(self.attempts))
            payload.pop("failure_reason", None)
            payload.pop("stderr_tail", None)
            payload.pop("stdout_tail", None)
            payload.pop("branch_name", None)
            payload.pop("pr_url", None)
            payload.pop("failing_verification", None)
            self.metadata.update(payload)
            self.touch()
            return dict(attempt)

        attempt = {
            "at": _utcnow().isoformat(),
            "exit_code": _coerce_exit_code(exit_code),
            "changed_files": list(changed_paths),
            "changed_paths": list(changed_paths),
            "test_output": _optional_text(test_output),
            "worker_outcome": _optional_text(worker_outcome),
            "failure_reason": _optional_text(failure_reason),
            "failing_verification": _coerce_dict(failing_verification),
            "stdout_tail": _optional_text(stdout_tail),
            "stderr_tail": _optional_text(stderr_tail),
        }
        self.attempts.append(attempt)
        self.retry_count = max(self.retry_count + 1, len(self.attempts))
        self.touch()
        return dict(attempt)

    def last_attempt(self) -> dict[str, Any] | None:
        if not self.attempts:
            return None
        return dict(self.attempts[-1])

    def should_resume(self) -> bool:
        last = self.last_attempt()
        if last is not None:
            if _attempt_changed_paths(last):
                return True
            if _optional_text(last.get("test_output")):
                return True
            if _optional_text(last.get("failure_reason")):
                return True
            if _coerce_failing_verification(last.get("failing_verification")) is not None:
                return True
            if _optional_text(last.get("worker_outcome")) not in {None, "completed"}:
                return True
        return bool(self.repair_journal) or self.phase not in {"explore", "plan"}

    def resume_context(self, *, max_attempts: int = 3) -> str:
        if not self.should_resume():
            return ""

        lines = [f"Resume from phase: {self.phase}"]
        if self.resume_hint:
            lines.append(f"Resume hint: {self.resume_hint}")

        recent_attempts = self.attempts[-max(1, int(max_attempts or 1)) :]
        if recent_attempts:
            lines.append("Prior attempts:")
            start_index = len(self.attempts) - len(recent_attempts) + 1
            for index, attempt in enumerate(recent_attempts, start=start_index):
                summary: list[str] = []
                exit_code = _coerce_exit_code(attempt.get("exit_code"))
                if exit_code is not None:
                    summary.append(f"exit={exit_code}")
                worker_outcome = _optional_text(attempt.get("worker_outcome"))
                if worker_outcome:
                    summary.append(worker_outcome)
                failure_reason = _optional_text(attempt.get("failure_reason"))
                if failure_reason:
                    summary.append(failure_reason)
                changed_paths = _attempt_changed_paths(attempt)
                if changed_paths:
                    summary.append(f"changed={', '.join(changed_paths[:5])}")
                lines.append(f"- Attempt {index}: {', '.join(summary) if summary else 'recorded'}")

                failing_verification = _coerce_failing_verification(
                    attempt.get("failing_verification")
                )
                if failing_verification is not None:
                    command = _optional_text(failing_verification.get("command"))
                    if command:
                        exit_suffix = ""
                        verification_exit = _coerce_exit_code(failing_verification.get("exit_code"))
                        if verification_exit is not None:
                            exit_suffix = f" (exit {verification_exit})"
                        lines.append(f"  Failing verification: {command}{exit_suffix}")
                    stderr_tail = _optional_text(failing_verification.get("stderr_tail"))
                    if stderr_tail:
                        lines.append(f"  Stderr: {stderr_tail[-240:]}")
                    stdout_tail = _optional_text(failing_verification.get("stdout_tail"))
                    if stdout_tail:
                        lines.append(f"  Stdout: {stdout_tail[-240:]}")
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

    def resume_payload(self, *, max_attempts: int = 3) -> dict[str, Any]:
        recent_attempts = [dict(item) for item in self.attempts[-max(1, int(max_attempts or 1)) :]]
        repair_journal = recent_attempts or [
            dict(item) for item in self.repair_journal[-max(1, int(max_attempts or 1)) :]
        ]
        context: dict[str, Any] = {
            "session_id": self.session_id,
            "issue_number": self.issue_number,
            "phase": self.phase,
            "status": self.status,
            "retry_count": self.retry_count,
            "resume_hint": self.resume_hint,
            "target_agent": self.target_agent,
            "runner_type": self.runner_type,
            "worktree_path": self.worktree_path,
            "branch_name": self.branch_name,
            "pr_url": self.pr_url,
            "repair_journal": repair_journal,
        }
        if recent_attempts:
            context["last_attempt"] = dict(recent_attempts[-1])
        return {key: value for key, value in context.items() if value not in (None, [], {})}

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

    def list_sessions(
        self,
        *,
        issue_number: int | None = None,
        repo_slug: str | None = None,
    ) -> list[SessionState]:
        items: list[SessionState] = []
        issue_filter = _coerce_issue_number(issue_number)
        repo_filter = _normalize_repo_slug(repo_slug)
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
            if repo_filter is not None and _session_repo_slug(state) != repo_filter:
                continue
            items.append(state)
        items.sort(
            key=lambda state: (state.updated_at, state.created_at, state.session_id),
            reverse=True,
        )
        return items

    def latest_for_issue(
        self,
        issue_number: int,
        *,
        repo_slug: str | None = None,
    ) -> SessionState | None:
        sessions = self.list_sessions(issue_number=issue_number, repo_slug=repo_slug)
        return sessions[0] if sessions else None

    def record_attempt(
        self,
        *,
        issue_number: int,
        repo_slug: str | None = None,
        status: str,
        outcome: str | None = None,
        exit_code: int | None = None,
        changed_files: list[str] | None = None,
        target_agent: str | None = None,
        runner_type: str | None = None,
        worktree_path: str | None = None,
        branch_name: str | None = None,
        pr_url: str | None = None,
        resume_hint: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        session_id: str | None = None,
    ) -> SessionState:
        issue_value = _coerce_issue_number(issue_number)
        if issue_value is None:
            raise ValueError("issue_number must be a positive integer")
        payload_metadata = dict(metadata or {})
        repo_value = _normalize_repo_slug(repo_slug)
        if repo_value is None:
            for key in ("repo_slug", "boss_repo", "repo", "repo_full_name"):
                repo_value = _normalize_repo_slug(payload_metadata.get(key))
                if repo_value is not None:
                    break
        if repo_value is not None:
            payload_metadata.setdefault("repo_slug", repo_value)
            payload_metadata.setdefault("boss_repo", repo_value)
        state = (
            self.load(session_id) if session_id is not None else None
        ) or self.latest_for_issue(issue_value, repo_slug=repo_value)
        if state is None:
            state = SessionState(
                session_id=session_id or _default_issue_session_id(issue_value, repo_value),
                issue_number=issue_value,
                status="created",
            )
        state.issue_number = issue_value
        if repo_value is not None:
            state.metadata.setdefault("repo_slug", repo_value)
            state.metadata.setdefault("boss_repo", repo_value)
        state.record_attempt(
            status=status,
            outcome=outcome,
            exit_code=exit_code,
            changed_files=changed_files,
            target_agent=target_agent,
            runner_type=runner_type,
            worktree_path=worktree_path,
            branch_name=branch_name,
            pr_url=pr_url,
            resume_hint=resume_hint,
            metadata=payload_metadata,
        )
        self.save(state)
        return state

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
            str(attempt.get("failure_reason") or "").strip(),
            str(attempt.get("test_output") or "").strip(),
        ]
        failing_verification = _coerce_failing_verification(attempt.get("failing_verification"))
        if failing_verification is not None:
            text_parts.extend(
                [
                    str(failing_verification.get("command") or "").strip(),
                    str(failing_verification.get("stderr_tail") or "").strip(),
                    str(failing_verification.get("stdout_tail") or "").strip(),
                ]
            )
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
        exit_code = _coerce_exit_code(attempt.get("exit_code"))
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


def load_resume_context_for_issue(
    issue_number: int | None,
    *,
    repo_slug: str | None = None,
) -> str:
    """Load resume context from prior session state for an issue (BC-02).

    Returns the resume_context string if a prior session exists with
    resumable state, otherwise returns empty string.
    """
    if not issue_number:
        return ""
    store = SessionStateStore()
    sessions = store.list_sessions(issue_number=issue_number, repo_slug=repo_slug)
    if not sessions:
        return ""
    latest = max(sessions, key=lambda s: s.updated_at)
    if latest.should_resume():
        context = latest.resume_context()
        if context and len(context.strip()) > 20:
            return context.strip()
    return ""


def summarize_session_blocker(state: SessionState | None) -> str | None:
    if state is None or state.issue_number is None:
        return None
    if not state.attempts:
        return f"Issue #{state.issue_number} exhausted retries, but no persisted attempt journal is available."

    last_attempt = state.attempts[-1]
    failing_verification = _coerce_failing_verification(last_attempt.get("failing_verification"))
    failure_reason = _optional_text(last_attempt.get("failure_reason")) or state.resume_hint
    changed_paths = _attempt_changed_paths(last_attempt)
    exit_code = _coerce_exit_code(last_attempt.get("exit_code"))
    pr_url = _optional_text(last_attempt.get("pr_url")) or state.pr_url
    worker_outcome = _optional_text(last_attempt.get("worker_outcome")) or state.status

    if failing_verification is not None:
        command = _optional_text(failing_verification.get("command")) or "verification command"
        verification_exit = _coerce_exit_code(failing_verification.get("exit_code"))
        suffix = f" (exit {verification_exit})" if verification_exit is not None else ""
        return (
            f"Issue #{state.issue_number} exhausted retries; last blocker was failing verification "
            f"`{command}`{suffix}."
        )
    if pr_url is not None:
        return (
            f"Issue #{state.issue_number} exhausted retries; last attempt produced reviewable output "
            f"at {pr_url} but still needs human follow-up."
        )
    if changed_paths and exit_code not in (None, 0):
        return (
            f"Issue #{state.issue_number} exhausted retries; last attempt changed "
            f"{len(changed_paths)} file(s) but exited {exit_code}: "
            f"{failure_reason or worker_outcome or 'worker failed'}."
        )
    if not changed_paths:
        return (
            f"Issue #{state.issue_number} exhausted retries; last attempt made no committed file changes"
            + (f": {failure_reason}." if failure_reason else ".")
        )
    return (
        f"Issue #{state.issue_number} exhausted retries; last recorded outcome was "
        f"{worker_outcome or 'unknown'}" + (f": {failure_reason}." if failure_reason else ".")
    )
