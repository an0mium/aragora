"""Read-only ingest of Codex (Desktop / CLI) on-disk session state.

The Codex app persists everything we need to *observe* a sibling agent locally,
with no network and no API:

* ``<codex_home>/sessions/YYYY/MM/DD/rollout-*.jsonl`` -- one append-only
  transcript per session. ``session_meta`` carries ``cwd`` (which repo/worktree
  the session is operating on) and ``originator``; ``event_msg`` lines of type
  ``agent_message`` carry the human-visible narration.
* ``<codex_home>/session_index.jsonl`` -- ``{id, thread_name, updated_at}`` per
  thread; the cheap recency cursor + human-readable thread names.
* ``<codex_home>/automations/<name>/ledger.jsonl`` -- already-structured
  steering records emitted by Codex's headless loops: ``action.target``
  (pr/head/branch), ``forbidden_actions``, ``git.head``/``origin_main``,
  ``summary`` (open PR count, runner blockers).

This module turns those into typed summaries so an operator (or the boss loop)
can cross-check a sibling agent's claimed state against live ``git``/``gh``
without copy-pasting transcripts. It is strictly read-only and defensive: a
malformed line is skipped, never raised, so a half-written file (Codex is
appending concurrently) can never crash the digest.

The ingest is inert unless pointed at a Codex home; the ``enable_codex_bridge``
feature flag (default OFF) gates any *orchestration* use. Reading for a one-shot
digest needs no flag -- it touches nothing.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any

# A rollout filename embeds the session UUID after the local timestamp, e.g.
# ``rollout-2026-06-15T15-40-37-019ecd03-c60c-7bc1-8b9c-6477893310f6.jsonl``.
_ROLLOUT_PREFIX = "rollout-"
_ROLLOUT_SUFFIX = ".jsonl"


def default_codex_home() -> Path:
    """Codex home directory. Override with ``ARAGORA_CODEX_HOME`` (operator-specific)."""
    override = os.environ.get("ARAGORA_CODEX_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".codex"


def _parse_iso(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp defensively; naive values are treated as UTC.

    Returns ``None`` for anything unparseable so callers can skip rather than
    crash on a malformed or partially-written record.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file, skipping blank/malformed lines."""
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError:
        return
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except ValueError:  # JSONDecodeError subclasses ValueError
                continue
            if isinstance(payload, dict):
                yield payload


@dataclass(slots=True)
class CodexSessionSummary:
    """A compact, human-readable view of one Codex session rollout."""

    session_id: str
    rollout_path: str
    cwd: str | None
    originator: str | None
    model_provider: str | None
    started_at: str | None
    updated_at: str | None
    agent_message_count: int
    last_agent_message: str | None
    thread_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "rollout_path": self.rollout_path,
            "cwd": self.cwd,
            "originator": self.originator,
            "model_provider": self.model_provider,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "agent_message_count": self.agent_message_count,
            "last_agent_message": self.last_agent_message,
            "thread_name": self.thread_name,
        }


@dataclass(slots=True)
class CodexLedgerEntry:
    """One structured steering record from an automation ledger."""

    automation: str
    ledger_path: str
    kind: str | None
    reason: str | None
    pr: int | None
    head: str | None
    branch: str | None
    url: str | None
    forbidden_actions: list[str] = field(default_factory=list)
    git_head: str | None = None
    git_origin_main: str | None = None
    git_status_ok: bool | None = None
    open_pr_count: int | None = None
    runner_blockers: list[str] = field(default_factory=list)
    generated_at: str | None = None
    repo_root: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "automation": self.automation,
            "ledger_path": self.ledger_path,
            "kind": self.kind,
            "reason": self.reason,
            "pr": self.pr,
            "head": self.head,
            "branch": self.branch,
            "url": self.url,
            "forbidden_actions": list(self.forbidden_actions),
            "git_head": self.git_head,
            "git_origin_main": self.git_origin_main,
            "git_status_ok": self.git_status_ok,
            "open_pr_count": self.open_pr_count,
            "runner_blockers": list(self.runner_blockers),
            "generated_at": self.generated_at,
            "repo_root": self.repo_root,
        }


def read_session_index(home: Path | None = None) -> dict[str, dict[str, Any]]:
    """Return ``{session_id: {thread_name, updated_at}}`` from ``session_index.jsonl``.

    Later lines win, matching the append-only "last write is current" semantics.
    """
    home = home or default_codex_home()
    index: dict[str, dict[str, Any]] = {}
    for entry in _iter_jsonl(home / "session_index.jsonl"):
        session_id = entry.get("id")
        if isinstance(session_id, str) and session_id:
            index[session_id] = entry
    return index


def summarize_rollout(path: Path) -> CodexSessionSummary | None:
    """Summarize one rollout file. ``None`` if it has no usable ``session_meta``."""
    session_id: str | None = None
    cwd: str | None = None
    originator: str | None = None
    model_provider: str | None = None
    started_at: str | None = None
    updated_at: str | None = None
    agent_message_count = 0
    last_agent_message: str | None = None

    for record in _iter_jsonl(path):
        ts = record.get("timestamp")
        if isinstance(ts, str) and ts:
            updated_at = ts
        rtype = record.get("type")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        if rtype == "session_meta":
            sid = payload.get("id")
            session_id = sid if isinstance(sid, str) else session_id
            cwd = payload.get("cwd") if isinstance(payload.get("cwd"), str) else cwd
            originator = (
                payload.get("originator")
                if isinstance(payload.get("originator"), str)
                else originator
            )
            model_provider = (
                payload.get("model_provider")
                if isinstance(payload.get("model_provider"), str)
                else model_provider
            )
            meta_ts = payload.get("timestamp")
            if isinstance(meta_ts, str) and meta_ts:
                started_at = meta_ts
        elif rtype == "event_msg" and payload.get("type") == "agent_message":
            message = payload.get("message")
            if isinstance(message, str) and message.strip():
                agent_message_count += 1
                last_agent_message = message.strip()

    if session_id is None:
        # Fall back to the timestamp+UUID portion of the filename so a rollout
        # without a readable meta line is still uniquely identifiable.
        stem = path.name
        if stem.startswith(_ROLLOUT_PREFIX) and stem.endswith(_ROLLOUT_SUFFIX):
            session_id = stem[len(_ROLLOUT_PREFIX) : -len(_ROLLOUT_SUFFIX)]
        else:
            return None

    return CodexSessionSummary(
        session_id=session_id,
        rollout_path=str(path),
        cwd=cwd,
        originator=originator,
        model_provider=model_provider,
        started_at=started_at,
        updated_at=updated_at,
        agent_message_count=agent_message_count,
        last_agent_message=last_agent_message,
    )


def _session_date_dirs(home: Path) -> Iterator[tuple[datetime, Path]]:
    """Yield ``(date, dir)`` for each ``sessions/YYYY/MM/DD`` directory."""
    sessions_root = home / "sessions"
    if not sessions_root.is_dir():
        return
    for year_dir in sessions_root.iterdir():
        if not year_dir.is_dir():
            continue
        for month_dir in year_dir.iterdir():
            if not month_dir.is_dir():
                continue
            for day_dir in month_dir.iterdir():
                if not day_dir.is_dir():
                    continue
                try:
                    day = datetime(
                        int(year_dir.name), int(month_dir.name), int(day_dir.name), tzinfo=UTC
                    )
                except ValueError:
                    continue
                yield day, day_dir


def recent_sessions(
    home: Path | None = None,
    *,
    hours: float | None = 24.0,
    limit: int | None = 50,
    now: datetime | None = None,
) -> list[CodexSessionSummary]:
    """Summaries of recently-updated Codex sessions, newest first.

    ``hours`` filters on each rollout's last record timestamp (``None`` = no
    time filter). Date directories are pre-filtered with a one-day margin so we
    only read files that can plausibly be in range. ``thread_name`` is joined in
    from ``session_index.jsonl`` when available.
    """
    home = home or default_codex_home()
    now = now or datetime.now(UTC)
    cutoff = None if hours is None else now - timedelta(hours=hours)
    # Pre-filter date dirs (cheap) before reading any file content.
    dir_cutoff = None if cutoff is None else (cutoff - timedelta(days=1))
    index = read_session_index(home)

    summaries: list[tuple[datetime, CodexSessionSummary]] = []
    for day, day_dir in _session_date_dirs(home):
        if dir_cutoff is not None and day < dir_cutoff:
            continue
        for rollout in day_dir.glob(f"{_ROLLOUT_PREFIX}*{_ROLLOUT_SUFFIX}"):
            summary = summarize_rollout(rollout)
            if summary is None:
                continue
            updated = _parse_iso(summary.updated_at) or _parse_iso(summary.started_at)
            if cutoff is not None and (updated is None or updated < cutoff):
                continue
            meta = index.get(summary.session_id)
            if meta is not None:
                thread_name = meta.get("thread_name")
                if isinstance(thread_name, str):
                    summary.thread_name = thread_name
            sort_key = updated or datetime.min.replace(tzinfo=UTC)
            summaries.append((sort_key, summary))

    summaries.sort(key=lambda item: item[0], reverse=True)
    ordered = [summary for _, summary in summaries]
    if limit is not None:
        ordered = ordered[:limit]
    return ordered


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _coerce_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _as_dict(value: Any) -> dict[str, Any]:
    """Return ``value`` if it is a mapping, else an empty dict (type-narrowing helper)."""
    return value if isinstance(value, dict) else {}


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _ledger_entry_from_record(
    record: dict[str, Any], *, automation: str, ledger_path: Path
) -> CodexLedgerEntry:
    action = _as_dict(record.get("action"))
    target = _as_dict(action.get("target"))
    git = _as_dict(record.get("git"))
    summary = _as_dict(record.get("summary"))

    return CodexLedgerEntry(
        automation=automation,
        ledger_path=str(ledger_path),
        kind=_str_or_none(action.get("kind")),
        reason=_str_or_none(action.get("reason")),
        pr=_coerce_int(target.get("pr")),
        head=_str_or_none(target.get("head")),
        branch=_str_or_none(target.get("branch")),
        url=_str_or_none(target.get("url")),
        forbidden_actions=_coerce_str_list(record.get("forbidden_actions")),
        git_head=_str_or_none(git.get("head")),
        git_origin_main=_str_or_none(git.get("origin_main")),
        git_status_ok=_bool_or_none(git.get("status_ok")),
        open_pr_count=_coerce_int(summary.get("open_pr_count")),
        runner_blockers=_coerce_str_list(summary.get("runner_blockers")),
        generated_at=_str_or_none(record.get("generated_at")),
        repo_root=_str_or_none(record.get("repo_root")),
    )


def read_ledgers(
    home: Path | None = None,
    *,
    hours: float | None = 24.0,
    now: datetime | None = None,
) -> list[CodexLedgerEntry]:
    """Structured steering records from ``automations/*/ledger.jsonl``, newest first.

    ``hours`` filters on each record's ``generated_at`` (``None`` = no filter).
    """
    home = home or default_codex_home()
    now = now or datetime.now(UTC)
    cutoff = None if hours is None else now - timedelta(hours=hours)
    automations_root = home / "automations"
    if not automations_root.is_dir():
        return []

    entries: list[tuple[datetime, CodexLedgerEntry]] = []
    for automation_dir in automations_root.iterdir():
        ledger_path = automation_dir / "ledger.jsonl"
        if not ledger_path.is_file():
            continue
        for record in _iter_jsonl(ledger_path):
            generated = _parse_iso(record.get("generated_at"))
            if cutoff is not None and (generated is None or generated < cutoff):
                continue
            entry = _ledger_entry_from_record(
                record, automation=automation_dir.name, ledger_path=ledger_path
            )
            sort_key = generated or datetime.min.replace(tzinfo=UTC)
            entries.append((sort_key, entry))

    entries.sort(key=lambda item: item[0], reverse=True)
    return [entry for _, entry in entries]
