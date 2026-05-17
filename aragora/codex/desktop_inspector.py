"""Read-only inspector for Codex Desktop sessions and threads.

Surfaces canonical thread metadata from ``state_5.sqlite`` and full session
transcripts from rollout JSONL files, with mandatory secret redaction on any
surfaced content.

This module is intentionally network-free and writes nothing to ``~/.codex/``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from aragora.debate.security_barrier import SecurityBarrier

from .desktop_paths import CodexDesktopPaths, resolve
from .jsonl_stream import iter_jsonl
from .sqlite_ro import sqlite_ro

# Extra redaction patterns layered on top of SecurityBarrier defaults to
# cover GitHub tokens, AWS access key IDs, and the OpenRouter prefix that the
# SecurityBarrier default ``sk-[…]`` pattern technically covers but worth
# being explicit about.
_EXTRA_REDACTION_PATTERNS = (
    r"gh[pousr]_[A-Za-z0-9_]{20,}",
    r"github_pat_[A-Za-z0-9_]{22,}",
    r"AKIA[0-9A-Z]{16}",
    r"sk-or-v1-[A-Za-z0-9]+",
)

LIST_TITLE_WIDTH = 160
LIST_FIRST_USER_MESSAGE_WIDTH = 240


def _rollout_path_from_db(value: str, *, paths: CodexDesktopPaths) -> Path:
    rollout = Path(value).expanduser()
    if rollout.is_absolute():
        return rollout
    return paths.home / rollout


def _build_barrier() -> SecurityBarrier:
    barrier = SecurityBarrier()
    for pattern in _EXTRA_REDACTION_PATTERNS:
        barrier.add_pattern(pattern)
    return barrier


def redact_display(value: str | Path | None) -> str | None:
    """Return a secret-redacted string for terminal/JSON display."""
    if value is None:
        return None
    return _build_barrier().redact(str(value))


@dataclass(frozen=True, slots=True)
class ThreadSummary:
    """Canonical metadata about one Codex Desktop thread.

    Pulled from the ``threads`` table in ``state_5.sqlite``. All string fields
    are redacted at construction time so that a thread whose title contains a
    leaked secret cannot leak it back through this object.
    """

    id: str
    title: str
    cwd: str
    model: str | None
    rollout_path: Path
    created_at: datetime
    updated_at: datetime
    tokens_used: int
    archived: bool
    git_sha: str | None
    git_branch: str | None
    source: str
    first_user_message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "cwd": self.cwd,
            "model": self.model,
            "rollout_path": redact_display(self.rollout_path),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tokens_used": self.tokens_used,
            "archived": self.archived,
            "git_sha": self.git_sha,
            "git_branch": self.git_branch,
            "source": self.source,
            "first_user_message": self.first_user_message,
        }

    def to_list_dict(self) -> dict[str, Any]:
        """Return bounded metadata safe for bulk thread-list output."""
        payload = self.to_dict()
        payload["title"] = truncate(str(payload["title"]), width=LIST_TITLE_WIDTH)
        payload["first_user_message"] = truncate(
            str(payload["first_user_message"]),
            width=LIST_FIRST_USER_MESSAGE_WIDTH,
        )
        return payload


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """Aggregate view of one rollout JSONL.

    Built by scanning up to ``max_events`` events. All string content
    (messages, titles) is already redacted.
    """

    rollout_path: Path
    events_scanned: int
    truncated: bool
    event_type_counts: dict[str, int]
    tool_call_counts: dict[str, int]
    first_user_message: str
    last_user_message: str
    model_provider: str | None
    started_at: datetime | None
    last_event_at: datetime | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollout_path": redact_display(self.rollout_path),
            "events_scanned": self.events_scanned,
            "truncated": self.truncated,
            "event_type_counts": dict(self.event_type_counts),
            "tool_call_counts": dict(self.tool_call_counts),
            "first_user_message": self.first_user_message,
            "last_user_message": self.last_user_message,
            "model_provider": self.model_provider,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_event_at": self.last_event_at.isoformat() if self.last_event_at else None,
        }


def _to_datetime(epoch_value: Any) -> datetime:
    """Best-effort conversion of integer/float epoch values into aware UTC datetimes.

    Codex Desktop stores some columns as seconds-since-epoch and others as
    milliseconds; we normalize by sniffing magnitude.
    """
    try:
        n = int(epoch_value)
    except (TypeError, ValueError):
        return datetime.fromtimestamp(0, tz=UTC)
    # Anything past ~3000-01-01 in seconds (32503680000) is almost certainly
    # milliseconds. Use that boundary to distinguish.
    if n > 32503680000:
        return datetime.fromtimestamp(n / 1000.0, tz=UTC)
    return datetime.fromtimestamp(n, tz=UTC)


def _extract_text(payload: Any) -> str:
    """Pull a flat text excerpt out of a Codex event payload.

    Handles the common shapes:
    - ``{"role": "user", "content": [{"type": "text", "text": "..."}]}``
    - ``{"role": "user", "content": "..."}``
    - ``{"text": "..."}``
    """
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        content = payload.get("content") or payload.get("text") or payload.get("message") or ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text_value = item.get("text") or item.get("content") or ""
                    if isinstance(text_value, str):
                        parts.append(text_value)
            return "\n".join(parts)
    return ""


def list_active_threads(
    *,
    since: timedelta,
    include_archived: bool = False,
    limit: int | None = None,
    paths: CodexDesktopPaths | None = None,
) -> list[ThreadSummary]:
    """Return Codex Desktop threads updated within the last ``since`` window.

    Threads are read from ``state_5.sqlite`` (read-only) and sorted newest
    first by ``updated_at``. Title and first-user-message fields are redacted
    so the returned objects are safe to print to a terminal.
    """
    paths = paths or resolve()
    barrier = _build_barrier()
    cutoff = datetime.now(UTC) - since
    cutoff_epoch = int(cutoff.timestamp())

    rows: list[ThreadSummary] = []
    sql = """
        SELECT
            id, COALESCE(title, '') AS title, cwd,
            COALESCE(model, '') AS model,
            rollout_path,
            created_at, updated_at,
            tokens_used,
            archived,
            git_sha, git_branch,
            source,
            COALESCE(first_user_message, '') AS first_user_message
        FROM threads
        WHERE updated_at >= ?
    """
    if not include_archived:
        sql += " AND archived = 0"
    sql += " ORDER BY updated_at DESC"
    if limit is not None and limit > 0:
        sql += f" LIMIT {int(limit)}"

    with sqlite_ro(paths.sqlite_path) as conn:
        cursor = conn.execute(sql, (cutoff_epoch,))
        for row in cursor:
            rows.append(
                ThreadSummary(
                    id=row["id"],
                    title=barrier.redact(row["title"] or ""),
                    cwd=barrier.redact(row["cwd"] or ""),
                    model=redact_display(row["model"]) if row["model"] else None,
                    rollout_path=_rollout_path_from_db(row["rollout_path"], paths=paths),
                    created_at=_to_datetime(row["created_at"]),
                    updated_at=_to_datetime(row["updated_at"]),
                    tokens_used=int(row["tokens_used"] or 0),
                    archived=bool(row["archived"]),
                    git_sha=redact_display(row["git_sha"]),
                    git_branch=redact_display(row["git_branch"]),
                    source=barrier.redact(row["source"] or ""),
                    first_user_message=barrier.redact(row["first_user_message"] or ""),
                )
            )
    return rows


def find_thread(thread_id: str, *, paths: CodexDesktopPaths | None = None) -> ThreadSummary | None:
    """Look up a single thread by id (or id-prefix of at least 8 chars).

    Returns ``None`` if no thread matches. Prefix matches are accepted to
    avoid forcing operators to paste the full UUID.
    """
    paths = paths or resolve()
    barrier = _build_barrier()
    cleaned = thread_id.strip()
    if len(cleaned) < 8:
        return None
    sql = """
        SELECT
            id, COALESCE(title, '') AS title, cwd,
            COALESCE(model, '') AS model,
            rollout_path,
            created_at, updated_at,
            tokens_used,
            archived,
            git_sha, git_branch,
            source,
            COALESCE(first_user_message, '') AS first_user_message
        FROM threads
        WHERE id = ? OR substr(id, 1, ?) = ?
        ORDER BY updated_at DESC
        LIMIT 2
    """
    with sqlite_ro(paths.sqlite_path) as conn:
        rows = conn.execute(sql, (cleaned, len(cleaned), cleaned)).fetchall()
    if len(rows) != 1:
        return None
    row = rows[0]
    return ThreadSummary(
        id=row["id"],
        title=barrier.redact(row["title"] or ""),
        cwd=barrier.redact(row["cwd"] or ""),
        model=redact_display(row["model"]) if row["model"] else None,
        rollout_path=_rollout_path_from_db(row["rollout_path"], paths=paths),
        created_at=_to_datetime(row["created_at"]),
        updated_at=_to_datetime(row["updated_at"]),
        tokens_used=int(row["tokens_used"] or 0),
        archived=bool(row["archived"]),
        git_sha=redact_display(row["git_sha"]),
        git_branch=redact_display(row["git_branch"]),
        source=barrier.redact(row["source"] or ""),
        first_user_message=barrier.redact(row["first_user_message"] or ""),
    )


def iter_session_events(
    rollout_path: str | Path,
    *,
    redact: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield one event dict per rollout JSONL line.

    With ``redact=True`` (the default), every string value reachable from the
    event is replaced by :class:`SecurityBarrier`'s redaction. Callers must
    pass ``redact=False`` explicitly to opt out — and even then, this module
    will never log unredacted content itself.
    """
    barrier = _build_barrier() if redact else None
    for event in iter_jsonl(rollout_path, strict=False):
        if barrier is not None:
            event = barrier.redact_dict(event)
        yield event


def iter_session_events_from_offset(
    rollout_path: str | Path,
    *,
    offset: int,
    redact: bool = True,
) -> Iterator[tuple[dict[str, Any], int]]:
    """Yield redacted events added after ``offset`` plus their raw byte offset.

    ``tail`` needs raw file offsets, not lengths of re-serialized redacted JSON.
    If a live rollout ends with a partial JSON object, stop before that line and
    leave the caller's offset unchanged for the incomplete event.
    """
    barrier = _build_barrier() if redact else None
    path = Path(rollout_path).expanduser()
    with path.open("rb") as handle:
        handle.seek(max(0, offset))
        while True:
            start = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            try:
                line = raw.decode("utf-8").strip()
            except UnicodeDecodeError:
                handle.seek(start)
                break
            if not line:
                yield {}, handle.tell()
                continue
            try:
                event = json.loads(line)
            except ValueError:
                if raw.endswith(b"\n"):
                    yield {}, handle.tell()
                    continue
                # Treat malformed trailing content as an in-progress write.
                handle.seek(start)
                break
            if not isinstance(event, dict):
                yield {}, handle.tell()
                continue
            if barrier is not None:
                event = barrier.redact_dict(event)
            yield event, handle.tell()


def summarize_session(
    rollout_path: str | Path,
    *,
    max_events: int = 2000,
) -> SessionSummary:
    """Walk up to ``max_events`` events from a rollout file and return a summary.

    Counts events by ``type``, counts tool calls by tool name, captures the
    first and last user messages (redacted), and notes the model provider
    seen in the session_meta event when present.
    """
    barrier = _build_barrier()
    event_type_counts: dict[str, int] = {}
    tool_call_counts: dict[str, int] = {}
    first_user: str = ""
    last_user: str = ""
    model_provider: str | None = None
    started_at: datetime | None = None
    last_event_at: datetime | None = None
    scanned = 0
    truncated = False

    for event in iter_jsonl(rollout_path, strict=False):
        if scanned >= max_events:
            truncated = True
            break
        scanned += 1
        event_type = str(event.get("type") or "")
        if event_type:
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1

        timestamp_raw = event.get("timestamp")
        if timestamp_raw:
            try:
                # Codex rollout timestamps are ISO-8601 strings.
                parsed = datetime.fromisoformat(str(timestamp_raw).replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                last_event_at = parsed
                if started_at is None:
                    started_at = parsed
            except ValueError:
                pass

        payload = event.get("payload") or {}
        if isinstance(payload, dict):
            if model_provider is None:
                candidate = payload.get("model_provider")
                if isinstance(candidate, str) and candidate:
                    model_provider = candidate

            role = str(payload.get("role") or "")
            if role == "user":
                text = barrier.redact(_extract_text(payload))
                if text:
                    if not first_user:
                        first_user = text
                    last_user = text

            tool_name = payload.get("tool_name") or payload.get("name")
            tool_call = payload.get("tool_call")
            if not tool_name and isinstance(tool_call, dict):
                tool_name = tool_call.get("name")
            if isinstance(tool_name, str) and tool_name and event_type.endswith("tool_call"):
                tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1

    return SessionSummary(
        rollout_path=Path(rollout_path),
        events_scanned=scanned if not truncated else max_events,
        truncated=truncated,
        event_type_counts=event_type_counts,
        tool_call_counts=tool_call_counts,
        first_user_message=first_user,
        last_user_message=last_user,
        model_provider=model_provider,
        started_at=started_at,
        last_event_at=last_event_at,
    )


# ---------------------------------------------------------------------------
# Lightweight helpers — kept small so the CLI module can stay thin.
# ---------------------------------------------------------------------------


def humanize_ago(when: datetime, *, now: datetime | None = None) -> str:
    """Return a short human-friendly delta like ``3m`` / ``2h`` / ``5d``."""
    now = now or datetime.now(UTC)
    delta = now - when
    seconds = max(0, int(delta.total_seconds()))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"


def truncate(value: str, *, width: int) -> str:
    """Truncate ``value`` to ``width`` characters with a trailing ellipsis if cut."""
    if width <= 0 or len(value) <= width:
        return value
    if width <= 1:
        return value[:width]
    return value[: width - 1] + "…"
