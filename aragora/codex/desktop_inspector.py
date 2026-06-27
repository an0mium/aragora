"""Read-only inspector for Codex Desktop sessions and threads.

Surfaces canonical thread metadata from ``state_5.sqlite`` and full session
transcripts from rollout JSONL files, with mandatory secret redaction on any
surfaced content.

This module is intentionally network-free and writes nothing to ``~/.codex/``.
"""

from __future__ import annotations

import json
import re
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

_PR_RE = re.compile(r"(?:\bPR\s*)?#(\d{2,7})\b", re.IGNORECASE)
_FILE_RE = re.compile(
    r"(?<![\w./-])((?:[\w.-]+/)+[\w.-]+\."
    r"(?:py|md|json|toml|yaml|yml|txt|html|css|js|jsx|ts|tsx|sh|sql))\b"
)
_BRANCH_RE = re.compile(
    r"\b(?:origin/main|main|(?:codex|droid|claude|vision-incubator|worktree|feat|fix|"
    r"chore|docs|review)/[A-Za-z0-9._/-]+)\b"
)

_QUEUE_PRESSURE_OPEN_PR_THRESHOLD = 10
_BROAD_BUILD_KEYWORDS = (
    "build",
    "implement",
    "new pr",
    "open pr",
    "broad",
    "feature",
    "yes to all",
    "ground up assessment",
)
_WATCH_KEYWORDS = ("watch", "ci", "checks", "pending", "head changed", "merge-packet")
_REVIEW_KEYWORDS = ("review", "audit", "read-only", "exact-head review")
_SETTLE_KEYWORDS = ("settle", "merge", "admin squash", "receipt")
_REPAIR_KEYWORDS = ("repair", "fix", "conflict", "rebase")
_PROMPT_CATEGORIES = {"pause", "watch", "review", "settle", "repair", "paste-needed"}


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


@dataclass(frozen=True, slots=True)
class PromptRoute:
    """Conservative prompt-router output for one Codex Desktop session."""

    category: str
    reason: str
    recommended_next_prompt: str

    def to_dict(self) -> dict[str, str]:
        return {
            "category": self.category,
            "reason": self.reason,
            "recommended_next_prompt": self.recommended_next_prompt,
        }


@dataclass(frozen=True, slots=True)
class SessionBrief:
    """Redacted, queue-aware briefing for one Codex Desktop session.

    Raw transcript text is intentionally absent. The brief exposes only
    redacted metadata, bounded summaries, and structured tokens such as PR
    numbers and repo-relative file names.
    """

    id: str
    title: str
    cwd: str
    branch: str | None
    sha: str | None
    rollout_path: Path | None
    updated_at: datetime | None
    age_seconds: int | None
    age: str | None
    pr_mentions: tuple[int, ...]
    files_mentioned: tuple[str, ...]
    branches_mentioned: tuple[str, ...]
    active_lane: dict[str, Any] | None
    conflict_risk: str
    prompt_needed: bool | str
    prompt_needed_reason: str
    last_user_intent_summary: str
    last_assistant_action_summary: str
    current_likely_state: str
    router: PromptRoute
    recent_turns: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "cwd": self.cwd,
            "branch": self.branch,
            "sha": self.sha,
            "rollout_path": redact_display(self.rollout_path) if self.rollout_path else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "age_seconds": self.age_seconds,
            "age": self.age,
            "pr_mentions": list(self.pr_mentions),
            "files_mentioned": list(self.files_mentioned),
            "branches_mentioned": list(self.branches_mentioned),
            "active_lane": self.active_lane,
            "conflict_risk": self.conflict_risk,
            "prompt_needed": self.prompt_needed,
            "prompt_needed_reason": self.prompt_needed_reason,
            "last_user_intent_summary": self.last_user_intent_summary,
            "last_assistant_action_summary": self.last_assistant_action_summary,
            "current_likely_state": self.current_likely_state,
            "router": self.router.to_dict(),
            "recent_turns": list(self.recent_turns),
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


def _unique_sorted_ints(values: list[int]) -> tuple[int, ...]:
    return tuple(sorted(dict.fromkeys(values)))


def _unique_sorted_strings(values: list[str]) -> tuple[str, ...]:
    return tuple(sorted(dict.fromkeys(v for v in values if v)))


def _extract_pr_mentions(text: str) -> list[int]:
    mentions: list[int] = []
    for match in _PR_RE.finditer(text):
        try:
            mentions.append(int(match.group(1)))
        except (TypeError, ValueError):
            continue
    return mentions


def _redact_structured_token(value: str) -> str:
    return _build_barrier().redact(value)


def _extract_file_mentions(text: str) -> list[str]:
    return [_redact_structured_token(match.group(1)) for match in _FILE_RE.finditer(text)]


def _extract_branch_mentions(text: str) -> list[str]:
    return [_redact_structured_token(match.group(0)) for match in _BRANCH_RE.finditer(text)]


def _message_role(payload: Any) -> str:
    if not isinstance(payload, dict):
        return "unknown"
    role = payload.get("role")
    return str(role) if role else "unknown"


def _summarize_intent(
    *,
    role: str,
    text: str,
    pr_mentions: tuple[int, ...],
    file_mentions: tuple[str, ...],
    branch_mentions: tuple[str, ...],
) -> str:
    lower = text.lower()
    parts: list[str] = []
    if any(keyword in lower for keyword in _SETTLE_KEYWORDS):
        parts.append("settlement or merge request")
    if any(keyword in lower for keyword in _WATCH_KEYWORDS):
        parts.append("watch/check-status request")
    if any(keyword in lower for keyword in _REVIEW_KEYWORDS):
        parts.append("review request")
    if any(keyword in lower for keyword in _REPAIR_KEYWORDS):
        parts.append("repair request")
    if any(keyword in lower for keyword in _BROAD_BUILD_KEYWORDS):
        parts.append("build/implementation request")
    if pr_mentions:
        parts.append("mentions " + ", ".join(f"PR #{n}" for n in pr_mentions[:5]))
    if file_mentions:
        parts.append(f"mentions {len(file_mentions)} file path(s)")
    if branch_mentions:
        parts.append(f"mentions {len(branch_mentions)} branch token(s)")
    if parts:
        return f"{role} " + "; ".join(parts)
    if text:
        return f"{role} message present; raw transcript text redacted"
    return f"no {role} message found"


def _recommended_prompt(
    *,
    category: str,
    pr_mentions: tuple[int, ...],
    files_mentioned: tuple[str, ...],
    queue_pressure_high: bool,
) -> str:
    pr_part = ", ".join(f"#{n}" for n in pr_mentions[:4]) or "<target PR>"
    file_part = ", ".join(files_mentioned[:3]) or "<changed files>"
    if category == "pause":
        return (
            "Stop new implementation. Start from live repo truth, report current task state, "
            "open PR pressure, and any active lane overlap; do not edit files or open PRs."
        )
    if category == "watch":
        return (
            f"Start from live repo truth. Watch {pr_part} only: pin current head, report CI/"
            "merge-packet state, and stop without edits, labels, mark-ready, or merge."
        )
    if category == "review":
        return (
            f"Start from live repo truth. Review {pr_part} only at exact head; read changed "
            "files, post or report a concise no-dissent/blocker verdict, and do not mutate "
            "anything else."
        )
    if category == "settle":
        return (
            f"Start from live repo truth. Settle {pr_part} only at exact-head: rerun review-queue "
            "merge-packet, merge only if admin_squash_allowed=true and not_ready=[], record "
            "receipt, and stop on any drift."
        )
    if category == "repair":
        return (
            f"Start from live repo truth. Repair one bounded lane only for {pr_part}; touch "
            f"only {file_part}, validate narrowly, push only the existing branch, and stop on "
            "head drift."
        )
    suffix = " Queue pressure is high, so prefer read-only routing." if queue_pressure_high else ""
    return (
        "Paste the last 2-4 turns or provide the exact session name/PR, because the local "
        f"redacted session record does not contain enough safe signal.{suffix}"
    )


def _route_session(
    *,
    all_text: str,
    has_user_signal: bool,
    has_assistant_signal: bool,
    pr_mentions: tuple[int, ...],
    files_mentioned: tuple[str, ...],
    repo_context: dict[str, Any] | None,
) -> PromptRoute:
    context = repo_context or {}
    try:
        open_pr_count = int(context.get("open_pr_count") or 0)
    except (TypeError, ValueError):
        open_pr_count = 0
    queue_pressure_high = open_pr_count >= _QUEUE_PRESSURE_OPEN_PR_THRESHOLD
    lower = all_text.lower()

    if not has_user_signal and not has_assistant_signal and not pr_mentions and not files_mentioned:
        category = "paste-needed"
        reason = "no safe recent user or assistant signal found"
    elif queue_pressure_high and any(keyword in lower for keyword in _BROAD_BUILD_KEYWORDS):
        category = "pause"
        reason = f"open PR pressure is high ({open_pr_count}) and session looks build-oriented"
    elif any(keyword in lower for keyword in _REPAIR_KEYWORDS):
        category = "repair"
        reason = "recent turns mention repair/conflict/rebase work"
    elif any(keyword in lower for keyword in _SETTLE_KEYWORDS):
        category = "settle"
        reason = "recent turns mention settlement/merge/receipt work"
    elif any(keyword in lower for keyword in _WATCH_KEYWORDS):
        category = "watch"
        reason = "recent turns mention CI/check/watch state"
    elif any(keyword in lower for keyword in _REVIEW_KEYWORDS):
        category = "review"
        reason = "recent turns mention review/audit work"
    elif queue_pressure_high:
        category = "pause"
        reason = f"open PR pressure is high ({open_pr_count}); defaulting to queue-drain posture"
    else:
        category = "paste-needed"
        reason = "redacted local signal is ambiguous"

    if category not in _PROMPT_CATEGORIES:
        category = "paste-needed"
        reason = "router produced an unknown category"
    return PromptRoute(
        category=category,
        reason=reason,
        recommended_next_prompt=_recommended_prompt(
            category=category,
            pr_mentions=pr_mentions,
            files_mentioned=files_mentioned,
            queue_pressure_high=queue_pressure_high,
        ),
    )


def _state_from_route(route: PromptRoute) -> str:
    if route.category == "pause":
        return "likely too broad or duplicate-prone; pause and re-ground"
    if route.category == "watch":
        return "likely waiting on CI/head/merge-packet evidence"
    if route.category == "review":
        return "likely needs exact-head read-only review"
    if route.category == "settle":
        return "likely ready for exact-head settlement gate"
    if route.category == "repair":
        return "likely needs bounded one-PR repair"
    return "insufficient redacted local signal; paste excerpt needed"


def _active_lane_records(repo_context: dict[str, Any] | None) -> list[dict[str, Any]]:
    context = repo_context or {}
    records = context.get("active_lane_records")
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def _redacted_lane_record(record: dict[str, Any]) -> dict[str, Any]:
    keys = ("lane_id", "owner_session", "status", "branch", "worktree", "pr_number", "goal")
    out: dict[str, Any] = {}
    for key in keys:
        value = record.get(key)
        if value in (None, ""):
            continue
        if key in {"branch", "worktree", "goal"}:
            out[key] = redact_display(str(value))
        else:
            out[key] = value
    return out


def _matching_active_lane(
    *,
    branch: str | None,
    cwd: str,
    pr_mentions: tuple[int, ...],
    repo_context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    branch_value = str(branch or "").strip()
    cwd_value = str(cwd or "").strip()
    pr_values = {int(value) for value in pr_mentions}
    for record in _active_lane_records(repo_context):
        lane_branch = str(record.get("branch") or "").strip()
        lane_worktree = str(record.get("worktree") or "").strip()
        lane_pr = record.get("pr_number")
        if branch_value and lane_branch and branch_value == lane_branch:
            return _redacted_lane_record(record)
        if cwd_value and lane_worktree and cwd_value == lane_worktree:
            return _redacted_lane_record(record)
        if isinstance(lane_pr, int) and lane_pr in pr_values:
            return _redacted_lane_record(record)
    return None


def _infer_prompt_needed(
    *,
    route: PromptRoute,
    active_lane: dict[str, Any] | None,
    last_message_role: str,
    has_user_signal: bool,
    has_assistant_signal: bool,
) -> tuple[bool | str, str]:
    if active_lane is not None:
        return False, "active_lane_owned"
    if route.category == "paste-needed":
        return "unknown", "raw_signal_insufficient"
    if last_message_role == "assistant":
        return True, "assistant_final_recent"
    if has_assistant_signal and route.category in {"watch", "review", "settle", "repair"}:
        return True, "pending_operator"
    if last_message_role == "user" and has_user_signal:
        return False, "user_turn_latest"
    if not has_user_signal and not has_assistant_signal:
        return "unknown", "raw_signal_insufficient"
    return "unknown", "raw_signal_insufficient"


def build_session_brief(
    thread: ThreadSummary,
    *,
    include_last_turns: int = 0,
    repo_context: dict[str, Any] | None = None,
    max_events: int = 2000,
) -> SessionBrief:
    """Build a redacted operator briefing for one Codex Desktop thread.

    The scan may read raw rollout JSONL locally to extract safe structured
    tokens (PR numbers, path-like file mentions, branch names), but raw
    transcript text is never returned.
    """
    include_last_turns = max(0, int(include_last_turns))
    pr_values: list[int] = []
    file_values: list[str] = []
    branch_values: list[str] = []
    turn_summaries: list[dict[str, Any]] = []
    all_text_parts: list[str] = []
    last_user_summary = "no user message found"
    last_assistant_summary = "no assistant action found"
    has_user_signal = False
    has_assistant_signal = False
    last_message_role = ""
    tool_names: list[str] = []
    scanned = 0

    for event in iter_jsonl(thread.rollout_path, strict=False):
        if scanned >= max_events:
            break
        scanned += 1
        payload = event.get("payload") or {}
        event_type = str(event.get("type") or "")
        text = _extract_text(payload)
        if text:
            all_text_parts.append(text)
            pr_values.extend(_extract_pr_mentions(text))
            file_values.extend(_extract_file_mentions(text))
            branch_values.extend(_extract_branch_mentions(text))

        if isinstance(payload, dict):
            tool_name = payload.get("tool_name") or payload.get("name")
            tool_call = payload.get("tool_call")
            if not tool_name and isinstance(tool_call, dict):
                tool_name = tool_call.get("name")
            if isinstance(tool_name, str) and tool_name:
                tool_names.append(tool_name)

        if event_type == "agent_message":
            role = _message_role(payload)
            turn_prs = _unique_sorted_ints(_extract_pr_mentions(text))
            turn_files = _unique_sorted_strings(_extract_file_mentions(text))
            turn_branches = _unique_sorted_strings(_extract_branch_mentions(text))
            summary = _summarize_intent(
                role=role,
                text=text,
                pr_mentions=turn_prs,
                file_mentions=turn_files,
                branch_mentions=turn_branches,
            )
            if role == "user":
                has_user_signal = bool(text)
                last_user_summary = summary
                last_message_role = role
            elif role == "assistant":
                has_assistant_signal = bool(text)
                last_assistant_summary = summary
                last_message_role = role
            turn_summaries.append(
                {
                    "timestamp": event.get("timestamp"),
                    "role": role,
                    "summary": summary,
                    "pr_mentions": list(turn_prs),
                    "files_mentioned": list(turn_files),
                    "branches_mentioned": list(turn_branches),
                }
            )

    pr_mentions = _unique_sorted_ints(pr_values)
    files_mentioned = _unique_sorted_strings(file_values)
    branches_mentioned = _unique_sorted_strings(
        branch_values + ([thread.git_branch] if thread.git_branch else [])
    )
    if tool_names and not has_assistant_signal:
        names = ", ".join(sorted(dict.fromkeys(tool_names))[:5])
        last_assistant_summary = f"assistant used tools: {names}"
        has_assistant_signal = True

    all_text = "\n".join(all_text_parts)
    route = _route_session(
        all_text=all_text,
        has_user_signal=has_user_signal,
        has_assistant_signal=has_assistant_signal,
        pr_mentions=pr_mentions,
        files_mentioned=files_mentioned,
        repo_context=repo_context,
    )
    now = datetime.now(UTC)
    age_seconds = max(0, int((now - thread.updated_at).total_seconds()))
    recent_turns = tuple(turn_summaries[-include_last_turns:]) if include_last_turns else ()
    active_lane = _matching_active_lane(
        branch=thread.git_branch,
        cwd=thread.cwd,
        pr_mentions=pr_mentions,
        repo_context=repo_context,
    )
    prompt_needed, prompt_needed_reason = _infer_prompt_needed(
        route=route,
        active_lane=active_lane,
        last_message_role=last_message_role,
        has_user_signal=has_user_signal,
        has_assistant_signal=has_assistant_signal,
    )
    return SessionBrief(
        id=thread.id,
        title=thread.title,
        cwd=thread.cwd,
        branch=thread.git_branch,
        sha=thread.git_sha,
        rollout_path=thread.rollout_path,
        updated_at=thread.updated_at,
        age_seconds=age_seconds,
        age=humanize_ago(thread.updated_at, now=now),
        pr_mentions=pr_mentions,
        files_mentioned=files_mentioned,
        branches_mentioned=branches_mentioned,
        active_lane=active_lane,
        conflict_risk="active-lane-overlap" if active_lane else "none",
        prompt_needed=prompt_needed,
        prompt_needed_reason=prompt_needed_reason,
        last_user_intent_summary=last_user_summary,
        last_assistant_action_summary=last_assistant_summary,
        current_likely_state=_state_from_route(route),
        router=route,
        recent_turns=recent_turns,
    )


def paste_needed_brief(session_id: str) -> SessionBrief:
    """Return a synthetic brief for a requested session not visible locally."""
    route = PromptRoute(
        category="paste-needed",
        reason="session id/prefix is not visible in Codex Desktop local state",
        recommended_next_prompt=_recommended_prompt(
            category="paste-needed",
            pr_mentions=(),
            files_mentioned=(),
            queue_pressure_high=False,
        ),
    )
    return SessionBrief(
        id=session_id,
        title="(session not found)",
        cwd="",
        branch=None,
        sha=None,
        rollout_path=None,
        updated_at=None,
        age_seconds=None,
        age=None,
        pr_mentions=(),
        files_mentioned=(),
        branches_mentioned=(),
        active_lane=None,
        conflict_risk="unknown",
        prompt_needed="unknown",
        prompt_needed_reason="raw_signal_insufficient",
        last_user_intent_summary="session not visible",
        last_assistant_action_summary="session not visible",
        current_likely_state=_state_from_route(route),
        router=route,
        recent_turns=(),
    )


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
