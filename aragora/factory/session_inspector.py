"""Read-only, redacted Factory/Droid session metadata inspector.

The inspector intentionally uses only metadata files: ``sessions-index.json``,
tmux ``*.meta.json`` records, lane registry rows, and git metadata for existing
worktrees. It does not read Factory history, raw transcript files, prompt logs,
or agent logs by default.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from aragora.debate.security_barrier import SecurityBarrier

_EXTRA_REDACTION_PATTERNS = (
    r"gh[pousr]_[A-Za-z0-9_]{20,}",
    r"github_pat_[A-Za-z0-9_]{22,}",
    r"AKIA[0-9A-Z]{16}",
    r"sk-or-v1-[A-Za-z0-9]+",
)

_PR_RE = re.compile(r"(?:\bPR\s*)?#(\d{2,7})\b", re.IGNORECASE)
_BRANCH_RE = re.compile(
    r"\b(?:origin/main|main|(?:codex|droid|claude|vision-incubator|worktree|feat|fix|"
    r"chore|docs|review)/[A-Za-z0-9._/-]+)\b"
)
_SAFE_TMUX_TARGET_RE = re.compile(r"^[A-Za-z0-9_.:@-]{1,128}$")
_ACTIVE_STATUSES = {"active", "running", "pending", "queued", "claimed", "conflict"}

DEFAULT_FACTORY_HOME = Path("~/.factory")
DEFAULT_LIMIT = 50


@dataclass(frozen=True, slots=True)
class PromptRoute:
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
class FactorySessionBrief:
    provider: str
    session_id: str
    cwd: str | None
    worktree: str | None
    branch: str | None
    head: str | None
    updated_at: datetime | None
    age_seconds: int | None
    age: str | None
    pr_number: int | None
    matched_lane: dict[str, Any] | None
    conflict_risk: str
    prompt_needed: bool | str
    prompt_needed_reason: str
    router: PromptRoute
    source_types: tuple[str, ...]
    direct_steering_available: bool
    steering_command: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "session_id": self.session_id,
            "cwd": self.cwd,
            "worktree": self.worktree,
            "branch": self.branch,
            "head": self.head,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "age_seconds": self.age_seconds,
            "age": self.age,
            "pr_number": self.pr_number,
            "matched_lane": self.matched_lane,
            "conflict_risk": self.conflict_risk,
            "prompt_needed": self.prompt_needed,
            "prompt_needed_reason": self.prompt_needed_reason,
            "router": self.router.to_dict(),
            "source_types": list(self.source_types),
            "direct_steering_available": self.direct_steering_available,
            "steering_command": self.steering_command,
        }


@dataclass(frozen=True, slots=True)
class _FactorySessionRecord:
    session_id: str
    cwd: str | None
    branch: str | None
    head: str | None
    pr_number: int | None
    updated_at: datetime | None
    search_text: str
    source_types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _TmuxTarget:
    name: str
    cwd: str | None


def _build_barrier() -> SecurityBarrier:
    barrier = SecurityBarrier()
    for pattern in _EXTRA_REDACTION_PATTERNS:
        barrier.add_pattern(pattern)
    return barrier


def redact_display(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return _build_barrier().redact(str(value))


def _safe_tmux_target_name(value: str | None) -> str | None:
    name = str(value or "").strip()
    if not name:
        return None
    redacted = redact_display(name)
    if not redacted or redacted != name:
        return None
    if not _SAFE_TMUX_TARGET_RE.fullmatch(name):
        return None
    return name


def _redact_lane(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "lane_id": redact_display(record.get("lane_id")),
            "owner_session": redact_display(record.get("owner_session")),
            "status": redact_display(record.get("status")),
            "pr_number": record.get("pr_number"),
            "branch": redact_display(record.get("branch")),
            "worktree": redact_display(record.get("worktree")),
            "updated_at": redact_display(record.get("updated_at")),
        }.items()
        if value not in (None, "")
    }


def _to_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if number > 32503680000:
            number = number / 1000.0
        return datetime.fromtimestamp(number, tz=UTC)
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.isdigit():
            return _to_datetime(int(text))
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except ValueError:
        return None


def humanize_ago(value: datetime | None, *, now: datetime | None = None) -> str | None:
    if value is None:
        return None
    now = now or datetime.now(UTC)
    seconds = max(0, int((now - value).total_seconds()))
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"


def _extract_pr_number(*values: Any) -> int | None:
    for value in values:
        if value in (None, ""):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        matches = _PR_RE.findall(str(value))
        if matches:
            return int(matches[0])
    return None


def _extract_branch(*values: Any) -> str | None:
    for value in values:
        if not isinstance(value, str):
            continue
        matches = _BRANCH_RE.findall(value)
        if matches:
            return redact_display(matches[0])
    return None


def _load_json_file(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _index_entries(factory_home: Path) -> list[dict[str, Any]]:
    payload = _load_json_file(factory_home / "sessions-index.json")
    if isinstance(payload, dict):
        entries = payload.get("entries") or payload.get("sessions") or []
    elif isinstance(payload, list):
        entries = payload
    else:
        entries = []
    return [entry for entry in entries if isinstance(entry, dict)]


def _git_metadata(cwd: str | None) -> tuple[str | None, str | None]:
    if not cwd:
        return None, None
    path = Path(cwd).expanduser()
    if not path.exists() or not path.is_dir():
        return None, None
    try:
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=str(path),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        head_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(path),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    branch = branch_result.stdout.strip() if branch_result.returncode == 0 else None
    head = head_result.stdout.strip() if head_result.returncode == 0 else None
    if head and not re.fullmatch(r"[0-9a-f]{40}", head):
        head = None
    return redact_display(branch), head


def _enrich_selected_records_with_git(
    records: Sequence[_FactorySessionRecord],
) -> list[_FactorySessionRecord]:
    enriched: list[_FactorySessionRecord] = []
    for record in records:
        if (record.branch and record.head) or not record.cwd:
            enriched.append(record)
            continue
        git_branch, git_head = _git_metadata(record.cwd)
        enriched.append(
            replace(
                record,
                branch=record.branch or git_branch,
                head=record.head or git_head,
            )
        )
    return enriched


def _session_records(factory_home: Path, *, limit: int) -> list[_FactorySessionRecord]:
    records: list[_FactorySessionRecord] = []
    for entry in _index_entries(factory_home):
        session_id = str(
            entry.get("sessionId")
            or entry.get("session_id")
            or entry.get("id")
            or entry.get("name")
            or ""
        ).strip()
        if not session_id:
            continue
        cwd_raw = entry.get("cwd") or entry.get("worktree")
        cwd = str(cwd_raw) if cwd_raw else None
        title = str(entry.get("title") or "")
        raw_tags = entry.get("tags")
        tags: list[Any] = raw_tags if isinstance(raw_tags, list) else []
        search_text = " ".join(
            str(value)
            for value in (
                session_id,
                cwd or "",
                title,
                entry.get("branch") or "",
                " ".join(str(tag) for tag in tags),
            )
            if value
        )
        branch = redact_display(entry.get("branch")) or _extract_branch(search_text)
        pr_number = _extract_pr_number(
            entry.get("pr_number"),
            entry.get("prNumber"),
            entry.get("pr"),
            search_text,
        )
        head_value = entry.get("head") or entry.get("sha") or entry.get("git_sha")
        head = str(head_value) if isinstance(head_value, str) else None
        if head and not re.fullmatch(r"[0-9a-f]{40}", head):
            head = None
        updated_at = _to_datetime(
            entry.get("mtime") or entry.get("updated_at") or entry.get("updatedAt")
        )
        records.append(
            _FactorySessionRecord(
                session_id=redact_display(session_id) or session_id,
                cwd=redact_display(cwd),
                branch=branch,
                head=head,
                pr_number=pr_number,
                updated_at=updated_at,
                search_text=search_text,
                source_types=("factory_sessions_index",),
            )
        )
    records.sort(
        key=lambda record: record.updated_at or datetime.fromtimestamp(0, tz=UTC), reverse=True
    )
    selected = records[:limit] if limit > 0 else records
    return _enrich_selected_records_with_git(selected)


def _active_lane_records(repo_root: Path | None) -> list[dict[str, Any]]:
    if repo_root is None:
        return []
    registry = repo_root / ".aragora" / "agent-bridge" / "lanes.json"
    payload = _load_json_file(registry)
    if isinstance(payload, dict):
        rows = payload.get("lanes") or payload.get("records") or []
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "").lower()
        if status not in _ACTIVE_STATUSES:
            continue
        out.append(row)
    return out


def _tmux_targets(repo_root: Path | None) -> list[_TmuxTarget]:
    if repo_root is None:
        return []
    tmux_dir = repo_root / ".aragora" / "tmux-sessions"
    if not tmux_dir.exists():
        tmux_dir = Path("~/.aragora/tmux-sessions").expanduser()
    targets: list[_TmuxTarget] = []
    for meta_path in sorted(tmux_dir.glob("*.meta.json")):
        payload = _load_json_file(meta_path)
        if not isinstance(payload, dict):
            continue
        agent = str(payload.get("agent") or "").lower()
        if agent not in {"droid", "factory"}:
            continue
        name = _safe_tmux_target_name(str(payload.get("name") or ""))
        if not name:
            continue
        cwd = str(payload.get("worktree") or payload.get("cwd") or "") or None
        targets.append(_TmuxTarget(name=name, cwd=redact_display(cwd)))
    return targets


def _same_owner(record: _FactorySessionRecord, lane: dict[str, Any]) -> bool:
    owner = str(lane.get("owner_session") or "")
    return bool(owner and (owner == record.session_id or owner.startswith(record.session_id)))


def _lane_matches(record: _FactorySessionRecord, lane: dict[str, Any]) -> bool:
    if _same_owner(record, lane):
        return True
    if record.pr_number is not None and lane.get("pr_number") == record.pr_number:
        return True
    if record.branch and lane.get("branch") == record.branch:
        return True
    if record.cwd and lane.get("worktree") == record.cwd:
        return True
    return False


def _matched_lane(
    record: _FactorySessionRecord, lanes: Sequence[dict[str, Any]]
) -> dict[str, Any] | None:
    owner_match = next((lane for lane in lanes if _same_owner(record, lane)), None)
    if owner_match is not None:
        return owner_match
    return next((lane for lane in lanes if _lane_matches(record, lane)), None)


def _tmux_target(
    record: _FactorySessionRecord,
    targets: Sequence[_TmuxTarget],
    lane: dict[str, Any] | None,
) -> _TmuxTarget | None:
    lane_owner = str(lane.get("owner_session") or "") if lane else ""
    for target in targets:
        if target.name == record.session_id:
            return target
        if lane_owner and target.name == lane_owner:
            return target
    return None


def _steering_prompt(record: _FactorySessionRecord, lane: dict[str, Any] | None) -> str:
    if lane and not _same_owner(record, lane):
        owner = redact_display(lane.get("owner_session")) or "the active owner"
        lane_id = redact_display(lane.get("lane_id")) or "the active lane"
        return (
            f"Do not edit, push, comment, mark-ready, merge, or label. "
            f"Another active owner appears to own this work: {owner} / {lane_id}. "
            "Report your local head and stop unless the operator explicitly reassigns ownership."
        )
    if lane:
        return (
            "Continue only if your local worktree/head still matches this active lane. "
            "Re-check live PR truth and report a bounded next action."
        )
    return (
        "Factory/Droid session is visible only through redacted metadata. "
        "Paste the relevant Factory excerpt for exact conversational advice, or assign a bounded lane."
    )


def _route(record: _FactorySessionRecord, lane: dict[str, Any] | None) -> PromptRoute:
    if lane and not _same_owner(record, lane):
        return PromptRoute(
            category="pause",
            reason="another active lane owns the same PR, branch, or worktree",
            recommended_next_prompt=_steering_prompt(record, lane),
        )
    if lane:
        return PromptRoute(
            category="watch",
            reason="session appears to own an active lane",
            recommended_next_prompt=_steering_prompt(record, lane),
        )
    if record.pr_number is not None:
        return PromptRoute(
            category="review",
            reason="session metadata mentions a PR but no active lane owns it",
            recommended_next_prompt=(
                f"Start from live repo truth. Duplicate-owner detection first. "
                f"Review or watch PR #{record.pr_number} only; do not mutate unless you claim a lane."
            ),
        )
    return PromptRoute(
        category="paste-needed",
        reason="metadata is insufficient to infer the latest conversational state",
        recommended_next_prompt=_steering_prompt(record, None),
    )


def _brief_from_record(
    record: _FactorySessionRecord,
    *,
    lanes: Sequence[dict[str, Any]],
    targets: Sequence[_TmuxTarget],
    now: datetime,
) -> FactorySessionBrief:
    lane = _matched_lane(record, lanes)
    route = _route(record, lane)
    same_owner = bool(lane and _same_owner(record, lane))
    conflict_risk = "active-owner-overlap" if lane and not same_owner else "none"
    target = _tmux_target(record, targets, lane)
    direct_steering_available = target is not None
    steering_command = None
    if target is not None:
        steering_command = (
            "python3 scripts/agent_bridge.py send "
            f"{shlex.quote(target.name)} {shlex.quote(route.recommended_next_prompt)}"
        )
    age_seconds = (
        max(0, int((now - record.updated_at).total_seconds())) if record.updated_at else None
    )
    return FactorySessionBrief(
        provider="factory",
        session_id=record.session_id,
        cwd=record.cwd,
        worktree=record.cwd,
        branch=record.branch,
        head=record.head,
        updated_at=record.updated_at,
        age_seconds=age_seconds,
        age=humanize_ago(record.updated_at, now=now),
        pr_number=record.pr_number or (lane.get("pr_number") if lane else None),
        matched_lane=_redact_lane(lane) if lane else None,
        conflict_risk=conflict_risk,
        prompt_needed=False if same_owner else True,
        prompt_needed_reason="active_lane_owned" if same_owner else route.reason,
        router=route,
        source_types=tuple(
            dict.fromkeys((*record.source_types, *(("agent_bridge_lanes",) if lanes else ())))
        ),
        direct_steering_available=direct_steering_available,
        steering_command=steering_command,
    )


def paste_needed_brief(session_id: str) -> FactorySessionBrief:
    route = PromptRoute(
        category="paste-needed",
        reason="session id/prefix is not visible in Factory local metadata",
        recommended_next_prompt=(
            "Paste the relevant Factory excerpt, or provide a Factory session id/worktree "
            "that appears in sessions-index.json."
        ),
    )
    return FactorySessionBrief(
        provider="factory",
        session_id=redact_display(session_id) or session_id,
        cwd=None,
        worktree=None,
        branch=None,
        head=None,
        updated_at=None,
        age_seconds=None,
        age=None,
        pr_number=None,
        matched_lane=None,
        conflict_risk="unknown",
        prompt_needed="unknown",
        prompt_needed_reason="raw_signal_insufficient",
        router=route,
        source_types=(),
        direct_steering_available=False,
        steering_command=None,
    )


def _filter_since(
    records: Iterable[_FactorySessionRecord], since: timedelta
) -> list[_FactorySessionRecord]:
    cutoff = datetime.now(UTC) - since
    return [
        record for record in records if record.updated_at is None or record.updated_at >= cutoff
    ]


def _filter_session(
    records: Sequence[_FactorySessionRecord], session: str | None
) -> list[_FactorySessionRecord]:
    if not session:
        return list(records)
    return [
        record
        for record in records
        if record.session_id == session or record.session_id.startswith(session)
    ]


def build_factory_session_briefs(
    *,
    factory_home: str | Path | None = None,
    repo_root: str | Path | None = None,
    since: timedelta = timedelta(hours=4),
    limit: int = DEFAULT_LIMIT,
    session: str | None = None,
) -> list[FactorySessionBrief]:
    home = (
        Path(factory_home).expanduser()
        if factory_home is not None
        else DEFAULT_FACTORY_HOME.expanduser()
    )
    repo = Path(repo_root).expanduser() if repo_root is not None else None
    records = _filter_since(_session_records(home, limit=limit), since)
    records = _filter_session(records, session)
    if session and not records:
        return [paste_needed_brief(session)]
    lanes = _active_lane_records(repo)
    targets = _tmux_targets(repo)
    now = datetime.now(UTC)
    return [_brief_from_record(record, lanes=lanes, targets=targets, now=now) for record in records]
