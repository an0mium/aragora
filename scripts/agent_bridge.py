#!/usr/bin/env python3
"""Agent bridge: action commands for cross-agent orchestration.

Provides send, approve, read, and lanes commands on top of the session
inventory from agent_bridge_sessions.py (PR #5306).

Usage:
  python3 scripts/agent_bridge.py sessions [--json]
  python3 scripts/agent_bridge.py launch --name codex-review --agent codex --cwd .worktrees/review --file /tmp/prompt.md
  python3 scripts/agent_bridge.py exec --agent droid --auto high --cwd . --file /tmp/prompt.md
  python3 scripts/agent_bridge.py send <name> "Fix the LOC ratchet"
  python3 scripts/agent_bridge.py send <name> --file /tmp/prompt.md
  python3 scripts/agent_bridge.py approve <name>
  python3 scripts/agent_bridge.py read <name> [--lines 20]
  python3 scripts/agent_bridge.py read-all [--lines 3] [--json]
  python3 scripts/agent_bridge.py lanes [--json]
  python3 scripts/agent_bridge.py owner --pr 7292 [--json]
  python3 scripts/agent_bridge.py processes [--json]
  python3 scripts/agent_bridge.py tmux-map
  python3 scripts/agent_bridge.py health [--json]
  python3 scripts/agent_bridge.py operator-snapshot [--json] [--summary-only]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    # When run as `python3 scripts/agent_bridge.py`, Python adds scripts/ to
    # sys.path automatically, so this direct import works.  For package-style
    # imports (e.g. `import scripts.agent_bridge`) or stale worktrees where
    # agent_bridge_sessions.py may not exist, fall back gracefully.
    import agent_bridge_sessions  # type: ignore[import-not-found]
except ModuleNotFoundError:
    _scripts_dir = str(Path(__file__).resolve().parent)
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    try:
        import agent_bridge_sessions  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        agent_bridge_sessions = None  # type: ignore[assignment]

AGENT_BRIDGE_DIR = Path.home() / ".aragora" / "agent-bridge"
SESSION_SNAPSHOT_FILE = AGENT_BRIDGE_DIR / "sessions.json"
LANE_REGISTRY_FILE = AGENT_BRIDGE_DIR / "lanes.json"
USER_HEARTBEATS_FILE = AGENT_BRIDGE_DIR / "heartbeats.json"
TMUX_SESSIONS_DIR = Path.home() / ".aragora" / "tmux-sessions"
TMUX_SESSION = "aragora"
HEARTBEATS_FILE = REPO_ROOT / ".aragora" / "agent-bridge" / "heartbeats.json"
CANONICAL_REPO_ROOT = REPO_ROOT
if agent_bridge_sessions is not None:
    try:
        CANONICAL_REPO_ROOT = agent_bridge_sessions.resolve_canonical_repo_root(REPO_ROOT)
    except (OSError, RuntimeError, ValueError):
        CANONICAL_REPO_ROOT = REPO_ROOT
CONFLICT_LANE_STATUSES = {"conflict"}
ACTIVE_LANE_STATUSES = {
    "active",
    "running",
    "pending",
    "queued",
    "claimed",
    "waiting_for_steering",
    "acknowledged",
    "working",
    "blocked",
}
COMPLETED_LANE_STATUSES = {"completed", "released", "superseded"}
CURRENT_SESSION_LIFECYCLES = {"live", "active_broker"}
HISTORICAL_SESSION_LIFECYCLES = {"historical", "dead", "stale", "orphaned"}
DEFAULT_STALE_TTL_HOURS = 24
HEARTBEAT_FRESH_SECONDS = 15 * 60
DEFAULT_ACTIVE_NEXT_ACTION = "unspecified active lane action"
DEFAULT_STEERING_OUTCOME = "unknown"
RESOLVED_STEERING_OUTCOMES = {
    "obeyed",
    "stale",
    "superseded",
    "completed",
}
DEFAULT_B0_SCORECARD_TIMEOUT_SECONDS = 5.0


class _LazyTransportError(Exception):
    """Placeholder so tests can monkeypatch create_transport without eager imports."""


TransportError: type[Exception] = _LazyTransportError
create_transport: Any | None = None


def _load_transport_runtime() -> tuple[type[Exception], Any]:
    global TransportError, create_transport
    if create_transport is None:
        from aragora.swarm.agent_bridge.exceptions import TransportError as bridge_error
        from aragora.swarm.agent_bridge.harnesses import create_transport as bridge_transport

        TransportError = bridge_error
        create_transport = bridge_transport
    return TransportError, create_transport


def _state_root_bridge_dir() -> Path:
    configured = os.environ.get("ARAGORA_AUTOMATION_STATE_ROOT")
    if configured:
        root = Path(configured).expanduser()
        state_dir = root if root.name == ".aragora" else root / ".aragora"
        return state_dir / "agent-bridge"
    return CANONICAL_REPO_ROOT / ".aragora" / "agent-bridge"


def _assert_writable_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".write-test"
    probe.write_text("", encoding="utf-8")
    probe.unlink(missing_ok=True)


def _bridge_file_for_read(default_path: Path) -> Path:
    if default_path.exists():
        return default_path
    fallback_path = _state_root_bridge_dir() / default_path.name
    if fallback_path.exists():
        return fallback_path
    return default_path


def _heartbeat_file_for_read() -> Path:
    repo_path = HEARTBEATS_FILE
    if repo_path.exists():
        return repo_path
    fallback_path = _state_root_bridge_dir() / repo_path.name
    if fallback_path.exists():
        return fallback_path
    if USER_HEARTBEATS_FILE.exists():
        return USER_HEARTBEATS_FILE
    return repo_path


def _bridge_files_for_lane_read() -> list[Path]:
    """Return all lane-registry locations that can contain live claims."""
    paths: list[Path] = []
    seen: set[Path] = set()
    for path in (LANE_REGISTRY_FILE, _state_root_bridge_dir() / LANE_REGISTRY_FILE.name):
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists():
            paths.append(path)
    return paths or [LANE_REGISTRY_FILE]


def _bridge_file_for_write(default_path: Path) -> Path:
    try:
        _assert_writable_dir(default_path.parent)
        return default_path
    except PermissionError:
        if os.environ.get("ARAGORA_AGENT_BRIDGE_DIR"):
            raise
        fallback_dir = _state_root_bridge_dir()
        _assert_writable_dir(fallback_dir)
        return fallback_dir / default_path.name


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


@dataclass
class Session:
    name: str
    agent: str
    status: str = "unknown"
    source: str = ""
    lifecycle: str = ""
    tmux_target: str = ""
    branch: str = ""
    worktree: str = ""
    session_id: str = ""
    updated_at: str = ""
    summary: str = ""
    log_file: str = ""
    transcript_file: str = ""
    pr_number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v}


@dataclass
class LaneRecord:
    lane_id: str
    owner_session: str
    goal: str = ""
    source: str = ""
    status: str = "active"
    next_action: str = ""
    updated_at: str = ""
    branch: str = ""
    worktree: str = ""
    pr_number: int | None = None
    conflict_session: str = ""
    conflict_reason: str = ""
    desktop_label: str = ""
    codex_thread_id: str = ""
    codex_rollout_path: str = ""
    session_title: str = ""
    contact_method: str = ""
    contact_payload: dict[str, Any] | None = None
    last_mailbox_check_at: str = ""
    last_delivery_at: str = ""
    last_ack_at: str = ""
    last_heartbeat_at: str = ""
    last_steering_outcome: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in ("", None)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LaneRecord":
        return cls(
            lane_id=str(payload.get("lane_id", "")),
            owner_session=str(payload.get("owner_session", "")),
            goal=str(payload.get("goal", "")),
            source=str(payload.get("source", "")),
            status=str(payload.get("status", "active")),
            next_action=str(payload.get("next_action", "")),
            updated_at=str(payload.get("updated_at", "")),
            branch=str(payload.get("branch", "")),
            worktree=str(payload.get("worktree", "")),
            pr_number=payload.get("pr_number"),
            conflict_session=str(payload.get("conflict_session", "")),
            conflict_reason=str(payload.get("conflict_reason", "")),
            desktop_label=str(payload.get("desktop_label", "")),
            codex_thread_id=str(payload.get("codex_thread_id", "")),
            codex_rollout_path=str(payload.get("codex_rollout_path", "")),
            session_title=str(payload.get("session_title", "")),
            contact_method=str(payload.get("contact_method", "")),
            contact_payload=payload.get("contact_payload")
            if isinstance(payload.get("contact_payload"), dict)
            else None,
            last_mailbox_check_at=str(payload.get("last_mailbox_check_at", "")),
            last_delivery_at=str(payload.get("last_delivery_at", "")),
            last_ack_at=str(payload.get("last_ack_at", "")),
            last_heartbeat_at=str(payload.get("last_heartbeat_at", "")),
            last_steering_outcome=str(payload.get("last_steering_outcome", "")),
        )


def discover(
    *,
    include_summaries: bool = True,
    include_historical: bool = True,
    active_broker_session_ids: set[str] | None = None,
) -> list[Session]:
    """Discover all sessions via agent_bridge_sessions.

    Falls back to minimal tmux-only discovery if agent_bridge_sessions
    is unavailable (stale worktree, package-style import, etc.).
    """
    if agent_bridge_sessions is not None:
        records = agent_bridge_sessions.collect_sessions(
            repo_root=REPO_ROOT,
            tmux_dir=TMUX_SESSIONS_DIR,
            claude_projects_root=Path.home() / ".claude" / "projects",
            include_summaries=include_summaries,
        )
        sessions: list[Session] = []
        for r in records:
            tmux_target = ""
            if r.status == "alive" and r.source == "tmux":
                tmux_target = f"{TMUX_SESSION}:{r.name}"
            lifecycle = _session_lifecycle(
                source=r.source,
                status=r.status,
                updated_at=r.updated_at,
                active_broker_session_ids=active_broker_session_ids,
                session_id=r.session_id,
            )
            if not include_historical and lifecycle not in CURRENT_SESSION_LIFECYCLES:
                continue
            sessions.append(
                Session(
                    name=r.name,
                    agent=r.agent,
                    status=_session_status_for_lifecycle(r.status, lifecycle),
                    source=r.source,
                    lifecycle=lifecycle,
                    tmux_target=tmux_target,
                    branch=r.branch or "",
                    worktree=r.cwd or "",
                    session_id=r.session_id,
                    updated_at=r.updated_at or "",
                    summary=r.summary or "",
                    log_file=r.log_file or "",
                    transcript_file=r.transcript_file or "",
                )
            )
        return sessions

    # Fallback: minimal tmux-only discovery
    sessions = _discover_tmux_fallback()
    if include_historical:
        return sessions
    return [session for session in sessions if _is_current_session(session)]


def _discover_with_broker_state(
    *,
    include_summaries: bool = True,
    include_historical: bool = True,
    broker_runs: list[dict[str, Any]] | None = None,
) -> tuple[list[Session], list[dict[str, Any]], set[str]]:
    runs = _load_broker_run_summaries() if broker_runs is None else broker_runs
    active_broker_ids = _active_broker_session_ids(runs)
    try:
        sessions = discover(
            include_summaries=include_summaries,
            include_historical=include_historical,
            active_broker_session_ids=active_broker_ids,
        )
    except TypeError:
        # Compatibility for tests or older in-process callers that monkeypatch discover().
        sessions = discover()
    return sessions, runs, active_broker_ids


def _discover_tmux_fallback() -> list[Session]:
    """Minimal fallback when agent_bridge_sessions is not available."""
    sessions: list[Session] = []
    if not TMUX_SESSIONS_DIR.exists():
        return sessions
    alive: set[str] = set()
    try:
        result = subprocess.run(
            ["tmux", "list-windows", "-t", TMUX_SESSION, "-F", "#{window_name}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            alive = set(result.stdout.strip().splitlines())
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    for meta_file in TMUX_SESSIONS_DIR.glob("*.meta.json"):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        name = meta.get("name", meta_file.stem)
        is_alive = name in alive
        status = "alive" if is_alive else "dead"
        lifecycle = _session_lifecycle(source="tmux", status=status, updated_at=None)
        sessions.append(
            Session(
                name=name,
                agent=meta.get("agent", "unknown"),
                status=_session_status_for_lifecycle(status, lifecycle),
                source="tmux",
                lifecycle=lifecycle,
                tmux_target=f"{TMUX_SESSION}:{name}" if is_alive else "",
            )
        )
    return sessions


def _write_session_snapshot(sessions: list[Session]) -> None:
    timestamp = datetime.now(UTC).isoformat()
    snapshot = [{"timestamp": timestamp, **s.to_dict()} for s in sessions]
    snapshot_file = _bridge_file_for_write(SESSION_SNAPSHOT_FILE)
    _atomic_write_json(snapshot_file, snapshot)


def _filter_current_sessions(sessions: list[Session]) -> list[Session]:
    return [session for session in sessions if _is_current_session(session)]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except ValueError:
        return None


def _is_older_than(value: str | None, *, hours: int) -> bool:
    parsed = _parse_timestamp(value)
    if parsed is None:
        return False
    return parsed < datetime.now(UTC) - timedelta(hours=hours)


def _session_lifecycle(
    *,
    source: str,
    status: str,
    updated_at: str | None,
    active_broker_session_ids: set[str] | None = None,
    session_id: str = "",
    ttl_hours: int = DEFAULT_STALE_TTL_HOURS,
) -> str:
    active_broker_session_ids = active_broker_session_ids or set()
    if session_id and session_id in active_broker_session_ids:
        return "active_broker"
    if source == "tmux":
        if status == "alive":
            return "live"
        if status == "dead":
            return "stale" if _is_older_than(updated_at, hours=ttl_hours) else "dead"
        return "unknown"
    if source == "claude_jsonl":
        return "historical"
    if status == "alive":
        return "live"
    if status == "dead":
        return "dead"
    return "unknown"


def _session_status_for_lifecycle(status: str, lifecycle: str) -> str:
    if lifecycle in {"historical", "stale", "orphaned", "active_broker", "live"}:
        return lifecycle
    return status


def _is_current_session(session: Session) -> bool:
    lifecycle = session.lifecycle or _session_lifecycle(
        source=session.source,
        status=session.status,
        updated_at=session.updated_at,
        session_id=session.session_id,
    )
    return lifecycle in CURRENT_SESSION_LIFECYCLES or session.status == "alive"


def _load_lane_registry() -> list[LaneRecord]:
    merged: dict[str, tuple[LaneRecord, int]] = {}
    anonymous: list[LaneRecord] = []

    for source_index, registry_file in enumerate(_bridge_files_for_lane_read()):
        if not registry_file.exists():
            continue
        try:
            payload = json.loads(registry_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            record = LaneRecord.from_dict(item)
            if not record.lane_id:
                anonymous.append(record)
                continue
            current = merged.get(record.lane_id)
            if current is None:
                merged[record.lane_id] = (record, source_index)
            elif _prefer_lane_record(record, source_index, current):
                merged[record.lane_id] = (
                    _fill_sparse_lane_identity(record, current[0]),
                    source_index,
                )
            else:
                merged[record.lane_id] = (
                    _fill_sparse_lane_identity(current[0], record),
                    current[1],
                )

    return anonymous + [record for record, _source_index in merged.values()]


def _prefer_lane_record(
    candidate: LaneRecord,
    candidate_source_index: int,
    current: tuple[LaneRecord, int],
) -> bool:
    current_record, current_source_index = current
    candidate_ts = _parse_timestamp(candidate.updated_at)
    current_ts = _parse_timestamp(current_record.updated_at)
    if candidate_ts is not None and current_ts is not None and candidate_ts != current_ts:
        return candidate_ts > current_ts
    # Later sources are repo-local fallbacks; prefer them when timestamps are
    # missing or tied so claim_active_agent_lane.py writes cannot be shadowed by
    # stale user-level bridge state.
    return candidate_source_index >= current_source_index


def _fill_sparse_lane_identity(preferred: LaneRecord, fallback: LaneRecord) -> LaneRecord:
    if not preferred.branch:
        preferred.branch = fallback.branch
    if not preferred.worktree:
        preferred.worktree = fallback.worktree
    if preferred.pr_number is None:
        preferred.pr_number = fallback.pr_number
    return preferred


def _write_lane_registry(records: list[LaneRecord]) -> None:
    registry_file = _bridge_file_for_write(LANE_REGISTRY_FILE)
    _atomic_write_json(registry_file, [record.to_dict() for record in records])


def _find_lane_record(records: list[LaneRecord], lane_id: str) -> LaneRecord | None:
    for record in records:
        if record.lane_id == lane_id:
            return record
    return None


def _sync_lane_records(records: list[LaneRecord], sessions: list[Session]) -> list[LaneRecord]:
    session_map = {session.name: session for session in sessions}
    for record in records:
        live = session_map.get(record.owner_session)
        if live is not None:
            if live.branch:
                record.branch = live.branch
            if live.worktree:
                record.worktree = live.worktree
            if live.pr_number is not None:
                record.pr_number = live.pr_number
    return records


def _conflict_lane_resolved_by_completed_owner(
    record: LaneRecord, records: list[LaneRecord]
) -> bool:
    if record.status not in CONFLICT_LANE_STATUSES or not record.conflict_session:
        return False
    conflict_updated_at = _parse_timestamp(record.updated_at)
    if conflict_updated_at is None:
        return False
    for candidate in records:
        if candidate.owner_session != record.conflict_session:
            continue
        if candidate.status not in COMPLETED_LANE_STATUSES:
            continue
        candidate_updated_at = _parse_timestamp(candidate.updated_at)
        if candidate_updated_at is not None and candidate_updated_at >= conflict_updated_at:
            return True
    return False


def _filter_current_lane_records(records: list[LaneRecord]) -> list[LaneRecord]:
    return [
        record
        for record in records
        if record.status in ACTIVE_LANE_STATUSES
        or (
            record.status in CONFLICT_LANE_STATUSES
            and not _conflict_lane_resolved_by_completed_owner(record, records)
        )
    ]


def _head_for_worktree(path: str | Path | None) -> str | None:
    if not path:
        return None
    worktree = Path(path)
    if not worktree.is_dir():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None
    head = result.stdout.strip()
    return head or None


def _worktree_matches(record_worktree: str, query_worktree: str | None) -> bool:
    if not query_worktree:
        return False
    if record_worktree == query_worktree:
        return True
    try:
        return Path(record_worktree).resolve() == Path(query_worktree).resolve()
    except OSError:
        return False


def _record_matches_owner_query(
    record: LaneRecord,
    *,
    pr_number: int | None,
    branch: str | None,
    worktree: str | None,
) -> bool:
    if pr_number is not None and record.pr_number == pr_number:
        return True
    if branch and record.branch == branch:
        return True
    return bool(record.worktree and _worktree_matches(record.worktree, worktree))


def _owner_action_for(record: LaneRecord) -> str:
    return (
        f"route mutation/comment work to owner_session {record.owner_session}; "
        "non-owners should stop or request release"
    )


def _unowned_owner_payload(
    *,
    pr_number: int | None,
    branch: str | None,
    worktree: str | None,
) -> dict[str, Any]:
    return {
        "owner_status": "unowned",
        "active_owner": False,
        "lane_id": None,
        "owner_session": None,
        "pr_number": pr_number,
        "branch": branch,
        "worktree": worktree,
        "head": None,
        "status": None,
        "updated_at": None,
        "recommended_operator_action": "no active owner found; claim the lane before mutation",
    }


def _owned_owner_payload(record: LaneRecord) -> dict[str, Any]:
    return {
        "owner_status": "owned",
        "active_owner": True,
        "lane_id": record.lane_id,
        "owner_session": record.owner_session,
        "pr_number": record.pr_number,
        "branch": record.branch or None,
        "worktree": record.worktree or None,
        "head": _head_for_worktree(record.worktree),
        "status": record.status,
        "updated_at": record.updated_at or None,
        "recommended_operator_action": _owner_action_for(record),
    }


def _historical_owner_payload(record: LaneRecord) -> dict[str, Any]:
    return {
        "owner_status": "unowned",
        "active_owner": False,
        "lane_id": record.lane_id,
        "owner_session": record.owner_session,
        "pr_number": record.pr_number,
        "branch": record.branch or None,
        "worktree": record.worktree or None,
        "head": _head_for_worktree(record.worktree),
        "status": record.status,
        "updated_at": record.updated_at or None,
        "recommended_operator_action": (
            f"latest matching lane is {record.status}; claim the lane before mutation"
        ),
    }


def _conflict_status_owner_payload(record: LaneRecord) -> dict[str, Any]:
    reason = record.conflict_reason or "resolve the recorded lane conflict"
    return {
        "owner_status": "conflict",
        "active_owner": False,
        "lane_id": record.lane_id,
        "owner_session": record.owner_session,
        "pr_number": record.pr_number,
        "branch": record.branch or None,
        "worktree": record.worktree or None,
        "head": _head_for_worktree(record.worktree),
        "status": record.status,
        "updated_at": record.updated_at or None,
        "conflict_session": record.conflict_session or None,
        "conflict_reason": record.conflict_reason or None,
        "recommended_operator_action": f"resolve lane conflict before mutation: {reason}",
    }


def _conflicted_owner_payload(records: list[LaneRecord]) -> dict[str, Any]:
    lane_ids = sorted({record.lane_id for record in records if record.lane_id})
    owner_sessions = sorted({record.owner_session for record in records if record.owner_session})
    branches = sorted({record.branch for record in records if record.branch})
    worktrees = sorted({record.worktree for record in records if record.worktree})
    pr_numbers = sorted({record.pr_number for record in records if record.pr_number is not None})
    conflict_sessions = sorted(
        {record.conflict_session for record in records if record.conflict_session}
    )
    conflict_reasons = sorted(
        {record.conflict_reason for record in records if record.conflict_reason}
    )
    updated_values = sorted(
        (record.updated_at for record in records if record.updated_at), reverse=True
    )
    return {
        "owner_status": "conflict",
        "active_owner": True,
        "lane_id": ",".join(lane_ids) or None,
        "owner_session": ",".join(owner_sessions) or None,
        "pr_number": pr_numbers[0] if len(pr_numbers) == 1 else None,
        "branch": branches[0] if len(branches) == 1 else None,
        "worktree": worktrees[0] if len(worktrees) == 1 else None,
        "head": None,
        "status": "conflict",
        "updated_at": updated_values[0] if updated_values else None,
        "conflict_session": ",".join(conflict_sessions) or None,
        "conflict_reason": " | ".join(conflict_reasons) or None,
        "recommended_operator_action": (
            "pause duplicate mutation; resolve active owner conflict before mutation"
        ),
    }


def _lane_updated_timestamp(record: LaneRecord) -> float:
    parsed = _parse_timestamp(record.updated_at)
    if parsed is None:
        return 0.0
    return parsed.timestamp()


def _newest_lane_record(records: list[LaneRecord]) -> LaneRecord:
    indexed = list(enumerate(records))
    _index, record = min(
        indexed,
        key=lambda item: (-_lane_updated_timestamp(item[1]), item[0]),
    )
    return record


def _active_owner_payload(
    records: list[LaneRecord],
    *,
    pr_number: int | None,
    branch: str | None,
    worktree: str | None,
) -> dict[str, Any]:
    matches = [
        record
        for record in records
        if _record_matches_owner_query(
            record, pr_number=pr_number, branch=branch, worktree=worktree
        )
    ]
    if not matches:
        return _unowned_owner_payload(pr_number=pr_number, branch=branch, worktree=worktree)

    active_matches = [record for record in matches if record.status in ACTIVE_LANE_STATUSES]
    if active_matches:
        owners = {record.owner_session for record in active_matches if record.owner_session}
        if len(owners) > 1:
            return _conflicted_owner_payload(active_matches)
        return _owned_owner_payload(_newest_lane_record(active_matches))

    conflict_matches = [record for record in matches if record.status in CONFLICT_LANE_STATUSES]
    if conflict_matches:
        return _conflict_status_owner_payload(_newest_lane_record(conflict_matches))

    completed_matches = [record for record in matches if record.status in COMPLETED_LANE_STATUSES]
    if completed_matches:
        return _historical_owner_payload(_newest_lane_record(completed_matches))

    return _unowned_owner_payload(pr_number=pr_number, branch=branch, worktree=worktree)


def _load_broker_run_summaries() -> list[dict[str, Any]]:
    runs_root = CANONICAL_REPO_ROOT / ".aragora" / "agent_bridge" / "runs"
    if not runs_root.exists():
        return []
    runs: list[dict[str, Any]] = []
    try:
        for run_path in runs_root.glob("*/run.json"):
            run_payload = json.loads(run_path.read_text(encoding="utf-8"))
            if not isinstance(run_payload, dict):
                continue
            run_id = str(run_payload.get("run_id") or run_path.parent.name)
            sessions: dict[str, dict[str, Any]] = {}
            sessions_path = run_path.parent / "sessions.json"
            try:
                sessions_payload = json.loads(sessions_path.read_text(encoding="utf-8"))
                raw_sessions = (
                    sessions_payload.get("sessions", {})
                    if isinstance(sessions_payload, dict)
                    else {}
                )
                if isinstance(raw_sessions, dict):
                    sessions = {
                        str(role): session
                        for role, session in raw_sessions.items()
                        if isinstance(session, dict)
                    }
            except (OSError, TypeError, json.JSONDecodeError):
                sessions = {}
            participants = run_payload.get("participants", [])
            if not isinstance(participants, list):
                participants = []
            runs.append(
                {
                    "run_id": run_id,
                    "status": run_payload.get("status"),
                    "updated_at": run_payload.get("updated_at"),
                    "next_actor": run_payload.get("next_actor"),
                    "last_turn_index": run_payload.get("last_turn_index"),
                    "participants": [
                        participant for participant in participants if isinstance(participant, dict)
                    ],
                    "sessions": sessions,
                }
            )
        runs.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
        return runs
    except (OSError, TypeError, json.JSONDecodeError, ValueError):
        return []


def _is_current_broker_run(run: dict[str, Any]) -> bool:
    return run.get("status") in ACTIVE_LANE_STATUSES or run.get("status") == "awaiting_human"


def _filter_current_broker_runs(broker_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [run for run in broker_runs if _is_current_broker_run(run)]


def _active_broker_session_ids(broker_runs: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for run in broker_runs:
        if not _is_current_broker_run(run):
            continue
        sessions = run.get("sessions", {})
        if not isinstance(sessions, dict):
            continue
        for raw_session in sessions.values():
            if not isinstance(raw_session, dict):
                continue
            session_id = raw_session.get("session_id")
            if isinstance(session_id, str) and session_id:
                ids.add(session_id)
    return ids


def _is_repo_root_path(path: str) -> bool:
    try:
        return Path(path).resolve() == CANONICAL_REPO_ROOT.resolve()
    except OSError:
        return False


def _lane_identity_values(record: "LaneRecord") -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    if record.pr_number is not None:
        values.append(("pr_number", str(record.pr_number)))
    if record.branch:
        values.append(("branch", record.branch))
    if record.worktree:
        values.append(("worktree", record.worktree))
    return values


def _active_lane_identity_conflicts(records: list["LaneRecord"]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[LaneRecord]] = {}
    for record in records:
        if record.status not in ACTIVE_LANE_STATUSES:
            continue
        for key in _lane_identity_values(record):
            buckets.setdefault(key, []).append(record)

    conflicts: list[dict[str, Any]] = []
    for (kind, value), grouped in buckets.items():
        owners = sorted({record.owner_session for record in grouped if record.owner_session})
        if len(owners) <= 1:
            continue
        lane_ids = sorted({record.lane_id for record in grouped if record.lane_id})
        conflicts.append(
            {
                "type": "lane_identity_conflict",
                "key_kind": kind,
                "key_value": value,
                "lane_ids": lane_ids,
                "owner_sessions": owners,
                "detail": (
                    f"active lanes share {kind}={value}: "
                    f"lanes={', '.join(lane_ids)} owners={', '.join(owners)}"
                ),
            }
        )
    conflicts.sort(key=lambda row: (row["key_kind"], row["key_value"]))
    return conflicts


def _collect_health_issues(
    sessions: list[Session], records: list[LaneRecord]
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    active_lane_owners = {
        record.owner_session for record in records if record.status in ACTIVE_LANE_STATUSES
    }

    # Missing paths are actionable for active/unknown sessions. A dead session
    # whose worktree is already gone has no remaining worktree cleanup action.
    # Dead historical sessions that merely remember the root checkout are also
    # not cleanup blockers. Claude transcript records are historical context;
    # if no active lane still names the transcript as owner, a removed scratch
    # worktree should not keep the operator health gate red.
    for s in sessions:
        if not s.worktree:
            continue
        lifecycle = s.lifecycle or _session_lifecycle(
            source=s.source,
            status=s.status,
            updated_at=s.updated_at,
            session_id=s.session_id,
        )
        worktree_exists = Path(s.worktree).is_dir()
        if lifecycle in {"dead", "stale", "orphaned"} or s.status == "dead":
            if worktree_exists and not _is_repo_root_path(s.worktree):
                issues.append(
                    {
                        "type": "stale_worktree",
                        "session": s.name,
                        "detail": f"dead session with lingering worktree: {s.worktree}",
                        "owner_state": "stale_session",
                        "lifecycle": lifecycle,
                        "worktree": s.worktree,
                        "worktree_exists": True,
                        "cleanup_state": "stale_lingering_worktree",
                        "recommended_operator_action": (
                            "inspect with safe_worktree_cleanup.py before any removal"
                        ),
                    }
                )
            continue
        if not worktree_exists:
            if lifecycle == "historical" and s.name not in active_lane_owners:
                continue
            issues.append(
                {
                    "type": "stale_worktree",
                    "session": s.name,
                    "detail": f"worktree path missing: {s.worktree}",
                    "owner_state": "active_or_current_session",
                    "lifecycle": lifecycle,
                    "worktree": s.worktree,
                    "worktree_exists": False,
                    "cleanup_state": "missing_path_metadata",
                    "recommended_operator_action": (
                        "verify lane ownership before pruning metadata"
                    ),
                }
            )

    # Check for ambiguous lane ownership (multiple active owners)
    lane_owners: dict[str, list[str]] = {}
    for r in records:
        if r.status in ACTIVE_LANE_STATUSES:
            lane_owners.setdefault(r.lane_id, []).append(r.owner_session)
    for lane_id, owners in lane_owners.items():
        if len(owners) > 1:
            issues.append(
                {
                    "type": "ambiguous_lane",
                    "session": ", ".join(owners),
                    "detail": f"lane '{lane_id}' claimed by multiple active sessions",
                    "owner_state": "duplicate_active_owner",
                    "lane_id": lane_id,
                    "owner_sessions": owners,
                    "recommended_operator_action": (
                        "resolve duplicate active owners before mutation or cleanup"
                    ),
                }
            )

    for r in records:
        if r.status not in ACTIVE_LANE_STATUSES:
            continue
        if not r.next_action or r.next_action == DEFAULT_ACTIVE_NEXT_ACTION:
            issues.append(
                {
                    "type": "lane_missing_next_action",
                    "session": r.owner_session,
                    "detail": f"active lane '{r.lane_id}' has no actionable next_action",
                    "owner_state": "active_lane_incomplete_metadata",
                    "lane_id": r.lane_id,
                    "status": r.status,
                    "worktree": r.worktree or None,
                    "recommended_operator_action": (
                        "refresh the lane with a concrete next_action before routing"
                    ),
                }
            )
        if not r.last_steering_outcome or r.last_steering_outcome == DEFAULT_STEERING_OUTCOME:
            issues.append(
                {
                    "type": "lane_missing_steering_outcome",
                    "session": r.owner_session,
                    "detail": f"active lane '{r.lane_id}' has no explicit steering outcome",
                    "owner_state": "active_lane_incomplete_metadata",
                    "lane_id": r.lane_id,
                    "status": r.status,
                    "worktree": r.worktree or None,
                    "recommended_operator_action": (
                        "record last_steering_outcome before claiming clean routing"
                    ),
                }
            )
        if not r.last_heartbeat_at:
            issues.append(
                {
                    "type": "lane_missing_heartbeat",
                    "session": r.owner_session,
                    "detail": f"active lane '{r.lane_id}' has no heartbeat timestamp",
                    "owner_state": "active_lane_missing_liveness",
                    "lane_id": r.lane_id,
                    "status": r.status,
                    "worktree": r.worktree or None,
                    "heartbeat_state": "missing",
                    "recommended_operator_action": (
                        "start or refresh agent_heartbeat.py before treating owner as live"
                    ),
                }
            )

    # Check for conflict-status lanes
    for r in records:
        if r.status == "conflict":
            issues.append(
                {
                    "type": "lane_conflict",
                    "session": r.owner_session,
                    "detail": f"lane '{r.lane_id}' in conflict with {r.conflict_session}: {r.conflict_reason}",
                    "owner_state": "lane_conflict",
                    "lane_id": r.lane_id,
                    "status": r.status,
                    "worktree": r.worktree or None,
                    "conflict_session": r.conflict_session,
                    "conflict_reason": r.conflict_reason,
                    "recommended_operator_action": (
                        "resolve_lane_conflicts.py dry-run before mutation or cleanup"
                    ),
                }
            )

    for conflict in _active_lane_identity_conflicts(records):
        issues.append(
            {
                "type": str(conflict["type"]),
                "session": ", ".join(conflict["owner_sessions"]),
                "detail": str(conflict["detail"]),
                "owner_state": "duplicate_active_owner",
                "key_kind": conflict["key_kind"],
                "key_value": conflict["key_value"],
                "lane_ids": conflict["lane_ids"],
                "owner_sessions": conflict["owner_sessions"],
                "recommended_operator_action": (
                    "resolve duplicate active owners before mutation or cleanup"
                ),
            }
        )

    return issues


def _classify_agent_process(command: str) -> str | None:
    """Classify known local agent/control-plane processes from a ps command line."""
    lowered = command.lower()
    if "scripts/agent_bridge.py" in lowered:
        return None
    if "codex_worktree_value_inventory.py" in lowered:
        return "worktree_inventory"
    if "run_boss_cycle.sh" in lowered:
        return "boss_cycle"
    if (
        "publish_codex_automation_branches.py" in lowered
        or "run_codex_automation_publisher.py" in lowered
    ):
        return "publisher"
    if "multi_agent_dialog.py" in lowered:
        return "multi_agent_dialog"
    if (
        " aragora.cli.main review-queue" in lowered
        or " -m aragora.cli.main review-queue" in lowered
    ):
        return "review_queue"
    if "droid exec" in lowered or "droid daemon" in lowered or "factory.app" in lowered:
        return "factory_droid"
    if re.search(r"(^|\s)(/[^ ]*/)?claude(\s|$)", command) or "claude-code" in lowered:
        return "claude_code"
    if "codex app-server" in lowered:
        return "codex_app_server"
    if re.search(r"(^|\s)(/[^ ]*/)?codex(\s|$)", command) or re.search(
        r"(^|\s)node\s+/[^ ]*/codex(\s|$)", command
    ):
        return "codex_cli"
    return None


def _process_summary_for_role(role: str) -> str:
    summaries = {
        "boss_cycle": "boss-loop control process",
        "claude_code": "Claude Code local session process",
        "codex_app_server": "Codex Desktop app server process",
        "codex_cli": "Codex CLI process",
        "factory_droid": "Factory/Droid local agent process",
        "multi_agent_dialog": "multi-agent review dialog process",
        "publisher": "Codex automation publisher process",
        "review_queue": "review-queue CLI process",
        "worktree_inventory": "worktree value inventory process",
    }
    return summaries.get(role, f"{role} process")


def _parse_ps_agent_process_line(line: str) -> dict[str, Any] | None:
    parts = line.strip().split(None, 2)
    if len(parts) < 3:
        return None
    raw_pid, elapsed, command = parts
    role = _classify_agent_process(command)
    if role is None:
        return None
    try:
        pid = int(raw_pid)
    except ValueError:
        return None
    return {
        "pid": pid,
        "elapsed": elapsed,
        "role": role,
        "summary": _process_summary_for_role(role),
    }


def _collect_agent_process_census(
    *,
    include_records: bool = True,
    record_limit: int | None = None,
    ps_lines: list[str] | None = None,
) -> dict[str, Any]:
    """Return a read-only, redacted census of active local agent processes."""
    error = ""
    if ps_lines is None:
        try:
            result = subprocess.run(
                ["ps", "-axo", "pid=,etime=,command="],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            result = None
            error = str(exc)
        if result is None:
            ps_lines = []
        elif result.returncode == 0:
            ps_lines = result.stdout.splitlines()
        else:
            error = (result.stderr or f"ps exited {result.returncode}").strip()
            ps_lines = []

    records = [
        record
        for record in (_parse_ps_agent_process_line(line) for line in ps_lines)
        if record is not None
    ]
    records.sort(key=lambda item: (str(item["role"]), int(item["pid"])))
    by_role: dict[str, int] = {}
    for record in records:
        role = str(record["role"])
        by_role[role] = by_role.get(role, 0) + 1

    payload: dict[str, Any] = {
        "ok": not error,
        "total": len(records),
        "by_role": dict(sorted(by_role.items())),
    }
    if error:
        payload["error"] = error
    if include_records:
        limited_records = records[:record_limit] if record_limit is not None else records
        payload["records"] = limited_records
        if len(limited_records) < len(records):
            payload["records_omitted"] = len(records) - len(limited_records)
    return payload


def _lane_conflict(
    records: list[LaneRecord],
    lane_id: str,
    owner_session: str,
) -> LaneRecord | None:
    record = _find_lane_record(records, lane_id)
    if record is None:
        return None
    if record.owner_session == owner_session:
        return None
    if record.status not in ACTIVE_LANE_STATUSES:
        return None
    return record


def _persist_lane_claim(
    records: list[LaneRecord],
    lane_id: str,
    session: Session,
    *,
    goal: str,
    source: str,
    status: str,
    next_action: str,
    allow_conflict: bool,
) -> None:
    existing = _find_lane_record(records, lane_id)
    conflict = _lane_conflict(records, lane_id, session.name)
    if conflict is not None and allow_conflict:
        conflict.status = "conflict"
        conflict.conflict_session = session.name
        conflict.conflict_reason = f"conflicting active owner claim from {session.name}"
        conflict.next_action = next_action or "resolve ambiguous lane ownership"
        conflict.updated_at = _now_iso()
        _write_lane_registry(records)
        return

    record = existing or LaneRecord(lane_id=lane_id, owner_session=session.name)
    record.owner_session = session.name
    record.goal = goal or record.goal
    record.source = source or record.source
    record.status = status or record.status
    record.next_action = next_action or record.next_action
    record.updated_at = _now_iso()
    record.branch = session.branch
    record.worktree = session.worktree
    record.pr_number = session.pr_number
    record.conflict_session = ""
    record.conflict_reason = ""
    if existing is None:
        records.append(record)
    _write_lane_registry(records)


def _find_session(sessions: list[Session], target: str) -> Session | None:
    for s in sessions:
        if target in s.name or target in (s.session_id or ""):
            return s
    return None


# ---------------------------------------------------------------------------
# tmux transport
# ---------------------------------------------------------------------------


def _send_tmux(target: str, prompt: str) -> bool:
    try:
        if "\n" in prompt:
            subprocess.run(
                ["tmux", "load-buffer", "-"],
                input=prompt,
                text=True,
                check=True,
                timeout=5,
            )
            subprocess.run(
                ["tmux", "paste-buffer", "-d", "-t", target],
                check=True,
                timeout=5,
            )
            time.sleep(float(os.environ.get("ARAGORA_TMUX_PASTE_SETTLE_SECONDS", "0.2")))
            subprocess.run(
                ["tmux", "send-keys", "-t", target, "Enter"],
                check=True,
                timeout=5,
            )
        else:
            subprocess.run(
                ["tmux", "send-keys", "-t", target, prompt, "Enter"],
                check=True,
                timeout=5,
            )
        return True
    except (subprocess.SubprocessError, OSError):
        return False


def _resolve_tmux_target(session: Session) -> str | None:
    if session.tmux_target:
        return session.tmux_target
    # Try finding window by name
    try:
        result = subprocess.run(
            ["tmux", "list-windows", "-t", TMUX_SESSION, "-F", "#{window_index} #{window_name}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = line.strip().split(None, 1)
                if len(parts) >= 2 and session.name in parts[1]:
                    return f"{TMUX_SESSION}:{parts[0]}"
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


# ---------------------------------------------------------------------------
# PR enrichment
# ---------------------------------------------------------------------------


def _enrich_prs(sessions: list[Session]) -> None:
    branches = [s.branch for s in sessions if s.branch]
    if not branches:
        return
    try:
        result = subprocess.run(
            [
                "gh",
                "pr",
                "list",
                "--state",
                "open",
                "--limit",
                "30",
                "--json",
                "number,headRefName",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if result.returncode != 0:
            return
        prs = json.loads(result.stdout)
        branch_pr = {pr["headRefName"]: pr["number"] for pr in prs}
        for s in sessions:
            if s.branch and s.branch in branch_pr:
                s.pr_number = branch_pr[s.branch]
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        pass


# ---------------------------------------------------------------------------
# tmux log reader
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07")


def _read_tmux_log(name: str, lines: int) -> list[str]:
    log_file = TMUX_SESSIONS_DIR / f"{name}.log"
    if not log_file.exists():
        return []
    try:
        raw = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        clean: list[str] = []
        for line in raw[-(lines * 5) :]:
            c = _ANSI_RE.sub("", line).strip()
            if c and len(c) > 5 and not c.startswith("[?"):
                clean.append(c[:150])
        return clean[-lines:]
    except OSError:
        return []


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_sessions(args: argparse.Namespace) -> int:
    sessions, _broker_runs, _active_broker_ids = _discover_with_broker_state()
    _write_session_snapshot(sessions)
    if args.json:
        print(json.dumps([s.to_dict() for s in sessions], indent=2))
        return 0
    if not sessions:
        print("No active sessions.")
        return 0
    print(f"{'NAME':<24} {'AGENT':<8} {'STATUS':<8} {'BRANCH':<28} SUMMARY")
    print("-" * 110)
    for s in sessions:
        branch = s.branch[:26] if s.branch else "-"
        summary = (s.summary[:40] + "..." if len(s.summary) > 40 else s.summary) or "-"
        print(f"{s.name:<24} {s.agent:<8} {s.status:<8} {branch:<28} {summary}")
    return 0


# ---------------------------------------------------------------------------
# Launch dispatch verification (issue #8317)
#
# A dispatched lane can die silently when the launcher pastes the prompt into
# the harness composer but the trailing Enter never registers as a submit: the
# lane registry shows a launched lane while the prompt sits staged in the
# composer forever.  These helpers read back the pane after launch, confirm
# the prompt actually left the composer, nudge it with exactly ONE Enter if it
# is still staged, and persist a "dispatch" receipt into the launcher's lane
# metadata entry so liveness checks can breach on staged-but-unsubmitted
# lanes.
# ---------------------------------------------------------------------------

DEFAULT_SUBMIT_VERIFY_TIMEOUT_SECONDS = 5.0
SUBMIT_VERIFY_POLLS = 3
# After a confident Enter nudge, give the harness a moment to clear the
# composer with a short second re-poll instead of a single fixed wait, so a
# harness that needs a beat is not recorded undelivered prematurely.
SUBMIT_VERIFY_NUDGE_REPOLLS = 2
SUBMIT_VERIFY_NUDGE_REPOLL_INTERVAL = 1.5
_PASTE_PLACEHOLDER_RE = re.compile(r"\[Pasted (?:Content|text)\b", re.IGNORECASE)


def _default_tmux_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a tmux command with a bounded timeout; injectable for tests."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )


def _launch_uses_interactive_paste(agent: str, *, autonomous: bool) -> bool:
    """True when tmux_session_launcher.sh delivers the prompt via paste.

    Claude lanes are always interactive (prompt pasted after readiness).
    Codex lanes paste only when not autonomous (autonomous prompted Codex
    uses ``codex exec`` which consumes the prompt directly).  Droid/Factory
    prompted lanes always use ``droid exec -f`` -- nothing to verify.
    """
    if agent == "claude":
        return True
    return agent == "codex" and not autonomous


def _launched_pane_target(name: str) -> tuple[str | None, str | None]:
    """Resolve the tmux pane target for a freshly launched lane.

    Returns ``(target, reason)``.  ``reason`` is ``None`` on success; on failure
    ``target`` is ``None`` and ``reason`` is a short diagnostic.

    The meta file is the source of truth for the launched window target.  A
    meta file that *exists but cannot be read* (OSError / JSONDecodeError, or a
    non-dict / missing ``tmux_window_target``) fails CLOSED with
    ``reason="meta-unreadable"`` (review fix #8338 / #8317): verifying against
    the documented ``TMUX_SESSION:name`` fallback could capture an unrelated
    pane and either falsely attest delivery or falsely flag an unrelated lane.
    When the meta file is *legitimately absent* the documented fallback target
    is returned (``reason=None``) -- there is no other pane to confuse it with.
    """
    meta_path = TMUX_SESSIONS_DIR / f"{name}.meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None, "meta-unreadable"
        if not isinstance(meta, dict):
            return None, "meta-unreadable"
        target = str(meta.get("tmux_window_target") or "")
        if not target:
            return None, "meta-unreadable"
        return target, None
    return f"{TMUX_SESSION}:{name}", None


def _prompt_tail_marker(prompt: str, *, max_chars: int = 80) -> str:
    """Last non-empty prompt line (tail-trimmed) used to spot a staged paste."""
    for line in reversed(prompt.splitlines()):
        cleaned = line.strip()
        if cleaned:
            return cleaned[-max_chars:]
    return ""


def _pane_shows_staged_prompt(
    pane_text: str, marker: str, *, tail_lines: int = 25
) -> tuple[bool, bool]:
    """Heuristic: is the pasted prompt still sitting in the composer?

    Returns ``(staged, placeholder)`` where ``staged`` is True when the prompt
    still appears un-submitted and ``placeholder`` is True only when a tmux/
    harness paste placeholder such as ``[Pasted Content N chars]`` is present.

    The two signals are deliberately split (review fix #8338 / #8317):

    * **Paste-placeholder detected** -- a high-confidence positive that the
      prompt is genuinely staged in the composer; this is the ONLY signal the
      caller acts on (single Enter nudge, then a confident True/False).
    * **Bare marker-in-tail (no placeholder)** -- low-confidence and too
      false-positive-prone to act on.  A harness routinely echoes the submitted
      prompt back as a quoted line *after* submit, so a substring match alone is
      not proof the composer still holds the prompt.  The verifier treats this
      as *unverifiable* (``delivered=None``), never as a confident
      still-staged (``False``) and never as grounds to send Enter.

    The tail window scans the last ``tail_lines`` non-empty lines after ANSI
    cleanup (wider than a handful of lines so a staged paste followed by a few
    lines of harness chrome does not scroll the marker off the tail).  Once a
    harness accepts a submit it appends output below the echoed prompt, so the
    pane tail eventually moves past both markers.  Known limitation: a freshly
    submitted prompt with no harness output yet can look staged; the bounded
    re-poll plus the single Enter nudge (a no-op on an empty composer) covers
    that.
    """
    cleaned_lines = [
        cleaned
        for cleaned in (_ANSI_RE.sub("", line).strip() for line in pane_text.splitlines())
        if cleaned
    ]
    tail = "\n".join(cleaned_lines[-tail_lines:])
    if _PASTE_PLACEHOLDER_RE.search(tail):
        return True, True
    return (bool(marker) and marker in tail), False


def _capture_pane_tail(
    target: str,
    runner: Any = None,
    *,
    lines: int = 40,
) -> str | None:
    """Read back the pane contents for submit verification.

    Returns the captured pane text on success (possibly an empty string for a
    genuinely empty pane), or ``None`` when the capture itself failed -- a tmux
    subprocess error, a non-zero returncode (e.g. dead/missing pane), or an
    OSError.  Callers must treat ``None`` as *unverifiable* and never collapse
    it into a clean "delivered" outcome (review fix #8338 / #8317): a
    capture-failure is not evidence the prompt left the composer.
    """
    run = runner or _default_tmux_runner
    try:
        result = run(["tmux", "capture-pane", "-t", target, "-p", "-S", f"-{lines}"])
    except (subprocess.SubprocessError, OSError):
        return None
    if getattr(result, "returncode", 1) != 0:
        return None
    return str(getattr(result, "stdout", "") or "")


def _verify_prompt_submission(
    target: str,
    prompt: str,
    *,
    timeout_seconds: float = DEFAULT_SUBMIT_VERIFY_TIMEOUT_SECONDS,
    polls: int = SUBMIT_VERIFY_POLLS,
    runner: Any = None,
    sleep: Any = None,
) -> dict[str, Any]:
    """Confirm a pasted prompt left the composer; nudge once if confidently staged.

    Polls the pane up to ``polls`` times within ``timeout_seconds`` and records
    a tri-state ``delivered`` outcome (review fix #8338 / #8317):

    * ``True``  -- a paste placeholder was positively detected and then cleared
      (submitted), or no staged signal was ever seen.
    * ``False`` -- a paste placeholder was positively detected and *persisted*
      through the single Enter nudge and the short re-poll (still staged).
    * ``None``  -- submission could not be verified either way.  This covers a
      failed pane capture (``reason: "capture-failed"``), a bare marker-in-tail
      match with no placeholder (``reason: "marker-only"`` -- too
      false-positive-prone to act on), and a nudge that ``tmux send-keys``
      rejected (``reason: "nudge-failed"``).

    A corrective Enter is sent only on a *positive* paste-placeholder
    detection -- the one high-confidence signal the prompt genuinely sits in
    the composer.  A bare marker-in-tail match is NEVER auto-submitted (a
    harness echoes the submitted prompt back as a quoted line, so it is not
    proof the composer still holds it) and is reported as unverifiable rather
    than a confident False.  At most ONE Enter is ever sent, and only when
    ``tmux send-keys`` actually succeeds (returncode 0); a rejected send is
    recorded as unverifiable rather than falsely attested.

    After a confident nudge the composer is re-polled with a short bounded
    loop (``SUBMIT_VERIFY_NUDGE_REPOLLS`` polls) so a harness that needs a beat
    to clear the composer is not recorded undelivered prematurely.
    """
    run = runner or _default_tmux_runner
    if sleep is None:
        sleep = time.sleep
    marker = _prompt_tail_marker(prompt)
    polls = max(1, int(polls))
    interval = max(0.1, float(timeout_seconds) / polls)
    attempts = 0
    enter_nudges = 0
    capture_failed = False
    saw_marker_only = False
    staged = False
    placeholder = False
    for poll_index in range(polls):
        attempts += 1
        pane = _capture_pane_tail(target, run)
        if pane is None:
            capture_failed = True
            break
        capture_failed = False
        marker_staged, placeholder = _pane_shows_staged_prompt(pane, marker)
        if placeholder:
            # High-confidence staged: a paste placeholder is in the composer.
            staged = True
            break
        if marker_staged:
            # Low-confidence: bare marker echo. Record it but keep polling --
            # the placeholder (if any) may surface, or the echo may scroll off.
            saw_marker_only = True
            staged = False
        else:
            staged = False
            saw_marker_only = False
            break
        if poll_index < polls - 1:
            sleep(interval)

    delivered: bool | None
    reason: str | None = None
    nudge_failed = False

    if capture_failed:
        delivered = None
        reason = "capture-failed"
    elif placeholder:
        # Positive staged signal -> send exactly one Enter, then re-poll briefly.
        sent = False
        try:
            result = run(["tmux", "send-keys", "-t", target, "Enter"])
            sent = getattr(result, "returncode", 1) == 0
        except (subprocess.SubprocessError, OSError):
            sent = False
        if not sent:
            nudge_failed = True
            delivered = None
            reason = "nudge-failed"
        else:
            enter_nudges = 1
            still_placeholder = True
            for _ in range(max(1, int(SUBMIT_VERIFY_NUDGE_REPOLLS))):
                sleep(min(interval, SUBMIT_VERIFY_NUDGE_REPOLL_INTERVAL))
                attempts += 1
                pane = _capture_pane_tail(target, run)
                if pane is None:
                    capture_failed = True
                    break
                _marker_staged, still_placeholder = _pane_shows_staged_prompt(pane, marker)
                if not still_placeholder:
                    break
            if capture_failed:
                delivered = None
                reason = "capture-failed"
            elif still_placeholder:
                # Placeholder positively persisted through the nudge+repoll.
                delivered = False
            else:
                delivered = True
    elif saw_marker_only:
        # Bare marker echo, never a placeholder: too false-positive-prone to
        # call either way. Unverifiable, and no Enter was sent.
        delivered = None
        reason = "marker-only"
    else:
        # No staged signal at all -> the prompt left the composer.
        delivered = True

    outcome: dict[str, Any] = {
        "delivered": delivered,
        "attempts": attempts,
        "enter_nudges": enter_nudges,
        "pane_target": target,
        "verified_at": _now_iso(),
        "method": "pane-tail-heuristic",
    }
    if reason is not None:
        outcome["error"] = reason
    if nudge_failed:
        outcome["nudge_failed"] = True
    return outcome


def _record_dispatch_receipt(name: str, dispatch: dict[str, Any]) -> Path | None:
    """Additively merge a ``dispatch`` sub-object into the lane meta entry.

    Extends the metadata file tmux_session_launcher.sh already writes at
    ``~/.aragora/tmux-sessions/<name>.meta.json``; all existing keys are
    preserved.  Sentinels (e.g. lane_liveness) can breach on
    ``dispatch.delivered == false`` for staged-but-unsubmitted lanes.
    """
    meta_path = TMUX_SESSIONS_DIR / f"{name}.meta.json"
    payload: dict[str, Any] = {}
    if meta_path.exists():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload.setdefault("name", name)
    payload["dispatch"] = dispatch
    try:
        _atomic_write_json(meta_path, payload)
    except OSError:
        return None
    return meta_path


def _install_sigpipe_hygiene() -> None:
    """Keep dispatch exit codes truthful when a stdout consumer exits early.

    Without this, a downstream reader closing its end of the pipe kills the
    launcher with SIGPIPE (exit 141) before it can report dispatch truth.
    With SIGPIPE ignored, writes raise BrokenPipeError instead, which the
    report paths catch explicitly.  POSIX-only; a no-op elsewhere.
    """
    if not hasattr(signal, "SIGPIPE"):
        return
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    except (OSError, ValueError):
        pass


def cmd_launch(args: argparse.Namespace) -> int:
    """Launch a tmux-managed harness lane, then let send/read manage it."""
    # SIGPIPE hygiene is scoped to launch (review fix #8338 / #8317): the
    # report writes below are wrapped in BrokenPipeError handling so a consumer
    # closing the pipe early cannot mask the dispatch outcome. Other subcommands
    # keep their default SIGPIPE behavior.
    _install_sigpipe_hygiene()
    if not args.name:
        print("No session name. Use --name", file=sys.stderr)
        return 1
    agent = str(args.agent or "codex").strip()
    if agent not in {"codex", "claude", "droid", "factory"}:
        print("Unsupported agent. Use codex, claude, droid, or factory.", file=sys.stderr)
        return 1
    launch_cwd = Path(args.cwd).expanduser() if args.cwd else Path.cwd()
    try:
        launch_cwd = launch_cwd.resolve()
    except OSError as exc:
        print(f"Invalid launch cwd: {exc}", file=sys.stderr)
        return 1
    if not launch_cwd.is_dir():
        print(f"Launch cwd does not exist or is not a directory: {launch_cwd}", file=sys.stderr)
        return 1
    if agent in {"droid", "factory"} and getattr(args, "autonomous", False):
        message = (
            "Interactive Droid/Factory tmux sessions cannot be made autonomous. "
            "Use `python3 scripts/agent_bridge.py exec --agent droid --auto high ...` "
            "or `python3 scripts/agent_bridge_broker.py` for non-interactive Droid evidence."
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "name": args.name,
                        "agent": agent,
                        "cwd": str(launch_cwd),
                        "error": message,
                    },
                    indent=2,
                )
            )
        else:
            print(message, file=sys.stderr)
        return 1

    launcher = CANONICAL_REPO_ROOT / "scripts" / "tmux_session_launcher.sh"
    cmd = [
        "bash",
        str(launcher),
        "--name",
        args.name,
        "--agent",
        agent,
        "--cwd",
        str(launch_cwd),
    ]
    if getattr(args, "autonomous", False):
        cmd.append("--autonomous")
    if args.file:
        cmd.extend(["--prompt-file", args.file])
    elif args.prompt:
        cmd.extend(["--prompt", " ".join(args.prompt)])

    try:
        result = subprocess.run(
            cmd,
            cwd=str(CANONICAL_REPO_ROOT),
            capture_output=bool(args.json),
            text=True,
            timeout=max(30, int(args.timeout_seconds)),
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        print(f"Launch failed: {exc}", file=sys.stderr)
        return 1

    dispatch = _verify_launch_dispatch(args, agent=agent, launch_ok=result.returncode == 0)
    exit_code = result.returncode
    # Verification is OBSERVATIONAL by default (review fix #8338 / #8317): the
    # tri-state dispatch receipt is ALWAYS written (that is the durable signal
    # lane_liveness reads), but it does NOT flip cmd_launch's exit code.  A
    # successful launch returns its normal rc regardless of whether the prompt
    # was confirmed submitted -- changing the default broke every caller (CI,
    # retry loops, the funnel automation) that treats rc!=0 as launch failure.
    # Exit-code enforcement is opt-in via --strict-verify: only then does a
    # non-delivered (False) or unverifiable (None) dispatch produce rc=1.
    if (
        getattr(args, "strict_verify", False)
        and exit_code == 0
        and dispatch is not None
        and dispatch.get("delivered") is not True
    ):
        exit_code = 1

    # Report writes are wrapped so a consumer closing the pipe early cannot
    # mask the dispatch outcome (see _install_sigpipe_hygiene / issue #8317).
    try:
        if args.json:
            payload: dict[str, Any] = {
                "ok": exit_code == 0,
                "name": args.name,
                "agent": agent,
                "cwd": str(launch_cwd),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            if dispatch is not None:
                payload["dispatch"] = dispatch
            print(json.dumps(payload, indent=2))
        else:
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr)
            if dispatch is not None and dispatch.get("delivered") is not True:
                if dispatch.get("delivered") is None:
                    print(
                        f"Prompt submission for '{args.name}' could not be "
                        f"verified after {dispatch.get('attempts', 0)} check(s) "
                        f"({dispatch.get('error', 'capture-failed')}); "
                        "dispatch receipt records delivered=null (unverifiable).",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"Prompt for '{args.name}' still staged in the composer "
                        f"after {dispatch.get('attempts', 0)} check(s); "
                        "dispatch receipt records delivered=false.",
                        file=sys.stderr,
                    )
    except BrokenPipeError:
        _mute_stdout_after_broken_pipe()

    return exit_code


def _verify_launch_dispatch(
    args: argparse.Namespace,
    *,
    agent: str,
    launch_ok: bool,
) -> dict[str, Any] | None:
    """Run post-launch submit verification and persist the dispatch receipt.

    Returns the dispatch outcome dict, or None when verification does not
    apply (launch failed, no prompt, exec-style delivery, or verification
    disabled via ``--submit-verify-timeout 0``).
    """
    if not launch_ok:
        return None
    timeout_seconds = float(
        getattr(args, "submit_verify_timeout", DEFAULT_SUBMIT_VERIFY_TIMEOUT_SECONDS) or 0.0
    )
    if timeout_seconds <= 0:
        return None
    if not _launch_uses_interactive_paste(
        agent, autonomous=bool(getattr(args, "autonomous", False))
    ):
        return None
    prompt = ""
    if getattr(args, "file", None):
        try:
            prompt = Path(args.file).read_text(encoding="utf-8")
        except OSError:
            return None
    elif getattr(args, "prompt", None):
        prompt = " ".join(args.prompt)
    if not prompt.strip():
        return None
    target, target_reason = _launched_pane_target(args.name)
    if target is None:
        # The meta file existed but could not be read for a trustworthy pane
        # target.  Fail CLOSED to unverifiable rather than capturing a possibly
        # unrelated fallback pane (review fix #8338 / #8317).
        dispatch: dict[str, Any] = {
            "delivered": None,
            "attempts": 0,
            "enter_nudges": 0,
            "pane_target": None,
            "verified_at": _now_iso(),
            "method": "pane-tail-heuristic",
            "error": target_reason or "meta-unreadable",
        }
    else:
        dispatch = _verify_prompt_submission(
            target,
            prompt,
            timeout_seconds=timeout_seconds,
        )
    _record_dispatch_receipt(args.name, dispatch)
    return dispatch


def cmd_exec(args: argparse.Namespace) -> int:
    """Run one non-interactive harness turn through the bridge transport layer."""
    agent = str(args.agent or "droid").strip()
    if agent not in {"codex", "claude", "droid", "factory"}:
        print("Unsupported agent. Use codex, claude, droid, or factory.", file=sys.stderr)
        return 1
    launch_cwd = Path(args.cwd).expanduser() if args.cwd else Path.cwd()
    try:
        launch_cwd = launch_cwd.resolve()
    except OSError as exc:
        print(f"Invalid exec cwd: {exc}", file=sys.stderr)
        return 1
    if not launch_cwd.is_dir():
        print(f"Exec cwd does not exist or is not a directory: {launch_cwd}", file=sys.stderr)
        return 1

    prompt = Path(args.file).read_text("utf-8") if args.file else " ".join(args.prompt or [])
    if not prompt:
        print("No prompt. Use text args or --file", file=sys.stderr)
        return 1

    harness_options: dict[str, Any] = {}
    auto_mode = str(args.auto or "").strip()
    if auto_mode:
        harness_options["auto"] = auto_mode
    elif agent in {"droid", "factory"}:
        harness_options["auto"] = "high"
    allowed_roles = set(args.allowed_role or ["reviewer"])

    try:
        transport_error, transport_factory = _load_transport_runtime()
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        if args.json:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "agent": agent,
                        "cwd": str(launch_cwd),
                        "error": str(exc),
                    },
                    indent=2,
                )
            )
        else:
            print(f"Exec failed: {exc}", file=sys.stderr)
        return 1

    try:
        transport = transport_factory(
            agent,
            cwd=launch_cwd,
            model=str(args.model).strip() if args.model else None,
            harness_options=harness_options,
        )
        result = transport.launch(prompt, allowed_roles=allowed_roles)
    except (transport_error, OSError, ValueError) as exc:
        if args.json:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "agent": agent,
                        "cwd": str(launch_cwd),
                        "error": str(exc),
                    },
                    indent=2,
                )
            )
        else:
            print(f"Exec failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        footer = result.parsed_turn.footer.to_dict() if result.parsed_turn.footer else None
        print(
            json.dumps(
                {
                    "ok": True,
                    "agent": agent,
                    "cwd": str(launch_cwd),
                    "session_id": result.session_id,
                    "command": result.command,
                    "exit_code": result.exit_code,
                    "message_text": result.message_text,
                    "parse_status": result.parsed_turn.parse_status,
                    "footer": footer,
                    "parse_errors": list(result.parsed_turn.parse_errors),
                    "usage": result.usage,
                },
                indent=2,
            )
        )
        return 0

    print(result.message_text)
    return 0


def cmd_send(args: argparse.Namespace) -> int:
    sessions = discover()
    _enrich_prs(sessions)
    session = _find_session(sessions, args.name)
    if not session:
        print(f"No session matching '{args.name}'", file=sys.stderr)
        return 1
    prompt = Path(args.file).read_text("utf-8") if args.file else " ".join(args.prompt or [])
    if not prompt:
        print("No prompt. Use text args or --file", file=sys.stderr)
        return 1
    target = _resolve_tmux_target(session)
    if not target:
        print(f"No tmux target for '{session.name}'", file=sys.stderr)
        return 1
    records = _sync_lane_records(_load_lane_registry(), sessions)
    lane_id = str(getattr(args, "lane", "") or "").strip()
    if lane_id:
        conflict = _lane_conflict(records, lane_id, session.name)
        if conflict is not None and not getattr(args, "allow_conflict", False):
            print(
                f"Lane '{lane_id}' already owned by active session '{conflict.owner_session}'",
                file=sys.stderr,
            )
            return 1
    if _send_tmux(target, prompt):
        if lane_id:
            _persist_lane_claim(
                records,
                lane_id,
                session,
                goal=str(getattr(args, "goal", "") or "").strip(),
                source=str(getattr(args, "source", "") or "").strip(),
                status=str(getattr(args, "status", "") or "active").strip(),
                next_action=str(getattr(args, "next_action", "") or "").strip(),
                allow_conflict=bool(getattr(args, "allow_conflict", False)),
            )
        print(f"Sent to '{session.name}' ({len(prompt)} chars)")
        return 0
    print(f"Send failed for '{session.name}'", file=sys.stderr)
    return 1


def cmd_approve(args: argparse.Namespace) -> int:
    sessions = discover()
    session = _find_session(sessions, args.name)
    if not session:
        print(f"No session matching '{args.name}'", file=sys.stderr)
        return 1
    target = _resolve_tmux_target(session)
    if not target:
        target = f"{TMUX_SESSION}:{session.name}"
    keys = ["Enter"] if session.agent in {"droid", "factory"} else ["y", "Enter"]
    try:
        subprocess.run(
            ["tmux", "send-keys", "-t", target, *keys],
            check=True,
            timeout=5,
        )
        print(f"Approved '{session.name}'")
        return 0
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"Approve failed: {exc}", file=sys.stderr)
        return 1


def cmd_read(args: argparse.Namespace) -> int:
    sessions = discover()
    session = _find_session(sessions, args.name)
    if not session:
        print(f"No session matching '{args.name}'", file=sys.stderr)
        return 1
    lines = _read_tmux_log(session.name, args.lines)
    print(f"Session: {session.name}  [{session.status}]  branch={session.branch or '-'}")
    print("-" * 80)
    for line in lines:
        print(f"  {line}")
    if not lines:
        print("  (no output)")
    return 0


def cmd_read_all(args: argparse.Namespace) -> int:
    sessions = discover()
    if not sessions:
        print("No sessions.")
        return 0
    if args.json:
        result = []
        for s in sessions:
            entry = s.to_dict()
            entry["recent_output"] = _read_tmux_log(s.name, args.lines)
            result.append(entry)
        print(json.dumps(result, indent=2))
        return 0
    for s in sessions:
        lines = _read_tmux_log(s.name, args.lines)
        print(f"\n{'=' * 80}")
        print(f"{s.name} [{s.agent}] [{s.status}] branch={s.branch or '-'}")
        print("-" * 80)
        for line in lines:
            print(f"  {line}")
        if not lines and s.summary:
            print(f"  {s.summary}")
        elif not lines:
            print("  (no output)")
    return 0


def cmd_lanes(args: argparse.Namespace) -> int:
    sessions, _broker_runs, _active_broker_ids = _discover_with_broker_state()
    _enrich_prs(sessions)
    _write_session_snapshot(sessions)
    records = _sync_lane_records(_load_lane_registry(), sessions)
    if records:
        _write_lane_registry(records)
        if args.json:
            print(json.dumps([record.to_dict() for record in records], indent=2))
            return 0
        print(f"{'LANE':<22} {'OWNER':<24} {'STATUS':<10} {'BRANCH':<26} {'PR':>5} NEXT ACTION")
        print("-" * 120)
        for record in records:
            branch = record.branch[:24] if record.branch else "-"
            pr = f"#{record.pr_number}" if record.pr_number else "-"
            next_action = (
                record.next_action[:40] + "..."
                if len(record.next_action) > 40
                else record.next_action
            ) or "-"
            print(
                f"{record.lane_id:<22} {record.owner_session:<24} {record.status:<10} "
                f"{branch:<26} {pr:>5} {next_action}"
            )
        return 0
    if args.json:
        print(json.dumps([s.to_dict() for s in sessions], indent=2))
        return 0
    print(f"{'NAME':<24} {'AGENT':<8} {'STATUS':<8} {'BRANCH':<26} {'PR':>5} SUMMARY")
    print("-" * 110)
    for s in sessions:
        branch = s.branch[:24] if s.branch else "-"
        pr = f"#{s.pr_number}" if s.pr_number else "-"
        summary = (s.summary[:30] + "..." if len(s.summary) > 30 else s.summary) or "-"
        print(f"{s.name:<24} {s.agent:<8} {s.status:<8} {branch:<26} {pr:>5} {summary}")
    return 0


def cmd_owner(args: argparse.Namespace) -> int:
    """Report the active lane owner for a PR, branch, or worktree."""
    pr_number = getattr(args, "pr", None)
    branch = str(getattr(args, "branch", "") or "").strip() or None
    worktree = str(getattr(args, "worktree", "") or "").strip() or None
    if pr_number is None and branch is None and worktree is None:
        print("Provide at least one of --pr, --branch, or --worktree.", file=sys.stderr)
        return 2

    sessions, _broker_runs, _active_broker_ids = _discover_with_broker_state(
        include_summaries=False,
        include_historical=False,
    )
    _enrich_prs(sessions)
    records = _sync_lane_records(_load_lane_registry(), sessions)
    payload = _active_owner_payload(
        records,
        pr_number=pr_number,
        branch=branch,
        worktree=worktree,
    )

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    if payload["owner_status"] == "unowned":
        print(f"unowned: {payload['recommended_operator_action']}")
        return 0
    print(
        f"{payload['owner_status']}: lane={payload['lane_id']} "
        f"owner={payload['owner_session']} pr={payload['pr_number'] or '-'} "
        f"branch={payload['branch'] or '-'} worktree={payload['worktree'] or '-'}"
    )
    print(payload["recommended_operator_action"])
    return 0


def cmd_processes(args: argparse.Namespace) -> int:
    """Report local agent/control-plane processes without exposing raw commands."""
    summary_only = bool(getattr(args, "summary_only", False))
    census = _collect_agent_process_census(
        include_records=not summary_only,
        record_limit=max(0, int(getattr(args, "limit", 50))),
    )
    if args.json:
        print(json.dumps(census, indent=2))
        return 0 if census.get("ok", False) else 1

    if not census.get("ok", False):
        print(
            f"Process census unavailable: {census.get('error', 'unknown error')}", file=sys.stderr
        )
        return 1

    records = census.get("records", [])
    if summary_only:
        roles = ", ".join(f"{role}={count}" for role, count in census.get("by_role", {}).items())
        print(f"Recognized processes: {census.get('total', 0)} ({roles or 'none'})")
        return 0
    if not records:
        print("No recognized local agent processes.")
        return 0

    print(f"{'PID':>8} {'ELAPSED':>12} {'ROLE':<22} SUMMARY")
    print("-" * 85)
    for record in records:
        print(
            f"{int(record['pid']):>8} {str(record['elapsed']):>12} "
            f"{str(record['role']):<22} {record['summary']}"
        )
    omitted = int(census.get("records_omitted", 0))
    if omitted:
        print(f"... {omitted} additional process record(s) omitted; use --limit to show more.")
    return 0


def _health_summary_payload(
    issues: list[dict[str, str]], *, example_limit: int = 3
) -> dict[str, Any]:
    issue_type_counts: dict[str, int] = {}
    for issue in issues:
        issue_type = str(issue.get("type") or "unknown")
        issue_type_counts[issue_type] = issue_type_counts.get(issue_type, 0) + 1
    examples = issues[: max(0, example_limit)]
    return {
        "ok": len(issues) == 0,
        "issue_count": len(issues),
        "issue_type_counts": dict(sorted(issue_type_counts.items())),
        "issue_examples": examples,
        "issues_omitted": max(0, len(issues) - len(examples)),
        "details_omitted": True,
    }


def cmd_health(args: argparse.Namespace) -> int:
    """Report stale worktrees, ambiguous lane ownership, and dead sessions."""
    summary_only = bool(getattr(args, "summary_only", False))
    sessions, _broker_runs, _active_broker_ids = _discover_with_broker_state()
    _enrich_prs(sessions)
    records = _sync_lane_records(_load_lane_registry(), sessions)

    issues = _collect_health_issues(sessions, records)

    # Check git worktree list for prunable entries
    try:
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            cwd=str(CANONICAL_REPO_ROOT),
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith("worktree "):
                    wt_path = line.split(" ", 1)[1]
                    if not Path(wt_path).is_dir():
                        issues.append(
                            {
                                "type": "prunable_worktree",
                                "session": "-",
                                "detail": f"git worktree missing on disk: {wt_path}",
                            }
                        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    if args.json:
        if summary_only:
            print(json.dumps(_health_summary_payload(issues), indent=2))
            return 0 if not issues else 1
        print(json.dumps({"ok": len(issues) == 0, "issues": issues}, indent=2))
        return 0 if not issues else 1

    if summary_only:
        payload = _health_summary_payload(issues)
        if not issues:
            print("Health OK: 0 issue(s).")
            return 0
        issue_counts = ", ".join(
            f"{issue_type}={count}" for issue_type, count in payload["issue_type_counts"].items()
        )
        print(f"Health has {payload['issue_count']} issue(s): {issue_counts}")
        return 1

    if not issues:
        print("Health OK: no stale worktrees, no lane conflicts.")
        return 0

    print(f"Found {len(issues)} issue(s):\n")
    print(f"{'TYPE':<22} {'SESSION':<26} DETAIL")
    print("-" * 100)
    for issue in issues:
        print(f"{issue['type']:<22} {issue['session']:<26} {issue['detail']}")
    return 1


def _collect_pending_steering_messages(
    session_name: str | None,
    steering_root: Path | None = None,
) -> dict[str, Any]:
    """Read the operator-steering mailbox(es) and surface pending counts.

    Phase C of the agent-steering primitive. Reads only — never
    mutates the mailbox. Companion to ``scripts/send_operator_steering.py``
    (Phase B writer) and ``scripts/identify_lane_owner.py``
    (Phase A consolidator) which both honour the same
    ``aragora-operator-steering/1.0`` schema.

    Scoping rules:
      - ``session_name`` set     → return only that recipient's count
                                   + latest_three.
      - ``session_name`` falsy   → operator roll-up: count across all
                                   recipient dirs + ``by_recipient``
                                   map + latest_three newest across all.

    Schema (stable for Phase D / future consumers):

        scoped:
          {count: int, latest_three: [{subject, sent_at_utc, priority,
                                        lane_id_hint, pr_hint}],
           unresolved_count: int, latest_unresolved_three: [...]}
        roll-up:
          {count: int, by_recipient: {<session>: int}, latest_three: [...],
           unresolved_count: int, unresolved_by_recipient: {<session>: int},
           latest_unresolved_three: [...]}

    Acknowledged messages live in the per-recipient ``_acked/`` subdir
    (Phase D convention; the directory name starts with ``_`` so this
    glob silently ignores it via ``*.json`` only matching the inbox
    top level).
    """

    if steering_root is None:
        steering_root = REPO_ROOT / ".aragora" / "operator-steering"
    if not steering_root.is_dir():
        if session_name:
            return {
                "count": 0,
                "latest_three": [],
                "unresolved_count": 0,
                "latest_unresolved_three": [],
            }
        return {
            "count": 0,
            "by_recipient": {},
            "latest_three": [],
            "unresolved_count": 0,
            "unresolved_by_recipient": {},
            "latest_unresolved_three": [],
        }

    def _message_key(path: Path) -> tuple[str, str]:
        try:
            message = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            message = {}
        if not isinstance(message, dict):
            message = {}
        return (path.name, str(message.get("message_sha256") or ""))

    def _summary_from(path: Path) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {
                "subject": "(unreadable)",
                "sent_at_utc": "",
                "priority": "",
                "lane_id_hint": None,
                "pr_hint": None,
            }
        if not isinstance(data, dict):
            return {
                "subject": "(invalid)",
                "sent_at_utc": "",
                "priority": "",
                "lane_id_hint": None,
                "pr_hint": None,
            }
        return {
            "subject": str(data.get("subject") or ""),
            "sent_at_utc": str(data.get("sent_at_utc") or ""),
            "priority": str(data.get("priority") or ""),
            "lane_id_hint": data.get("lane_id_hint"),
            "pr_hint": data.get("pr_hint"),
        }

    def _inbox_files(dir_path: Path) -> list[Path]:
        if not dir_path.is_dir():
            return []
        # Only count top-level *.json files — _acked/ subdir holds
        # consumed messages per the Phase D convention.
        return [p for p in dir_path.glob("*.json") if p.is_file()]

    def _read_receipt_summary(
        dir_path: Path,
        files: list[Path],
    ) -> tuple[dict[str, Any], set[tuple[str, str]]]:
        receipt_dir = dir_path / "_read_receipts"
        if not receipt_dir.is_dir():
            return (
                {
                    "read_receipt_count": 0,
                    "unread_message_count": len(files),
                    "latest_read_receipt": None,
                },
                set(),
            )

        receipts: list[dict[str, Any]] = []
        read_keys: set[tuple[str, str]] = set()
        resolved_keys: set[tuple[str, str]] = set()
        for receipt_path in receipt_dir.glob("*.json"):
            if not receipt_path.is_file():
                continue
            try:
                data = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            data["_receipt_filename"] = receipt_path.name
            receipts.append(data)
            key = (
                str(data.get("message_filename") or ""),
                str(data.get("message_sha256") or ""),
            )
            read_keys.add(key)
            if str(data.get("outcome") or "").strip().lower() in RESOLVED_STEERING_OUTCOMES:
                resolved_keys.add(key)

        unread = 0
        for message_path in files:
            key = _message_key(message_path)
            if key not in read_keys:
                unread += 1

        receipts.sort(key=lambda r: str(r.get("read_at_utc") or ""), reverse=True)
        latest = None
        if receipts:
            raw = receipts[0]
            latest = {
                "receipt_filename": raw.get("_receipt_filename"),
                "read_at_utc": raw.get("read_at_utc"),
                "read_by_session": raw.get("read_by_session"),
                "message_filename": raw.get("message_filename"),
                "message_sha256": raw.get("message_sha256"),
                "outcome": raw.get("outcome"),
                "subject": raw.get("subject"),
            }
        return (
            {
                "read_receipt_count": len(receipts),
                "unread_message_count": unread,
                "latest_read_receipt": latest,
            },
            resolved_keys,
        )

    def _unresolved_files(files: list[Path], resolved_keys: set[tuple[str, str]]) -> list[Path]:
        return [
            message_path
            for message_path in files
            if _message_key(message_path) not in resolved_keys
        ]

    if session_name:
        inbox_dir = steering_root / session_name
        files = _inbox_files(inbox_dir)
        receipt_summary, resolved_keys = _read_receipt_summary(inbox_dir, files)
        unresolved_files = _unresolved_files(files, resolved_keys)
        summaries = sorted(
            (_summary_from(p) for p in files),
            key=lambda s: s["sent_at_utc"],
            reverse=True,
        )
        unresolved_summaries = sorted(
            (_summary_from(p) for p in unresolved_files),
            key=lambda s: s["sent_at_utc"],
            reverse=True,
        )
        return {
            "count": len(files),
            "latest_three": summaries[:3],
            "unresolved_count": len(unresolved_files),
            "latest_unresolved_three": unresolved_summaries[:3],
            **receipt_summary,
        }

    # Roll-up across all recipient dirs.
    by_recipient: dict[str, int] = {}
    unresolved_by_recipient: dict[str, int] = {}
    read_receipts_by_recipient: dict[str, int] = {}
    all_summaries: list[dict[str, Any]] = []
    all_unresolved_summaries: list[dict[str, Any]] = []
    total_read_receipts = 0
    total_unread = 0
    total_unresolved = 0
    latest_receipts: list[dict[str, Any]] = []
    for child in sorted(steering_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        files = _inbox_files(child)
        receipt_summary, resolved_keys = _read_receipt_summary(child, files)
        unresolved_files = _unresolved_files(files, resolved_keys)
        if files:
            by_recipient[child.name] = len(files)
            all_summaries.extend(_summary_from(p) for p in files)
        if unresolved_files:
            unresolved_by_recipient[child.name] = len(unresolved_files)
            all_unresolved_summaries.extend(_summary_from(p) for p in unresolved_files)
        total_read_receipts += int(receipt_summary["read_receipt_count"])
        total_unread += int(receipt_summary["unread_message_count"])
        total_unresolved += len(unresolved_files)
        if receipt_summary["read_receipt_count"]:
            read_receipts_by_recipient[child.name] = int(receipt_summary["read_receipt_count"])
        latest = receipt_summary["latest_read_receipt"]
        if isinstance(latest, dict):
            latest_receipts.append(latest)
    all_summaries.sort(key=lambda s: s["sent_at_utc"], reverse=True)
    all_unresolved_summaries.sort(key=lambda s: s["sent_at_utc"], reverse=True)
    latest_receipts.sort(key=lambda r: str(r.get("read_at_utc") or ""), reverse=True)
    return {
        "count": sum(by_recipient.values()),
        "by_recipient": by_recipient,
        "latest_three": all_summaries[:3],
        "unresolved_count": total_unresolved,
        "unresolved_by_recipient": unresolved_by_recipient,
        "latest_unresolved_three": all_unresolved_summaries[:3],
        "read_receipt_count": total_read_receipts,
        "unread_message_count": total_unread,
        "read_receipts_by_recipient": read_receipts_by_recipient,
        "latest_read_receipt": latest_receipts[0] if latest_receipts else None,
    }


def _parse_heartbeat_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _heartbeat_summary(
    row: dict[str, Any],
    *,
    now_dt: datetime,
    freshness_seconds: int,
) -> dict[str, Any]:
    seen = _parse_heartbeat_timestamp(row.get("last_seen_at"))
    age_seconds: int | None = None
    fresh = False
    if seen is not None:
        age_seconds = max(0, int((now_dt - seen).total_seconds()))
        fresh = age_seconds <= freshness_seconds
    return {
        "lane_id": row.get("lane_id"),
        "owner_session": row.get("owner_session"),
        "thread_id": row.get("thread_id"),
        "pid": row.get("pid"),
        "cwd": row.get("cwd"),
        "worktree": row.get("worktree"),
        "branch": row.get("branch"),
        "pr_number": row.get("pr_number"),
        "last_seen_at": row.get("last_seen_at"),
        "age_seconds": age_seconds,
        "fresh": fresh,
    }


def _collect_agent_heartbeats(
    heartbeat_path: Path | None = None,
    *,
    now: str | None = None,
    freshness_seconds: int = HEARTBEAT_FRESH_SECONDS,
) -> dict[str, Any]:
    """Summarize harness heartbeat rows without exposing transcripts."""

    path = heartbeat_path or _heartbeat_file_for_read()
    if not path.exists():
        return {"count": 0, "fresh_count": 0, "stale_count": 0, "latest_by_owner": {}}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"count": 0, "fresh_count": 0, "stale_count": 0, "latest_by_owner": {}}
    rows = [row for row in raw if isinstance(row, dict)] if isinstance(raw, list) else []
    now_dt = _parse_heartbeat_timestamp(now) if now else datetime.now(UTC)
    if now_dt is None:
        now_dt = datetime.now(UTC)
    summaries = [
        _heartbeat_summary(row, now_dt=now_dt, freshness_seconds=freshness_seconds) for row in rows
    ]
    latest_by_owner: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        owner = str(summary.get("owner_session") or "")
        if not owner:
            continue
        existing = latest_by_owner.get(owner)
        summary_seen = _parse_heartbeat_timestamp(summary.get("last_seen_at"))
        existing_seen = (
            _parse_heartbeat_timestamp(existing.get("last_seen_at")) if existing else None
        )
        if existing is None or (
            summary_seen is not None and (existing_seen is None or summary_seen > existing_seen)
        ):
            latest_by_owner[owner] = summary
    return {
        "count": len(summaries),
        "fresh_count": sum(1 for summary in summaries if summary.get("fresh") is True),
        "stale_count": sum(1 for summary in summaries if summary.get("fresh") is False),
        "latest_by_owner": latest_by_owner,
    }


def _operator_pending_steering_count(pending_steering: dict[str, Any]) -> int:
    key = "unresolved_count" if "unresolved_count" in pending_steering else "count"
    try:
        return max(0, int(pending_steering.get(key, 0)))
    except (TypeError, ValueError):
        return 0


def _operator_queue_depth(summary: dict[str, Any], pending_steering: dict[str, Any]) -> int:
    """Return the actionable operator-visible work count for the snapshot."""

    return (
        int(summary.get("active_lanes", 0))
        + int(summary.get("active_broker_runs", 0))
        + _operator_pending_steering_count(pending_steering)
    )


def _operator_summary_int(summary: dict[str, Any], key: str) -> int:
    try:
        return max(0, int(summary.get(key, 0)))
    except (TypeError, ValueError):
        return 0


def _operator_boss_loop_status(summary: dict[str, Any]) -> dict[str, Any]:
    """Explain the legacy boss-loop liveness boolean in operator snapshots."""

    raw_roles = summary.get("active_process_roles") or []
    if isinstance(raw_roles, str):
        raw_roles = [raw_roles]
    try:
        active_roles = sorted(str(role) for role in raw_roles)
    except TypeError:
        active_roles = []
    active_role_set = set(active_roles)
    active_broker_runs = _operator_summary_int(summary, "active_broker_runs")
    fresh_agent_heartbeats = _operator_summary_int(summary, "fresh_agent_heartbeats")
    has_boss_cycle_process = "boss_cycle" in active_role_set

    if active_broker_runs:
        reason = "active_broker_runs"
    elif fresh_agent_heartbeats:
        reason = "fresh_agent_heartbeats"
    elif has_boss_cycle_process:
        reason = "boss_cycle_process"
    else:
        reason = "idle_no_live_boss_loop_signal"

    return {
        "alive": bool(active_broker_runs or fresh_agent_heartbeats or has_boss_cycle_process),
        "reason": reason,
        "active_broker_runs": active_broker_runs,
        "fresh_agent_heartbeats": fresh_agent_heartbeats,
        "has_boss_cycle_process": has_boss_cycle_process,
        "active_process_roles": active_roles,
    }


def _operator_boss_loop_alive(summary: dict[str, Any]) -> bool:
    return bool(_operator_boss_loop_status(summary)["alive"])


def _operator_recent_blockers(
    issues: list[dict[str, str]],
    pending_steering: dict[str, Any],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if _operator_pending_steering_count(pending_steering):
        messages = pending_steering.get("latest_unresolved_three")
        if messages is None:
            messages = pending_steering.get("latest_three", [])
        for message in messages:
            if not isinstance(message, dict):
                continue
            blockers.append(
                {
                    "type": "pending_steering",
                    "source": "operator_steering",
                    "detail": str(message.get("subject") or "(pending steering message)"),
                    "priority": str(message.get("priority") or ""),
                    "lane_id_hint": message.get("lane_id_hint"),
                    "pr_hint": message.get("pr_hint"),
                }
            )

    for issue in issues:
        blockers.append(
            {
                "type": str(issue.get("type") or "health_issue"),
                "source": "health",
                "detail": str(issue.get("detail") or ""),
                "session": str(issue.get("session") or ""),
            }
        )
    return blockers[:limit]


def _coerce_success_rate(raw_rate: Any) -> float | None:
    if raw_rate is None or isinstance(raw_rate, bool):
        return None
    if isinstance(raw_rate, int | float):
        rate = float(raw_rate)
    else:
        try:
            rate = float(str(raw_rate))
        except ValueError:
            return None
    if not math.isfinite(rate) or rate < 0.0 or rate > 1.0:
        return None
    return rate


def _b0_scorecard_timeout_seconds() -> float:
    raw = os.environ.get("AGENT_BRIDGE_B0_SCORECARD_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_B0_SCORECARD_TIMEOUT_SECONDS
    try:
        timeout = float(raw)
    except ValueError:
        return DEFAULT_B0_SCORECARD_TIMEOUT_SECONDS
    return max(0.1, timeout)


def _collect_b0_success_rate(repo_root: Path | None = None) -> float | None:
    root = repo_root or CANONICAL_REPO_ROOT
    scorecard_script = root / "scripts" / "measure_b0_scorecard.py"
    corpus_path = root / "docs" / "benchmarks" / "corpus.json"
    if not scorecard_script.exists() or not corpus_path.exists():
        return None
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(scorecard_script),
                "--json",
                "--corpus",
                str(corpus_path),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=_b0_scorecard_timeout_seconds(),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return _coerce_success_rate(payload.get("no_rescue_success_rate", payload.get("success_rate")))


def _mute_stdout_after_broken_pipe() -> None:
    """Avoid interpreter-shutdown tracebacks after downstream pipes close.

    This is an end-of-process CLI guard; stdout is intentionally redirected
    rather than restored because the downstream reader has already gone away.
    """
    try:
        sys.stdout.close()
    except OSError:
        pass
    sys.stdout = open(os.devnull, "w", encoding="utf-8")


def _emit_text(output: str) -> int:
    try:
        print(output)
    except BrokenPipeError:
        _mute_stdout_after_broken_pipe()
    return 0


def cmd_operator_snapshot(args: argparse.Namespace) -> int:
    """Output a unified operator snapshot combining sessions, lanes, and health."""
    summary_only = bool(getattr(args, "summary_only", False))
    include_historical = bool(getattr(args, "include_historical", False)) or (
        getattr(args, "scope", "current") == "all"
    )
    discovered_sessions, broker_runs, _active_broker_ids = _discover_with_broker_state(
        include_summaries=not summary_only,
        include_historical=include_historical or not summary_only,
    )
    if not include_historical:
        broker_runs = _filter_current_broker_runs(broker_runs)
    sessions = (
        discovered_sessions if include_historical else _filter_current_sessions(discovered_sessions)
    )
    if not summary_only:
        _enrich_prs(discovered_sessions)
        _write_session_snapshot(discovered_sessions)
    records = _sync_lane_records(_load_lane_registry(), sessions)
    if not include_historical:
        records = _filter_current_lane_records(records)

    issues = _collect_health_issues(sessions, records)
    lane_conflicts = _active_lane_identity_conflicts(records)
    computed_conflict_lane_ids = {
        lane_id
        for conflict in lane_conflicts
        for lane_id in conflict.get("lane_ids", [])
        if isinstance(lane_id, str)
    }
    process_census = _collect_agent_process_census(
        include_records=not summary_only,
        record_limit=50,
    )

    # Phase C: surface operator-steering mailbox counts for the
    # current session (if ARAGORA_SESSION_ID set or
    # --steering-recipient passed) or as a roll-up across all
    # recipient dirs (operator view).
    steering_recipient = getattr(args, "steering_recipient", None) or os.environ.get(
        "ARAGORA_SESSION_ID"
    )
    pending_steering = _collect_pending_steering_messages(steering_recipient)
    agent_heartbeats = _collect_agent_heartbeats()
    summary: dict[str, Any] = {
        "total_sessions": len(sessions),
        "alive_sessions": sum(1 for s in sessions if s.status == "alive"),
        "live_sessions": sum(1 for s in sessions if _is_current_session(s)),
        "dead_sessions": sum(1 for s in sessions if s.status == "dead"),
        "historical_sessions": sum(
            1 for s in sessions if (s.lifecycle or s.status) in HISTORICAL_SESSION_LIFECYCLES
        ),
        "active_broker_runs": sum(
            1 for run in broker_runs if run.get("status") in {"running", "awaiting_human"}
        ),
        "active_lanes": sum(1 for r in records if r.status in ACTIVE_LANE_STATUSES),
        "conflict_lanes": sum(1 for r in records if r.status == "conflict")
        + len(computed_conflict_lane_ids),
        "health_issues": len(issues),
        "active_processes": int(process_census.get("total", 0)),
        "active_process_roles": sorted(process_census.get("by_role", {}).keys()),
        "agent_heartbeats": int(agent_heartbeats.get("count", 0)),
        "fresh_agent_heartbeats": int(agent_heartbeats.get("fresh_count", 0)),
    }

    boss_loop_status = _operator_boss_loop_status(summary)
    snapshot: dict[str, Any] = {
        "timestamp": _now_iso(),
        "sessions": [s.to_dict() for s in sessions],
        "broker_runs": broker_runs,
        "lanes": [r.to_dict() for r in records],
        "lane_conflicts": lane_conflicts,
        "process_census": process_census,
        "health": {"ok": len(issues) == 0, "issues": issues},
        "pending_steering_messages": pending_steering,
        "agent_heartbeats": agent_heartbeats,
        "queue_depth": _operator_queue_depth(summary, pending_steering),
        "success_rate": _collect_b0_success_rate(),
        "recent_blockers": _operator_recent_blockers(issues, pending_steering),
        "boss_loop_alive": bool(boss_loop_status["alive"]),
        "boss_loop_status": boss_loop_status,
        "summary": summary,
    }
    if summary_only:
        snapshot.pop("sessions")
        snapshot.pop("lanes")
        snapshot.pop("broker_runs")
        snapshot["agent_heartbeats"] = {
            "count": int(agent_heartbeats.get("count", 0)),
            "fresh_count": int(agent_heartbeats.get("fresh_count", 0)),
            "stale_count": int(agent_heartbeats.get("stale_count", 0)),
        }
        snapshot["records_omitted"] = True

    if args.json:
        return _emit_text(json.dumps(snapshot, indent=2))

    lines: list[str] = [
        f"Operator Snapshot @ {snapshot['timestamp']}",
        "=" * 80,
        f"Sessions: {summary['live_sessions']} live / {summary['historical_sessions']} historical / {summary['total_sessions']} total",
    ]
    lines.append(f"Broker:   {summary['active_broker_runs']} active run(s)")
    lines.append(
        f"Lanes:    {summary['active_lanes']} active / {summary['conflict_lanes']} conflict"
    )
    active_process_roles = [str(role) for role in summary.get("active_process_roles", [])]
    process_roles = ", ".join(active_process_roles) or "-"
    lines.append(f"Processes:{summary['active_processes']} recognized ({process_roles})")
    boss_loop_label = "alive" if boss_loop_status["alive"] else "idle"
    lines.append(f"BossLoop: {boss_loop_label} ({boss_loop_status['reason']})")
    health_status = "OK" if snapshot["health"]["ok"] else f"{summary['health_issues']} issue(s)"
    lines.append(f"Health:   {health_status}")

    if sessions and not summary_only:
        lines.extend(["", f"{'NAME':<24} {'AGENT':<8} {'STATUS':<8} {'BRANCH':<28} SUMMARY"])
        lines.append("-" * 110)
        for s in sessions:
            branch = s.branch[:26] if s.branch else "-"
            summary_text = (s.summary[:40] + "..." if len(s.summary) > 40 else s.summary) or "-"
            lines.append(f"{s.name:<24} {s.agent:<8} {s.status:<8} {branch:<28} {summary_text}")

    if records and not summary_only:
        lines.extend(["", f"{'LANE':<22} {'OWNER':<24} {'STATUS':<10} NEXT ACTION"])
        lines.append("-" * 90)
        for r in records:
            next_action = (
                r.next_action[:40] + "..." if len(r.next_action) > 40 else r.next_action
            ) or "-"
            lines.append(f"{r.lane_id:<22} {r.owner_session:<24} {r.status:<10} {next_action}")

    process_records = snapshot.get("process_census", {}).get("records", [])
    if process_records and not summary_only:
        lines.extend(["", f"{'PID':>8} {'ELAPSED':>12} {'ROLE':<22} SUMMARY"])
        lines.append("-" * 85)
        for process in process_records:
            lines.append(
                f"{int(process['pid']):>8} {str(process['elapsed']):>12} "
                f"{str(process['role']):<22} {process['summary']}"
            )

    if issues:
        lines.extend(["", f"{'TYPE':<22} {'SESSION':<26} DETAIL"])
        lines.append("-" * 100)
        for issue in issues:
            lines.append(f"{issue['type']:<22} {issue['session']:<26} {issue['detail']}")

    return _emit_text("\n".join(lines))


def cmd_tmux_map(args: argparse.Namespace) -> int:
    try:
        result = subprocess.run(
            [
                "tmux",
                "list-panes",
                "-a",
                "-F",
                "#{session_name}:#{window_name} #{pane_pid} #{pane_current_command}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            print("No tmux sessions.")
            return 0
        print(f"{'WINDOW':<40} {'PID':<8} COMMAND")
        print("-" * 65)
        for line in result.stdout.strip().splitlines():
            parts = line.strip().split(None, 2)
            if len(parts) >= 3 and TMUX_SESSION in parts[0]:
                print(f"{parts[0]:<40} {parts[1]:<8} {parts[2]}")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        print("tmux not available.")
    return 0


def _gc_tmux_candidates(*, ttl_hours: int) -> list[dict[str, Any]]:
    if agent_bridge_sessions is None or not TMUX_SESSIONS_DIR.exists():
        return []
    candidates: list[dict[str, Any]] = []
    broker_runs = _load_broker_run_summaries()
    active_broker_session_ids = _active_broker_session_ids(broker_runs)
    records = agent_bridge_sessions.load_tmux_sessions(
        repo_root=CANONICAL_REPO_ROOT,
        tmux_dir=TMUX_SESSIONS_DIR,
        include_summaries=False,
    )
    for record in records:
        lifecycle = _session_lifecycle(
            source=record.source,
            status=record.status,
            updated_at=record.updated_at,
            active_broker_session_ids=active_broker_session_ids,
            session_id=record.session_id,
            ttl_hours=ttl_hours,
        )
        if lifecycle != "stale":
            continue
        meta_path = TMUX_SESSIONS_DIR / f"{record.name}.meta.json"
        files = [meta_path]
        if record.log_file:
            files.append(Path(record.log_file))
        existing_files = [path for path in files if path.exists()]
        if not existing_files:
            continue
        candidates.append(
            {
                "name": record.name,
                "lifecycle": lifecycle,
                "updated_at": record.updated_at,
                "reason": f"dead bridge-owned tmux session older than {ttl_hours}h",
                "files": [str(path) for path in existing_files],
            }
        )
    return candidates


def cmd_gc(args: argparse.Namespace) -> int:
    ttl_hours = max(1, int(args.ttl_hours))
    write = bool(args.write)
    candidates = _gc_tmux_candidates(ttl_hours=ttl_hours)
    archive_dir = TMUX_SESSIONS_DIR / "archive" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    actions: list[dict[str, Any]] = []

    for candidate in candidates:
        archived_files: list[str] = []
        for raw_path in candidate["files"]:
            source = Path(raw_path)
            destination = archive_dir / source.name
            if write:
                archive_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
                archived_files.append(str(destination))
            else:
                archived_files.append(str(destination))
        actions.append(
            {
                "action": "archive_tmux_session",
                "name": candidate["name"],
                "reason": candidate["reason"],
                "dry_run": not write,
                "files": candidate["files"],
                "archive_files": archived_files,
            }
        )

    if write:
        sessions, _broker_runs, _active_broker_ids = _discover_with_broker_state(
            include_summaries=True,
            include_historical=True,
        )
        _write_session_snapshot(sessions)

    payload = {
        "ok": True,
        "dry_run": not write,
        "ttl_hours": ttl_hours,
        "archive_dir": str(archive_dir),
        "actions": actions,
        "external_transcripts_touched": False,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    if not actions:
        print("No stale bridge-owned tmux sessions to archive.")
        return 0
    for action in actions:
        mode = "would archive" if action["dry_run"] else "archived"
        print(f"{mode}: {action['name']} ({len(action['files'])} file(s))")
    return 0


def _json_parent() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    return parent


def main() -> int:
    # SIGPIPE hygiene is scoped to ``launch`` only (review fix #8338 / #8317):
    # other subcommands (sessions/exec/send/approve/read) keep their prior
    # clean SIGPIPE-killed behavior under ``| head`` and must not be affected.
    parser = argparse.ArgumentParser(
        description="Agent bridge: send, approve, read, lanes",
    )
    parser.add_argument("--json", action="store_true")
    sub = parser.add_subparsers(dest="command")
    json_parent = _json_parent()

    sub.add_parser("sessions", parents=[json_parent], help="List sessions")

    launch_p = sub.add_parser(
        "launch",
        parents=[json_parent],
        help="Launch a tmux-managed agent session",
    )
    launch_p.add_argument("--name", required=True)
    launch_p.add_argument(
        "--agent", default="codex", choices=("codex", "claude", "droid", "factory")
    )
    launch_p.add_argument(
        "--cwd",
        help=(
            "Working directory/worktree for the launched harness "
            "(defaults to the caller's current directory)"
        ),
    )
    launch_p.add_argument("prompt", nargs="*")
    launch_p.add_argument("--file", help="Prompt file")
    launch_p.add_argument(
        "--autonomous", action="store_true", help="Grant launcher autonomy where supported"
    )
    launch_p.add_argument("--timeout-seconds", type=int, default=120)
    launch_p.add_argument(
        "--submit-verify-timeout",
        type=float,
        default=DEFAULT_SUBMIT_VERIFY_TIMEOUT_SECONDS,
        help=(
            "Seconds to spend confirming a pasted prompt left the composer "
            "after launch (0 disables submit verification)."
        ),
    )
    launch_p.add_argument(
        "--strict-verify",
        action="store_true",
        help=(
            "Enforce submit verification on the exit code: exit 1 when the "
            "dispatch is not confirmed delivered (still staged or unverifiable). "
            "Off by default -- verification is observational and the tri-state "
            "dispatch receipt is always written regardless of this flag."
        ),
    )

    exec_p = sub.add_parser(
        "exec",
        parents=[json_parent],
        help="Run one non-interactive harness turn",
    )
    exec_p.add_argument("--agent", default="droid", choices=("codex", "claude", "droid", "factory"))
    exec_p.add_argument(
        "--cwd",
        help=(
            "Working directory/worktree for the exec harness "
            "(defaults to the caller's current directory)"
        ),
    )
    exec_p.add_argument("--model", help="Model id to pass to the harness")
    exec_p.add_argument("--auto", choices=("low", "medium", "high"), help="Droid autonomy level")
    exec_p.add_argument("--allowed-role", action="append", help="Allowed bridge footer role")
    exec_p.add_argument("prompt", nargs="*")
    exec_p.add_argument("--file", help="Prompt file")

    send_p = sub.add_parser("send", parents=[json_parent], help="Send prompt to session")
    send_p.add_argument("name")
    send_p.add_argument("prompt", nargs="*")
    send_p.add_argument("--file", help="Prompt file")
    send_p.add_argument("--lane", help="Lane identifier to claim/update")
    send_p.add_argument("--goal", default="", help="Lane goal summary")
    send_p.add_argument("--source", default="", help="Source issue or PR reference")
    send_p.add_argument("--status", default="active", help="Lane status")
    send_p.add_argument("--next-action", default="", help="Next action for the lane")
    send_p.add_argument(
        "--allow-conflict",
        action="store_true",
        help="Mark an explicit conflict instead of rejecting a second active owner",
    )

    approve_p = sub.add_parser("approve", parents=[json_parent], help="Approve Codex permission")
    approve_p.add_argument("name")

    read_p = sub.add_parser("read", parents=[json_parent], help="Read session output")
    read_p.add_argument("name")
    read_p.add_argument("--lines", type=int, default=20)

    ra_p = sub.add_parser("read-all", parents=[json_parent], help="Read all sessions")
    ra_p.add_argument("--lines", type=int, default=5)

    sub.add_parser("lanes", parents=[json_parent], help="Sessions + PR state")
    owner_p = sub.add_parser(
        "owner",
        parents=[json_parent],
        help="Find the active lane owner for a PR, branch, or worktree",
    )
    owner_p.add_argument("--pr", type=int, help="Pull request number to query")
    owner_p.add_argument("--branch", help="Branch name to query")
    owner_p.add_argument("--worktree", help="Worktree path to query")
    processes_p = sub.add_parser(
        "processes",
        parents=[json_parent],
        help="Read-only census of local agent/control-plane processes",
    )
    processes_p.add_argument(
        "--summary-only",
        action="store_true",
        help="Show counts by process role without per-process records.",
    )
    processes_p.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum process records to include when not using --summary-only.",
    )
    sub.add_parser("tmux-map", parents=[json_parent], help="Show tmux panes")
    sub.add_parser(
        "health", parents=[json_parent], help="Check for stale worktrees and lane conflicts"
    ).add_argument(
        "--summary-only",
        action="store_true",
        help="Emit compact health counts and bounded examples for automation probes.",
    )
    gc_p = sub.add_parser(
        "gc",
        parents=[json_parent],
        help="Archive stale bridge-owned tmux metadata/logs; dry-run by default.",
    )
    gc_p.add_argument("--write", action="store_true", help="Apply archive actions")
    gc_p.add_argument("--ttl-hours", type=int, default=DEFAULT_STALE_TTL_HOURS)
    operator_snapshot_p = sub.add_parser(
        "operator-snapshot",
        parents=[json_parent],
        help="Unified operator snapshot (sessions + lanes + health)",
    )
    operator_snapshot_p.add_argument(
        "--summary-only",
        action="store_true",
        help="Omit session and lane records from output for compact automation checks.",
    )
    operator_snapshot_p.add_argument(
        "--include-historical",
        action="store_true",
        help="Include historical Claude/Factory transcript records in the snapshot.",
    )
    operator_snapshot_p.add_argument(
        "--scope",
        choices=("current", "all"),
        default="current",
        help="Snapshot scope. Default 'current' includes live bridge truth only.",
    )
    operator_snapshot_p.add_argument(
        "--steering-recipient",
        default=None,
        metavar="SESSION",
        help=(
            "Scope pending_steering_messages lookup to one recipient "
            "session. Default: env ARAGORA_SESSION_ID, then roll-up across "
            "all recipient inbox dirs."
        ),
    )

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    cmds = {
        "sessions": cmd_sessions,
        "launch": cmd_launch,
        "exec": cmd_exec,
        "send": cmd_send,
        "approve": cmd_approve,
        "read": cmd_read,
        "read-all": cmd_read_all,
        "lanes": cmd_lanes,
        "owner": cmd_owner,
        "processes": cmd_processes,
        "tmux-map": cmd_tmux_map,
        "health": cmd_health,
        "gc": cmd_gc,
        "operator-snapshot": cmd_operator_snapshot,
    }
    return cmds[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
