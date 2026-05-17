#!/usr/bin/env python3
"""Agent bridge: action commands for cross-agent orchestration.

Provides send, approve, read, and lanes commands on top of the session
inventory from agent_bridge_sessions.py (PR #5306).

Usage:
  python3 scripts/agent_bridge.py sessions [--json]
  python3 scripts/agent_bridge.py launch --name codex-review --agent codex --cwd .worktrees/review --file /tmp/prompt.md
  python3 scripts/agent_bridge.py send <name> "Fix the LOC ratchet"
  python3 scripts/agent_bridge.py send <name> --file /tmp/prompt.md
  python3 scripts/agent_bridge.py approve <name>
  python3 scripts/agent_bridge.py read <name> [--lines 20]
  python3 scripts/agent_bridge.py read-all [--lines 3] [--json]
  python3 scripts/agent_bridge.py lanes [--json]
  python3 scripts/agent_bridge.py processes [--json]
  python3 scripts/agent_bridge.py tmux-map
  python3 scripts/agent_bridge.py health [--json]
  python3 scripts/agent_bridge.py operator-snapshot [--json] [--summary-only]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
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
TMUX_SESSIONS_DIR = Path.home() / ".aragora" / "tmux-sessions"
TMUX_SESSION = "aragora"
REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_REPO_ROOT = REPO_ROOT
if agent_bridge_sessions is not None:
    try:
        CANONICAL_REPO_ROOT = agent_bridge_sessions.resolve_canonical_repo_root(REPO_ROOT)
    except (OSError, RuntimeError, ValueError):
        CANONICAL_REPO_ROOT = REPO_ROOT
ACTIVE_LANE_STATUSES = {"active", "running", "pending", "queued", "claimed"}
CURRENT_SESSION_LIFECYCLES = {"live", "active_broker"}
HISTORICAL_SESSION_LIFECYCLES = {"historical", "dead", "stale", "orphaned"}
DEFAULT_STALE_TTL_HOURS = 24


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
    registry_file = _bridge_file_for_read(LANE_REGISTRY_FILE)
    if not registry_file.exists():
        return []
    try:
        payload = json.loads(registry_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(payload, list):
        return []
    return [LaneRecord.from_dict(item) for item in payload if isinstance(item, dict)]


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
            record.branch = live.branch
            record.worktree = live.worktree
            record.pr_number = live.pr_number
    return records


def _load_broker_run_summaries() -> list[dict[str, Any]]:
    try:
        from aragora.swarm.agent_bridge.store import BridgeStore
    except ImportError:
        return []

    try:
        store = BridgeStore(CANONICAL_REPO_ROOT)
        runs = []
        for run_path in store.runs_root().glob("*/run.json"):
            run = store.load_run(run_path.parent.name)
            try:
                registry = store.load_sessions(run.run_id)
                sessions = {role: session.to_dict() for role, session in registry.sessions.items()}
            except (OSError, TypeError, json.JSONDecodeError, KeyError):
                sessions = {}
            runs.append(
                {
                    "run_id": run.run_id,
                    "status": run.status,
                    "updated_at": run.updated_at,
                    "next_actor": run.next_actor,
                    "last_turn_index": run.last_turn_index,
                    "participants": [participant.to_dict() for participant in run.participants],
                    "sessions": sessions,
                }
            )
        runs.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
        return runs
    except (OSError, TypeError, json.JSONDecodeError, KeyError, ValueError):
        return []


def _active_broker_session_ids(broker_runs: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for run in broker_runs:
        if run.get("status") not in ACTIVE_LANE_STATUSES and run.get("status") != "awaiting_human":
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
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
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
                }
            )

    for conflict in _active_lane_identity_conflicts(records):
        issues.append(
            {
                "type": str(conflict["type"]),
                "session": ", ".join(conflict["owner_sessions"]),
                "detail": str(conflict["detail"]),
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


def cmd_launch(args: argparse.Namespace) -> int:
    """Launch a tmux-managed harness lane, then let send/read manage it."""
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

    if args.json:
        print(
            json.dumps(
                {
                    "ok": result.returncode == 0,
                    "name": args.name,
                    "agent": agent,
                    "cwd": str(launch_cwd),
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
                indent=2,
            )
        )
    else:
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)

    return result.returncode


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


def cmd_health(args: argparse.Namespace) -> int:
    """Report stale worktrees, ambiguous lane ownership, and dead sessions."""
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
        print(json.dumps({"ok": len(issues) == 0, "issues": issues}, indent=2))
        return 0 if not issues else 1

    if not issues:
        print("Health OK: no stale worktrees, no lane conflicts.")
        return 0

    print(f"Found {len(issues)} issue(s):\n")
    print(f"{'TYPE':<22} {'SESSION':<26} DETAIL")
    print("-" * 100)
    for issue in issues:
        print(f"{issue['type']:<22} {issue['session']:<26} {issue['detail']}")
    return 1


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
    sessions = (
        discovered_sessions if include_historical else _filter_current_sessions(discovered_sessions)
    )
    if not summary_only:
        _enrich_prs(discovered_sessions)
        _write_session_snapshot(discovered_sessions)
    records = _sync_lane_records(_load_lane_registry(), sessions)

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

    snapshot: dict[str, Any] = {
        "timestamp": _now_iso(),
        "sessions": [s.to_dict() for s in sessions],
        "broker_runs": broker_runs,
        "lanes": [r.to_dict() for r in records],
        "lane_conflicts": lane_conflicts,
        "process_census": process_census,
        "health": {"ok": len(issues) == 0, "issues": issues},
        "summary": {
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
        },
    }
    if summary_only:
        snapshot.pop("sessions")
        snapshot.pop("lanes")
        snapshot.pop("broker_runs")
        snapshot["records_omitted"] = True

    if args.json:
        print(json.dumps(snapshot, indent=2))
        return 0

    summary = snapshot["summary"]
    print(f"Operator Snapshot @ {snapshot['timestamp']}")
    print("=" * 80)
    print(
        f"Sessions: {summary['live_sessions']} live / {summary['historical_sessions']} historical / {summary['total_sessions']} total"
    )
    print(f"Broker:   {summary['active_broker_runs']} active run(s)")
    print(f"Lanes:    {summary['active_lanes']} active / {summary['conflict_lanes']} conflict")
    process_roles = ", ".join(summary["active_process_roles"]) or "-"
    print(f"Processes:{summary['active_processes']} recognized ({process_roles})")
    health_status = "OK" if snapshot["health"]["ok"] else f"{summary['health_issues']} issue(s)"
    print(f"Health:   {health_status}")

    if sessions and not summary_only:
        print(f"\n{'NAME':<24} {'AGENT':<8} {'STATUS':<8} {'BRANCH':<28} SUMMARY")
        print("-" * 110)
        for s in sessions:
            branch = s.branch[:26] if s.branch else "-"
            summary_text = (s.summary[:40] + "..." if len(s.summary) > 40 else s.summary) or "-"
            print(f"{s.name:<24} {s.agent:<8} {s.status:<8} {branch:<28} {summary_text}")

    if records and not summary_only:
        print(f"\n{'LANE':<22} {'OWNER':<24} {'STATUS':<10} NEXT ACTION")
        print("-" * 90)
        for r in records:
            next_action = (
                r.next_action[:40] + "..." if len(r.next_action) > 40 else r.next_action
            ) or "-"
            print(f"{r.lane_id:<22} {r.owner_session:<24} {r.status:<10} {next_action}")

    process_records = snapshot.get("process_census", {}).get("records", [])
    if process_records and not summary_only:
        print(f"\n{'PID':>8} {'ELAPSED':>12} {'ROLE':<22} SUMMARY")
        print("-" * 85)
        for process in process_records:
            print(
                f"{int(process['pid']):>8} {str(process['elapsed']):>12} "
                f"{str(process['role']):<22} {process['summary']}"
            )

    if issues:
        print(f"\n{'TYPE':<22} {'SESSION':<26} DETAIL")
        print("-" * 100)
        for issue in issues:
            print(f"{issue['type']:<22} {issue['session']:<26} {issue['detail']}")

    return 0


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

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    cmds = {
        "sessions": cmd_sessions,
        "launch": cmd_launch,
        "send": cmd_send,
        "approve": cmd_approve,
        "read": cmd_read,
        "read-all": cmd_read_all,
        "lanes": cmd_lanes,
        "processes": cmd_processes,
        "tmux-map": cmd_tmux_map,
        "health": cmd_health,
        "gc": cmd_gc,
        "operator-snapshot": cmd_operator_snapshot,
    }
    return cmds[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
