#!/usr/bin/env python3
"""Agent bridge: action commands for cross-agent orchestration.

Provides send, approve, read, and lanes commands on top of the session
inventory from agent_bridge_sessions.py (PR #5306).

Usage:
  python3 scripts/agent_bridge.py sessions [--json]
  python3 scripts/agent_bridge.py send <name> "Fix the LOC ratchet"
  python3 scripts/agent_bridge.py send <name> --file /tmp/prompt.md
  python3 scripts/agent_bridge.py approve <name>
  python3 scripts/agent_bridge.py read <name> [--lines 20]
  python3 scripts/agent_bridge.py read-all [--lines 3] [--json]
  python3 scripts/agent_bridge.py lanes [--json]
  python3 scripts/agent_bridge.py tmux-map
  python3 scripts/agent_bridge.py health [--json]
  python3 scripts/agent_bridge.py operator-snapshot [--json]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC
from datetime import datetime
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


@dataclass
class Session:
    name: str
    agent: str
    status: str = "unknown"
    tmux_target: str = ""
    branch: str = ""
    worktree: str = ""
    session_id: str = ""
    summary: str = ""
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
        )


def discover() -> list[Session]:
    """Discover all sessions via agent_bridge_sessions.

    Falls back to minimal tmux-only discovery if agent_bridge_sessions
    is unavailable (stale worktree, package-style import, etc.).
    """
    if agent_bridge_sessions is not None:
        records = agent_bridge_sessions.collect_sessions(
            repo_root=REPO_ROOT,
            tmux_dir=TMUX_SESSIONS_DIR,
            claude_projects_root=Path.home() / ".claude" / "projects",
        )
        sessions: list[Session] = []
        for r in records:
            tmux_target = ""
            if r.status == "alive" and r.source == "tmux":
                tmux_target = f"{TMUX_SESSION}:{r.name}"
            sessions.append(
                Session(
                    name=r.name,
                    agent=r.agent,
                    status=r.status,
                    tmux_target=tmux_target,
                    branch=r.branch or "",
                    worktree=r.cwd or "",
                    session_id=r.session_id,
                    summary=r.summary or "",
                )
            )
        return sessions

    # Fallback: minimal tmux-only discovery
    return _discover_tmux_fallback()


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
        sessions.append(
            Session(
                name=name,
                agent=meta.get("agent", "unknown"),
                status="alive" if is_alive else "dead",
                tmux_target=f"{TMUX_SESSION}:{name}" if is_alive else "",
            )
        )
    return sessions


def _write_session_snapshot(sessions: list[Session]) -> None:
    timestamp = datetime.now(UTC).isoformat()
    snapshot = [{"timestamp": timestamp, **s.to_dict()} for s in sessions]
    AGENT_BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = SESSION_SNAPSHOT_FILE.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(SESSION_SNAPSHOT_FILE)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _load_lane_registry() -> list[LaneRecord]:
    if not LANE_REGISTRY_FILE.exists():
        return []
    try:
        payload = json.loads(LANE_REGISTRY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(payload, list):
        return []
    return [LaneRecord.from_dict(item) for item in payload if isinstance(item, dict)]


def _write_lane_registry(records: list[LaneRecord]) -> None:
    AGENT_BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = LANE_REGISTRY_FILE.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps([record.to_dict() for record in records], indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(LANE_REGISTRY_FILE)


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


def _is_repo_root_path(path: str) -> bool:
    try:
        return Path(path).resolve() == CANONICAL_REPO_ROOT.resolve()
    except OSError:
        return False


def _collect_health_issues(
    sessions: list[Session], records: list[LaneRecord]
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []

    # Missing paths are actionable stale-session residue. Dead historical
    # sessions that merely remember the root checkout are not cleanup blockers.
    for s in sessions:
        if s.worktree and not Path(s.worktree).is_dir():
            issues.append(
                {
                    "type": "stale_worktree",
                    "session": s.name,
                    "detail": f"worktree path missing: {s.worktree}",
                }
            )
        if (
            s.status == "dead"
            and s.worktree
            and Path(s.worktree).is_dir()
            and not _is_repo_root_path(s.worktree)
        ):
            issues.append(
                {
                    "type": "stale_worktree",
                    "session": s.name,
                    "detail": f"dead session with lingering worktree: {s.worktree}",
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

    return issues


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
            subprocess.run(["tmux", "set-buffer", "-b", "bridge", prompt], check=True, timeout=5)
            subprocess.run(
                ["tmux", "paste-buffer", "-b", "bridge", "-t", target],
                check=True,
                timeout=5,
            )
            subprocess.run(
                ["tmux", "send-keys", "-t", target, "", "Enter"],
                check=True,
                timeout=5,
            )
            subprocess.run(["tmux", "delete-buffer", "-b", "bridge"], check=False, timeout=5)
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
    sessions = discover()
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
    try:
        subprocess.run(
            ["tmux", "send-keys", "-t", target, "y", "Enter"],
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
    sessions = discover()
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


def cmd_health(args: argparse.Namespace) -> int:
    """Report stale worktrees, ambiguous lane ownership, and dead sessions."""
    sessions = discover()
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
    sessions = discover()
    _enrich_prs(sessions)
    _write_session_snapshot(sessions)
    records = _sync_lane_records(_load_lane_registry(), sessions)

    issues = _collect_health_issues(sessions, records)

    snapshot: dict[str, Any] = {
        "timestamp": _now_iso(),
        "sessions": [s.to_dict() for s in sessions],
        "lanes": [r.to_dict() for r in records],
        "health": {"ok": len(issues) == 0, "issues": issues},
        "summary": {
            "total_sessions": len(sessions),
            "alive_sessions": sum(1 for s in sessions if s.status == "alive"),
            "dead_sessions": sum(1 for s in sessions if s.status == "dead"),
            "active_lanes": sum(1 for r in records if r.status in ACTIVE_LANE_STATUSES),
            "conflict_lanes": sum(1 for r in records if r.status == "conflict"),
            "health_issues": len(issues),
        },
    }

    if args.json:
        print(json.dumps(snapshot, indent=2))
        return 0

    summary = snapshot["summary"]
    print(f"Operator Snapshot @ {snapshot['timestamp']}")
    print("=" * 80)
    print(
        f"Sessions: {summary['alive_sessions']} alive / {summary['dead_sessions']} dead / {summary['total_sessions']} total"
    )
    print(f"Lanes:    {summary['active_lanes']} active / {summary['conflict_lanes']} conflict")
    health_status = "OK" if snapshot["health"]["ok"] else f"{summary['health_issues']} issue(s)"
    print(f"Health:   {health_status}")

    if sessions:
        print(f"\n{'NAME':<24} {'AGENT':<8} {'STATUS':<8} {'BRANCH':<28} SUMMARY")
        print("-" * 110)
        for s in sessions:
            branch = s.branch[:26] if s.branch else "-"
            summary_text = (s.summary[:40] + "..." if len(s.summary) > 40 else s.summary) or "-"
            print(f"{s.name:<24} {s.agent:<8} {s.status:<8} {branch:<28} {summary_text}")

    if records:
        print(f"\n{'LANE':<22} {'OWNER':<24} {'STATUS':<10} NEXT ACTION")
        print("-" * 90)
        for r in records:
            next_action = (
                r.next_action[:40] + "..." if len(r.next_action) > 40 else r.next_action
            ) or "-"
            print(f"{r.lane_id:<22} {r.owner_session:<24} {r.status:<10} {next_action}")

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
    sub.add_parser("tmux-map", parents=[json_parent], help="Show tmux panes")
    sub.add_parser(
        "health", parents=[json_parent], help="Check for stale worktrees and lane conflicts"
    )
    sub.add_parser(
        "operator-snapshot",
        parents=[json_parent],
        help="Unified operator snapshot (sessions + lanes + health)",
    )

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    cmds = {
        "sessions": cmd_sessions,
        "send": cmd_send,
        "approve": cmd_approve,
        "read": cmd_read,
        "read-all": cmd_read_all,
        "lanes": cmd_lanes,
        "tmux-map": cmd_tmux_map,
        "health": cmd_health,
        "operator-snapshot": cmd_operator_snapshot,
    }
    return cmds[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
