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
TMUX_SESSIONS_DIR = Path.home() / ".aragora" / "tmux-sessions"
TMUX_SESSION = "aragora"
REPO_ROOT = Path(__file__).resolve().parents[1]


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
    if _send_tmux(target, prompt):
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Agent bridge: send, approve, read, lanes",
    )
    parser.add_argument("--json", action="store_true")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("sessions", help="List sessions")

    send_p = sub.add_parser("send", help="Send prompt to session")
    send_p.add_argument("name")
    send_p.add_argument("prompt", nargs="*")
    send_p.add_argument("--file", help="Prompt file")

    approve_p = sub.add_parser("approve", help="Approve Codex permission")
    approve_p.add_argument("name")

    read_p = sub.add_parser("read", help="Read session output")
    read_p.add_argument("name")
    read_p.add_argument("--lines", type=int, default=20)

    ra_p = sub.add_parser("read-all", help="Read all sessions")
    ra_p.add_argument("--lines", type=int, default=5)

    sub.add_parser("lanes", help="Sessions + PR state")
    sub.add_parser("tmux-map", help="Show tmux panes")

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
    }
    return cmds[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
