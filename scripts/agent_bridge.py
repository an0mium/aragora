#!/usr/bin/env python3
"""Unified agent bridge: session listing, prompt dispatch, and lane supervision.

Combines Codex and Claude session discovery, tmux transport, lane tracking,
and GitHub PR state into a single orchestration tool.

Usage:
  # List all active agent sessions (Codex + Claude)
  python3 scripts/agent_bridge.py sessions [--json]

  # Read latest from a specific session
  python3 scripts/agent_bridge.py read <name-or-id> [--lines 10]

  # Read latest from ALL sessions
  python3 scripts/agent_bridge.py read-all [--lines 3] [--json]

  # Send a prompt to a session
  python3 scripts/agent_bridge.py send <name-or-id> "Fix the LOC ratchet"
  python3 scripts/agent_bridge.py send <name-or-id> --file /tmp/prompt.md

  # Show lane status (session -> branch -> PR -> CI)
  python3 scripts/agent_bridge.py lanes [--json]

  # Approve a Codex permission prompt
  python3 scripts/agent_bridge.py approve <name>

  # Show tmux pane map
  python3 scripts/agent_bridge.py tmux-map
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CLAUDE_DIR = Path.home() / ".claude"
PROJECTS_DIR = CLAUDE_DIR / "projects"
TMUX_SESSIONS_DIR = Path.home() / ".aragora" / "tmux-sessions"
TMUX_SESSION_NAME = "aragora"
ARAGORA_REPO_SLUG = "synaptent/aragora"
MAX_TEXT = 2000


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class AgentSession:
    name: str
    agent: str  # codex | claude
    transport: str  # tmux | process
    tmux_target: str = ""  # aragora:window_name
    worktree: str = ""
    branch: str = ""
    session_id: str = ""  # Claude JSONL session ID
    status: str = "unknown"  # alive | idle | dead
    last_activity: str = ""
    message_count: int = 0
    pr_number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "agent": self.agent,
            "transport": self.transport,
            "status": self.status,
            "branch": self.branch,
        }
        if self.tmux_target:
            d["tmux_target"] = self.tmux_target
        if self.worktree:
            d["worktree"] = self.worktree
        if self.session_id:
            d["session_id"] = self.session_id
        if self.last_activity:
            d["last_activity"] = self.last_activity
        if self.message_count:
            d["message_count"] = self.message_count
        if self.pr_number:
            d["pr_number"] = self.pr_number
        return d


# ---------------------------------------------------------------------------
# Session discovery: tmux-managed sessions (Codex + Claude)
# ---------------------------------------------------------------------------


def _discover_tmux_sessions() -> list[AgentSession]:
    """Find sessions from tmux metadata files."""
    sessions: list[AgentSession] = []
    if not TMUX_SESSIONS_DIR.exists():
        return sessions

    # Check which tmux windows are alive
    alive_windows: set[str] = set()
    try:
        result = subprocess.run(
            ["tmux", "list-windows", "-t", TMUX_SESSION_NAME, "-F", "#{window_name}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            alive_windows = set(result.stdout.strip().splitlines())
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    for meta_file in TMUX_SESSIONS_DIR.glob("*.meta.json"):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        name = meta.get("name", meta_file.stem)
        agent = meta.get("agent", "unknown")
        is_alive = name in alive_windows

        # Try to get last log line
        log_file = TMUX_SESSIONS_DIR / f"{name}.log"
        last_activity = ""
        if log_file.exists():
            try:
                lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
                # Strip ANSI and find last meaningful line
                for line in reversed(lines[-20:]):
                    clean = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", line).strip()
                    clean = re.sub(r"\x1b\][^\x07]*\x07", "", clean).strip()
                    if clean and len(clean) > 10 and not clean.startswith("[?"):
                        last_activity = clean[:120]
                        break
            except OSError:
                pass

        sessions.append(
            AgentSession(
                name=name,
                agent=agent,
                transport="tmux",
                tmux_target=f"{TMUX_SESSION_NAME}:{name}" if is_alive else "",
                status="alive" if is_alive else "dead",
                last_activity=last_activity,
            )
        )

    return sessions


# ---------------------------------------------------------------------------
# Session discovery: Claude Code JSONL sessions
# ---------------------------------------------------------------------------


def _discover_claude_sessions() -> list[AgentSession]:
    """Find active Claude Code sessions from JSONL logs."""
    sessions: list[AgentSession] = []
    if not PROJECTS_DIR.exists():
        return sessions

    # Find aragora project dirs
    project_dirs = [d for d in PROJECTS_DIR.iterdir() if d.is_dir() and "aragora" in d.name.lower()]

    # Find running Claude processes
    running_session_ids: set[str] = set()
    for f in CLAUDE_DIR.glob("security_warnings_state_*.json"):
        sid = f.stem.replace("security_warnings_state_", "")
        if re.match(r"^[0-9a-f]{8}-", sid):
            running_session_ids.add(sid)

    seen: set[str] = set()
    for proj_dir in project_dirs:
        jsonl_files: list[Path] = []
        for f in proj_dir.glob("*.jsonl"):
            try:
                f.stat()
                jsonl_files.append(f)
            except OSError:
                continue

        for jsonl_file in sorted(jsonl_files, key=lambda f: f.stat().st_mtime, reverse=True):
            sid = jsonl_file.stem
            if sid in seen:
                continue

            # Only last 24h
            try:
                age_hours = (datetime.now().timestamp() - jsonl_file.stat().st_mtime) / 3600
                if age_hours > 24:
                    continue
            except OSError:
                continue

            seen.add(sid)

            # Parse last few entries for metadata
            meta = _parse_jsonl_tail(jsonl_file)
            actual_id = meta.get("session_id") or sid
            is_running = actual_id in running_session_ids

            # Get last message
            last_msg = _last_assistant_message(jsonl_file)

            sessions.append(
                AgentSession(
                    name=f"claude-{actual_id[:8]}",
                    agent="claude",
                    transport="jsonl",
                    session_id=actual_id,
                    branch=meta.get("git_branch", ""),
                    worktree=meta.get("cwd", ""),
                    status="alive" if is_running else "idle",
                    last_activity=last_msg[:120] if last_msg else "",
                    message_count=meta.get("message_count", 0),
                )
            )

    return sessions


def _parse_jsonl_tail(path: Path, tail: int = 20) -> dict[str, Any]:
    """Parse last N lines of JSONL for metadata."""
    meta: dict[str, Any] = {"message_count": 0}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        meta["message_count"] = len(lines)
        for line in reversed(lines[-tail:]):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key, meta_key in [
                ("cwd", "cwd"),
                ("gitBranch", "git_branch"),
                ("sessionId", "session_id"),
                ("timestamp", "timestamp"),
            ]:
                if not meta.get(meta_key) and entry.get(key):
                    meta[meta_key] = entry[key]
    except OSError:
        pass
    return meta


def _last_assistant_message(path: Path) -> str:
    """Extract the last assistant text message from JSONL."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for line in reversed(lines[-100:]):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("type") != "assistant":
            continue
        msg = entry.get("message", {})
        text = _extract_text(msg)
        if text.strip():
            return text[:MAX_TEXT]
    return ""


def _extract_text(msg: Any) -> str:
    """Extract readable text from a message content field."""
    if isinstance(msg, str):
        return msg
    if isinstance(msg, dict):
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
                    elif block.get("type") == "tool_use":
                        parts.append(f"[tool: {block.get('name', '?')}]")
            return "\n".join(parts)
    return ""


# ---------------------------------------------------------------------------
# Combined discovery
# ---------------------------------------------------------------------------


def discover_all_sessions() -> list[AgentSession]:
    """Merge tmux + Claude sessions, dedup by name."""
    tmux = _discover_tmux_sessions()
    claude = _discover_claude_sessions()

    # Dedup: tmux sessions take priority (they have send capability)
    seen_branches: set[str] = set()
    result: list[AgentSession] = []
    for s in tmux:
        result.append(s)
        if s.branch:
            seen_branches.add(s.branch)

    for s in claude:
        if s.branch and s.branch in seen_branches:
            continue
        result.append(s)

    return result


# ---------------------------------------------------------------------------
# Lane discovery (session -> branch -> PR -> CI)
# ---------------------------------------------------------------------------


def _enrich_with_pr_state(sessions: list[AgentSession]) -> None:
    """Look up PR state for each session's branch."""
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
                "number,headRefName,title,statusCheckRollup,isDraft",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if result.returncode != 0:
            return
        prs = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return

    branch_to_pr: dict[str, dict[str, Any]] = {}
    for pr in prs:
        branch_to_pr[pr.get("headRefName", "")] = pr

    for session in sessions:
        if not session.branch:
            continue
        pr = branch_to_pr.get(session.branch)
        if pr:
            session.pr_number = pr.get("number")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_sessions(args: argparse.Namespace) -> int:
    sessions = discover_all_sessions()
    if args.json:
        print(json.dumps([s.to_dict() for s in sessions], indent=2))
        return 0

    if not sessions:
        print("No active agent sessions found.")
        return 0

    print(f"{'NAME':<24} {'AGENT':<8} {'STATUS':<8} {'BRANCH':<30} LAST ACTIVITY")
    print("-" * 110)
    for s in sessions:
        branch = s.branch[:28] if s.branch else "-"
        activity = (
            s.last_activity[:40] + "..." if len(s.last_activity) > 40 else s.last_activity
        ) or "-"
        print(f"{s.name:<24} {s.agent:<8} {s.status:<8} {branch:<30} {activity}")
    return 0


def cmd_read(args: argparse.Namespace) -> int:
    sessions = discover_all_sessions()
    target = args.name
    matches = [s for s in sessions if target in s.name or target in (s.session_id or "")]
    if not matches:
        print(f"No session matching '{target}'", file=sys.stderr)
        return 1

    session = matches[0]

    # For Claude sessions, read from JSONL
    if session.session_id:
        from scripts.claude_sessions import _parse_conversation, discover_sessions

        claude_sessions = discover_sessions()
        for cs in claude_sessions:
            if cs.session_id == session.session_id:
                messages = _parse_conversation(cs.jsonl_path, max_messages=args.lines)
                print(f"Session: {session.name} ({session.session_id})")
                print(f"Branch:  {session.branch}")
                print(f"Status:  {session.status}")
                print("-" * 80)
                for msg in messages:
                    prefix = "USER" if msg.role == "user" else "CLAUDE"
                    ts = msg.timestamp[:19] if msg.timestamp else ""
                    print(f"\n[{prefix}] {ts}")
                    print(f"  {msg.text[:500]}")
                return 0

    # For tmux sessions, harvest from log
    log_file = TMUX_SESSIONS_DIR / f"{session.name}.log"
    if log_file.exists():
        lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        clean_lines: list[str] = []
        for line in lines[-(args.lines * 5) :]:
            clean = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", line)
            clean = re.sub(r"\x1b\][^\x07]*\x07", "", clean).strip()
            if clean and len(clean) > 5:
                clean_lines.append(clean)
        print(f"Session: {session.name}")
        print(f"Agent:   {session.agent}")
        print(f"Status:  {session.status}")
        print("-" * 80)
        for line in clean_lines[-args.lines :]:
            print(f"  {line[:120]}")
        return 0

    print(f"No readable output for session '{session.name}'")
    return 1


def cmd_read_all(args: argparse.Namespace) -> int:
    sessions = discover_all_sessions()
    if not sessions:
        print("No sessions found.")
        return 0

    if args.json:
        result = []
        for s in sessions:
            entry: dict[str, Any] = s.to_dict()
            entry["recent_output"] = _get_recent_output(s, args.lines)
            result.append(entry)
        print(json.dumps(result, indent=2))
        return 0

    for s in sessions:
        output = _get_recent_output(s, args.lines)
        print(f"\n{'=' * 80}")
        print(f"{s.name} [{s.agent}] [{s.status}] branch={s.branch or '-'}")
        print("-" * 80)
        for line in output:
            print(f"  {line}")
        if not output:
            print("  (no output)")
    return 0


def _get_recent_output(session: AgentSession, lines: int) -> list[str]:
    """Get recent readable output from a session."""
    output: list[str] = []

    # Try tmux log first
    log_file = TMUX_SESSIONS_DIR / f"{session.name}.log"
    if log_file.exists():
        try:
            raw_lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in raw_lines[-(lines * 3) :]:
                clean = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", line)
                clean = re.sub(r"\x1b\][^\x07]*\x07", "", clean).strip()
                if clean and len(clean) > 5 and not clean.startswith("[?"):
                    output.append(clean[:150])
            return output[-lines:]
        except OSError:
            pass

    # Fall back to JSONL for Claude
    if session.session_id:
        for proj_dir in PROJECTS_DIR.iterdir():
            if not proj_dir.is_dir() or "aragora" not in proj_dir.name.lower():
                continue
            jsonl = proj_dir / f"{session.session_id}.jsonl"
            if jsonl.exists():
                last = _last_assistant_message(jsonl)
                if last:
                    output.append(last[:150])
                break

    return output


def cmd_send(args: argparse.Namespace) -> int:
    sessions = discover_all_sessions()
    target = args.name
    matches = [s for s in sessions if target in s.name or target in (s.session_id or "")]
    if not matches:
        print(f"No session matching '{target}'", file=sys.stderr)
        return 1

    session = matches[0]

    # Resolve prompt
    if args.file:
        prompt = Path(args.file).read_text(encoding="utf-8")
    elif args.prompt:
        prompt = " ".join(args.prompt)
    else:
        print("No prompt. Use positional text or --file", file=sys.stderr)
        return 1

    # Send via tmux
    if session.tmux_target:
        return _send_tmux(session.tmux_target, prompt, session.name)

    # Try finding tmux window by name
    try:
        result = subprocess.run(
            [
                "tmux",
                "list-windows",
                "-t",
                TMUX_SESSION_NAME,
                "-F",
                "#{window_index} #{window_name}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = line.strip().split(None, 1)
                if len(parts) >= 2 and session.name in parts[1]:
                    target_window = f"{TMUX_SESSION_NAME}:{parts[0]}"
                    return _send_tmux(target_window, prompt, session.name)
    except (subprocess.TimeoutExpired, OSError):
        pass

    print(f"Session '{session.name}' has no tmux target. Cannot send.", file=sys.stderr)
    return 1


def _send_tmux(target: str, prompt: str, name: str) -> int:
    try:
        if "\n" in prompt:
            subprocess.run(["tmux", "set-buffer", "-b", "bridge", prompt], check=True, timeout=5)
            subprocess.run(
                ["tmux", "paste-buffer", "-b", "bridge", "-t", target], check=True, timeout=5
            )
            subprocess.run(["tmux", "send-keys", "-t", target, "", "Enter"], check=True, timeout=5)
            subprocess.run(["tmux", "delete-buffer", "-b", "bridge"], check=False, timeout=5)
        else:
            subprocess.run(
                ["tmux", "send-keys", "-t", target, prompt, "Enter"], check=True, timeout=5
            )
        print(f"Sent to '{name}' ({len(prompt)} chars)")
        return 0
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"Send failed: {exc}", file=sys.stderr)
        return 1


def cmd_approve(args: argparse.Namespace) -> int:
    """Send 'y' + Enter to approve a Codex permission prompt."""
    sessions = discover_all_sessions()
    matches = [s for s in sessions if args.name in s.name]
    if not matches:
        print(f"No session matching '{args.name}'", file=sys.stderr)
        return 1
    session = matches[0]
    target = session.tmux_target
    if not target:
        # Try by name
        target = f"{TMUX_SESSION_NAME}:{session.name}"
    try:
        subprocess.run(["tmux", "send-keys", "-t", target, "y", "Enter"], check=True, timeout=5)
        print(f"Approved permission in '{session.name}'")
        return 0
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"Approve failed: {exc}", file=sys.stderr)
        return 1


def cmd_lanes(args: argparse.Namespace) -> int:
    """Show lane status: session -> branch -> PR -> CI blocker."""
    sessions = discover_all_sessions()
    _enrich_with_pr_state(sessions)

    if args.json:
        print(json.dumps([s.to_dict() for s in sessions], indent=2))
        return 0

    print(f"{'NAME':<24} {'AGENT':<8} {'STATUS':<8} {'BRANCH':<28} {'PR':>5} ACTIVITY")
    print("-" * 110)
    for s in sessions:
        branch = s.branch[:26] if s.branch else "-"
        pr = f"#{s.pr_number}" if s.pr_number else "-"
        activity = (
            s.last_activity[:30] + "..." if len(s.last_activity) > 30 else s.last_activity
        ) or "-"
        print(f"{s.name:<24} {s.agent:<8} {s.status:<8} {branch:<28} {pr:>5} {activity}")
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
            if len(parts) >= 3 and TMUX_SESSION_NAME in parts[0]:
                print(f"{parts[0]:<40} {parts[1]:<8} {parts[2]}")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        print("tmux not available.")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Agent bridge: session listing, prompt dispatch, lane supervision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="JSON output")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("sessions", help="List all agent sessions")

    read_p = sub.add_parser("read", help="Read output from a session")
    read_p.add_argument("name", help="Session name or ID prefix")
    read_p.add_argument("--lines", type=int, default=20)

    ra_p = sub.add_parser("read-all", help="Read latest from ALL sessions")
    ra_p.add_argument("--lines", type=int, default=5)

    send_p = sub.add_parser("send", help="Send prompt to a session")
    send_p.add_argument("name", help="Session name")
    send_p.add_argument("prompt", nargs="*", help="Prompt text")
    send_p.add_argument("--file", help="Prompt file path")

    approve_p = sub.add_parser("approve", help="Approve Codex permission prompt")
    approve_p.add_argument("name", help="Session name")

    sub.add_parser("lanes", help="Show lane status with PR/CI state")
    sub.add_parser("tmux-map", help="Show tmux pane mapping")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    cmds = {
        "sessions": cmd_sessions,
        "read": cmd_read,
        "read-all": cmd_read_all,
        "send": cmd_send,
        "approve": cmd_approve,
        "lanes": cmd_lanes,
        "tmux-map": cmd_tmux_map,
    }
    return cmds[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
