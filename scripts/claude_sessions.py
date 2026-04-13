#!/usr/bin/env python3
"""Claude Code session inspector and prompt submitter.

Designed for Codex (or any external agent) to:
  1. List all active Claude Code sessions working on aragora
  2. Read the latest conversation from any session
  3. Submit prompts to sessions (via tmux or claude --resume)

Usage:
  # List all aragora sessions with status
  python3 scripts/claude_sessions.py list

  # Read last N messages from a session
  python3 scripts/claude_sessions.py read <session-id> [--lines 20]

  # Read last N messages from ALL active sessions
  python3 scripts/claude_sessions.py read-all [--lines 5]

  # Submit a prompt to a session (via tmux if available, else --resume)
  python3 scripts/claude_sessions.py send <session-id> "Fix the bug in spec.py"
  python3 scripts/claude_sessions.py send <session-id> --file /tmp/prompt.md

  # Show which sessions are in tmux panes
  python3 scripts/claude_sessions.py tmux-map
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
ARAGORA_REPO = "aragora"
TMUX_SESSION = "aragora"
MAX_MESSAGE_CHARS = 2000


@dataclass
class SessionInfo:
    session_id: str
    project_dir: str
    jsonl_path: Path
    cwd: str = ""
    git_branch: str = ""
    last_timestamp: str = ""
    last_activity: str = ""
    message_count: int = 0
    is_running: bool = False
    pid: int | None = None
    tmux_window: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "project_dir": self.project_dir,
            "cwd": self.cwd,
            "git_branch": self.git_branch,
            "last_timestamp": self.last_timestamp,
            "last_activity": self.last_activity,
            "message_count": self.message_count,
            "is_running": self.is_running,
            "pid": self.pid,
            "tmux_window": self.tmux_window,
        }


@dataclass
class ConversationMessage:
    role: str  # user | assistant | tool_result | progress
    text: str
    timestamp: str = ""
    tool_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "text": self.text}
        if self.timestamp:
            d["timestamp"] = self.timestamp
        if self.tool_name:
            d["tool_name"] = self.tool_name
        return d


def _find_aragora_project_dirs() -> list[Path]:
    """Find all project directories that relate to aragora."""
    if not PROJECTS_DIR.exists():
        return []
    dirs: list[Path] = []
    for entry in PROJECTS_DIR.iterdir():
        if not entry.is_dir():
            continue
        name_lower = entry.name.lower()
        if ARAGORA_REPO in name_lower:
            dirs.append(entry)
    return sorted(dirs)


def _find_running_claude_pids() -> dict[str, int]:
    """Map session IDs to PIDs for running Claude processes."""
    session_pids: dict[str, int] = {}
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        for line in result.stdout.splitlines():
            if "claude" not in line.lower():
                continue
            if any(skip in line for skip in ["grep", "claude_sessions.py", "chroma-mcp"]):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            pid = int(parts[1])
            # Try to extract session ID from process args or from lock files
            for part in parts:
                if re.match(r"^[0-9a-f]{8}-", part):
                    session_pids[part] = pid
                    break
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass
    return session_pids


def _find_running_sessions_from_locks() -> dict[str, int]:
    """Find active sessions from security_warnings_state files (indicate recent activity)."""
    sessions: dict[str, int] = {}
    if not CLAUDE_DIR.exists():
        return sessions
    for f in CLAUDE_DIR.glob("security_warnings_state_*.json"):
        session_id = f.stem.replace("security_warnings_state_", "")
        if re.match(r"^[0-9a-f]{8}-", session_id):
            sessions[session_id] = 0
    return sessions


def _get_tmux_window_map() -> dict[str, str]:
    """Map: PID or session marker -> tmux window name."""
    window_map: dict[str, str] = {}
    try:
        result = subprocess.run(
            ["tmux", "list-panes", "-a", "-F", "#{session_name}:#{window_name} #{pane_pid}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    window_name = parts[0]
                    pane_pid = parts[1]
                    if TMUX_SESSION in window_name:
                        window_map[pane_pid] = window_name
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return window_map


def _parse_jsonl_metadata(jsonl_path: Path) -> dict[str, Any]:
    """Read the last few entries to extract session metadata."""
    meta: dict[str, Any] = {
        "cwd": "",
        "git_branch": "",
        "last_timestamp": "",
        "message_count": 0,
        "session_id": "",
    }
    if not jsonl_path.exists():
        return meta

    try:
        lines = jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return meta

    meta["message_count"] = len(lines)

    # Read last 20 lines for metadata
    for line in reversed(lines[-20:]):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not meta["cwd"] and entry.get("cwd"):
            meta["cwd"] = entry["cwd"]
        if not meta["git_branch"] and entry.get("gitBranch"):
            meta["git_branch"] = entry["gitBranch"]
        if not meta["last_timestamp"] and entry.get("timestamp"):
            meta["last_timestamp"] = entry["timestamp"]
        if not meta["session_id"] and entry.get("sessionId"):
            meta["session_id"] = entry["sessionId"]
        if all(meta[k] for k in ("cwd", "git_branch", "last_timestamp", "session_id")):
            break

    return meta


def _extract_text_from_message(msg: Any) -> str:
    """Extract readable text from a Claude message content field."""
    if isinstance(msg, str):
        return msg
    if isinstance(msg, dict):
        # Direct message dict
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
                        tool = block.get("name", "tool")
                        parts.append(f"[tool: {tool}]")
                    elif block.get("type") == "tool_result":
                        parts.append("[tool_result]")
            return "\n".join(parts)
    if isinstance(msg, list):
        parts = []
        for block in msg:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif block.get("type") == "tool_use":
                    parts.append(f"[tool: {block.get('name', 'tool')}]")
        return "\n".join(parts)
    return str(msg)[:200]


def _parse_conversation(jsonl_path: Path, max_messages: int = 20) -> list[ConversationMessage]:
    """Extract the last N human-readable messages from a JSONL log."""
    if not jsonl_path.exists():
        return []

    try:
        lines = jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []

    messages: list[ConversationMessage] = []
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        entry_type = entry.get("type", "")
        timestamp = entry.get("timestamp", "")

        if entry_type == "user":
            msg = entry.get("message", {})
            text = _extract_text_from_message(msg)
            if text.strip():
                messages.append(
                    ConversationMessage(
                        role="user",
                        text=text[:MAX_MESSAGE_CHARS],
                        timestamp=timestamp,
                    )
                )

        elif entry_type == "assistant":
            msg = entry.get("message", {})
            text = _extract_text_from_message(msg)
            if text.strip():
                messages.append(
                    ConversationMessage(
                        role="assistant",
                        text=text[:MAX_MESSAGE_CHARS],
                        timestamp=timestamp,
                    )
                )

    # Return last N
    return messages[-max_messages:]


def discover_sessions() -> list[SessionInfo]:
    """Find all aragora-related Claude Code sessions."""
    project_dirs = _find_aragora_project_dirs()
    running_pids = _find_running_claude_pids()
    lock_sessions = _find_running_sessions_from_locks()
    tmux_map = _get_tmux_window_map()

    sessions: list[SessionInfo] = []
    seen_ids: set[str] = set()

    for proj_dir in project_dirs:
        jsonl_files: list[Path] = []
        for f in proj_dir.glob("*.jsonl"):
            try:
                f.stat()
                jsonl_files.append(f)
            except OSError:
                continue
        jsonl_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        for jsonl_file in jsonl_files:
            session_id = jsonl_file.stem
            if session_id in seen_ids:
                continue

            # Only include sessions modified in the last 24 hours
            try:
                mtime = jsonl_file.stat().st_mtime
                age_hours = (datetime.now().timestamp() - mtime) / 3600
                if age_hours > 24:
                    continue
            except OSError:
                continue

            meta = _parse_jsonl_metadata(jsonl_file)
            actual_id = meta.get("session_id") or session_id
            if actual_id in seen_ids:
                continue
            seen_ids.add(actual_id)

            pid = running_pids.get(actual_id) or lock_sessions.get(actual_id)
            is_running = actual_id in running_pids or actual_id in lock_sessions

            # Check tmux
            tmux_window = None
            if pid:
                tmux_window = tmux_map.get(str(pid))

            info = SessionInfo(
                session_id=actual_id,
                project_dir=proj_dir.name,
                jsonl_path=jsonl_file,
                cwd=meta.get("cwd", ""),
                git_branch=meta.get("git_branch", ""),
                last_timestamp=meta.get("last_timestamp", ""),
                message_count=meta.get("message_count", 0),
                is_running=is_running,
                pid=pid if pid else None,
                tmux_window=tmux_window,
            )

            # Compute last activity summary
            msgs = _parse_conversation(jsonl_file, max_messages=1)
            if msgs:
                last = msgs[-1]
                info.last_activity = f"[{last.role}] {last.text[:80]}..."

            sessions.append(info)

    # Sort by last timestamp (most recent first)
    sessions.sort(key=lambda s: s.last_timestamp or "", reverse=True)
    return sessions


def cmd_list(args: argparse.Namespace) -> int:
    sessions = discover_sessions()
    if not sessions:
        print("No active aragora Claude Code sessions found.")
        return 0

    if args.json:
        print(json.dumps([s.to_dict() for s in sessions], indent=2))
        return 0

    print(f"{'SESSION ID':<40} {'BRANCH':<35} {'STATUS':<10} {'MSGS':>5}  LAST ACTIVITY")
    print("-" * 130)
    for s in sessions:
        status = "RUNNING" if s.is_running else "idle"
        if s.tmux_window:
            status = f"tmux:{s.tmux_window.split(':')[-1]}"
        branch = s.git_branch[:33] if s.git_branch else "-"
        activity = (
            (s.last_activity[:50] + "...")
            if len(s.last_activity) > 50
            else (s.last_activity or "-")
        )
        print(f"{s.session_id:<40} {branch:<35} {status:<10} {s.message_count:>5}  {activity}")

    return 0


def cmd_read(args: argparse.Namespace) -> int:
    sessions = discover_sessions()
    target = args.session_id

    # Allow partial ID match
    matches = [s for s in sessions if s.session_id.startswith(target)]
    if not matches:
        print(f"No session found matching '{target}'", file=sys.stderr)
        return 1
    if len(matches) > 1:
        print(
            f"Ambiguous session ID '{target}', matches: {[s.session_id for s in matches]}",
            file=sys.stderr,
        )
        return 1

    session = matches[0]
    messages = _parse_conversation(session.jsonl_path, max_messages=args.lines)

    if args.json:
        print(
            json.dumps(
                {
                    "session_id": session.session_id,
                    "branch": session.git_branch,
                    "cwd": session.cwd,
                    "messages": [m.to_dict() for m in messages],
                },
                indent=2,
            )
        )
        return 0

    print(f"Session: {session.session_id}")
    print(f"Branch:  {session.git_branch}")
    print(f"CWD:     {session.cwd}")
    print(f"Status:  {'RUNNING' if session.is_running else 'idle'}")
    print("-" * 80)
    for msg in messages:
        prefix = "USER" if msg.role == "user" else "CLAUDE"
        ts = msg.timestamp[:19] if msg.timestamp else ""
        text = msg.text.replace("\n", "\n  ")
        print(f"\n[{prefix}] {ts}")
        print(f"  {text}")

    return 0


def cmd_read_all(args: argparse.Namespace) -> int:
    sessions = discover_sessions()
    if not sessions:
        print("No active aragora sessions found.")
        return 0

    if args.json:
        result = []
        for s in sessions:
            msgs = _parse_conversation(s.jsonl_path, max_messages=args.lines)
            result.append(
                {
                    "session_id": s.session_id,
                    "branch": s.git_branch,
                    "cwd": s.cwd,
                    "is_running": s.is_running,
                    "messages": [m.to_dict() for m in msgs],
                }
            )
        print(json.dumps(result, indent=2))
        return 0

    for s in sessions:
        msgs = _parse_conversation(s.jsonl_path, max_messages=args.lines)
        status = "RUNNING" if s.is_running else "idle"
        print(f"\n{'=' * 80}")
        print(f"Session: {s.session_id}  [{status}]  branch={s.git_branch}")
        print(f"CWD:     {s.cwd}")
        print("-" * 80)
        for msg in msgs:
            prefix = "USER" if msg.role == "user" else "CLAUDE"
            ts = msg.timestamp[:19] if msg.timestamp else ""
            text = msg.text[:200].replace("\n", " ")
            print(f"  [{prefix}] {ts} {text}")
        if not msgs:
            print("  (no messages)")

    return 0


def cmd_send(args: argparse.Namespace) -> int:
    sessions = discover_sessions()
    target = args.session_id

    matches = [s for s in sessions if s.session_id.startswith(target)]
    if not matches:
        print(f"No session found matching '{target}'", file=sys.stderr)
        return 1

    session = matches[0]

    # Resolve prompt text
    if args.file:
        prompt = Path(args.file).read_text(encoding="utf-8")
    elif args.prompt:
        prompt = " ".join(args.prompt)
    else:
        print("No prompt specified. Use positional args or --file", file=sys.stderr)
        return 1

    # Strategy 1: tmux send-keys (if session is in tmux)
    if session.tmux_window:
        return _send_via_tmux(session.tmux_window, prompt)

    # Strategy 2: tmux session with matching name
    tmux_result = _try_tmux_by_branch(session.git_branch, prompt)
    if tmux_result == 0:
        return 0

    # Strategy 3: claude --resume --print (creates new process)
    return _send_via_resume(session, prompt)


def _send_via_tmux(window: str, prompt: str) -> int:
    """Send prompt via tmux paste buffer."""
    try:
        subprocess.run(
            ["tmux", "set-buffer", "-b", "codex-bridge", prompt],
            check=True,
            timeout=5,
        )
        subprocess.run(
            ["tmux", "paste-buffer", "-b", "codex-bridge", "-t", window],
            check=True,
            timeout=5,
        )
        subprocess.run(
            ["tmux", "send-keys", "-t", window, "", "Enter"],
            check=True,
            timeout=5,
        )
        subprocess.run(
            ["tmux", "delete-buffer", "-b", "codex-bridge"],
            check=False,
            timeout=5,
        )
        print(f"Prompt sent via tmux to {window} ({len(prompt)} chars)")
        return 0
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"tmux send failed: {exc}", file=sys.stderr)
        return 1


def _try_tmux_by_branch(branch: str, prompt: str) -> int:
    """Try to find a tmux window matching the branch name."""
    if not branch:
        return 1
    try:
        result = subprocess.run(
            ["tmux", "list-windows", "-t", TMUX_SESSION, "-F", "#{window_index} #{window_name}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return 1
        for line in result.stdout.strip().splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) >= 2 and branch in parts[1]:
                target = f"{TMUX_SESSION}:{parts[0]}"
                return _send_via_tmux(target, prompt)
    except (subprocess.SubprocessError, OSError):
        pass
    return 1


def _send_via_resume(session: SessionInfo, prompt: str) -> int:
    """Send prompt by launching claude --resume --print."""
    cwd = session.cwd or os.getcwd()
    cmd = [
        "claude",
        "--resume",
        session.session_id,
        "--print",
        "-p",
        prompt,
    ]
    print("Sending via claude --resume (new process, non-interactive)...")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if result.stdout.strip():
            print(result.stdout)
        if result.returncode != 0 and result.stderr.strip():
            print(f"stderr: {result.stderr[:500]}", file=sys.stderr)
        return result.returncode
    except subprocess.TimeoutExpired:
        print("claude --resume timed out after 300s", file=sys.stderr)
        return 1
    except (OSError, FileNotFoundError) as exc:
        print(f"Failed to launch claude: {exc}", file=sys.stderr)
        return 1


def cmd_tmux_map(args: argparse.Namespace) -> int:
    """Show which sessions are in tmux panes."""
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
            print("No tmux sessions found.")
            return 0
        print(f"{'TMUX WINDOW':<40} {'PID':<8} {'COMMAND'}")
        print("-" * 70)
        for line in result.stdout.strip().splitlines():
            parts = line.strip().split(None, 2)
            if len(parts) >= 3 and TMUX_SESSION in parts[0]:
                print(f"{parts[0]:<40} {parts[1]:<8} {parts[2]}")
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        print("tmux not available.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Claude Code session inspector for cross-agent orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    sub = parser.add_subparsers(dest="command")

    # list
    sub.add_parser("list", help="List all aragora Claude Code sessions")

    # read
    read_p = sub.add_parser("read", help="Read conversation from a session")
    read_p.add_argument("session_id", help="Session ID (or prefix)")
    read_p.add_argument("--lines", type=int, default=20, help="Number of messages")

    # read-all
    ra_p = sub.add_parser("read-all", help="Read latest from ALL active sessions")
    ra_p.add_argument("--lines", type=int, default=5, help="Messages per session")

    # send
    send_p = sub.add_parser("send", help="Send a prompt to a session")
    send_p.add_argument("session_id", help="Session ID (or prefix)")
    send_p.add_argument("prompt", nargs="*", help="Prompt text")
    send_p.add_argument("--file", help="Read prompt from file")

    # tmux-map
    sub.add_parser("tmux-map", help="Show tmux pane mapping")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    handlers = {
        "list": cmd_list,
        "read": cmd_read,
        "read-all": cmd_read_all,
        "send": cmd_send,
        "tmux-map": cmd_tmux_map,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
