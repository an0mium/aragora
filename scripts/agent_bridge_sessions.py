#!/usr/bin/env python3
"""List Codex/Claude session state for a repository.

This is a read-only bridge inventory tool. It combines:
- tmux-managed session metadata/logs under ``~/.aragora/tmux-sessions``
- Claude Code transcript tails under ``~/.claude/projects/.../*.jsonl``

The goal is to give a small supervisor process one machine-readable view of:
- which sessions exist
- which repo/lane they appear to be working on
- their latest visible local output / conversation turn
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

TAIL_BYTES = 256 * 1024
ANSI_RE = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*(?:\x07|\x1b\\)|\x1b[@-_]",
)
TAG_RE = re.compile(
    r"<(?:system-reminder|task-notification|command-[^>]+|local-command-[^>]+)>.*?</(?:system-reminder|task-notification|command-[^>]+|local-command-[^>]+)>",
    re.DOTALL,
)
SEPARATOR_RE = re.compile(r"^[\s\-\u2500-\u257f=\ufffd]+$")
MODEL_STATUS_RE = re.compile(
    r"^(?:gpt|claude|codex|openai|gemini)-[^\s]+(?:\s+\S+)?\s+·\s+.+$",
    re.I,
)
UI_CHROME_RE = re.compile(
    r"\bnavigate\s+enter\s+select\s+esc\s+cancel\b"
    r"|\bshift\+tab\s+to\s+cycle\b"
    r"|\?\s+for\s+help\b.*\bide\b"
    r"|\balways\s+allow\s+(?:low|medium|high)\s+impact\s+commands\b"
    r"|\bauto\s+\((?:low|medium|high)\)\s+-\s+edits\b"
    r"|\bpermissions\s*dialog\s*dismissed\b",
    re.I,
)


@dataclass(slots=True)
class SessionRecord:
    source: str
    session_id: str
    name: str
    agent: str
    status: str
    updated_at: str | None
    branch: str | None
    cwd: str | None
    prompt_file: str | None
    summary: str
    log_file: str | None = None
    transcript_file: str | None = None
    last_role: str | None = None
    last_user_text: str | None = None
    last_assistant_text: str | None = None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _run_git(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def resolve_canonical_repo_root(path: Path) -> Path:
    """Resolve the shared repo root even when running from a linked worktree."""
    common_dir_proc = _run_git(path, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if common_dir_proc.returncode == 0:
        common_dir = common_dir_proc.stdout.strip()
        if common_dir.endswith("/.git"):
            return Path(common_dir).resolve().parent

    root_proc = _run_git(path, "rev-parse", "--show-toplevel")
    if root_proc.returncode == 0 and root_proc.stdout.strip():
        return Path(root_proc.stdout.strip()).resolve()
    return path.resolve()


def _claude_project_slug(repo_root: Path) -> str:
    normalized = str(repo_root.resolve()).replace("\\", "/").replace(":", "-")
    return normalized.replace("/", "-")


def _repo_match(candidate_root: Path | None, repo_root: Path) -> bool:
    if candidate_root is None:
        return False
    try:
        if candidate_root == repo_root:
            return True
        candidate_root.relative_to(repo_root)
        return True
    except ValueError:
        return False


def _safe_repo_root(candidate: str | None) -> Path | None:
    if not candidate:
        return None
    path = Path(candidate).expanduser()
    try:
        return resolve_canonical_repo_root(path)
    except (OSError, RuntimeError, ValueError):
        if path.exists():
            return path.resolve()
        return path


def _tail_lines(path: Path, *, max_bytes: int = TAIL_BYTES) -> list[str]:
    if not path.exists():
        return []
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[-max_bytes:]
        newline = data.find(b"\n")
        if newline >= 0:
            data = data[newline + 1 :]
    return data.decode("utf-8", errors="replace").splitlines()


def _collapse(text: str, *, max_chars: int = 160) -> str:
    text = ANSI_RE.sub("", text).replace("\r", "")
    text = TAG_RE.sub(" ", text)
    text = "".join(ch if ch >= " " or ch in "\n\t" else " " for ch in text)
    collapsed = " ".join(part.strip() for part in text.splitlines() if part.strip())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3] + "..."


def _clean_display_line(text: str, *, max_chars: int = 220) -> str:
    text = ANSI_RE.sub("", text).replace("\r", "")
    text = TAG_RE.sub(" ", text)
    text = text.replace("…", "...")
    text = "".join(ch if ch >= " " or ch in "\t" else " " for ch in text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[•›■]+\s*", "", text)
    text = re.sub(r"\s*·\s*(?:gpt|claude|openai|gemini|codex)-.*$", "", text, flags=re.I)
    if not text or SEPARATOR_RE.fullmatch(text):
        return ""
    if UI_CHROME_RE.search(text):
        return ""
    if MODEL_STATUS_RE.match(text):
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _summary_score(text: str) -> int:
    if not text or SEPARATOR_RE.fullmatch(text):
        return -100

    lower = text.lower()
    if lower.startswith("conversation interrupted"):
        return -100
    if lower.startswith("/"):
        return -50

    score = 0
    if len(text) >= 20:
        score += 1
    if lower.startswith(("i’m ", "i'm ", "i am ", "hold ", "stay ", "no additional prs")):
        score += 3
    if lower.startswith(("pr #", "the commit/push", "next ", "working", "blocked")):
        score += 2
    if any(
        needle in lower
        for needle in (
            "pr #",
            "opened",
            "pushed",
            "push ",
            "merge",
            "blocked",
            "passed",
            "failed",
            "waiting",
            "monitor",
            "parked",
            "next",
        )
    ):
        score += 2
    if lower.startswith(("ran ", "explored", "read ", "search ", "waited for background terminal")):
        score -= 2
    return score


def _select_summary(lines: list[str]) -> str:
    fallback = ""
    for raw in reversed(lines):
        cleaned = _clean_display_line(raw)
        if not cleaned:
            continue
        if not fallback:
            fallback = cleaned
        if _summary_score(cleaned) > 0:
            return cleaned
    return fallback


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return _collapse(content)
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            if item.strip():
                parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text = str(item.get("text", "")).strip()
            if text:
                parts.append(text)
    return _collapse("\n".join(parts))


def _extract_recent_claude_turns(path: Path) -> dict[str, Any] | None:
    last_user_text: str | None = None
    last_assistant_text: str | None = None
    last_role: str | None = None
    last_text: str | None = None
    updated_at: str | None = None
    cwd: str | None = None
    branch: str | None = None
    session_id = path.stem

    for raw in reversed(_tail_lines(path)):
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue

        if cwd is None:
            raw_cwd = payload.get("cwd")
            if isinstance(raw_cwd, str) and raw_cwd.strip():
                cwd = raw_cwd
        if branch is None:
            raw_branch = payload.get("gitBranch")
            if isinstance(raw_branch, str) and raw_branch.strip():
                branch = raw_branch
        if updated_at is None:
            raw_ts = payload.get("timestamp")
            if isinstance(raw_ts, str) and raw_ts.strip():
                updated_at = raw_ts

        entry_type = payload.get("type")
        if entry_type not in {"user", "assistant"}:
            continue

        message = payload.get("message")
        if not isinstance(message, dict):
            continue
        text = _extract_text(message.get("content"))
        if not text:
            continue

        if last_role is None:
            last_role = str(entry_type)
            last_text = text
        if entry_type == "user" and last_user_text is None:
            last_user_text = text
        elif entry_type == "assistant" and last_assistant_text is None:
            last_assistant_text = text

        if last_user_text and last_assistant_text and last_role and updated_at:
            break

    if last_role is None or last_text is None:
        return None

    return {
        "session_id": session_id,
        "updated_at": updated_at or datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
        "cwd": cwd,
        "branch": branch,
        "last_role": last_role,
        "last_text": last_text,
        "last_user_text": last_user_text,
        "last_assistant_text": last_assistant_text,
    }


def _tmux_alive(name: str, *, session_name: str = "aragora") -> str:
    has_session = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        text=True,
        capture_output=True,
        check=False,
    )
    if has_session.returncode != 0:
        return "dead"

    windows = subprocess.run(
        ["tmux", "list-windows", "-t", session_name, "-F", "#{window_name}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if windows.returncode != 0:
        return "unknown"
    return "alive" if name in windows.stdout.splitlines() else "dead"


def _capture_tmux_summary(name: str, *, session_name: str = "aragora") -> str:
    captured = subprocess.run(
        ["tmux", "capture-pane", "-t", f"{session_name}:{name}", "-p", "-S", "-120"],
        text=True,
        capture_output=True,
        check=False,
    )
    if captured.returncode != 0 or not captured.stdout.strip():
        return ""
    return _select_summary(captured.stdout.splitlines())


def _latest_log_line(log_file: Path) -> str:
    return _select_summary(_tail_lines(log_file, max_bytes=64 * 1024))


def load_tmux_sessions(*, repo_root: Path, tmux_dir: Path) -> list[SessionRecord]:
    records: list[SessionRecord] = []
    if not tmux_dir.exists():
        return records

    for meta_path in sorted(tmux_dir.glob("*.meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(meta, dict):
            continue

        candidate_root = _safe_repo_root(str(meta.get("repo_root", "") or ""))
        if not _repo_match(candidate_root, repo_root):
            continue
        raw_cwd = meta.get("cwd") or meta.get("worktree")
        display_cwd = str(candidate_root) if candidate_root else None
        if isinstance(raw_cwd, str) and raw_cwd.strip():
            cwd_path = Path(raw_cwd).expanduser()
            try:
                resolved_cwd = cwd_path.resolve()
            except OSError:
                resolved_cwd = cwd_path
            if _repo_match(_safe_repo_root(str(resolved_cwd)), repo_root):
                display_cwd = str(resolved_cwd)

        name = meta_path.stem.removesuffix(".meta")
        log_file = Path(str(meta.get("log_file", "") or "")).expanduser()
        updated_at = None
        if log_file.exists():
            updated_at = datetime.fromtimestamp(log_file.stat().st_mtime, UTC).isoformat()
        else:
            started = meta.get("started")
            if isinstance(started, str) and started.strip():
                updated_at = started

        status = _tmux_alive(name)
        summary = _capture_tmux_summary(name) if status == "alive" else ""
        if not summary and log_file.exists():
            summary = _latest_log_line(log_file)

        records.append(
            SessionRecord(
                source="tmux",
                session_id=name,
                name=name,
                agent=str(meta.get("agent", "unknown") or "unknown"),
                status=status,
                updated_at=updated_at,
                branch=None,
                cwd=display_cwd,
                prompt_file=str(meta.get("prompt_file", "") or "") or None,
                summary=summary,
                log_file=str(log_file) if log_file else None,
            )
        )
    return records


def _candidate_claude_logs(*, repo_root: Path, projects_root: Path) -> list[Path]:
    preferred = projects_root / _claude_project_slug(repo_root)
    if preferred.exists():
        return sorted(preferred.glob("*.jsonl"))

    candidates: list[Path] = []
    for path in projects_root.glob("*/*.jsonl"):
        transcript = _extract_recent_claude_turns(path)
        if transcript is None:
            continue
        cwd = transcript.get("cwd")
        if isinstance(cwd, str) and _repo_match(_safe_repo_root(cwd), repo_root):
            candidates.append(path)
    return sorted(candidates)


def load_claude_sessions(*, repo_root: Path, projects_root: Path) -> list[SessionRecord]:
    records: list[SessionRecord] = []
    for jsonl_path in _candidate_claude_logs(repo_root=repo_root, projects_root=projects_root):
        transcript = _extract_recent_claude_turns(jsonl_path)
        if transcript is None:
            continue
        cwd = transcript.get("cwd")
        if isinstance(cwd, str) and not _repo_match(_safe_repo_root(cwd), repo_root):
            continue

        session_id = str(transcript["session_id"])
        branch = transcript.get("branch")
        records.append(
            SessionRecord(
                source="claude_jsonl",
                session_id=session_id,
                name=f"claude-{session_id[:8]}",
                agent="claude",
                status="unknown",
                updated_at=str(transcript.get("updated_at") or ""),
                branch=str(branch) if branch else None,
                cwd=str(cwd) if cwd else None,
                prompt_file=None,
                summary=str(transcript["last_text"]),
                transcript_file=str(jsonl_path),
                last_role=str(transcript.get("last_role") or ""),
                last_user_text=(
                    str(transcript["last_user_text"]) if transcript.get("last_user_text") else None
                ),
                last_assistant_text=(
                    str(transcript["last_assistant_text"])
                    if transcript.get("last_assistant_text")
                    else None
                ),
            )
        )
    return records


def collect_sessions(
    *,
    repo_root: Path,
    tmux_dir: Path,
    claude_projects_root: Path,
    source: str = "all",
    limit: int = 100,
    resolve_repo: bool = True,
) -> list[SessionRecord]:
    normalized_repo = (
        resolve_canonical_repo_root(repo_root) if resolve_repo else repo_root.resolve()
    )
    records: list[SessionRecord] = []
    if source in {"all", "tmux"}:
        records.extend(load_tmux_sessions(repo_root=normalized_repo, tmux_dir=tmux_dir))
    if source in {"all", "claude"}:
        records.extend(
            load_claude_sessions(repo_root=normalized_repo, projects_root=claude_projects_root)
        )
    records.sort(key=lambda item: item.updated_at or "", reverse=True)
    return records[:limit]


def _print_table(repo_root: Path, sessions: list[SessionRecord]) -> None:
    print(f"repo_root: {repo_root}")
    print(f"sessions: {len(sessions)}")
    print("")
    header = f"{'source':<13} {'name':<24} {'agent':<8} {'status':<8} {'branch':<28} summary"
    print(header)
    print("-" * len(header))
    for row in sessions:
        branch = row.branch or "-"
        print(
            f"{row.source:<13} {row.name[:24]:<24} {row.agent[:8]:<8} {row.status[:8]:<8} "
            f"{branch[:28]:<28} {row.summary}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List local Codex/Claude sessions for a repo")
    parser.add_argument("--repo", default=".", help="Repo path or worktree path (default: cwd)")
    parser.add_argument(
        "--source",
        choices=("all", "tmux", "claude"),
        default="all",
        help="Which local session sources to read",
    )
    parser.add_argument("--limit", type=int, default=100, help="Maximum sessions to return")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument(
        "--tmux-dir",
        default=str(Path.home() / ".aragora" / "tmux-sessions"),
        help="tmux metadata/log directory",
    )
    parser.add_argument(
        "--claude-projects-root",
        default=str(Path.home() / ".claude" / "projects"),
        help="Claude project transcript root",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = resolve_canonical_repo_root(Path(args.repo).expanduser())
    sessions = collect_sessions(
        repo_root=repo_root,
        tmux_dir=Path(args.tmux_dir).expanduser(),
        claude_projects_root=Path(args.claude_projects_root).expanduser(),
        source=args.source,
        limit=max(1, int(args.limit)),
        resolve_repo=False,
    )

    payload = {
        "generated_at": _utc_now_iso(),
        "repo_root": str(repo_root),
        "count": len(sessions),
        "sessions": [asdict(item) for item in sessions],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_table(repo_root, sessions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
