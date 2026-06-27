#!/usr/bin/env python3
"""Read-only digest of what sibling Codex sessions/automations did recently.

Replaces hand copy-pasting Codex transcripts: reads the local Codex on-disk
state (``aragora.swarm.agent_bridge.codex_source``) and prints a compact,
cross-checkable summary -- recent sessions (what each was doing + where) and
automation ledger steering records (target PR/head, forbidden actions, blockers).

Strictly observational. Touches no repo files and makes no mutations. With
``--live`` it additionally runs read-only ``gh pr view`` to flag when a Codex
prompt is grounded on a stale head or a PR that already merged/closed.

Examples::

    python scripts/codex_bridge_digest.py --hours 12
    python scripts/codex_bridge_digest.py --hours 6 --live --repo synaptent/aragora
    python scripts/codex_bridge_digest.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Allow running as a bare script from anywhere in the repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aragora.swarm.agent_bridge.codex_source import (  # noqa: E402
    CodexLedgerEntry,
    CodexSessionSummary,
    default_codex_home,
    read_ledgers,
    recent_sessions,
)

_TRUNC = 160


def _default_shared_root() -> str | None:
    """The main repo worktree -- a Codex session here is editing shared dirt.

    Resolved portably: ``ARAGORA_SHARED_ROOT`` if set, else the parent of git's
    common dir (the primary worktree even when run from a linked worktree).
    """
    override = os.environ.get("ARAGORA_SHARED_ROOT")
    if override:
        return str(Path(override).expanduser().resolve())
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return str(Path(result.stdout.strip()).parent.resolve())


def _truncate(text: str | None, length: int = _TRUNC) -> str:
    if not text:
        return ""
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= length else collapsed[: length - 1] + "…"


def _gh_pr_view(pr: int, repo: str) -> dict | None:
    try:
        result = subprocess.run(
            [
                "gh",
                "pr",
                "view",
                str(pr),
                "--repo",
                repo,
                "--json",
                "state,headRefOid,isDraft,title",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
    except (ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _session_flags(
    sessions: list[CodexSessionSummary], shared_root: str | None
) -> dict[str, list[str]]:
    """Compute per-session watchlist flags keyed by rollout path."""
    flags: dict[str, list[str]] = {}
    cwd_counts: dict[str, int] = {}
    for session in sessions:
        if session.cwd:
            cwd_counts[session.cwd] = cwd_counts.get(session.cwd, 0) + 1
    for session in sessions:
        session_flags: list[str] = []
        if shared_root and session.cwd == shared_root:
            session_flags.append("operating directly on SHARED ROOT (dirt risk)")
        if session.cwd and cwd_counts.get(session.cwd, 0) > 1:
            session_flags.append(f"shares cwd with {cwd_counts[session.cwd] - 1} other session(s)")
        if session_flags:
            flags[session.rollout_path] = session_flags
    return flags


def _live_ledger_flags(entries: list[CodexLedgerEntry], repo: str) -> dict[int, list[str]]:
    """For distinct target PRs, flag stale-head / already-merged prompts via gh."""
    flags: dict[int, list[str]] = {}
    seen: set[int] = set()
    for entry in entries:
        if entry.pr is None or entry.pr in seen:
            continue
        seen.add(entry.pr)
        view = _gh_pr_view(entry.pr, repo)
        if view is None:
            continue
        pr_flags: list[str] = []
        state = view.get("state")
        if isinstance(state, str) and state.upper() in {"MERGED", "CLOSED"}:
            pr_flags.append(f"PR already {state.upper()} (prompt is stale)")
        live_head = view.get("headRefOid")
        if entry.head and isinstance(live_head, str) and live_head != entry.head:
            pr_flags.append(
                f"head moved {entry.head[:12]} -> {live_head[:12]} (Codex grounded on stale head)"
            )
        if pr_flags:
            flags[entry.pr] = pr_flags
    return flags


def _render_text(
    *,
    home: Path,
    hours: float | None,
    sessions: list[CodexSessionSummary],
    ledgers: list[CodexLedgerEntry],
    session_flags: dict[str, list[str]],
    ledger_flags: dict[int, list[str]],
) -> str:
    out: list[str] = []
    window = "all time" if hours is None else f"last {hours:g}h"
    out.append(f"# Codex bridge digest  (home={home}, window={window})")
    out.append("")

    out.append(f"## Sessions ({len(sessions)})")
    if not sessions:
        out.append("  (none in window)")
    for session in sessions:
        name = session.thread_name or session.session_id[:12]
        origin = session.originator or "?"
        out.append(f"- {name}  [{origin}]  updated={session.updated_at or '?'}")
        out.append(f"    cwd: {session.cwd or '?'}")
        out.append(
            f"    msgs={session.agent_message_count}  last: {_truncate(session.last_agent_message)}"
        )
        for flag in session_flags.get(session.rollout_path, []):
            out.append(f"    ⚠ {flag}")
    out.append("")

    out.append(f"## Automation ledgers ({len(ledgers)})")
    if not ledgers:
        out.append("  (none in window)")
    for entry in ledgers:
        target = f"PR #{entry.pr}" if entry.pr is not None else "(no PR target)"
        head = f" @ {entry.head[:12]}" if entry.head else ""
        out.append(f"- [{entry.automation}] {entry.kind or '?'} -> {target}{head}")
        if entry.reason:
            out.append(f"    reason: {_truncate(entry.reason)}")
        if entry.git_head and entry.git_origin_main and entry.git_head != entry.git_origin_main:
            out.append(
                f"    ⚠ git.head {entry.git_head[:12]} != origin/main {entry.git_origin_main[:12]}"
            )
        if entry.runner_blockers:
            out.append(f"    ⚠ runner_blockers: {entry.runner_blockers}")
        if entry.forbidden_actions:
            out.append(f"    forbidden: {', '.join(entry.forbidden_actions)}")
        for flag in ledger_flags.get(entry.pr, []) if entry.pr is not None else []:
            out.append(f"    ⚠ {flag}")
    out.append("")

    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codex-home",
        type=Path,
        default=None,
        help="Codex home (default: $ARAGORA_CODEX_HOME or ~/.codex)",
    )
    parser.add_argument(
        "--hours", type=float, default=24.0, help="Recency window in hours (0 = all)"
    )
    parser.add_argument("--limit", type=int, default=50, help="Max sessions to show")
    parser.add_argument(
        "--live", action="store_true", help="Cross-check target PRs against gh (read-only)"
    )
    parser.add_argument("--repo", default="synaptent/aragora", help="Repo for --live gh checks")
    parser.add_argument(
        "--shared-root",
        default=None,
        help="Main worktree path to flag as dirt-risk (default: $ARAGORA_SHARED_ROOT or git common dir)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args(argv)

    home = args.codex_home or default_codex_home()
    hours = None if args.hours == 0 else args.hours
    shared_root = args.shared_root or _default_shared_root()

    sessions = recent_sessions(home, hours=hours, limit=args.limit)
    ledgers = read_ledgers(home, hours=hours)
    session_flags = _session_flags(sessions, shared_root)
    ledger_flags = _live_ledger_flags(ledgers, args.repo) if args.live else {}

    if args.json:
        print(
            json.dumps(
                {
                    "codex_home": str(home),
                    "hours": hours,
                    "sessions": [s.to_dict() for s in sessions],
                    "ledgers": [entry.to_dict() for entry in ledgers],
                    "session_flags": session_flags,
                    "ledger_flags": {str(k): v for k, v in ledger_flags.items()},
                },
                indent=2,
            )
        )
    else:
        print(
            _render_text(
                home=home,
                hours=hours,
                sessions=sessions,
                ledgers=ledgers,
                session_flags=session_flags,
                ledger_flags=ledger_flags,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
