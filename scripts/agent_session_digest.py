#!/usr/bin/env python3
"""Digest a sibling agent's Codex session rollout into a compact, reviewable summary.

Cross-agent visibility: Codex writes a full per-session transcript to
``~/.codex/sessions/<YYYY>/<MM>/<DD>/rollout-*.jsonl`` (every user prompt, agent
message, and ``exec_command`` call). Those files are large (MBs), so reading one
wholesale blows a reviewer's context. This tool extracts the *relevant* turns
deterministically and — when available — runs them through Aragora's own RLM
(``aragora rlm compress`` + ``query``) for a recursive summary, so reviewing what
a sibling agent did is one command instead of hand-parsing JSONL.

Usage:
    python3 scripts/agent_session_digest.py --latest
    python3 scripts/agent_session_digest.py --session 019f197d --json
    python3 scripts/agent_session_digest.py --path <rollout.jsonl> --rlm "what did it do?"
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_SESSIONS_ROOT = Path.home() / ".codex" / "sessions"
_PR_RE = re.compile(r"#(\d{3,6})\b")
_CMD_RE = re.compile(r'"cmd"\s*:\s*"([^"]{0,160})')


def find_rollout(*, path: str | None, session: str | None, latest: bool, root: Path) -> Path | None:
    """Resolve a rollout file from an explicit path, a session-id substring, or --latest."""
    if path:
        p = Path(path)
        return p if p.is_file() else None
    candidates = sorted(
        root.rglob("rollout-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not candidates:
        return None
    if latest:
        return candidates[0]
    if session:
        for c in candidates:
            if session in c.name:
                return c
    return None


def coordinator_view(
    *, root: Path, since_hours: float, now: float | None = None
) -> list[dict[str, Any]]:
    """Compact per-session digests for every rollout modified within the window.

    The single cross-agent view: run before editing shared files to see what
    sibling agents are touching (which PRs/files) and avoid collisions.
    """
    cutoff = (now if now is not None else time.time()) - since_hours * 3600.0
    rows: list[dict[str, Any]] = []
    for r in sorted(root.rglob("rollout-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            if r.stat().st_mtime < cutoff:
                continue
        except OSError:
            continue
        turns = extract_turns(r)
        sid = r.name.split("-")[-1].replace(".jsonl", "")
        rows.append(
            {
                "session_id": sid,
                "rollout": str(r),
                "counts": turns["counts"],
                "prs_referenced": turns["prs_referenced"],
                "last_decision": (turns["decisions"][-1] if turns["decisions"] else ""),
            }
        )
    return rows


def extract_turns(rollout: Path) -> dict[str, Any]:
    """Deterministically extract the reviewable signal from a rollout JSONL.

    Returns prompts, agent decisions, exec commands, and referenced PRs — without
    loading the full transcript into a reviewer's context.
    """
    prompts: list[str] = []
    decisions: list[str] = []
    commands: list[str] = []
    prs: set[str] = set()
    for line in rollout.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (ValueError, TypeError):
            continue
        if rec.get("type") != "response_item":
            continue
        item = rec.get("payload") or rec
        itype = item.get("type")
        if itype == "function_call":
            args = str(item.get("arguments", ""))
            m = _CMD_RE.search(args)
            cmd = m.group(1) if m else f"{item.get('name', 'call')}: {args[:80]}"
            commands.append(cmd)
            prs.update(_PR_RE.findall(args))
        elif item.get("role") in ("user", "assistant"):
            content = item.get("content")
            text = ""
            if isinstance(content, list):
                text = " ".join(x.get("text", "") for x in content if isinstance(x, dict)).strip()
            elif isinstance(content, str):
                text = content.strip()
            if not text:
                continue
            prs.update(_PR_RE.findall(text))
            (prompts if item.get("role") == "user" else decisions).append(text[:300])
    return {
        "rollout": str(rollout),
        "prompts": prompts,
        "decisions": decisions,
        "commands": commands,
        "prs_referenced": sorted(prs, key=int),
        "counts": {
            "prompts": len(prompts),
            "decisions": len(decisions),
            "commands": len(commands),
            "prs": len(prs),
        },
    }


def rlm_summary(turns: dict[str, Any], question: str) -> str | None:
    """Best-effort recursive summary via Aragora's RLM CLI; None if unavailable."""
    import tempfile

    body = "\n".join(
        ["# Agent session decisions", *turns["decisions"], "# Commands", *turns["commands"]]
    )
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as src:
            src.write(body)
            src_path = src.name
        ctx_path = src_path + ".ctx.json"
        comp = subprocess.run(
            [
                sys.executable,
                "-m",
                "aragora.cli.main",
                "rlm",
                "compress",
                src_path,
                "-o",
                ctx_path,
                "-t",
                "document",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if comp.returncode != 0 or not Path(ctx_path).exists():
            return None
        q = subprocess.run(
            [
                sys.executable,
                "-m",
                "aragora.cli.main",
                "rlm",
                "query",
                question,
                "--context",
                ctx_path,
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        return q.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def render_text(turns: dict[str, Any], summary: str | None) -> str:
    c = turns["counts"]
    lines = [
        f"Session: {turns['rollout']}",
        f"Activity: {c['prompts']} prompts · {c['decisions']} decisions · "
        f"{c['commands']} commands · PRs: {', '.join('#' + p for p in turns['prs_referenced']) or 'none'}",
        "",
        "Key decisions:",
    ]
    lines += [f"  • {d[:160]}" for d in turns["decisions"][:8]]
    if summary:
        lines += ["", "RLM summary:", summary]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Digest a sibling agent's Codex session rollout.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--latest", action="store_true", help="Most recent rollout")
    g.add_argument("--session", help="Match a session-id substring")
    g.add_argument("--path", help="Explicit rollout .jsonl path")
    g.add_argument(
        "--all",
        action="store_true",
        help="Coordinator view: one-line digest of every rollout in the window",
    )
    ap.add_argument(
        "--since-hours",
        type=float,
        default=24.0,
        help="With --all: only rollouts modified within this many hours (default 24)",
    )
    ap.add_argument(
        "--rlm",
        nargs="?",
        const="Summarize the main actions, files/PRs touched, and any duplicated or wasted work.",
        default=None,
        help="Also produce a recursive RLM summary (optional custom question)",
    )
    ap.add_argument("--sessions-root", default=str(DEFAULT_SESSIONS_ROOT))
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = ap.parse_args(argv)

    if args.all:
        lines = coordinator_view(root=Path(args.sessions_root), since_hours=args.since_hours)
        if args.json:
            print(json.dumps(lines, indent=2))
        else:
            if not lines:
                print("no rollouts in window")
            for row in lines:
                c = row["counts"]
                prs = ", ".join("#" + p for p in row["prs_referenced"]) or "none"
                print(
                    f"{row['session_id']}  {c['commands']}cmds {c['decisions']}dec  "
                    f"PRs:{prs}  | {row['last_decision'][:90]}"
                )
        return 0

    rollout = find_rollout(
        path=args.path,
        session=args.session,
        latest=args.latest,
        root=Path(args.sessions_root),
    )
    if rollout is None:
        print("no matching rollout found", file=sys.stderr)
        return 1
    turns = extract_turns(rollout)
    summary = rlm_summary(turns, args.rlm) if args.rlm is not None else None
    if args.json:
        print(json.dumps({**turns, "rlm_summary": summary}, indent=2))
    else:
        print(render_text(turns, summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
