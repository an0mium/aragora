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
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aragora.swarm.agent_bridge.codex_source import default_codex_home


def _repo_python_env() -> dict[str, str]:
    """Give child Python processes the same repo import root as this script."""
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(REPO_ROOT)
    )
    return env


def default_sessions_root() -> Path:
    return default_codex_home() / "sessions"


MAX_ROLLOUT_SCAN = 200
_PR_RE = re.compile(r"#(\d{3,6})\b")
_CMD_RE = re.compile(r'"cmd"\s*:\s*"([^"]{0,160})')


def _rollout_session_id(path: Path, meta_id: str | None = None) -> str:
    if meta_id:
        return meta_id
    name = path.name
    if name.startswith("rollout-") and name.endswith(".jsonl"):
        stem = name[len("rollout-") : -len(".jsonl")]
        match = re.match(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-(?P<sid>.+)", stem)
        return match.group("sid") if match else stem
    return path.stem


def _iter_rollouts(
    root: Path,
    *,
    since_hours: float | None = None,
    now: float | None = None,
    limit: int | None = MAX_ROLLOUT_SCAN,
):
    """Yield rollout files newest-first.

    ``limit`` caps the scan to the N most-recent rollouts (the default keeps the
    coordinator view bounded). Pass ``limit=None`` for an exhaustive scan — an
    exact ``--session <id>`` lookup must not silently miss an older session just
    because more than ``MAX_ROLLOUT_SCAN`` rollouts exist.
    """
    cutoff = (
        None
        if since_hours is None
        else (now if now is not None else time.time()) - since_hours * 3600.0
    )
    candidates: list[tuple[float, Path]] = []
    patterns = ("rollout-*.jsonl", "*/*/*/rollout-*.jsonl")
    for pattern in patterns:
        for rollout in root.glob(pattern):
            try:
                mtime = rollout.stat().st_mtime
            except OSError:
                continue
            if cutoff is not None and mtime < cutoff:
                continue
            candidates.append((mtime, rollout))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates if limit is None else candidates[:limit]
    for _, rollout in selected:
        yield rollout


def find_rollout(*, path: str | None, session: str | None, latest: bool, root: Path) -> Path | None:
    """Resolve a rollout file from an explicit path, a session-id substring, or --latest."""
    if path:
        p = Path(path)
        return p if p.is_file() else None
    if latest:
        return next(iter(_iter_rollouts(root)), None)
    if session:
        # Explicit session lookup is an exact request, not a recency scan — scan
        # uncapped so an older session (beyond MAX_ROLLOUT_SCAN) is still found.
        for c in _iter_rollouts(root, limit=None):
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
    rows: list[dict[str, Any]] = []
    for r in _iter_rollouts(root, since_hours=since_hours, now=now):
        turns = extract_turns(r)
        sid = _rollout_session_id(r, turns.get("session_id"))
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
    session_id: str | None = None
    # Real rollouts carry the same human-visible turn as BOTH an ``event_msg``
    # (agent_message/user_message) and a ``response_item`` (role assistant/user);
    # dedupe by text so a turn present in both representations is counted once.
    seen_prompts: set[str] = set()
    seen_decisions: set[str] = set()

    def _add(bucket: list[str], seen: set[str], text: str) -> None:
        if text in seen:
            return
        seen.add(text)
        bucket.append(text)

    try:
        handle = rollout.open("r", encoding="utf-8", errors="replace")
    except OSError:
        handle = None
    if handle is None:
        return {
            "rollout": str(rollout),
            "session_id": _rollout_session_id(rollout),
            "prompts": [],
            "decisions": [],
            "commands": [],
            "prs_referenced": [],
            "counts": {"prompts": 0, "decisions": 0, "commands": 0, "prs": 0},
        }
    with handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (ValueError, TypeError):
                continue
            payload = rec.get("payload") if isinstance(rec.get("payload"), dict) else {}
            if rec.get("type") == "session_meta":
                # Canonical rollouts nest the id under ``payload`` (codex_source.py);
                # tolerate a top-level ``id`` too so either shape resolves.
                sid = payload.get("id") or rec.get("id")
                if isinstance(sid, str) and sid:
                    session_id = sid
                continue
            if rec.get("type") == "event_msg":
                etype = payload.get("type")
                message = payload.get("message")
                if isinstance(message, str) and message.strip():
                    text = message.strip()[:300]
                    if etype == "agent_message":
                        _add(decisions, seen_decisions, text)
                        prs.update(_PR_RE.findall(text))
                    elif etype == "user_message":
                        # Codex stores human prompts as event_msg/user_message;
                        # without this the tool's advertised prompt summary is empty.
                        _add(prompts, seen_prompts, text)
                        prs.update(_PR_RE.findall(text))
                continue
            if rec.get("type") != "response_item":
                continue
            item = payload or rec
            itype = item.get("type")
            if itype == "function_call":
                args = item.get("arguments", "")
                parsed_args = _parse_call_arguments(args)
                if isinstance(parsed_args, dict) and isinstance(parsed_args.get("cmd"), str):
                    cmd = parsed_args["cmd"][:160]
                    arg_text = json.dumps(parsed_args, sort_keys=True, default=str)
                    prs.update(_pr_ids_from_call_args(parsed_args))
                else:
                    arg_text = (
                        args
                        if isinstance(args, str)
                        else json.dumps(args, sort_keys=True, default=str)
                    )
                    m = _CMD_RE.search(arg_text)
                    cmd = m.group(1) if m else f"{item.get('name', 'call')}: {arg_text[:80]}"
                commands.append(cmd)
                prs.update(_PR_RE.findall(arg_text))
            elif item.get("role") in ("user", "assistant"):
                content = item.get("content")
                text = ""
                if isinstance(content, list):
                    text = " ".join(
                        x.get("text", "") for x in content if isinstance(x, dict)
                    ).strip()
                elif isinstance(content, str):
                    text = content.strip()
                if not text:
                    continue
                prs.update(_PR_RE.findall(text))
                if item.get("role") == "user":
                    _add(prompts, seen_prompts, text[:300])
                else:
                    _add(decisions, seen_decisions, text[:300])
    session_id = _rollout_session_id(rollout, session_id)
    return {
        "rollout": str(rollout),
        "session_id": session_id,
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


def _parse_call_arguments(arguments: Any) -> Any:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return arguments
    try:
        return json.loads(arguments)
    except (ValueError, TypeError):
        return arguments


def _pr_ids_from_call_args(arguments: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for key in ("pr", "pr_number", "pull_request", "pull_request_number"):
        value = arguments.get(key)
        if isinstance(value, int):
            ids.add(str(value))
        elif isinstance(value, str) and value.isdecimal():
            # isdecimal (not isdigit): a Unicode-digit like "²" passes isdigit()
            # but raises ValueError under the ``sorted(prs, key=int)`` at the end
            # of extract_turns, crashing the whole digest.
            ids.add(value)
    return ids


def rlm_summary(turns: dict[str, Any], question: str) -> str | None:
    """Best-effort recursive summary via Aragora's RLM CLI; None if unavailable."""
    import tempfile

    child_env = _repo_python_env()
    body = "\n".join(
        ["# Agent session decisions", *turns["decisions"], "# Commands", *turns["commands"]]
    )
    try:
        with tempfile.TemporaryDirectory(prefix="agent-session-digest-") as tmp:
            src_path = Path(tmp) / "session.txt"
            ctx_path = Path(tmp) / "session.ctx.json"
            src_path.write_text(body, encoding="utf-8")
            comp = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "aragora.cli.main",
                    "rlm",
                    "compress",
                    str(src_path),
                    "-o",
                    str(ctx_path),
                    "-t",
                    "document",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                env=child_env,
            )
            if comp.returncode != 0 or not ctx_path.exists():
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
                    str(ctx_path),
                ],
                capture_output=True,
                text=True,
                timeout=180,
                env=child_env,
            )
            if q.returncode != 0:
                # A failed query (e.g. missing context) must not be surfaced as
                # a "summary"; the compress step already guards its returncode.
                return None
            return _extract_rlm_answer(q.stdout)
    except (subprocess.SubprocessError, OSError):
        return None


def _extract_rlm_answer(stdout: str) -> str | None:
    """Return just the ANSWER block from ``aragora rlm query`` stdout.

    The CLI decorates the answer with chrome (``Loaded context…``, ``Query:``,
    ``Strategy:``, separator bars, an ``ANSWER`` banner) and a trailing stats
    footer (``Ready:``/``Confidence:``/``Tokens processed:``…). Surfacing raw
    stdout as the "summary" buries the answer; extract the text between the
    ANSWER banner and the footer separator. Falls back to the stripped stdout
    if the expected markers are absent.
    """
    lines = stdout.splitlines()
    try:
        marker = next(i for i, ln in enumerate(lines) if ln.strip() == "ANSWER")
    except StopIteration:
        return stdout.strip() or None
    body: list[str] = []
    for ln in lines[marker + 1 :]:
        stripped = ln.strip()
        # Skip the ``===`` banner immediately under ANSWER.
        if not body and set(stripped) <= {"="} and stripped:
            continue
        # Stop at the footer: the ``---`` separator or the first stats line.
        if (set(stripped) <= {"-"} and stripped) or stripped.startswith(
            ("Ready:", "Confidence:", "Iteration:", "Tokens processed:", "Sub-calls made:")
        ):
            break
        body.append(ln)
    answer = "\n".join(body).strip()
    return answer or (stdout.strip() or None)


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
        help=(
            "Coordinator view: one-line digest of each rollout in the window "
            f"(up to the {MAX_ROLLOUT_SCAN} most recent; older ones are skipped)"
        ),
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
    ap.add_argument("--sessions-root", default=str(default_sessions_root()))
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = ap.parse_args(argv)

    if args.all:
        if args.rlm is not None:
            # --rlm summarizes a single session; it has no effect in coordinator
            # view. Say so rather than silently ignoring it.
            print("note: --rlm is ignored with --all (coordinator view)", file=sys.stderr)
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
