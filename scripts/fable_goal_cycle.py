#!/usr/bin/env python3
"""One bounded goal cycle: package live context, consult Claude Fable 5, emit the next prompt.

This is the connective tissue of the "evolving adaptive factory" loop. Each
cycle the calling agent (normally the Codex conductor) runs this script once:

1. **Package** — gather a bounded snapshot of live state: repo truth (origin
   SHA, open PRs, worktrees, recent plans), recent agent activity via
   ``scripts/agent_session_digest.py``, plus any operator-supplied context
   files and a standing mission statement.
2. **Consult** — send the packet to Claude Fable 5 through the bounded
   consult tool (``consult_claude.py``), asking for ranked next goals, a
   one-cycle plan, and exactly one paste-ready next prompt.
3. **Emit** — persist the packet and response under ``.aragora/goal_cycles/``
   and print the extracted next prompt (or a JSON envelope with ``--json``).

Every context source is best-effort with its own short timeout; failures are
recorded in the packet as gaps rather than aborting the cycle, so Fable knows
what it is *not* seeing. The consult itself is hard-bounded (default 900s).

The response is strategy input, not authority: executing agents remain bound
by the operating contract (tier gates, quorum, anti-treadmill rules).

Examples::

    python3 scripts/fable_goal_cycle.py --goal "Close the loop on settlement throughput"
    python3 scripts/fable_goal_cycle.py --dry-run --json      # build packet only
    python3 scripts/fable_goal_cycle.py --context-file /tmp/cycle-report.md
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_TIMEOUT_SECONDS = 900
CONTEXT_STEP_TIMEOUT_SECONDS = 30
DIGEST_TIMEOUT_SECONDS = 120
MAX_CONTEXT_FILE_BYTES = 64 * 1024
DEFAULT_OUTPUT_DIR = ".aragora/goal_cycles"
DEFAULT_MODEL = "claude-fable-5"
NEXT_PROMPT_HEADING = "## NEXT PROMPT"
NEXT_PROMPT_HEADING_RE = re.compile(
    rf"^{re.escape(NEXT_PROMPT_HEADING)}\s*$", re.IGNORECASE | re.MULTILINE
)
SECTION_HEADING_RE = re.compile(r"^##\s+", re.MULTILINE)
SENSITIVE_CONTEXT_MARKERS = (
    ".env",
    "credential",
    "credentials",
    "id_rsa",
    "passwd",
    "password",
    "private-key",
    "private_key",
    "secret",
    "token",
)

EXIT_OK = 0
EXIT_CONSULT_FAILED = 2
EXIT_NO_NEXT_PROMPT = 5

RESPONSE_FORMAT_INSTRUCTIONS = """\
Respond in exactly this structure (all four sections, these exact headings):

## ASSESSMENT
Your read of the current state in a short paragraph: what is working, what is
blocked, where the highest-leverage gap is.

## NEXT GOALS
Ranked list, at most 3, each one line: goal + why it is the best use of the
next cycles.

## NEXT PLAN
Bounded steps for ONE cycle only (not a roadmap). Each step must be completable
and verifiable within that cycle.

## NEXT PROMPT
Exactly one paste-ready prompt for the executing agent, inside a single fenced
code block. It must be self-contained (the agent starts from live repo truth,
not this conversation), name concrete targets (PR numbers, files, commands),
and end with reporting requirements.

Constraints on your recommendations:
- The executing agent is bound by docs/AGENT_OPERATING_CONTRACT.md: tier gates
  and model quorum decide what merges; no --admin, no force-push, no workflow
  or branch-protection edits, shared checkout stays read-only.
- One bounded progress unit per cycle; if a blocker repeats, switch progress
  class rather than retrying.
- Prefer finishing and settling in-flight work over starting new work.
- Your output is strategy input, not authority: nothing you write authorizes
  merging, settlement, or evidence posting outside normal gates.
"""


def _run(command: list[str], timeout: float, cwd: Path | None = None) -> tuple[bool, str]:
    """Run one bounded read-only command; never raises."""
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "").strip()[-500:]
    return True, proc.stdout.strip()


def _repo_root() -> Path:
    ok, out = _run(["git", "rev-parse", "--show-toplevel"], CONTEXT_STEP_TIMEOUT_SECONDS)
    return Path(out) if ok else Path.cwd()


def _find_consult_script(root: Path) -> Path | None:
    candidates = [
        Path(__file__).resolve().parent / "consult_claude.py",
        root / "scripts" / "consult_claude.py",
        Path.home() / ".codex" / "skills" / "consult-fable" / "consult_claude.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _cycle_dir(root: Path, stamp: str) -> Path:
    """Create a collision-safe artifact directory for this cycle."""
    base = root / stamp
    candidate = base
    suffix = 1
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = base.with_name(f"{base.name}-{suffix}")
            suffix += 1


def _sensitive_context_path(path: Path) -> bool:
    lower_parts = [part.lower() for part in path.parts]
    return any(marker in part for part in lower_parts for marker in SENSITIVE_CONTEXT_MARKERS)


def _read_context_file(path: Path) -> tuple[str | None, str | None]:
    """Read a context file with a hard cap and basic secret-path guard."""
    if _sensitive_context_path(path):
        return None, f"refused potentially sensitive context path: {path}"
    try:
        if not path.is_file():
            return None, f"context file is not a regular file: {path}"
        with path.open("rb") as handle:
            data = handle.read(MAX_CONTEXT_FILE_BYTES + 1)
    except OSError as exc:
        return None, f"context file unreadable: {path}: {exc}"

    truncated = len(data) > MAX_CONTEXT_FILE_BYTES
    body = data[:MAX_CONTEXT_FILE_BYTES].decode("utf-8", errors="replace").strip()
    if truncated:
        note = f"context file truncated to {MAX_CONTEXT_FILE_BYTES} bytes: {path}"
    else:
        note = None
    return body, note


def gather_context(root: Path, since_hours: float, max_prs: int, skip_digest: bool) -> dict:
    """Collect bounded, read-only context sections. Failures become gaps."""
    sections: dict[str, str] = {}
    gaps: list[str] = []

    def section(
        name: str, command: list[str], timeout: float = CONTEXT_STEP_TIMEOUT_SECONDS
    ) -> None:
        ok, out = _run(command, timeout, cwd=root)
        if ok and out:
            sections[name] = out
        else:
            gaps.append(f"{name}: {out or 'empty output'}")

    section("origin/main", ["git", "rev-parse", "origin/main"])
    section("branch status", ["git", "status", "--short", "--branch"])
    section("recent commits (main)", ["git", "log", "--oneline", "-10", "origin/main"])
    section("worktrees", ["git", "worktree", "list"])
    if shutil.which("gh"):
        section(
            f"open non-draft PRs (up to {max_prs})",
            [
                "gh",
                "pr",
                "list",
                "--state",
                "open",
                "--limit",
                str(max_prs),
                "--json",
                "number,title,isDraft,headRefName,updatedAt",
                "--jq",
                "[.[] | select(.isDraft | not)]",
            ],
        )
        section(
            "open epics",
            [
                "gh",
                "issue",
                "list",
                "--state",
                "open",
                "--label",
                "epic",
                "--limit",
                "15",
                "--json",
                "number,title",
            ],
        )
    else:
        gaps.append("gh CLI not on PATH: no PR/epic snapshot")
    ok, out = _run(["ls", "-t", str(root / "docs" / "plans")], CONTEXT_STEP_TIMEOUT_SECONDS)
    if ok and out:
        sections["recent plans (docs/plans, newest first)"] = "\n".join(out.splitlines()[:10])
    else:
        gaps.append(f"docs/plans listing: {out}")

    if skip_digest:
        gaps.append("agent activity digest: skipped by flag")
    else:
        digest_script = root / "scripts" / "agent_session_digest.py"
        if digest_script.is_file():
            ok, out = _run(
                [sys.executable, str(digest_script), "--all", "--since-hours", str(since_hours)],
                DIGEST_TIMEOUT_SECONDS,
                cwd=root,
            )
            if ok and out:
                sections[f"agent activity digest (last {since_hours:g}h)"] = out
            else:
                gaps.append(f"agent activity digest: {out or 'empty output'}")
        else:
            gaps.append("agent activity digest: scripts/agent_session_digest.py not found")

    return {"sections": sections, "gaps": gaps}


def build_packet(
    context: dict,
    goal: str | None,
    extra_files: list[Path],
    since_hours: float,
) -> str:
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts: list[str] = [
        "# Goal-cycle advisory packet",
        f"Generated {now}. You are Claude Fable 5, acting as the strategy",
        "advisor for an autonomous agent loop working on the aragora repo",
        "(Decision Integrity Platform). Recommend the best next goals, a",
        "one-cycle plan, and exactly one next prompt.",
    ]
    if goal:
        parts += ["", "## Standing mission", goal.strip()]
    parts += ["", "## Live state"]
    for name, body in context["sections"].items():
        parts += ["", f"### {name}", "```", body, "```"]
    if context["gaps"]:
        parts += [
            "",
            "## Context gaps (sources that failed or were skipped this cycle)",
            *[f"- {gap}" for gap in context["gaps"]],
        ]
    for path in extra_files:
        body, note = _read_context_file(path)
        if note:
            parts += ["", f"### Operator context {path} ({note})"]
        if body is None:
            continue
        parts += ["", f"### Operator context: {path.name}", body]
    parts += ["", "## Required response format", RESPONSE_FORMAT_INSTRUCTIONS]
    return "\n".join(parts) + "\n"


def extract_next_prompt(response: str) -> str | None:
    """Pull the fenced block out of the NEXT PROMPT section, if present."""
    matches = list(NEXT_PROMPT_HEADING_RE.finditer(response))
    if not matches:
        return None
    match = matches[-1]
    tail = response[match.end() :]
    next_section = SECTION_HEADING_RE.search(tail)
    section = tail[: next_section.start()] if next_section else tail
    fence_open = section.find("```")
    if fence_open < 0:
        # Unfenced fallback: everything after the heading line.
        return section.strip() or None
    after_open = section[fence_open + 3 :]
    # Drop a language tag on the opening fence line.
    after_open = after_open.split("\n", 1)[1] if "\n" in after_open else after_open
    fence_close = after_open.find("```")
    block = after_open[:fence_close] if fence_close >= 0 else after_open
    return block.strip() or None


def run_consult(consult_script: Path, packet_path: Path, model: str, timeout: float) -> dict:
    command = [
        sys.executable,
        str(consult_script),
        "--prompt-file",
        str(packet_path),
        "--model",
        model,
        "--timeout",
        str(timeout),
        "--overall-timeout",
        str(timeout),
        "--json",
    ]
    # Outer bound gives the consult helper a small cleanup/reporting grace
    # around its own overall timeout.
    ok, out = _run(command, timeout + 60)
    if not ok:
        return {"ok": False, "error": f"consult tool failed: {out}"}
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"ok": False, "error": f"consult tool returned non-JSON: {out[:300]}"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--goal", help="Standing mission statement to include in the packet")
    parser.add_argument(
        "--context-file",
        action="append",
        default=[],
        help="Extra context file to include verbatim (repeatable)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Consult timeout in seconds (default {DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--since-hours",
        type=float,
        default=24.0,
        help="Agent-activity digest window (default 24)",
    )
    parser.add_argument("--max-prs", type=int, default=30)
    parser.add_argument("--skip-digest", action="store_true", help="Skip the session digest step")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Cycle artifact root (default <repo>/{DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build the packet, skip the consult")
    parser.add_argument("--json", action="store_true", help="Emit a JSON result envelope")
    args = parser.parse_args(argv)

    root = _repo_root()
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cycle_root = Path(args.output_dir) if args.output_dir else root / DEFAULT_OUTPUT_DIR
    cycle_dir = _cycle_dir(cycle_root, stamp)

    context = gather_context(root, args.since_hours, args.max_prs, args.skip_digest)
    packet = build_packet(
        context, args.goal, [Path(p) for p in args.context_file], args.since_hours
    )
    packet_path = cycle_dir / "packet.md"
    packet_path.write_text(packet, encoding="utf-8")

    result: dict = {
        "cycle_dir": str(cycle_dir),
        "packet": str(packet_path),
        "context_gaps": context["gaps"],
    }

    if args.dry_run:
        result["ok"] = True
        result["dry_run"] = True
        print(json.dumps(result, indent=2) if args.json else f"packet written: {packet_path}")
        return EXIT_OK

    consult_script = _find_consult_script(root)
    if consult_script is None:
        result["ok"] = False
        result["error"] = (
            "consult_claude.py not found (repo scripts/ or ~/.codex/skills/consult-fable/)"
        )
        print(json.dumps(result, indent=2) if args.json else result["error"], file=sys.stderr)
        return EXIT_CONSULT_FAILED

    consult = run_consult(consult_script, packet_path, args.model, args.timeout)
    result["consult"] = {
        k: consult.get(k) for k in ("ok", "model", "backend", "elapsed_s", "error")
    }
    if not consult.get("ok"):
        result["ok"] = False
        print(
            json.dumps(result, indent=2)
            if args.json
            else f"consult failed: {consult.get('error')}",
            file=sys.stderr,
        )
        return EXIT_CONSULT_FAILED

    response = consult["text"]
    response_path = cycle_dir / "response.md"
    response_path.write_text(response + "\n", encoding="utf-8")
    result["response"] = str(response_path)

    next_prompt = extract_next_prompt(response)
    if next_prompt:
        next_prompt_path = cycle_dir / "next_prompt.md"
        next_prompt_path.write_text(next_prompt + "\n", encoding="utf-8")
        result["ok"] = True
        result["next_prompt"] = str(next_prompt_path)
        if args.json:
            result["next_prompt_text"] = next_prompt
            print(json.dumps(result, indent=2))
        else:
            print(next_prompt)
        return EXIT_OK

    result["ok"] = False
    result["error"] = f"response missing '{NEXT_PROMPT_HEADING}' block; see {response_path}"
    print(json.dumps(result, indent=2) if args.json else result["error"], file=sys.stderr)
    return EXIT_NO_NEXT_PROMPT


if __name__ == "__main__":
    sys.exit(main())
