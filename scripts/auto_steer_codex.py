#!/usr/bin/env python3
"""Recommend (and, with --apply, write) an advisory steering directive for Codex.

Closes the observe->steer loop so the operator stops hand-relaying recursive
prompts: it reads recent Codex automation ledgers (the bridge) + live gh/git
state, detects steerable conditions, and composes a single monotonic-restrictive
:class:`SteeringDirective` via :func:`aragora.swarm.agent_bridge.auto_steer.build_recommendation`.

Default is a DRY RUN -- it prints the recommendation and writes nothing. Pass
``--apply`` to append it to the steering mailbox (the one mutating action, gated
behind an explicit flag). The directive can only ever ADD caution; the
merge-quorum gate stays the sole authority.

    python3 scripts/auto_steer_codex.py                 # dry run: show recommendation
    python3 scripts/auto_steer_codex.py --apply         # write it to the mailbox
    python3 scripts/auto_steer_codex.py --json           # machine-readable
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aragora.swarm.agent_bridge.auto_steer import SteerSignals  # noqa: E402
from aragora.swarm.agent_bridge.auto_steer import build_recommendation  # noqa: E402
from aragora.swarm.agent_bridge.codex_source import default_codex_home  # noqa: E402
from aragora.swarm.agent_bridge.codex_source import read_ledgers  # noqa: E402
from aragora.swarm.agent_bridge.codex_steer import default_mailbox_path  # noqa: E402
from aragora.swarm.agent_bridge.codex_steer import write_directive  # noqa: E402


def _gh_json(args: list[str], *, timeout: float = 30.0) -> object | None:
    """Run a ``gh`` command returning parsed JSON, or ``None`` on any failure."""
    try:
        proc = subprocess.run(
            ["gh", *args], capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        return json.loads(proc.stdout)
    except ValueError:
        return None


def _gather(
    repo: str, *, since_hours: float, max_ledger_check: int
) -> tuple[SteerSignals, list[str]]:
    warnings: list[str] = []

    # Open PRs -> backlog size + Claude-owned PRs (one call).
    open_prs = _gh_json(
        [
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--limit",
            "400",
            "--json",
            "number,headRefName",
        ]
    )
    open_codex = 0
    claude_owned: list[int] = []
    if isinstance(open_prs, list):
        for pr in open_prs:
            head = str(pr.get("headRefName", ""))
            if head.startswith("codex/"):
                open_codex += 1
            if head.startswith("claude/"):
                num = pr.get("number")
                if isinstance(num, int):
                    claude_owned.append(num)
    else:
        warnings.append("could not read open PR list from gh (backlog/claude-owned unknown)")

    # Ledger target PRs that are no longer open -> stale entries.
    ledger_prs: list[int] = []
    for entry in read_ledgers(default_codex_home(), hours=since_hours):
        if entry.pr is not None and entry.pr > 0:
            ledger_prs.append(entry.pr)
    stale: list[int] = []
    for pr in sorted(set(ledger_prs))[:max_ledger_check]:
        view = _gh_json(["pr", "view", str(pr), "--repo", repo, "--json", "state"], timeout=20.0)
        if isinstance(view, dict) and str(view.get("state", "")).upper() in {"MERGED", "CLOSED"}:
            stale.append(pr)

    signals = SteerSignals(
        issued_at=datetime.now(UTC).isoformat(),
        open_codex_prs=open_codex,
        stale_ledger_prs=tuple(stale),
        claude_owned_prs=tuple(claude_owned),
    )
    return signals, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="synaptent/aragora")
    parser.add_argument("--since-hours", type=float, default=24.0)
    parser.add_argument("--backlog-threshold", type=int, default=140)
    parser.add_argument("--max-ledger-check", type=int, default=40)
    parser.add_argument("--apply", action="store_true", help="Write the directive to the mailbox")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    signals, warnings = _gather(
        args.repo, since_hours=args.since_hours, max_ledger_check=args.max_ledger_check
    )
    signals = SteerSignals(
        issued_at=signals.issued_at,
        open_codex_prs=signals.open_codex_prs,
        backlog_threshold=args.backlog_threshold,
        stale_ledger_prs=signals.stale_ledger_prs,
        claude_owned_prs=signals.claude_owned_prs,
    )
    rec = build_recommendation(signals)

    if args.json:
        print(
            json.dumps(
                {"signals": signals.__dict__, "warnings": warnings, **rec.to_dict()},
                default=list,
                indent=2,
            )
        )
    else:
        for w in warnings:
            print(f"warning: {w}")
        print(
            f"signals: open_codex_prs={signals.open_codex_prs} "
            f"stale_ledger={list(signals.stale_ledger_prs)} "
            f"claude_owned={list(signals.claude_owned_prs)}"
        )
        print("rationale:")
        for r in rec.rationale:
            print(f"  - {r}")
        if rec.directive is not None:
            print("recommended directive:")
            print(json.dumps(rec.directive.to_dict(), indent=2))

    if rec.directive is None:
        print("\nNo directive recommended.")
        return 0
    if args.apply:
        path = write_directive(rec.directive)
        print(f"\nAPPLIED -> {path}")
    else:
        print("\nDRY RUN -- nothing written. Re-run with --apply to write to the mailbox.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
