#!/usr/bin/env python3
"""Loop Control Plane v1 - fleet status CLI (read-only).

Unifies Aragora's standing loops (boss loop, merge arbiter, proof-first shift,
publisher, worktree autopilot, nomic) into one budgeted, halt-aware inventory and
reports, per loop, whether it is safe to continue. Computed entirely from
existing read-only surfaces; performs no mutations.

The single governance question per loop: *is it safe to keep running?* A loop
that is merely waiting on not-ready work keeps waiting (``wait``); a loop with an
operational fault or exhausted budget should ``halt``; an unreadable loop is
``report_only`` (fail-closed), never an implied continue.

Examples
--------
::

    python3 scripts/loop_control_status.py
    python3 scripts/loop_control_status.py --json
    python3 scripts/loop_control_status.py --loop merge_arbiter --json
    python3 scripts/loop_control_status.py --no-network --exit-nonzero-on-halt
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aragora.swarm.loop_control import (  # noqa: E402
    LoopKind,
    LoopRecord,
    NextAction,
    summarize,
)
from aragora.swarm.loop_control_io import build_records, collect_all  # noqa: E402


def _parse_kinds(values: list[str] | None) -> list[LoopKind] | None:
    if not values:
        return None
    valid = {kind.value: kind for kind in LoopKind}
    kinds: list[LoopKind] = []
    for value in values:
        if value not in valid:
            raise SystemExit(f"unknown loop kind: {value} (choices: {', '.join(sorted(valid))})")
        kinds.append(valid[value])
    return kinds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Loop Control Plane v1 - fleet status (read-only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo", default=".", help="Path inside the target repository (default: cwd)"
    )
    parser.add_argument(
        "--loop",
        action="append",
        dest="loops",
        metavar="KIND",
        help="Restrict to loop kind(s); repeatable",
    )
    parser.add_argument(
        "--timeout", type=float, default=15.0, help="Per-source collector timeout (seconds)"
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Skip collectors that may reach the network (operator snapshot)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON (records + fleet summary)")
    parser.add_argument(
        "--exit-nonzero-on-halt",
        action="store_true",
        help="Exit 1 when any loop should halt/escalate or the fleet is unsafe to continue",
    )
    return parser


def _print_table(records: list[LoopRecord], summary: dict[str, object], repo_root: Path) -> None:
    print(f"Loop Control Plane @ {repo_root}")
    print("=" * 96)
    print(f"{'LOOP':<20} {'STATE':<16} {'ACTION':<14} {'HALT':<11} {'SRC':<11} BLOCKER")
    print("-" * 96)
    for record in records:
        blocker = record.blocker or "-"
        if len(blocker) > 28:
            blocker = blocker[:25] + "..."
        print(
            f"{record.kind:<20} {record.state:<16} {record.next_action:<14} "
            f"{record.halt_readiness.verdict:<11} {record.source_status:<11} {blocker}"
        )
    print("-" * 96)
    print(
        f"fleet_safe_to_continue: {summary['fleet_safe_to_continue']}  | "
        f"loops={summary['loops']}  by_state={summary['by_state']}"
    )
    gaps = summary.get("halt_readiness_gaps") or []
    if isinstance(gaps, list) and gaps:
        print("halt-readiness gaps:")
        for gap in gaps:
            print(f"  - {gap['loop']} ({gap['verdict']}): {'; '.join(gap['gaps'])}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo).resolve()
    kinds = _parse_kinds(args.loops)
    raw = collect_all(
        repo_root, timeout=args.timeout, allow_network=not args.no_network, kinds=kinds
    )
    records = build_records(raw)
    summary = summarize(records)

    if args.json:
        print(
            json.dumps(
                {
                    "generated_at": datetime.now(tz=timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "repo": str(repo_root),
                    "records": [record.to_dict() for record in records],
                    "summary": summary,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        _print_table(records, summary, repo_root)

    if args.exit_nonzero_on_halt:
        unsafe = (not summary["fleet_safe_to_continue"]) or any(
            record.next_action in (NextAction.HALT.value, NextAction.ESCALATE_HUMAN.value)
            for record in records
        )
        return 1 if unsafe else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
