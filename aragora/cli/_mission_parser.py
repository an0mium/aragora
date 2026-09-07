"""Argument-parser registration for the ``mission`` subcommand.

Extracted from :mod:`aragora.cli.parser` to keep that module under its LOC
ratchet. This is pure ``argparse`` wiring with no heavy imports; the command
handler stays lazy-loaded via the injected ``lazy`` factory so CLI startup
remains fast.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def add_mission_parser(subparsers: Any, lazy: Callable[[str, str], Callable[..., Any]]) -> None:
    """Add the ``mission`` subcommand parser."""
    mission_parser = subparsers.add_parser(
        "mission",
        help="Run or manage native missions",
        description="Seed, run, resume, inspect, or reconcile native missions.",
    )
    mission_parser.add_argument(
        "mission_action",
        nargs="?",
        help=(
            "Mission action: seed, status, run, resume, or reconcile. Any other value "
            "is treated as a legacy goal alias for 'seed'."
        ),
    )
    mission_parser.add_argument(
        "goal",
        nargs="*",
        help="The high-level goal description for mission seed",
    )
    mission_parser.add_argument(
        "--state",
        help="Path to the mission state JSON. Defaults to .aragora/missions/<id>/state.json for seed.",
    )
    mission_parser.add_argument(
        "--autonomy",
        choices=["report", "safe-clean", "auto-drain"],
        default="report",
        help="Mission autonomy level for run/reconcile.",
    )
    mission_parser.add_argument(
        "--max-ticks",
        type=int,
        default=10_000,
        help="Maximum orchestrator ticks for run/resume.",
    )
    mission_parser.add_argument(
        "--operator-tier",
        type=int,
        default=3,
        help="Tier at or above which live dispatch parks for operator settlement.",
    )
    mission_parser.add_argument(
        "--repo-root",
        help=(
            "Repository root for live auto-drain git/gh checks. Defaults to the git root "
            "containing --state, then the current git root."
        ),
    )
    mission_parser.add_argument(
        "--artifact-fixture",
        help="Load reconcile artifacts from a JSON fixture instead of live inventory helpers.",
    )
    mission_parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum live inventory artifacts to inspect during reconcile.",
    )
    mission_parser.add_argument(
        "--admission-max-unresolved",
        type=int,
        default=0,
        help=(
            "Maximum parked backlog artifacts allowed when seeding producer missions. "
            "Cleanup, evidence, settlement, drain, and repair goals bypass this limit "
            "only when they name backlog objects such as PRs, branches, worktrees, or CI."
        ),
    )
    mission_parser.add_argument(
        "--json", action="store_true", help="Emit JSON output where supported"
    )
    mission_parser.add_argument("--budget", type=float, help="The USD budget limit for the mission")
    mission_parser.add_argument("--max-hours", type=float, help="The maximum run time in hours")
    mission_parser.add_argument(
        "--relay",
        choices=["none", "slack", "email"],
        default="none",
        help="The notification relay channel for hard-halt/park actions",
    )
    mission_parser.add_argument(
        "--auto-settle-max-tier",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help=(
            "Highest merge-quorum tier the mission may settle autonomously (0-2). "
            "Tier-3+ always parks for human risk acceptance — the gate, not the "
            "mission, is the sole authority for Tier-3/4 settlement."
        ),
    )
    mission_parser.add_argument(
        "--tracks", help="Comma-separated focus tracks (e.g. sme,qa,billing)"
    )
    mission_parser.add_argument(
        "--paths",
        help=(
            "Comma-separated repo paths this mission is allowed to mutate during auto-drain "
            "foreign-commit checks."
        ),
    )
    mission_parser.set_defaults(func=lazy("aragora.cli.commands.mission", "cmd_mission"))
