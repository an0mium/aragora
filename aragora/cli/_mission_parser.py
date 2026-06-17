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
        description="Ingest a free-text goal, decompose it, and register it to the mission loop.",
    )
    mission_parser.add_argument("goal", help="The high-level goal description")
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
    mission_parser.set_defaults(func=lazy("aragora.cli.commands.mission", "cmd_mission"))
