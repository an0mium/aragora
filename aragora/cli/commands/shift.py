"""CLI command: aragora shift — run a bounded autonomous improvement shift."""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


def register_shift_command(subparsers) -> None:
    """Register the 'shift' subcommand."""
    parser = subparsers.add_parser(
        "shift",
        help="Run a bounded autonomous improvement shift",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=480,
        help="Maximum shift duration in minutes (default: 480 = 8 hours)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=10,
        help="Maximum items to process per shift (default: 10)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=50.0,
        help="Budget limit in USD (default: 50.0)",
    )
    parser.add_argument(
        "--skip-assessment",
        action="store_true",
        help="Skip pre-shift assessment",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run assessment only, don't execute goals",
    )
    parser.set_defaults(func=cmd_shift)


def cmd_shift(args) -> None:
    """Execute the shift command."""
    from pathlib import Path

    from aragora.nomic.shift_controller import ShiftConfig, ShiftController

    config = ShiftConfig(
        max_duration_seconds=args.max_duration * 60,
        max_items_per_shift=args.max_items,
        budget_limit_usd=args.budget,
        require_assessment_before_start=not args.skip_assessment,
    )

    if args.dry_run:
        config.max_items_per_shift = 0

    controller = ShiftController(repo_path=Path("."), config=config)
    result = asyncio.run(controller.run_shift())

    print(f"\nShift {result.shift_id} completed")
    print(f"  Duration: {result.duration_seconds:.1f}s")
    print(
        f"  Items: {result.items_attempted} attempted, "
        f"{result.items_landed} landed, {result.items_failed} failed"
    )
    print(f"  Budget: ${result.budget_spent_usd:.2f} / ${config.budget_limit_usd:.2f}")
    print(f"  Health: {result.health_score_before:.2f} -> {result.health_score_after:.2f}")
    print(f"  Stop reason: {result.stop_reason}")
    if result.error:
        print(f"  Error: {result.error}")
