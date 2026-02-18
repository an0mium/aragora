"""
Outcome CLI commands: record and search decision outcomes.

Commands:
- record: Record an outcome for a debate/decision
- search: Search past outcomes by topic or tags
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)


def add_outcome_parser(subparsers: Any) -> None:
    """Register the 'outcome' subcommand with record/search actions."""
    outcome_parser = subparsers.add_parser(
        "outcome",
        help="Record and search decision outcomes",
        description="""
Track real-world outcomes of decisions made through Aragora debates.
Closes the feedback loop: decision -> action -> outcome -> learning.

Subcommands:
  record  Record an outcome for a debate
  search  Search past outcomes

Examples:
  aragora outcome record dbt-123 --type success --description "Vendor delivered on time"
  aragora outcome record dbt-456 --type failure --impact 0.8 --lessons "Need stricter SLA"
  aragora outcome search "vendor selection"
  aragora outcome search --tags hiring,engineering
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    outcome_sub = outcome_parser.add_subparsers(dest="outcome_command")

    # --- record ---
    record_parser = outcome_sub.add_parser(
        "record",
        help="Record an outcome for a debate/decision",
    )
    record_parser.add_argument("debate_id", help="Debate ID to record outcome for")
    record_parser.add_argument(
        "--type",
        "-t",
        required=True,
        choices=["success", "failure", "partial", "unknown"],
        help="Outcome type",
    )
    record_parser.add_argument(
        "--description",
        "-d",
        required=True,
        help="Description of the outcome",
    )
    record_parser.add_argument(
        "--impact",
        type=float,
        default=0.5,
        help="Impact score 0.0-1.0 (default: 0.5)",
    )
    record_parser.add_argument(
        "--lessons",
        default="",
        help="Lessons learned from this outcome",
    )
    record_parser.add_argument(
        "--tags",
        default="",
        help="Comma-separated tags",
    )
    record_parser.add_argument(
        "--decision-id",
        default=None,
        help="Decision ID (defaults to debate_id)",
    )
    record_parser.set_defaults(func=_cmd_record)

    # --- search ---
    search_parser = outcome_sub.add_parser(
        "search",
        help="Search past outcomes",
    )
    search_parser.add_argument(
        "query",
        nargs="?",
        default="",
        help="Search query text",
    )
    search_parser.add_argument(
        "--tags",
        default="",
        help="Comma-separated tags to filter by",
    )
    search_parser.add_argument(
        "--type",
        choices=["success", "failure", "partial", "unknown"],
        default=None,
        help="Filter by outcome type",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum results (default: 20)",
    )
    search_parser.set_defaults(func=_cmd_search)


def _cmd_record(args: argparse.Namespace) -> None:
    """Record a decision outcome."""
    from aragora.knowledge.mound.adapters.outcome_adapter import get_outcome_adapter

    debate_id = args.debate_id
    decision_id = args.decision_id or debate_id
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []

    outcome_data = {
        "outcome_id": f"cli_{debate_id}_{int(__import__('time').time())}",
        "decision_id": decision_id,
        "debate_id": debate_id,
        "outcome_type": args.type,
        "outcome_description": args.description,
        "impact_score": max(0.0, min(1.0, args.impact)),
        "kpis_before": {},
        "kpis_after": {},
        "lessons_learned": args.lessons,
        "tags": tags,
    }

    adapter = get_outcome_adapter()
    success = adapter.ingest(outcome_data)

    if success:
        print(f"Outcome recorded for debate {debate_id}")
        print(f"  Type: {args.type}")
        print(f"  Impact: {args.impact:.2f}")
        if args.lessons:
            print(f"  Lessons: {args.lessons}")
        if tags:
            print(f"  Tags: {', '.join(tags)}")
    else:
        print("Failed to record outcome", file=sys.stderr)
        sys.exit(1)


def _cmd_search(args: argparse.Namespace) -> None:
    """Search past outcomes."""
    from aragora.knowledge.mound.adapters.outcome_adapter import get_outcome_adapter

    adapter = get_outcome_adapter()
    stats = adapter.get_stats()

    print(f"Outcome store: {stats['outcomes_processed']} outcomes indexed")

    if stats["outcomes_processed"] == 0:
        print("No outcomes recorded yet. Use 'aragora outcome record' to add one.")
        return

    print(f"Total items ingested: {stats['total_items_ingested']}")
    if stats["total_errors"] > 0:
        print(f"Errors: {stats['total_errors']}")
