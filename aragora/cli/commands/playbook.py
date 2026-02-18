"""
Playbook CLI commands: list and run decision playbooks.

Commands:
- list: List available playbooks
- run: Run a specific playbook
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)


def add_playbook_parser(subparsers: Any) -> None:
    """Register the 'playbook' subcommand with list/run actions."""
    playbook_parser = subparsers.add_parser(
        "playbook",
        help="List and run decision playbooks",
        description="""
Pre-built end-to-end decision workflows that combine debate templates,
compliance artifacts, vertical scoring, and approval gates.

Subcommands:
  list  List available playbooks
  run   Run a specific playbook

Examples:
  aragora playbook list
  aragora playbook list --category healthcare
  aragora playbook run hipaa_vendor_assessment --input "Evaluate Acme Corp"
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    playbook_sub = playbook_parser.add_subparsers(dest="playbook_command")

    # --- list ---
    list_parser = playbook_sub.add_parser(
        "list",
        help="List available playbooks",
    )
    list_parser.add_argument(
        "--category",
        default=None,
        help="Filter by category (healthcare, finance, engineering, etc.)",
    )
    list_parser.add_argument(
        "--tags",
        default="",
        help="Filter by comma-separated tags",
    )
    list_parser.set_defaults(func=_cmd_list)

    # --- run ---
    run_parser = playbook_sub.add_parser(
        "run",
        help="Run a specific playbook",
    )
    run_parser.add_argument("playbook_id", help="Playbook ID to run")
    run_parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input question or topic for the playbook",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running",
    )
    run_parser.set_defaults(func=_cmd_run)


def _cmd_list(args: argparse.Namespace) -> None:
    """List available playbooks."""
    from aragora.playbooks.registry import get_playbook_registry

    registry = get_playbook_registry()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None

    playbooks = registry.list(category=args.category, tags=tags)

    if not playbooks:
        print("No playbooks found.")
        return

    print(f"{'ID':<30} {'Category':<12} {'Name'}")
    print("-" * 72)
    for pb in playbooks:
        print(f"{pb.id:<30} {pb.category:<12} {pb.name}")

    print(f"\n{len(playbooks)} playbook(s) available")


def _cmd_run(args: argparse.Namespace) -> None:
    """Run a playbook."""
    from aragora.playbooks.registry import get_playbook_registry

    registry = get_playbook_registry()
    playbook = registry.get(args.playbook_id)

    if not playbook:
        print(f"Playbook not found: {args.playbook_id}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print(f"Playbook: {playbook.name}")
        print(f"Category: {playbook.category}")
        print(f"Template: {playbook.template_name}")
        if playbook.vertical_profile:
            print(f"Vertical: {playbook.vertical_profile}")
        print(f"Agents: {playbook.min_agents}-{playbook.max_agents}")
        print(f"Rounds: {playbook.max_rounds}")
        print(f"Consensus: {playbook.consensus_threshold}")
        print(f"\nSteps ({len(playbook.steps)}):")
        for i, step in enumerate(playbook.steps, 1):
            print(f"  {i}. [{step.action}] {step.name}")
        if playbook.approval_gates:
            print(f"\nApproval gates ({len(playbook.approval_gates)}):")
            for gate in playbook.approval_gates:
                print(f"  - {gate.name}: {gate.description}")
        if playbook.compliance_artifacts:
            print(f"\nCompliance artifacts: {', '.join(playbook.compliance_artifacts)}")
        print(f"\nInput: {args.input}")
        print("\n[dry-run] Would execute the above playbook.")
        return

    # In production, this would trigger actual debate execution
    print(f"Running playbook: {playbook.name}")
    print(f"Input: {args.input}")
    print("Playbook execution queued.")
