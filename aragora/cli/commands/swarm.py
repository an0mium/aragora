"""Swarm Commander CLI command.

Launches the full swarm lifecycle: interrogate -> spec -> dispatch -> report.

Usage:
    aragora swarm "Make the dashboard faster"
    aragora swarm "Fix tests" --skip-interrogation
    aragora swarm --spec my-spec.yaml
    aragora swarm "Add auth" --budget-limit 10
    aragora swarm "Improve UX" --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone


def cmd_swarm(args: argparse.Namespace) -> None:
    """Handle 'swarm' command."""
    from aragora.swarm import SwarmCommander, SwarmCommanderConfig, SwarmSpec
    from aragora.swarm.config import InterrogatorConfig

    goal = getattr(args, "goal", None)
    spec_file = getattr(args, "spec", None)
    skip_interrogation = getattr(args, "skip_interrogation", False)
    dry_run = getattr(args, "dry_run", False)
    budget_limit = getattr(args, "budget_limit", 5.0)
    require_approval = getattr(args, "require_approval", False)

    if not goal and not spec_file:
        print("Error: provide a goal or --spec file")
        print('Usage: aragora swarm "your goal here"')
        return

    config = SwarmCommanderConfig(
        interrogator=InterrogatorConfig(),
        budget_limit_usd=budget_limit,
        require_approval=require_approval,
    )
    commander = SwarmCommander(config=config)

    if spec_file:
        spec_path = Path(spec_file)
        if not spec_path.exists():
            print(f"Error: spec file not found: {spec_file}")
            return
        spec = SwarmSpec.from_yaml(spec_path.read_text())
        print(f"\nLoaded spec from {spec_file}")
        print(spec.summary())
        asyncio.run(commander.run_from_spec(spec))
    elif dry_run:
        spec = asyncio.run(commander.dry_run(goal))
        # Also save to file if requested
        save_path = getattr(args, "save_spec", None)
        if save_path:
            Path(save_path).write_text(spec.to_yaml())
            print(f"\nSpec saved to {save_path}")
    elif skip_interrogation:
        spec = SwarmSpec(
            id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            raw_goal=goal,
            refined_goal=goal,
            budget_limit_usd=budget_limit,
            requires_approval=require_approval,
            interrogation_turns=0,
            user_expertise="developer",
        )
        print("\nSkipping interrogation (developer mode)")
        print(spec.summary())
        asyncio.run(commander.run_from_spec(spec))
    else:
        asyncio.run(commander.run(goal))
