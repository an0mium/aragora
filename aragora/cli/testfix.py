"""
Aragora testfix command - automated test failure diagnosis and repair.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any
import sys
from pathlib import Path

from aragora.nomic.testfixer import FixLoopConfig, TestFixerOrchestrator
from aragora.nomic.testfixer.proposer import AgentCodeGenerator

DEFAULT_TEST_COMMAND = os.environ.get("ARAGORA_TESTFIX_COMMAND", "pytest tests/ -q --maxfail=1")


def _parse_agents(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [agent.strip() for agent in raw.split(",") if agent.strip()]


def _build_generators(agent_types: list[str]) -> list[Any] | None:
    if not agent_types:
        return None
    generators: list[AgentCodeGenerator] = []
    for agent_type in agent_types:
        try:
            generators.append(AgentCodeGenerator(agent_type))
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            print(f"[testfix] Skipping agent '{agent_type}': {exc}")
    return generators or None


async def _run_testfix(args: argparse.Namespace) -> int:
    repo_path = Path(args.repo).resolve()
    agents = _parse_agents(getattr(args, "agents", None))
    generators = _build_generators(agents)

    config = FixLoopConfig(
        max_iterations=args.max_iterations,
        max_same_failure=args.max_same_failure,
        min_confidence_to_apply=args.min_confidence,
        min_confidence_for_auto=args.min_auto_confidence,
        require_debate_consensus=args.require_consensus,
        revert_on_failure=not args.no_revert,
        stop_on_first_success=args.stop_on_first_success,
    )

    if args.attempts_dir:
        config.attempts_dir = Path(args.attempts_dir)
        config.save_attempts = True

    if args.require_approval:

        async def approve(proposal):
            print("\nProposed fix:")
            print(f"  {proposal.description}")
            print(f"  Confidence: {proposal.post_debate_confidence:.0%}")
            diff = proposal.as_diff()
            if diff:
                print("\nDiff:\n")
                print(diff)
            if not sys.stdin.isatty():
                print("[testfix] Approval required but no TTY; rejecting.")
                return False
            response = input("Apply this fix? [y/N]: ").strip().lower()
            return response in ("y", "yes")

        config.on_fix_proposed = approve

    fixer = TestFixerOrchestrator(
        repo_path=repo_path,
        test_command=args.test_command,
        config=config,
        generators=generators,
        test_timeout=args.test_timeout,
    )

    result = await fixer.run_fix_loop(max_iterations=args.max_iterations)
    print(result.summary())

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"[testfix] Wrote result to {output_path}")

    return 0 if result.status.value == "success" else 1


def cmd_testfix(args: argparse.Namespace) -> None:
    """Handle 'testfix' command."""
    exit_code = asyncio.run(_run_testfix(args))
    raise SystemExit(exit_code)
