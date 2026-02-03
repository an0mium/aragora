"""CLI command for TestFixer automation."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from aragora.nomic.testfixer.generators import AgentCodeGenerator, AgentGeneratorConfig
from aragora.nomic.testfixer.orchestrator import FixLoopConfig, TestFixerOrchestrator
from aragora.nomic.testfixer.store import TestFixerAttemptStore


def _parse_agents(agents: str) -> list[AgentCodeGenerator]:
    generator_configs: list[AgentGeneratorConfig] = []
    for spec in agents.split(","):
        spec = spec.strip()
        if not spec:
            continue
        if ":" in spec:
            agent_type, model = spec.split(":", 1)
        else:
            agent_type, model = spec, None
        generator_configs.append(AgentGeneratorConfig(agent_type=agent_type, model=model))
    return [AgentCodeGenerator(cfg) for cfg in generator_configs]


async def _run(args: argparse.Namespace) -> int:
    repo_path = Path(args.repo_path).resolve()
    test_command = args.test_command
    generators = _parse_agents(args.agents)

    attempt_store = None
    if args.attempt_store:
        attempt_store = TestFixerAttemptStore(Path(args.attempt_store))

    config = FixLoopConfig(
        max_iterations=args.max_iterations,
        min_confidence_to_apply=args.min_confidence,
        min_confidence_for_auto=args.min_confidence_auto,
        require_debate_consensus=args.require_consensus,
        revert_on_failure=not args.no_revert,
        attempt_store=attempt_store,
    )
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
                print("[testfixer] Approval required but no TTY; rejecting.")
                return False
            response = input("Apply this fix? [y/N]: ").strip().lower()
            return response in ("y", "yes")

        config.on_fix_proposed = approve

    fixer = TestFixerOrchestrator(
        repo_path=repo_path,
        test_command=test_command,
        config=config,
        generators=generators,
        test_timeout=args.timeout_seconds,
    )

    result = await fixer.run_fix_loop(max_iterations=args.max_iterations)
    print(result.summary())
    if result.status.value not in ("success",):
        return 1
    return 0


def cmd_testfixer(args: argparse.Namespace) -> None:
    exit_code = asyncio.run(_run(args))
    if exit_code:
        sys.exit(exit_code)


def build_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "testfixer",
        help="Run automated test-fix loop",
        description="Run automated test-fix loop with multi-agent debate",
    )
    parser.add_argument("repo_path", help="Path to repository")
    parser.add_argument("--test-command", default="pytest tests/ -q --maxfail=1")
    parser.add_argument("--agents", default="codex,claude")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-confidence-auto", type=float, default=0.7)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--attempt-store", default=None)
    parser.add_argument("--require-consensus", action="store_true")
    parser.add_argument("--no-revert", action="store_true")
    parser.add_argument(
        "--require-approval",
        action="store_true",
        help="Require manual approval before applying fixes",
    )
    parser.set_defaults(func=cmd_testfixer)
