"""CLI command for TestFixer automation."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

from aragora.nomic.testfixer.generators import AgentCodeGenerator, AgentGeneratorConfig
from aragora.nomic.testfixer.orchestrator import FixLoopConfig, TestFixerOrchestrator
from aragora.nomic.testfixer.store import TestFixerAttemptStore


def _setup_logging(repo_path: Path, log_file: str | None, log_level: str) -> Path | None:
    level_name = (log_level or "info").upper()
    level = getattr(logging, level_name, logging.INFO)

    if log_file == "-":
        handlers = [logging.StreamHandler()]
        log_path = None
    else:
        if log_file:
            log_path = Path(log_file)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = repo_path / ".testfixer" / "logs" / f"testfixer_{ts}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers = [
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ]

    logging.basicConfig(
        level=level,
        handlers=handlers,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    return log_path


def _setup_logging(repo_path: Path, log_file: str | None, log_level: str) -> Path | None:
    level_name = (log_level or "info").upper()
    level = getattr(logging, level_name, logging.INFO)

    if log_file == "-":
        handlers = [logging.StreamHandler()]
        log_path = None
    else:
        if log_file:
            log_path = Path(log_file)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = repo_path / ".testfixer" / "logs" / f"testfixer_{ts}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers = [
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ]

    logging.basicConfig(
        level=level,
        handlers=handlers,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    return log_path


def _parse_agents(agents: str) -> list[AgentCodeGenerator]:
    if not agents or agents.strip().lower() in ("none", "false", "off", "0"):
        return []
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

    log_path = _setup_logging(repo_path, args.log_file, args.log_level)
    if log_path:
        print(f"[testfixer] logging to {log_path}")

    attempt_store = None
    if args.attempt_store:
        attempt_store = TestFixerAttemptStore(Path(args.attempt_store))

    run_id = args.run_id or uuid.uuid4().hex
    config = FixLoopConfig(
        max_iterations=args.max_iterations,
        min_confidence_to_apply=args.min_confidence,
        min_confidence_for_auto=args.min_confidence_auto,
        require_debate_consensus=args.require_consensus,
        revert_on_failure=not args.no_revert,
        attempt_store=attempt_store,
        run_id=run_id,
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
    parser.add_argument("--log-file", default=None, help="Path to log file (or '-' for stderr)")
    parser.add_argument("--log-level", default="info", help="Log level (debug, info, warning)")
    parser.add_argument("--run-id", default=None, help="Optional run id for correlation")
    parser.set_defaults(func=cmd_testfixer)
