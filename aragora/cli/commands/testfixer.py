"""CLI command for TestFixer automation."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

from aragora.nomic.testfixer.analyzers import LLMAnalyzerConfig
from aragora.nomic.testfixer.generators import AgentCodeGenerator, AgentGeneratorConfig
from aragora.nomic.testfixer.orchestrator import FixLoopConfig, TestFixerOrchestrator
from aragora.nomic.testfixer.store import TestFixerAttemptStore
from aragora.nomic.testfixer.validators import ArenaValidatorConfig, RedTeamValidatorConfig


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


def _parse_agents(agents: str, timeout_seconds: float | None = None) -> list[AgentCodeGenerator]:
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
        generator_configs.append(
            AgentGeneratorConfig(
                agent_type=agent_type,
                model=model,
                timeout_seconds=timeout_seconds,
            )
        )
    return [AgentCodeGenerator(cfg) for cfg in generator_configs]


def _parse_agent_specs(specs: str) -> tuple[list[str], dict[str, str] | None]:
    if not specs or specs.strip().lower() in ("none", "false", "off", "0"):
        return ([], None)
    agent_types: list[str] = []
    models: dict[str, str] = {}
    for spec in specs.split(","):
        spec = spec.strip()
        if not spec:
            continue
        if ":" in spec:
            agent_type, model = spec.split(":", 1)
            models[agent_type] = model
        else:
            agent_type = spec
        agent_types.append(agent_type)
    return agent_types, models or None


async def _run(args: argparse.Namespace) -> int:
    repo_path = Path(args.repo_path).resolve()
    test_command = args.test_command
    generators = _parse_agents(args.agents, timeout_seconds=args.generation_timeout_seconds)

    log_path = _setup_logging(repo_path, args.log_file, args.log_level)
    if log_path:
        print(f"[testfixer] logging to {log_path}")

    attempt_store = None
    if args.attempt_store:
        attempt_store = TestFixerAttemptStore(Path(args.attempt_store))

    run_id = args.run_id or uuid.uuid4().hex

    analysis_agents_spec = args.analysis_agents or args.agents
    analysis_agent_types, analysis_models = _parse_agent_specs(analysis_agents_spec)
    llm_analyzer_config = None
    if args.llm_analyzer:
        default_analysis = LLMAnalyzerConfig()
        llm_analyzer_config = LLMAnalyzerConfig(
            agent_types=analysis_agent_types or default_analysis.agent_types,
            models=analysis_models,
            require_consensus=args.analysis_require_consensus,
            consensus_threshold=args.analysis_consensus_threshold,
            agent_timeout=args.generation_timeout_seconds,
        )

    arena_config = None
    if args.arena_validate:
        arena_agents_spec = args.arena_agents or analysis_agents_spec or args.agents
        arena_agent_types, arena_models = _parse_agent_specs(arena_agents_spec)
        default_arena = ArenaValidatorConfig()
        arena_config = ArenaValidatorConfig(
            agent_types=arena_agent_types or default_arena.agent_types,
            models=arena_models,
            debate_rounds=args.arena_rounds,
            min_confidence_to_pass=args.arena_min_confidence,
            require_consensus=args.arena_require_consensus,
            consensus_threshold=args.arena_consensus_threshold,
            agent_timeout=args.generation_timeout_seconds,
            debate_timeout=max(
                args.generation_timeout_seconds * 2,
                default_arena.debate_timeout,
            ),
        )

    redteam_config = None
    if args.redteam_validate:
        redteam_agents_spec = args.redteam_attackers or analysis_agents_spec or args.agents
        redteam_attackers, _ = _parse_agent_specs(redteam_agents_spec)
        default_redteam = RedTeamValidatorConfig()
        defender = args.redteam_defender or (
            redteam_attackers[0] if redteam_attackers else default_redteam.defender_type
        )
        redteam_config = RedTeamValidatorConfig(
            attacker_types=redteam_attackers or default_redteam.attacker_types,
            defender_type=defender,
            attack_rounds=args.redteam_rounds,
            attacks_per_round=args.redteam_attacks_per_round,
            min_robustness_score=args.redteam_min_robustness,
            agent_timeout=args.generation_timeout_seconds,
            total_timeout=max(
                args.generation_timeout_seconds * 4,
                default_redteam.total_timeout,
            ),
        )

    config = FixLoopConfig(
        max_iterations=args.max_iterations,
        min_confidence_to_apply=args.min_confidence,
        min_confidence_for_auto=args.min_confidence_auto,
        require_debate_consensus=args.require_consensus,
        revert_on_failure=not args.no_revert,
        attempt_store=attempt_store,
        run_id=run_id,
        artifacts_dir=Path(args.artifacts_dir).resolve() if args.artifacts_dir else None,
        enable_diagnostics=not args.no_diagnostics,
        use_llm_analyzer=args.llm_analyzer,
        llm_analyzer_config=llm_analyzer_config,
        enable_arena_validation=args.arena_validate,
        arena_validator_config=arena_config,
        enable_redteam_validation=args.redteam_validate,
        redteam_validator_config=redteam_config,
        enable_pattern_learning=args.pattern_learning,
        pattern_store_path=Path(args.pattern_store).resolve() if args.pattern_store else None,
        generation_timeout_seconds=args.generation_timeout_seconds,
        critique_timeout_seconds=args.critique_timeout_seconds,
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
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Directory for per-run artifacts (default: .testfixer/runs)",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Disable crash diagnostics and artifact collection",
    )
    parser.add_argument(
        "--llm-analyzer",
        action="store_true",
        help="Enable LLM-powered failure analysis",
    )
    parser.add_argument(
        "--analysis-agents",
        default="",
        help="Agent types for analysis (comma-separated, default: --agents)",
    )
    parser.add_argument(
        "--analysis-require-consensus",
        action="store_true",
        help="Require consensus between analysis agents",
    )
    parser.add_argument(
        "--analysis-consensus-threshold",
        type=float,
        default=0.7,
        help="Consensus threshold for analysis agents",
    )
    parser.add_argument(
        "--arena-validate",
        action="store_true",
        help="Enable Arena validator for proposed fixes",
    )
    parser.add_argument(
        "--arena-agents",
        default="",
        help="Agent types for Arena validation (default: --analysis-agents)",
    )
    parser.add_argument("--arena-rounds", type=int, default=2)
    parser.add_argument("--arena-min-confidence", type=float, default=0.6)
    parser.add_argument(
        "--arena-require-consensus",
        action="store_true",
        help="Require consensus for Arena validation",
    )
    parser.add_argument(
        "--arena-consensus-threshold",
        type=float,
        default=0.7,
        help="Consensus threshold for Arena validation",
    )
    parser.add_argument(
        "--redteam-validate",
        action="store_true",
        help="Enable red team validation for proposed fixes",
    )
    parser.add_argument(
        "--redteam-attackers",
        default="",
        help="Agent types for red team attackers (default: --analysis-agents)",
    )
    parser.add_argument(
        "--redteam-defender",
        default="",
        help="Agent type for red team defender (default: first attacker)",
    )
    parser.add_argument("--redteam-rounds", type=int, default=2)
    parser.add_argument("--redteam-attacks-per-round", type=int, default=3)
    parser.add_argument("--redteam-min-robustness", type=float, default=0.6)
    parser.add_argument(
        "--pattern-learning",
        action="store_true",
        help="Enable pattern learning for repeated fixes",
    )
    parser.add_argument(
        "--pattern-store",
        default=None,
        help="Pattern store path (default: .testfixer/patterns.json)",
    )
    parser.add_argument(
        "--generation-timeout-seconds",
        type=float,
        default=600.0,
        help="Timeout for each generator proposal",
    )
    parser.add_argument(
        "--critique-timeout-seconds",
        type=float,
        default=300.0,
        help="Timeout for each critique pass",
    )
    parser.set_defaults(func=cmd_testfixer)
