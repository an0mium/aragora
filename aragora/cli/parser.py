"""
CLI argument parser construction.

Builds the argparse parser with all subcommands and their arguments.
Separated from command implementations for clarity and maintainability.
"""

import argparse
import os

from aragora.config import DEFAULT_AGENTS, DEFAULT_CONSENSUS, DEFAULT_ROUNDS

from aragora.cli.commands.debate import cmd_ask
from aragora.cli.commands.stats import (
    cmd_stats,
    cmd_patterns,
    cmd_elo,
    cmd_cross_pollination,
)
from aragora.cli.commands.status import (
    cmd_status,
    cmd_validate_env,
    cmd_doctor,
    cmd_validate,
)
from aragora.cli.commands.server import cmd_serve
from aragora.cli.commands.tools import (
    cmd_modes,
    cmd_templates,
    cmd_improve,
    cmd_context,
)
from aragora.cli.commands.delegated import (
    cmd_agents,
    cmd_demo,
    cmd_export,
    cmd_init,
    cmd_setup,
    cmd_repl,
    cmd_config,
    cmd_replay,
    cmd_bench,
    cmd_badge,
    cmd_mcp_server,
    cmd_marketplace,
    cmd_control_plane,
)
from aragora.cli.commands.decide import (
    cmd_decide,
    cmd_plans,
    cmd_plans_show,
    cmd_plans_approve,
    cmd_plans_reject,
    cmd_plans_execute,
)
from aragora.cli.commands.testfixer import build_parser as build_testfixer_parser

# Default API URL from environment or localhost fallback
DEFAULT_API_URL = os.environ.get("ARAGORA_API_URL", "http://localhost:8080")
DEFAULT_API_KEY = os.environ.get("ARAGORA_API_KEY")


def get_version() -> str:
    """Get package version from pyproject.toml or fallback."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("aragora")
    except ImportError:
        # importlib.metadata not available (Python < 3.8)
        return "0.8.0-dev"
    except PackageNotFoundError:
        # Package not installed in editable mode - use dev version
        return "0.8.0-dev"


def build_parser() -> argparse.ArgumentParser:
    """Build and return the complete CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Aragora - Control plane for multi-agent vetted decisionmaking across org knowledge and channels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  aragora ask "Design a rate limiter" --agents grok,anthropic-api,openai-api,deepseek,mistral,gemini,qwen,kimi
  aragora ask "Implement auth" --agents grok,anthropic-api,openai-api,gemini --rounds 9
  aragora stats
  aragora patterns --type security
        """,
    )

    parser.add_argument("--version", "-V", action="version", version=f"aragora {get_version()}")
    parser.add_argument("--db", default="agora_memory.db", help="SQLite database path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    _add_ask_parser(subparsers)
    _add_stats_parser(subparsers)
    _add_status_parser(subparsers)
    _add_agents_parser(subparsers)
    _add_modes_parser(subparsers)
    _add_patterns_parser(subparsers)
    _add_demo_parser(subparsers)
    _add_templates_parser(subparsers)
    _add_export_parser(subparsers)
    _add_doctor_parser(subparsers)
    _add_validate_parser(subparsers)
    _add_validate_env_parser(subparsers)
    _add_improve_parser(subparsers)
    _add_context_parser(subparsers)
    _add_serve_parser(subparsers)
    _add_init_parser(subparsers)
    _add_setup_parser(subparsers)
    _add_backup_parser(subparsers)
    _add_repl_parser(subparsers)
    _add_config_parser(subparsers)
    _add_replay_parser(subparsers)
    _add_bench_parser(subparsers)
    _add_external_parsers(subparsers)
    _add_badge_parser(subparsers)
    _add_memory_parser(subparsers)
    _add_elo_parser(subparsers)
    _add_cross_pollination_parser(subparsers)
    _add_mcp_parser(subparsers)
    _add_marketplace_parser(subparsers)
    _add_skills_parser(subparsers)
    _add_nomic_parser(subparsers)
    _add_workflow_parser(subparsers)
    _add_deploy_parser(subparsers)
    _add_control_plane_parser(subparsers)
    _add_decide_parser(subparsers)
    _add_plans_parser(subparsers)
    build_testfixer_parser(subparsers)

    return parser


def _add_ask_parser(subparsers) -> None:
    """Add the 'ask' subcommand parser."""
    ask_parser = subparsers.add_parser("ask", help="Run a decision stress-test (debate engine)")
    ask_parser.add_argument("task", help="The task/question to debate")
    ask_parser.add_argument(
        "--agents",
        "-a",
        default=DEFAULT_AGENTS,
        help=(
            "Comma-separated agents. Formats: "
            "'provider' (auto-assign role), "
            "'provider:role' (e.g., anthropic-api:critic), "
            "'provider:persona' (e.g., anthropic-api:philosopher), "
            "'provider|model|persona|role' (full spec). "
            "Valid roles: proposer, critic, synthesizer, judge. "
            "Also accepts JSON list of dicts with provider/model/persona/role."
        ),
    )
    ask_parser.add_argument(
        "--auto-select",
        action="store_true",
        help="Auto-select an optimal agent team for the task",
    )
    ask_parser.add_argument(
        "--auto-select-config",
        help=(
            "JSON config for auto-selection (e.g. "
            '\'{"min_agents":3,"max_agents":5,"diversity_preference":0.5}\')'
        ),
    )
    ask_parser.add_argument(
        "--rounds",
        "-r",
        type=int,
        default=DEFAULT_ROUNDS,
        help=f"Number of debate rounds (default: {DEFAULT_ROUNDS})",
    )
    ask_parser.add_argument(
        "--consensus",
        "-c",
        choices=["majority", "unanimous", "judge", "hybrid", "none"],
        default=DEFAULT_CONSENSUS,
        help=f"Consensus mechanism (default: {DEFAULT_CONSENSUS})",
    )
    ask_parser.add_argument("--context", help="Additional context for the task")
    ask_parser.add_argument(
        "--no-learn", dest="learn", action="store_false", help="Don't store patterns"
    )
    ask_parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with built-in demo agents (no API keys required)",
    )
    ask_parser.add_argument(
        "--mode",
        "-m",
        choices=["architect", "coder", "reviewer", "debugger", "orchestrator"],
        help="Operational mode for agents (architect, coder, reviewer, debugger, orchestrator)",
    )
    ask_parser.add_argument(
        "--enable-verticals",
        action="store_true",
        help="Enable vertical specialists (auto-detected by task)",
    )
    ask_parser.add_argument(
        "--vertical",
        help="Explicit vertical specialist ID to inject (e.g., software, legal, healthcare)",
    )
    run_mode = ask_parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--api",
        action="store_true",
        help="Run debate via API server (uses shared storage and audit trails)",
    )
    run_mode.add_argument(
        "--local",
        action="store_true",
        help="Run debate locally without API server (offline/air-gapped mode)",
    )
    ask_parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"API server URL (default: {DEFAULT_API_URL})",
    )
    ask_parser.add_argument(
        "--api-key",
        default=None if DEFAULT_API_KEY is None else DEFAULT_API_KEY,
        help="API key for server authentication (default: ARAGORA_API_KEY)",
    )
    debate_type = ask_parser.add_mutually_exclusive_group()
    debate_type.add_argument(
        "--graph",
        action="store_true",
        help="Run a graph debate with branching (API mode only)",
    )
    debate_type.add_argument(
        "--matrix",
        action="store_true",
        help="Run a matrix debate with scenarios (API mode only)",
    )
    ask_parser.add_argument(
        "--graph-rounds",
        type=int,
        default=5,
        help="Max rounds per graph branch (default: 5)",
    )
    ask_parser.add_argument(
        "--branch-threshold",
        type=float,
        default=0.5,
        help="Divergence threshold for graph branching (0-1, default: 0.5)",
    )
    ask_parser.add_argument(
        "--max-branches",
        type=int,
        default=5,
        help="Maximum graph branches (default: 5)",
    )
    ask_parser.add_argument(
        "--matrix-rounds",
        type=int,
        default=3,
        help="Max rounds per matrix scenario (default: 3)",
    )
    ask_parser.add_argument(
        "--scenario",
        action="append",
        help="Matrix scenario JSON or name (repeatable)",
    )
    ask_parser.add_argument(
        "--decision-integrity",
        action="store_true",
        help="Build decision integrity package (receipt + plan) after debate completes",
    )
    ask_parser.add_argument(
        "--di-include-context",
        action="store_true",
        help="Include memory/knowledge snapshot in decision integrity package",
    )
    ask_parser.add_argument(
        "--di-plan-strategy",
        choices=["single_task", "gemini"],
        default="single_task",
        help="Decision integrity plan strategy (default: single_task)",
    )
    ask_parser.add_argument(
        "--di-execution-mode",
        choices=[
            "plan_only",
            "request_approval",
            "execute",
            "workflow",
            "workflow_execute",
            "execute_workflow",
            "hybrid",
            "computer_use",
        ],
        help="Decision integrity execution mode (API mode only)",
    )
    # Cross-pollination feature flags
    ask_parser.add_argument(
        "--no-elo-weighting",
        dest="elo_weighting",
        action="store_false",
        default=True,
        help="Disable ELO skill-based vote weighting",
    )
    ask_parser.add_argument(
        "--no-calibration",
        dest="calibration",
        action="store_false",
        default=True,
        help="Disable calibration tracking and confidence adjustment",
    )
    ask_parser.add_argument(
        "--no-evidence-weighting",
        dest="evidence_weighting",
        action="store_false",
        default=True,
        help="Disable evidence quality-based consensus weighting",
    )
    ask_parser.add_argument(
        "--no-trending",
        dest="trending",
        action="store_false",
        default=True,
        help="Disable trending topic injection from Pulse",
    )
    ask_parser.set_defaults(func=cmd_ask)


def _add_stats_parser(subparsers) -> None:
    """Add the 'stats' subcommand parser."""
    stats_parser = subparsers.add_parser("stats", help="Show memory statistics")
    stats_parser.set_defaults(func=cmd_stats)


def _add_status_parser(subparsers) -> None:
    """Add the 'status' subcommand parser."""
    status_parser = subparsers.add_parser(
        "status", help="Show environment health and agent availability"
    )
    status_parser.add_argument(
        "--server",
        "-s",
        default=DEFAULT_API_URL,
        help=f"Server URL to check (default: {DEFAULT_API_URL})",
    )
    status_parser.set_defaults(func=cmd_status)


def _add_agents_parser(subparsers) -> None:
    """Add the 'agents' subcommand parser."""
    agents_parser = subparsers.add_parser(
        "agents",
        help="List available agents and their configuration",
        description="Show all available agent types, their API key requirements, and configuration status.",
    )
    agents_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed descriptions"
    )
    agents_parser.set_defaults(func=cmd_agents)


def _add_modes_parser(subparsers) -> None:
    """Add the 'modes' subcommand parser."""
    modes_parser = subparsers.add_parser(
        "modes",
        help="List available operational modes",
        description="Show all available operational modes (architect, coder, reviewer, etc.) for debates.",
    )
    modes_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show full system prompts"
    )
    modes_parser.set_defaults(func=cmd_modes)


def _add_patterns_parser(subparsers) -> None:
    """Add the 'patterns' subcommand parser."""
    patterns_parser = subparsers.add_parser("patterns", help="Show learned patterns")
    patterns_parser.add_argument("--type", "-t", help="Filter by issue type")
    patterns_parser.add_argument("--min-success", type=int, default=1, help="Minimum success count")
    patterns_parser.add_argument("--limit", "-l", type=int, default=10, help="Max patterns to show")
    patterns_parser.set_defaults(func=cmd_patterns)


def _add_demo_parser(subparsers) -> None:
    """Add the 'demo' subcommand parser."""
    demo_parser = subparsers.add_parser("demo", help="Run a quick demo debate")
    demo_parser.add_argument("name", nargs="?", help="Demo name (rate-limiter, auth, cache)")
    demo_parser.set_defaults(func=cmd_demo)


def _add_templates_parser(subparsers) -> None:
    """Add the 'templates' subcommand parser."""
    templates_parser = subparsers.add_parser("templates", help="List available debate templates")
    templates_parser.set_defaults(func=cmd_templates)


def _add_export_parser(subparsers) -> None:
    """Add the 'export' subcommand parser."""
    export_parser = subparsers.add_parser("export", help="Export debate artifacts")
    export_parser.add_argument("--debate-id", "-d", help="Debate ID to export")
    export_parser.add_argument(
        "--format",
        "-f",
        choices=["html", "json", "md"],
        default="html",
        help="Output format (default: html)",
    )
    export_parser.add_argument(
        "--output",
        "-o",
        default=".",
        help="Output directory (default: current)",
    )
    export_parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate a demo export",
    )
    export_parser.set_defaults(func=cmd_export)


def _add_doctor_parser(subparsers) -> None:
    """Add the 'doctor' subcommand parser."""
    doctor_parser = subparsers.add_parser("doctor", help="Run system health checks")
    doctor_parser.add_argument(
        "--validate", "-v", action="store_true", help="Validate API keys by making test calls"
    )
    doctor_parser.set_defaults(func=cmd_doctor)


def _add_validate_parser(subparsers) -> None:
    """Add the 'validate' subcommand parser."""
    validate_parser = subparsers.add_parser(
        "validate", help="Validate API keys by making test calls"
    )
    validate_parser.set_defaults(func=cmd_validate)


def _add_validate_env_parser(subparsers) -> None:
    """Add the 'validate-env' subcommand parser."""
    validate_env_parser = subparsers.add_parser(
        "validate-env",
        help="Validate environment configuration and backend connectivity",
        description=(
            "Validates that the environment is properly configured for production "
            "deployment, including Redis/PostgreSQL connectivity, encryption keys, "
            "and AI provider configuration."
        ),
    )
    validate_env_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed messages"
    )
    validate_env_parser.add_argument(
        "--json", "-j", action="store_true", help="Output results as JSON"
    )
    validate_env_parser.add_argument(
        "--strict", "-s", action="store_true", help="Fail on warnings (for CI/CD enforcement)"
    )
    validate_env_parser.set_defaults(func=cmd_validate_env)


def _add_improve_parser(subparsers) -> None:
    """Add the 'improve' subcommand parser."""
    improve_parser = subparsers.add_parser(
        "improve",
        help="Self-improvement mode using AutonomousOrchestrator",
        description="""
Run self-improvement on the codebase using the Nomic AutonomousOrchestrator.

The orchestrator decomposes high-level goals into subtasks, routes them to
appropriate agents based on domain expertise, and executes them with
verification and feedback loops.

Examples:
  aragora improve --goal "Improve test coverage" --tracks qa
  aragora improve --goal "Refactor authentication" --dry-run
  aragora improve --goal "Add SDK endpoints" --tracks developer --max-cycles 3
  aragora improve --goal "Security audit" --tracks security --require-approval
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    improve_parser.add_argument(
        "--goal",
        "-g",
        required=True,
        help="The improvement goal to execute (required)",
    )
    improve_parser.add_argument(
        "--tracks",
        "-t",
        help="Comma-separated tracks to focus on (sme, developer, self_hosted, qa, core, security)",
    )
    improve_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview decomposition without executing (uses TaskDecomposer)",
    )
    improve_parser.add_argument(
        "--max-cycles",
        type=int,
        default=5,
        help="Maximum improvement cycles per subtask (default: 5)",
    )
    improve_parser.add_argument(
        "--require-approval",
        action="store_true",
        help="Require human approval at checkpoint gates",
    )
    improve_parser.add_argument(
        "--debate",
        action="store_true",
        help="Use multi-agent debate for goal decomposition (slower but better for abstract goals)",
    )
    improve_parser.add_argument(
        "--max-parallel",
        type=int,
        default=4,
        help="Maximum parallel tasks across all tracks (default: 4)",
    )
    improve_parser.add_argument(
        "--path",
        "-p",
        help="Path to codebase (default: current dir)",
    )
    improve_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress and checkpoint information",
    )
    improve_parser.set_defaults(func=cmd_improve)


def _add_context_parser(subparsers) -> None:
    """Add the 'context' subcommand parser."""
    context_parser = subparsers.add_parser(
        "context",
        help="Build codebase context for RLM-powered analysis",
        description=(
            "Indexes the codebase and optionally builds a TRUE RLM context "
            "for deep codebase analysis (up to 10M tokens)."
        ),
    )
    context_parser.add_argument("--path", "-p", help="Path to codebase (default: current dir)")
    context_parser.add_argument(
        "--rlm",
        action="store_true",
        help="Build TRUE RLM context (REPL-based) when available",
    )
    context_parser.add_argument(
        "--full-corpus",
        action="store_true",
        help="Include full-corpus RLM summary (expensive)",
    )
    context_parser.add_argument(
        "--max-bytes",
        type=int,
        help="Max context bytes (overrides env, supports 10M tokens ~40MB)",
    )
    tests_group = context_parser.add_mutually_exclusive_group()
    tests_group.add_argument(
        "--include-tests",
        action="store_true",
        help="Include test files in the index",
    )
    tests_group.add_argument(
        "--exclude-tests",
        action="store_true",
        help="Exclude test files from the index",
    )
    context_parser.add_argument(
        "--summary-out",
        help="Write the debate context summary to a file",
    )
    context_parser.add_argument(
        "--preview",
        action="store_true",
        help="Print a short preview of the context summary",
    )
    context_parser.set_defaults(func=cmd_context)


def _add_serve_parser(subparsers) -> None:
    """Add the 'serve' subcommand parser."""
    serve_parser = subparsers.add_parser(
        "serve",
        help="Run live debate server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Production deployment:
    aragora serve --workers 4 --host 0.0.0.0

    Use a load balancer to distribute traffic across workers.
        """,
    )
    serve_parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket port")
    serve_parser.add_argument("--api-port", type=int, default=8080, help="HTTP API port")
    serve_parser.add_argument("--host", default="localhost", help="Host to bind to")
    serve_parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Number of worker processes (default: 1). For production, use 2-4x CPU cores.",
    )
    serve_parser.set_defaults(func=cmd_serve)


def _add_init_parser(subparsers) -> None:
    """Add the 'init' subcommand parser."""
    init_parser = subparsers.add_parser("init", help="Initialize Aragora project")
    init_parser.add_argument("directory", nargs="?", help="Target directory (default: current)")
    init_parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing files")
    init_parser.add_argument("--no-git", action="store_true", help="Don't modify .gitignore")
    init_parser.set_defaults(func=cmd_init)


def _add_setup_parser(subparsers) -> None:
    """Add the 'setup' subcommand parser."""
    setup_parser = subparsers.add_parser(
        "setup",
        help="Interactive setup wizard for API keys and configuration",
        description=(
            "Guides you through configuring Aragora including API keys, "
            "database settings, and optional integrations. Generates a .env file."
        ),
    )
    setup_parser.add_argument(
        "--output", "-o", help="Output directory for .env file (default: current)"
    )
    setup_parser.add_argument(
        "--minimal", "-m", action="store_true", help="Only configure essential settings"
    )
    setup_parser.add_argument("--skip-test", action="store_true", help="Skip API key validation")
    setup_parser.add_argument(
        "-y", "--yes", action="store_true", help="Non-interactive mode (use defaults)"
    )
    setup_parser.set_defaults(func=cmd_setup)


def _add_backup_parser(subparsers) -> None:
    """Add the 'backup' subcommand parser."""
    from aragora.cli.backup import add_backup_subparsers

    add_backup_subparsers(subparsers)


def _add_repl_parser(subparsers) -> None:
    """Add the 'repl' subcommand parser."""
    repl_parser = subparsers.add_parser("repl", help="Interactive debate mode")
    repl_parser.add_argument(
        "--agents",
        "-a",
        default="anthropic-api,openai-api",
        help="Comma-separated agents for debates",
    )
    repl_parser.add_argument(
        "--rounds", "-r", type=int, default=8, help="Debate rounds (default: 8)"
    )
    repl_parser.set_defaults(func=cmd_repl)


def _add_config_parser(subparsers) -> None:
    """Add the 'config' subcommand parser."""
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_parser.add_argument(
        "action",
        nargs="?",
        default="show",
        choices=["show", "get", "set", "env", "path"],
        help="Config action",
    )
    config_parser.add_argument("key", nargs="?", help="Config key (for get/set)")
    config_parser.add_argument("value", nargs="?", help="Config value (for set)")
    config_parser.set_defaults(func=cmd_config)


def _add_replay_parser(subparsers) -> None:
    """Add the 'replay' subcommand parser."""
    replay_parser = subparsers.add_parser("replay", help="Replay stored debates")
    replay_parser.add_argument(
        "action", nargs="?", default="list", choices=["list", "show", "play"], help="Replay action"
    )
    replay_parser.add_argument("id", nargs="?", help="Replay ID (for show/play)")
    replay_parser.add_argument("--directory", "-d", help="Replays directory")
    replay_parser.add_argument("--limit", "-n", type=int, default=10, help="Max replays to list")
    replay_parser.add_argument("--speed", "-s", type=float, default=1.0, help="Playback speed")
    replay_parser.set_defaults(func=cmd_replay)


def _add_bench_parser(subparsers) -> None:
    """Add the 'bench' subcommand parser."""
    bench_parser = subparsers.add_parser("bench", help="Benchmark agents")
    bench_parser.add_argument(
        "--agents",
        "-a",
        default="anthropic-api,openai-api",
        help="Comma-separated agents to benchmark",
    )
    bench_parser.add_argument("--iterations", "-n", type=int, default=3, help="Iterations per task")
    bench_parser.add_argument("--task", "-t", help="Custom benchmark task")
    bench_parser.add_argument("--quick", "-q", action="store_true", help="Quick mode (1 iteration)")
    bench_parser.set_defaults(func=cmd_bench)


def _add_external_parsers(subparsers) -> None:
    """Add subcommand parsers that are defined in external modules."""
    # Review command (AI red team code review)
    from aragora.cli.review import create_review_parser

    create_review_parser(subparsers)

    # Gauntlet command (adversarial stress-testing)
    from aragora.cli.gauntlet import create_gauntlet_parser

    create_gauntlet_parser(subparsers)

    # Batch command (process multiple debates)
    from aragora.cli.batch import create_batch_parser

    create_batch_parser(subparsers)

    # Billing command
    from aragora.cli.billing import create_billing_parser

    create_billing_parser(subparsers)

    # Audit command (compliance audit logs)
    from aragora.cli.audit import create_audit_parser

    create_audit_parser(subparsers)

    # Document audit command (document analysis)
    from aragora.cli.document_audit import create_document_audit_parser

    create_document_audit_parser(subparsers)

    # Documents command (upload, list, show with folder support)
    from aragora.cli.documents import create_documents_parser

    create_documents_parser(subparsers)

    # Knowledge command (knowledge base operations)
    from aragora.cli.knowledge import create_knowledge_parser

    create_knowledge_parser(subparsers)

    # RLM command (recursive language model operations)
    from aragora.cli.rlm import create_rlm_parser

    create_rlm_parser(subparsers)

    # Template command (workflow template management)
    from aragora.cli.template import create_template_parser

    create_template_parser(subparsers)

    # Security command (encryption, key rotation)
    from aragora.cli.security import create_security_parser

    create_security_parser(subparsers)

    # Tenant command (multi-tenant management)
    from aragora.cli.tenant import create_tenant_parser

    create_tenant_parser(subparsers)

    # OpenClaw command (enterprise gateway management)
    from aragora.cli.openclaw import create_openclaw_parser

    create_openclaw_parser(subparsers)


def _add_badge_parser(subparsers) -> None:
    """Add the 'badge' subcommand parser."""
    badge_parser = subparsers.add_parser(
        "badge",
        help="Generate Aragora badge for your README",
        description="Generate shareable badges to show your project uses Aragora.",
    )
    badge_parser.add_argument(
        "--type",
        "-t",
        choices=["reviewed", "consensus", "gauntlet"],
        default="reviewed",
        help="Badge type: reviewed (blue), consensus (green), gauntlet (orange)",
    )
    badge_parser.add_argument(
        "--style",
        "-s",
        choices=["flat", "flat-square", "for-the-badge", "plastic"],
        default="flat",
        help="Badge style (default: flat)",
    )
    badge_parser.add_argument(
        "--repo",
        "-r",
        help="Link to specific repo (default: aragora repo)",
    )
    badge_parser.set_defaults(func=cmd_badge)


def _add_memory_parser(subparsers) -> None:
    """Add the 'memory' subcommand parser with API-backed sub-subcommands."""
    from aragora.cli.commands.memory_ops import add_memory_ops_parser

    add_memory_ops_parser(subparsers)


def _add_elo_parser(subparsers) -> None:
    """Add the 'elo' subcommand parser."""
    elo_parser = subparsers.add_parser(
        "elo",
        help="View ELO ratings, leaderboards, and match history",
        description="Inspect agent skill ratings, match history, and leaderboards.",
    )
    elo_parser.add_argument(
        "action",
        nargs="?",
        default="leaderboard",
        choices=["leaderboard", "history", "matches", "agent"],
        help="Action: leaderboard (default), history, matches, agent",
    )
    elo_parser.add_argument("--agent", "-a", help="Agent name (for history/agent actions)")
    elo_parser.add_argument("--domain", "-d", help="Filter by domain (for leaderboard)")
    elo_parser.add_argument("--limit", "-n", type=int, default=10, help="Max entries to show")
    elo_parser.add_argument("--db", help="Database path (default: from config)")
    elo_parser.set_defaults(func=cmd_elo)


def _add_cross_pollination_parser(subparsers) -> None:
    """Add the 'cross-pollination' subcommand parser."""
    xpoll_parser = subparsers.add_parser(
        "cross-pollination",
        aliases=["xpoll"],
        help="Cross-pollination event system diagnostics",
        description="View cross-subsystem event statistics and handler status.",
    )
    xpoll_parser.add_argument(
        "action",
        nargs="?",
        default="stats",
        choices=["stats", "subscribers", "reset"],
        help="Action: stats (default), subscribers, reset",
    )
    xpoll_parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output in JSON format",
    )
    xpoll_parser.set_defaults(func=cmd_cross_pollination)


def _add_mcp_parser(subparsers) -> None:
    """Add the 'mcp-server' subcommand parser."""
    mcp_parser = subparsers.add_parser(
        "mcp-server",
        help="Run the MCP (Model Context Protocol) server",
        description="""
Run the Aragora MCP server for integration with Claude and other MCP clients.

The MCP server exposes Aragora's capabilities as tools:
- run_debate: Run decision stress-tests (debate engine)
- run_gauntlet: Stress-test documents
- list_agents: List available agents
- get_debate: Retrieve debate results

Configure in claude_desktop_config.json:
{
    "mcpServers": {
        "aragora": {
            "command": "aragora",
            "args": ["mcp-server"]
        }
    }
}
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mcp_parser.set_defaults(func=cmd_mcp_server)


def _add_marketplace_parser(subparsers) -> None:
    """Add the 'marketplace' subcommand parser."""
    marketplace_parser = subparsers.add_parser(
        "marketplace",
        help="Manage agent template marketplace",
        description="List, search, import, and export agent templates. Use 'aragora marketplace --help' for subcommands.",
    )
    marketplace_parser.add_argument(
        "subcommand",
        nargs="?",
        help="Subcommand (list, search, get, export, import, categories, rate, use)",
    )
    marketplace_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Subcommand arguments",
    )
    marketplace_parser.set_defaults(func=cmd_marketplace)


def _add_skills_parser(subparsers) -> None:
    """Add the 'skills' subcommand parser for skill marketplace."""
    from aragora.cli.commands.skills import add_skills_parser

    add_skills_parser(subparsers)


def _add_nomic_parser(subparsers) -> None:
    """Add the 'nomic' subcommand parser for self-improvement loop."""
    from aragora.cli.commands.nomic import add_nomic_parser

    add_nomic_parser(subparsers)


def _add_workflow_parser(subparsers) -> None:
    """Add the 'workflow' subcommand parser for workflow engine."""
    from aragora.cli.commands.workflow import add_workflow_parser

    add_workflow_parser(subparsers)


def _add_deploy_parser(subparsers) -> None:
    """Add the 'deploy' subcommand parser for deployment validation."""
    from aragora.cli.commands.deploy import add_deploy_parser

    add_deploy_parser(subparsers)


def _add_control_plane_parser(subparsers) -> None:
    """Add the 'control-plane' subcommand parser."""
    cp_parser = subparsers.add_parser(
        "control-plane",
        help="Control plane status and management",
        description="""
Aragora Control Plane - orchestrate multi-agent vetted decisionmaking.

Show control plane status, list registered agents, and view connected channels.

Subcommands:
  status   - Show control plane overview (default)
  agents   - List registered agents and their status
  channels - List connected communication channels
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    cp_parser.add_argument(
        "subcommand",
        nargs="?",
        default="status",
        choices=["status", "agents", "channels"],
        help="Subcommand (default: status)",
    )
    cp_parser.add_argument(
        "--server",
        default=DEFAULT_API_URL,
        help=f"API server URL (default: {DEFAULT_API_URL})",
    )
    cp_parser.set_defaults(func=cmd_control_plane)


def _add_decide_parser(subparsers) -> None:
    """Add the 'decide' subcommand parser for the full gold path pipeline."""
    decide_parser = subparsers.add_parser(
        "decide",
        help="Run full decision pipeline: debate → plan → execute",
        description="""
Run the full decision pipeline (gold path):

  1. Debate: Multi-agent debate on the task
  2. Plan: Create decision plan from debate outcome
  3. Approve: Get approval (or auto-approve)
  4. Execute: Run the plan tasks
  5. Verify: Check execution results
  6. Learn: Store lessons in Knowledge Mound

Examples:
  aragora decide "Design a rate limiter" --agents grok,anthropic-api,openai-api
  aragora decide "Implement auth" --auto-approve --budget-limit 10.00
  aragora decide "Refactor database" --dry-run  # Create plan but don't execute
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    decide_parser.add_argument("task", help="The task/question to decide on")
    decide_parser.add_argument(
        "--agents",
        "-a",
        default=DEFAULT_AGENTS,
        help="Comma-separated agents for debate",
    )
    decide_parser.add_argument(
        "--auto-select",
        action="store_true",
        help="Auto-select an optimal agent team for the task",
    )
    decide_parser.add_argument(
        "--auto-select-config",
        help=(
            "JSON config for auto-selection (e.g. "
            '\'{"min_agents":3,"max_agents":5,"diversity_preference":0.5}\')'
        ),
    )
    decide_parser.add_argument(
        "--rounds",
        "-r",
        type=int,
        default=DEFAULT_ROUNDS,
        help=f"Number of debate rounds (default: {DEFAULT_ROUNDS})",
    )
    decide_parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically approve plans (skip approval step)",
    )
    decide_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create plan but don't execute",
    )
    execution_group = decide_parser.add_mutually_exclusive_group()
    execution_group.add_argument(
        "--execution-mode",
        choices=["workflow", "hybrid", "computer_use"],
        help="Execution engine for implementation tasks",
    )
    execution_group.add_argument(
        "--hybrid",
        action="store_true",
        help="Use hybrid executor (Claude + Codex)",
    )
    execution_group.add_argument(
        "--computer-use",
        action="store_true",
        help="Use browser-based computer use executor",
    )
    decide_parser.add_argument(
        "--budget-limit",
        type=float,
        help="Maximum budget for plan execution in USD",
    )
    decide_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )
    decide_parser.set_defaults(func=cmd_decide)


def _add_plans_parser(subparsers) -> None:
    """Add the 'plans' subcommand parser for decision plan management."""
    plans_parser = subparsers.add_parser(
        "plans",
        help="Manage decision plans",
        description="""
Manage decision plans created by the 'decide' command or API.

Subcommands:
  list              - List all plans (default)
  show <id>         - Show plan details
  approve <id>      - Approve a pending plan
  reject <id>       - Reject a pending plan
  execute <id>      - Execute an approved plan

Examples:
  aragora plans                          # List plans
  aragora plans list --status pending    # List pending plans
  aragora plans show abc123              # Show plan details
  aragora plans approve abc123           # Approve plan
  aragora plans execute abc123           # Execute plan
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    plans_subparsers = plans_parser.add_subparsers(dest="plans_action")

    # plans list
    list_parser = plans_subparsers.add_parser("list", help="List decision plans")
    list_parser.add_argument(
        "--status",
        "-s",
        choices=[
            "created",
            "awaiting_approval",
            "approved",
            "rejected",
            "executing",
            "completed",
            "failed",
        ],
        help="Filter by status",
    )
    list_parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=20,
        help="Maximum plans to show (default: 20)",
    )
    list_parser.set_defaults(func=cmd_plans)

    # plans show
    show_parser = plans_subparsers.add_parser("show", help="Show plan details")
    show_parser.add_argument("plan_id", help="Plan ID (full or prefix)")
    show_parser.set_defaults(func=cmd_plans_show)

    # plans approve
    approve_parser = plans_subparsers.add_parser("approve", help="Approve a plan")
    approve_parser.add_argument("plan_id", help="Plan ID to approve")
    approve_parser.add_argument(
        "--reason",
        "-r",
        help="Reason for approval",
    )
    approve_parser.set_defaults(func=cmd_plans_approve)

    # plans reject
    reject_parser = plans_subparsers.add_parser("reject", help="Reject a plan")
    reject_parser.add_argument("plan_id", help="Plan ID to reject")
    reject_parser.add_argument(
        "--reason",
        "-r",
        help="Reason for rejection",
    )
    reject_parser.set_defaults(func=cmd_plans_reject)

    # plans execute
    execute_parser = plans_subparsers.add_parser("execute", help="Execute a plan")
    execute_parser.add_argument("plan_id", help="Plan ID to execute")
    execute_exec_group = execute_parser.add_mutually_exclusive_group()
    execute_exec_group.add_argument(
        "--execution-mode",
        choices=["workflow", "hybrid", "computer_use"],
        help="Execution engine for implementation tasks",
    )
    execute_exec_group.add_argument(
        "--hybrid",
        action="store_true",
        help="Use hybrid executor (Claude + Codex)",
    )
    execute_exec_group.add_argument(
        "--computer-use",
        action="store_true",
        help="Use browser-based computer use executor",
    )
    execute_parser.set_defaults(func=cmd_plans_execute)

    # Default behavior when just 'aragora plans' is called
    plans_parser.set_defaults(func=cmd_plans)
