#!/usr/bin/env python3
"""
Aragora CLI - Control Plane for Multi-Agent Deliberation

Orchestrate multi-agent vetted decisionmaking across your organization's knowledge and channels.

Usage:
    aragora ask "Design a rate limiter" --agents grok,anthropic-api,openai-api,deepseek,mistral,gemini,qwen,kimi --rounds 9
    aragora ask "Implement auth system" --agents grok,anthropic-api,openai-api,gemini --rounds 9
    aragora stats

Environment Variables:
    ARAGORA_API_URL: API server URL (default: http://localhost:8080)

This module serves as the entry point for the CLI. All command implementations
have been split into submodules under aragora.cli.commands/ for maintainability:

    - aragora.cli.commands.debate   : Debate execution (run_debate, cmd_ask, parse_agents)
    - aragora.cli.commands.stats    : Statistics and data inspection (cmd_stats, cmd_patterns, etc.)
    - aragora.cli.commands.status   : Environment health and validation (cmd_status, cmd_validate_env)
    - aragora.cli.commands.server   : Server management (cmd_serve)
    - aragora.cli.commands.tools    : Modes, templates, improve, context commands
    - aragora.cli.commands.delegated: Thin wrappers delegating to other cli modules
    - aragora.cli.parser            : Argument parser construction (build_parser)
"""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Default API URL from environment or localhost fallback
DEFAULT_API_URL = os.environ.get("ARAGORA_API_URL", "http://localhost:8080")

# ---------------------------------------------------------------------------
# Re-exports for backwards compatibility
#
# Heavy imports (debate engine, agents, memory) are deferred via __getattr__
# to avoid loading scipy/numpy (~13s) on every CLI invocation.
# Lightweight parser/config imports remain eager.
# ---------------------------------------------------------------------------
from aragora.cli.parser import get_version, build_parser  # noqa: E402, F401

# Lazy re-export mapping: name -> (module, attr)
_LAZY_REEXPORTS: dict[str, tuple[str, str]] = {
    # From aragora.cli.commands.debate
    "get_event_emitter_if_available": (
        "aragora.cli.commands.debate",
        "get_event_emitter_if_available",
    ),
    "parse_agents": ("aragora.cli.commands.debate", "parse_agents"),
    "run_debate": ("aragora.cli.commands.debate", "run_debate"),
    "cmd_ask": ("aragora.cli.commands.debate", "cmd_ask"),
    # From aragora.cli.commands.stats
    "cmd_stats": ("aragora.cli.commands.stats", "cmd_stats"),
    "cmd_patterns": ("aragora.cli.commands.stats", "cmd_patterns"),
    "cmd_memory": ("aragora.cli.commands.stats", "cmd_memory"),
    "cmd_elo": ("aragora.cli.commands.stats", "cmd_elo"),
    "cmd_cross_pollination": ("aragora.cli.commands.stats", "cmd_cross_pollination"),
    # From aragora.cli.commands.status
    "cmd_status": ("aragora.cli.commands.status", "cmd_status"),
    "cmd_validate_env": ("aragora.cli.commands.status", "cmd_validate_env"),
    "cmd_doctor": ("aragora.cli.commands.status", "cmd_doctor"),
    "cmd_validate": ("aragora.cli.commands.status", "cmd_validate"),
    # From aragora.cli.commands.server
    "cmd_serve": ("aragora.cli.commands.server", "cmd_serve"),
    # From aragora.cli.commands.tools
    "cmd_modes": ("aragora.cli.commands.tools", "cmd_modes"),
    "cmd_templates": ("aragora.cli.commands.tools", "cmd_templates"),
    "cmd_improve": ("aragora.cli.commands.tools", "cmd_improve"),
    "cmd_context": ("aragora.cli.commands.tools", "cmd_context"),
    # From aragora.cli.commands.delegated
    "cmd_agents": ("aragora.cli.commands.delegated", "cmd_agents"),
    "cmd_demo": ("aragora.cli.commands.delegated", "cmd_demo"),
    "cmd_export": ("aragora.cli.commands.delegated", "cmd_export"),
    "cmd_init": ("aragora.cli.commands.delegated", "cmd_init"),
    "cmd_setup": ("aragora.cli.commands.delegated", "cmd_setup"),
    "cmd_repl": ("aragora.cli.commands.delegated", "cmd_repl"),
    "cmd_config": ("aragora.cli.commands.delegated", "cmd_config"),
    "cmd_replay": ("aragora.cli.commands.delegated", "cmd_replay"),
    "cmd_bench": ("aragora.cli.commands.delegated", "cmd_bench"),
    "cmd_review": ("aragora.cli.commands.delegated", "cmd_review"),
    "cmd_gauntlet": ("aragora.cli.commands.delegated", "cmd_gauntlet"),
    "cmd_badge": ("aragora.cli.commands.delegated", "cmd_badge"),
    "cmd_billing": ("aragora.cli.commands.delegated", "cmd_billing"),
    "cmd_mcp_server": ("aragora.cli.commands.delegated", "cmd_mcp_server"),
    "cmd_marketplace": ("aragora.cli.commands.delegated", "cmd_marketplace"),
    "cmd_control_plane": ("aragora.cli.commands.delegated", "cmd_control_plane"),
    # From aragora.cli.commands.testfix
    "cmd_testfix": ("aragora.cli.commands.testfix", "cmd_testfix"),
    # Essential objects used by other modules (e.g., aragora.cli.batch)
    "AgentSpec": ("aragora.agents.spec", "AgentSpec"),
    "CritiqueStore": ("aragora.memory.store", "CritiqueStore"),
    "create_agent": ("aragora.agents.base", "create_agent"),
    "Arena": ("aragora.debate.orchestrator", "Arena"),
    "DebateProtocol": ("aragora.debate.orchestrator", "DebateProtocol"),
    "Environment": ("aragora.core", "Environment"),
    "DEFAULT_AGENTS": ("aragora.config", "DEFAULT_AGENTS"),
    "DEFAULT_CONSENSUS": ("aragora.config", "DEFAULT_CONSENSUS"),
    "DEFAULT_ROUNDS": ("aragora.config", "DEFAULT_ROUNDS"),
}


def __getattr__(name: str):
    if name in _LAZY_REEXPORTS:
        module_path, attr_name = _LAZY_REEXPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'aragora.cli.main' has no attribute {name!r}")


def main() -> None:
    # Register built-in modes here (not at module level) to avoid import-time cost
    from aragora.modes import register_all_builtins

    register_all_builtins()

    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
