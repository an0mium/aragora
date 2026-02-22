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

# Built-in modes registered lazily inside main() to avoid import-time hang

# Default API URL from environment or localhost fallback
DEFAULT_API_URL = os.environ.get("ARAGORA_API_URL", "http://localhost:8080")

# ---------------------------------------------------------------------------
# Re-exports for backwards compatibility
#
# All public symbols that were previously defined in this module are now
# imported from their new locations. Any code doing
#   from aragora.cli.main import run_debate
# will continue to work.
# ---------------------------------------------------------------------------
from aragora.cli.commands.debate import (  # noqa: E402, F401
    get_event_emitter_if_available,
    parse_agents,
    run_debate,
    cmd_ask,
)
from aragora.cli.commands.stats import (  # noqa: E402, F401
    cmd_stats,
    cmd_patterns,
    cmd_memory,
    cmd_elo,
    cmd_cross_pollination,
)
from aragora.cli.commands.status import (  # noqa: E402, F401
    cmd_status,
    cmd_validate_env,
    cmd_doctor,
    cmd_validate,
)
from aragora.cli.commands.server import cmd_serve  # noqa: E402, F401
from aragora.cli.commands.tools import (  # noqa: E402, F401
    cmd_modes,
    cmd_templates,
    cmd_improve,
    cmd_context,
)
from aragora.cli.commands.delegated import (  # noqa: E402, F401
    cmd_agents,
    cmd_demo,
    cmd_export,
    cmd_init,
    cmd_setup,
    cmd_repl,
    cmd_config,
    cmd_replay,
    cmd_bench,
    cmd_review,
    cmd_gauntlet,
    cmd_badge,
    cmd_billing,
    cmd_mcp_server,
    cmd_marketplace,
    cmd_control_plane,
)
from aragora.cli.commands.testfix import cmd_testfix  # noqa: E402, F401
from aragora.cli.parser import get_version, build_parser  # noqa: E402, F401

# Re-export essential objects used by other modules (e.g., aragora.cli.batch)
from aragora.agents.spec import AgentSpec  # noqa: E402, F401
from aragora.memory.store import CritiqueStore  # noqa: E402, F401
from aragora.agents.base import create_agent  # noqa: E402, F401
from aragora.debate.orchestrator import Arena, DebateProtocol  # noqa: E402, F401
from aragora.core import Environment  # noqa: E402, F401
from aragora.config import DEFAULT_AGENTS, DEFAULT_CONSENSUS, DEFAULT_ROUNDS  # noqa: E402, F401


def main() -> None:
    # Register built-in modes here (not at module level) to avoid import-time hang
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
