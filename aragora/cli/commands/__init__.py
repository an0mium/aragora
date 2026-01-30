"""
CLI command implementations, split by functional domain.

All public command handlers and functions are re-exported here for convenience.
"""

from aragora.cli.commands.debate import (
    get_event_emitter_if_available,
    parse_agents,
    run_debate,
    cmd_ask,
)
from aragora.cli.commands.stats import (
    cmd_stats,
    cmd_patterns,
    cmd_memory,
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
    cmd_review,
    cmd_gauntlet,
    cmd_badge,
    cmd_billing,
    cmd_mcp_server,
    cmd_marketplace,
    cmd_control_plane,
)

__all__ = [
    # debate
    "get_event_emitter_if_available",
    "parse_agents",
    "run_debate",
    "cmd_ask",
    # stats
    "cmd_stats",
    "cmd_patterns",
    "cmd_memory",
    "cmd_elo",
    "cmd_cross_pollination",
    # status
    "cmd_status",
    "cmd_validate_env",
    "cmd_doctor",
    "cmd_validate",
    # server
    "cmd_serve",
    # tools
    "cmd_modes",
    "cmd_templates",
    "cmd_improve",
    "cmd_context",
    # delegated
    "cmd_agents",
    "cmd_demo",
    "cmd_export",
    "cmd_init",
    "cmd_setup",
    "cmd_repl",
    "cmd_config",
    "cmd_replay",
    "cmd_bench",
    "cmd_review",
    "cmd_gauntlet",
    "cmd_badge",
    "cmd_billing",
    "cmd_mcp_server",
    "cmd_marketplace",
    "cmd_control_plane",
]
