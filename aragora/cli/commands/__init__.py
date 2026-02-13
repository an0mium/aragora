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
from aragora.cli.commands.testfix import cmd_testfix
from aragora.cli.commands.skills import cmd_skills, add_skills_parser
from aragora.cli.commands.nomic import cmd_nomic, add_nomic_parser
from aragora.cli.commands.workflow import cmd_workflow, add_workflow_parser
from aragora.cli.commands.receipt import (
    cmd_receipt,
    cmd_receipt_verify,
    cmd_receipt_inspect,
    cmd_receipt_export,
    setup_receipt_parser,
)
from aragora.cli.commands.deploy import cmd_deploy, add_deploy_parser
from aragora.cli.commands.memory_ops import cmd_memory_ops, add_memory_ops_parser
from aragora.cli.commands.publish import cmd_publish, add_publish_parser
from aragora.cli.commands.autopilot import cmd_autopilot, add_autopilot_parser

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
    "cmd_testfix",
    # skills (API-backed)
    "cmd_skills",
    "add_skills_parser",
    # nomic loop
    "cmd_nomic",
    "add_nomic_parser",
    # workflow engine
    "cmd_workflow",
    "add_workflow_parser",
    # receipt verification
    "cmd_receipt",
    "cmd_receipt_verify",
    "cmd_receipt_inspect",
    "cmd_receipt_export",
    "setup_receipt_parser",
    # deployment CLI
    "cmd_deploy",
    "add_deploy_parser",
    # memory operations (API-backed)
    "cmd_memory_ops",
    "add_memory_ops_parser",
    # package publishing
    "cmd_publish",
    "add_publish_parser",
    # autopilot GTM
    "cmd_autopilot",
    "add_autopilot_parser",
]
