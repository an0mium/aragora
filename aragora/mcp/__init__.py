"""
MCP (Model Context Protocol) Server for Aragora.

Exposes Aragora capabilities as MCP tools for integration with
any MCP-compatible AI agent (Claude Code, Cursor, etc.).
"""

from aragora.mcp.server import (
    AragoraMCPServer,
    MCPTool,
    MCPResource,
    MCPPrompt,
    create_mcp_server,
    run_server,
)
from aragora.mcp.tools_module import (
    run_debate_tool,
    run_gauntlet_tool,
    list_agents_tool,
    get_debate_tool,
)
from aragora.mcp.tools_module.self_improve import (
    assess_codebase_tool,
    generate_improvement_goals_tool,
    run_self_improvement_tool,
    get_daemon_status_tool,
    trigger_improvement_cycle_tool,
)

__all__ = [
    "AragoraMCPServer",
    "MCPTool",
    "MCPResource",
    "MCPPrompt",
    "create_mcp_server",
    "run_server",
    "run_debate_tool",
    "run_gauntlet_tool",
    "list_agents_tool",
    "get_debate_tool",
    "assess_codebase_tool",
    "generate_improvement_goals_tool",
    "run_self_improvement_tool",
    "get_daemon_status_tool",
    "trigger_improvement_cycle_tool",
]
