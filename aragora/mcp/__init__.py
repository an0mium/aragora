"""
Aragora MCP (Model Context Protocol) Server.

Exposes Aragora's debate and gauntlet capabilities as MCP tools,
allowing Claude and other MCP-compatible clients to:

1. Run debates with multiple agents
2. Execute gauntlet stress-tests
3. Access debate results as resources

Usage:
    # Start the MCP server
    aragora mcp-server

    # Or run directly
    python -m aragora.mcp.server

Configuration in claude_desktop_config.json:
    {
        "mcpServers": {
            "aragora": {
                "command": "aragora",
                "args": ["mcp-server"]
            }
        }
    }
"""

from .server import AragoraMCPServer, run_server
from .tools import (
    run_debate_tool,
    run_gauntlet_tool,
    list_agents_tool,
    get_debate_tool,
)

__all__ = [
    "AragoraMCPServer",
    "run_server",
    "run_debate_tool",
    "run_gauntlet_tool",
    "list_agents_tool",
    "get_debate_tool",
]
