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
    MCPCapability,
    create_mcp_server,
)

__all__ = [
    "AragoraMCPServer",
    "MCPTool",
    "MCPResource",
    "MCPPrompt",
    "MCPCapability",
    "create_mcp_server",
]
