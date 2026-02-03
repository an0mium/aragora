"""
Tests for Aragora MCP Server (lightweight registry layer).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from aragora.mcp.server import AragoraMCPServer, MCPPrompt, MCPResource, MCPTool, create_mcp_server


TEST_TOOLS_METADATA = [
    {
        "name": "echo",
        "description": "Echo input text",
        "function": AsyncMock(return_value={"echo": "ok"}),
        "parameters": {
            "text": {"type": "string", "required": True, "description": "Text to echo"},
        },
    },
    {
        "name": "ping",
        "description": "Health check",
        "function": AsyncMock(return_value={"status": "ok"}),
        "parameters": {},
    },
]


@pytest.fixture
def server() -> AragoraMCPServer:
    return AragoraMCPServer(
        name="aragora-test",
        version="1.0.0-test",
        tools_metadata=TEST_TOOLS_METADATA,
        require_mcp=False,
    )


def test_server_creation() -> None:
    server = AragoraMCPServer(require_mcp=False)
    assert server.name == "aragora"
    assert server.version == "1.0.0"


def test_factory_function() -> None:
    server = create_mcp_server(name="factory", version="9.9.9", require_mcp=False)
    assert server.name == "factory"
    assert server.version == "9.9.9"


def test_builtin_prompts_registered(server: AragoraMCPServer) -> None:
    assert "debate-decision" in server._prompts


@pytest.mark.asyncio
async def test_list_tools(server: AragoraMCPServer) -> None:
    response = await server.list_tools()
    assert "tools" in response
    names = {t["name"] for t in response["tools"]}
    assert names == {"echo", "ping"}


@pytest.mark.asyncio
async def test_call_tool_success(server: AragoraMCPServer) -> None:
    response = await server.call_tool("ping", {})
    assert "content" in response
    payload = json.loads(response["content"][0]["text"])
    assert payload["status"] == "ok"


@pytest.mark.asyncio
async def test_call_tool_unknown(server: AragoraMCPServer) -> None:
    response = await server.call_tool("missing", {})
    payload = json.loads(response["content"][0]["text"])
    assert "error" in payload


@pytest.mark.asyncio
async def test_list_resources_empty(server: AragoraMCPServer) -> None:
    response = await server.list_resources()
    assert response["resources"] == []


@pytest.mark.asyncio
async def test_read_resource_unknown(server: AragoraMCPServer) -> None:
    response = await server.read_resource("unknown://id")
    payload = json.loads(response["contents"][0]["text"])
    assert "error" in payload


@pytest.mark.asyncio
async def test_list_prompts(server: AragoraMCPServer) -> None:
    response = await server.list_prompts()
    assert len(response["prompts"]) >= 1
