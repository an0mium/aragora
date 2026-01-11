"""
Tests for the Aragora MCP Server.

Tests cover:
- Server initialization
- Tool listing and schema validation
- Resource listing and templates
- Tool execution with mocked dependencies
- Error handling
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict


# Check if MCP is available
try:
    from mcp.types import Tool, TextContent, Resource, ResourceTemplate
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Tool = None
    TextContent = None
    Resource = None
    ResourceTemplate = None


pytestmark = pytest.mark.skipif(
    not MCP_AVAILABLE,
    reason="MCP package not installed"
)


class TestAragoraMCPServerInitialization:
    """Test server initialization."""

    def test_server_creation_without_mcp_raises(self):
        """Test that server raises ImportError without MCP package."""
        with patch.dict('sys.modules', {'mcp': None, 'mcp.server': None}):
            # Force reimport with mocked modules
            import importlib
            from aragora.mcp import server as mcp_server

            # Store original value
            original_available = mcp_server.MCP_AVAILABLE
            mcp_server.MCP_AVAILABLE = False

            try:
                with pytest.raises(ImportError, match="MCP package not installed"):
                    mcp_server.AragoraMCPServer()
            finally:
                mcp_server.MCP_AVAILABLE = original_available

    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    def test_server_creation_with_mcp(self):
        """Test successful server creation when MCP is available."""
        from aragora.mcp.server import AragoraMCPServer

        server = AragoraMCPServer()

        assert server.server is not None
        assert server._debates_cache == {}

    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    def test_server_has_correct_name(self):
        """Test server is registered with correct name."""
        from aragora.mcp.server import AragoraMCPServer

        server = AragoraMCPServer()

        # The Server class stores the name
        assert server.server.name == "aragora"


class TestMCPToolListing:
    """Test tool listing functionality."""

    @pytest.fixture
    def server(self):
        """Create an MCP server instance."""
        from aragora.mcp.server import AragoraMCPServer
        return AragoraMCPServer()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    async def test_list_tools_returns_four_tools(self, server):
        """Test that list_tools returns expected tools."""
        # Access the registered handler
        tools = await server.server._tool_handlers["list_tools"]()

        assert len(tools) == 4
        tool_names = {t.name for t in tools}
        assert tool_names == {"run_debate", "run_gauntlet", "list_agents", "get_debate"}

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    async def test_run_debate_tool_schema(self, server):
        """Test run_debate tool has correct schema."""
        tools = await server.server._tool_handlers["list_tools"]()

        run_debate = next(t for t in tools if t.name == "run_debate")

        assert run_debate.inputSchema["type"] == "object"
        assert "question" in run_debate.inputSchema["properties"]
        assert "agents" in run_debate.inputSchema["properties"]
        assert "rounds" in run_debate.inputSchema["properties"]
        assert "consensus" in run_debate.inputSchema["properties"]
        assert run_debate.inputSchema["required"] == ["question"]

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    async def test_run_gauntlet_tool_schema(self, server):
        """Test run_gauntlet tool has correct schema."""
        tools = await server.server._tool_handlers["list_tools"]()

        run_gauntlet = next(t for t in tools if t.name == "run_gauntlet")

        assert run_gauntlet.inputSchema["type"] == "object"
        assert "content" in run_gauntlet.inputSchema["properties"]
        assert "content_type" in run_gauntlet.inputSchema["properties"]
        assert "profile" in run_gauntlet.inputSchema["properties"]
        assert run_gauntlet.inputSchema["required"] == ["content"]

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    async def test_list_agents_tool_schema(self, server):
        """Test list_agents tool has minimal schema."""
        tools = await server.server._tool_handlers["list_tools"]()

        list_agents = next(t for t in tools if t.name == "list_agents")

        assert list_agents.inputSchema["type"] == "object"
        assert list_agents.inputSchema["properties"] == {}

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    async def test_get_debate_tool_schema(self, server):
        """Test get_debate tool has correct schema."""
        tools = await server.server._tool_handlers["list_tools"]()

        get_debate = next(t for t in tools if t.name == "get_debate")

        assert "debate_id" in get_debate.inputSchema["properties"]
        assert get_debate.inputSchema["required"] == ["debate_id"]


class TestMCPResourceListing:
    """Test resource listing functionality."""

    @pytest.fixture
    def server(self):
        """Create an MCP server instance."""
        from aragora.mcp.server import AragoraMCPServer
        return AragoraMCPServer()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    async def test_list_resources_empty_initially(self, server):
        """Test that resources list is empty initially."""
        resources = await server.server._tool_handlers["list_resources"]()

        assert resources == []

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    async def test_list_resources_after_debate(self, server):
        """Test that resources include cached debates."""
        # Add a debate to the cache
        server._debates_cache["test_123"] = {
            "debate_id": "test_123",
            "task": "Test debate question",
            "timestamp": "2026-01-11 12:00:00",
        }

        resources = await server.server._tool_handlers["list_resources"]()

        assert len(resources) == 1
        assert resources[0].uri == "debate://test_123"
        assert "Test debate" in resources[0].name

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    async def test_list_resource_templates(self, server):
        """Test resource templates are listed."""
        templates = await server.server._tool_handlers["list_resource_templates"]()

        assert len(templates) == 1
        assert templates[0].uriTemplate == "debate://{debate_id}"
        assert templates[0].mimeType == "application/json"


class TestMCPResourceReading:
    """Test resource reading functionality."""

    @pytest.fixture
    def server(self):
        """Create an MCP server instance."""
        from aragora.mcp.server import AragoraMCPServer
        return AragoraMCPServer()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    async def test_read_cached_debate(self, server):
        """Test reading a cached debate resource."""
        debate_data = {
            "debate_id": "test_456",
            "task": "What is the meaning of life?",
            "final_answer": "42",
        }
        server._debates_cache["test_456"] = debate_data

        result = await server.server._tool_handlers["read_resource"]("debate://test_456")

        parsed = json.loads(result)
        assert parsed["debate_id"] == "test_456"
        assert parsed["final_answer"] == "42"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    async def test_read_missing_debate(self, server):
        """Test reading a non-existent debate resource."""
        result = await server.server._tool_handlers["read_resource"]("debate://missing")

        parsed = json.loads(result)
        assert "error" in parsed
        assert "not found" in parsed["error"]

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    async def test_read_unknown_resource_type(self, server):
        """Test reading an unknown resource type."""
        result = await server.server._tool_handlers["read_resource"]("unknown://something")

        parsed = json.loads(result)
        assert "error" in parsed
        assert "Unknown resource" in parsed["error"]


class TestMCPToolExecution:
    """Test tool execution with mocked dependencies."""

    @pytest.fixture
    def server(self):
        """Create an MCP server instance."""
        from aragora.mcp.server import AragoraMCPServer
        return AragoraMCPServer()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    async def test_list_agents_tool_execution(self, server):
        """Test list_agents tool returns agents list."""
        with patch("aragora.agents.registry.list_available_agents") as mock_list:
            mock_list.return_value = ["anthropic-api", "openai-api", "gemini"]

            result = await server._list_agents()

            assert result["agents"] == ["anthropic-api", "openai-api", "gemini"]
            assert result["count"] == 3

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    async def test_list_agents_fallback_on_error(self, server):
        """Test list_agents returns fallback on error."""
        with patch("aragora.agents.registry.list_available_agents") as mock_list:
            mock_list.side_effect = Exception("Registry unavailable")

            result = await server._list_agents()

            assert "agents" in result
            assert len(result["agents"]) >= 5  # Fallback list
            assert "note" in result

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    async def test_get_debate_from_cache(self, server):
        """Test get_debate retrieves from cache."""
        server._debates_cache["cached_123"] = {
            "debate_id": "cached_123",
            "final_answer": "Cached result",
        }

        result = await server._get_debate({"debate_id": "cached_123"})

        assert result["debate_id"] == "cached_123"
        assert result["final_answer"] == "Cached result"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    async def test_get_debate_missing_id(self, server):
        """Test get_debate with missing ID returns error."""
        result = await server._get_debate({})

        assert "error" in result
        assert "required" in result["error"]

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    async def test_get_debate_not_found(self, server):
        """Test get_debate returns error for non-existent debate."""
        with patch("aragora.server.storage.get_debates_db") as mock_db:
            mock_db.return_value = None

            result = await server._get_debate({"debate_id": "nonexistent"})

            assert "error" in result
            assert "not found" in result["error"]

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    async def test_run_debate_missing_question(self, server):
        """Test run_debate with missing question returns error."""
        result = await server._run_debate({})

        assert "error" in result
        assert "Question is required" in result["error"]

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    async def test_run_debate_no_valid_agents(self, server):
        """Test run_debate with no valid agents returns error."""
        with patch("aragora.agents.base.create_agent") as mock_create:
            mock_create.side_effect = Exception("No API key")

            result = await server._run_debate({
                "question": "Test question",
                "agents": "fake-agent",
            })

            assert "error" in result
            assert "No valid agents" in result["error"]

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    async def test_run_gauntlet_missing_content(self, server):
        """Test run_gauntlet with missing content returns error."""
        result = await server._run_gauntlet({})

        assert "error" in result
        assert "Content is required" in result["error"]


class TestMCPToolCallHandler:
    """Test the unified tool call handler."""

    @pytest.fixture
    def server(self):
        """Create an MCP server instance."""
        from aragora.mcp.server import AragoraMCPServer
        return AragoraMCPServer()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    async def test_call_tool_unknown_tool(self, server):
        """Test calling unknown tool returns error."""
        handler = server.server._tool_handlers["call_tool"]

        result = await handler("unknown_tool", {})

        assert len(result) == 1
        parsed = json.loads(result[0].text)
        assert "error" in parsed
        assert "Unknown tool" in parsed["error"]

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    async def test_call_tool_handles_exceptions(self, server):
        """Test tool call handler catches exceptions."""
        handler = server.server._tool_handlers["call_tool"]

        # Mock _list_agents to raise an exception
        original_method = server._list_agents
        server._list_agents = AsyncMock(side_effect=RuntimeError("Test error"))

        try:
            result = await handler("list_agents", {})

            assert len(result) == 1
            parsed = json.loads(result[0].text)
            assert "error" in parsed
            assert "Test error" in parsed["error"]
        finally:
            server._list_agents = original_method

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    async def test_call_tool_returns_text_content(self, server):
        """Test tool calls return TextContent objects."""
        handler = server.server._tool_handlers["call_tool"]

        with patch("aragora.agents.registry.list_available_agents") as mock_list:
            mock_list.return_value = ["test-agent"]

            result = await handler("list_agents", {})

            assert len(result) == 1
            assert result[0].type == "text"
            parsed = json.loads(result[0].text)
            assert parsed["agents"] == ["test-agent"]


class TestMCPServerRun:
    """Test server run functionality."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    async def test_run_server_without_mcp_exits(self):
        """Test run_server exits with error when MCP unavailable."""
        from aragora.mcp import server as mcp_server

        original_available = mcp_server.MCP_AVAILABLE
        mcp_server.MCP_AVAILABLE = False

        try:
            with pytest.raises(SystemExit) as exc_info:
                await mcp_server.run_server()
            assert exc_info.value.code == 1
        finally:
            mcp_server.MCP_AVAILABLE = original_available


class TestMCPDebateCaching:
    """Test debate caching behavior."""

    @pytest.fixture
    def server(self):
        """Create an MCP server instance."""
        from aragora.mcp.server import AragoraMCPServer
        return AragoraMCPServer()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    async def test_debate_cached_after_run(self, server):
        """Test that debates are cached after successful run."""
        # Create mock result
        mock_result = MagicMock()
        mock_result.final_answer = "Test answer"
        mock_result.consensus_reached = True
        mock_result.confidence = 0.9
        mock_result.rounds_used = 3

        # Create mock agent
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"

        with patch("aragora.agents.base.create_agent") as mock_create, \
             patch("aragora.debate.orchestrator.Arena") as mock_arena, \
             patch("aragora.core.Environment"):

            mock_create.return_value = mock_agent
            mock_arena_instance = MagicMock()
            mock_arena_instance.run = AsyncMock(return_value=mock_result)
            mock_arena.return_value = mock_arena_instance

            result = await server._run_debate({
                "question": "Test question?",
                "agents": "test-agent",
                "rounds": 3,
            })

            # Verify result structure
            assert "debate_id" in result
            assert result["debate_id"].startswith("mcp_")
            assert result["final_answer"] == "Test answer"

            # Verify caching
            debate_id = result["debate_id"]
            assert debate_id in server._debates_cache
            assert server._debates_cache[debate_id]["task"] == "Test question?"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    async def test_debate_id_format(self, server):
        """Test debate ID has correct format."""
        mock_result = MagicMock()
        mock_result.final_answer = "Answer"
        mock_result.consensus_reached = False
        mock_result.confidence = 0.5
        mock_result.rounds_used = 1

        mock_agent = MagicMock()
        mock_agent.name = "agent"

        with patch("aragora.agents.base.create_agent") as mock_create, \
             patch("aragora.debate.orchestrator.Arena") as mock_arena, \
             patch("aragora.core.Environment"):

            mock_create.return_value = mock_agent
            mock_arena_instance = MagicMock()
            mock_arena_instance.run = AsyncMock(return_value=mock_result)
            mock_arena.return_value = mock_arena_instance

            result = await server._run_debate({"question": "Q?"})

            assert result["debate_id"].startswith("mcp_")
            assert len(result["debate_id"]) == 12  # "mcp_" + 8 hex chars


class TestMCPRoundsValidation:
    """Test rounds parameter validation."""

    @pytest.fixture
    def server(self):
        """Create an MCP server instance."""
        from aragora.mcp.server import AragoraMCPServer
        return AragoraMCPServer()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    async def test_rounds_clamped_to_minimum(self, server):
        """Test rounds below 1 are clamped to 1."""
        mock_result = MagicMock()
        mock_result.final_answer = "A"
        mock_result.consensus_reached = True
        mock_result.confidence = 1.0
        mock_result.rounds_used = 1

        mock_agent = MagicMock()
        mock_agent.name = "a"

        with patch("aragora.agents.base.create_agent") as mock_create, \
             patch("aragora.debate.orchestrator.Arena") as mock_arena, \
             patch("aragora.debate.orchestrator.DebateProtocol") as mock_protocol, \
             patch("aragora.core.Environment") as mock_env:

            mock_create.return_value = mock_agent
            mock_arena_instance = MagicMock()
            mock_arena_instance.run = AsyncMock(return_value=mock_result)
            mock_arena.return_value = mock_arena_instance

            await server._run_debate({"question": "Q?", "rounds": -5})

            # Check that Environment was called with rounds=1
            mock_env.assert_called_once()
            call_kwargs = mock_env.call_args[1]
            assert call_kwargs["max_rounds"] == 1

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP required")
    async def test_rounds_clamped_to_maximum(self, server):
        """Test rounds above 10 are clamped to 10."""
        mock_result = MagicMock()
        mock_result.final_answer = "A"
        mock_result.consensus_reached = True
        mock_result.confidence = 1.0
        mock_result.rounds_used = 10

        mock_agent = MagicMock()
        mock_agent.name = "a"

        with patch("aragora.agents.base.create_agent") as mock_create, \
             patch("aragora.debate.orchestrator.Arena") as mock_arena, \
             patch("aragora.debate.orchestrator.DebateProtocol") as mock_protocol, \
             patch("aragora.core.Environment") as mock_env:

            mock_create.return_value = mock_agent
            mock_arena_instance = MagicMock()
            mock_arena_instance.run = AsyncMock(return_value=mock_result)
            mock_arena.return_value = mock_arena_instance

            await server._run_debate({"question": "Q?", "rounds": 100})

            # Check that Environment was called with rounds=10
            mock_env.assert_called_once()
            call_kwargs = mock_env.call_args[1]
            assert call_kwargs["max_rounds"] == 10
