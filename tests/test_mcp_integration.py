"""
Tests for Aragora MCP Server Integration.

Tests the MCP server initialization and module structure.
For detailed tool tests, see test_mcp_tools.py.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch


class TestMCPModuleStructure:
    """Tests for MCP module structure and exports."""

    def test_mcp_module_imports(self):
        """MCP module can be imported."""
        from aragora import mcp

        assert mcp is not None

    def test_mcp_exports_server_class(self):
        """MCP module exports AragoraMCPServer."""
        from aragora.mcp import AragoraMCPServer

        assert AragoraMCPServer is not None

    def test_mcp_exports_run_server(self):
        """MCP module exports run_server function."""
        from aragora.mcp import run_server

        assert callable(run_server)

    def test_mcp_exports_tool_functions(self):
        """MCP module exports all tool functions."""
        from aragora.mcp import (
            run_debate_tool,
            run_gauntlet_tool,
            list_agents_tool,
            get_debate_tool,
        )

        assert callable(run_debate_tool)
        assert callable(run_gauntlet_tool)
        assert callable(list_agents_tool)
        assert callable(get_debate_tool)


class TestMCPServerInitialization:
    """Tests for AragoraMCPServer initialization."""

    def test_server_requires_mcp_package(self):
        """Server raises ImportError when MCP package not available."""
        import aragora.mcp.server as server_module

        original_available = server_module.MCP_AVAILABLE

        try:
            server_module.MCP_AVAILABLE = False

            with pytest.raises(ImportError) as exc_info:
                server_module.AragoraMCPServer()

            assert "MCP package not installed" in str(exc_info.value)
        finally:
            server_module.MCP_AVAILABLE = original_available

    def test_server_module_has_expected_exports(self):
        """Server module exports expected classes and functions."""
        from aragora.mcp import server

        assert hasattr(server, "AragoraMCPServer")
        assert hasattr(server, "run_server")
        assert hasattr(server, "MCP_AVAILABLE")

    def test_mcp_available_is_boolean(self):
        """MCP_AVAILABLE flag is a boolean."""
        from aragora.mcp.server import MCP_AVAILABLE

        assert isinstance(MCP_AVAILABLE, bool)


class TestMCPServerRun:
    """Tests for MCP server run functionality."""

    @pytest.mark.asyncio
    async def test_run_server_fails_without_mcp(self):
        """run_server exits when MCP not available."""
        import aragora.mcp.server as server_module

        original_available = server_module.MCP_AVAILABLE

        try:
            server_module.MCP_AVAILABLE = False

            with pytest.raises(SystemExit) as exc_info:
                await server_module.run_server()

            assert exc_info.value.code == 1
        finally:
            server_module.MCP_AVAILABLE = original_available


class TestMCPToolsModuleStructure:
    """Tests for tools module structure."""

    def test_tools_module_imports(self):
        """Tools module can be imported."""
        from aragora.mcp import tools

        assert tools is not None

    def test_tools_metadata_defined(self):
        """TOOLS_METADATA is defined and has expected structure."""
        from aragora.mcp.tools import TOOLS_METADATA

        assert isinstance(TOOLS_METADATA, list)
        assert len(TOOLS_METADATA) == 24  # Full tool set from tools_module

        tool_names = [t["name"] for t in TOOLS_METADATA]
        # Core debate tools
        assert "run_debate" in tool_names
        assert "run_gauntlet" in tool_names
        assert "list_agents" in tool_names
        assert "search_debates" in tool_names
        assert "get_agent_history" in tool_names
        assert "get_consensus_proofs" in tool_names
        assert "list_trending_topics" in tool_names
        assert "get_debate" in tool_names
        # Additional tools from tools_module
        assert "fork_debate" in tool_names
        assert "query_memory" in tool_names
        assert "create_checkpoint" in tool_names
        assert "verify_consensus" in tool_names
        assert "search_evidence" in tool_names

    def test_tools_metadata_has_required_fields(self):
        """Each tool in TOOLS_METADATA has required fields."""
        from aragora.mcp.tools import TOOLS_METADATA

        for tool in TOOLS_METADATA:
            assert "name" in tool, f"Tool missing 'name' field"
            assert (
                "description" in tool
            ), f"Tool {tool.get('name', 'unknown')} missing 'description'"
            assert "function" in tool, f"Tool {tool.get('name', 'unknown')} missing 'function'"
            assert "parameters" in tool, f"Tool {tool.get('name', 'unknown')} missing 'parameters'"
            assert callable(tool["function"]), f"Tool {tool['name']} function not callable"

    def test_run_debate_has_required_parameter(self):
        """run_debate tool requires question parameter."""
        from aragora.mcp.tools import TOOLS_METADATA

        run_debate = next(t for t in TOOLS_METADATA if t["name"] == "run_debate")
        assert "question" in run_debate["parameters"]
        assert run_debate["parameters"]["question"].get("required") is True

    def test_run_gauntlet_has_required_parameter(self):
        """run_gauntlet tool requires content parameter."""
        from aragora.mcp.tools import TOOLS_METADATA

        run_gauntlet = next(t for t in TOOLS_METADATA if t["name"] == "run_gauntlet")
        assert "content" in run_gauntlet["parameters"]
        assert run_gauntlet["parameters"]["content"].get("required") is True

    def test_get_debate_has_required_parameter(self):
        """get_debate tool requires debate_id parameter."""
        from aragora.mcp.tools import TOOLS_METADATA

        get_debate = next(t for t in TOOLS_METADATA if t["name"] == "get_debate")
        assert "debate_id" in get_debate["parameters"]
        assert get_debate["parameters"]["debate_id"].get("required") is True

    def test_list_agents_has_no_required_parameters(self):
        """list_agents tool has no required parameters."""
        from aragora.mcp.tools import TOOLS_METADATA

        list_agents = next(t for t in TOOLS_METADATA if t["name"] == "list_agents")
        assert list_agents["parameters"] == {}


class TestMCPToolFunctions:
    """Basic tests for MCP tool functions."""

    @pytest.mark.asyncio
    async def test_run_debate_empty_question_returns_error(self):
        """run_debate returns error for empty question."""
        from aragora.mcp.tools import run_debate_tool

        result = await run_debate_tool(question="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_run_gauntlet_empty_content_returns_error(self):
        """run_gauntlet returns error for empty content."""
        from aragora.mcp.tools import run_gauntlet_tool

        result = await run_gauntlet_tool(content="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_debate_empty_id_returns_error(self):
        """get_debate returns error for empty debate_id."""
        from aragora.mcp.tools import get_debate_tool

        result = await get_debate_tool(debate_id="")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_list_agents_returns_agents_list(self):
        """list_agents returns a list of agents (fallback or registry)."""
        from aragora.mcp.tools import list_agents_tool

        result = await list_agents_tool()

        # Should return either real agents or fallback list
        assert "agents" in result
        assert "count" in result
        assert isinstance(result["agents"], list)
        # Fallback list has at least 5 agents
        assert result["count"] >= 1


class TestMCPToolDefaults:
    """Tests for MCP tool default values."""

    def test_run_debate_default_agents(self):
        """run_debate has default agents."""
        from aragora.mcp.tools import TOOLS_METADATA

        run_debate = next(t for t in TOOLS_METADATA if t["name"] == "run_debate")
        assert run_debate["parameters"]["agents"]["default"] == "anthropic-api,openai-api"

    def test_run_debate_default_rounds(self):
        """run_debate has default rounds."""
        from aragora.mcp.tools import TOOLS_METADATA

        run_debate = next(t for t in TOOLS_METADATA if t["name"] == "run_debate")
        assert run_debate["parameters"]["rounds"]["default"] == 3

    def test_run_debate_default_consensus(self):
        """run_debate has default consensus."""
        from aragora.mcp.tools import TOOLS_METADATA

        run_debate = next(t for t in TOOLS_METADATA if t["name"] == "run_debate")
        assert run_debate["parameters"]["consensus"]["default"] == "majority"

    def test_run_gauntlet_default_content_type(self):
        """run_gauntlet has default content_type."""
        from aragora.mcp.tools import TOOLS_METADATA

        run_gauntlet = next(t for t in TOOLS_METADATA if t["name"] == "run_gauntlet")
        assert run_gauntlet["parameters"]["content_type"]["default"] == "spec"

    def test_run_gauntlet_default_profile(self):
        """run_gauntlet has default profile."""
        from aragora.mcp.tools import TOOLS_METADATA

        run_gauntlet = next(t for t in TOOLS_METADATA if t["name"] == "run_gauntlet")
        assert run_gauntlet["parameters"]["profile"]["default"] == "quick"


class TestMCPToolValidation:
    """Tests for MCP tool input validation."""

    @pytest.mark.asyncio
    async def test_run_debate_invalid_agents_returns_error(self):
        """run_debate returns error when agents can't be created."""
        from aragora.mcp.tools import run_debate_tool

        # Patch at the import location inside the function
        with patch("aragora.agents.base.create_agent", side_effect=ValueError("No API key")):
            result = await run_debate_tool(
                question="Test?",
                agents="invalid-agent",
            )

            # Should fail on agent creation
            assert "error" in result
            assert "No valid agents" in result["error"]


class TestMCPAsyncFunctions:
    """Tests for async nature of MCP tool functions."""

    def test_run_debate_is_async(self):
        """run_debate_tool is an async function."""
        import asyncio
        from aragora.mcp.tools import run_debate_tool

        assert asyncio.iscoroutinefunction(run_debate_tool)

    def test_run_gauntlet_is_async(self):
        """run_gauntlet_tool is an async function."""
        import asyncio
        from aragora.mcp.tools import run_gauntlet_tool

        assert asyncio.iscoroutinefunction(run_gauntlet_tool)

    def test_list_agents_is_async(self):
        """list_agents_tool is an async function."""
        import asyncio
        from aragora.mcp.tools import list_agents_tool

        assert asyncio.iscoroutinefunction(list_agents_tool)

    def test_get_debate_is_async(self):
        """get_debate_tool is an async function."""
        import asyncio
        from aragora.mcp.tools import get_debate_tool

        assert asyncio.iscoroutinefunction(get_debate_tool)
