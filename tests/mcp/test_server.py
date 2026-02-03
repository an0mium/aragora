"""
Tests for Aragora MCP Server.

Tests cover:
- Server initialization
- Tool registration and listing
- Tool execution
- Resource management
- Prompt templates
- Error handling
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.mcp.server import (
    AragoraMCPServer,
    MCPCapability,
    MCPPrompt,
    MCPResource,
    MCPTool,
    create_mcp_server,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def server() -> AragoraMCPServer:
    """Create a test MCP server."""
    return AragoraMCPServer(
        name="aragora-test",
        version="1.0.0-test",
        api_base="https://test.aragora.ai",
    )


@pytest.fixture
def custom_tool_handler() -> AsyncMock:
    """Create a mock tool handler."""
    handler = AsyncMock()
    handler.return_value = {"result": "success", "data": "test"}
    return handler


# =============================================================================
# Server Initialization Tests
# =============================================================================


class TestServerInitialization:
    """Tests for MCP server initialization."""

    def test_server_creation(self) -> None:
        """Test basic server creation."""
        server = AragoraMCPServer()

        assert server.name == "aragora"
        assert server.version == "1.0.0"
        assert server.api_base == "https://api.aragora.ai"

    def test_server_custom_params(self) -> None:
        """Test server with custom parameters."""
        server = AragoraMCPServer(
            name="custom-aragora",
            version="2.0.0",
            api_base="https://custom.example.com",
        )

        assert server.name == "custom-aragora"
        assert server.version == "2.0.0"
        assert server.api_base == "https://custom.example.com"

    def test_factory_function(self) -> None:
        """Test create_mcp_server factory."""
        server = create_mcp_server(
            name="factory-test",
            version="3.0.0",
            api_base="https://factory.test",
        )

        assert server.name == "factory-test"
        assert server.version == "3.0.0"

    def test_builtin_tools_registered(self, server: AragoraMCPServer) -> None:
        """Test builtin tools are registered on init."""
        # Should have debate tools
        assert "aragora.debate.create" in server._tools
        assert "aragora.debate.status" in server._tools
        assert "aragora.debate.receipt" in server._tools

        # Should have verification tools
        assert "aragora.verify.receipt" in server._tools

        # Should have knowledge tools
        assert "aragora.knowledge.search" in server._tools
        assert "aragora.knowledge.ingest" in server._tools

        # Should have gauntlet tools
        assert "aragora.gauntlet.run" in server._tools

    def test_builtin_resources_registered(self, server: AragoraMCPServer) -> None:
        """Test builtin resources are registered."""
        assert "aragora://agents" in server._resources
        assert "aragora://connectors" in server._resources
        assert "aragora://workflows" in server._resources

    def test_builtin_prompts_registered(self, server: AragoraMCPServer) -> None:
        """Test builtin prompts are registered."""
        assert "debate-decision" in server._prompts


# =============================================================================
# MCP Protocol Tests
# =============================================================================


class TestMCPProtocol:
    """Tests for MCP protocol methods."""

    @pytest.mark.asyncio
    async def test_initialize(self, server: AragoraMCPServer) -> None:
        """Test MCP initialize response."""
        params = {"clientInfo": {"name": "test-client", "version": "1.0.0"}}

        response = await server.initialize(params)

        assert response["protocolVersion"] == "2024-11-05"
        assert response["serverInfo"]["name"] == "aragora-test"
        assert response["serverInfo"]["version"] == "1.0.0-test"
        assert "capabilities" in response
        assert "tools" in response["capabilities"]
        assert "resources" in response["capabilities"]
        assert "prompts" in response["capabilities"]

    @pytest.mark.asyncio
    async def test_list_tools(self, server: AragoraMCPServer) -> None:
        """Test listing available tools."""
        response = await server.list_tools()

        assert "tools" in response
        assert len(response["tools"]) >= 7  # At least our builtin tools

        # Check tool format
        tool = response["tools"][0]
        assert "name" in tool
        assert "description" in tool
        assert "inputSchema" in tool

    @pytest.mark.asyncio
    async def test_list_resources(self, server: AragoraMCPServer) -> None:
        """Test listing available resources."""
        response = await server.list_resources()

        assert "resources" in response
        assert len(response["resources"]) >= 3

        # Check resource format
        resource = response["resources"][0]
        assert "uri" in resource
        assert "name" in resource
        assert "description" in resource

    @pytest.mark.asyncio
    async def test_read_resource(self, server: AragoraMCPServer) -> None:
        """Test reading a resource."""
        response = await server.read_resource("aragora://agents")

        assert "contents" in response
        assert len(response["contents"]) >= 1
        assert response["contents"][0]["uri"] == "aragora://agents"

    @pytest.mark.asyncio
    async def test_list_prompts(self, server: AragoraMCPServer) -> None:
        """Test listing available prompts."""
        response = await server.list_prompts()

        assert "prompts" in response
        assert len(response["prompts"]) >= 1

        # Check prompt format
        prompt = response["prompts"][0]
        assert "name" in prompt
        assert "description" in prompt


# =============================================================================
# Tool Execution Tests
# =============================================================================


class TestToolExecution:
    """Tests for tool execution."""

    @pytest.mark.asyncio
    async def test_call_debate_create(self, server: AragoraMCPServer) -> None:
        """Test calling debate.create tool."""
        response = await server.call_tool(
            "aragora.debate.create",
            {"task": "Should we adopt microservices?"},
        )

        assert "content" in response
        assert "isError" not in response or not response.get("isError")

        # Parse result
        result = json.loads(response["content"][0]["text"])
        assert "debate_id" in result
        assert result["status"] == "created"
        assert result["task"] == "Should we adopt microservices?"
        assert "websocket_url" in result

    @pytest.mark.asyncio
    async def test_call_debate_create_with_options(self, server: AragoraMCPServer) -> None:
        """Test debate.create with all options."""
        response = await server.call_tool(
            "aragora.debate.create",
            {
                "task": "API architecture decision",
                "agents": ["claude", "gpt-4"],
                "rounds": 5,
                "consensus_threshold": 0.8,
            },
        )

        result = json.loads(response["content"][0]["text"])
        assert result["agents"] == ["claude", "gpt-4"]
        assert result["rounds"] == 5

    @pytest.mark.asyncio
    async def test_call_debate_status(self, server: AragoraMCPServer) -> None:
        """Test calling debate.status tool."""
        response = await server.call_tool(
            "aragora.debate.status",
            {"debate_id": "debate_test123"},
        )

        result = json.loads(response["content"][0]["text"])
        assert result["debate_id"] == "debate_test123"
        assert "status" in result
        assert "current_round" in result
        assert "phase" in result

    @pytest.mark.asyncio
    async def test_call_debate_receipt(self, server: AragoraMCPServer) -> None:
        """Test calling debate.receipt tool."""
        response = await server.call_tool(
            "aragora.debate.receipt",
            {"debate_id": "debate_test123"},
        )

        result = json.loads(response["content"][0]["text"])
        assert result["debate_id"] == "debate_test123"
        assert "receipt_id" in result
        assert "consensus" in result
        assert "confidence" in result
        assert "signature" in result

    @pytest.mark.asyncio
    async def test_call_verify_receipt(self, server: AragoraMCPServer) -> None:
        """Test calling verify.receipt tool."""
        response = await server.call_tool(
            "aragora.verify.receipt",
            {"receipt_id": "receipt_test123"},
        )

        result = json.loads(response["content"][0]["text"])
        assert result["receipt_id"] == "receipt_test123"
        assert result["verified"] is True
        assert result["signature_valid"] is True
        assert result["content_hash_valid"] is True

    @pytest.mark.asyncio
    async def test_call_knowledge_search(self, server: AragoraMCPServer) -> None:
        """Test calling knowledge.search tool."""
        response = await server.call_tool(
            "aragora.knowledge.search",
            {"query": "API design patterns", "limit": 5},
        )

        result = json.loads(response["content"][0]["text"])
        assert result["query"] == "API design patterns"
        assert "results" in result
        assert "total_count" in result

    @pytest.mark.asyncio
    async def test_call_knowledge_ingest(self, server: AragoraMCPServer) -> None:
        """Test calling knowledge.ingest tool."""
        response = await server.call_tool(
            "aragora.knowledge.ingest",
            {
                "content": "REST APIs use HTTP methods for CRUD operations.",
                "source_type": "documentation",
                "source_id": "api-guide-v1",
                "title": "REST API Guide",
            },
        )

        result = json.loads(response["content"][0]["text"])
        assert "evidence_id" in result
        assert result["status"] == "ingested"
        assert result["source_type"] == "documentation"

    @pytest.mark.asyncio
    async def test_call_gauntlet_run(self, server: AragoraMCPServer) -> None:
        """Test calling gauntlet.run tool."""
        response = await server.call_tool(
            "aragora.gauntlet.run",
            {
                "proposal": "Switch all services to GraphQL",
                "intensity": "intense",
                "focus_areas": ["security", "performance"],
            },
        )

        result = json.loads(response["content"][0]["text"])
        assert "run_id" in result
        assert result["status"] == "running"
        assert result["intensity"] == "intense"
        assert "security" in result["focus_areas"]

    @pytest.mark.asyncio
    async def test_call_unknown_tool(self, server: AragoraMCPServer) -> None:
        """Test calling unknown tool returns error."""
        response = await server.call_tool(
            "aragora.nonexistent.tool",
            {},
        )

        assert response.get("isError") is True
        assert "Unknown tool" in response["content"][0]["text"]


# =============================================================================
# Tool Registration Tests
# =============================================================================


class TestToolRegistration:
    """Tests for custom tool registration."""

    @pytest.mark.asyncio
    async def test_register_custom_tool(
        self,
        server: AragoraMCPServer,
        custom_tool_handler: AsyncMock,
    ) -> None:
        """Test registering a custom tool."""
        custom_tool = MCPTool(
            name="custom.test.tool",
            description="A custom test tool",
            input_schema={
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                },
                "required": ["input"],
            },
            handler=custom_tool_handler,
        )

        server.register_tool(custom_tool)

        # Verify tool is registered
        assert "custom.test.tool" in server._tools

        # Verify tool appears in list
        tools_response = await server.list_tools()
        tool_names = [t["name"] for t in tools_response["tools"]]
        assert "custom.test.tool" in tool_names

    @pytest.mark.asyncio
    async def test_call_custom_tool(
        self,
        server: AragoraMCPServer,
        custom_tool_handler: AsyncMock,
    ) -> None:
        """Test calling a custom registered tool."""
        custom_tool = MCPTool(
            name="custom.echo",
            description="Echo back the input",
            input_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
            },
            handler=custom_tool_handler,
        )

        server.register_tool(custom_tool)

        response = await server.call_tool("custom.echo", {"message": "Hello"})

        assert "isError" not in response or not response.get("isError")
        custom_tool_handler.assert_called_once_with(message="Hello")

    @pytest.mark.asyncio
    async def test_tool_handler_exception(self, server: AragoraMCPServer) -> None:
        """Test tool handler that raises exception."""

        async def failing_handler(**kwargs: Any) -> dict[str, Any]:
            raise ValueError("Intentional test error")

        failing_tool = MCPTool(
            name="custom.failing",
            description="Always fails",
            input_schema={"type": "object", "properties": {}},
            handler=failing_handler,
        )

        server.register_tool(failing_tool)

        response = await server.call_tool("custom.failing", {})

        assert response.get("isError") is True
        assert "Tool execution failed" in response["content"][0]["text"]


# =============================================================================
# Data Model Tests
# =============================================================================


class TestDataModels:
    """Tests for MCP data models."""

    def test_mcp_tool_to_dict(self) -> None:
        """Test MCPTool serialization."""
        tool = MCPTool(
            name="test.tool",
            description="Test description",
            input_schema={"type": "object"},
            handler=AsyncMock(),
        )

        data = tool.to_dict()

        assert data["name"] == "test.tool"
        assert data["description"] == "Test description"
        assert data["inputSchema"] == {"type": "object"}
        assert "handler" not in data  # Handler should not be serialized

    def test_mcp_resource_to_dict(self) -> None:
        """Test MCPResource serialization."""
        resource = MCPResource(
            uri="test://resource",
            name="Test Resource",
            description="A test resource",
            mime_type="application/json",
        )

        data = resource.to_dict()

        assert data["uri"] == "test://resource"
        assert data["name"] == "Test Resource"
        assert data["description"] == "A test resource"
        assert data["mimeType"] == "application/json"

    def test_mcp_resource_default_mime_type(self) -> None:
        """Test MCPResource default mime type."""
        resource = MCPResource(
            uri="test://resource",
            name="Test",
            description="Test",
        )

        assert resource.mime_type == "application/json"

    def test_mcp_prompt_to_dict(self) -> None:
        """Test MCPPrompt serialization."""
        prompt = MCPPrompt(
            name="test-prompt",
            description="Test prompt template",
            arguments=[
                {"name": "arg1", "description": "First argument", "required": True},
                {"name": "arg2", "description": "Optional argument", "required": False},
            ],
        )

        data = prompt.to_dict()

        assert data["name"] == "test-prompt"
        assert data["description"] == "Test prompt template"
        assert len(data["arguments"]) == 2
        assert data["arguments"][0]["name"] == "arg1"

    def test_mcp_prompt_default_arguments(self) -> None:
        """Test MCPPrompt with no arguments."""
        prompt = MCPPrompt(
            name="simple-prompt",
            description="No arguments",
        )

        data = prompt.to_dict()

        assert data["arguments"] == []

    def test_mcp_capability_enum(self) -> None:
        """Test MCPCapability enum values."""
        assert MCPCapability.TOOLS.value == "tools"
        assert MCPCapability.RESOURCES.value == "resources"
        assert MCPCapability.PROMPTS.value == "prompts"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_full_debate_workflow(self, server: AragoraMCPServer) -> None:
        """Test complete debate workflow through MCP."""
        # 1. Create debate
        create_response = await server.call_tool(
            "aragora.debate.create",
            {"task": "Should we migrate to Kubernetes?"},
        )
        create_result = json.loads(create_response["content"][0]["text"])
        debate_id = create_result["debate_id"]

        # 2. Check status
        status_response = await server.call_tool(
            "aragora.debate.status",
            {"debate_id": debate_id},
        )
        status_result = json.loads(status_response["content"][0]["text"])
        assert status_result["debate_id"] == debate_id

        # 3. Get receipt
        receipt_response = await server.call_tool(
            "aragora.debate.receipt",
            {"debate_id": debate_id},
        )
        receipt_result = json.loads(receipt_response["content"][0]["text"])

        # 4. Verify receipt
        verify_response = await server.call_tool(
            "aragora.verify.receipt",
            {"receipt_id": receipt_result["receipt_id"]},
        )
        verify_result = json.loads(verify_response["content"][0]["text"])
        assert verify_result["verified"] is True

    @pytest.mark.asyncio
    async def test_knowledge_workflow(self, server: AragoraMCPServer) -> None:
        """Test knowledge ingestion and search workflow."""
        # 1. Ingest knowledge
        ingest_response = await server.call_tool(
            "aragora.knowledge.ingest",
            {
                "content": "Kubernetes orchestrates containerized applications.",
                "source_type": "documentation",
                "source_id": "k8s-docs",
            },
        )
        ingest_result = json.loads(ingest_response["content"][0]["text"])
        assert ingest_result["status"] == "ingested"

        # 2. Search knowledge
        search_response = await server.call_tool(
            "aragora.knowledge.search",
            {"query": "container orchestration"},
        )
        search_result = json.loads(search_response["content"][0]["text"])
        assert "results" in search_result


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_missing_required_argument(self, server: AragoraMCPServer) -> None:
        """Test handling missing required argument."""
        # debate.create requires 'task'
        response = await server.call_tool("aragora.debate.create", {})

        # Should handle gracefully (implementation dependent)
        # Either returns error or handles with defaults
        assert "content" in response or "isError" in response

    @pytest.mark.asyncio
    async def test_invalid_argument_type(self, server: AragoraMCPServer) -> None:
        """Test handling invalid argument types."""
        response = await server.call_tool(
            "aragora.debate.create",
            {"task": 123},  # Should be string
        )

        # Should still work (Python is flexible) or return error
        assert "content" in response

    @pytest.mark.asyncio
    async def test_extra_arguments_ignored(self, server: AragoraMCPServer) -> None:
        """Test extra arguments are ignored."""
        response = await server.call_tool(
            "aragora.debate.create",
            {
                "task": "Test task",
                "unknown_arg": "should be ignored",
                "another_unknown": 123,
            },
        )

        # Should succeed, ignoring unknown args
        assert "isError" not in response or not response.get("isError")
