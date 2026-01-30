"""
Tests for A2A Protocol Client.

Tests cover:
- Client initialization
- Agent registration and lookup
- Agent listing with capability filter
- Error handling for missing agents
- Context manager behavior
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.protocols.a2a.client import A2AClient, A2AClientError
from aragora.protocols.a2a.types import (
    AgentCapability,
    AgentCard,
    TaskPriority,
    TaskResult,
    TaskStatus,
)


# ============================================================================
# A2AClientError Tests
# ============================================================================


class TestA2AClientError:
    """Tests for A2AClientError."""

    def test_basic_error(self):
        err = A2AClientError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.agent_name is None
        assert err.task_id is None

    def test_error_with_context(self):
        err = A2AClientError("Failed", agent_name="agent-1", task_id="t_123")
        assert err.agent_name == "agent-1"
        assert err.task_id == "t_123"

    def test_is_exception(self):
        assert issubclass(A2AClientError, Exception)


# ============================================================================
# Client Initialization Tests
# ============================================================================


class TestClientInit:
    """Tests for A2AClient initialization."""

    def test_default_config(self):
        client = A2AClient()
        assert client._timeout == 300.0
        assert client._max_retries == 3
        assert client._agents == {}
        assert client._client is None

    def test_custom_config(self):
        client = A2AClient(timeout=60.0, max_retries=5)
        assert client._timeout == 60.0
        assert client._max_retries == 5


# ============================================================================
# Agent Registration Tests
# ============================================================================


class TestAgentRegistration:
    """Tests for agent registration and lookup."""

    def test_register_agent(self):
        client = A2AClient()
        agent = AgentCard(name="test-agent", description="Test")
        client.register_agent(agent)
        assert client.get_agent("test-agent") is agent

    def test_get_agent_not_found(self):
        client = A2AClient()
        assert client.get_agent("nonexistent") is None

    def test_register_overwrites(self):
        client = A2AClient()
        agent1 = AgentCard(name="agent", description="First")
        agent2 = AgentCard(name="agent", description="Second")
        client.register_agent(agent1)
        client.register_agent(agent2)
        assert client.get_agent("agent").description == "Second"


# ============================================================================
# Agent Listing Tests
# ============================================================================


class TestAgentListing:
    """Tests for listing agents."""

    def test_list_all_agents(self):
        client = A2AClient()
        client.register_agent(AgentCard(name="a1", description="Agent 1"))
        client.register_agent(AgentCard(name="a2", description="Agent 2"))
        agents = client.list_agents()
        assert len(agents) == 2

    def test_list_agents_by_capability(self):
        client = A2AClient()
        client.register_agent(
            AgentCard(
                name="debater",
                description="Debater",
                capabilities=[AgentCapability.DEBATE],
            )
        )
        client.register_agent(
            AgentCard(
                name="auditor",
                description="Auditor",
                capabilities=[AgentCapability.AUDIT],
            )
        )
        debate_agents = client.list_agents(capability=AgentCapability.DEBATE)
        assert len(debate_agents) == 1
        assert debate_agents[0].name == "debater"

    def test_list_agents_empty(self):
        client = A2AClient()
        assert client.list_agents() == []


# ============================================================================
# Invoke Tests
# ============================================================================


class TestInvoke:
    """Tests for agent invocation."""

    @pytest.mark.asyncio
    async def test_invoke_agent_not_found(self):
        """Invoke raises error for unknown agent."""
        client = A2AClient()
        with pytest.raises(A2AClientError, match="Agent not found"):
            await client.invoke("unknown-agent", "Do something")

    @pytest.mark.asyncio
    async def test_invoke_no_endpoint(self):
        """Invoke raises error for agent without endpoint."""
        client = A2AClient()
        client.register_agent(AgentCard(name="no-endpoint", description="No endpoint"))
        with pytest.raises(A2AClientError, match="no endpoint"):
            await client.invoke("no-endpoint", "Do something")

    @pytest.mark.asyncio
    async def test_invoke_success(self):
        """Invoke returns TaskResult on success."""
        client = A2AClient()
        client.register_agent(
            AgentCard(
                name="good-agent",
                description="Good agent",
                endpoint="https://agent.example.com",
            )
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "task_id": "t_123",
            "agent_name": "good-agent",
            "status": "completed",
            "output": "Result",
        }

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        client._client = mock_http_client

        result = await client.invoke("good-agent", "Do something")
        assert result.status == TaskStatus.COMPLETED
        assert result.output == "Result"


# ============================================================================
# Context Manager Tests
# ============================================================================


class TestContextManager:
    """Tests for async context manager behavior."""

    @pytest.mark.asyncio
    async def test_context_manager_creates_client(self):
        """Async context manager creates HTTP client."""
        async with A2AClient() as client:
            assert client._client is not None

    @pytest.mark.asyncio
    async def test_context_manager_closes_client(self):
        """Async context manager closes HTTP client on exit."""
        client = A2AClient()
        async with client:
            assert client._client is not None
        assert client._client is None


# ============================================================================
# Task Status and Cancel Tests
# ============================================================================


class TestTaskStatusAndCancel:
    """Tests for task status and cancel operations."""

    @pytest.mark.asyncio
    async def test_get_task_status_no_agent(self):
        """get_task_status raises for unknown agent."""
        client = A2AClient()
        with pytest.raises(A2AClientError, match="Agent not found"):
            await client.get_task_status("unknown", "t_123")

    @pytest.mark.asyncio
    async def test_cancel_task_no_agent(self):
        """cancel_task returns False for unknown agent."""
        client = A2AClient()
        result = await client.cancel_task("unknown", "t_123")
        assert result is False
