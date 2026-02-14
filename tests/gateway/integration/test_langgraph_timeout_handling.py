"""
Integration tests for LangGraph agent timeout and error handling with Aragora Gateway.

Tests LangGraph-specific features:
- Agent registration
- Graph-based execution
- Timeout handling for long-running graphs
- State management between nodes
- Node whitelist enforcement
- Error recovery in graph nodes
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.gateway.integration.conftest import (
    MockExternalFrameworkServer,
    register_external_agent,
)


class TestLangGraphTimeoutHandling:
    """Integration tests for LangGraph agents with gateway."""

    @pytest.mark.asyncio
    async def test_langgraph_agent_registration(
        self,
        gateway_server_context: dict,
        mock_external_server: MockExternalFrameworkServer,
    ) -> None:
        """Test registering a LangGraph agent via the gateway.

        Verifies that a LangGraph agent can be registered with the gateway
        and that the registration includes LangGraph-specific metadata.
        """
        # Register LangGraph agent
        agent_info = register_external_agent(
            gateway_server_context,
            name="langgraph-test",
            framework="langgraph",
            base_url="https://langgraph.example.com/api",
        )

        # Verify registration succeeded
        assert "langgraph-test" in gateway_server_context["external_agents"]
        assert agent_info["framework_type"] == "langgraph"
        assert agent_info["status"] == "registered"
        assert agent_info["base_url"] == "https://langgraph.example.com/api"

    @pytest.mark.asyncio
    async def test_langgraph_graph_execution(
        self,
        gateway_server_context: dict,
        mock_external_server: MockExternalFrameworkServer,
    ) -> None:
        """Test LangGraph graph-based execution works.

        Verifies that the LangGraph agent can execute graph workflows
        and properly handle the graph state and node transitions.
        """
        from aragora.agents.api_agents.langgraph_agent import (
            LangGraphAgent,
            LangGraphConfig,
        )

        # Configure LangGraph agent
        config = LangGraphConfig(
            base_url="http://localhost:8123",
            graph_id="test-workflow",
            recursion_limit=25,
            stream_mode="values",
        )

        # Mock the HTTP session
        with patch(
            "aragora.agents.api_agents.langgraph_agent.create_client_session"
        ) as mock_session_factory:
            mock_session = MagicMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={
                    "output": {
                        "messages": [
                            {"role": "user", "content": "Process this task"},
                            {"role": "assistant", "content": "Task processed via graph"},
                        ]
                    },
                    "thread_id": "thread-graph-123",
                }
            )

            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = MagicMock(return_value=mock_cm)

            session_cm = AsyncMock()
            session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session_factory.return_value = session_cm

            agent = LangGraphAgent(
                name="langgraph-executor",
                config=config,
                api_key="test-key",
            )

            # Verify config
            assert agent.langgraph_config.graph_id == "test-workflow"
            assert agent.langgraph_config.recursion_limit == 25

            # Execute the graph
            result = await agent.generate("Process this task")

            assert "Task processed" in result or result is not None
            # Thread ID should be stored after execution
            assert agent._thread_id == "thread-graph-123"

    @pytest.mark.asyncio
    async def test_langgraph_timeout_exceeded(
        self,
        gateway_server_context: dict,
        mock_external_server: MockExternalFrameworkServer,
    ) -> None:
        """Test that long-running graphs timeout properly.

        Verifies that the timeout mechanism works correctly for
        LangGraph executions that exceed the configured timeout.
        """
        from aragora.agents.api_agents.common import AgentTimeoutError
        from aragora.agents.api_agents.langgraph_agent import (
            LangGraphAgent,
            LangGraphConfig,
        )

        # Configure with short timeout
        config = LangGraphConfig(
            base_url="http://localhost:8123",
            graph_id="slow-graph",
            timeout=1,  # 1 second timeout
        )

        agent = LangGraphAgent(
            name="langgraph-timeout",
            config=config,
            api_key="test-key",
        )

        # Verify timeout is set
        assert agent.timeout == 1

        # Mock session that simulates a timeout
        with patch(
            "aragora.agents.api_agents.langgraph_agent.create_client_session"
        ) as mock_session_factory:
            mock_session = MagicMock()

            # Create an async context manager that raises TimeoutError
            async def raise_timeout(*args, **kwargs):
                raise TimeoutError("Request timed out")

            mock_cm = AsyncMock()
            mock_cm.__aenter__ = raise_timeout
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = MagicMock(return_value=mock_cm)

            session_cm = AsyncMock()
            session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session_factory.return_value = session_cm

            # Attempt to execute - should raise timeout error
            with pytest.raises(AgentTimeoutError) as exc_info:
                await agent.generate("Long running task")

            assert "timed out" in str(exc_info.value).lower()
            assert exc_info.value.agent_name == "langgraph-timeout"

    @pytest.mark.asyncio
    async def test_langgraph_state_management(
        self,
        gateway_server_context: dict,
        mock_external_server: MockExternalFrameworkServer,
    ) -> None:
        """Test that state is passed correctly between nodes.

        Verifies that the LangGraph agent properly manages state
        including state size validation and thread state operations.
        """
        from aragora.agents.api_agents.common import AgentAPIError
        from aragora.agents.api_agents.langgraph_agent import (
            LangGraphAgent,
            LangGraphConfig,
        )

        config = LangGraphConfig(
            base_url="http://localhost:8123",
            graph_id="stateful-graph",
            max_state_size=1024,  # 1KB limit for testing
        )

        agent = LangGraphAgent(
            name="langgraph-stateful",
            config=config,
            api_key="test-key",
        )

        # Test state size validation with small state (should pass)
        small_state = {"key": "value", "count": 42}
        # Should not raise
        agent._validate_state_size(small_state)

        # Test state size validation with large state (should fail)
        large_state = {"data": "x" * 2000}  # Exceeds 1KB limit

        with pytest.raises(AgentAPIError) as exc_info:
            agent._validate_state_size(large_state)

        assert "exceeds limit" in str(exc_info.value).lower()

        # Test thread ID management
        assert agent.get_thread_id() is None

        agent.set_thread_id("thread-test-123")
        assert agent.get_thread_id() == "thread-test-123"

        agent.clear_thread()
        assert agent.get_thread_id() is None

        # Test get_state requires thread_id
        with pytest.raises(AgentAPIError) as exc_info:
            await agent.get_state()

        assert "thread_id" in str(exc_info.value).lower()

        # Test update_state requires thread_id
        with pytest.raises(AgentAPIError) as exc_info:
            await agent.update_state({"key": "value"})

        assert "thread_id" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_langgraph_node_whitelist(
        self,
        gateway_server_context: dict,
        mock_external_server: MockExternalFrameworkServer,
    ) -> None:
        """Test that only whitelisted nodes are executed.

        Verifies that the node whitelist is enforced, blocking
        events from nodes that are not explicitly allowed.
        """
        from aragora.agents.api_agents.common import AgentAPIError
        from aragora.agents.api_agents.langgraph_agent import (
            LangGraphAgent,
            LangGraphConfig,
        )

        # Configure with limited allowed nodes
        config = LangGraphConfig(
            base_url="http://localhost:8123",
            graph_id="restricted-graph",
            allowed_nodes=["generate", "review", "output"],  # Only these are allowed
        )

        agent = LangGraphAgent(
            name="langgraph-whitelist",
            config=config,
            api_key="test-key",
        )

        # Test node validation
        assert agent._validate_node_allowed("generate") is True
        assert agent._validate_node_allowed("review") is True
        assert agent._validate_node_allowed("output") is True

        # Non-whitelisted nodes should be blocked
        assert agent._validate_node_allowed("execute_code") is False
        assert agent._validate_node_allowed("shell") is False
        assert agent._validate_node_allowed("arbitrary_node") is False

        # Test filtering response events
        events = [
            {"node": "generate", "data": "Generated content"},
            {"node": "execute_code", "data": "Dangerous code"},  # Should be filtered
            {"node": "review", "data": "Review complete"},
            {"node": "shell", "data": "Shell command"},  # Should be filtered
            {"node": "output", "data": "Final output"},
            {"data": "No node specified"},  # Should pass (no node)
        ]

        filtered = agent._filter_response_nodes(events)

        # Should only include allowed nodes and events without node
        assert len(filtered) == 4
        node_names = [e.get("node") for e in filtered]
        assert "execute_code" not in node_names
        assert "shell" not in node_names
        assert "generate" in node_names
        assert "review" in node_names
        assert "output" in node_names

        # Test empty whitelist allows all nodes
        config_no_whitelist = LangGraphConfig(
            base_url="http://localhost:8123",
            allowed_nodes=[],  # Empty = all allowed
        )
        agent_no_whitelist = LangGraphAgent(
            name="langgraph-no-whitelist",
            config=config_no_whitelist,
            api_key="test-key",
        )

        assert agent_no_whitelist._validate_node_allowed("any_node") is True
        assert agent_no_whitelist._validate_node_allowed("shell") is True

        # Test update_state with disallowed node
        agent.set_thread_id("thread-123")

        # Mock the HTTP session for update_state
        with patch(
            "aragora.agents.api_agents.langgraph_agent.create_client_session"
        ) as mock_session_factory:
            mock_session = MagicMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"success": True})

            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = MagicMock(return_value=mock_cm)
            mock_session.get = MagicMock(return_value=mock_cm)

            session_cm = AsyncMock()
            session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session_factory.return_value = session_cm

            # Attempt to update state as disallowed node
            with pytest.raises(AgentAPIError) as exc_info:
                await agent.update_state(
                    {"key": "value"},
                    as_node="execute_code",  # Not in whitelist
                )

            assert "not in allowed_nodes" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_langgraph_error_recovery(
        self,
        gateway_server_context: dict,
        mock_external_server: MockExternalFrameworkServer,
    ) -> None:
        """Test that errors in nodes are handled gracefully.

        Verifies that the LangGraph agent properly handles various
        error conditions including API errors, connection errors,
        and malformed responses.
        """
        from aragora.agents.api_agents.common import (
            AgentAPIError,
            AgentConnectionError,
        )
        from aragora.agents.api_agents.langgraph_agent import (
            LangGraphAgent,
            LangGraphConfig,
        )

        config = LangGraphConfig(
            base_url="http://localhost:8123",
            graph_id="error-prone-graph",
        )

        agent = LangGraphAgent(
            name="langgraph-recovery",
            config=config,
            api_key="test-key",
        )

        # Test handling of API error response
        with patch(
            "aragora.agents.api_agents.langgraph_agent.create_client_session"
        ) as mock_session_factory:
            mock_session = MagicMock()
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Internal server error")

            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = MagicMock(return_value=mock_cm)

            session_cm = AsyncMock()
            session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session_factory.return_value = session_cm

            with pytest.raises(AgentAPIError) as exc_info:
                await agent.generate("Test prompt")

            assert "500" in str(exc_info.value)
            assert exc_info.value.agent_name == "langgraph-recovery"

        # Test handling of connection error
        with patch(
            "aragora.agents.api_agents.langgraph_agent.create_client_session"
        ) as mock_session_factory:
            mock_session = MagicMock()

            # Simulate connection error
            import aiohttp

            async def raise_connection_error(*args, **kwargs):
                raise aiohttp.ClientConnectorError(MagicMock(), OSError("Connection refused"))

            mock_cm = AsyncMock()
            mock_cm.__aenter__ = raise_connection_error
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = MagicMock(return_value=mock_cm)

            session_cm = AsyncMock()
            session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session_factory.return_value = session_cm

            with pytest.raises(AgentConnectionError) as exc_info:
                await agent.generate("Test prompt")

            assert "Cannot connect" in str(exc_info.value)

        # Test response extraction from various formats
        # Format 1: Standard output with messages
        data1 = {"output": {"messages": [{"role": "assistant", "content": "Response 1"}]}}
        result1 = agent._extract_langgraph_response(data1)
        assert "Response 1" in result1

        # Format 2: Values format
        data2 = {"values": {"messages": [{"role": "assistant", "content": "Response 2"}]}}
        result2 = agent._extract_langgraph_response(data2)
        assert "Response 2" in result2

        # Format 3: Direct result
        data3 = {"output": {"result": "Direct result"}}
        result3 = agent._extract_langgraph_response(data3)
        assert "Direct result" in result3

        # Format 4: String output
        data4 = {"output": "String output"}
        result4 = agent._extract_langgraph_response(data4)
        assert "String output" in result4

        # Format 5: Fallback to JSON string for unknown format
        data5 = {"unknown_key": {"nested": "data"}}
        result5 = agent._extract_langgraph_response(data5)
        # Should be JSON serialized
        assert "unknown_key" in result5 or "nested" in result5

        # Format 6: Content blocks format
        data6 = {
            "output": {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"text": "Part 1"},
                            {"text": " Part 2"},
                        ],
                    }
                ]
            }
        }
        result6 = agent._extract_langgraph_response(data6)
        assert "Part 1" in result6
        assert "Part 2" in result6
