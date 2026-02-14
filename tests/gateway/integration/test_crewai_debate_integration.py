"""
Integration tests for CrewAI agent with Aragora Gateway.

Tests CrewAI-specific features:
- Agent registration through gateway
- Process modes (sequential, hierarchical)
- Proposal generation
- Tool whitelisting
- Rate limiting
"""

from __future__ import annotations

import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.gateway.integration.conftest import (
    MockExternalFrameworkServer,
    register_external_agent,
)


@pytest.fixture(autouse=True)
def skip_ssrf_validation(monkeypatch):
    """Skip SSRF validation in tests to allow localhost URLs."""
    # Patch the SSRF validation to allow localhost in tests
    monkeypatch.setattr(
        "aragora.agents.api_agents.external_framework.ExternalFrameworkAgent._validate_endpoint_url",
        lambda self, url: None,
    )


class TestCrewAIDebateIntegration:
    """Integration tests for CrewAI agents with gateway."""

    @pytest.mark.asyncio
    async def test_crewai_agent_registration(
        self,
        gateway_server_context: dict,
        mock_external_server: MockExternalFrameworkServer,
    ) -> None:
        """Test registering a CrewAI agent through the gateway.

        Verifies that a CrewAI agent can be registered with the gateway
        and that the registration includes framework-specific metadata.
        """
        # Register CrewAI agent
        agent_info = register_external_agent(
            gateway_server_context,
            name="crewai-test",
            framework="crewai",
            base_url="https://crewai.example.com/api",
        )

        # Verify registration succeeded
        assert "crewai-test" in gateway_server_context["external_agents"]
        assert agent_info["framework_type"] == "crewai"
        assert agent_info["status"] == "registered"
        assert agent_info["base_url"] == "https://crewai.example.com/api"

    @pytest.mark.asyncio
    async def test_crewai_proposal_generation(
        self,
        gateway_server_context: dict,
        mock_external_server: MockExternalFrameworkServer,
    ) -> None:
        """Test CrewAI agent generating a proposal through gateway.

        Verifies that the CrewAI agent can generate proposals when
        called through the gateway, with proper request handling.
        """
        # Setup mock server response for kickoff endpoint
        mock_external_server.set_response(
            "/v1/crew/kickoff",
            {
                "status": "success",
                "result": "CrewAI generated proposal for the task",
                "crew_output": {
                    "final_output": "Detailed proposal for implementing a caching system",
                    "tasks_outputs": [
                        {"task": "research", "output": "Analyzed requirements"},
                        {"task": "design", "output": "Created architecture"},
                    ],
                },
            },
        )

        # Register agent
        register_external_agent(gateway_server_context, "crewai-proposal", "crewai")

        # Mock the CrewAI agent
        with patch("aragora.agents.api_agents.crewai_agent.CrewAIAgent") as MockCrewAIAgent:
            mock_instance = AsyncMock()
            mock_instance.name = "crewai-proposal"
            mock_instance.generate = AsyncMock(
                return_value="CrewAI proposal: Implement Redis-based caching layer"
            )
            mock_instance.kickoff = AsyncMock(
                return_value={
                    "success": True,
                    "output": "Detailed proposal for caching system",
                    "agent": "crewai-proposal",
                    "tools_used": ["search"],
                    "process": "sequential",
                    "execution_time": 2.5,
                }
            )
            MockCrewAIAgent.return_value = mock_instance

            # Test generate method
            result = await mock_instance.generate("Design a caching system")
            assert "CrewAI proposal" in result
            mock_instance.generate.assert_called_once_with("Design a caching system")

            # Test kickoff method
            kickoff_result = await mock_instance.kickoff(
                "Design a distributed cache", tools=["search"]
            )
            assert kickoff_result["success"] is True
            assert kickoff_result["process"] == "sequential"

    @pytest.mark.asyncio
    async def test_crewai_process_mode_sequential(
        self,
        gateway_server_context: dict,
        mock_external_server: MockExternalFrameworkServer,
    ) -> None:
        """Test CrewAI sequential process mode works correctly.

        Verifies that when configured for sequential mode, the CrewAI
        agent executes tasks in order, one at a time.
        """
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent, CrewAIConfig

        # Configure for sequential mode
        config = CrewAIConfig(
            base_url="http://localhost:8000",
            process="sequential",
            allowed_tools=["search"],
            max_rpm=100,  # High limit for testing
        )

        # Mock the HTTP session
        with patch(
            "aragora.agents.api_agents.crewai_agent.create_client_session"
        ) as mock_session_factory:
            mock_session = MagicMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={
                    "status": "success",
                    "result": "Sequential execution complete",
                    "process_mode": "sequential",
                }
            )
            mock_response.text = AsyncMock(return_value="")

            # Setup context manager for session.post
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = MagicMock(return_value=mock_cm)

            # Setup context manager for session itself
            session_cm = AsyncMock()
            session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session_factory.return_value = session_cm

            agent = CrewAIAgent(
                name="crewai-sequential",
                config=config,
                api_key="test-key",
            )

            # Verify config
            assert agent.crewai_config.process == "sequential"
            assert agent.crewai_config.validate_process() is True

            # Execute kickoff
            result = await agent.kickoff(
                task="Process documents in order",
                tools=["search"],
            )

            assert result["success"] is True
            assert result["process"] == "sequential"

    @pytest.mark.asyncio
    async def test_crewai_process_mode_hierarchical(
        self,
        gateway_server_context: dict,
        mock_external_server: MockExternalFrameworkServer,
    ) -> None:
        """Test CrewAI hierarchical process mode works correctly.

        Verifies that when configured for hierarchical mode, the CrewAI
        agent uses a manager to delegate tasks to worker agents.
        """
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent, CrewAIConfig

        # Configure for hierarchical mode
        config = CrewAIConfig(
            base_url="http://localhost:8000",
            process="hierarchical",
            allowed_tools=["search", "calculator"],
            max_rpm=100,
        )

        # Mock the HTTP session
        with patch(
            "aragora.agents.api_agents.crewai_agent.create_client_session"
        ) as mock_session_factory:
            mock_session = MagicMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={
                    "status": "success",
                    "result": "Hierarchical execution complete",
                    "process_mode": "hierarchical",
                    "manager_delegated": True,
                }
            )
            mock_response.text = AsyncMock(return_value="")

            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = MagicMock(return_value=mock_cm)

            session_cm = AsyncMock()
            session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session_factory.return_value = session_cm

            agent = CrewAIAgent(
                name="crewai-hierarchical",
                config=config,
                api_key="test-key",
            )

            # Verify config
            assert agent.crewai_config.process == "hierarchical"
            assert agent.crewai_config.validate_process() is True

            # Execute kickoff
            result = await agent.kickoff(
                task="Coordinate team to complete project",
                tools=["search", "calculator"],
            )

            assert result["success"] is True
            assert result["process"] == "hierarchical"

    @pytest.mark.asyncio
    async def test_crewai_tool_whitelist(
        self,
        gateway_server_context: dict,
        mock_external_server: MockExternalFrameworkServer,
    ) -> None:
        """Test only whitelisted tools are allowed in CrewAI.

        Verifies that the tool whitelist is enforced, blocking
        tools that are not explicitly allowed.
        """
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent, CrewAIConfig

        # Configure with limited allowed tools
        config = CrewAIConfig(
            base_url="http://localhost:8000",
            process="sequential",
            allowed_tools=["search", "calculator"],  # Only these are allowed
            max_rpm=100,
        )

        agent = CrewAIAgent(
            name="crewai-whitelist",
            config=config,
            api_key="test-key",
        )

        # Test tool filtering
        # Allowed tools should pass through
        assert agent._is_tool_allowed("search") is True
        assert agent._is_tool_allowed("calculator") is True

        # Non-whitelisted tools should be blocked
        assert agent._is_tool_allowed("shell") is False
        assert agent._is_tool_allowed("file_delete") is False
        assert agent._is_tool_allowed("code_execution") is False

        # Test filtering a list of tools
        requested_tools = ["search", "shell", "calculator", "file_delete"]
        filtered_tools = agent._filter_tools(requested_tools)

        assert "search" in filtered_tools
        assert "calculator" in filtered_tools
        assert "shell" not in filtered_tools
        assert "file_delete" not in filtered_tools
        assert len(filtered_tools) == 2

        # Test empty whitelist blocks all tools
        config_no_tools = CrewAIConfig(
            base_url="http://localhost:8000",
            allowed_tools=[],  # Empty - all blocked
        )
        agent_no_tools = CrewAIAgent(
            name="crewai-no-tools",
            config=config_no_tools,
            api_key="test-key",
        )

        assert agent_no_tools._is_tool_allowed("search") is False
        assert agent_no_tools._filter_tools(["search", "calc"]) == []

    @pytest.mark.asyncio
    async def test_crewai_rate_limiting(
        self,
        gateway_server_context: dict,
        mock_external_server: MockExternalFrameworkServer,
    ) -> None:
        """Test rate limits are applied to CrewAI calls.

        Verifies that the rate limiter correctly enforces the maximum
        requests per minute, raising errors when exceeded.
        """
        from aragora.agents.api_agents.common import AgentRateLimitError
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent, CrewAIConfig

        # Configure with very low rate limit for testing
        config = CrewAIConfig(
            base_url="http://localhost:8000",
            process="sequential",
            allowed_tools=["search"],
            max_rpm=3,  # Only 3 requests per minute
        )

        agent = CrewAIAgent(
            name="crewai-ratelimit",
            config=config,
            api_key="test-key",
        )

        # Simulate requests to fill the rate limit window
        # Record 3 requests (at the limit)
        for _ in range(3):
            agent._record_request()

        # Verify rate limit is now exceeded
        assert agent._check_rate_limit() is False

        # Verify wait time is calculated
        wait_time = agent._get_rate_limit_wait()
        assert wait_time > 0
        assert wait_time <= 60.0  # Should be within the 1-minute window

        # Mock the HTTP session for generate call
        with patch(
            "aragora.agents.api_agents.crewai_agent.create_client_session"
        ) as mock_session_factory:
            mock_session = MagicMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"response": "test"})

            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = MagicMock(return_value=mock_cm)

            session_cm = AsyncMock()
            session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session_factory.return_value = session_cm

            # Attempt to generate when rate limited - should raise error
            with pytest.raises(AgentRateLimitError) as exc_info:
                await agent.generate("Test prompt")

            assert "Rate limit exceeded" in str(exc_info.value)
            assert exc_info.value.agent_name == "crewai-ratelimit"
            assert exc_info.value.retry_after is not None

        # Reset rate limit and verify requests work again
        agent.reset_rate_limit()
        assert agent._check_rate_limit() is True
        assert agent._get_rate_limit_wait() == 0.0
