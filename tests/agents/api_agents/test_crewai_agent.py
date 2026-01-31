"""
Tests for CrewAI Agent.

Tests cover:
- Initialization and configuration
- CrewAIConfig dataclass
- Tool whitelist filtering (allowed_tools)
- Rate limiting via max_rpm
- generate() method with tool prefixes
- kickoff() method for crew execution
- get_crew_status() method for monitoring
- Process mode tests (sequential vs hierarchical)
- Health check (is_available)
- Error handling
- Agent registry integration
"""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from aragora.agents.api_agents.common import (
    AgentAPIError,
    AgentConnectionError,
    AgentRateLimitError,
)


@pytest.fixture
def sample_context():
    """Sample message context for testing."""
    from aragora.core import Message

    return [
        Message(
            agent="agent1",
            content="Previous context message",
            role="proposer",
            round=1,
        ),
    ]


@pytest.fixture
def mock_crew_response():
    """Standard CrewAI API response."""
    return {
        "output": "Crew completed: Task was executed successfully by the team.",
        "status": "completed",
        "crew_id": "crew-123",
    }


@pytest.fixture
def mock_openai_response():
    """Standard OpenAI-compatible API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "CrewAI response: Task completed successfully.",
                },
            }
        ],
    }


class TestCrewAIConfig:
    """Tests for CrewAIConfig dataclass."""

    def test_config_defaults(self):
        """Should have secure default values."""
        from aragora.agents.api_agents.crewai_agent import CrewAIConfig

        config = CrewAIConfig(base_url="http://localhost:8000")
        config.__post_init__()

        assert config.process == "sequential"
        assert config.verbose is False
        assert config.memory is True
        assert config.max_rpm == 10
        assert config.allowed_tools == []
        assert config.crew_timeout == 600
        assert config.audit_all_requests is True

    def test_config_post_init_sets_endpoints(self):
        """Should set CrewAI-specific endpoints."""
        from aragora.agents.api_agents.crewai_agent import CrewAIConfig

        config = CrewAIConfig(base_url="")
        config.__post_init__()

        assert config.generate_endpoint == "/v1/crew/kickoff"
        assert config.health_endpoint == "/health"

    def test_config_post_init_reads_env_url(self, monkeypatch):
        """Should read CREWAI_URL from environment."""
        from aragora.agents.api_agents.crewai_agent import CrewAIConfig

        monkeypatch.setenv("CREWAI_URL", "http://crewai.example.com")

        config = CrewAIConfig(base_url="")
        config.__post_init__()

        assert config.base_url == "http://crewai.example.com"

    def test_config_preserves_explicit_base_url(self, monkeypatch):
        """Should prefer explicit base_url over environment."""
        from aragora.agents.api_agents.crewai_agent import CrewAIConfig

        monkeypatch.setenv("CREWAI_URL", "http://from-env.com")

        config = CrewAIConfig(base_url="http://explicit.com")
        config.__post_init__()

        assert config.base_url == "http://explicit.com"

    def test_config_custom_values(self):
        """Should accept custom configuration values."""
        from aragora.agents.api_agents.crewai_agent import CrewAIConfig

        config = CrewAIConfig(
            base_url="https://crewai.enterprise.com",
            process="hierarchical",
            verbose=True,
            memory=False,
            max_rpm=5,
            allowed_tools=["search", "calculator", "code_interpreter"],
            crew_timeout=1200,
            audit_all_requests=False,
        )

        assert config.base_url == "https://crewai.enterprise.com"
        assert config.process == "hierarchical"
        assert config.verbose is True
        assert config.memory is False
        assert config.max_rpm == 5
        assert config.allowed_tools == ["search", "calculator", "code_interpreter"]
        assert config.crew_timeout == 1200
        assert config.audit_all_requests is False

    def test_config_validate_process_sequential(self):
        """Should validate sequential process mode."""
        from aragora.agents.api_agents.crewai_agent import CrewAIConfig

        config = CrewAIConfig(base_url="http://localhost", process="sequential")
        assert config.validate_process() is True

    def test_config_validate_process_hierarchical(self):
        """Should validate hierarchical process mode."""
        from aragora.agents.api_agents.crewai_agent import CrewAIConfig

        config = CrewAIConfig(base_url="http://localhost", process="hierarchical")
        assert config.validate_process() is True

    def test_config_validate_process_invalid(self):
        """Should reject invalid process modes."""
        from aragora.agents.api_agents.crewai_agent import CrewAIConfig

        config = CrewAIConfig(base_url="http://localhost", process="invalid")
        assert config.validate_process() is False


class TestCrewAIAgentInitialization:
    """Tests for agent initialization."""

    def test_init_with_defaults(self, monkeypatch):
        """Should initialize with default values."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent

        # Clear env vars to test defaults
        monkeypatch.delenv("CREWAI_URL", raising=False)
        monkeypatch.delenv("CREWAI_API_KEY", raising=False)

        agent = CrewAIAgent()

        assert agent.name == "crewai"
        assert agent.model == "crewai"
        assert agent.agent_type == "crewai"
        assert agent.crewai_config is not None
        assert agent.crewai_config.process == "sequential"
        assert agent.crewai_config.allowed_tools == []

    def test_init_with_custom_config(self):
        """Should initialize with custom configuration."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent, CrewAIConfig

        config = CrewAIConfig(
            base_url="https://custom.crewai.com",
            process="hierarchical",
            allowed_tools=["search"],
            max_rpm=5,
        )
        config.__post_init__()

        agent = CrewAIAgent(
            name="custom-crewai",
            config=config,
        )

        assert agent.name == "custom-crewai"
        assert agent.base_url == "https://custom.crewai.com"
        assert agent.crewai_config.process == "hierarchical"
        assert agent.crewai_config.allowed_tools == ["search"]
        assert agent.crewai_config.max_rpm == 5

    def test_init_with_api_key(self):
        """Should accept API key."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent

        agent = CrewAIAgent(api_key="test-crewai-key")

        assert agent.api_key == "test-crewai-key"

    def test_init_reads_api_key_from_env(self, monkeypatch):
        """Should read API key from environment variable."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent

        monkeypatch.setenv("CREWAI_API_KEY", "env-crewai-key")

        agent = CrewAIAgent()

        assert agent.api_key == "env-crewai-key"

    def test_init_invalid_process_raises(self):
        """Should raise ValueError for invalid process mode."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent, CrewAIConfig

        config = CrewAIConfig(
            base_url="http://localhost",
            process="invalid_mode",
        )

        with pytest.raises(ValueError) as exc_info:
            CrewAIAgent(config=config)

        assert "Invalid process mode" in str(exc_info.value)

    def test_agent_registry_registration(self):
        """Should be registered in agent registry."""
        from aragora.agents.registry import AgentRegistry

        # Import to trigger registration
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent  # noqa: F401

        spec = AgentRegistry.get_spec("crewai")

        assert spec is not None
        assert spec.default_model == "crewai"
        assert spec.agent_type == "API"
        assert spec.accepts_api_key is True


class TestCrewAIToolFiltering:
    """Tests for tool whitelist filtering."""

    def test_get_allowed_tools_none(self):
        """Should return empty list when no tools configured."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent

        agent = CrewAIAgent()

        tools = agent._get_allowed_tools()

        assert tools == []

    def test_get_allowed_tools_configured(self):
        """Should return configured tools."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent, CrewAIConfig

        config = CrewAIConfig(
            base_url="http://localhost:8000",
            allowed_tools=["search", "calculator", "code_interpreter"],
        )
        agent = CrewAIAgent(config=config)

        tools = agent._get_allowed_tools()

        assert "search" in tools
        assert "calculator" in tools
        assert "code_interpreter" in tools

    def test_is_tool_allowed_true(self):
        """Should return True for allowed tools."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent, CrewAIConfig

        config = CrewAIConfig(
            base_url="http://localhost:8000",
            allowed_tools=["search", "calculator"],
        )
        agent = CrewAIAgent(config=config)

        assert agent._is_tool_allowed("search") is True
        assert agent._is_tool_allowed("calculator") is True

    def test_is_tool_allowed_false(self):
        """Should return False for disallowed tools."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent, CrewAIConfig

        config = CrewAIConfig(
            base_url="http://localhost:8000",
            allowed_tools=["search"],
        )
        agent = CrewAIAgent(config=config)

        assert agent._is_tool_allowed("shell") is False
        assert agent._is_tool_allowed("file_write") is False

    def test_is_tool_allowed_case_insensitive(self):
        """Should match tools case-insensitively."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent, CrewAIConfig

        config = CrewAIConfig(
            base_url="http://localhost:8000",
            allowed_tools=["Search", "Calculator"],
        )
        agent = CrewAIAgent(config=config)

        assert agent._is_tool_allowed("search") is True
        assert agent._is_tool_allowed("CALCULATOR") is True

    def test_is_tool_allowed_empty_list(self):
        """Should return False when no tools allowed."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent

        agent = CrewAIAgent()  # Empty allowed_tools

        assert agent._is_tool_allowed("search") is False
        assert agent._is_tool_allowed("calculator") is False

    def test_filter_tools_returns_only_allowed(self):
        """Should filter to only allowed tools."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent, CrewAIConfig

        config = CrewAIConfig(
            base_url="http://localhost:8000",
            allowed_tools=["search", "calculator"],
        )
        agent = CrewAIAgent(config=config)

        requested = ["search", "shell", "calculator", "file_write"]
        filtered = agent._filter_tools(requested)

        assert "search" in filtered
        assert "calculator" in filtered
        assert "shell" not in filtered
        assert "file_write" not in filtered

    def test_filter_tools_empty_when_no_allowed(self):
        """Should return empty list when no tools allowed."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent

        agent = CrewAIAgent()

        requested = ["search", "calculator"]
        filtered = agent._filter_tools(requested)

        assert filtered == []

    def test_build_capability_prefix_none_allowed(self):
        """Should indicate no tools allowed."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent

        agent = CrewAIAgent()

        prefix = agent._build_capability_prefix()

        assert "No tools allowed" in prefix
        assert "disabled" in prefix

    def test_build_capability_prefix_tools_allowed(self):
        """Should list allowed tools."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent, CrewAIConfig

        config = CrewAIConfig(
            base_url="http://localhost:8000",
            allowed_tools=["search", "calculator"],
        )
        agent = CrewAIAgent(config=config)

        prefix = agent._build_capability_prefix()

        assert "Allowed tools:" in prefix
        assert "search" in prefix
        assert "calculator" in prefix


class TestCrewAIRateLimiting:
    """Tests for rate limiting via max_rpm."""

    def test_check_rate_limit_under_limit(self):
        """Should return True when under rate limit."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent, CrewAIConfig

        config = CrewAIConfig(base_url="http://localhost:8000", max_rpm=10)
        agent = CrewAIAgent(config=config)

        # Make a few requests
        for _ in range(5):
            agent._record_request()

        assert agent._check_rate_limit() is True

    def test_check_rate_limit_at_limit(self):
        """Should return False when at rate limit."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent, CrewAIConfig

        config = CrewAIConfig(base_url="http://localhost:8000", max_rpm=5)
        agent = CrewAIAgent(config=config)

        # Fill up to the limit
        for _ in range(5):
            agent._record_request()

        assert agent._check_rate_limit() is False

    def test_check_rate_limit_clears_old_timestamps(self):
        """Should clear timestamps outside the 1-minute window."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent, CrewAIConfig

        config = CrewAIConfig(base_url="http://localhost:8000", max_rpm=5)
        agent = CrewAIAgent(config=config)

        # Add old timestamps (more than 60 seconds ago)
        old_time = time.time() - 120  # 2 minutes ago
        agent._request_timestamps = [old_time] * 5

        # Should be under limit because old timestamps are cleared
        assert agent._check_rate_limit() is True
        assert len(agent._request_timestamps) == 0

    def test_get_rate_limit_wait_no_wait(self):
        """Should return 0 when no wait needed."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent, CrewAIConfig

        config = CrewAIConfig(base_url="http://localhost:8000", max_rpm=10)
        agent = CrewAIAgent(config=config)

        assert agent._get_rate_limit_wait() == 0.0

    def test_get_rate_limit_wait_positive(self):
        """Should return positive wait time when at limit."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent, CrewAIConfig

        config = CrewAIConfig(base_url="http://localhost:8000", max_rpm=2)
        agent = CrewAIAgent(config=config)

        # Fill up the limit
        agent._record_request()
        agent._record_request()

        wait = agent._get_rate_limit_wait()
        assert wait > 0
        assert wait <= 60.0  # Should be within the window

    def test_reset_rate_limit(self):
        """Should clear rate limit counters."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent, CrewAIConfig

        config = CrewAIConfig(base_url="http://localhost:8000", max_rpm=5)
        agent = CrewAIAgent(config=config)

        for _ in range(5):
            agent._record_request()

        assert agent._check_rate_limit() is False

        agent.reset_rate_limit()

        assert agent._check_rate_limit() is True
        assert len(agent._request_timestamps) == 0


class TestCrewAIGenerate:
    """Tests for generate method."""

    @pytest.mark.asyncio
    async def test_generate_adds_tool_prefix(self, mock_openai_response):
        """Should add tool prefix to prompts."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = CrewAIAgent()

        captured_prompt = None

        async def mock_parent_generate(self, prompt, context=None, **kwargs):
            nonlocal captured_prompt
            captured_prompt = prompt
            return "Test response"

        with patch.object(
            ExternalFrameworkAgent,
            "generate",
            mock_parent_generate,
        ):
            await agent.generate("Execute task")

        assert captured_prompt is not None
        assert "No tools allowed" in captured_prompt
        assert "Execute task" in captured_prompt

    @pytest.mark.asyncio
    async def test_generate_with_tools_enabled(self, mock_openai_response):
        """Should show enabled tools in prefix."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent, CrewAIConfig
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        config = CrewAIConfig(
            base_url="http://localhost:8000",
            allowed_tools=["search", "calculator"],
        )
        agent = CrewAIAgent(config=config)

        captured_prompt = None

        async def mock_parent_generate(self, prompt, context=None, **kwargs):
            nonlocal captured_prompt
            captured_prompt = prompt
            return "Test response"

        with patch.object(
            ExternalFrameworkAgent,
            "generate",
            mock_parent_generate,
        ):
            await agent.generate("Search for information")

        assert "Allowed tools: search, calculator" in captured_prompt

    @pytest.mark.asyncio
    async def test_generate_rate_limited(self):
        """Should raise AgentRateLimitError when rate limited."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent, CrewAIConfig

        config = CrewAIConfig(base_url="http://localhost:8000", max_rpm=2)
        agent = CrewAIAgent(config=config)

        # Fill up the rate limit
        agent._record_request()
        agent._record_request()

        with pytest.raises(AgentRateLimitError) as exc_info:
            await agent.generate("Test prompt")

        assert "Rate limit exceeded" in str(exc_info.value)
        assert "2 rpm" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_logs_when_audit_enabled(self, mock_openai_response):
        """Should log requests when audit is enabled."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent

        agent = CrewAIAgent()

        with patch.object(
            agent.__class__.__bases__[0],
            "generate",
            AsyncMock(return_value="Response"),
        ):
            with patch("aragora.agents.api_agents.crewai_agent.logger") as mock_logger:
                await agent.generate("Test prompt")

                mock_logger.info.assert_called()


class TestCrewAIKickoff:
    """Tests for kickoff method."""

    @pytest.mark.asyncio
    async def test_kickoff_success(self, mock_crew_response):
        """Should execute crew and return structured result."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent

        agent = CrewAIAgent()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_crew_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.crewai_agent.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.kickoff("Research AI trends")

        assert result["success"] is True
        assert "completed" in result["output"].lower()
        assert result["agent"] == "crewai"
        assert result["process"] == "sequential"
        assert "execution_time" in result

    @pytest.mark.asyncio
    async def test_kickoff_with_inputs(self, mock_crew_response):
        """Should pass inputs to the crew."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent

        agent = CrewAIAgent()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_crew_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        captured_payload = {}

        def capture_post(url, **kwargs):
            captured_payload["json"] = kwargs.get("json")
            return mock_response

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=capture_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.crewai_agent.create_client_session",
            return_value=mock_session,
        ):
            await agent.kickoff(
                "Research topic",
                inputs={"topic": "AI", "depth": "detailed"},
            )

        assert "inputs" in captured_payload["json"]
        assert captured_payload["json"]["inputs"]["topic"] == "AI"

    @pytest.mark.asyncio
    async def test_kickoff_filters_tools(self, mock_crew_response):
        """Should filter tools to only allowed ones."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent, CrewAIConfig

        config = CrewAIConfig(
            base_url="http://localhost:8000",
            allowed_tools=["search", "calculator"],
        )
        agent = CrewAIAgent(config=config)

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_crew_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        captured_payload = {}

        def capture_post(url, **kwargs):
            captured_payload["json"] = kwargs.get("json")
            return mock_response

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=capture_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.crewai_agent.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.kickoff(
                "Research topic",
                tools=["search", "shell", "calculator", "file_write"],
            )

        # Should only include allowed tools
        assert "search" in captured_payload["json"]["tools"]
        assert "calculator" in captured_payload["json"]["tools"]
        assert "shell" not in captured_payload["json"]["tools"]
        assert "file_write" not in captured_payload["json"]["tools"]

        # Result should show filtered tools
        assert "search" in result["tools_used"]
        assert "calculator" in result["tools_used"]
        assert "shell" not in result["tools_used"]

    @pytest.mark.asyncio
    async def test_kickoff_rate_limited(self):
        """Should raise AgentRateLimitError when rate limited."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent, CrewAIConfig

        config = CrewAIConfig(base_url="http://localhost:8000", max_rpm=2)
        agent = CrewAIAgent(config=config)

        # Fill up the rate limit
        agent._record_request()
        agent._record_request()

        with pytest.raises(AgentRateLimitError) as exc_info:
            await agent.kickoff("Test task")

        assert "Rate limit exceeded" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_kickoff_handles_api_error(self):
        """Should raise AgentAPIError on API failure."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent

        agent = CrewAIAgent(name="test-api-error", enable_circuit_breaker=False)

        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal server error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.crewai_agent.create_client_session",
            return_value=mock_session,
        ):
            with pytest.raises(AgentAPIError) as exc_info:
                await agent.kickoff("Test task")

            assert "500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_kickoff_handles_connection_error(self):
        """Should raise AgentConnectionError on connection failure."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent

        agent = CrewAIAgent(name="test-conn-error", enable_circuit_breaker=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            side_effect=aiohttp.ClientConnectorError(MagicMock(), OSError("Connection refused"))
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.crewai_agent.create_client_session",
            return_value=mock_session,
        ):
            with pytest.raises(AgentConnectionError) as exc_info:
                await agent.kickoff("Test task")

            assert "Cannot connect" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_kickoff_handles_exception_gracefully(self):
        """Should handle unexpected exceptions gracefully."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent

        agent = CrewAIAgent(name="test-exception", enable_circuit_breaker=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=RuntimeError("Unexpected error"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.crewai_agent.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.kickoff("Test task")

        assert result["success"] is False
        assert "Unexpected error" in result["output"]


class TestCrewAIGetCrewStatus:
    """Tests for get_crew_status method."""

    @pytest.mark.asyncio
    async def test_get_crew_status_running(self):
        """Should return running status."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent

        agent = CrewAIAgent()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "status": "running",
                "progress": 0.5,
                "message": "Processing tasks",
                "crew_id": "crew-123",
            }
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.crewai_agent.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.get_crew_status("crew-123")

        assert result["status"] == "running"
        assert result["progress"] == 0.5
        assert result["crew_id"] == "crew-123"

    @pytest.mark.asyncio
    async def test_get_crew_status_completed(self):
        """Should return completed status."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent

        agent = CrewAIAgent()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "status": "completed",
                "progress": 1.0,
                "message": "Crew finished successfully",
                "crew_id": "crew-456",
            }
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.crewai_agent.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.get_crew_status("crew-456")

        assert result["status"] == "completed"
        assert result["progress"] == 1.0

    @pytest.mark.asyncio
    async def test_get_crew_status_not_found(self):
        """Should handle 404 for unknown crew."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent

        agent = CrewAIAgent()

        mock_response = MagicMock()
        mock_response.status = 404
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.crewai_agent.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.get_crew_status("unknown-crew")

        assert result["status"] == "unknown"
        assert "not found" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_get_crew_status_connection_error(self):
        """Should handle connection errors gracefully."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent

        agent = CrewAIAgent()

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=aiohttp.ClientError())
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.crewai_agent.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.get_crew_status("crew-123")

        assert result["status"] == "error"
        assert "Connection error" in result["message"]


class TestCrewAIProcessModes:
    """Tests for sequential vs hierarchical process modes."""

    @pytest.mark.asyncio
    async def test_sequential_process_mode(self, mock_crew_response):
        """Should use sequential process mode in request."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent, CrewAIConfig

        config = CrewAIConfig(
            base_url="http://localhost:8000",
            process="sequential",
        )
        agent = CrewAIAgent(config=config)

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_crew_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        captured_payload = {}

        def capture_post(url, **kwargs):
            captured_payload["json"] = kwargs.get("json")
            return mock_response

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=capture_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.crewai_agent.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.kickoff("Test task")

        assert captured_payload["json"]["process"] == "sequential"
        assert result["process"] == "sequential"

    @pytest.mark.asyncio
    async def test_hierarchical_process_mode(self, mock_crew_response):
        """Should use hierarchical process mode in request."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent, CrewAIConfig

        config = CrewAIConfig(
            base_url="http://localhost:8000",
            process="hierarchical",
        )
        agent = CrewAIAgent(config=config)

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_crew_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        captured_payload = {}

        def capture_post(url, **kwargs):
            captured_payload["json"] = kwargs.get("json")
            return mock_response

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=capture_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.crewai_agent.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.kickoff("Test task")

        assert captured_payload["json"]["process"] == "hierarchical"
        assert result["process"] == "hierarchical"


class TestCrewAIIsAvailable:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_is_available_calls_parent(self):
        """Should delegate to parent class."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent

        agent = CrewAIAgent()

        with patch.object(
            agent.__class__.__bases__[0],
            "is_available",
            AsyncMock(return_value=True),
        ):
            result = await agent.is_available()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_available_returns_false_on_error(self):
        """Should return False when server unavailable."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent

        agent = CrewAIAgent()

        with patch.object(
            agent.__class__.__bases__[0],
            "is_available",
            AsyncMock(return_value=False),
        ):
            result = await agent.is_available()

        assert result is False


class TestCrewAIConfigStatus:
    """Tests for get_config_status method."""

    def test_get_config_status(self):
        """Should return full configuration status."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent, CrewAIConfig

        config = CrewAIConfig(
            base_url="http://localhost:8000",
            process="hierarchical",
            verbose=True,
            memory=True,
            max_rpm=5,
            allowed_tools=["search", "calculator"],
            crew_timeout=1200,
            audit_all_requests=True,
        )
        agent = CrewAIAgent(config=config)

        status = agent.get_config_status()

        assert status["process"] == "hierarchical"
        assert status["verbose"] is True
        assert status["memory_enabled"] is True
        assert status["max_rpm"] == 5
        assert status["allowed_tools"] == ["search", "calculator"]
        assert status["crew_timeout"] == 1200
        assert status["audit_enabled"] is True
        assert "base_url" in status
        assert "current_rpm_usage" in status


class TestCrewAIInheritedBehavior:
    """Tests for behavior inherited from ExternalFrameworkAgent."""

    def test_inherits_from_external_framework_agent(self):
        """Should inherit from ExternalFrameworkAgent."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = CrewAIAgent()

        assert isinstance(agent, ExternalFrameworkAgent)

    def test_has_circuit_breaker_support(self):
        """Should support circuit breaker configuration."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent

        agent = CrewAIAgent(
            enable_circuit_breaker=True,
            circuit_breaker_threshold=5,
        )

        assert agent._circuit_breaker is not None

    def test_supports_response_sanitization(self):
        """Should inherit response sanitization."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent

        agent = CrewAIAgent()

        # Test sanitization method exists and works
        result = agent._sanitize_response("Test\x00with\x00nulls")
        assert "\x00" not in result

    def test_supports_generation_params(self):
        """Should support generation parameters."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent

        agent = CrewAIAgent()
        agent.set_generation_params(temperature=0.7, top_p=0.9)

        assert agent.temperature == 0.7
        assert agent.top_p == 0.9


class TestCrewAIModuleExports:
    """Tests for module exports."""

    def test_exports_agent_class(self):
        """Should export CrewAIAgent."""
        from aragora.agents.api_agents.crewai_agent import CrewAIAgent

        assert CrewAIAgent is not None

    def test_exports_config_class(self):
        """Should export CrewAIConfig."""
        from aragora.agents.api_agents.crewai_agent import CrewAIConfig

        assert CrewAIConfig is not None

    def test_all_exports(self):
        """Should have correct __all__ exports."""
        from aragora.agents.api_agents import crewai_agent

        assert "CrewAIAgent" in crewai_agent.__all__
        assert "CrewAIConfig" in crewai_agent.__all__
