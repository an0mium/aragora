"""
Tests for OpenClaw Agent.

Tests cover:
- Initialization and configuration
- OpenClawConfig dataclass
- Capability filtering (shell, files, browser)
- Channel restrictions for messaging
- generate() method with capability prefixes
- execute_task() method
- send_message() method
- Health check (is_available)
- Error handling
- Agent registry integration
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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
def mock_openai_response():
    """Standard OpenAI-compatible API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "OpenClaw response: Task completed successfully.",
                },
            }
        ],
    }


class TestOpenClawConfig:
    """Tests for OpenClawConfig dataclass."""

    def test_config_defaults(self):
        """Should have secure default values."""
        from aragora.agents.api_agents.openclaw import OpenClawConfig

        config = OpenClawConfig(base_url="http://localhost:3000")
        config.__post_init__()

        assert config.gateway_mode == "local"
        assert config.enable_shell is False
        assert config.enable_file_ops is False
        assert config.enable_browser is False
        assert config.allowed_channels == []
        assert config.task_timeout == 300
        assert config.audit_all_requests is True

    def test_config_post_init_sets_endpoints(self):
        """Should set OpenClaw-specific endpoints."""
        from aragora.agents.api_agents.openclaw import OpenClawConfig

        config = OpenClawConfig(base_url="")
        config.__post_init__()

        assert config.generate_endpoint == "/api/chat"
        assert config.health_endpoint == "/health"

    def test_config_post_init_reads_env_url(self, monkeypatch):
        """Should read OPENCLAW_URL from environment."""
        from aragora.agents.api_agents.openclaw import OpenClawConfig

        monkeypatch.setenv("OPENCLAW_URL", "http://openclaw.example.com")

        config = OpenClawConfig(base_url="")
        config.__post_init__()

        assert config.base_url == "http://openclaw.example.com"

    def test_config_preserves_explicit_base_url(self, monkeypatch):
        """Should prefer explicit base_url over environment."""
        from aragora.agents.api_agents.openclaw import OpenClawConfig

        monkeypatch.setenv("OPENCLAW_URL", "http://from-env.com")

        config = OpenClawConfig(base_url="http://explicit.com")
        config.__post_init__()

        assert config.base_url == "http://explicit.com"

    def test_config_custom_values(self):
        """Should accept custom configuration values."""
        from aragora.agents.api_agents.openclaw import OpenClawConfig

        config = OpenClawConfig(
            base_url="https://openclaw.enterprise.com",
            gateway_mode="cloud",
            enable_shell=True,
            enable_file_ops=True,
            enable_browser=True,
            allowed_channels=["slack", "teams"],
            task_timeout=600,
            audit_all_requests=False,
        )

        assert config.base_url == "https://openclaw.enterprise.com"
        assert config.gateway_mode == "cloud"
        assert config.enable_shell is True
        assert config.enable_file_ops is True
        assert config.enable_browser is True
        assert config.allowed_channels == ["slack", "teams"]
        assert config.task_timeout == 600
        assert config.audit_all_requests is False


class TestOpenClawAgentInitialization:
    """Tests for agent initialization."""

    def test_init_with_defaults(self, monkeypatch):
        """Should initialize with default values."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent

        # Clear env vars to test defaults
        monkeypatch.delenv("OPENCLAW_URL", raising=False)
        monkeypatch.delenv("OPENCLAW_API_KEY", raising=False)

        agent = OpenClawAgent()

        assert agent.name == "openclaw"
        assert agent.model == "openclaw"
        assert agent.agent_type == "openclaw"
        assert agent.openclaw_config is not None
        assert agent.openclaw_config.enable_shell is False
        assert agent.openclaw_config.enable_file_ops is False

    def test_init_with_custom_config(self):
        """Should initialize with custom configuration."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent, OpenClawConfig

        config = OpenClawConfig(
            base_url="https://custom.openclaw.com",
            enable_shell=True,
            allowed_channels=["telegram"],
        )
        config.__post_init__()

        agent = OpenClawAgent(
            name="custom-openclaw",
            config=config,
        )

        assert agent.name == "custom-openclaw"
        assert agent.base_url == "https://custom.openclaw.com"
        assert agent.openclaw_config.enable_shell is True
        assert agent.openclaw_config.allowed_channels == ["telegram"]

    def test_init_with_api_key(self):
        """Should accept API key."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent

        agent = OpenClawAgent(api_key="test-openclaw-key")

        assert agent.api_key == "test-openclaw-key"

    def test_init_reads_api_key_from_env(self, monkeypatch):
        """Should read API key from environment variable."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent

        monkeypatch.setenv("OPENCLAW_API_KEY", "env-openclaw-key")

        agent = OpenClawAgent()

        assert agent.api_key == "env-openclaw-key"

    def test_agent_registry_registration(self):
        """Should be registered in agent registry."""
        from aragora.agents.registry import AgentRegistry

        # Import to trigger registration
        from aragora.agents.api_agents.openclaw import OpenClawAgent  # noqa: F401

        spec = AgentRegistry.get_spec("openclaw")

        assert spec is not None
        assert spec.default_model == "openclaw"
        assert spec.agent_type == "API"
        assert spec.accepts_api_key is True


class TestOpenClawCapabilities:
    """Tests for capability filtering."""

    def test_get_enabled_capabilities_none(self):
        """Should return empty list when no capabilities enabled."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent

        agent = OpenClawAgent()

        capabilities = agent._get_enabled_capabilities()

        assert capabilities == []

    def test_get_enabled_capabilities_all(self):
        """Should return all enabled capabilities."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent, OpenClawConfig

        config = OpenClawConfig(
            base_url="http://localhost:3000",
            enable_shell=True,
            enable_file_ops=True,
            enable_browser=True,
        )
        agent = OpenClawAgent(config=config)

        capabilities = agent._get_enabled_capabilities()

        assert "shell" in capabilities
        assert "files" in capabilities
        assert "browser" in capabilities

    def test_get_enabled_capabilities_partial(self):
        """Should return only enabled capabilities."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent, OpenClawConfig

        config = OpenClawConfig(
            base_url="http://localhost:3000",
            enable_shell=True,
            enable_file_ops=False,
            enable_browser=True,
        )
        agent = OpenClawAgent(config=config)

        capabilities = agent._get_enabled_capabilities()

        assert "shell" in capabilities
        assert "files" not in capabilities
        assert "browser" in capabilities

    def test_build_capability_prefix_none_enabled(self):
        """Should indicate no dangerous capabilities allowed."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent

        agent = OpenClawAgent()

        prefix = agent._build_capability_prefix()

        assert "No dangerous capabilities allowed" in prefix
        assert "disabled" in prefix

    def test_build_capability_prefix_some_enabled(self):
        """Should list enabled capabilities."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent, OpenClawConfig

        config = OpenClawConfig(
            base_url="http://localhost:3000",
            enable_shell=True,
            enable_browser=True,
        )
        agent = OpenClawAgent(config=config)

        prefix = agent._build_capability_prefix()

        assert "Allowed capabilities:" in prefix
        assert "shell" in prefix
        assert "browser" in prefix
        assert "files" not in prefix

    def test_get_capability_status(self):
        """Should return full capability status."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent, OpenClawConfig

        config = OpenClawConfig(
            base_url="http://localhost:3000",
            gateway_mode="cloud",
            enable_shell=True,
            enable_file_ops=False,
            enable_browser=True,
            allowed_channels=["slack"],
            audit_all_requests=True,
        )
        agent = OpenClawAgent(config=config)

        status = agent.get_capability_status()

        assert status["gateway_mode"] == "cloud"
        assert status["shell_enabled"] is True
        assert status["file_ops_enabled"] is False
        assert status["browser_enabled"] is True
        assert status["allowed_channels"] == ["slack"]
        assert status["audit_enabled"] is True


class TestOpenClawGenerate:
    """Tests for generate method."""

    @pytest.mark.asyncio
    async def test_generate_adds_capability_prefix(self, mock_openai_response):
        """Should add capability prefix to prompts."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = OpenClawAgent()

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
            await agent.generate("List files in /tmp")

        assert captured_prompt is not None
        assert "No dangerous capabilities allowed" in captured_prompt
        assert "List files in /tmp" in captured_prompt

    @pytest.mark.asyncio
    async def test_generate_with_capabilities_enabled(self, mock_openai_response):
        """Should show enabled capabilities in prefix."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent, OpenClawConfig
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        config = OpenClawConfig(
            base_url="http://localhost:3000",
            enable_shell=True,
        )
        agent = OpenClawAgent(config=config)

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
            await agent.generate("Run ls -la")

        assert "Allowed capabilities: shell" in captured_prompt

    @pytest.mark.asyncio
    async def test_generate_logs_when_audit_enabled(self, mock_openai_response):
        """Should log requests when audit is enabled."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent

        agent = OpenClawAgent()

        with patch.object(
            agent.__class__.__bases__[0],
            "generate",
            AsyncMock(return_value="Response"),
        ):
            with patch("aragora.agents.api_agents.openclaw.logger") as mock_logger:
                await agent.generate("Test prompt")

                mock_logger.info.assert_called()


class TestOpenClawExecuteTask:
    """Tests for execute_task method."""

    @pytest.mark.asyncio
    async def test_execute_task_success(self):
        """Should execute task and return structured result."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent

        agent = OpenClawAgent()

        with patch.object(agent, "generate", AsyncMock(return_value="Task completed")):
            result = await agent.execute_task("Check system status")

        assert result["success"] is True
        assert result["output"] == "Task completed"
        assert result["agent"] == "openclaw"

    @pytest.mark.asyncio
    async def test_execute_task_with_capabilities(self):
        """Should return capabilities used."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent, OpenClawConfig

        config = OpenClawConfig(
            base_url="http://localhost:3000",
            enable_shell=True,
            enable_file_ops=True,
        )
        agent = OpenClawAgent(config=config)

        with patch.object(agent, "generate", AsyncMock(return_value="Done")):
            result = await agent.execute_task("Run command")

        assert "shell" in result["capabilities_used"]
        assert "files" in result["capabilities_used"]

    @pytest.mark.asyncio
    async def test_execute_task_missing_required_capabilities(self):
        """Should fail when required capabilities not enabled."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent

        agent = OpenClawAgent()  # No capabilities enabled

        result = await agent.execute_task(
            "Run dangerous command",
            require_capabilities=["shell"],
        )

        assert result["success"] is False
        assert "shell" in result["output"]
        assert result["capabilities_used"] == []

    @pytest.mark.asyncio
    async def test_execute_task_handles_exception(self):
        """Should handle exceptions gracefully."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent

        agent = OpenClawAgent()

        with patch.object(agent, "generate", AsyncMock(side_effect=RuntimeError("Network error"))):
            result = await agent.execute_task("Test task")

        assert result["success"] is False
        assert "Network error" in result["output"]


class TestOpenClawSendMessage:
    """Tests for send_message method."""

    @pytest.mark.asyncio
    async def test_send_message_to_allowed_channel(self):
        """Should send message to allowed channel."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent, OpenClawConfig

        config = OpenClawConfig(
            base_url="http://localhost:3000",
            allowed_channels=["slack", "teams"],
        )
        agent = OpenClawAgent(config=config)

        with patch.object(agent, "generate", AsyncMock(return_value="Message sent")):
            result = await agent.send_message("slack", "Hello world")

        assert result["success"] is True
        assert result["channel"] == "slack"

    @pytest.mark.asyncio
    async def test_send_message_to_disallowed_channel(self):
        """Should reject message to disallowed channel."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent, OpenClawConfig

        config = OpenClawConfig(
            base_url="http://localhost:3000",
            allowed_channels=["slack"],
        )
        agent = OpenClawAgent(config=config)

        result = await agent.send_message("telegram", "Hello world")

        assert result["success"] is False
        assert "not allowed" in result["error"]
        assert result["channel"] == "telegram"

    @pytest.mark.asyncio
    async def test_send_message_no_channels_allowed(self):
        """Should reject all messages when no channels allowed."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent

        agent = OpenClawAgent()  # Empty allowed_channels by default

        result = await agent.send_message("slack", "Hello")

        assert result["success"] is False
        assert "No messaging channels are allowed" in result["error"]

    @pytest.mark.asyncio
    async def test_send_message_with_recipient(self):
        """Should include recipient in prompt."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent, OpenClawConfig

        config = OpenClawConfig(
            base_url="http://localhost:3000",
            allowed_channels=["slack"],
        )
        agent = OpenClawAgent(config=config)

        captured_prompt = None

        async def capture_generate(prompt, context=None, **kwargs):
            nonlocal captured_prompt
            captured_prompt = prompt
            return "Sent"

        with patch.object(agent, "generate", capture_generate):
            await agent.send_message("slack", "Hello", recipient="@user123")

        assert "recipient @user123" in captured_prompt

    @pytest.mark.asyncio
    async def test_send_message_case_insensitive_channel(self):
        """Should match channels case-insensitively."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent, OpenClawConfig

        config = OpenClawConfig(
            base_url="http://localhost:3000",
            allowed_channels=["Slack"],  # Capital S
        )
        agent = OpenClawAgent(config=config)

        with patch.object(agent, "generate", AsyncMock(return_value="Sent")):
            result = await agent.send_message("slack", "Hello")  # lowercase

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_send_message_handles_exception(self):
        """Should handle exceptions gracefully."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent, OpenClawConfig

        config = OpenClawConfig(
            base_url="http://localhost:3000",
            allowed_channels=["slack"],
        )
        agent = OpenClawAgent(config=config)

        with patch.object(agent, "generate", AsyncMock(side_effect=RuntimeError("API error"))):
            result = await agent.send_message("slack", "Hello")

        assert result["success"] is False
        assert "API error" in result["error"]


class TestOpenClawIsAvailable:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_is_available_calls_parent(self):
        """Should delegate to parent class."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent

        agent = OpenClawAgent()

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
        from aragora.agents.api_agents.openclaw import OpenClawAgent

        agent = OpenClawAgent()

        with patch.object(
            agent.__class__.__bases__[0],
            "is_available",
            AsyncMock(return_value=False),
        ):
            result = await agent.is_available()

        assert result is False


class TestOpenClawInheritedBehavior:
    """Tests for behavior inherited from ExternalFrameworkAgent."""

    def test_inherits_from_external_framework_agent(self):
        """Should inherit from ExternalFrameworkAgent."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent
        from aragora.agents.api_agents.openclaw import OpenClawAgent

        agent = OpenClawAgent()

        assert isinstance(agent, ExternalFrameworkAgent)

    def test_has_circuit_breaker_support(self):
        """Should support circuit breaker configuration."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent

        agent = OpenClawAgent(
            enable_circuit_breaker=True,
            circuit_breaker_threshold=5,
        )

        assert agent._circuit_breaker is not None

    def test_supports_response_sanitization(self):
        """Should inherit response sanitization."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent

        agent = OpenClawAgent()

        # Test sanitization method exists and works
        result = agent._sanitize_response("Test\x00with\x00nulls")
        assert "\x00" not in result

    def test_supports_generation_params(self):
        """Should support generation parameters."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent

        agent = OpenClawAgent()
        agent.set_generation_params(temperature=0.7, top_p=0.9)

        assert agent.temperature == 0.7
        assert agent.top_p == 0.9


class TestOpenClawModuleExports:
    """Tests for module exports."""

    def test_exports_agent_class(self):
        """Should export OpenClawAgent."""
        from aragora.agents.api_agents.openclaw import OpenClawAgent

        assert OpenClawAgent is not None

    def test_exports_config_class(self):
        """Should export OpenClawConfig."""
        from aragora.agents.api_agents.openclaw import OpenClawConfig

        assert OpenClawConfig is not None

    def test_all_exports(self):
        """Should have correct __all__ exports."""
        from aragora.agents.api_agents import openclaw

        assert "OpenClawAgent" in openclaw.__all__
        assert "OpenClawConfig" in openclaw.__all__
