"""
Tests for AutoGen Agent.

Tests cover:
- Initialization and configuration
- AutoGenConfig dataclass
- Mode tests (groupchat vs two_agent)
- Speaker selection tests
- Code execution permission tests
- Round limit tests
- Conversation management (initiate, continue, get, clear)
- generate() method with configuration prefixes
- Mock HTTP responses
- Error handling
- Agent registry integration
"""

from unittest.mock import AsyncMock, patch

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
                    "content": "AutoGen response: Discussion completed successfully.",
                },
            }
        ],
    }


class TestAutoGenConfig:
    """Tests for AutoGenConfig dataclass."""

    def test_config_defaults(self):
        """Should have secure default values."""
        from aragora.agents.api_agents.autogen_agent import AutoGenConfig

        config = AutoGenConfig(base_url="http://localhost:8000")
        config.__post_init__()

        assert config.mode == "groupchat"
        assert config.max_round == 10
        assert config.speaker_selection_method == "auto"
        assert config.allow_code_execution is False
        assert config.work_dir is None
        assert config.human_input_mode == "NEVER"
        assert config.conversation_timeout == 300
        assert config.audit_all_requests is True

    def test_config_post_init_sets_endpoints(self):
        """Should set AutoGen-specific endpoints."""
        from aragora.agents.api_agents.autogen_agent import AutoGenConfig

        config = AutoGenConfig(base_url="")
        config.__post_init__()

        assert config.generate_endpoint == "/api/chat"
        assert config.health_endpoint == "/health"

    def test_config_post_init_reads_env_url(self, monkeypatch):
        """Should read AUTOGEN_URL from environment."""
        from aragora.agents.api_agents.autogen_agent import AutoGenConfig

        monkeypatch.setenv("AUTOGEN_URL", "http://autogen.example.com")

        config = AutoGenConfig(base_url="")
        config.__post_init__()

        assert config.base_url == "http://autogen.example.com"

    def test_config_preserves_explicit_base_url(self, monkeypatch):
        """Should prefer explicit base_url over environment."""
        from aragora.agents.api_agents.autogen_agent import AutoGenConfig

        monkeypatch.setenv("AUTOGEN_URL", "http://from-env.com")

        config = AutoGenConfig(base_url="http://explicit.com")
        config.__post_init__()

        assert config.base_url == "http://explicit.com"

    def test_config_custom_values(self):
        """Should accept custom configuration values."""
        from aragora.agents.api_agents.autogen_agent import AutoGenConfig

        config = AutoGenConfig(
            base_url="https://autogen.enterprise.com",
            mode="two_agent",
            max_round=20,
            speaker_selection_method="round_robin",
            allow_code_execution=True,
            work_dir="/tmp/autogen_workspace",
            human_input_mode="TERMINATE",
            conversation_timeout=600,
            audit_all_requests=False,
        )

        assert config.base_url == "https://autogen.enterprise.com"
        assert config.mode == "two_agent"
        assert config.max_round == 20
        assert config.speaker_selection_method == "round_robin"
        assert config.allow_code_execution is True
        assert config.work_dir == "/tmp/autogen_workspace"
        assert config.human_input_mode == "TERMINATE"
        assert config.conversation_timeout == 600
        assert config.audit_all_requests is False


class TestAutoGenConfigValidation:
    """Tests for AutoGenConfig validation."""

    def test_validate_work_dir_disabled_code_execution(self):
        """Should pass validation when code execution is disabled."""
        from aragora.agents.api_agents.autogen_agent import AutoGenConfig

        config = AutoGenConfig(
            base_url="http://localhost:8000",
            allow_code_execution=False,
            work_dir=None,
        )

        assert config.validate_work_dir() is True

    def test_validate_work_dir_enabled_but_none(self):
        """Should fail validation when code execution enabled but no work_dir."""
        from aragora.agents.api_agents.autogen_agent import AutoGenConfig

        config = AutoGenConfig(
            base_url="http://localhost:8000",
            allow_code_execution=True,
            work_dir=None,
        )

        assert config.validate_work_dir() is False

    def test_validate_work_dir_relative_path(self):
        """Should fail validation for relative path."""
        from aragora.agents.api_agents.autogen_agent import AutoGenConfig

        config = AutoGenConfig(
            base_url="http://localhost:8000",
            allow_code_execution=True,
            work_dir="relative/path",
        )

        assert config.validate_work_dir() is False

    def test_validate_work_dir_valid_existing(self, tmp_path):
        """Should pass validation for existing absolute directory."""
        from aragora.agents.api_agents.autogen_agent import AutoGenConfig

        config = AutoGenConfig(
            base_url="http://localhost:8000",
            allow_code_execution=True,
            work_dir=str(tmp_path),
        )

        assert config.validate_work_dir() is True

    def test_validate_work_dir_creatable(self, tmp_path):
        """Should pass validation for creatable path."""
        from aragora.agents.api_agents.autogen_agent import AutoGenConfig

        # Path that doesn't exist but parent does
        new_dir = tmp_path / "new_workspace"

        config = AutoGenConfig(
            base_url="http://localhost:8000",
            allow_code_execution=True,
            work_dir=str(new_dir),
        )

        assert config.validate_work_dir() is True


class TestAutoGenAgentInitialization:
    """Tests for agent initialization."""

    def test_init_with_defaults(self, monkeypatch):
        """Should initialize with default values."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent

        # Clear env vars to test defaults
        monkeypatch.delenv("AUTOGEN_URL", raising=False)
        monkeypatch.delenv("AUTOGEN_API_KEY", raising=False)

        agent = AutoGenAgent()

        assert agent.name == "autogen"
        assert agent.model == "autogen"
        assert agent.agent_type == "autogen"
        assert agent.autogen_config is not None
        assert agent.autogen_config.allow_code_execution is False
        assert agent.autogen_config.mode == "groupchat"

    def test_init_with_custom_config(self):
        """Should initialize with custom configuration."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent, AutoGenConfig

        config = AutoGenConfig(
            base_url="https://custom.autogen.com",
            mode="two_agent",
            max_round=5,
        )
        config.__post_init__()

        agent = AutoGenAgent(
            name="custom-autogen",
            config=config,
        )

        assert agent.name == "custom-autogen"
        assert agent.base_url == "https://custom.autogen.com"
        assert agent.autogen_config.mode == "two_agent"
        assert agent.autogen_config.max_round == 5

    def test_init_with_api_key(self):
        """Should accept API key."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent

        agent = AutoGenAgent(api_key="test-autogen-key")

        assert agent.api_key == "test-autogen-key"

    def test_init_reads_api_key_from_env(self, monkeypatch):
        """Should read API key from environment variable."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent

        monkeypatch.setenv("AUTOGEN_API_KEY", "env-autogen-key")

        agent = AutoGenAgent()

        assert agent.api_key == "env-autogen-key"

    def test_init_rejects_invalid_work_dir(self):
        """Should reject invalid work_dir when code execution is enabled."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent, AutoGenConfig

        config = AutoGenConfig(
            base_url="http://localhost:8000",
            allow_code_execution=True,
            work_dir=None,  # Invalid - must be set
        )

        with pytest.raises(ValueError, match="work_dir is invalid"):
            AutoGenAgent(config=config)

    def test_init_accepts_valid_work_dir(self, tmp_path):
        """Should accept valid work_dir when code execution is enabled."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent, AutoGenConfig

        config = AutoGenConfig(
            base_url="http://localhost:8000",
            allow_code_execution=True,
            work_dir=str(tmp_path),
        )

        agent = AutoGenAgent(config=config)

        assert agent.autogen_config.work_dir == str(tmp_path)
        assert agent.autogen_config.allow_code_execution is True

    def test_agent_registry_registration(self):
        """Should be registered in agent registry."""
        from aragora.agents.registry import AgentRegistry

        # Import to trigger registration
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent  # noqa: F401

        spec = AgentRegistry.get_spec("autogen")

        assert spec is not None
        assert spec.default_model == "autogen"
        assert spec.agent_type == "API"
        assert spec.accepts_api_key is True


class TestAutoGenModes:
    """Tests for different conversation modes."""

    def test_groupchat_mode(self):
        """Should configure groupchat mode correctly."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent, AutoGenConfig

        config = AutoGenConfig(
            base_url="http://localhost:8000",
            mode="groupchat",
        )
        agent = AutoGenAgent(config=config)

        assert agent.autogen_config.mode == "groupchat"
        prefix = agent._build_autogen_prefix()
        assert "multi-agent groupchat" in prefix

    def test_two_agent_mode(self):
        """Should configure two_agent mode correctly."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent, AutoGenConfig

        config = AutoGenConfig(
            base_url="http://localhost:8000",
            mode="two_agent",
        )
        agent = AutoGenAgent(config=config)

        assert agent.autogen_config.mode == "two_agent"
        prefix = agent._build_autogen_prefix()
        assert "two-agent conversation" in prefix


class TestAutoGenSpeakerSelection:
    """Tests for speaker selection methods."""

    @pytest.mark.parametrize(
        "method",
        ["auto", "random", "round_robin", "manual"],
    )
    def test_speaker_selection_methods(self, method):
        """Should accept all valid speaker selection methods."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent, AutoGenConfig

        config = AutoGenConfig(
            base_url="http://localhost:8000",
            speaker_selection_method=method,
        )
        agent = AutoGenAgent(config=config)

        assert agent.autogen_config.speaker_selection_method == method


class TestAutoGenRoundLimits:
    """Tests for conversation round limits."""

    def test_default_max_round(self):
        """Should have default max_round of 10."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent

        agent = AutoGenAgent()

        assert agent.autogen_config.max_round == 10

    def test_custom_max_round(self):
        """Should accept custom max_round."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent, AutoGenConfig

        config = AutoGenConfig(
            base_url="http://localhost:8000",
            max_round=25,
        )
        agent = AutoGenAgent(config=config)

        assert agent.autogen_config.max_round == 25
        prefix = agent._build_autogen_prefix()
        assert "Max Rounds: 25" in prefix


class TestAutoGenCodeExecution:
    """Tests for code execution controls."""

    def test_code_execution_disabled_by_default(self):
        """Should have code execution disabled by default."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent

        agent = AutoGenAgent()

        assert agent.autogen_config.allow_code_execution is False
        prefix = agent._build_autogen_prefix()
        assert "Code Execution: disabled" in prefix

    def test_code_execution_enabled_with_work_dir(self, tmp_path):
        """Should enable code execution with valid work_dir."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent, AutoGenConfig

        config = AutoGenConfig(
            base_url="http://localhost:8000",
            allow_code_execution=True,
            work_dir=str(tmp_path),
        )
        agent = AutoGenAgent(config=config)

        assert agent.autogen_config.allow_code_execution is True
        prefix = agent._build_autogen_prefix()
        assert "Code Execution: enabled" in prefix
        assert str(tmp_path) in prefix


class TestAutoGenGenerate:
    """Tests for generate method."""

    @pytest.mark.asyncio
    async def test_generate_adds_config_prefix(self, mock_openai_response):
        """Should add configuration prefix to prompts."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = AutoGenAgent()

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
            await agent.generate("Discuss the solution")

        assert captured_prompt is not None
        assert "[AutoGen Mode:" in captured_prompt
        assert "Discuss the solution" in captured_prompt

    @pytest.mark.asyncio
    async def test_generate_logs_when_audit_enabled(self, mock_openai_response):
        """Should log requests when audit is enabled."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent

        agent = AutoGenAgent()

        with patch.object(
            agent.__class__.__bases__[0],
            "generate",
            AsyncMock(return_value="Response"),
        ):
            with patch("aragora.agents.api_agents.autogen_agent.logger") as mock_logger:
                await agent.generate("Test prompt")

                mock_logger.info.assert_called()


class TestAutoGenConversationManagement:
    """Tests for conversation management methods."""

    @pytest.mark.asyncio
    async def test_initiate_chat_success(self):
        """Should initiate chat and return conversation ID."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent

        agent = AutoGenAgent()

        with patch.object(agent, "generate", AsyncMock(return_value="Welcome to the chat")):
            result = await agent.initiate_chat("Start a discussion about AI")

        assert result["success"] is True
        assert "conversation_id" in result
        assert result["response"] == "Welcome to the chat"

    @pytest.mark.asyncio
    async def test_initiate_chat_stores_history(self):
        """Should store conversation history."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent

        agent = AutoGenAgent()

        with patch.object(agent, "generate", AsyncMock(return_value="Response")):
            result = await agent.initiate_chat("Hello")

        conv_id = result["conversation_id"]
        history = agent.get_conversation(conv_id)

        assert history is not None
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_initiate_chat_handles_error(self):
        """Should handle errors gracefully."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent

        agent = AutoGenAgent()

        with patch.object(agent, "generate", AsyncMock(side_effect=RuntimeError("Network error"))):
            result = await agent.initiate_chat("Hello")

        assert result["success"] is False
        assert "Network error" in result["response"]

    @pytest.mark.asyncio
    async def test_continue_chat_success(self):
        """Should continue existing conversation."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent

        agent = AutoGenAgent()

        with patch.object(agent, "generate", AsyncMock(return_value="Initial response")):
            init_result = await agent.initiate_chat("Hello")

        conv_id = init_result["conversation_id"]

        with patch.object(agent, "generate", AsyncMock(return_value="Follow-up response")):
            result = await agent.continue_chat(conv_id, "Tell me more")

        assert result["success"] is True
        assert result["response"] == "Follow-up response"

        history = agent.get_conversation(conv_id)
        assert len(history) == 4  # 2 from init, 2 from continue

    @pytest.mark.asyncio
    async def test_continue_chat_invalid_conversation(self):
        """Should fail for non-existent conversation."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent

        agent = AutoGenAgent()

        result = await agent.continue_chat("invalid-id", "Hello")

        assert result["success"] is False
        assert "not found" in result["response"]

    def test_get_conversation_exists(self):
        """Should return conversation history."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent

        agent = AutoGenAgent()
        agent._conversations["test-id"] = [{"role": "user", "content": "test"}]

        history = agent.get_conversation("test-id")

        assert history is not None
        assert len(history) == 1

    def test_get_conversation_not_exists(self):
        """Should return None for non-existent conversation."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent

        agent = AutoGenAgent()

        history = agent.get_conversation("non-existent")

        assert history is None

    def test_get_all_conversations(self):
        """Should return all conversations."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent

        agent = AutoGenAgent()
        agent._conversations["id1"] = [{"role": "user", "content": "test1"}]
        agent._conversations["id2"] = [{"role": "user", "content": "test2"}]

        all_convs = agent.get_all_conversations()

        assert len(all_convs) == 2
        assert "id1" in all_convs
        assert "id2" in all_convs

    def test_clear_conversation(self):
        """Should clear specific conversation."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent

        agent = AutoGenAgent()
        agent._conversations["test-id"] = [{"role": "user", "content": "test"}]

        result = agent.clear_conversation("test-id")

        assert result is True
        assert "test-id" not in agent._conversations

    def test_clear_conversation_not_exists(self):
        """Should return False for non-existent conversation."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent

        agent = AutoGenAgent()

        result = agent.clear_conversation("non-existent")

        assert result is False

    def test_clear_all_conversations(self):
        """Should clear all conversations."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent

        agent = AutoGenAgent()
        agent._conversations["id1"] = [{"role": "user", "content": "test1"}]
        agent._conversations["id2"] = [{"role": "user", "content": "test2"}]

        count = agent.clear_all_conversations()

        assert count == 2
        assert len(agent._conversations) == 0


class TestAutoGenIsAvailable:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_is_available_calls_parent(self):
        """Should delegate to parent class."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent

        agent = AutoGenAgent()

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
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent

        agent = AutoGenAgent()

        with patch.object(
            agent.__class__.__bases__[0],
            "is_available",
            AsyncMock(return_value=False),
        ):
            result = await agent.is_available()

        assert result is False


class TestAutoGenConfigStatus:
    """Tests for configuration status reporting."""

    def test_get_config_status(self, tmp_path):
        """Should return full configuration status."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent, AutoGenConfig

        config = AutoGenConfig(
            base_url="http://localhost:8000",
            mode="groupchat",
            max_round=15,
            speaker_selection_method="round_robin",
            allow_code_execution=True,
            work_dir=str(tmp_path),
            human_input_mode="TERMINATE",
            conversation_timeout=600,
            audit_all_requests=True,
        )
        agent = AutoGenAgent(config=config)

        status = agent.get_config_status()

        assert status["mode"] == "groupchat"
        assert status["max_round"] == 15
        assert status["speaker_selection_method"] == "round_robin"
        assert status["code_execution_enabled"] is True
        assert status["work_dir"] == str(tmp_path)
        assert status["human_input_mode"] == "TERMINATE"
        assert status["conversation_timeout"] == 600
        assert status["audit_enabled"] is True
        assert status["active_conversations"] == 0


class TestAutoGenInheritedBehavior:
    """Tests for behavior inherited from ExternalFrameworkAgent."""

    def test_inherits_from_external_framework_agent(self):
        """Should inherit from ExternalFrameworkAgent."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent

        agent = AutoGenAgent()

        assert isinstance(agent, ExternalFrameworkAgent)

    def test_has_circuit_breaker_support(self):
        """Should support circuit breaker configuration."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent

        agent = AutoGenAgent(
            enable_circuit_breaker=True,
            circuit_breaker_threshold=5,
        )

        assert agent._circuit_breaker is not None

    def test_supports_response_sanitization(self):
        """Should inherit response sanitization."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent

        agent = AutoGenAgent()

        # Test sanitization method exists and works
        result = agent._sanitize_response("Test\x00with\x00nulls")
        assert "\x00" not in result

    def test_supports_generation_params(self):
        """Should support generation parameters."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent

        agent = AutoGenAgent()
        agent.set_generation_params(temperature=0.7, top_p=0.9)

        assert agent.temperature == 0.7
        assert agent.top_p == 0.9


class TestAutoGenRequestPayload:
    """Tests for request payload building."""

    def test_build_request_payload_basic(self):
        """Should build basic request payload."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent

        agent = AutoGenAgent()

        payload = agent._build_request_payload("Test message")

        assert payload["message"] == "Test message"
        assert payload["model"] == "autogen"
        assert "config" in payload
        assert payload["config"]["mode"] == "groupchat"
        assert payload["config"]["max_round"] == 10

    def test_build_request_payload_with_conversation_id(self):
        """Should include conversation ID in payload."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent

        agent = AutoGenAgent()

        payload = agent._build_request_payload("Test message", conversation_id="test-123")

        assert payload["conversation_id"] == "test-123"

    def test_build_request_payload_with_code_execution(self, tmp_path):
        """Should include code execution config in payload."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent, AutoGenConfig

        config = AutoGenConfig(
            base_url="http://localhost:8000",
            allow_code_execution=True,
            work_dir=str(tmp_path),
        )
        agent = AutoGenAgent(config=config)

        payload = agent._build_request_payload("Test")

        assert payload["config"]["code_execution_config"]["enabled"] is True
        assert payload["config"]["code_execution_config"]["work_dir"] == str(tmp_path)


class TestAutoGenModuleExports:
    """Tests for module exports."""

    def test_exports_agent_class(self):
        """Should export AutoGenAgent."""
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent

        assert AutoGenAgent is not None

    def test_exports_config_class(self):
        """Should export AutoGenConfig."""
        from aragora.agents.api_agents.autogen_agent import AutoGenConfig

        assert AutoGenConfig is not None

    def test_all_exports(self):
        """Should have correct __all__ exports."""
        from aragora.agents.api_agents import autogen_agent

        assert "AutoGenAgent" in autogen_agent.__all__
        assert "AutoGenConfig" in autogen_agent.__all__
