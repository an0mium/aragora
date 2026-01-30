"""
Tests for Tinker API Agent.

Tests cover:
- Initialization and configuration
- TinkerAgent class
- respond() method (equivalent to generate)
- respond_stream() method
- critique() method
- Model selection
- API request handling
- Streaming response handling
- Error handling
- Convenience subclasses

Note: TinkerAgent uses respond() instead of generate() for the main
generation method. The tests mock the TinkerClient to avoid actual API calls.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core import Critique, Message


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_env_with_tinker_key(monkeypatch):
    """Set up environment with Tinker API key."""
    monkeypatch.setenv("TINKER_API_KEY", "test-tinker-key")
    # Also set other common keys for consistency
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    # Clear circuit breaker cache to avoid state leaking between tests
    from aragora.resilience import _registry_lock, _v2_registry

    with _registry_lock:
        _v2_registry.clear()


@pytest.fixture
def mock_env_no_tinker_key(monkeypatch):
    """Clear Tinker API key from environment."""
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    # Clear circuit breaker cache
    from aragora.resilience import _registry_lock, _v2_registry

    with _registry_lock:
        _v2_registry.clear()


@pytest.fixture
def mock_tinker_client():
    """Mock TinkerClient for testing."""
    client = MagicMock()
    client.sample = AsyncMock(return_value="This is a test response from Tinker.")
    client.sample_stream = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_tinker_response():
    """Standard Tinker API response text."""
    return "This is a test response from Tinker fine-tuned model."


@pytest.fixture
def sample_context():
    """Sample message context for testing."""
    return [
        Message(
            agent="agent1",
            content="First message",
            role="proposer",
            round=1,
        ),
        Message(
            agent="agent2",
            content="Response to first message",
            role="critic",
            round=1,
        ),
    ]


# ============================================================================
# TinkerAgent Initialization Tests
# ============================================================================


class TestTinkerAgentInitialization:
    """Tests for TinkerAgent initialization.

    Note: We patch the abstract method 'generate' since TinkerAgent doesn't
    implement it directly (it uses respond() instead).
    """

    def test_init_with_defaults(self, mock_env_with_tinker_key):
        """Should initialize with default values."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        # Patch generate to make class instantiable
        with patch.object(TinkerAgent, "generate", AsyncMock(return_value="")):
            agent = TinkerAgent()

            assert agent.name == "tinker"
            assert agent.model == "llama-3.3-70b"
            assert agent.role == "proposer"
            assert agent.timeout == 120
            assert agent.agent_type == "tinker"
            assert agent.temperature == 0.7
            assert agent.max_tokens == 2048
            assert agent.model_id is None
            assert agent.adapter is None
            assert agent.api_key == "test-tinker-key"

    def test_init_with_custom_config(self, mock_env_with_tinker_key):
        """Should initialize with custom configuration."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        with patch.object(TinkerAgent, "generate", AsyncMock(return_value="")):
            agent = TinkerAgent(
                name="custom-tinker",
                model="qwen-2.5-72b",
                role="critic",
                timeout=180,
                model_id="aragora-security-v1",
                adapter="security-expert",
                temperature=0.9,
                max_tokens=4096,
            )

            assert agent.name == "custom-tinker"
            assert agent.model == "qwen-2.5-72b"
            assert agent.role == "critic"
            assert agent.timeout == 180
            assert agent.model_id == "aragora-security-v1"
            assert agent.adapter == "security-expert"
            assert agent.temperature == 0.9
            assert agent.max_tokens == 4096

    def test_init_with_explicit_api_key(self, mock_env_no_tinker_key):
        """Should use explicitly provided API key."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        with patch.object(TinkerAgent, "generate", AsyncMock(return_value="")):
            agent = TinkerAgent(api_key="explicit-tinker-key")

            assert agent.api_key == "explicit-tinker-key"

    def test_init_without_api_key(self, mock_env_no_tinker_key):
        """Should initialize with empty API key if not provided."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        with patch.object(TinkerAgent, "generate", AsyncMock(return_value="")):
            agent = TinkerAgent()

            assert agent.api_key == ""

    def test_agent_registry_registration(self, mock_env_with_tinker_key):
        """Should be registered in agent registry."""
        from aragora.agents.registry import AgentRegistry

        spec = AgentRegistry.get_spec("tinker")

        assert spec is not None
        assert spec.default_model == "llama-3.3-70b"
        assert spec.agent_type == "API"
        assert spec.env_vars == "TINKER_API_KEY"

    def test_config_initialization(self, mock_env_with_tinker_key):
        """Should initialize TinkerConfig correctly."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        with patch.object(TinkerAgent, "generate", AsyncMock(return_value="")):
            agent = TinkerAgent(timeout=200)

            assert agent._config is not None
            assert agent._config.api_key == "test-tinker-key"
            assert agent._config.base_model == "llama-3.3-70b"
            assert agent._config.timeout == 200.0


class TestTinkerAgentModelSelection:
    """Tests for model selection."""

    def test_supported_models_mapping(self, mock_env_with_tinker_key):
        """Should have correct model mappings."""
        from aragora.agents.api_agents.tinker import TinkerAgent
        from aragora.training.tinker_client import TinkerModel

        assert "llama-3.3-70b" in TinkerAgent.SUPPORTED_MODELS
        assert "llama-3.1-8b" in TinkerAgent.SUPPORTED_MODELS
        assert "qwen-2.5-72b" in TinkerAgent.SUPPORTED_MODELS
        assert "qwen-3-32b" in TinkerAgent.SUPPORTED_MODELS
        assert "deepseek-v3" in TinkerAgent.SUPPORTED_MODELS
        assert "deepseek-r1" in TinkerAgent.SUPPORTED_MODELS

        assert TinkerAgent.SUPPORTED_MODELS["llama-3.3-70b"] == TinkerModel.LLAMA_3_3_70B

    def test_set_adapter(self, mock_env_with_tinker_key):
        """Should set adapter and update model_id."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        with patch.object(TinkerAgent, "generate", AsyncMock(return_value="")):
            agent = TinkerAgent()

            assert agent.adapter is None
            assert agent.model_id is None

            agent.set_adapter("security-expert")

            assert agent.adapter == "security-expert"
            assert agent.model_id == "llama-3.3-70b-security-expert"

    def test_clear_adapter(self, mock_env_with_tinker_key):
        """Should clear adapter and reset model_id."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        with patch.object(TinkerAgent, "generate", AsyncMock(return_value="")):
            agent = TinkerAgent(adapter="test-adapter")
            agent.model_id = "llama-3.3-70b-test-adapter"

            agent.set_adapter(None)

            assert agent.adapter is None
            assert agent.model_id is None

    def test_get_model_info(self, mock_env_with_tinker_key):
        """Should return model info dictionary."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        with patch.object(TinkerAgent, "generate", AsyncMock(return_value="")):
            agent = TinkerAgent(
                name="test-agent",
                model="qwen-2.5-72b",
                role="judge",
                model_id="custom-model",
                adapter="custom-adapter",
                temperature=0.8,
                max_tokens=3000,
            )

            info = agent.get_model_info()

            assert info["agent_name"] == "test-agent"
            assert info["agent_type"] == "tinker"
            assert info["base_model"] == "qwen-2.5-72b"
            assert info["model_id"] == "custom-model"
            assert info["adapter"] == "custom-adapter"
            assert info["role"] == "judge"
            assert info["temperature"] == 0.8
            assert info["max_tokens"] == 3000


# ============================================================================
# TinkerAgent Client Management Tests
# ============================================================================


class TestTinkerAgentClientManagement:
    """Tests for client management."""

    def test_lazy_client_initialization(self, mock_env_with_tinker_key):
        """Should lazily initialize client on first access."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        with patch.object(TinkerAgent, "generate", AsyncMock(return_value="")):
            agent = TinkerAgent()

            assert agent._client is None

            # Access client property
            client = agent.client

            assert agent._client is not None
            assert client is agent._client

    @pytest.mark.asyncio
    async def test_close_client(self, mock_env_with_tinker_key):
        """Should close the client properly."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        with patch.object(TinkerAgent, "generate", AsyncMock(return_value="")):
            agent = TinkerAgent()

            # Initialize client
            _ = agent.client

            assert agent._client is not None

            # Mock the close method
            agent._client.close = AsyncMock()

            await agent.close()

            assert agent._client is None


# ============================================================================
# TinkerAgent Generate (Respond) Tests
# ============================================================================


class TestTinkerAgentRespond:
    """Tests for respond method."""

    @pytest.mark.asyncio
    async def test_respond_basic(self, mock_env_with_tinker_key, mock_tinker_response):
        """Should generate response from API."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        with patch.object(TinkerAgent, "generate", AsyncMock(return_value="")):
            agent = TinkerAgent()

            with patch.object(agent, "client") as mock_client:
                mock_client.sample = AsyncMock(return_value=mock_tinker_response)

                result = await agent.respond("Test prompt")

            assert "test response from Tinker" in result

    @pytest.mark.asyncio
    async def test_respond_with_context(
        self, mock_env_with_tinker_key, mock_tinker_response, sample_context
    ):
        """Should include context in prompt."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        with patch.object(TinkerAgent, "generate", AsyncMock(return_value="")):
            agent = TinkerAgent()

            with patch.object(agent, "client") as mock_client:
                mock_client.sample = AsyncMock(return_value=mock_tinker_response)

                result = await agent.respond("Test prompt", context=sample_context)

                # Verify sample was called with prompt containing context
                call_kwargs = mock_client.sample.call_args
                prompt_arg = call_kwargs.kwargs.get("prompt") or call_kwargs.args[0]
                assert "Previous discussion" in prompt_arg
                assert "agent1" in prompt_arg or "agent2" in prompt_arg

            assert result is not None

    @pytest.mark.asyncio
    async def test_respond_with_system_prompt(self, mock_env_with_tinker_key, mock_tinker_response):
        """Should include system prompt when provided."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        with patch.object(TinkerAgent, "generate", AsyncMock(return_value="")):
            agent = TinkerAgent()

            with patch.object(agent, "client") as mock_client:
                mock_client.sample = AsyncMock(return_value=mock_tinker_response)

                await agent.respond("Test prompt", system_prompt="You are a security expert.")

                call_kwargs = mock_client.sample.call_args
                prompt_arg = call_kwargs.kwargs.get("prompt") or call_kwargs.args[0]
                assert "security expert" in prompt_arg

    @pytest.mark.asyncio
    async def test_respond_uses_model_id(self, mock_env_with_tinker_key, mock_tinker_response):
        """Should pass model_id to client."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        with patch.object(TinkerAgent, "generate", AsyncMock(return_value="")):
            agent = TinkerAgent(model_id="custom-fine-tuned-model")

            with patch.object(agent, "client") as mock_client:
                mock_client.sample = AsyncMock(return_value=mock_tinker_response)

                await agent.respond("Test prompt")

                mock_client.sample.assert_called_once()
                call_kwargs = mock_client.sample.call_args.kwargs
                assert call_kwargs["model_id"] == "custom-fine-tuned-model"

    @pytest.mark.asyncio
    async def test_respond_records_token_usage(
        self, mock_env_with_tinker_key, mock_tinker_response
    ):
        """Should estimate and record token usage."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        with patch.object(TinkerAgent, "generate", AsyncMock(return_value="")):
            agent = TinkerAgent()
            agent.reset_token_usage()

            with patch.object(agent, "client") as mock_client:
                mock_client.sample = AsyncMock(return_value=mock_tinker_response)

                await agent.respond("Test prompt")

            # Token usage should be recorded (estimated based on text length)
            assert agent.total_tokens_in > 0
            assert agent.total_tokens_out > 0


# ============================================================================
# TinkerAgent Streaming Tests
# ============================================================================


class TestTinkerAgentRespondStream:
    """Tests for streaming responses."""

    @pytest.mark.asyncio
    async def test_respond_stream_yields_chunks(self, mock_env_with_tinker_key):
        """Should yield text chunks from stream."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        with patch.object(TinkerAgent, "generate", AsyncMock(return_value="")):
            agent = TinkerAgent()

            async def mock_stream(*args, **kwargs):
                for chunk in ["Hello", " from", " Tinker", "!"]:
                    yield chunk

            with patch.object(agent, "client") as mock_client:
                mock_client.sample_stream = mock_stream

                chunks = []
                async for chunk in agent.respond_stream("Test prompt"):
                    chunks.append(chunk)

            assert len(chunks) == 4
            assert "".join(chunks) == "Hello from Tinker!"

    @pytest.mark.asyncio
    async def test_respond_stream_with_context(self, mock_env_with_tinker_key, sample_context):
        """Should include context in streaming prompt."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        with patch.object(TinkerAgent, "generate", AsyncMock(return_value="")):
            agent = TinkerAgent()
            captured_prompt = None

            async def mock_stream(prompt, **kwargs):
                nonlocal captured_prompt
                captured_prompt = prompt
                yield "response"

            with patch.object(agent, "client") as mock_client:
                mock_client.sample_stream = mock_stream

                async for _ in agent.respond_stream("Test prompt", context=sample_context):
                    pass

            assert captured_prompt is not None
            assert "Previous discussion" in captured_prompt


# ============================================================================
# TinkerAgent Critique Tests
# ============================================================================


class TestTinkerAgentCritique:
    """Tests for critique method."""

    @pytest.mark.asyncio
    async def test_critique_returns_structured_feedback(self, mock_env_with_tinker_key):
        """Should return structured critique."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        with patch.object(TinkerAgent, "generate", AsyncMock(return_value="")):
            agent = TinkerAgent()

            critique_response = """ISSUES:
- Issue one: Missing error handling
- Issue two: Inefficient algorithm

SUGGESTIONS:
- Add try/catch blocks
- Use binary search instead

SEVERITY: 6.0
REASONING: The proposal has significant gaps in robustness."""

            with patch.object(agent, "respond", new_callable=AsyncMock) as mock_respond:
                mock_respond.return_value = critique_response

                critique = await agent.critique(
                    proposal="Test proposal",
                    task="Design an API",
                    target_agent="test-agent",
                )

            assert isinstance(critique, Critique)
            assert critique.agent == "tinker"
            assert critique.target_agent == "test-agent"
            assert len(critique.issues) > 0
            assert len(critique.suggestions) > 0
            assert 0 <= critique.severity <= 10

    @pytest.mark.asyncio
    async def test_critique_includes_target_agent_in_prompt(self, mock_env_with_tinker_key):
        """Should include target agent in critique prompt."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        with patch.object(TinkerAgent, "generate", AsyncMock(return_value="")):
            agent = TinkerAgent()

            with patch.object(agent, "respond", new_callable=AsyncMock) as mock_respond:
                mock_respond.return_value = "No issues found."

                await agent.critique(
                    proposal="Test proposal",
                    task="Test task",
                    target_agent="claude-agent",
                )

                # Check that target agent is mentioned in the prompt
                call_args = mock_respond.call_args
                prompt = call_args.args[0] if call_args.args else call_args.kwargs.get("task")
                assert "claude-agent" in prompt

    @pytest.mark.asyncio
    async def test_critique_without_target_agent(self, mock_env_with_tinker_key):
        """Should handle critique without target agent."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        with patch.object(TinkerAgent, "generate", AsyncMock(return_value="")):
            agent = TinkerAgent()

            with patch.object(agent, "respond", new_callable=AsyncMock) as mock_respond:
                mock_respond.return_value = "Minor issues detected."

                critique = await agent.critique(
                    proposal="Test proposal",
                    task="Test task",
                )

            assert critique.target_agent == "proposal"


# ============================================================================
# TinkerAgent Error Handling Tests
# ============================================================================


class TestTinkerAgentErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_handles_api_error(self, mock_env_with_tinker_key):
        """Should propagate API errors."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        agent = TinkerAgent()

        with patch.object(agent, "client") as mock_client:
            mock_client.sample = AsyncMock(side_effect=RuntimeError("API error"))

            with pytest.raises(RuntimeError) as exc_info:
                await agent.respond("Test prompt")

            assert "API error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handles_timeout_error(self, mock_env_with_tinker_key):
        """Should propagate timeout errors."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        agent = TinkerAgent()

        with patch.object(agent, "client") as mock_client:
            mock_client.sample = AsyncMock(side_effect=TimeoutError("Request timed out"))

            with pytest.raises(TimeoutError):
                await agent.respond("Test prompt")

    @pytest.mark.asyncio
    async def test_handles_os_error(self, mock_env_with_tinker_key):
        """Should propagate OS errors (network issues)."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        agent = TinkerAgent()

        with patch.object(agent, "client") as mock_client:
            mock_client.sample = AsyncMock(side_effect=OSError("Network error"))

            with pytest.raises(OSError):
                await agent.respond("Test prompt")

    @pytest.mark.asyncio
    async def test_handles_value_error(self, mock_env_with_tinker_key):
        """Should propagate value errors."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        agent = TinkerAgent()

        with patch.object(agent, "client") as mock_client:
            mock_client.sample = AsyncMock(side_effect=ValueError("Invalid input"))

            with pytest.raises(ValueError):
                await agent.respond("Test prompt")

    @pytest.mark.asyncio
    async def test_records_circuit_breaker_failure(self, mock_env_with_tinker_key):
        """Should record failure in circuit breaker on error."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        agent = TinkerAgent()

        # Mock circuit breaker
        mock_breaker = MagicMock()
        mock_breaker.record_failure = MagicMock()
        agent._circuit_breaker = mock_breaker

        with patch.object(agent, "client") as mock_client:
            mock_client.sample = AsyncMock(side_effect=RuntimeError("API error"))

            with pytest.raises(RuntimeError):
                await agent.respond("Test prompt")

        mock_breaker.record_failure.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_handles_error(self, mock_env_with_tinker_key):
        """Should handle errors in streaming."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        agent = TinkerAgent()

        async def mock_stream(*args, **kwargs):
            yield "partial"
            raise RuntimeError("Stream error")

        with patch.object(agent, "client") as mock_client:
            mock_client.sample_stream = mock_stream

            chunks = []
            with pytest.raises(RuntimeError):
                async for chunk in agent.respond_stream("Test prompt"):
                    chunks.append(chunk)

        assert len(chunks) == 1


# ============================================================================
# TinkerAgent Prompt Building Tests
# ============================================================================


class TestTinkerAgentPromptBuilding:
    """Tests for prompt building."""

    def test_build_prompt_basic(self, mock_env_with_tinker_key):
        """Should build basic prompt correctly."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        agent = TinkerAgent(name="test-tinker", role="proposer")

        prompt = agent._build_prompt("Test task", None, None)

        assert "Task: Test task" in prompt
        assert "test-tinker (proposer)" in prompt

    def test_build_prompt_with_context(self, mock_env_with_tinker_key, sample_context):
        """Should include context in prompt."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        agent = TinkerAgent()

        prompt = agent._build_prompt("Test task", sample_context, None)

        assert "Previous discussion" in prompt
        assert "First message" in prompt

    def test_build_prompt_with_system_prompt(self, mock_env_with_tinker_key):
        """Should include custom system prompt."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        agent = TinkerAgent()

        prompt = agent._build_prompt("Test task", None, "You are an expert in security.")

        assert "System: You are an expert in security." in prompt

    def test_get_default_system_prompt_proposer(self, mock_env_with_tinker_key):
        """Should return proposer system prompt."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        agent = TinkerAgent(role="proposer")

        prompt = agent._get_default_system_prompt()

        assert "expert debater" in prompt
        assert "propose" in prompt.lower()

    def test_get_default_system_prompt_critic(self, mock_env_with_tinker_key):
        """Should return critic system prompt."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        agent = TinkerAgent(role="critic")

        prompt = agent._get_default_system_prompt()

        assert "critical analyst" in prompt

    def test_get_default_system_prompt_judge(self, mock_env_with_tinker_key):
        """Should return judge system prompt."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        agent = TinkerAgent(role="judge")

        prompt = agent._get_default_system_prompt()

        assert "impartial judge" in prompt

    def test_get_default_system_prompt_synthesizer(self, mock_env_with_tinker_key):
        """Should return synthesizer system prompt."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        agent = TinkerAgent(role="synthesizer")

        prompt = agent._get_default_system_prompt()

        assert "synthesizer" in prompt


# ============================================================================
# Convenience Subclass Tests
# ============================================================================


class TestTinkerLlamaAgent:
    """Tests for TinkerLlamaAgent subclass."""

    def test_init_with_defaults(self, mock_env_with_tinker_key):
        """Should initialize with Llama defaults."""
        from aragora.agents.api_agents.tinker import TinkerLlamaAgent

        agent = TinkerLlamaAgent()

        assert agent.name == "tinker-llama"
        assert agent.model == "llama-3.3-70b"
        assert agent.role == "proposer"

    def test_init_with_custom_role(self, mock_env_with_tinker_key):
        """Should accept custom role."""
        from aragora.agents.api_agents.tinker import TinkerLlamaAgent

        agent = TinkerLlamaAgent(role="critic")

        assert agent.role == "critic"
        assert agent.model == "llama-3.3-70b"

    def test_registry_registration(self, mock_env_with_tinker_key):
        """Should be registered in agent registry."""
        from aragora.agents.registry import AgentRegistry

        spec = AgentRegistry.get_spec("tinker-llama")

        assert spec is not None
        assert spec.default_model == "llama-3.3-70b"


class TestTinkerQwenAgent:
    """Tests for TinkerQwenAgent subclass."""

    def test_init_with_defaults(self, mock_env_with_tinker_key):
        """Should initialize with Qwen defaults."""
        from aragora.agents.api_agents.tinker import TinkerQwenAgent

        agent = TinkerQwenAgent()

        assert agent.name == "tinker-qwen"
        assert agent.model == "qwen-2.5-72b"
        assert agent.role == "proposer"

    def test_init_with_custom_name(self, mock_env_with_tinker_key):
        """Should accept custom name."""
        from aragora.agents.api_agents.tinker import TinkerQwenAgent

        agent = TinkerQwenAgent(name="custom-qwen")

        assert agent.name == "custom-qwen"
        assert agent.model == "qwen-2.5-72b"

    def test_registry_registration(self, mock_env_with_tinker_key):
        """Should be registered in agent registry."""
        from aragora.agents.registry import AgentRegistry

        spec = AgentRegistry.get_spec("tinker-qwen")

        assert spec is not None
        assert spec.default_model == "qwen-2.5-72b"


class TestTinkerDeepSeekAgent:
    """Tests for TinkerDeepSeekAgent subclass."""

    def test_init_with_defaults(self, mock_env_with_tinker_key):
        """Should initialize with DeepSeek defaults."""
        from aragora.agents.api_agents.tinker import TinkerDeepSeekAgent

        agent = TinkerDeepSeekAgent()

        assert agent.name == "tinker-deepseek"
        assert agent.model == "deepseek-v3"
        assert agent.role == "proposer"

    def test_init_with_kwargs(self, mock_env_with_tinker_key):
        """Should pass kwargs to parent."""
        from aragora.agents.api_agents.tinker import TinkerDeepSeekAgent

        agent = TinkerDeepSeekAgent(
            timeout=300,
            temperature=0.5,
            max_tokens=8000,
        )

        assert agent.timeout == 300
        assert agent.temperature == 0.5
        assert agent.max_tokens == 8000

    def test_registry_registration(self, mock_env_with_tinker_key):
        """Should be registered in agent registry."""
        from aragora.agents.registry import AgentRegistry

        spec = AgentRegistry.get_spec("tinker-deepseek")

        assert spec is not None
        assert spec.default_model == "deepseek-v3"


# ============================================================================
# Token Usage Tracking Tests
# ============================================================================


class TestTinkerAgentTokenUsage:
    """Tests for token usage tracking."""

    def test_get_token_usage(self, mock_env_with_tinker_key):
        """Should return token usage summary."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        agent = TinkerAgent()
        agent._record_token_usage(100, 50)

        usage = agent.get_token_usage()

        assert usage["tokens_in"] == 100
        assert usage["tokens_out"] == 50
        assert usage["total_tokens_in"] == 100
        assert usage["total_tokens_out"] == 50

    def test_reset_token_usage(self, mock_env_with_tinker_key):
        """Should reset token counters."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        agent = TinkerAgent()
        agent._record_token_usage(100, 50)
        agent.reset_token_usage()

        assert agent.last_tokens_in == 0
        assert agent.last_tokens_out == 0
        assert agent.total_tokens_in == 0
        assert agent.total_tokens_out == 0

    def test_accumulates_token_usage(self, mock_env_with_tinker_key):
        """Should accumulate total token usage."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        agent = TinkerAgent()
        agent._record_token_usage(100, 50)
        agent._record_token_usage(200, 100)

        assert agent.last_tokens_in == 200
        assert agent.last_tokens_out == 100
        assert agent.total_tokens_in == 300
        assert agent.total_tokens_out == 150


# ============================================================================
# Circuit Breaker Tests
# ============================================================================


class TestTinkerAgentCircuitBreaker:
    """Tests for circuit breaker functionality."""

    def test_circuit_breaker_is_enabled_by_default(self, mock_env_with_tinker_key):
        """Should have circuit breaker enabled by default."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        agent = TinkerAgent()

        assert agent.enable_circuit_breaker is True
        assert agent._circuit_breaker is not None

    def test_is_circuit_open(self, mock_env_with_tinker_key):
        """Should check circuit breaker state."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        agent = TinkerAgent()

        mock_breaker = MagicMock()
        mock_breaker.can_proceed.return_value = True
        agent._circuit_breaker = mock_breaker

        assert agent.is_circuit_open() is False

        mock_breaker.can_proceed.return_value = False
        assert agent.is_circuit_open() is True

    def test_circuit_breaker_can_be_disabled(self, mock_env_with_tinker_key):
        """Should allow disabling circuit breaker."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        agent = TinkerAgent(enable_circuit_breaker=False)

        assert agent._circuit_breaker is None
        assert agent.is_circuit_open() is False


# ============================================================================
# Generation Parameters Tests
# ============================================================================


class TestTinkerAgentGenerationParams:
    """Tests for generation parameters."""

    def test_set_generation_params(self, mock_env_with_tinker_key):
        """Should set generation parameters."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        agent = TinkerAgent()
        agent.set_generation_params(temperature=0.5, top_p=0.9)

        assert agent.temperature == 0.5
        assert agent.top_p == 0.9

    def test_get_generation_params(self, mock_env_with_tinker_key):
        """Should return non-None generation parameters."""
        from aragora.agents.api_agents.tinker import TinkerAgent

        agent = TinkerAgent()
        agent.temperature = 0.8
        agent.top_p = None

        params = agent.get_generation_params()

        assert "temperature" in params
        assert params["temperature"] == 0.8
        assert "top_p" not in params
