"""
Tests for External Framework Agent.

Tests cover:
- Initialization and configuration
- ExternalFrameworkConfig dataclass
- generate() method with various inputs
- critique() method
- vote() method
- Health check (is_available)
- Error handling (connection errors, API errors, timeouts)
- Response sanitization
- Circuit breaker integration
- Header building and authentication
- Session reuse (_get_session, close)
"""

import asyncio
import json
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


@pytest.fixture
def mock_openai_response():
    """Standard OpenAI-compatible API response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response from the external framework.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


@pytest.fixture
def mock_simple_response():
    """Simple response format."""
    return {"response": "Simple response text."}


class TestExternalFrameworkConfig:
    """Tests for ExternalFrameworkConfig dataclass."""

    def test_config_defaults(self):
        """Should have sensible default values."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkConfig

        config = ExternalFrameworkConfig(base_url="https://api.example.com:8000")

        assert config.base_url == "https://api.example.com:8000"
        assert config.generate_endpoint == "/generate"
        assert config.critique_endpoint == "/critique"
        assert config.vote_endpoint == "/vote"
        assert config.health_endpoint == "/health"
        assert config.api_key_header == "Authorization"
        assert config.api_key_prefix == "Bearer"
        assert config.timeout == 120
        assert config.max_retries == 3
        assert config.enable_response_sanitization is True
        assert config.max_response_length == 100000

    def test_config_custom_values(self):
        """Should accept custom configuration values."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkConfig

        config = ExternalFrameworkConfig(
            base_url="https://api.example.com",
            generate_endpoint="/v1/chat/completions",
            critique_endpoint="/v1/critique",
            vote_endpoint="/v1/vote",
            health_endpoint="/healthz",
            api_key_header="X-API-Key",
            api_key_prefix="",
            extra_headers={"X-Custom": "value"},
            timeout=300,
            max_retries=5,
            retry_delay=2.0,
            retry_backoff=3.0,
            enable_response_sanitization=False,
            max_response_length=50000,
        )

        assert config.base_url == "https://api.example.com"
        assert config.generate_endpoint == "/v1/chat/completions"
        assert config.api_key_header == "X-API-Key"
        assert config.api_key_prefix == ""
        assert config.extra_headers == {"X-Custom": "value"}
        assert config.timeout == 300
        assert config.max_retries == 5


class TestExternalFrameworkAgentInitialization:
    """Tests for agent initialization."""

    def test_init_with_defaults(self):
        """Should initialize with default values."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")

        assert agent.name == "external-framework"
        assert agent.model == "external"
        assert agent.role == "proposer"
        assert agent.agent_type == "external-framework"
        assert agent.timeout == 120

    def test_init_with_custom_config(self):
        """Should initialize with custom configuration."""
        from aragora.agents.api_agents.external_framework import (
            ExternalFrameworkAgent,
            ExternalFrameworkConfig,
        )

        config = ExternalFrameworkConfig(
            base_url="https://custom.example.com",
            timeout=300,
        )

        agent = ExternalFrameworkAgent(
            name="custom-external",
            model="gpt-4",
            role="critic",
            config=config,
        )

        assert agent.name == "custom-external"
        assert agent.model == "gpt-4"
        assert agent.role == "critic"
        assert agent.base_url == "https://custom.example.com"
        assert agent.timeout == 300

    def test_init_base_url_overrides_config(self):
        """Should prefer explicit base_url over config.base_url."""
        from aragora.agents.api_agents.external_framework import (
            ExternalFrameworkAgent,
            ExternalFrameworkConfig,
        )

        config = ExternalFrameworkConfig(base_url="https://config-url.com")

        agent = ExternalFrameworkAgent(config=config, base_url="https://override-url.com")

        assert agent.base_url == "https://override-url.com"

    def test_init_with_api_key(self):
        """Should accept API key."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(
            api_key="test-api-key-123", base_url="https://api.example.com"
        )

        assert agent.api_key == "test-api-key-123"

    def test_init_with_env_var(self, monkeypatch):
        """Should read base_url and api_key from environment variables."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        monkeypatch.setenv("EXTERNAL_FRAMEWORK_URL", "https://env-url.com")
        monkeypatch.setenv("EXTERNAL_FRAMEWORK_API_KEY", "env-api-key")

        agent = ExternalFrameworkAgent()

        assert agent.base_url == "https://env-url.com"
        assert agent.api_key == "env-api-key"

    def test_agent_registry_registration(self):
        """Should be registered in agent registry."""
        from aragora.agents.registry import AgentRegistry

        spec = AgentRegistry.get_spec("external-framework")

        assert spec is not None
        assert spec.default_model == "external"
        assert spec.agent_type == "API"
        assert spec.accepts_api_key is True

    def test_circuit_breaker_configuration(self):
        """Should configure circuit breaker."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(
            base_url="https://api.example.com",
            enable_circuit_breaker=True,
            circuit_breaker_threshold=10,
            circuit_breaker_cooldown=120.0,
        )

        assert agent._circuit_breaker is not None

    def test_circuit_breaker_disabled(self):
        """Should allow disabling circuit breaker."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(
            base_url="https://api.example.com", enable_circuit_breaker=False
        )

        assert agent._circuit_breaker is None


class TestExternalFrameworkHeaders:
    """Tests for header building and authentication."""

    def test_build_headers_basic(self):
        """Should build basic headers."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")
        headers = agent._build_headers()

        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"
        assert "Authorization" not in headers

    def test_build_headers_with_api_key(self):
        """Should include API key in headers."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(api_key="test-key-123", base_url="https://api.example.com")
        headers = agent._build_headers()

        assert headers["Authorization"] == "Bearer test-key-123"

    def test_build_headers_custom_prefix(self):
        """Should support custom API key prefix."""
        from aragora.agents.api_agents.external_framework import (
            ExternalFrameworkAgent,
            ExternalFrameworkConfig,
        )

        config = ExternalFrameworkConfig(
            base_url="https://api.example.com",
            api_key_header="X-API-Key",
            api_key_prefix="",
        )
        agent = ExternalFrameworkAgent(config=config, api_key="raw-key")
        headers = agent._build_headers()

        assert headers["X-API-Key"] == "raw-key"

    def test_build_headers_with_extra(self):
        """Should include extra headers."""
        from aragora.agents.api_agents.external_framework import (
            ExternalFrameworkAgent,
            ExternalFrameworkConfig,
        )

        config = ExternalFrameworkConfig(
            base_url="https://api.example.com",
            extra_headers={"X-Custom": "value", "X-Another": "header"},
        )
        agent = ExternalFrameworkAgent(config=config)
        headers = agent._build_headers()

        assert headers["X-Custom"] == "value"
        assert headers["X-Another"] == "header"


class TestExternalFrameworkIsAvailable:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_is_available_returns_true(self):
        """Should return True when server is healthy."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)

        with patch.object(agent, "_get_session", new_callable=AsyncMock, return_value=mock_session):
            result = await agent.is_available()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_available_returns_true_on_204(self):
        """Should return True on 204 No Content."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")

        mock_response = MagicMock()
        mock_response.status = 204
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)

        with patch.object(agent, "_get_session", new_callable=AsyncMock, return_value=mock_session):
            result = await agent.is_available()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_available_returns_false_on_error(self):
        """Should return False on connection error."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=aiohttp.ClientError())

        with patch.object(agent, "_get_session", new_callable=AsyncMock, return_value=mock_session):
            result = await agent.is_available()

        assert result is False

    @pytest.mark.asyncio
    async def test_is_available_returns_false_on_500(self):
        """Should return False on server error."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")

        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)

        with patch.object(agent, "_get_session", new_callable=AsyncMock, return_value=mock_session):
            result = await agent.is_available()

        assert result is False


class TestExternalFrameworkGenerate:
    """Tests for generate method."""

    @pytest.mark.asyncio
    async def test_generate_openai_format(self, mock_openai_response):
        """Should parse OpenAI-compatible response format."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_openai_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        with patch.object(agent, "_get_session", new_callable=AsyncMock, return_value=mock_session):
            result = await agent.generate("Test prompt")

        assert "test response from the external framework" in result

    @pytest.mark.asyncio
    async def test_generate_simple_format(self, mock_simple_response):
        """Should parse simple response format."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_simple_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        with patch.object(agent, "_get_session", new_callable=AsyncMock, return_value=mock_session):
            result = await agent.generate("Test prompt")

        assert result == "Simple response text."

    @pytest.mark.asyncio
    async def test_generate_with_context(self, sample_context, mock_simple_response):
        """Should include context in request."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_simple_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        captured_payload = {}

        def capture_post(url, json=None, headers=None):
            captured_payload["json"] = json
            return mock_response

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=capture_post)

        with patch.object(agent, "_get_session", new_callable=AsyncMock, return_value=mock_session):
            await agent.generate("Test prompt", context=sample_context)

        # Should include context in prompt
        assert "First message" in captured_payload["json"]["prompt"]

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self, mock_simple_response):
        """Should include system prompt."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")
        agent.system_prompt = "You are a helpful assistant."

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_simple_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        captured_payload = {}

        def capture_post(url, json=None, headers=None):
            captured_payload["json"] = json
            return mock_response

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=capture_post)

        with patch.object(agent, "_get_session", new_callable=AsyncMock, return_value=mock_session):
            await agent.generate("Test prompt")

        assert "System:" in captured_payload["json"]["prompt"]
        assert "helpful assistant" in captured_payload["json"]["prompt"]

    @pytest.mark.asyncio
    async def test_generate_includes_generation_params(self, mock_simple_response):
        """Should include generation parameters in request."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")
        agent.set_generation_params(temperature=0.7, top_p=0.9)

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_simple_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        captured_payload = {}

        def capture_post(url, json=None, headers=None):
            captured_payload["json"] = json
            return mock_response

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=capture_post)

        with patch.object(agent, "_get_session", new_callable=AsyncMock, return_value=mock_session):
            await agent.generate("Test prompt")

        assert captured_payload["json"]["temperature"] == 0.7
        assert captured_payload["json"]["top_p"] == 0.9


class TestExternalFrameworkResponseExtraction:
    """Tests for response extraction from various formats."""

    def test_extract_openai_chat_format(self):
        """Should extract from OpenAI chat completion format."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")
        data = {
            "choices": [{"message": {"content": "Hello from chat"}}],
        }

        result = agent._extract_response_text(data)
        assert result == "Hello from chat"

    def test_extract_openai_completion_format(self):
        """Should extract from OpenAI completion format."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")
        data = {
            "choices": [{"text": "Hello from completion"}],
        }

        result = agent._extract_response_text(data)
        assert result == "Hello from completion"

    def test_extract_simple_response(self):
        """Should extract from simple response format."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")
        data = {"response": "Simple response"}

        result = agent._extract_response_text(data)
        assert result == "Simple response"

    def test_extract_text_format(self):
        """Should extract from text format."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")
        data = {"text": "Text response"}

        result = agent._extract_response_text(data)
        assert result == "Text response"

    def test_extract_content_format(self):
        """Should extract from content format."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")
        data = {"content": "Content response"}

        result = agent._extract_response_text(data)
        assert result == "Content response"

    def test_extract_output_format(self):
        """Should extract from output format."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")
        data = {"output": "Output response"}

        result = agent._extract_response_text(data)
        assert result == "Output response"

    def test_extract_fallback_to_json(self):
        """Should fall back to JSON stringification for unknown formats."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")
        data = {"unknown_field": "value", "another": 123}

        result = agent._extract_response_text(data)
        assert "unknown_field" in result
        assert "value" in result


class TestExternalFrameworkResponseSanitization:
    """Tests for response sanitization."""

    def test_sanitize_truncates_long_response(self):
        """Should truncate responses exceeding max length."""
        from aragora.agents.api_agents.external_framework import (
            ExternalFrameworkAgent,
            ExternalFrameworkConfig,
        )

        config = ExternalFrameworkConfig(
            base_url="https://api.example.com",
            max_response_length=100,
        )
        agent = ExternalFrameworkAgent(config=config)

        long_text = "x" * 200
        result = agent._sanitize_response(long_text)

        assert len(result) < 200
        assert "[truncated]" in result

    def test_sanitize_removes_null_bytes(self):
        """Should remove null bytes."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")

        text = "Hello\x00World"
        result = agent._sanitize_response(text)

        assert "\x00" not in result
        assert result == "HelloWorld"

    def test_sanitize_removes_ansi_escapes(self):
        """Should remove ANSI escape sequences."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")

        text = "Hello \x1b[31mRed\x1b[0m World"
        result = agent._sanitize_response(text)

        assert "\x1b" not in result
        assert "Red" in result

    def test_sanitize_disabled(self):
        """Should skip sanitization when disabled."""
        from aragora.agents.api_agents.external_framework import (
            ExternalFrameworkAgent,
            ExternalFrameworkConfig,
        )

        config = ExternalFrameworkConfig(
            base_url="https://api.example.com",
            enable_response_sanitization=False,
            max_response_length=100,
        )
        agent = ExternalFrameworkAgent(config=config)

        long_text = "x" * 200
        result = agent._sanitize_response(long_text)

        assert result == long_text  # Not truncated


class TestExternalFrameworkCritique:
    """Tests for critique method."""

    @pytest.mark.asyncio
    async def test_critique_returns_structured_feedback(self):
        """Should return structured critique."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")

        with patch.object(agent, "generate", new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = """ISSUES:
- Issue one: The proposal lacks specificity
- Issue two: Missing error handling

SUGGESTIONS:
- Suggestion one: Add more detail
- Suggestion two: Include try-catch blocks

SEVERITY: 6.5
REASONING: The proposal addresses the task but needs refinement."""

            critique = await agent.critique(
                proposal="Test proposal implementation",
                task="Implement a rate limiter",
                target_agent="test-agent",
            )

            assert critique is not None
            assert critique.agent == "external-framework"
            assert critique.target_agent == "test-agent"
            assert len(critique.issues) > 0
            assert len(critique.suggestions) > 0

    @pytest.mark.asyncio
    async def test_critique_with_context(self, sample_context):
        """Should include context when critiquing."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")

        with patch.object(agent, "generate", new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = """ISSUES:
- The proposal doesn't build on previous discussion

SUGGESTIONS:
- Reference earlier points

SEVERITY: 4.0
REASONING: Needs better context integration."""

            critique = await agent.critique(
                proposal="Standalone proposal",
                task="Design API endpoints",
                context=sample_context,
                target_agent="other-agent",
            )

            assert critique is not None
            mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_critique_dedicated_endpoint(self):
        """Should try dedicated critique endpoint first."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "response": """ISSUES:
- Issue from dedicated endpoint

SUGGESTIONS:
- Suggestion from dedicated endpoint

SEVERITY: 5.0
REASONING: Critique from dedicated endpoint."""
            }
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        with patch.object(agent, "_get_session", new_callable=AsyncMock, return_value=mock_session):
            critique = await agent.critique(
                proposal="Test proposal",
                task="Test task",
                target_agent="test-agent",
            )

            assert critique is not None
            assert "dedicated endpoint" in critique.reasoning.lower()


class TestExternalFrameworkVote:
    """Tests for vote method."""

    @pytest.mark.asyncio
    async def test_vote_selects_best_proposal(self):
        """Should vote for best proposal."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")

        proposals = {
            "agent1": "Proposal A: Simple implementation",
            "agent2": "Proposal B: Comprehensive solution with error handling",
        }

        with patch.object(agent, "generate", new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = """CHOICE: agent2
CONFIDENCE: 0.85
CONTINUE: no
REASONING: Proposal B is more comprehensive and production-ready."""

            vote = await agent.vote(proposals, "Implement authentication system")

            assert vote is not None
            assert vote.agent == "external-framework"
            assert vote.choice == "agent2"
            assert vote.confidence == 0.85
            assert vote.continue_debate is False

    @pytest.mark.asyncio
    async def test_vote_with_continue_debate_yes(self):
        """Should indicate debate should continue when appropriate."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")

        proposals = {
            "agent1": "Partial solution A",
            "agent2": "Partial solution B",
        }

        with patch.object(agent, "generate", new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = """CHOICE: agent1
CONFIDENCE: 0.6
CONTINUE: yes
REASONING: Both proposals have merit but need refinement."""

            vote = await agent.vote(proposals, "Complex design task")

            assert vote.continue_debate is True
            assert vote.confidence == 0.6

    @pytest.mark.asyncio
    async def test_vote_parse_handles_unknown_choice(self):
        """Should handle unknown agent choice gracefully."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")

        proposals = {
            "agent1": "Proposal A",
            "agent2": "Proposal B",
        }

        # Simulate response with unknown agent name
        vote = agent._parse_vote(
            """CHOICE: unknown-agent
CONFIDENCE: 0.7
CONTINUE: no
REASONING: Test reasoning.""",
            proposals,
        )

        # Should default to first agent when choice is invalid
        assert vote.choice in proposals


class TestExternalFrameworkErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_handles_api_error(self):
        """Should raise AgentAPIError on API failure."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        # Use unique name to avoid circuit breaker state from other tests
        agent = ExternalFrameworkAgent(
            name="test-api-error", base_url="https://api.example.com", enable_circuit_breaker=False
        )

        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal server error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        with patch.object(agent, "_get_session", new_callable=AsyncMock, return_value=mock_session):
            with pytest.raises(AgentAPIError) as exc_info:
                await agent.generate("Test prompt")

            assert "500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handles_rate_limit_error(self):
        """Should raise AgentRateLimitError on 429."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        # Disable circuit breaker to test raw rate limit error
        agent = ExternalFrameworkAgent(
            name="test-rate-limit", base_url="https://api.example.com", enable_circuit_breaker=False
        )

        mock_response = MagicMock()
        mock_response.status = 429
        mock_response.text = AsyncMock(return_value="Rate limit exceeded")
        mock_response.headers = {"Retry-After": "60"}
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        with patch.object(agent, "_get_session", new_callable=AsyncMock, return_value=mock_session):
            with pytest.raises(AgentRateLimitError) as exc_info:
                await agent.generate("Test prompt")

            assert "Rate limited" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handles_connection_error(self):
        """Should raise AgentConnectionError on connection failure."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        # Use unique name and disable circuit breaker
        agent = ExternalFrameworkAgent(
            name="test-conn-error", base_url="https://api.example.com", enable_circuit_breaker=False
        )

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            side_effect=aiohttp.ClientConnectorError(MagicMock(), OSError("Connection refused"))
        )

        with patch.object(agent, "_get_session", new_callable=AsyncMock, return_value=mock_session):
            with pytest.raises(AgentConnectionError) as exc_info:
                await agent.generate("Test prompt")

            assert "Cannot connect" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handles_invalid_json_response(self):
        """Should raise AgentAPIError on invalid JSON response."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        # Use unique name and disable circuit breaker
        agent = ExternalFrameworkAgent(
            name="test-invalid-json",
            base_url="https://api.example.com",
            enable_circuit_breaker=False,
        )

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(side_effect=json.JSONDecodeError("", "", 0))
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        with patch.object(agent, "_get_session", new_callable=AsyncMock, return_value=mock_session):
            with pytest.raises(AgentAPIError) as exc_info:
                await agent.generate("Test prompt")

            assert "invalid JSON" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_failure(self):
        """Should record failure with circuit breaker."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent
        from aragora.resilience import BaseCircuitBreaker

        # Create agent with unique name to get fresh circuit breaker
        agent = ExternalFrameworkAgent(
            name="test-cb-failure",
            base_url="https://api.example.com",
            enable_circuit_breaker=True,
            circuit_breaker_threshold=10,  # High threshold to avoid opening
        )

        # Mock the circuit breaker
        mock_circuit_breaker = MagicMock(spec=BaseCircuitBreaker)
        mock_circuit_breaker.can_proceed.return_value = True
        agent._circuit_breaker = mock_circuit_breaker

        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Server error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        with patch.object(agent, "_get_session", new_callable=AsyncMock, return_value=mock_session):
            with pytest.raises(AgentAPIError):
                await agent.generate("Test prompt")

            mock_circuit_breaker.record_failure.assert_called()


class TestExternalFrameworkTokenUsage:
    """Tests for token usage tracking (inherited from APIAgent base)."""

    def test_token_usage_tracking(self):
        """Should track token usage."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")
        agent._record_token_usage(100, 50)

        usage = agent.get_token_usage()

        assert usage["tokens_in"] == 100
        assert usage["tokens_out"] == 50
        assert usage["total_tokens_in"] == 100
        assert usage["total_tokens_out"] == 50

    def test_reset_token_usage(self):
        """Should reset token counters."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")
        agent._record_token_usage(100, 50)
        agent.reset_token_usage()

        assert agent.last_tokens_in == 0
        assert agent.last_tokens_out == 0
        assert agent.total_tokens_in == 0
        assert agent.total_tokens_out == 0


class TestExternalFrameworkGenerationParams:
    """Tests for generation parameters."""

    def test_set_generation_params(self):
        """Should set generation parameters."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")
        agent.set_generation_params(temperature=0.7, top_p=0.9)

        assert agent.temperature == 0.7
        assert agent.top_p == 0.9

    def test_get_generation_params(self):
        """Should return non-None generation parameters."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")
        agent.temperature = 0.8
        agent.top_p = None

        params = agent.get_generation_params()

        assert "temperature" in params
        assert params["temperature"] == 0.8
        assert "top_p" not in params


class TestCredentialProxyIntegration:
    """Tests for optional CredentialProxy integration."""

    def test_init_without_credential_proxy(self):
        """Default agent should work without credential proxy."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")
        assert agent._credential_proxy is None
        assert agent._credential_id is None

    def test_init_with_credential_proxy(self):
        """Agent should accept credential proxy."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        mock_proxy = MagicMock()
        agent = ExternalFrameworkAgent(
            base_url="https://api.example.com",
            credential_proxy=mock_proxy,
            credential_id="test-cred",
        )
        assert agent._credential_proxy is mock_proxy
        assert agent._credential_id == "test-cred"

    def test_resolve_api_key_from_proxy(self):
        """Should resolve API key from credential proxy."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        mock_cred = MagicMock()
        mock_cred.api_key = "proxy-secret-key"
        mock_cred.is_expired = False
        mock_proxy = MagicMock()
        mock_proxy.get_credential.return_value = mock_cred

        agent = ExternalFrameworkAgent(
            base_url="https://api.example.com",
            credential_proxy=mock_proxy,
            credential_id="test-cred",
        )
        assert agent._resolve_api_key() == "proxy-secret-key"

    def test_resolve_api_key_fallback_to_direct(self):
        """Should fall back to direct api_key if proxy fails."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        mock_proxy = MagicMock()
        mock_proxy.get_credential.side_effect = Exception("Proxy error")

        agent = ExternalFrameworkAgent(
            base_url="https://api.example.com",
            api_key="direct-key",
            credential_proxy=mock_proxy,
            credential_id="test-cred",
        )
        assert agent._resolve_api_key() == "direct-key"

    def test_resolve_api_key_skips_expired_credential(self):
        """Should skip expired credentials from proxy."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        mock_cred = MagicMock()
        mock_cred.is_expired = True
        mock_proxy = MagicMock()
        mock_proxy.get_credential.return_value = mock_cred

        agent = ExternalFrameworkAgent(
            base_url="https://api.example.com",
            api_key="direct-key",
            credential_proxy=mock_proxy,
            credential_id="test-cred",
        )
        assert agent._resolve_api_key() == "direct-key"

    def test_build_headers_uses_proxy_credential(self):
        """Headers should use credential from proxy."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        mock_cred = MagicMock()
        mock_cred.api_key = "proxy-key"
        mock_cred.is_expired = False
        mock_proxy = MagicMock()
        mock_proxy.get_credential.return_value = mock_cred

        agent = ExternalFrameworkAgent(
            base_url="https://api.example.com",
            credential_proxy=mock_proxy,
            credential_id="test-cred",
        )
        headers = agent._build_headers()
        assert "proxy-key" in headers.get("Authorization", "")

    def test_no_proxy_no_credential_id(self):
        """Should work normally without proxy even if credential_id set."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(
            base_url="https://api.example.com",
            credential_id="orphan-id",
        )
        # Without a proxy, _resolve_api_key falls back to self.api_key (None)
        assert agent._resolve_api_key() is None

    def test_proxy_returns_none_credential(self):
        """Should handle None from proxy gracefully."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        mock_proxy = MagicMock()
        mock_proxy.get_credential.return_value = None

        agent = ExternalFrameworkAgent(
            base_url="https://api.example.com",
            api_key="fallback",
            credential_proxy=mock_proxy,
            credential_id="test-cred",
        )
        assert agent._resolve_api_key() == "fallback"


class TestSessionReuse:
    """Tests for HTTP session reuse."""

    def test_session_initially_none(self):
        """Session should be None after initialization."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")
        assert agent._session is None

    @pytest.mark.asyncio
    async def test_get_session_creates_session(self):
        """_get_session should create a new session when none exists."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")
        session = await agent._get_session()
        assert session is not None
        assert not session.closed
        await agent.close()

    @pytest.mark.asyncio
    async def test_get_session_reuses_existing(self):
        """_get_session should return the same session on repeated calls."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")
        session1 = await agent._get_session()
        session2 = await agent._get_session()
        assert session1 is session2
        await agent.close()

    @pytest.mark.asyncio
    async def test_close_cleans_up_session(self):
        """close() should clean up the session and set it to None."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")
        await agent._get_session()
        assert agent._session is not None
        await agent.close()
        assert agent._session is None

    @pytest.mark.asyncio
    async def test_get_session_after_close_creates_new(self):
        """_get_session should create a new session after close."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")
        session1 = await agent._get_session()
        await agent.close()
        session2 = await agent._get_session()
        assert session1 is not session2
        await agent.close()

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        """close() should be safe to call multiple times."""
        from aragora.agents.api_agents.external_framework import ExternalFrameworkAgent

        agent = ExternalFrameworkAgent(base_url="https://api.example.com")
        await agent.close()  # Should not raise
        await agent.close()  # Still should not raise
