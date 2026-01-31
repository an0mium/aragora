"""
Tests for LM Studio API Agent.

Tests cover:
- Initialization and configuration
- Local server communication (is_available, list_models, get_loaded_model)
- generate() method with various inputs
- generate_stream() for streaming responses
- critique() method
- vote() method (inherited from base Agent)
- Error handling (connection errors, API errors, invalid responses)
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from aragora.agents.api_agents.common import (
    AgentAPIError,
    AgentConnectionError,
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
    """Standard OpenAI-compatible API response from LM Studio."""
    return {
        "id": "chatcmpl-lmstudio-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "local-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response from LM Studio.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


@pytest.fixture
def mock_sse_chunks():
    """SSE chunks for streaming response tests (OpenAI format)."""
    return [
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"!"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]


@pytest.fixture
def mock_models_response():
    """Mock response for /models endpoint."""
    return {
        "object": "list",
        "data": [
            {
                "id": "llama-3.2-8b-instruct",
                "object": "model",
                "owned_by": "meta",
            },
            {
                "id": "mistral-7b-instruct",
                "object": "model",
                "owned_by": "mistralai",
            },
        ],
    }


class TestLMStudioAgentInitialization:
    """Tests for agent initialization."""

    def test_init_with_defaults(self):
        """Should initialize with default values."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

        assert agent.name == "lm-studio"
        assert agent.model == "local-model"
        assert agent.role == "proposer"
        assert agent.timeout == 180
        assert agent.agent_type == "lm-studio"
        assert agent.base_url == "http://localhost:1234/v1"
        assert agent.max_tokens == 4096

    def test_init_with_custom_config(self):
        """Should initialize with custom configuration."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent(
            name="custom-lm-studio",
            model="codellama-34b",
            role="critic",
            timeout=300,
            base_url="http://192.168.1.100:1234",
            max_tokens=8192,
        )

        assert agent.name == "custom-lm-studio"
        assert agent.model == "codellama-34b"
        assert agent.role == "critic"
        assert agent.timeout == 300
        assert agent.base_url == "http://192.168.1.100:1234/v1"
        assert agent.max_tokens == 8192

    def test_init_with_lm_studio_host_env_var(self, monkeypatch):
        """Should use LM_STUDIO_HOST environment variable when base_url not provided."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        monkeypatch.setenv("LM_STUDIO_HOST", "http://remote-lmstudio:5000")

        agent = LMStudioAgent()

        assert agent.base_url == "http://remote-lmstudio:5000/v1"

    def test_init_base_url_overrides_env_var(self, monkeypatch):
        """Should prefer explicit base_url over environment variable."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        monkeypatch.setenv("LM_STUDIO_HOST", "http://remote-lmstudio:5000")

        agent = LMStudioAgent(base_url="http://explicit-host:1234")

        assert agent.base_url == "http://explicit-host:1234/v1"

    def test_init_adds_v1_suffix_if_missing(self):
        """Should add /v1 suffix to base_url if not present."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent(base_url="http://localhost:1234")
        assert agent.base_url == "http://localhost:1234/v1"

        agent2 = LMStudioAgent(base_url="http://localhost:1234/")
        assert agent2.base_url == "http://localhost:1234/v1"

    def test_init_preserves_v1_suffix_if_present(self):
        """Should not duplicate /v1 suffix."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent(base_url="http://localhost:1234/v1")
        assert agent.base_url == "http://localhost:1234/v1"

    def test_agent_registry_registration(self):
        """Should be registered in agent registry."""
        from aragora.agents.registry import AgentRegistry

        spec = AgentRegistry.get_spec("lm-studio")

        assert spec is not None
        assert spec.default_model == "local-model"
        assert spec.agent_type == "API"


class TestLMStudioServerCommunication:
    """Tests for local server communication methods."""

    @pytest.mark.asyncio
    async def test_is_available_returns_true_when_server_running(self):
        """Should return True when LM Studio server responds successfully."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.lm_studio.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.is_available()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_available_returns_false_when_server_not_running(self):
        """Should return False when LM Studio server is not accessible."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=aiohttp.ClientError())
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.lm_studio.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.is_available()

        assert result is False

    @pytest.mark.asyncio
    async def test_is_available_returns_false_on_timeout(self):
        """Should return False when connection times out."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=asyncio.TimeoutError())
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.lm_studio.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.is_available()

        assert result is False

    @pytest.mark.asyncio
    async def test_is_available_returns_false_on_os_error(self):
        """Should return False on OS-level connection errors."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=OSError("Connection refused"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.lm_studio.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.is_available()

        assert result is False

    @pytest.mark.asyncio
    async def test_list_models_returns_models_list(self, mock_models_response):
        """Should return list of available models."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_models_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.lm_studio.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.list_models()

        assert len(result) == 2
        assert result[0]["id"] == "llama-3.2-8b-instruct"
        assert result[1]["id"] == "mistral-7b-instruct"

    @pytest.mark.asyncio
    async def test_list_models_returns_empty_on_error(self):
        """Should return empty list when server returns error."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.lm_studio.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.list_models()

        assert result == []

    @pytest.mark.asyncio
    async def test_list_models_handles_connection_error(self):
        """Should return empty list on connection error."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=aiohttp.ClientError())
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.lm_studio.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.list_models()

        assert result == []

    @pytest.mark.asyncio
    async def test_list_models_handles_invalid_json(self):
        """Should return empty list on JSON parse error."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(side_effect=ValueError("Invalid JSON"))
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.lm_studio.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.list_models()

        assert result == []

    @pytest.mark.asyncio
    async def test_get_loaded_model_returns_first_model(self, mock_models_response):
        """Should return the first model ID from the list."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_models_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.lm_studio.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.get_loaded_model()

        assert result == "llama-3.2-8b-instruct"

    @pytest.mark.asyncio
    async def test_get_loaded_model_returns_none_when_no_models(self):
        """Should return None when no models are loaded."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"object": "list", "data": []})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.lm_studio.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.get_loaded_model()

        assert result is None


class TestLMStudioGenerate:
    """Tests for generate method."""

    @pytest.mark.asyncio
    async def test_generate_basic_response(self, mock_openai_response):
        """Should generate response from LM Studio API."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_openai_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.lm_studio.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.generate("Test prompt")

        assert "test response from LM Studio" in result

    @pytest.mark.asyncio
    async def test_generate_with_context(self, sample_context, mock_openai_response):
        """Should include context in messages."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

        captured_payload = {}

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_openai_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        def capture_post(url, json=None):
            captured_payload["json"] = json
            return mock_response

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=capture_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.lm_studio.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.generate("Test prompt", context=sample_context)

        assert result is not None
        # Verify context messages were included (2 context messages + 1 user prompt)
        assert len(captured_payload["json"]["messages"]) == 3

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self, mock_openai_response):
        """Should include system prompt when set."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()
        agent.system_prompt = "You are a helpful coding assistant."

        captured_payload = {}

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_openai_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        def capture_post(url, json=None):
            captured_payload["json"] = json
            return mock_response

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=capture_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.lm_studio.create_client_session",
            return_value=mock_session,
        ):
            await agent.generate("Test prompt")

        # Verify system message was included
        messages = captured_payload["json"]["messages"]
        assert messages[0]["role"] == "system"
        assert "helpful coding assistant" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_generate_returns_empty_string_on_missing_choices(self):
        """Should return empty string when choices are empty."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

        empty_response = {"id": "test", "choices": []}

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=empty_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.lm_studio.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.generate("Test prompt")

        assert result == ""

    @pytest.mark.asyncio
    async def test_generate_includes_max_tokens(self, mock_openai_response):
        """Should include max_tokens in payload."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent(max_tokens=2048)

        captured_payload = {}

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_openai_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        def capture_post(url, json=None):
            captured_payload["json"] = json
            return mock_response

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=capture_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.lm_studio.create_client_session",
            return_value=mock_session,
        ):
            await agent.generate("Test prompt")

        assert captured_payload["json"]["max_tokens"] == 2048

    @pytest.mark.asyncio
    async def test_generate_uses_correct_endpoint(self, mock_openai_response):
        """Should POST to /chat/completions endpoint."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent(base_url="http://test-host:1234")

        captured_url = None

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_openai_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        def capture_post(url, json=None):
            nonlocal captured_url
            captured_url = url
            return mock_response

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=capture_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.lm_studio.create_client_session",
            return_value=mock_session,
        ):
            await agent.generate("Test prompt")

        assert captured_url == "http://test-host:1234/v1/chat/completions"


class TestLMStudioGenerateStream:
    """Tests for streaming generation."""

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self, mock_sse_chunks):
        """Should yield text chunks from streaming response."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

        class MockContent:
            def __init__(self, chunks):
                self._chunks = chunks
                self._index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._index >= len(self._chunks):
                    raise StopAsyncIteration
                chunk = self._chunks[self._index]
                self._index += 1
                return chunk

            def iter_any(self):
                """Return async iterator for iter_chunks_with_timeout compatibility."""
                return self

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content = MockContent(mock_sse_chunks)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.lm_studio.create_client_session",
            return_value=mock_session,
        ):
            chunks = []
            async for chunk in agent.generate_stream("Test prompt"):
                chunks.append(chunk)

        assert len(chunks) == 3
        assert "".join(chunks) == "Hello world!"

    @pytest.mark.asyncio
    async def test_stream_with_system_prompt(self, mock_sse_chunks):
        """Should include system prompt in streaming request."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()
        agent.system_prompt = "Be concise."

        captured_payload = {}

        class MockContent:
            def __init__(self, chunks):
                self._chunks = chunks
                self._index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._index >= len(self._chunks):
                    raise StopAsyncIteration
                chunk = self._chunks[self._index]
                self._index += 1
                return chunk

            def iter_any(self):
                """Return async iterator for iter_chunks_with_timeout compatibility."""
                return self

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content = MockContent(mock_sse_chunks)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        def capture_post(url, json=None):
            captured_payload["json"] = json
            return mock_response

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=capture_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.lm_studio.create_client_session",
            return_value=mock_session,
        ):
            chunks = []
            async for chunk in agent.generate_stream("Test prompt"):
                chunks.append(chunk)

        assert captured_payload["json"]["stream"] is True
        assert captured_payload["json"]["messages"][0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_stream_handles_malformed_json(self):
        """Should skip malformed JSON lines gracefully."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

        stream_chunks = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
            b"data: malformed json line\n\n",
            b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]

        class MockContent:
            def __init__(self, chunks):
                self._chunks = chunks
                self._index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._index >= len(self._chunks):
                    raise StopAsyncIteration
                chunk = self._chunks[self._index]
                self._index += 1
                return chunk

            def iter_any(self):
                """Return async iterator for iter_chunks_with_timeout compatibility."""
                return self

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content = MockContent(stream_chunks)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.lm_studio.create_client_session",
            return_value=mock_session,
        ):
            chunks = []
            async for chunk in agent.generate_stream("Test prompt"):
                chunks.append(chunk)

        # Should have received valid chunks despite malformed line
        assert "".join(chunks) == "Hello world"

    @pytest.mark.asyncio
    async def test_stream_skips_empty_lines(self):
        """Should skip empty lines in stream."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

        stream_chunks = [
            b"\n",
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
            b"   \n",
            b"data: [DONE]\n\n",
        ]

        class MockContent:
            def __init__(self, chunks):
                self._chunks = chunks
                self._index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._index >= len(self._chunks):
                    raise StopAsyncIteration
                chunk = self._chunks[self._index]
                self._index += 1
                return chunk

            def iter_any(self):
                """Return async iterator for iter_chunks_with_timeout compatibility."""
                return self

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content = MockContent(stream_chunks)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.lm_studio.create_client_session",
            return_value=mock_session,
        ):
            chunks = []
            async for chunk in agent.generate_stream("Test prompt"):
                chunks.append(chunk)

        assert chunks == ["Hello"]


class TestLMStudioCritique:
    """Tests for critique method."""

    @pytest.mark.asyncio
    async def test_critique_returns_structured_feedback(self):
        """Should return structured critique."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

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
            assert critique.agent == "lm-studio"
            assert critique.target_agent == "test-agent"
            assert len(critique.issues) > 0
            assert len(critique.suggestions) > 0

    @pytest.mark.asyncio
    async def test_critique_with_context(self, sample_context):
        """Should include context when critiquing."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

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
            # Verify generate was called with context
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args
            assert call_args[0][1] == sample_context

    @pytest.mark.asyncio
    async def test_critique_without_target_agent(self):
        """Should handle critique without explicit target agent."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

        with patch.object(agent, "generate", new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = """ISSUES:
- Generic issue

SEVERITY: 3.0
REASONING: Minor concerns."""

            critique = await agent.critique(
                proposal="Some proposal",
                task="Some task",
            )

            assert critique is not None
            assert critique.target_agent == "proposal"

    @pytest.mark.asyncio
    async def test_critique_prompt_includes_task_and_proposal(self):
        """Should include task and proposal in critique prompt."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

        captured_prompt = None

        async def capture_generate(prompt, context=None):
            nonlocal captured_prompt
            captured_prompt = prompt
            return "ISSUES:\n- Issue\nSEVERITY: 5.0\nREASONING: reason"

        with patch.object(agent, "generate", side_effect=capture_generate):
            await agent.critique(
                proposal="My test proposal",
                task="Implement feature X",
                target_agent="agent1",
            )

        assert "Implement feature X" in captured_prompt
        assert "My test proposal" in captured_prompt
        assert "agent1" in captured_prompt


class TestLMStudioVote:
    """Tests for vote method (inherited from base Agent)."""

    @pytest.mark.asyncio
    async def test_vote_selects_best_proposal(self):
        """Should vote for best proposal."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

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
            assert vote.agent == "lm-studio"
            assert vote.choice == "agent2"
            assert vote.confidence == 0.85
            assert vote.continue_debate is False

    @pytest.mark.asyncio
    async def test_vote_with_continue_debate_yes(self):
        """Should indicate debate should continue when appropriate."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

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
    async def test_vote_handles_default_confidence(self):
        """Should use default confidence when parsing fails."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

        proposals = {"agent1": "Proposal A"}

        with patch.object(agent, "generate", new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = """CHOICE: agent1
CONFIDENCE: invalid
CONTINUE: no
REASONING: Some reasoning."""

            vote = await agent.vote(proposals, "Test task")

            assert vote.confidence == 0.5  # Default confidence


class TestLMStudioErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_handles_api_error(self):
        """Should raise AgentAPIError on API failure."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

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
            "aragora.agents.api_agents.lm_studio.create_client_session",
            return_value=mock_session,
        ):
            with pytest.raises(AgentAPIError) as exc_info:
                await agent.generate("Test prompt")

            assert "500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handles_connection_error(self):
        """Should raise AgentConnectionError when LM Studio is not running."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            side_effect=aiohttp.ClientConnectorError(MagicMock(), OSError("Connection refused"))
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.lm_studio.create_client_session",
            return_value=mock_session,
        ):
            with pytest.raises(AgentConnectionError) as exc_info:
                await agent.generate("Test prompt")

            assert "Cannot connect to LM Studio" in str(exc_info.value)
            assert "model loaded" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handles_invalid_json_response(self):
        """Should raise AgentAPIError on invalid JSON response."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(side_effect=json.JSONDecodeError("", "", 0))
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.lm_studio.create_client_session",
            return_value=mock_session,
        ):
            with pytest.raises(AgentAPIError) as exc_info:
                await agent.generate("Test prompt")

            assert "invalid JSON" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handles_content_type_error(self):
        """Should raise AgentAPIError on content type error."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            side_effect=aiohttp.ContentTypeError(MagicMock(), MagicMock())
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.lm_studio.create_client_session",
            return_value=mock_session,
        ):
            with pytest.raises(AgentAPIError) as exc_info:
                await agent.generate("Test prompt")

            assert "invalid JSON" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stream_handles_api_error(self):
        """Should raise AgentAPIError on streaming API failure."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

        mock_response = MagicMock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value="Bad request")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.lm_studio.create_client_session",
            return_value=mock_session,
        ):
            with pytest.raises(AgentAPIError):
                async for _ in agent.generate_stream("Test prompt"):
                    pass

    @pytest.mark.asyncio
    async def test_stream_handles_connection_error(self):
        """Should raise AgentConnectionError on streaming connection failure."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            side_effect=aiohttp.ClientConnectorError(MagicMock(), OSError("Connection refused"))
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.lm_studio.create_client_session",
            return_value=mock_session,
        ):
            with pytest.raises(AgentConnectionError):
                async for _ in agent.generate_stream("Test prompt"):
                    pass


class TestLMStudioTokenUsage:
    """Tests for token usage tracking (inherited from APIAgent base)."""

    def test_token_usage_tracking(self):
        """Should track token usage."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()
        agent._record_token_usage(100, 50)

        usage = agent.get_token_usage()

        assert usage["tokens_in"] == 100
        assert usage["tokens_out"] == 50
        assert usage["total_tokens_in"] == 100
        assert usage["total_tokens_out"] == 50

    def test_reset_token_usage(self):
        """Should reset token counters."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()
        agent._record_token_usage(100, 50)
        agent.reset_token_usage()

        assert agent.last_tokens_in == 0
        assert agent.last_tokens_out == 0
        assert agent.total_tokens_in == 0
        assert agent.total_tokens_out == 0

    def test_accumulates_token_usage(self):
        """Should accumulate total token usage across calls."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()
        agent._record_token_usage(100, 50)
        agent._record_token_usage(200, 100)

        assert agent.last_tokens_in == 200
        assert agent.last_tokens_out == 100
        assert agent.total_tokens_in == 300
        assert agent.total_tokens_out == 150


class TestLMStudioGenerationParams:
    """Tests for generation parameters."""

    def test_set_generation_params(self):
        """Should set generation parameters."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()
        agent.set_generation_params(temperature=0.7, top_p=0.9)

        assert agent.temperature == 0.7
        assert agent.top_p == 0.9

    def test_get_generation_params(self):
        """Should return non-None generation parameters."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        agent = LMStudioAgent()
        agent.temperature = 0.8
        agent.top_p = None

        params = agent.get_generation_params()

        assert "temperature" in params
        assert params["temperature"] == 0.8
        assert "top_p" not in params


class TestLMStudioModuleExports:
    """Tests for module exports."""

    def test_exports_lm_studio_agent(self):
        """Should export LMStudioAgent."""
        from aragora.agents.api_agents.lm_studio import __all__

        assert "LMStudioAgent" in __all__

    def test_agent_can_be_imported(self):
        """Should be importable from the module."""
        from aragora.agents.api_agents.lm_studio import LMStudioAgent

        assert LMStudioAgent is not None
