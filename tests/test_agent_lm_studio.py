"""
Tests for aragora.agents.api_agents.lm_studio module.

Tests LM Studio agent with mocked HTTP responses.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from aragora.agents.api_agents.lm_studio import LMStudioAgent
from aragora.agents.errors import AgentAPIError, AgentConnectionError
from aragora.core import Message


class TestLMStudioAgentInit:
    """Test LMStudioAgent initialization."""

    def test_default_initialization(self):
        """Test agent with default parameters."""
        agent = LMStudioAgent()
        assert agent.name == "lm-studio"
        assert agent.model == "local-model"
        assert agent.role == "proposer"
        assert agent.timeout == 180
        assert agent.base_url == "http://localhost:1234/v1"
        assert agent.max_tokens == 4096

    def test_custom_name_and_model(self):
        """Test agent with custom name and model."""
        agent = LMStudioAgent(name="my-agent", model="llama-2-7b")
        assert agent.name == "my-agent"
        assert agent.model == "llama-2-7b"

    def test_custom_base_url(self):
        """Test agent with custom base URL."""
        agent = LMStudioAgent(base_url="http://192.168.1.100:8080")
        assert agent.base_url == "http://192.168.1.100:8080/v1"

    def test_base_url_with_v1_suffix(self):
        """Test base URL already has /v1 suffix."""
        agent = LMStudioAgent(base_url="http://localhost:1234/v1")
        assert agent.base_url == "http://localhost:1234/v1"

    def test_base_url_with_trailing_slash(self):
        """Test base URL with trailing slash."""
        agent = LMStudioAgent(base_url="http://localhost:1234/")
        assert agent.base_url == "http://localhost:1234/v1"

    def test_env_var_base_url(self):
        """Test base URL from environment variable."""
        with patch.dict("os.environ", {"LM_STUDIO_HOST": "http://custom:5000"}):
            agent = LMStudioAgent()
            assert agent.base_url == "http://custom:5000/v1"

    def test_custom_max_tokens(self):
        """Test agent with custom max tokens."""
        agent = LMStudioAgent(max_tokens=8192)
        assert agent.max_tokens == 8192


class TestLMStudioIsAvailable:
    """Test is_available method."""

    @pytest.mark.asyncio
    async def test_available_when_server_responds(self):
        """Test returns True when server is accessible."""
        agent = LMStudioAgent()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await agent.is_available()
            assert result is True

    @pytest.mark.asyncio
    async def test_not_available_when_connection_fails(self):
        """Test returns False when connection fails."""
        agent = LMStudioAgent()

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=aiohttp.ClientError())
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await agent.is_available()
            assert result is False

    @pytest.mark.asyncio
    async def test_not_available_when_server_errors(self):
        """Test returns False when server returns error status."""
        agent = LMStudioAgent()

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await agent.is_available()
            assert result is False


class TestLMStudioListModels:
    """Test list_models method."""

    @pytest.mark.asyncio
    async def test_list_models_success(self):
        """Test listing models successfully."""
        agent = LMStudioAgent()

        models_response = {
            "data": [
                {"id": "llama-2-7b", "object": "model", "owned_by": "local"},
                {"id": "mistral-7b", "object": "model", "owned_by": "local"},
            ]
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=models_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            models = await agent.list_models()
            assert len(models) == 2
            assert models[0]["id"] == "llama-2-7b"
            assert models[1]["id"] == "mistral-7b"

    @pytest.mark.asyncio
    async def test_list_models_empty(self):
        """Test listing when no models loaded."""
        agent = LMStudioAgent()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": []})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            models = await agent.list_models()
            assert models == []

    @pytest.mark.asyncio
    async def test_list_models_server_error(self):
        """Test listing when server errors returns empty list."""
        agent = LMStudioAgent()

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            models = await agent.list_models()
            assert models == []


class TestLMStudioGetLoadedModel:
    """Test get_loaded_model method."""

    @pytest.mark.asyncio
    async def test_get_loaded_model_success(self):
        """Test getting the currently loaded model."""
        agent = LMStudioAgent()

        with patch.object(
            agent, "list_models", return_value=[{"id": "llama-2-7b"}, {"id": "other"}]
        ):
            model_id = await agent.get_loaded_model()
            assert model_id == "llama-2-7b"

    @pytest.mark.asyncio
    async def test_get_loaded_model_none_loaded(self):
        """Test when no model is loaded."""
        agent = LMStudioAgent()

        with patch.object(agent, "list_models", return_value=[]):
            model_id = await agent.get_loaded_model()
            assert model_id is None


class TestLMStudioGenerate:
    """Test generate method."""

    @pytest.mark.asyncio
    async def test_generate_success(self):
        """Test successful generation."""
        agent = LMStudioAgent()

        api_response = {"choices": [{"message": {"content": "Hello, I am an AI assistant."}}]}

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=api_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await agent.generate("Say hello")
            assert result == "Hello, I am an AI assistant."

    @pytest.mark.asyncio
    async def test_generate_with_context(self):
        """Test generation with conversation context."""
        agent = LMStudioAgent()

        api_response = {"choices": [{"message": {"content": "Continued response"}}]}

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=api_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        context = [
            Message(role="user", agent="user", content="Hello"),
            Message(role="assistant", agent="lm-studio", content="Hi there!"),
        ]

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await agent.generate("Continue", context=context)
            assert result == "Continued response"

            # Verify the request included context
            call_args = mock_session.post.call_args
            payload = call_args[1]["json"]
            assert len(payload["messages"]) >= 3  # context + prompt

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self):
        """Test generation includes system prompt."""
        agent = LMStudioAgent()
        agent.system_prompt = "You are a helpful assistant."

        api_response = {"choices": [{"message": {"content": "Response"}}]}

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=api_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await agent.generate("Test")

            call_args = mock_session.post.call_args
            payload = call_args[1]["json"]
            assert payload["messages"][0]["role"] == "system"
            assert payload["messages"][0]["content"] == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_generate_empty_choices(self):
        """Test handling of empty choices array."""
        agent = LMStudioAgent()

        api_response = {"choices": []}

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=api_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await agent.generate("Test")
            assert result == ""

    @pytest.mark.asyncio
    async def test_generate_api_error(self):
        """Test handling of API errors."""
        agent = LMStudioAgent()

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(AgentAPIError) as exc_info:
                await agent.generate("Test")
            assert "500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_connection_error(self):
        """Test handling of connection errors."""
        agent = LMStudioAgent()

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            side_effect=aiohttp.ClientConnectorError(
                connection_key=MagicMock(), os_error=OSError("Connection refused")
            )
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(AgentConnectionError) as exc_info:
                await agent.generate("Test")
            assert "LM Studio" in str(exc_info.value)


class TestLMStudioGenerateStream:
    """Test generate_stream method."""

    @pytest.mark.asyncio
    async def test_stream_success(self):
        """Test successful streaming generation."""
        agent = LMStudioAgent()

        # Simulate SSE stream
        stream_lines = [
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n',
            b'data: {"choices": [{"delta": {"content": " world"}}]}\n',
            b"data: [DONE]\n",
        ]

        async def mock_content_iter():
            for line in stream_lines:
                yield line

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.content = mock_content_iter()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            chunks = []
            async for chunk in agent.generate_stream("Test"):
                chunks.append(chunk)

            assert chunks == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_skips_malformed_json(self):
        """Test streaming skips malformed JSON lines."""
        agent = LMStudioAgent()

        stream_lines = [
            b'data: {"choices": [{"delta": {"content": "Valid"}}]}\n',
            b"data: {invalid json}\n",
            b'data: {"choices": [{"delta": {"content": "Also valid"}}]}\n',
            b"data: [DONE]\n",
        ]

        async def mock_content_iter():
            for line in stream_lines:
                yield line

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.content = mock_content_iter()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            chunks = []
            async for chunk in agent.generate_stream("Test"):
                chunks.append(chunk)

            assert chunks == ["Valid", "Also valid"]


class TestLMStudioCritique:
    """Test critique method."""

    @pytest.mark.asyncio
    async def test_critique_generates_structured_response(self):
        """Test critique calls generate with proper prompt."""
        agent = LMStudioAgent()

        critique_response = """ISSUES:
- Issue 1

SUGGESTIONS:
- Suggestion 1

SEVERITY: 0.5
REASONING: Test reasoning"""

        with patch.object(agent, "generate", return_value=critique_response) as mock_gen:
            critique = await agent.critique(
                proposal="Test proposal",
                task="Test task",
                target_agent="other-agent",
            )

            # Verify generate was called
            mock_gen.assert_called_once()
            call_prompt = mock_gen.call_args[0][0]
            assert "Test proposal" in call_prompt
            assert "Test task" in call_prompt
            assert "other-agent" in call_prompt

            # Verify critique object returned
            assert critique is not None


class TestLMStudioRegistration:
    """Test agent registry integration."""

    def test_registered_in_registry(self):
        """Test agent is registered with correct metadata."""
        from aragora.agents.registry import AgentRegistry

        # Check if registered
        assert AgentRegistry.is_registered("lm-studio")

        # Get spec and verify
        spec = AgentRegistry.get_spec("lm-studio")
        assert spec is not None
        assert spec.agent_type == "API"
        assert "LM Studio" in spec.requires
