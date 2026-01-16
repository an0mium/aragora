"""
Tests for aragora.agents.api_agents.ollama module.

Tests Ollama agent with mocked HTTP responses.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from aragora.agents.api_agents.ollama import OllamaAgent
from aragora.agents.errors import AgentAPIError, AgentConnectionError
from aragora.core import Message


class TestOllamaAgentInit:
    """Test OllamaAgent initialization."""

    def test_default_initialization(self):
        """Test agent with default parameters."""
        agent = OllamaAgent()
        assert agent.name == "ollama"
        assert agent.model == "llama3.2"
        assert agent.role == "proposer"
        assert agent.timeout == 180
        assert agent.base_url == "http://localhost:11434"

    def test_custom_name_and_model(self):
        """Test agent with custom name and model."""
        agent = OllamaAgent(name="my-ollama", model="codellama")
        assert agent.name == "my-ollama"
        assert agent.model == "codellama"

    def test_custom_base_url(self):
        """Test agent with custom base URL."""
        agent = OllamaAgent(base_url="http://192.168.1.100:11434")
        assert agent.base_url == "http://192.168.1.100:11434"

    def test_env_var_base_url(self):
        """Test base URL from environment variable."""
        with patch.dict("os.environ", {"OLLAMA_HOST": "http://custom:5000"}):
            agent = OllamaAgent()
            assert agent.base_url == "http://custom:5000"


class TestOllamaIsAvailable:
    """Test is_available method."""

    @pytest.mark.asyncio
    async def test_available_when_server_responds(self):
        """Test returns True when server is accessible."""
        agent = OllamaAgent()

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
            # Verify it checked /api/tags endpoint
            mock_session.get.assert_called_once()
            call_args = mock_session.get.call_args
            assert "/api/tags" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_not_available_when_connection_fails(self):
        """Test returns False when connection fails."""
        agent = OllamaAgent()

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=aiohttp.ClientError())
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await agent.is_available()
            assert result is False


class TestOllamaListModels:
    """Test list_models method."""

    @pytest.mark.asyncio
    async def test_list_models_success(self):
        """Test listing models successfully."""
        agent = OllamaAgent()

        models_response = {
            "models": [
                {"name": "llama3.2:latest", "size": 4000000000, "modified_at": "2024-01-01"},
                {"name": "codellama:latest", "size": 3500000000, "modified_at": "2024-01-02"},
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
            assert models[0]["name"] == "llama3.2:latest"

    @pytest.mark.asyncio
    async def test_list_models_empty(self):
        """Test listing when no models available."""
        agent = OllamaAgent()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"models": []})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            models = await agent.list_models()
            assert models == []


class TestOllamaModelInfo:
    """Test model_info method."""

    @pytest.mark.asyncio
    async def test_model_info_success(self):
        """Test getting model info successfully."""
        agent = OllamaAgent(model="llama3.2")

        info_response = {
            "modelfile": "FROM llama3.2",
            "parameters": "temperature 0.8",
            "template": "{{ .Prompt }}",
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=info_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            info = await agent.model_info()
            assert info["modelfile"] == "FROM llama3.2"

    @pytest.mark.asyncio
    async def test_model_info_not_found(self):
        """Test model info when model not found."""
        agent = OllamaAgent()

        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            info = await agent.model_info()
            assert info == {}


class TestOllamaGenerate:
    """Test generate method."""

    @pytest.mark.asyncio
    async def test_generate_success(self):
        """Test successful generation."""
        agent = OllamaAgent()

        api_response = {"response": "Hello, I am Ollama!"}

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
            assert result == "Hello, I am Ollama!"

            # Verify request payload
            call_args = mock_session.post.call_args
            payload = call_args[1]["json"]
            assert payload["model"] == "llama3.2"
            assert payload["stream"] is False
            assert "Say hello" in payload["prompt"]

    @pytest.mark.asyncio
    async def test_generate_with_context(self):
        """Test generation with conversation context."""
        agent = OllamaAgent()

        api_response = {"response": "Continued response"}

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
            Message(role="assistant", agent="ollama", content="Hi there!"),
        ]

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await agent.generate("Continue", context=context)
            assert result == "Continued response"

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self):
        """Test generation includes system prompt."""
        agent = OllamaAgent()
        agent.system_prompt = "You are a helpful assistant."

        api_response = {"response": "Response"}

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
            assert "System context" in payload["prompt"]
            assert "helpful assistant" in payload["prompt"]

    @pytest.mark.asyncio
    async def test_generate_api_error(self):
        """Test handling of API errors."""
        agent = OllamaAgent()

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
        agent = OllamaAgent()

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
            assert "Ollama" in str(exc_info.value)


class TestOllamaGenerateStream:
    """Test generate_stream method."""

    @pytest.mark.asyncio
    async def test_stream_success(self):
        """Test successful streaming generation."""
        agent = OllamaAgent()

        # Simulate Ollama streaming format (JSON per line)
        stream_lines = [
            b'{"response": "Hello"}\n',
            b'{"response": " world"}\n',
            b'{"response": "!", "done": true}\n',
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

            assert chunks == ["Hello", " world", "!"]

    @pytest.mark.asyncio
    async def test_stream_stops_on_done(self):
        """Test streaming stops when done flag is received."""
        agent = OllamaAgent()

        stream_lines = [
            b'{"response": "Complete"}\n',
            b'{"done": true}\n',
            b'{"response": "Should not appear"}\n',
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

            assert chunks == ["Complete"]


class TestOllamaPullModel:
    """Test pull_model method."""

    @pytest.mark.asyncio
    async def test_pull_model_progress(self):
        """Test model pull with progress updates."""
        agent = OllamaAgent()

        progress_lines = [
            b'{"status": "pulling manifest"}\n',
            b'{"status": "downloading", "completed": 50, "total": 100}\n',
            b'{"status": "success"}\n',
        ]

        async def mock_content_iter():
            for line in progress_lines:
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
            updates = []
            async for update in agent.pull_model("llama3.2"):
                updates.append(update)

            assert len(updates) == 3
            assert updates[0]["status"] == "pulling manifest"
            assert updates[1]["completed"] == 50
            assert updates[2]["status"] == "success"


class TestOllamaCritique:
    """Test critique method."""

    @pytest.mark.asyncio
    async def test_critique_generates_structured_response(self):
        """Test critique calls generate with proper prompt."""
        agent = OllamaAgent()

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

            mock_gen.assert_called_once()
            call_prompt = mock_gen.call_args[0][0]
            assert "Test proposal" in call_prompt
            assert "Test task" in call_prompt

            assert critique is not None


class TestOllamaRegistration:
    """Test agent registry integration."""

    def test_registered_in_registry(self):
        """Test agent is registered with correct metadata."""
        from aragora.agents.registry import AgentRegistry

        assert AgentRegistry.is_registered("ollama")

        spec = AgentRegistry.get_spec("ollama")
        assert spec is not None
        assert spec.agent_type == "API"
        assert "Ollama" in spec.requires
        assert spec.default_model == "llama3.2"
