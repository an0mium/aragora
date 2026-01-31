"""
Tests for Ollama API Agent.

Tests cover:
- Initialization and configuration
- Local server communication (is_available, list_models, model_info)
- generate() method with various inputs
- generate_stream() for streaming responses
- critique() method
- vote() method (inherited from base Agent)
- pull_model() for model management
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
def mock_ollama_response():
    """Standard Ollama API response."""
    return {
        "model": "llama3.2",
        "response": "This is a test response from Ollama.",
        "done": True,
    }


@pytest.fixture
def mock_ollama_stream_chunks():
    """Streaming response chunks for Ollama (NDJSON format)."""
    return [
        b'{"model":"llama3.2","response":"Hello"}\n',
        b'{"model":"llama3.2","response":" world"}\n',
        b'{"model":"llama3.2","response":"!"}\n',
        b'{"model":"llama3.2","response":"","done":true}\n',
    ]


class TestOllamaAgentInitialization:
    """Tests for agent initialization."""

    def test_init_with_defaults(self):
        """Should initialize with default values."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()

        assert agent.name == "ollama"
        assert agent.model == "llama3.2"
        assert agent.role == "proposer"
        assert agent.timeout == 180
        assert agent.agent_type == "ollama"
        assert agent.base_url == "http://localhost:11434"

    def test_init_with_custom_config(self):
        """Should initialize with custom configuration."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent(
            name="custom-ollama",
            model="codellama",
            role="critic",
            timeout=300,
            base_url="http://192.168.1.100:11434",
        )

        assert agent.name == "custom-ollama"
        assert agent.model == "codellama"
        assert agent.role == "critic"
        assert agent.timeout == 300
        assert agent.base_url == "http://192.168.1.100:11434"

    def test_init_with_ollama_host_env_var(self, monkeypatch):
        """Should use OLLAMA_HOST environment variable when base_url not provided."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        monkeypatch.setenv("OLLAMA_HOST", "http://remote-ollama:11434")

        agent = OllamaAgent()

        assert agent.base_url == "http://remote-ollama:11434"

    def test_init_base_url_overrides_env_var(self, monkeypatch):
        """Should prefer explicit base_url over environment variable."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        monkeypatch.setenv("OLLAMA_HOST", "http://remote-ollama:11434")

        agent = OllamaAgent(base_url="http://explicit-host:11434")

        assert agent.base_url == "http://explicit-host:11434"

    def test_agent_registry_registration(self):
        """Should be registered in agent registry."""
        from aragora.agents.registry import AgentRegistry

        spec = AgentRegistry.get_spec("ollama")

        assert spec is not None
        assert spec.default_model == "llama3.2"
        assert spec.agent_type == "API"


class TestOllamaServerCommunication:
    """Tests for local server communication methods."""

    @pytest.mark.asyncio
    async def test_is_available_returns_true_when_server_running(self):
        """Should return True when Ollama server responds successfully."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.ollama.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.is_available()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_available_returns_false_when_server_not_running(self):
        """Should return False when Ollama server is not accessible."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=aiohttp.ClientError())
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.ollama.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.is_available()

        assert result is False

    @pytest.mark.asyncio
    async def test_is_available_returns_false_on_timeout(self):
        """Should return False when connection times out."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=asyncio.TimeoutError())
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.ollama.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.is_available()

        assert result is False

    @pytest.mark.asyncio
    async def test_list_models_returns_models_list(self):
        """Should return list of available models."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()

        models_response = {
            "models": [
                {"name": "llama3.2:latest", "size": 4700000000, "modified_at": "2024-01-15"},
                {"name": "codellama:latest", "size": 3800000000, "modified_at": "2024-01-14"},
            ]
        }

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=models_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.ollama.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.list_models()

        assert len(result) == 2
        assert result[0]["name"] == "llama3.2:latest"
        assert result[1]["name"] == "codellama:latest"

    @pytest.mark.asyncio
    async def test_list_models_returns_empty_on_error(self):
        """Should return empty list when server returns error."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()

        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.ollama.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.list_models()

        assert result == []

    @pytest.mark.asyncio
    async def test_list_models_handles_connection_error(self):
        """Should return empty list on connection error."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=aiohttp.ClientError())
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.ollama.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.list_models()

        assert result == []

    @pytest.mark.asyncio
    async def test_model_info_returns_model_details(self):
        """Should return model details."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()

        model_info_response = {
            "license": "Meta License",
            "modelfile": "FROM llama3.2",
            "parameters": "stop [/INST]",
            "template": "{{ .System }}\n{{ .Prompt }}",
        }

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=model_info_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.ollama.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.model_info()

        assert "license" in result
        assert "template" in result

    @pytest.mark.asyncio
    async def test_model_info_returns_empty_on_error(self):
        """Should return empty dict on error."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()

        mock_response = MagicMock()
        mock_response.status = 404
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.ollama.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.model_info("nonexistent-model")

        assert result == {}


class TestOllamaGenerate:
    """Tests for generate method."""

    @pytest.mark.asyncio
    async def test_generate_basic_response(self):
        """Should generate response from Ollama API."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()

        ollama_response = {
            "model": "llama3.2",
            "response": "This is a test response from Ollama.",
            "done": True,
        }

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=ollama_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.ollama.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.generate("Test prompt")

        assert "test response from Ollama" in result

    @pytest.mark.asyncio
    async def test_generate_with_context(self, sample_context):
        """Should include context in prompt."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()

        ollama_response = {
            "model": "llama3.2",
            "response": "Response considering context.",
            "done": True,
        }

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=ollama_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.ollama.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.generate("Test prompt", context=sample_context)

        assert result is not None
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self):
        """Should include system prompt when set."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()
        agent.system_prompt = "You are a helpful coding assistant."

        ollama_response = {
            "model": "llama3.2",
            "response": "Response with system context.",
            "done": True,
        }

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=ollama_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        captured_payload = {}

        def capture_post(url, json=None):
            captured_payload["json"] = json
            return mock_response

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=capture_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.ollama.create_client_session",
            return_value=mock_session,
        ):
            await agent.generate("Test prompt")

        # Verify system prompt was included in the full prompt
        assert "System context:" in captured_payload["json"]["prompt"]
        assert "helpful coding assistant" in captured_payload["json"]["prompt"]

    @pytest.mark.asyncio
    async def test_generate_returns_empty_string_on_missing_response(self):
        """Should return empty string when response field is missing."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()

        ollama_response = {"model": "llama3.2", "done": True}

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=ollama_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.ollama.create_client_session",
            return_value=mock_session,
        ):
            result = await agent.generate("Test prompt")

        assert result == ""


class TestOllamaGenerateStream:
    """Tests for streaming generation."""

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self):
        """Should yield text chunks from streaming response."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()

        # Ollama streaming format: NDJSON (newline-delimited JSON)
        stream_chunks = [
            b'{"model":"llama3.2","response":"Hello"}\n',
            b'{"model":"llama3.2","response":" world"}\n',
            b'{"model":"llama3.2","response":"!"}\n',
            b'{"model":"llama3.2","response":"","done":true}\n',
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
            "aragora.agents.api_agents.ollama.create_client_session",
            return_value=mock_session,
        ):
            chunks = []
            async for chunk in agent.generate_stream("Test prompt"):
                chunks.append(chunk)

        assert len(chunks) == 3
        assert "".join(chunks) == "Hello world!"

    @pytest.mark.asyncio
    async def test_stream_with_system_prompt(self):
        """Should include system prompt in streaming request."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()
        agent.system_prompt = "Be concise."

        stream_chunks = [
            b'{"model":"llama3.2","response":"OK"}\n',
            b'{"model":"llama3.2","response":"","done":true}\n',
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

        captured_payload = {}

        def capture_post(url, json=None):
            captured_payload["json"] = json
            return mock_response

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=capture_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.ollama.create_client_session",
            return_value=mock_session,
        ):
            chunks = []
            async for chunk in agent.generate_stream("Test prompt"):
                chunks.append(chunk)

        assert "System context:" in captured_payload["json"]["prompt"]
        assert captured_payload["json"]["stream"] is True

    @pytest.mark.asyncio
    async def test_stream_handles_malformed_json(self):
        """Should skip malformed JSON lines gracefully."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()

        stream_chunks = [
            b'{"model":"llama3.2","response":"Hello"}\n',
            b"malformed json line\n",
            b'{"model":"llama3.2","response":" world"}\n',
            b'{"model":"llama3.2","response":"","done":true}\n',
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
            "aragora.agents.api_agents.ollama.create_client_session",
            return_value=mock_session,
        ):
            chunks = []
            async for chunk in agent.generate_stream("Test prompt"):
                chunks.append(chunk)

        # Should have received valid chunks despite malformed line
        assert "".join(chunks) == "Hello world"


class TestOllamaCritique:
    """Tests for critique method."""

    @pytest.mark.asyncio
    async def test_critique_returns_structured_feedback(self):
        """Should return structured critique."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()

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
            assert critique.agent == "ollama"
            assert critique.target_agent == "test-agent"
            assert len(critique.issues) > 0
            assert len(critique.suggestions) > 0

    @pytest.mark.asyncio
    async def test_critique_with_context(self, sample_context):
        """Should include context when critiquing."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()

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
            # Verify generate was called with context (passed as positional arg)
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args
            # Context is passed as second positional argument to generate()
            assert call_args[0][1] == sample_context

    @pytest.mark.asyncio
    async def test_critique_without_target_agent(self):
        """Should handle critique without explicit target agent."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()

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


class TestOllamaVote:
    """Tests for vote method (inherited from base Agent)."""

    @pytest.mark.asyncio
    async def test_vote_selects_best_proposal(self):
        """Should vote for best proposal."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()

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
            assert vote.agent == "ollama"
            assert vote.choice == "agent2"
            assert vote.confidence == 0.85
            assert vote.continue_debate is False

    @pytest.mark.asyncio
    async def test_vote_with_continue_debate_yes(self):
        """Should indicate debate should continue when appropriate."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()

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


class TestOllamaPullModel:
    """Tests for model pulling functionality."""

    @pytest.mark.asyncio
    async def test_pull_model_yields_progress(self):
        """Should yield progress updates during model pull."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()

        progress_chunks = [
            b'{"status":"pulling manifest"}\n',
            b'{"status":"downloading","completed":1000000,"total":5000000}\n',
            b'{"status":"downloading","completed":3000000,"total":5000000}\n',
            b'{"status":"downloading","completed":5000000,"total":5000000}\n',
            b'{"status":"success"}\n',
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
        mock_response.content = MockContent(progress_chunks)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.ollama.create_client_session",
            return_value=mock_session,
        ):
            progress_updates = []
            async for update in agent.pull_model("llama3.2"):
                progress_updates.append(update)

        assert len(progress_updates) == 5
        assert progress_updates[0]["status"] == "pulling manifest"
        assert progress_updates[-1]["status"] == "success"


class TestOllamaErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_handles_api_error(self):
        """Should raise AgentAPIError on API failure."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()

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
            "aragora.agents.api_agents.ollama.create_client_session",
            return_value=mock_session,
        ):
            with pytest.raises(AgentAPIError) as exc_info:
                await agent.generate("Test prompt")

            assert "500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handles_connection_error(self):
        """Should raise AgentConnectionError when Ollama is not running."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            side_effect=aiohttp.ClientConnectorError(MagicMock(), OSError("Connection refused"))
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.ollama.create_client_session",
            return_value=mock_session,
        ):
            with pytest.raises(AgentConnectionError) as exc_info:
                await agent.generate("Test prompt")

            assert "Cannot connect to Ollama" in str(exc_info.value)
            assert "ollama serve" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handles_invalid_json_response(self):
        """Should raise AgentAPIError on invalid JSON response."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()

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
            "aragora.agents.api_agents.ollama.create_client_session",
            return_value=mock_session,
        ):
            with pytest.raises(AgentAPIError) as exc_info:
                await agent.generate("Test prompt")

            assert "invalid JSON" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stream_handles_api_error(self):
        """Should raise AgentAPIError on streaming API failure."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()

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
            "aragora.agents.api_agents.ollama.create_client_session",
            return_value=mock_session,
        ):
            with pytest.raises(AgentAPIError):
                async for _ in agent.generate_stream("Test prompt"):
                    pass

    @pytest.mark.asyncio
    async def test_stream_handles_connection_error(self):
        """Should raise AgentConnectionError on streaming connection failure."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            side_effect=aiohttp.ClientConnectorError(MagicMock(), OSError("Connection refused"))
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.agents.api_agents.ollama.create_client_session",
            return_value=mock_session,
        ):
            with pytest.raises(AgentConnectionError):
                async for _ in agent.generate_stream("Test prompt"):
                    pass


class TestOllamaTokenUsage:
    """Tests for token usage tracking (inherited from APIAgent base)."""

    def test_token_usage_tracking(self):
        """Should track token usage."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()
        agent._record_token_usage(100, 50)

        usage = agent.get_token_usage()

        assert usage["tokens_in"] == 100
        assert usage["tokens_out"] == 50
        assert usage["total_tokens_in"] == 100
        assert usage["total_tokens_out"] == 50

    def test_reset_token_usage(self):
        """Should reset token counters."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()
        agent._record_token_usage(100, 50)
        agent.reset_token_usage()

        assert agent.last_tokens_in == 0
        assert agent.last_tokens_out == 0
        assert agent.total_tokens_in == 0
        assert agent.total_tokens_out == 0

    def test_accumulates_token_usage(self):
        """Should accumulate total token usage across calls."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()
        agent._record_token_usage(100, 50)
        agent._record_token_usage(200, 100)

        assert agent.last_tokens_in == 200
        assert agent.last_tokens_out == 100
        assert agent.total_tokens_in == 300
        assert agent.total_tokens_out == 150


class TestOllamaGenerationParams:
    """Tests for generation parameters."""

    def test_set_generation_params(self):
        """Should set generation parameters."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()
        agent.set_generation_params(temperature=0.7, top_p=0.9)

        assert agent.temperature == 0.7
        assert agent.top_p == 0.9

    def test_get_generation_params(self):
        """Should return non-None generation parameters."""
        from aragora.agents.api_agents.ollama import OllamaAgent

        agent = OllamaAgent()
        agent.temperature = 0.8
        agent.top_p = None

        params = agent.get_generation_params()

        assert "temperature" in params
        assert params["temperature"] == 0.8
        assert "top_p" not in params
