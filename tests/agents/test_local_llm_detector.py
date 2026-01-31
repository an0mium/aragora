"""
Tests for aragora.agents.local_llm_detector module.

Tests the LocalLLMDetector functionality including:
- LocalLLMServer and LocalLLMStatus dataclasses
- Local server detection (Ollama, LM Studio)
- Timeout handling during detection
- Health check functionality
- Failure scenarios (unreachable servers, slow responses)
- Model listing and validation
- Environment variable overrides
- Convenience functions (detect_local_llms, detect_local_llms_sync)
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from aragora.agents.local_llm_detector import (
    LocalLLMDetector,
    LocalLLMServer,
    LocalLLMStatus,
    detect_local_llms,
    detect_local_llms_sync,
)


# =============================================================================
# Test Fixtures for httpx-based HTTP client pool mocking
# =============================================================================


def create_mock_response(status_code: int = 200, json_data: dict | None = None):
    """Create a mock httpx response."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json = MagicMock(return_value=json_data or {})
    return mock_response


def create_mock_pool(responses: dict[str, MagicMock] | None = None, error: Exception | None = None):
    """Create a mock HTTP client pool.

    Args:
        responses: Dict mapping URL patterns to mock responses
        error: Exception to raise on request (simulates connection failure)

    Returns:
        Mock pool configured for get_http_pool() patching
    """
    mock_pool = MagicMock()

    @asynccontextmanager
    async def mock_get_session(provider: str):
        mock_client = MagicMock()

        async def mock_get(url: str, timeout: float | None = None):
            if error:
                raise error
            if responses:
                for pattern, response in responses.items():
                    if pattern in url:
                        return response
            # Default empty response
            return create_mock_response(200, {})

        mock_client.get = mock_get
        yield mock_client

    mock_pool.get_session = mock_get_session
    return mock_pool


# =============================================================================
# LocalLLMServer Dataclass Tests
# =============================================================================


class TestLocalLLMServer:
    """Test LocalLLMServer dataclass."""

    def test_basic_initialization(self):
        """Test basic server initialization."""
        server = LocalLLMServer(
            name="ollama",
            base_url="http://localhost:11434",
            available=True,
        )

        assert server.name == "ollama"
        assert server.base_url == "http://localhost:11434"
        assert server.available is True
        assert server.models == []
        assert server.default_model is None
        assert server.version is None

    def test_initialization_with_all_fields(self):
        """Test server with all fields populated."""
        server = LocalLLMServer(
            name="lm-studio",
            base_url="http://localhost:1234/v1",
            available=True,
            models=["llama3.2", "codellama", "mistral"],
            default_model="llama3.2",
            version="0.2.5",
        )

        assert server.name == "lm-studio"
        assert server.base_url == "http://localhost:1234/v1"
        assert server.available is True
        assert len(server.models) == 3
        assert "llama3.2" in server.models
        assert server.default_model == "llama3.2"
        assert server.version == "0.2.5"

    def test_unavailable_server(self):
        """Test unavailable server state."""
        server = LocalLLMServer(
            name="ollama",
            base_url="http://localhost:11434",
            available=False,
        )

        assert server.available is False
        assert server.models == []
        assert server.default_model is None


# =============================================================================
# LocalLLMStatus Dataclass Tests
# =============================================================================


class TestLocalLLMStatus:
    """Test LocalLLMStatus dataclass."""

    def test_empty_status(self):
        """Test empty status initialization."""
        status = LocalLLMStatus()

        assert status.servers == []
        assert status.total_models == 0
        assert status.recommended_server is None
        assert status.recommended_model is None

    def test_any_available_property_true(self):
        """Test any_available returns True when at least one server is available."""
        server = LocalLLMServer(
            name="ollama",
            base_url="http://localhost:11434",
            available=True,
        )
        status = LocalLLMStatus(servers=[server])

        assert status.any_available is True

    def test_any_available_property_false(self):
        """Test any_available returns False when no servers available."""
        server = LocalLLMServer(
            name="ollama",
            base_url="http://localhost:11434",
            available=False,
        )
        status = LocalLLMStatus(servers=[server])

        assert status.any_available is False

    def test_any_available_with_mixed_servers(self):
        """Test any_available with mixed availability."""
        servers = [
            LocalLLMServer(name="ollama", base_url="http://localhost:11434", available=False),
            LocalLLMServer(name="lm-studio", base_url="http://localhost:1234/v1", available=True),
        ]
        status = LocalLLMStatus(servers=servers)

        assert status.any_available is True

    def test_get_available_agents_returns_available_names(self):
        """Test get_available_agents returns names of available servers."""
        servers = [
            LocalLLMServer(name="ollama", base_url="http://localhost:11434", available=True),
            LocalLLMServer(name="lm-studio", base_url="http://localhost:1234/v1", available=False),
            LocalLLMServer(name="custom", base_url="http://localhost:8080", available=True),
        ]
        status = LocalLLMStatus(servers=servers)

        agents = status.get_available_agents()

        assert len(agents) == 2
        assert "ollama" in agents
        assert "custom" in agents
        assert "lm-studio" not in agents

    def test_get_available_agents_empty_when_none_available(self):
        """Test get_available_agents returns empty list when none available."""
        servers = [
            LocalLLMServer(name="ollama", base_url="http://localhost:11434", available=False),
            LocalLLMServer(name="lm-studio", base_url="http://localhost:1234/v1", available=False),
        ]
        status = LocalLLMStatus(servers=servers)

        agents = status.get_available_agents()

        assert agents == []


# =============================================================================
# LocalLLMDetector Initialization Tests
# =============================================================================


class TestLocalLLMDetectorInit:
    """Test LocalLLMDetector initialization."""

    def test_default_timeout(self):
        """Test default timeout value."""
        detector = LocalLLMDetector()

        assert detector.timeout == 5.0

    def test_custom_timeout(self):
        """Test custom timeout value."""
        detector = LocalLLMDetector(timeout=10.0)

        assert detector.timeout == 10.0

    def test_servers_config_exists(self):
        """Test SERVERS configuration exists."""
        assert hasattr(LocalLLMDetector, "SERVERS")
        assert "ollama" in LocalLLMDetector.SERVERS
        assert "lm-studio" in LocalLLMDetector.SERVERS

    def test_ollama_config(self):
        """Test Ollama server configuration."""
        config = LocalLLMDetector.SERVERS["ollama"]

        assert config["base_url"] == "http://localhost:11434"
        assert config["env_var"] == "OLLAMA_HOST"
        assert config["health_endpoint"] == "/api/tags"
        assert config["models_key"] == "models"
        assert config["model_name_key"] == "name"

    def test_lm_studio_config(self):
        """Test LM Studio server configuration."""
        config = LocalLLMDetector.SERVERS["lm-studio"]

        assert config["base_url"] == "http://localhost:1234/v1"
        assert config["env_var"] == "LM_STUDIO_HOST"
        assert config["health_endpoint"] == "/models"
        assert config["models_key"] == "data"
        assert config["model_name_key"] == "id"

    def test_model_preferences_exist(self):
        """Test MODEL_PREFERENCES configuration exists."""
        assert hasattr(LocalLLMDetector, "MODEL_PREFERENCES")
        assert len(LocalLLMDetector.MODEL_PREFERENCES) > 0
        assert "llama3.2" in LocalLLMDetector.MODEL_PREFERENCES


# =============================================================================
# Model Selection Tests
# =============================================================================


class TestPickBestModel:
    """Test _pick_best_model method."""

    def test_pick_preferred_model_first(self):
        """Test preferred model is selected first."""
        detector = LocalLLMDetector()
        models = ["gemma:latest", "llama3.2:7b", "phi:latest"]

        best = detector._pick_best_model(models)

        assert best == "llama3.2:7b"

    def test_pick_second_preference_if_first_missing(self):
        """Test second preference when first is missing."""
        detector = LocalLLMDetector()
        models = ["gemma:latest", "llama3.1:7b", "phi:latest"]

        best = detector._pick_best_model(models)

        assert best == "llama3.1:7b"

    def test_pick_codellama_over_mistral(self):
        """Test codellama is preferred over mistral."""
        detector = LocalLLMDetector()
        models = ["mistral:latest", "codellama:34b"]

        best = detector._pick_best_model(models)

        assert best == "codellama:34b"

    def test_pick_first_model_when_no_preference_match(self):
        """Test first model is returned when no preference matches."""
        detector = LocalLLMDetector()
        models = ["custom-model-1", "custom-model-2"]

        best = detector._pick_best_model(models)

        assert best == "custom-model-1"

    def test_pick_returns_default_for_empty_list(self):
        """Test returns 'default' for empty list."""
        detector = LocalLLMDetector()
        models = []

        best = detector._pick_best_model(models)

        assert best == "default"

    def test_pick_case_insensitive(self):
        """Test model matching is case insensitive."""
        detector = LocalLLMDetector()
        models = ["LLAMA3.2:LATEST", "gemma"]

        best = detector._pick_best_model(models)

        assert best == "LLAMA3.2:LATEST"


# =============================================================================
# Ollama Detection Tests
# =============================================================================


class TestDetectOllama:
    """Test Ollama server detection."""

    @pytest.mark.asyncio
    async def test_detect_ollama_available(self):
        """Test detecting available Ollama server."""
        detector = LocalLLMDetector()

        ollama_response = {
            "models": [
                {"name": "llama3.2:latest"},
                {"name": "codellama:7b"},
            ]
        }

        mock_response = create_mock_response(200, ollama_response)
        mock_pool = create_mock_pool({"/api/tags": mock_response})

        with patch("aragora.agents.local_llm_detector.get_http_pool", return_value=mock_pool):
            server = await detector.detect_ollama()

            assert server.name == "ollama"
            assert server.available is True
            assert len(server.models) == 2
            assert "llama3.2:latest" in server.models
            assert server.default_model == "llama3.2:latest"

    @pytest.mark.asyncio
    async def test_detect_ollama_unavailable(self):
        """Test detecting unavailable Ollama server."""
        detector = LocalLLMDetector()

        mock_pool = create_mock_pool(error=ConnectionError("Connection refused"))

        with patch("aragora.agents.local_llm_detector.get_http_pool", return_value=mock_pool):
            server = await detector.detect_ollama()

            assert server.name == "ollama"
            assert server.available is False
            assert server.models == []

    @pytest.mark.asyncio
    async def test_detect_ollama_with_env_var_override(self):
        """Test Ollama detection with environment variable override."""
        detector = LocalLLMDetector()

        ollama_response = {"models": [{"name": "test-model"}]}
        mock_response = create_mock_response(200, ollama_response)

        # Track URLs called
        called_urls = []

        mock_pool = MagicMock()

        @asynccontextmanager
        async def mock_get_session(provider: str):
            mock_client = MagicMock()

            async def mock_get(url: str, timeout: float | None = None):
                called_urls.append(url)
                return mock_response

            mock_client.get = mock_get
            yield mock_client

        mock_pool.get_session = mock_get_session

        with patch.dict("os.environ", {"OLLAMA_HOST": "http://custom-host:9999"}):
            with patch("aragora.agents.local_llm_detector.get_http_pool", return_value=mock_pool):
                server = await detector.detect_ollama()

                assert server.base_url == "http://custom-host:9999"
                # Verify URL used
                assert len(called_urls) == 1
                assert "custom-host:9999" in called_urls[0]


# =============================================================================
# LM Studio Detection Tests
# =============================================================================


class TestDetectLMStudio:
    """Test LM Studio server detection."""

    @pytest.mark.asyncio
    async def test_detect_lm_studio_available(self):
        """Test detecting available LM Studio server."""
        detector = LocalLLMDetector()

        lm_studio_response = {
            "data": [
                {"id": "TheBloke/Llama-2-7B-Chat-GGUF"},
                {"id": "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"},
            ]
        }

        mock_response = create_mock_response(200, lm_studio_response)
        mock_pool = create_mock_pool({"/models": mock_response})

        with patch("aragora.agents.local_llm_detector.get_http_pool", return_value=mock_pool):
            server = await detector.detect_lm_studio()

            assert server.name == "lm-studio"
            assert server.available is True
            assert len(server.models) == 2
            assert server.base_url == "http://localhost:1234/v1"

    @pytest.mark.asyncio
    async def test_detect_lm_studio_unavailable(self):
        """Test detecting unavailable LM Studio server."""
        detector = LocalLLMDetector()

        mock_pool = create_mock_pool(error=ConnectionError("Connection refused"))

        with patch("aragora.agents.local_llm_detector.get_http_pool", return_value=mock_pool):
            server = await detector.detect_lm_studio()

            assert server.name == "lm-studio"
            assert server.available is False
            assert server.models == []

    @pytest.mark.asyncio
    async def test_detect_lm_studio_with_env_var_override(self):
        """Test LM Studio detection with environment variable override."""
        detector = LocalLLMDetector()

        lm_studio_response = {"data": [{"id": "test-model"}]}

        mock_response = create_mock_response(200, lm_studio_response)
        mock_pool = create_mock_pool({"/models": mock_response})

        with patch.dict("os.environ", {"LM_STUDIO_HOST": "http://192.168.1.100:5000/v1"}):
            with patch("aragora.agents.local_llm_detector.get_http_pool", return_value=mock_pool):
                server = await detector.detect_lm_studio()

                assert server.base_url == "http://192.168.1.100:5000/v1"


# =============================================================================
# detect_all Tests
# =============================================================================


class TestDetectAll:
    """Test detect_all method."""

    @pytest.mark.asyncio
    async def test_detect_all_both_available(self):
        """Test detecting all servers when both are available."""
        detector = LocalLLMDetector()

        ollama_response = {"models": [{"name": "llama3.2:latest"}]}
        lm_studio_response = {"data": [{"id": "mistral-7b"}]}

        mock_pool = create_mock_pool(
            {
                "/api/tags": create_mock_response(200, ollama_response),
                "/models": create_mock_response(200, lm_studio_response),
            }
        )

        with patch("aragora.agents.local_llm_detector.get_http_pool", return_value=mock_pool):
            status = await detector.detect_all()

            assert len(status.servers) == 2
            assert status.any_available is True
            assert status.total_models == 2
            assert status.recommended_server is not None
            assert status.recommended_model is not None

    @pytest.mark.asyncio
    async def test_detect_all_none_available(self):
        """Test detecting all servers when none are available."""
        detector = LocalLLMDetector()

        mock_pool = create_mock_pool(error=ConnectionError("Connection refused"))

        with patch("aragora.agents.local_llm_detector.get_http_pool", return_value=mock_pool):
            status = await detector.detect_all()

            assert len(status.servers) == 2
            assert status.any_available is False
            assert status.total_models == 0
            assert status.recommended_server is None
            assert status.recommended_model is None

    @pytest.mark.asyncio
    async def test_detect_all_partial_availability(self):
        """Test detecting all servers when only some are available."""
        detector = LocalLLMDetector()

        ollama_response = {"models": [{"name": "llama3.2:latest"}]}

        # Custom pool that returns success for Ollama, error for LM Studio
        mock_pool = MagicMock()

        @asynccontextmanager
        async def mock_get_session(provider: str):
            mock_client = MagicMock()

            async def mock_get(url: str, timeout: float | None = None):
                if "/api/tags" in url:
                    return create_mock_response(200, ollama_response)
                raise ConnectionError("Connection refused")

            mock_client.get = mock_get
            yield mock_client

        mock_pool.get_session = mock_get_session

        with patch("aragora.agents.local_llm_detector.get_http_pool", return_value=mock_pool):
            status = await detector.detect_all()

            assert status.any_available is True
            available_servers = [s for s in status.servers if s.available]
            assert len(available_servers) == 1
            assert available_servers[0].name == "ollama"


# =============================================================================
# Timeout Handling Tests
# =============================================================================


class TestTimeoutHandling:
    """Test timeout handling during detection."""

    @pytest.mark.asyncio
    async def test_timeout_returns_unavailable(self):
        """Test that timeout returns server as unavailable."""
        detector = LocalLLMDetector(timeout=0.1)

        mock_pool = create_mock_pool(error=asyncio.TimeoutError())

        with patch("aragora.agents.local_llm_detector.get_http_pool", return_value=mock_pool):
            server = await detector.detect_ollama()

            assert server.available is False

    @pytest.mark.asyncio
    async def test_timeout_configuration_used(self):
        """Test that timeout configuration is passed to request."""
        detector = LocalLLMDetector(timeout=15.0)

        # Track timeout passed to get()
        captured_timeout = []

        mock_pool = MagicMock()

        @asynccontextmanager
        async def mock_get_session(provider: str):
            mock_client = MagicMock()

            async def mock_get(url: str, timeout: float | None = None):
                captured_timeout.append(timeout)
                return create_mock_response(200, {"models": []})

            mock_client.get = mock_get
            yield mock_client

        mock_pool.get_session = mock_get_session

        with patch("aragora.agents.local_llm_detector.get_http_pool", return_value=mock_pool):
            await detector.detect_ollama()

            # Verify timeout was set
            assert len(captured_timeout) == 1
            assert captured_timeout[0] == 15.0


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_non_200_status_returns_unavailable(self):
        """Test non-200 status returns server as unavailable."""
        detector = LocalLLMDetector()

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            server = await detector.detect_ollama()

            assert server.available is False

    @pytest.mark.asyncio
    async def test_invalid_json_returns_unavailable(self):
        """Test invalid JSON response returns server as unavailable."""
        detector = LocalLLMDetector()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(side_effect=ValueError("Invalid JSON"))
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            server = await detector.detect_ollama()

            assert server.available is False

    @pytest.mark.asyncio
    async def test_missing_models_key_handles_gracefully(self):
        """Test missing models key in response."""
        detector = LocalLLMDetector()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"other_key": "value"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            server = await detector.detect_ollama()

            # Server is available but has no models
            assert server.available is True
            assert server.models == []

    @pytest.mark.asyncio
    async def test_unexpected_exception_returns_unavailable(self):
        """Test unexpected exception returns server as unavailable."""
        detector = LocalLLMDetector()

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=Exception("Unexpected error"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            server = await detector.detect_ollama()

            assert server.available is False

    @pytest.mark.asyncio
    async def test_key_error_in_model_extraction(self):
        """Test KeyError during model extraction."""
        detector = LocalLLMDetector()

        # Models without expected 'name' key
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"models": [{"different_key": "value"}, {"name": "valid-model"}]}
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            server = await detector.detect_ollama()

            # Should still work, extracting valid models
            assert server.available is True
            assert "valid-model" in server.models

    @pytest.mark.asyncio
    async def test_non_dict_model_entries_skipped(self):
        """Test non-dict model entries are skipped."""
        detector = LocalLLMDetector()

        # Models with non-dict entries
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"models": ["string-model", {"name": "valid-model"}, 123]}
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            server = await detector.detect_ollama()

            assert server.available is True
            # Only dict entries with 'name' key should be included
            assert len(server.models) == 1
            assert "valid-model" in server.models


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.asyncio
    async def test_detect_local_llms_function(self):
        """Test detect_local_llms convenience function."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"models": [{"name": "test-model"}]})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            status = await detect_local_llms()

            assert isinstance(status, LocalLLMStatus)
            assert len(status.servers) == 2

    def test_detect_local_llms_sync_function(self):
        """Test detect_local_llms_sync convenience function."""
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
            status = detect_local_llms_sync()

            assert isinstance(status, LocalLLMStatus)


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Test module exports."""

    def test_local_llm_detector_exportable(self):
        """Test LocalLLMDetector can be imported."""
        from aragora.agents.local_llm_detector import LocalLLMDetector

        assert LocalLLMDetector is not None

    def test_local_llm_server_exportable(self):
        """Test LocalLLMServer can be imported."""
        from aragora.agents.local_llm_detector import LocalLLMServer

        assert LocalLLMServer is not None

    def test_local_llm_status_exportable(self):
        """Test LocalLLMStatus can be imported."""
        from aragora.agents.local_llm_detector import LocalLLMStatus

        assert LocalLLMStatus is not None

    def test_detect_local_llms_exportable(self):
        """Test detect_local_llms can be imported."""
        from aragora.agents.local_llm_detector import detect_local_llms

        assert detect_local_llms is not None

    def test_detect_local_llms_sync_exportable(self):
        """Test detect_local_llms_sync can be imported."""
        from aragora.agents.local_llm_detector import detect_local_llms_sync

        assert detect_local_llms_sync is not None

    def test_all_exports_in_module_all(self):
        """Test __all__ contains expected exports."""
        from aragora.agents import local_llm_detector

        assert hasattr(local_llm_detector, "__all__")
        exports = local_llm_detector.__all__

        assert "LocalLLMDetector" in exports
        assert "LocalLLMServer" in exports
        assert "LocalLLMStatus" in exports
        assert "detect_local_llms" in exports
        assert "detect_local_llms_sync" in exports


# =============================================================================
# URL Construction Tests
# =============================================================================


class TestURLConstruction:
    """Test URL construction for health endpoints."""

    @pytest.mark.asyncio
    async def test_url_construction_strips_trailing_slash(self):
        """Test trailing slash is handled in URL construction."""
        detector = LocalLLMDetector()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"models": []})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        # Test with trailing slash in env var
        with patch.dict("os.environ", {"OLLAMA_HOST": "http://localhost:11434/"}):
            with patch("aiohttp.ClientSession", return_value=mock_session):
                await detector.detect_ollama()

                call_url = mock_session.get.call_args[0][0]
                # Should not have double slashes
                assert "//" not in call_url.replace("http://", "")


# =============================================================================
# Caching Behavior Tests
# =============================================================================


class TestCachingBehavior:
    """Test that detector doesn't cache results between calls."""

    @pytest.mark.asyncio
    async def test_multiple_detections_make_new_requests(self):
        """Test that each detection makes fresh HTTP requests."""
        detector = LocalLLMDetector()

        call_count = 0

        def mock_get(url, **kwargs):
            nonlocal call_count
            call_count += 1

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"models": []})
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            return mock_response

        mock_session = MagicMock()
        mock_session.get = mock_get
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await detector.detect_ollama()
            await detector.detect_ollama()

            # Should make 2 separate requests
            assert call_count == 2


# =============================================================================
# Recommendation Tests
# =============================================================================


class TestRecommendations:
    """Test server and model recommendation logic."""

    @pytest.mark.asyncio
    async def test_recommendation_from_first_available_server(self):
        """Test recommendation comes from first available server."""
        detector = LocalLLMDetector()

        ollama_response = {"models": [{"name": "codellama:latest"}]}

        def mock_get(url, **kwargs):
            mock_response = AsyncMock()
            if "/api/tags" in url:
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value=ollama_response)
            else:
                raise aiohttp.ClientConnectorError(
                    connection_key=MagicMock(), os_error=OSError("Connection refused")
                )
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            return mock_response

        mock_session = MagicMock()
        mock_session.get = mock_get
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            status = await detector.detect_all()

            assert status.recommended_server == "ollama"
            assert status.recommended_model == "codellama:latest"

    @pytest.mark.asyncio
    async def test_no_recommendation_when_no_models(self):
        """Test no recommendation when available server has no models."""
        detector = LocalLLMDetector()

        # Available but no models
        ollama_response = {"models": []}

        def mock_get(url, **kwargs):
            mock_response = AsyncMock()
            if "/api/tags" in url:
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value=ollama_response)
            else:
                raise aiohttp.ClientConnectorError(
                    connection_key=MagicMock(), os_error=OSError("Connection refused")
                )
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            return mock_response

        mock_session = MagicMock()
        mock_session.get = mock_get
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            status = await detector.detect_all()

            # Server is available but no models, so no recommendation
            assert status.recommended_server is None
            assert status.recommended_model is None


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.mark.asyncio
    async def test_typical_development_setup(self):
        """Test typical development setup with Ollama running."""
        detector = LocalLLMDetector()

        ollama_response = {
            "models": [
                {"name": "llama3.2:latest"},
                {"name": "llama3.1:8b"},
                {"name": "codellama:7b"},
                {"name": "phi:latest"},
            ]
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=ollama_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            server = await detector.detect_ollama()

            assert server.available is True
            assert len(server.models) == 4
            # llama3.2 should be preferred
            assert server.default_model == "llama3.2:latest"

    @pytest.mark.asyncio
    async def test_privacy_first_deployment_both_servers(self):
        """Test privacy-first deployment with both servers available."""
        detector = LocalLLMDetector()

        ollama_response = {"models": [{"name": "llama3.2:latest"}]}
        lm_studio_response = {"data": [{"id": "mistral-7b-instruct"}]}

        responses = {
            "/api/tags": ollama_response,
            "/models": lm_studio_response,
        }

        def mock_get(url, **kwargs):
            mock_response = AsyncMock()
            mock_response.status = 200
            for endpoint, data in responses.items():
                if endpoint in url:
                    mock_response.json = AsyncMock(return_value=data)
                    break
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            return mock_response

        mock_session = MagicMock()
        mock_session.get = mock_get
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            status = await detect_local_llms()

            assert status.any_available is True
            assert status.total_models == 2
            assert len(status.get_available_agents()) == 2
