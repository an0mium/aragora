"""Tests for AragoraClient and AragoraAsyncClient initialization and configuration."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient
from aragora_sdk.namespaces.agents import AgentsAPI, AsyncAgentsAPI
from aragora_sdk.namespaces.control_plane import AsyncControlPlaneAPI, ControlPlaneAPI
from aragora_sdk.namespaces.debates import AsyncDebatesAPI, DebatesAPI


class TestAragoraClientInitialization:
    """Tests for synchronous client initialization."""

    def test_client_initializes_with_base_url(self) -> None:
        """Client can be initialized with just a base URL."""
        client = AragoraClient(base_url="https://api.aragora.ai")
        assert client.base_url == "https://api.aragora.ai"
        client.close()

    def test_client_strips_trailing_slash_from_base_url(self) -> None:
        """Client strips trailing slash from base URL."""
        client = AragoraClient(base_url="https://api.aragora.ai/")
        assert client.base_url == "https://api.aragora.ai"
        client.close()

    def test_client_stores_api_key(self) -> None:
        """Client stores the API key."""
        client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
        assert client.api_key == "test-key"
        client.close()

    def test_client_default_timeout(self) -> None:
        """Client has default timeout of 30 seconds."""
        client = AragoraClient(base_url="https://api.aragora.ai")
        assert client.timeout == 30.0
        client.close()

    def test_client_custom_timeout(self) -> None:
        """Client accepts custom timeout."""
        client = AragoraClient(base_url="https://api.aragora.ai", timeout=60.0)
        assert client.timeout == 60.0
        client.close()

    def test_client_default_max_retries(self) -> None:
        """Client has default max_retries of 3."""
        client = AragoraClient(base_url="https://api.aragora.ai")
        assert client.max_retries == 3
        client.close()

    def test_client_custom_max_retries(self) -> None:
        """Client accepts custom max_retries."""
        client = AragoraClient(base_url="https://api.aragora.ai", max_retries=5)
        assert client.max_retries == 5
        client.close()

    def test_client_default_retry_delay(self) -> None:
        """Client has default retry_delay of 1.0 seconds."""
        client = AragoraClient(base_url="https://api.aragora.ai")
        assert client.retry_delay == 1.0
        client.close()

    def test_client_custom_retry_delay(self) -> None:
        """Client accepts custom retry_delay."""
        client = AragoraClient(base_url="https://api.aragora.ai", retry_delay=2.0)
        assert client.retry_delay == 2.0
        client.close()


class TestAragoraClientNamespaces:
    """Tests for synchronous client namespace initialization."""

    def test_client_has_debates_namespace(self) -> None:
        """Client has debates namespace."""
        client = AragoraClient(base_url="https://api.aragora.ai")
        assert isinstance(client.debates, DebatesAPI)
        client.close()

    def test_client_has_agents_namespace(self) -> None:
        """Client has agents namespace."""
        client = AragoraClient(base_url="https://api.aragora.ai")
        assert isinstance(client.agents, AgentsAPI)
        client.close()

    def test_client_has_control_plane_namespace(self) -> None:
        """Client has control_plane namespace."""
        client = AragoraClient(base_url="https://api.aragora.ai")
        assert isinstance(client.control_plane, ControlPlaneAPI)
        client.close()

    def test_namespaces_reference_client(self) -> None:
        """Namespaces hold reference to parent client."""
        client = AragoraClient(base_url="https://api.aragora.ai")
        assert client.debates._client is client
        assert client.agents._client is client
        assert client.control_plane._client is client
        client.close()


class TestAragoraClientHeaders:
    """Tests for client header construction."""

    def test_headers_include_content_type(self) -> None:
        """Headers include Content-Type: application/json."""
        client = AragoraClient(base_url="https://api.aragora.ai")
        headers = client._build_headers()
        assert headers["Content-Type"] == "application/json"
        client.close()

    def test_headers_include_accept(self) -> None:
        """Headers include Accept: application/json."""
        client = AragoraClient(base_url="https://api.aragora.ai")
        headers = client._build_headers()
        assert headers["Accept"] == "application/json"
        client.close()

    def test_headers_include_user_agent(self) -> None:
        """Headers include User-Agent."""
        client = AragoraClient(base_url="https://api.aragora.ai")
        headers = client._build_headers()
        assert "User-Agent" in headers
        assert "aragora-python" in headers["User-Agent"]
        client.close()

    def test_headers_include_authorization_with_api_key(self) -> None:
        """Headers include Authorization with Bearer token when API key is set."""
        client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
        headers = client._build_headers()
        assert headers["Authorization"] == "Bearer test-key"
        client.close()

    def test_headers_omit_authorization_without_api_key(self) -> None:
        """Headers omit Authorization when API key is not set."""
        client = AragoraClient(base_url="https://api.aragora.ai")
        headers = client._build_headers()
        assert "Authorization" not in headers
        client.close()


class TestAragoraClientContextManager:
    """Tests for synchronous client context manager."""

    def test_client_context_manager(self) -> None:
        """Client can be used as context manager."""
        with AragoraClient(base_url="https://api.aragora.ai") as client:
            assert isinstance(client, AragoraClient)

    def test_client_context_manager_closes_client(self) -> None:
        """Context manager closes client on exit."""
        with patch.object(AragoraClient, "close") as mock_close:
            with AragoraClient(base_url="https://api.aragora.ai"):
                pass
            mock_close.assert_called_once()


class TestAragoraAsyncClientInitialization:
    """Tests for async client initialization."""

    @pytest.mark.asyncio
    async def test_async_client_initializes_with_base_url(self) -> None:
        """Async client can be initialized with just a base URL."""
        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            assert client.base_url == "https://api.aragora.ai"

    @pytest.mark.asyncio
    async def test_async_client_strips_trailing_slash(self) -> None:
        """Async client strips trailing slash from base URL."""
        async with AragoraAsyncClient(base_url="https://api.aragora.ai/") as client:
            assert client.base_url == "https://api.aragora.ai"

    @pytest.mark.asyncio
    async def test_async_client_stores_api_key(self) -> None:
        """Async client stores the API key."""
        async with AragoraAsyncClient(
            base_url="https://api.aragora.ai", api_key="test-key"
        ) as client:
            assert client.api_key == "test-key"

    @pytest.mark.asyncio
    async def test_async_client_stores_ws_url(self) -> None:
        """Async client stores explicit WebSocket URL."""
        async with AragoraAsyncClient(
            base_url="https://api.aragora.ai", ws_url="wss://ws.aragora.ai"
        ) as client:
            assert client.ws_url == "wss://ws.aragora.ai"

    @pytest.mark.asyncio
    async def test_async_client_default_timeout(self) -> None:
        """Async client has default timeout of 30 seconds."""
        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            assert client.timeout == 30.0

    @pytest.mark.asyncio
    async def test_async_client_custom_timeout(self) -> None:
        """Async client accepts custom timeout."""
        async with AragoraAsyncClient(base_url="https://api.aragora.ai", timeout=60.0) as client:
            assert client.timeout == 60.0


class TestAragoraAsyncClientNamespaces:
    """Tests for async client namespace initialization."""

    @pytest.mark.asyncio
    async def test_async_client_has_debates_namespace(self) -> None:
        """Async client has debates namespace."""
        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            assert isinstance(client.debates, AsyncDebatesAPI)

    @pytest.mark.asyncio
    async def test_async_client_has_agents_namespace(self) -> None:
        """Async client has agents namespace."""
        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            assert isinstance(client.agents, AsyncAgentsAPI)

    @pytest.mark.asyncio
    async def test_async_client_has_control_plane_namespace(self) -> None:
        """Async client has control_plane namespace."""
        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            assert isinstance(client.control_plane, AsyncControlPlaneAPI)


class TestAragoraAsyncClientContextManager:
    """Tests for async client context manager."""

    @pytest.mark.asyncio
    async def test_async_client_context_manager(self) -> None:
        """Async client can be used as async context manager."""
        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            assert isinstance(client, AragoraAsyncClient)


class TestAragoraClientFromEnv:
    """Tests for AragoraClient.from_env() factory method."""

    def test_from_env_uses_defaults(self) -> None:
        """from_env() uses default values when no env vars are set."""
        env = {}
        with patch.dict(os.environ, env, clear=True):
            client = AragoraClient.from_env()
            assert client.base_url == "http://localhost:8080"
            assert client.api_key is None
            assert client.timeout == 30.0
            assert client.max_retries == 3
            assert client.retry_delay == 1.0
            client.close()

    def test_from_env_reads_api_url(self) -> None:
        """from_env() reads ARAGORA_API_URL."""
        with patch.dict(os.environ, {"ARAGORA_API_URL": "https://custom.aragora.ai"}):
            client = AragoraClient.from_env()
            assert client.base_url == "https://custom.aragora.ai"
            client.close()

    def test_from_env_reads_api_key(self) -> None:
        """from_env() reads ARAGORA_API_KEY."""
        with patch.dict(os.environ, {"ARAGORA_API_KEY": "ara_test_key"}):
            client = AragoraClient.from_env()
            assert client.api_key == "ara_test_key"
            client.close()

    def test_from_env_reads_timeout(self) -> None:
        """from_env() reads ARAGORA_TIMEOUT."""
        with patch.dict(os.environ, {"ARAGORA_TIMEOUT": "60"}):
            client = AragoraClient.from_env()
            assert client.timeout == 60.0
            client.close()

    def test_from_env_reads_max_retries(self) -> None:
        """from_env() reads ARAGORA_MAX_RETRIES."""
        with patch.dict(os.environ, {"ARAGORA_MAX_RETRIES": "5"}):
            client = AragoraClient.from_env()
            assert client.max_retries == 5
            client.close()

    def test_from_env_reads_retry_delay(self) -> None:
        """from_env() reads ARAGORA_RETRY_DELAY."""
        with patch.dict(os.environ, {"ARAGORA_RETRY_DELAY": "2.5"}):
            client = AragoraClient.from_env()
            assert client.retry_delay == 2.5
            client.close()

    def test_from_env_kwargs_override_env(self) -> None:
        """Explicit kwargs override environment variables."""
        with patch.dict(os.environ, {
            "ARAGORA_API_URL": "https://env.aragora.ai",
            "ARAGORA_API_KEY": "ara_env_key",
            "ARAGORA_TIMEOUT": "60",
        }):
            client = AragoraClient.from_env(
                base_url="https://override.aragora.ai",
                api_key="ara_override_key",
                timeout=90.0,
            )
            assert client.base_url == "https://override.aragora.ai"
            assert client.api_key == "ara_override_key"
            assert client.timeout == 90.0
            client.close()

    def test_from_env_all_vars(self) -> None:
        """from_env() reads all supported environment variables."""
        env = {
            "ARAGORA_API_URL": "https://full.aragora.ai",
            "ARAGORA_API_KEY": "ara_full_key",
            "ARAGORA_TIMEOUT": "45",
            "ARAGORA_MAX_RETRIES": "7",
            "ARAGORA_RETRY_DELAY": "0.5",
        }
        with patch.dict(os.environ, env):
            client = AragoraClient.from_env()
            assert client.base_url == "https://full.aragora.ai"
            assert client.api_key == "ara_full_key"
            assert client.timeout == 45.0
            assert client.max_retries == 7
            assert client.retry_delay == 0.5
            client.close()

    def test_from_env_returns_aragoraclient(self) -> None:
        """from_env() returns an AragoraClient instance."""
        client = AragoraClient.from_env()
        assert isinstance(client, AragoraClient)
        client.close()


class TestAragoraAsyncClientFromEnv:
    """Tests for AragoraAsyncClient.from_env() factory method."""

    def test_async_from_env_uses_defaults(self) -> None:
        """Async from_env() uses default values when no env vars are set."""
        env = {}
        with patch.dict(os.environ, env, clear=True):
            client = AragoraAsyncClient.from_env()
            assert client.base_url == "http://localhost:8080"
            assert client.api_key is None
            assert client.ws_url is None
            assert client.timeout == 30.0

    def test_async_from_env_reads_ws_url(self) -> None:
        """Async from_env() reads ARAGORA_WS_URL."""
        with patch.dict(os.environ, {"ARAGORA_WS_URL": "wss://ws.aragora.ai"}):
            client = AragoraAsyncClient.from_env()
            assert client.ws_url == "wss://ws.aragora.ai"

    def test_async_from_env_reads_all_vars(self) -> None:
        """Async from_env() reads all supported environment variables."""
        env = {
            "ARAGORA_API_URL": "https://async.aragora.ai",
            "ARAGORA_API_KEY": "ara_async_key",
            "ARAGORA_WS_URL": "wss://ws.async.aragora.ai",
            "ARAGORA_TIMEOUT": "45",
            "ARAGORA_MAX_RETRIES": "7",
            "ARAGORA_RETRY_DELAY": "0.5",
        }
        with patch.dict(os.environ, env):
            client = AragoraAsyncClient.from_env()
            assert client.base_url == "https://async.aragora.ai"
            assert client.api_key == "ara_async_key"
            assert client.ws_url == "wss://ws.async.aragora.ai"
            assert client.timeout == 45.0
            assert client.max_retries == 7
            assert client.retry_delay == 0.5

    def test_async_from_env_kwargs_override_env(self) -> None:
        """Explicit kwargs override environment variables for async client."""
        with patch.dict(os.environ, {"ARAGORA_API_URL": "https://env.aragora.ai"}):
            client = AragoraAsyncClient.from_env(base_url="https://override.aragora.ai")
            assert client.base_url == "https://override.aragora.ai"

    def test_async_from_env_returns_async_client(self) -> None:
        """Async from_env() returns an AragoraAsyncClient instance."""
        client = AragoraAsyncClient.from_env()
        assert isinstance(client, AragoraAsyncClient)
