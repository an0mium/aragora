"""Tests for AragoraClient and AragoraAsyncClient initialization and configuration."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient
from aragora.namespaces.agents import AgentsAPI, AsyncAgentsAPI
from aragora.namespaces.control_plane import AsyncControlPlaneAPI, ControlPlaneAPI
from aragora.namespaces.debates import AsyncDebatesAPI, DebatesAPI


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
