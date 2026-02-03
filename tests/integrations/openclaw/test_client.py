"""Tests for integrations/openclaw/client.py — OpenClaw API client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from aragora.integrations.openclaw.client import (
    OpenClawClient,
    OpenClawConfig,
    OpenClawResponse,
)


# =============================================================================
# Dataclasses
# =============================================================================


class TestOpenClawConfig:
    def test_defaults(self):
        cfg = OpenClawConfig()
        assert cfg.base_url == "http://localhost:8080"
        assert cfg.api_key is None
        assert cfg.timeout_seconds == 30
        assert cfg.verify_ssl is True
        assert cfg.max_retries == 3

    def test_custom(self):
        cfg = OpenClawConfig(
            base_url="https://claw.example.com",
            api_key="key-1",
            auth_token="tok-1",
            max_retries=5,
        )
        assert cfg.base_url == "https://claw.example.com"
        assert cfg.api_key == "key-1"
        assert cfg.auth_token == "tok-1"


class TestOpenClawResponse:
    def test_success(self):
        resp = OpenClawResponse(success=True, data={"result": "ok"}, status_code=200)
        assert resp.success is True
        assert resp.data["result"] == "ok"

    def test_error(self):
        resp = OpenClawResponse(success=False, error="timeout", status_code=504)
        assert resp.success is False
        assert resp.error == "timeout"


# =============================================================================
# OpenClawClient — init
# =============================================================================


class TestClientInit:
    def test_defaults(self):
        client = OpenClawClient()
        assert client._config.base_url == "http://localhost:8080"
        assert client._session is None
        assert client._ws is None

    def test_custom_config(self):
        cfg = OpenClawConfig(base_url="http://custom:9090")
        client = OpenClawClient(config=cfg)
        assert client._config.base_url == "http://custom:9090"

    def test_event_callback(self):
        cb = MagicMock()
        client = OpenClawClient(event_callback=cb)
        assert client._event_callback is cb


# =============================================================================
# _get_headers
# =============================================================================


class TestGetHeaders:
    def test_no_auth(self):
        client = OpenClawClient()
        headers = client._get_headers()
        assert headers["Content-Type"] == "application/json"
        assert "X-API-Key" not in headers
        assert "Authorization" not in headers

    def test_api_key(self):
        client = OpenClawClient(config=OpenClawConfig(api_key="k-1"))
        headers = client._get_headers()
        assert headers["X-API-Key"] == "k-1"

    def test_auth_token(self):
        client = OpenClawClient(config=OpenClawConfig(auth_token="tok"))
        headers = client._get_headers()
        assert headers["Authorization"] == "Bearer tok"

    def test_both_keys(self):
        client = OpenClawClient(config=OpenClawConfig(api_key="k", auth_token="t"))
        headers = client._get_headers()
        assert "X-API-Key" in headers
        assert "Authorization" in headers


# =============================================================================
# get_stats
# =============================================================================


class TestGetStats:
    def test_initial_stats(self):
        client = OpenClawClient()
        stats = client.get_stats()
        assert stats["requests_made"] == 0
        assert stats["avg_latency_ms"] == 0.0
        assert stats["connected"] is False
        assert stats["ws_connected"] is False


# =============================================================================
# close
# =============================================================================


class TestClose:
    @pytest.mark.asyncio
    async def test_close_no_session(self):
        client = OpenClawClient()
        await client.close()  # should not raise

    @pytest.mark.asyncio
    async def test_close_with_session(self):
        client = OpenClawClient()
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.closed = False
        client._session = mock_session
        await client.close()
        mock_session.close.assert_called_once()
        assert client._session is None

    @pytest.mark.asyncio
    async def test_close_with_ws(self):
        client = OpenClawClient()
        mock_ws = AsyncMock()
        client._ws = mock_ws
        client._ws_connected = True
        await client.close()
        mock_ws.close.assert_called_once()
        assert client._ws is None
        assert client._ws_connected is False


# =============================================================================
# Context manager
# =============================================================================


class TestContextManager:
    @pytest.mark.asyncio
    async def test_aenter_aexit(self):
        async with OpenClawClient() as client:
            assert isinstance(client, OpenClawClient)


# =============================================================================
# API methods (test they call _request correctly)
# =============================================================================


class TestAPIMethods:
    @pytest.fixture()
    def client(self):
        return OpenClawClient(config=OpenClawConfig(base_url="http://test:8080"))

    @pytest.mark.asyncio
    async def test_execute_shell(self, client):
        client._request = AsyncMock(return_value=OpenClawResponse(success=True, data="output"))
        result = await client.execute_shell("ls -la")
        client._request.assert_called_once_with("POST", "/api/shell", {"command": "ls -la"})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_read_file(self, client):
        client._request = AsyncMock(return_value=OpenClawResponse(success=True, data="content"))
        result = await client.read_file("/tmp/file.txt")
        client._request.assert_called_once_with("GET", "/api/files?path=/tmp/file.txt")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_write_file(self, client):
        client._request = AsyncMock(return_value=OpenClawResponse(success=True))
        result = await client.write_file("/tmp/file.txt", "hello")
        client._request.assert_called_once_with(
            "POST", "/api/files", {"path": "/tmp/file.txt", "content": "hello"}
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_delete_file(self, client):
        client._request = AsyncMock(return_value=OpenClawResponse(success=True))
        result = await client.delete_file("/tmp/file.txt")
        client._request.assert_called_once_with("DELETE", "/api/files?path=/tmp/file.txt")

    @pytest.mark.asyncio
    async def test_navigate(self, client):
        client._request = AsyncMock(return_value=OpenClawResponse(success=True))
        await client.navigate("https://example.com")
        client._request.assert_called_once_with(
            "POST", "/api/browser/navigate", {"url": "https://example.com"}
        )

    @pytest.mark.asyncio
    async def test_screenshot(self, client):
        client._request = AsyncMock(return_value=OpenClawResponse(success=True, data="base64..."))
        result = await client.screenshot()
        client._request.assert_called_once_with("GET", "/api/browser/screenshot")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_type_text(self, client):
        client._request = AsyncMock(return_value=OpenClawResponse(success=True))
        await client.type_text("hello")
        client._request.assert_called_once_with("POST", "/api/input/keyboard", {"text": "hello"})

    @pytest.mark.asyncio
    async def test_click(self, client):
        client._request = AsyncMock(return_value=OpenClawResponse(success=True))
        await client.click(100, 200)
        client._request.assert_called_once_with(
            "POST", "/api/input/mouse", {"action": "click", "x": 100, "y": 200}
        )

    @pytest.mark.asyncio
    async def test_api_call(self, client):
        client._request = AsyncMock(return_value=OpenClawResponse(success=True))
        await client.api_call("https://api.example.com/data", params={"key": "val"})
        client._request.assert_called_once_with(
            "POST", "/api/proxy", {"url": "https://api.example.com/data", "params": {"key": "val"}}
        )

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        client._request = AsyncMock(
            return_value=OpenClawResponse(success=True, data={"status": "healthy"})
        )
        result = await client.health_check()
        client._request.assert_called_once_with("GET", "/health")
        assert result.data["status"] == "healthy"
