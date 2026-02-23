"""Tests for the Spectate SDK namespace."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("aragora_sdk", reason="aragora-sdk not installed")

from aragora_sdk.namespaces.spectate import AsyncSpectateAPI, SpectateAPI  # noqa: E402


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.request.return_value = {"stream_url": "https://api.aragora.ai/sse/debate-1"}
    return client


@pytest.fixture
def api(mock_client):
    return SpectateAPI(mock_client)


class TestSpectateAPI:
    def test_connect_sse(self, api, mock_client):
        result = api.connect_sse("debate-123")
        mock_client.request.assert_called_once_with("GET", "/api/v1/spectate/debate-123/stream")
        assert "stream_url" in result

    def test_connect_sse_different_debate(self, api, mock_client):
        api.connect_sse("debate-456")
        mock_client.request.assert_called_once_with("GET", "/api/v1/spectate/debate-456/stream")

    def test_async_class_exists(self):
        """Verify AsyncSpectateAPI can be instantiated."""
        mock_client = MagicMock()
        async_api = AsyncSpectateAPI(mock_client)
        assert async_api is not None

    def test_sync_api_init(self):
        """Verify SpectateAPI stores client reference."""
        mock_client = MagicMock()
        api = SpectateAPI(mock_client)
        assert api._client is mock_client

    def test_connect_returns_response(self, api, mock_client):
        mock_client.request.return_value = {"stream_url": "/sse/test", "debate_id": "test"}
        result = api.connect_sse("test")
        assert result["debate_id"] == "test"
