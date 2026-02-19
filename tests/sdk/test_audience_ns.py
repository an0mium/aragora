"""Tests for the Audience SDK namespace."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("aragora_sdk", reason="aragora-sdk not installed")

from aragora_sdk.namespaces.audience import AsyncAudienceAPI, AudienceAPI  # noqa: E402


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.request.return_value = {"suggestions": []}
    return client


@pytest.fixture
def api(mock_client):
    return AudienceAPI(mock_client)


class TestAudienceAPI:
    def test_get_suggestions(self, api, mock_client):
        result = api.get_suggestions("debate-123")
        mock_client.request.assert_called_once_with(
            "GET", "/api/v1/debates/debate-123/audience/suggestions"
        )
        assert result == {"suggestions": []}

    def test_submit_suggestion(self, api, mock_client):
        suggestion = {"text": "Consider risk factors", "author": "user-1"}
        api.submit_suggestion("debate-456", suggestion)
        mock_client.request.assert_called_once_with(
            "POST",
            "/api/v1/debates/debate-456/audience/suggestions",
            json=suggestion,
        )

    def test_get_suggestions_different_debate(self, api, mock_client):
        api.get_suggestions("debate-789")
        mock_client.request.assert_called_once_with(
            "GET", "/api/v1/debates/debate-789/audience/suggestions"
        )

    def test_submit_empty_suggestion(self, api, mock_client):
        api.submit_suggestion("debate-1", {})
        mock_client.request.assert_called_once_with(
            "POST",
            "/api/v1/debates/debate-1/audience/suggestions",
            json={},
        )

    def test_async_class_exists(self):
        """Verify AsyncAudienceAPI can be instantiated."""
        mock_client = MagicMock()
        async_api = AsyncAudienceAPI(mock_client)
        assert async_api is not None
