"""Tests for the Moderation SDK namespace."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aragora_sdk.namespaces.moderation import AsyncModerationAPI, ModerationAPI


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.request.return_value = {"status": "ok"}
    return client


@pytest.fixture
def api(mock_client):
    return ModerationAPI(mock_client)


class TestModerationAPI:
    def test_get_config(self, api, mock_client):
        result = api.get_config()
        mock_client.request.assert_called_once_with("GET", "/api/v1/moderation/config")
        assert result == {"status": "ok"}

    def test_update_config(self, api, mock_client):
        config = {"spam_threshold": 0.8}
        api.update_config(config)
        mock_client.request.assert_called_once_with("PUT", "/api/v1/moderation/config", json=config)

    def test_get_stats(self, api, mock_client):
        api.get_stats()
        mock_client.request.assert_called_once_with("GET", "/api/v1/moderation/stats")

    def test_get_queue(self, api, mock_client):
        api.get_queue()
        mock_client.request.assert_called_once_with("GET", "/api/v1/moderation/queue")

    def test_approve_item(self, api, mock_client):
        api.approve_item("item-1")
        mock_client.request.assert_called_once_with("POST", "/api/v1/moderation/queue/item-1/approve")

    def test_reject_item(self, api, mock_client):
        api.reject_item("item-2")
        mock_client.request.assert_called_once_with("POST", "/api/v1/moderation/queue/item-2/reject")
