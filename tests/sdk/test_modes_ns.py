"""Tests for the Modes SDK namespace."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("aragora_sdk", reason="aragora-sdk not installed")

from aragora_sdk.namespaces.modes import AsyncModesAPI, ModesAPI  # noqa: E402


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.request.return_value = {"modes": []}
    return client


@pytest.fixture
def api(mock_client):
    return ModesAPI(mock_client)


class TestModesAPI:
    def test_list_modes(self, api, mock_client):
        result = api.list_modes()
        mock_client.request.assert_called_once_with("GET", "/api/v1/modes")
        assert result == {"modes": []}

    def test_get_mode(self, api, mock_client):
        api.get_mode("architect")
        mock_client.request.assert_called_once_with("GET", "/api/v1/modes/architect")

    def test_get_mode_coder(self, api, mock_client):
        api.get_mode("coder")
        mock_client.request.assert_called_once_with("GET", "/api/v1/modes/coder")

    def test_get_mode_reviewer(self, api, mock_client):
        api.get_mode("reviewer")
        mock_client.request.assert_called_once_with("GET", "/api/v1/modes/reviewer")

    def test_async_class_exists(self):
        """Verify AsyncModesAPI can be instantiated."""
        mock_client = MagicMock()
        async_api = AsyncModesAPI(mock_client)
        assert async_api is not None
