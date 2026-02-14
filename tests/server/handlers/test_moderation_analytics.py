"""
Tests for Moderation Analytics Dashboard Handler.

Covers:
- GET /api/v1/moderation/stats (block rate, queue size, false positive rate)
- GET /api/v1/moderation/queue (pending review items)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.moderation_analytics import ModerationAnalyticsHandler


# ============================================================================
# Fixtures
# ============================================================================


def _make_mock_handler(method="GET"):
    mock = MagicMock()
    mock.command = method
    mock.rfile = MagicMock()
    mock.rfile.read = MagicMock(return_value=b"{}")
    mock.headers = {"Content-Length": "2"}
    return mock


def _parse(result) -> dict[str, Any]:
    if result is None:
        return {}
    body = result.body
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    return json.loads(body) if body else {}


@pytest.fixture
def handler():
    return ModerationAnalyticsHandler(ctx={})


# ============================================================================
# Route Tests
# ============================================================================


class TestRouting:
    def test_can_handle_stats(self, handler):
        assert handler.can_handle("/api/v1/moderation/stats") is True

    def test_can_handle_queue(self, handler):
        assert handler.can_handle("/api/v1/moderation/queue") is True

    def test_cannot_handle_other_paths(self, handler):
        assert handler.can_handle("/api/v1/moderation/config") is False
        assert handler.can_handle("/api/v1/other") is False


# ============================================================================
# GET /api/v1/moderation/stats
# ============================================================================


class TestModerationStats:
    @patch("aragora.server.handlers.moderation_analytics._get_moderation")
    @patch("aragora.server.handlers.moderation_analytics._get_queue_size")
    def test_stats_with_moderation(self, mock_queue_size, mock_moderation, handler):
        mock_mod = MagicMock()
        mock_mod.statistics = {
            "total_checks": 1000,
            "blocked": 50,
            "flagged": 30,
            "clean": 920,
            "false_positives": 5,
        }
        mock_moderation.return_value = mock_mod
        mock_queue_size.return_value = 12

        http = _make_mock_handler()
        result = handler.handle("/api/v1/moderation/stats", {}, http)
        assert result.status_code == 200
        body = _parse(result)
        assert body["total_checks"] == 1000
        assert body["blocked_count"] == 50
        assert body["flagged_count"] == 30
        assert body["clean_count"] == 920
        assert body["block_rate"] == 0.05
        assert body["false_positive_rate"] == 0.1
        assert body["queue_size"] == 12
        assert body["available"] is True

    @patch("aragora.server.handlers.moderation_analytics._get_moderation")
    def test_stats_without_moderation(self, mock_moderation, handler):
        mock_moderation.return_value = None
        http = _make_mock_handler()
        result = handler.handle("/api/v1/moderation/stats", {}, http)
        assert result.status_code == 200
        body = _parse(result)
        assert body["available"] is False
        assert body["total_checks"] == 0
        assert body["block_rate"] == 0.0

    @patch("aragora.server.handlers.moderation_analytics._get_moderation")
    @patch("aragora.server.handlers.moderation_analytics._get_queue_size")
    def test_stats_zero_total(self, mock_queue_size, mock_moderation, handler):
        mock_mod = MagicMock()
        mock_mod.statistics = {"total_checks": 0}
        mock_moderation.return_value = mock_mod
        mock_queue_size.return_value = 0

        http = _make_mock_handler()
        result = handler.handle("/api/v1/moderation/stats", {}, http)
        body = _parse(result)
        assert body["block_rate"] == 0.0
        assert body["false_positive_rate"] == 0.0

    @patch("aragora.server.handlers.moderation_analytics._get_moderation")
    @patch("aragora.server.handlers.moderation_analytics._get_queue_size")
    def test_stats_alternative_key_names(self, mock_queue_size, mock_moderation, handler):
        """Test stats extraction with alternative key naming."""
        mock_mod = MagicMock()
        mock_mod.statistics = {
            "total_checks": 500,
            "blocked_count": 25,
            "flagged_count": 10,
            "clean_count": 465,
        }
        mock_moderation.return_value = mock_mod
        mock_queue_size.return_value = 0

        http = _make_mock_handler()
        result = handler.handle("/api/v1/moderation/stats", {}, http)
        body = _parse(result)
        assert body["blocked_count"] == 25

    def test_returns_none_for_unhandled_path(self, handler):
        http = _make_mock_handler()
        result = handler.handle("/api/v1/other", {}, http)
        assert result is None


# ============================================================================
# GET /api/v1/moderation/queue
# ============================================================================


class TestModerationQueue:
    @patch("aragora.server.handlers.moderation_analytics._list_queue")
    def test_queue_with_items(self, mock_list, handler):
        mock_item = MagicMock()
        mock_item.to_dict.return_value = {
            "id": "item-001",
            "content": "Suspicious content",
            "verdict": "suspicious",
        }
        mock_list.return_value = [mock_item]

        http = _make_mock_handler()
        result = handler.handle("/api/v1/moderation/queue", {}, http)
        assert result.status_code == 200
        body = _parse(result)
        assert body["count"] == 1
        assert body["items"][0]["id"] == "item-001"

    @patch("aragora.server.handlers.moderation_analytics._list_queue")
    def test_queue_empty(self, mock_list, handler):
        mock_list.return_value = []
        http = _make_mock_handler()
        result = handler.handle("/api/v1/moderation/queue", {}, http)
        body = _parse(result)
        assert body["count"] == 0
        assert body["items"] == []

    @patch("aragora.server.handlers.moderation_analytics._list_queue")
    def test_queue_with_limit(self, mock_list, handler):
        mock_list.return_value = []
        http = _make_mock_handler()
        result = handler.handle("/api/v1/moderation/queue", {"limit": "10", "offset": "5"}, http)
        body = _parse(result)
        assert body["limit"] == 10
        assert body["offset"] == 5
        mock_list.assert_called_once_with(limit=10, offset=5)

    @patch("aragora.server.handlers.moderation_analytics._list_queue")
    def test_queue_limit_capped(self, mock_list, handler):
        mock_list.return_value = []
        http = _make_mock_handler()
        result = handler.handle("/api/v1/moderation/queue", {"limit": "500"}, http)
        body = _parse(result)
        assert body["limit"] == 200  # Capped at 200

    @patch("aragora.server.handlers.moderation_analytics._list_queue")
    def test_queue_with_dict_items(self, mock_list, handler):
        mock_list.return_value = [
            {"id": "item-001", "content": "Test"},
        ]
        http = _make_mock_handler()
        result = handler.handle("/api/v1/moderation/queue", {}, http)
        body = _parse(result)
        assert body["items"][0]["id"] == "item-001"
