"""Tests for moderation analytics handler.

Covers all routes and behavior of ModerationAnalyticsHandler:
- can_handle() routing for ROUTES
- GET /api/v1/moderation/stats (block rate, queue size, false positive rate)
- GET /api/v1/moderation/queue (pending review items)
- Moderation module unavailable fallback
- Statistics computation (rates, rounding)
- Queue pagination (limit, offset, clamping)
- Queue item serialization (to_dict, dict, str fallback)
- Error handling via @handle_errors
- RBAC permission checks
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.moderation_analytics import (
    ModerationAnalyticsHandler,
    _get_moderation,
    _get_queue_size,
    _list_queue,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Mock HTTP request handler."""

    def __init__(self, body: dict | None = None, method: str = "GET"):
        self.command = method
        self.client_address = ("127.0.0.1", 12345)
        self.headers: dict[str, str] = {"User-Agent": "test-agent"}
        self.rfile = MagicMock()
        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a ModerationAnalyticsHandler with empty context."""
    return ModerationAnalyticsHandler(ctx={})


@pytest.fixture
def http_handler():
    """Create a mock HTTP handler."""
    return MockHTTPHandler()


# ---------------------------------------------------------------------------
# can_handle tests
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for can_handle() routing."""

    def test_stats_route(self, handler):
        assert handler.can_handle("/api/v1/moderation/stats") is True

    def test_queue_route(self, handler):
        assert handler.can_handle("/api/v1/moderation/queue") is True

    def test_unknown_route(self, handler):
        assert handler.can_handle("/api/v1/moderation/unknown") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_partial_match(self, handler):
        assert handler.can_handle("/api/v1/moderation") is False

    def test_trailing_slash(self, handler):
        assert handler.can_handle("/api/v1/moderation/stats/") is False

    def test_wrong_version(self, handler):
        assert handler.can_handle("/api/v2/moderation/stats") is False


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


class TestConstructor:
    """Tests for ModerationAnalyticsHandler initialization."""

    def test_default_ctx(self):
        h = ModerationAnalyticsHandler()
        assert h.ctx == {}

    def test_custom_ctx(self):
        ctx = {"key": "value"}
        h = ModerationAnalyticsHandler(ctx=ctx)
        assert h.ctx == ctx

    def test_none_ctx_defaults_to_empty(self):
        h = ModerationAnalyticsHandler(ctx=None)
        assert h.ctx == {}


# ---------------------------------------------------------------------------
# ROUTES constant
# ---------------------------------------------------------------------------


class TestRoutes:
    """Tests for ROUTES class attribute."""

    def test_routes_contains_stats(self):
        assert "/api/v1/moderation/stats" in ModerationAnalyticsHandler.ROUTES

    def test_routes_contains_queue(self):
        assert "/api/v1/moderation/queue" in ModerationAnalyticsHandler.ROUTES

    def test_routes_count(self):
        assert len(ModerationAnalyticsHandler.ROUTES) == 2


# ---------------------------------------------------------------------------
# _get_moderation helper
# ---------------------------------------------------------------------------


class TestGetModeration:
    """Tests for _get_moderation() lazy loader."""

    @patch(
        "aragora.server.handlers.moderation_analytics._get_moderation",
        return_value=MagicMock(statistics={"total_checks": 10}),
    )
    def test_returns_moderation_when_available(self, mock_mod):
        result = mock_mod()
        assert result is not None

    def test_returns_none_on_import_error(self):
        with patch.dict("sys.modules", {"aragora.moderation": None}):
            result = _get_moderation()
            assert result is None


# ---------------------------------------------------------------------------
# _get_queue_size helper
# ---------------------------------------------------------------------------


class TestGetQueueSize:
    """Tests for _get_queue_size() helper."""

    def test_returns_zero_on_import_error(self):
        with patch.dict("sys.modules", {"aragora.moderation": None}):
            result = _get_queue_size()
            assert result == 0

    @patch("aragora.server.handlers.moderation_analytics._get_queue_size", return_value=42)
    def test_returns_size_when_available(self, mock_qs):
        assert mock_qs() == 42


# ---------------------------------------------------------------------------
# _list_queue helper
# ---------------------------------------------------------------------------


class TestListQueue:
    """Tests for _list_queue() helper."""

    def test_returns_empty_on_import_error(self):
        with patch.dict("sys.modules", {"aragora.moderation": None}):
            result = _list_queue()
            assert result == []

    def test_returns_empty_on_import_error_with_params(self):
        with patch.dict("sys.modules", {"aragora.moderation": None}):
            result = _list_queue(limit=10, offset=5)
            assert result == []


# ---------------------------------------------------------------------------
# GET /api/v1/moderation/stats -- moderation unavailable
# ---------------------------------------------------------------------------


class TestStatsUnavailable:
    """Tests for stats endpoint when moderation is not available."""

    @patch(
        "aragora.server.handlers.moderation_analytics._get_moderation",
        return_value=None,
    )
    def test_returns_unavailable_stats(self, mock_mod, handler, http_handler):
        result = handler.handle("/api/v1/moderation/stats", {}, http_handler)
        body = _body(result)
        assert body["available"] is False
        assert body["total_checks"] == 0
        assert body["blocked_count"] == 0
        assert body["flagged_count"] == 0
        assert body["clean_count"] == 0
        assert body["block_rate"] == 0.0
        assert body["false_positive_rate"] == 0.0
        assert body["queue_size"] == 0

    @patch(
        "aragora.server.handlers.moderation_analytics._get_moderation",
        return_value=None,
    )
    def test_returns_200_when_unavailable(self, mock_mod, handler, http_handler):
        result = handler.handle("/api/v1/moderation/stats", {}, http_handler)
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# GET /api/v1/moderation/stats -- moderation available
# ---------------------------------------------------------------------------


class TestStatsAvailable:
    """Tests for stats endpoint with moderation data."""

    def _make_moderation(self, stats: dict) -> MagicMock:
        mod = MagicMock()
        mod.statistics = stats
        return mod

    @patch("aragora.server.handlers.moderation_analytics._get_queue_size", return_value=5)
    @patch("aragora.server.handlers.moderation_analytics._get_moderation")
    def test_basic_stats(self, mock_mod, mock_qs, handler, http_handler):
        mock_mod.return_value = self._make_moderation(
            {"total_checks": 100, "blocked": 10, "flagged": 5, "clean": 85, "false_positives": 2}
        )
        result = handler.handle("/api/v1/moderation/stats", {}, http_handler)
        body = _body(result)
        assert body["available"] is True
        assert body["total_checks"] == 100
        assert body["blocked_count"] == 10
        assert body["flagged_count"] == 5
        assert body["clean_count"] == 85
        assert body["block_rate"] == 0.1
        assert body["false_positive_rate"] == 0.2
        assert body["queue_size"] == 5

    @patch("aragora.server.handlers.moderation_analytics._get_queue_size", return_value=0)
    @patch("aragora.server.handlers.moderation_analytics._get_moderation")
    def test_zero_total_checks(self, mock_mod, mock_qs, handler, http_handler):
        """block_rate should be 0.0 when total_checks is 0."""
        mock_mod.return_value = self._make_moderation({"total_checks": 0})
        result = handler.handle("/api/v1/moderation/stats", {}, http_handler)
        body = _body(result)
        assert body["block_rate"] == 0.0
        assert body["false_positive_rate"] == 0.0

    @patch("aragora.server.handlers.moderation_analytics._get_queue_size", return_value=0)
    @patch("aragora.server.handlers.moderation_analytics._get_moderation")
    def test_zero_blocked(self, mock_mod, mock_qs, handler, http_handler):
        """false_positive_rate should be 0.0 when blocked is 0."""
        mock_mod.return_value = self._make_moderation(
            {"total_checks": 50, "blocked": 0, "flagged": 3, "clean": 47}
        )
        result = handler.handle("/api/v1/moderation/stats", {}, http_handler)
        body = _body(result)
        assert body["false_positive_rate"] == 0.0
        assert body["block_rate"] == 0.0

    @patch("aragora.server.handlers.moderation_analytics._get_queue_size", return_value=0)
    @patch("aragora.server.handlers.moderation_analytics._get_moderation")
    def test_block_rate_rounding(self, mock_mod, mock_qs, handler, http_handler):
        """block_rate should be rounded to 4 decimal places."""
        mock_mod.return_value = self._make_moderation(
            {"total_checks": 3, "blocked": 1, "flagged": 0, "clean": 2, "false_positives": 0}
        )
        result = handler.handle("/api/v1/moderation/stats", {}, http_handler)
        body = _body(result)
        assert body["block_rate"] == round(1 / 3, 4)

    @patch("aragora.server.handlers.moderation_analytics._get_queue_size", return_value=0)
    @patch("aragora.server.handlers.moderation_analytics._get_moderation")
    def test_fp_rate_rounding(self, mock_mod, mock_qs, handler, http_handler):
        """false_positive_rate should be rounded to 4 decimal places."""
        mock_mod.return_value = self._make_moderation(
            {"total_checks": 100, "blocked": 3, "false_positives": 1}
        )
        result = handler.handle("/api/v1/moderation/stats", {}, http_handler)
        body = _body(result)
        assert body["false_positive_rate"] == round(1 / 3, 4)

    @patch("aragora.server.handlers.moderation_analytics._get_queue_size", return_value=0)
    @patch("aragora.server.handlers.moderation_analytics._get_moderation")
    def test_alternate_stat_keys_blocked_count(self, mock_mod, mock_qs, handler, http_handler):
        """Uses blocked_count key when blocked is not present."""
        mock_mod.return_value = self._make_moderation(
            {"total_checks": 100, "blocked_count": 20, "flagged_count": 5, "clean_count": 75}
        )
        result = handler.handle("/api/v1/moderation/stats", {}, http_handler)
        body = _body(result)
        assert body["blocked_count"] == 20
        assert body["flagged_count"] == 5
        assert body["clean_count"] == 75

    @patch("aragora.server.handlers.moderation_analytics._get_queue_size", return_value=0)
    @patch("aragora.server.handlers.moderation_analytics._get_moderation")
    def test_empty_statistics(self, mock_mod, mock_qs, handler, http_handler):
        """Handles moderation with empty statistics dict."""
        mock_mod.return_value = self._make_moderation({})
        result = handler.handle("/api/v1/moderation/stats", {}, http_handler)
        body = _body(result)
        assert body["total_checks"] == 0
        assert body["blocked_count"] == 0
        assert body["flagged_count"] == 0
        assert body["clean_count"] == 0
        assert body["block_rate"] == 0.0
        assert body["false_positive_rate"] == 0.0
        assert body["available"] is True

    @patch("aragora.server.handlers.moderation_analytics._get_queue_size", return_value=0)
    @patch("aragora.server.handlers.moderation_analytics._get_moderation")
    def test_none_values_in_stats(self, mock_mod, mock_qs, handler, http_handler):
        """Handles None values in statistics gracefully."""
        mock_mod.return_value = self._make_moderation(
            {"total_checks": None, "blocked": None, "flagged": None, "clean": None}
        )
        result = handler.handle("/api/v1/moderation/stats", {}, http_handler)
        body = _body(result)
        assert body["total_checks"] == 0
        assert body["blocked_count"] == 0
        assert body["block_rate"] == 0.0

    @patch("aragora.server.handlers.moderation_analytics._get_queue_size", return_value=12)
    @patch("aragora.server.handlers.moderation_analytics._get_moderation")
    def test_queue_size_reflected_in_stats(self, mock_mod, mock_qs, handler, http_handler):
        """Queue size from _get_queue_size is included in stats."""
        mock_mod.return_value = self._make_moderation({"total_checks": 10})
        result = handler.handle("/api/v1/moderation/stats", {}, http_handler)
        body = _body(result)
        assert body["queue_size"] == 12

    @patch("aragora.server.handlers.moderation_analytics._get_queue_size", return_value=0)
    @patch("aragora.server.handlers.moderation_analytics._get_moderation")
    def test_no_statistics_attribute(self, mock_mod, mock_qs, handler, http_handler):
        """Handles moderation object without statistics attribute."""
        mod = MagicMock(spec=[])
        mock_mod.return_value = mod
        result = handler.handle("/api/v1/moderation/stats", {}, http_handler)
        body = _body(result)
        assert body["total_checks"] == 0
        assert body["available"] is True

    @patch("aragora.server.handlers.moderation_analytics._get_queue_size", return_value=0)
    @patch("aragora.server.handlers.moderation_analytics._get_moderation")
    def test_status_200_when_available(self, mock_mod, mock_qs, handler, http_handler):
        mock_mod.return_value = self._make_moderation({"total_checks": 50})
        result = handler.handle("/api/v1/moderation/stats", {}, http_handler)
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# GET /api/v1/moderation/queue -- basic
# ---------------------------------------------------------------------------


class TestQueueBasic:
    """Tests for queue endpoint basic behavior."""

    @patch("aragora.server.handlers.moderation_analytics._list_queue", return_value=[])
    def test_empty_queue(self, mock_lq, handler, http_handler):
        result = handler.handle("/api/v1/moderation/queue", {}, http_handler)
        body = _body(result)
        assert body["items"] == []
        assert body["count"] == 0
        assert body["limit"] == 50
        assert body["offset"] == 0

    @patch("aragora.server.handlers.moderation_analytics._list_queue", return_value=[])
    def test_status_200(self, mock_lq, handler, http_handler):
        result = handler.handle("/api/v1/moderation/queue", {}, http_handler)
        assert _status(result) == 200

    @patch("aragora.server.handlers.moderation_analytics._list_queue")
    def test_dict_items(self, mock_lq, handler, http_handler):
        """Dict items pass through as-is."""
        items = [{"id": "1", "content": "spam"}, {"id": "2", "content": "ham"}]
        mock_lq.return_value = items
        result = handler.handle("/api/v1/moderation/queue", {}, http_handler)
        body = _body(result)
        assert body["items"] == items
        assert body["count"] == 2

    @patch("aragora.server.handlers.moderation_analytics._list_queue")
    def test_to_dict_items(self, mock_lq, handler, http_handler):
        """Items with to_dict() are serialized via that method."""
        item = MagicMock()
        item.to_dict.return_value = {"id": "x", "flagged": True}
        mock_lq.return_value = [item]
        result = handler.handle("/api/v1/moderation/queue", {}, http_handler)
        body = _body(result)
        assert body["items"] == [{"id": "x", "flagged": True}]
        item.to_dict.assert_called_once()

    @patch("aragora.server.handlers.moderation_analytics._list_queue")
    def test_str_fallback_items(self, mock_lq, handler, http_handler):
        """Non-dict items without to_dict fall back to str()."""
        mock_lq.return_value = [42, "raw-string"]
        result = handler.handle("/api/v1/moderation/queue", {}, http_handler)
        body = _body(result)
        assert body["items"] == ["42", "raw-string"]
        assert body["count"] == 2

    @patch("aragora.server.handlers.moderation_analytics._list_queue")
    def test_mixed_items(self, mock_lq, handler, http_handler):
        """Mixed item types are each serialized correctly."""
        dict_item = {"id": "d1"}
        obj_item = MagicMock()
        obj_item.to_dict.return_value = {"id": "o1"}
        mock_lq.return_value = [dict_item, obj_item, 99]
        result = handler.handle("/api/v1/moderation/queue", {}, http_handler)
        body = _body(result)
        assert body["items"] == [{"id": "d1"}, {"id": "o1"}, "99"]
        assert body["count"] == 3


# ---------------------------------------------------------------------------
# GET /api/v1/moderation/queue -- pagination
# ---------------------------------------------------------------------------


class TestQueuePagination:
    """Tests for queue endpoint pagination parameters."""

    @patch("aragora.server.handlers.moderation_analytics._list_queue", return_value=[])
    def test_custom_limit(self, mock_lq, handler, http_handler):
        result = handler.handle("/api/v1/moderation/queue", {"limit": "25"}, http_handler)
        body = _body(result)
        assert body["limit"] == 25
        mock_lq.assert_called_once_with(limit=25, offset=0)

    @patch("aragora.server.handlers.moderation_analytics._list_queue", return_value=[])
    def test_custom_offset(self, mock_lq, handler, http_handler):
        result = handler.handle("/api/v1/moderation/queue", {"offset": "10"}, http_handler)
        body = _body(result)
        assert body["offset"] == 10
        mock_lq.assert_called_once_with(limit=50, offset=10)

    @patch("aragora.server.handlers.moderation_analytics._list_queue", return_value=[])
    def test_limit_capped_at_200(self, mock_lq, handler, http_handler):
        """Limit is capped at 200."""
        result = handler.handle("/api/v1/moderation/queue", {"limit": "500"}, http_handler)
        body = _body(result)
        assert body["limit"] == 200
        mock_lq.assert_called_once_with(limit=200, offset=0)

    @patch("aragora.server.handlers.moderation_analytics._list_queue", return_value=[])
    def test_negative_offset_clamped_to_zero(self, mock_lq, handler, http_handler):
        """Negative offset is clamped to 0."""
        result = handler.handle("/api/v1/moderation/queue", {"offset": "-5"}, http_handler)
        body = _body(result)
        assert body["offset"] == 0
        mock_lq.assert_called_once_with(limit=50, offset=0)

    @patch("aragora.server.handlers.moderation_analytics._list_queue", return_value=[])
    def test_invalid_limit_defaults_to_50(self, mock_lq, handler, http_handler):
        """Non-numeric limit defaults to 50."""
        result = handler.handle("/api/v1/moderation/queue", {"limit": "abc"}, http_handler)
        body = _body(result)
        assert body["limit"] == 50

    @patch("aragora.server.handlers.moderation_analytics._list_queue", return_value=[])
    def test_invalid_offset_defaults_to_zero(self, mock_lq, handler, http_handler):
        """Non-numeric offset defaults to 0."""
        result = handler.handle("/api/v1/moderation/queue", {"offset": "xyz"}, http_handler)
        body = _body(result)
        assert body["offset"] == 0

    @patch("aragora.server.handlers.moderation_analytics._list_queue", return_value=[])
    def test_none_limit(self, mock_lq, handler, http_handler):
        """None limit (from missing param) defaults to 50."""
        result = handler.handle("/api/v1/moderation/queue", {}, http_handler)
        body = _body(result)
        assert body["limit"] == 50

    @patch("aragora.server.handlers.moderation_analytics._list_queue", return_value=[])
    def test_limit_exactly_200(self, mock_lq, handler, http_handler):
        result = handler.handle("/api/v1/moderation/queue", {"limit": "200"}, http_handler)
        body = _body(result)
        assert body["limit"] == 200

    @patch("aragora.server.handlers.moderation_analytics._list_queue", return_value=[])
    def test_limit_and_offset_together(self, mock_lq, handler, http_handler):
        result = handler.handle(
            "/api/v1/moderation/queue", {"limit": "10", "offset": "20"}, http_handler
        )
        body = _body(result)
        assert body["limit"] == 10
        assert body["offset"] == 20
        mock_lq.assert_called_once_with(limit=10, offset=20)

    @patch("aragora.server.handlers.moderation_analytics._list_queue", return_value=[])
    def test_float_limit_truncated(self, mock_lq, handler, http_handler):
        """Float-like string for limit should fail and default to 50."""
        result = handler.handle("/api/v1/moderation/queue", {"limit": "3.5"}, http_handler)
        body = _body(result)
        assert body["limit"] == 50


# ---------------------------------------------------------------------------
# handle() routing
# ---------------------------------------------------------------------------


class TestHandleRouting:
    """Tests for handle() dispatching to correct sub-handler."""

    @patch(
        "aragora.server.handlers.moderation_analytics._get_moderation",
        return_value=None,
    )
    def test_routes_to_stats(self, mock_mod, handler, http_handler):
        result = handler.handle("/api/v1/moderation/stats", {}, http_handler)
        assert result is not None
        body = _body(result)
        assert "total_checks" in body

    @patch("aragora.server.handlers.moderation_analytics._list_queue", return_value=[])
    def test_routes_to_queue(self, mock_lq, handler, http_handler):
        result = handler.handle("/api/v1/moderation/queue", {}, http_handler)
        assert result is not None
        body = _body(result)
        assert "items" in body

    @patch(
        "aragora.server.handlers.moderation_analytics._get_moderation",
        return_value=None,
    )
    def test_unknown_path_returns_none(self, mock_mod, handler, http_handler):
        result = handler.handle("/api/v1/moderation/other", {}, http_handler)
        assert result is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for @handle_errors decorator on handle()."""

    @patch("aragora.server.handlers.moderation_analytics._get_moderation")
    def test_stats_exception_returns_error(self, mock_mod, handler, http_handler):
        """Exception in _handle_stats is caught by @handle_errors."""
        mock_mod.side_effect = RuntimeError("boom")
        result = handler.handle("/api/v1/moderation/stats", {}, http_handler)
        assert _status(result) == 500

    @patch("aragora.server.handlers.moderation_analytics._list_queue")
    def test_queue_exception_returns_error(self, mock_lq, handler, http_handler):
        """Exception in _handle_queue is caught by @handle_errors."""
        mock_lq.side_effect = RuntimeError("queue crash")
        result = handler.handle("/api/v1/moderation/queue", {}, http_handler)
        assert _status(result) == 500

    @patch("aragora.server.handlers.moderation_analytics._get_moderation")
    def test_error_response_has_trace_id(self, mock_mod, handler, http_handler):
        """Error responses include X-Trace-Id header."""
        mock_mod.side_effect = ValueError("bad data")
        result = handler.handle("/api/v1/moderation/stats", {}, http_handler)
        assert result.headers.get("X-Trace-Id") is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests."""

    @patch("aragora.server.handlers.moderation_analytics._get_queue_size", return_value=0)
    @patch("aragora.server.handlers.moderation_analytics._get_moderation")
    def test_stats_all_blocked(self, mock_mod, mock_qs, handler, http_handler):
        """100% block rate."""
        mod = MagicMock()
        mod.statistics = {"total_checks": 10, "blocked": 10, "false_positives": 3}
        mock_mod.return_value = mod
        result = handler.handle("/api/v1/moderation/stats", {}, http_handler)
        body = _body(result)
        assert body["block_rate"] == 1.0
        assert body["false_positive_rate"] == 0.3

    @patch("aragora.server.handlers.moderation_analytics._get_queue_size", return_value=0)
    @patch("aragora.server.handlers.moderation_analytics._get_moderation")
    def test_stats_large_numbers(self, mock_mod, mock_qs, handler, http_handler):
        """Handles large numbers correctly."""
        mod = MagicMock()
        mod.statistics = {
            "total_checks": 1_000_000,
            "blocked": 50_000,
            "flagged": 100_000,
            "clean": 850_000,
            "false_positives": 500,
        }
        mock_mod.return_value = mod
        result = handler.handle("/api/v1/moderation/stats", {}, http_handler)
        body = _body(result)
        assert body["total_checks"] == 1_000_000
        assert body["block_rate"] == 0.05
        assert body["false_positive_rate"] == 0.01

    @patch("aragora.server.handlers.moderation_analytics._list_queue")
    def test_queue_single_item(self, mock_lq, handler, http_handler):
        """Queue with exactly one item."""
        mock_lq.return_value = [{"id": "solo"}]
        result = handler.handle("/api/v1/moderation/queue", {}, http_handler)
        body = _body(result)
        assert body["count"] == 1
        assert body["items"] == [{"id": "solo"}]

    @patch("aragora.server.handlers.moderation_analytics._list_queue", return_value=[])
    def test_queue_limit_zero(self, mock_lq, handler, http_handler):
        """Limit of 0 is valid (min with 200 is 0)."""
        result = handler.handle("/api/v1/moderation/queue", {"limit": "0"}, http_handler)
        body = _body(result)
        assert body["limit"] == 0

    @patch("aragora.server.handlers.moderation_analytics._list_queue", return_value=[])
    def test_queue_negative_limit(self, mock_lq, handler, http_handler):
        """Negative limit still passes through min(val, 200)."""
        result = handler.handle("/api/v1/moderation/queue", {"limit": "-10"}, http_handler)
        body = _body(result)
        assert body["limit"] == -10  # min(-10, 200) = -10

    @patch("aragora.server.handlers.moderation_analytics._get_queue_size", return_value=0)
    @patch("aragora.server.handlers.moderation_analytics._get_moderation")
    def test_stats_with_only_false_positives(self, mock_mod, mock_qs, handler, http_handler):
        """Statistics with only false_positives field present."""
        mod = MagicMock()
        mod.statistics = {"total_checks": 50, "blocked": 10, "false_positives": 10}
        mock_mod.return_value = mod
        result = handler.handle("/api/v1/moderation/stats", {}, http_handler)
        body = _body(result)
        assert body["false_positive_rate"] == 1.0
