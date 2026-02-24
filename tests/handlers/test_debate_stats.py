"""Tests for the DebateStatsHandler REST endpoints.

Covers:
- GET /api/v1/debates/stats -- aggregate debate statistics
- GET /api/v1/debates/stats/agents -- per-agent statistics
- Routing (can_handle, method filtering, unmatched paths)
- Period validation
- Limit query parameter with bounds
- Error paths (storage unavailable, import errors, runtime errors)
"""

from __future__ import annotations

import builtins
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.debate_stats import DebateStatsHandler

# Patch target: the handler does ``from aragora.analytics.debate_analytics import DebateAnalytics``
# inside the method body.  Patching at the *source* module ensures the local
# import picks up the mock.
_ANALYTICS_CLS = "aragora.analytics.debate_analytics.DebateAnalytics"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _make_import_raiser(target_module: str):
    """Return an __import__ replacement that raises ImportError for *target_module*."""
    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == target_module:
            raise ImportError(f"mocked: no module named {name}")
        return real_import(name, *args, **kwargs)

    return _import


class _MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to DebateStatsHandler.handle."""

    def __init__(self, method: str = "GET", body: dict[str, Any] | None = None):
        self.command = method
        self.headers = {"Content-Length": "0"}
        self.rfile = MagicMock()

        if body is not None:
            raw = json.dumps(body).encode()
            self.rfile.read.return_value = raw
            self.headers = {"Content-Length": str(len(raw))}
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {"Content-Length": "2"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a DebateStatsHandler with an empty server context."""
    return DebateStatsHandler({})


@pytest.fixture
def handler_with_storage():
    """Create a DebateStatsHandler whose context has a mock storage."""
    storage = MagicMock()
    return DebateStatsHandler({"storage": storage}), storage


@pytest.fixture
def mock_http():
    """Factory for creating mock HTTP handlers."""

    def _create(method: str = "GET") -> _MockHTTPHandler:
        return _MockHTTPHandler(method=method)

    return _create


# ---------------------------------------------------------------------------
# can_handle routing
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Test DebateStatsHandler.can_handle routing."""

    def test_stats_versioned(self, handler):
        assert handler.can_handle("/api/v1/debates/stats") is True

    def test_stats_agents_versioned(self, handler):
        assert handler.can_handle("/api/v1/debates/stats/agents") is True

    def test_stats_unversioned(self, handler):
        assert handler.can_handle("/api/debates/stats") is True

    def test_stats_agents_unversioned(self, handler):
        assert handler.can_handle("/api/debates/stats/agents") is True

    def test_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_stats_trailing_slash(self, handler):
        assert handler.can_handle("/api/v1/debates/stats/") is False

    def test_stats_extra_segment(self, handler):
        assert handler.can_handle("/api/v1/debates/stats/agents/extra") is False

    def test_different_version(self, handler):
        assert handler.can_handle("/api/v2/debates/stats") is True

    def test_root_path(self, handler):
        assert handler.can_handle("/") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("") is False


# ---------------------------------------------------------------------------
# ROUTES class attribute
# ---------------------------------------------------------------------------


class TestRoutes:
    """Test ROUTES class attribute is correctly declared."""

    def test_routes_defined(self):
        assert hasattr(DebateStatsHandler, "ROUTES")

    def test_routes_contains_stats(self):
        assert "/api/v1/debates/stats" in DebateStatsHandler.ROUTES

    def test_routes_contains_agent_stats(self):
        assert "/api/v1/debates/stats/agents" in DebateStatsHandler.ROUTES


# ---------------------------------------------------------------------------
# Method Not Allowed
# ---------------------------------------------------------------------------


class TestMethodNotAllowed:
    """Test that non-GET requests are rejected."""

    @pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
    def test_non_get_returns_405_stats(self, handler, method):
        http = _MockHTTPHandler(method=method)
        result = handler.handle("/api/v1/debates/stats", {}, http)
        assert _status(result) == 405
        assert "Method not allowed" in _body(result).get("error", "")

    @pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
    def test_non_get_returns_405_agent_stats(self, handler, method):
        http = _MockHTTPHandler(method=method)
        result = handler.handle("/api/v1/debates/stats/agents", {}, http)
        assert _status(result) == 405
        assert "Method not allowed" in _body(result).get("error", "")


# ---------------------------------------------------------------------------
# GET /api/v1/debates/stats -- success paths
# ---------------------------------------------------------------------------


class TestGetStats:
    """Test the aggregate stats endpoint."""

    def test_success_default_period(self, handler_with_storage, mock_http):
        handler, storage = handler_with_storage
        http = mock_http()
        mock_stats = MagicMock()
        mock_stats.to_dict.return_value = {
            "total_debates": 42,
            "consensus_rate": 0.75,
        }
        with patch(_ANALYTICS_CLS) as MockAnalytics:
            MockAnalytics.return_value.get_debate_stats.return_value = mock_stats
            result = handler.handle("/api/v1/debates/stats", {}, http)

        assert _status(result) == 200
        body = _body(result)
        assert body["total_debates"] == 42
        assert body["consensus_rate"] == 0.75
        MockAnalytics.assert_called_once_with(storage)
        MockAnalytics.return_value.get_debate_stats.assert_called_once_with(days_back=3650)

    @pytest.mark.parametrize(
        "period,days",
        [("all", 3650), ("day", 1), ("week", 7), ("month", 30)],
    )
    def test_success_valid_periods(self, handler_with_storage, mock_http, period, days):
        handler, storage = handler_with_storage
        http = mock_http()
        mock_stats = MagicMock()
        mock_stats.to_dict.return_value = {"period": period}
        with patch(_ANALYTICS_CLS) as MockAnalytics:
            MockAnalytics.return_value.get_debate_stats.return_value = mock_stats
            result = handler.handle("/api/v1/debates/stats", {"period": period}, http)

        assert _status(result) == 200
        assert _body(result)["period"] == period
        MockAnalytics.return_value.get_debate_stats.assert_called_once_with(days_back=days)

    def test_invalid_period_returns_400(self, handler_with_storage, mock_http):
        handler, _ = handler_with_storage
        http = mock_http()
        result = handler.handle("/api/v1/debates/stats", {"period": "year"}, http)
        assert _status(result) == 400
        assert "period" in _body(result).get("error", "")

    def test_invalid_period_hour(self, handler_with_storage, mock_http):
        handler, _ = handler_with_storage
        http = mock_http()
        result = handler.handle("/api/v1/debates/stats", {"period": "hour"}, http)
        assert _status(result) == 400

    def test_storage_none_returns_503(self, handler, mock_http):
        """When context has no storage, a 503 is returned."""
        http = mock_http()
        with patch(_ANALYTICS_CLS):
            result = handler.handle("/api/v1/debates/stats", {}, http)
        assert _status(result) == 503
        assert "Storage" in _body(result).get("error", "")

    def test_import_error_returns_500(self, handler_with_storage, mock_http):
        handler, _ = handler_with_storage
        http = mock_http()
        with patch(
            "builtins.__import__",
            side_effect=_make_import_raiser("aragora.analytics.debate_analytics"),
        ):
            result = handler.handle("/api/v1/debates/stats", {}, http)
        assert _status(result) == 500
        assert "Failed to get debate stats" in _body(result).get("error", "")

    def test_value_error_returns_500(self, handler_with_storage, mock_http):
        handler, _ = handler_with_storage
        http = mock_http()
        with patch(_ANALYTICS_CLS) as MockAnalytics:
            MockAnalytics.return_value.get_debate_stats.side_effect = ValueError("bad value")
            result = handler.handle("/api/v1/debates/stats", {}, http)
        assert _status(result) == 500

    def test_type_error_returns_500(self, handler_with_storage, mock_http):
        handler, _ = handler_with_storage
        http = mock_http()
        with patch(_ANALYTICS_CLS) as MockAnalytics:
            MockAnalytics.return_value.get_debate_stats.side_effect = TypeError("wrong type")
            result = handler.handle("/api/v1/debates/stats", {}, http)
        assert _status(result) == 500

    def test_key_error_returns_500(self, handler_with_storage, mock_http):
        handler, _ = handler_with_storage
        http = mock_http()
        with patch(_ANALYTICS_CLS) as MockAnalytics:
            MockAnalytics.return_value.get_debate_stats.side_effect = KeyError("key")
            result = handler.handle("/api/v1/debates/stats", {}, http)
        assert _status(result) == 500

    def test_attribute_error_returns_500(self, handler_with_storage, mock_http):
        handler, _ = handler_with_storage
        http = mock_http()
        with patch(_ANALYTICS_CLS) as MockAnalytics:
            MockAnalytics.return_value.get_debate_stats.side_effect = AttributeError("no attr")
            result = handler.handle("/api/v1/debates/stats", {}, http)
        assert _status(result) == 500

    def test_os_error_returns_500(self, handler_with_storage, mock_http):
        handler, _ = handler_with_storage
        http = mock_http()
        with patch(_ANALYTICS_CLS) as MockAnalytics:
            MockAnalytics.return_value.get_debate_stats.side_effect = OSError("disk error")
            result = handler.handle("/api/v1/debates/stats", {}, http)
        assert _status(result) == 500

    def test_runtime_error_returns_500(self, handler_with_storage, mock_http):
        handler, _ = handler_with_storage
        http = mock_http()
        with patch(_ANALYTICS_CLS) as MockAnalytics:
            MockAnalytics.return_value.get_debate_stats.side_effect = RuntimeError(
                "something broke"
            )
            result = handler.handle("/api/v1/debates/stats", {}, http)
        assert _status(result) == 500


# ---------------------------------------------------------------------------
# GET /api/v1/debates/stats/agents -- success paths
# ---------------------------------------------------------------------------


class TestGetAgentStats:
    """Test the per-agent stats endpoint."""

    def test_success_default_limit(self, handler_with_storage, mock_http):
        handler, storage = handler_with_storage
        http = mock_http()
        agent_data = [
            {"agent": "claude", "wins": 10},
            {"agent": "gpt4", "wins": 8},
        ]
        with patch(_ANALYTICS_CLS) as MockAnalytics:
            MockAnalytics.return_value.get_agent_leaderboard.return_value = agent_data
            result = handler.handle("/api/v1/debates/stats/agents", {}, http)

        assert _status(result) == 200
        body = _body(result)
        assert body["agents"] == agent_data
        assert body["count"] == 2
        MockAnalytics.assert_called_once_with(storage)
        MockAnalytics.return_value.get_agent_leaderboard.assert_called_once_with(limit=20)

    def test_custom_limit(self, handler_with_storage, mock_http):
        handler, _ = handler_with_storage
        http = mock_http()
        with patch(_ANALYTICS_CLS) as MockAnalytics:
            MockAnalytics.return_value.get_agent_leaderboard.return_value = []
            result = handler.handle("/api/v1/debates/stats/agents", {"limit": "5"}, http)

        assert _status(result) == 200
        MockAnalytics.return_value.get_agent_leaderboard.assert_called_once_with(limit=5)

    def test_limit_clamped_to_max(self, handler_with_storage, mock_http):
        """Limit above 100 should be clamped to 100 by safe_query_int."""
        handler, _ = handler_with_storage
        http = mock_http()
        with patch(_ANALYTICS_CLS) as MockAnalytics:
            MockAnalytics.return_value.get_agent_leaderboard.return_value = []
            result = handler.handle("/api/v1/debates/stats/agents", {"limit": "999"}, http)

        assert _status(result) == 200
        MockAnalytics.return_value.get_agent_leaderboard.assert_called_once_with(limit=100)

    def test_limit_clamped_to_min(self, handler_with_storage, mock_http):
        """Limit below 1 should be clamped to 1 by safe_query_int."""
        handler, _ = handler_with_storage
        http = mock_http()
        with patch(_ANALYTICS_CLS) as MockAnalytics:
            MockAnalytics.return_value.get_agent_leaderboard.return_value = []
            result = handler.handle("/api/v1/debates/stats/agents", {"limit": "0"}, http)

        assert _status(result) == 200
        MockAnalytics.return_value.get_agent_leaderboard.assert_called_once_with(limit=1)

    def test_invalid_limit_uses_default(self, handler_with_storage, mock_http):
        """Non-numeric limit falls back to default 20."""
        handler, _ = handler_with_storage
        http = mock_http()
        with patch(_ANALYTICS_CLS) as MockAnalytics:
            MockAnalytics.return_value.get_agent_leaderboard.return_value = []
            result = handler.handle("/api/v1/debates/stats/agents", {"limit": "abc"}, http)

        assert _status(result) == 200
        MockAnalytics.return_value.get_agent_leaderboard.assert_called_once_with(limit=20)

    def test_empty_agent_stats(self, handler_with_storage, mock_http):
        handler, _ = handler_with_storage
        http = mock_http()
        with patch(_ANALYTICS_CLS) as MockAnalytics:
            MockAnalytics.return_value.get_agent_leaderboard.return_value = []
            result = handler.handle("/api/v1/debates/stats/agents", {}, http)

        assert _status(result) == 200
        body = _body(result)
        assert body["agents"] == []
        assert body["count"] == 0

    def test_storage_none_returns_503(self, handler, mock_http):
        http = mock_http()
        with patch(_ANALYTICS_CLS):
            result = handler.handle("/api/v1/debates/stats/agents", {}, http)
        assert _status(result) == 503
        assert "Storage" in _body(result).get("error", "")

    def test_import_error_returns_500(self, handler_with_storage, mock_http):
        handler, _ = handler_with_storage
        http = mock_http()
        with patch(
            "builtins.__import__",
            side_effect=_make_import_raiser("aragora.analytics.debate_analytics"),
        ):
            result = handler.handle("/api/v1/debates/stats/agents", {}, http)
        assert _status(result) == 500
        assert "Failed to get agent stats" in _body(result).get("error", "")

    def test_value_error_returns_500(self, handler_with_storage, mock_http):
        handler, _ = handler_with_storage
        http = mock_http()
        with patch(_ANALYTICS_CLS) as MockAnalytics:
            MockAnalytics.return_value.get_agent_leaderboard.side_effect = ValueError("bad")
            result = handler.handle("/api/v1/debates/stats/agents", {}, http)
        assert _status(result) == 500

    def test_runtime_error_returns_500(self, handler_with_storage, mock_http):
        handler, _ = handler_with_storage
        http = mock_http()
        with patch(_ANALYTICS_CLS) as MockAnalytics:
            MockAnalytics.return_value.get_agent_leaderboard.side_effect = RuntimeError("broken")
            result = handler.handle("/api/v1/debates/stats/agents", {}, http)
        assert _status(result) == 500

    def test_os_error_returns_500(self, handler_with_storage, mock_http):
        handler, _ = handler_with_storage
        http = mock_http()
        with patch(_ANALYTICS_CLS) as MockAnalytics:
            MockAnalytics.return_value.get_agent_leaderboard.side_effect = OSError("io err")
            result = handler.handle("/api/v1/debates/stats/agents", {}, http)
        assert _status(result) == 500


# ---------------------------------------------------------------------------
# Unmatched paths after method check
# ---------------------------------------------------------------------------


class TestUnmatchedPath:
    """Test that handle returns None for paths that pass method check but don't match."""

    def test_unmatched_path_returns_none(self, handler, mock_http):
        """A GET to a path not matching either route returns None."""
        http = mock_http()
        result = handler.handle("/api/v1/debates/other", {}, http)
        assert result is None

    def test_unmatched_stripped_path_returns_none(self, handler, mock_http):
        http = mock_http()
        result = handler.handle("/api/debates/unknown", {}, http)
        assert result is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Miscellaneous edge cases."""

    def test_limit_negative(self, handler_with_storage, mock_http):
        """Negative limit should be clamped to min_val (1)."""
        handler, _ = handler_with_storage
        http = mock_http()
        with patch(_ANALYTICS_CLS) as MockAnalytics:
            MockAnalytics.return_value.get_agent_leaderboard.return_value = []
            result = handler.handle("/api/v1/debates/stats/agents", {"limit": "-5"}, http)
        assert _status(result) == 200
        MockAnalytics.return_value.get_agent_leaderboard.assert_called_once_with(limit=1)

    def test_limit_boundary_100(self, handler_with_storage, mock_http):
        """Limit exactly 100 should be accepted."""
        handler, _ = handler_with_storage
        http = mock_http()
        with patch(_ANALYTICS_CLS) as MockAnalytics:
            MockAnalytics.return_value.get_agent_leaderboard.return_value = []
            result = handler.handle("/api/v1/debates/stats/agents", {"limit": "100"}, http)
        assert _status(result) == 200
        MockAnalytics.return_value.get_agent_leaderboard.assert_called_once_with(limit=100)

    def test_limit_boundary_1(self, handler_with_storage, mock_http):
        """Limit exactly 1 should be accepted."""
        handler, _ = handler_with_storage
        http = mock_http()
        with patch(_ANALYTICS_CLS) as MockAnalytics:
            MockAnalytics.return_value.get_agent_leaderboard.return_value = []
            result = handler.handle("/api/v1/debates/stats/agents", {"limit": "1"}, http)
        assert _status(result) == 200
        MockAnalytics.return_value.get_agent_leaderboard.assert_called_once_with(limit=1)

    def test_period_empty_string_invalid(self, handler_with_storage, mock_http):
        """An empty period string should fail validation."""
        handler, _ = handler_with_storage
        http = mock_http()
        result = handler.handle("/api/v1/debates/stats", {"period": ""}, http)
        assert _status(result) == 400

    def test_period_case_sensitive(self, handler_with_storage, mock_http):
        """Period values are case sensitive -- 'Day' should be rejected."""
        handler, _ = handler_with_storage
        http = mock_http()
        result = handler.handle("/api/v1/debates/stats", {"period": "Day"}, http)
        assert _status(result) == 400

    def test_handler_instantiation_with_empty_ctx(self):
        """Handler can be created with an empty dict context."""
        h = DebateStatsHandler({})
        assert h.ctx == {}

    def test_handler_ctx_contains_storage(self, handler_with_storage):
        """Handler context holds the provided storage."""
        handler, storage = handler_with_storage
        assert handler.ctx.get("storage") is storage

    def test_stats_to_dict_called(self, handler_with_storage, mock_http):
        """Verify to_dict() is called on the stats result."""
        handler, _ = handler_with_storage
        http = mock_http()
        mock_stats = MagicMock()
        mock_stats.to_dict.return_value = {"x": 1}
        with patch(_ANALYTICS_CLS) as MockAnalytics:
            MockAnalytics.return_value.get_debate_stats.return_value = mock_stats
            result = handler.handle("/api/v1/debates/stats", {}, http)
        mock_stats.to_dict.assert_called_once()
        assert _body(result) == {"x": 1}

    def test_agent_stats_count_matches_list_length(self, handler_with_storage, mock_http):
        """The count field matches the actual number of agent entries."""
        handler, _ = handler_with_storage
        http = mock_http()
        agents = [{"a": i} for i in range(7)]
        with patch(_ANALYTICS_CLS) as MockAnalytics:
            MockAnalytics.return_value.get_agent_leaderboard.return_value = agents
            result = handler.handle("/api/v1/debates/stats/agents", {}, http)
        body = _body(result)
        assert body["count"] == 7
        assert len(body["agents"]) == 7

    def test_analytics_receives_storage_from_ctx(self, handler_with_storage, mock_http):
        """DebateAnalytics is constructed with the storage from context."""
        handler, storage = handler_with_storage
        http = mock_http()
        mock_stats = MagicMock()
        mock_stats.to_dict.return_value = {}
        with patch(_ANALYTICS_CLS) as MockAnalytics:
            MockAnalytics.return_value.get_debate_stats.return_value = mock_stats
            handler.handle("/api/v1/debates/stats", {}, http)
        MockAnalytics.assert_called_once_with(storage)
