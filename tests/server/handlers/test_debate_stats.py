"""
Tests for aragora.server.handlers.debate_stats - Debate statistics handler.

Tests cover:
- Instantiation and ROUTES
- can_handle() route matching with version prefix stripping
- GET /api/v1/debates/stats - aggregate debate statistics
- GET /api/v1/debates/stats/agents - per-agent statistics
- Method not allowed (non-GET)
- Period query parameter validation
- Limit query parameter
- Error handling when storage or analytics service fails
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.debate_stats import DebateStatsHandler


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def handler():
    """Create a DebateStatsHandler with mocked context."""
    ctx: dict[str, Any] = {"storage": MagicMock()}
    return DebateStatsHandler(ctx)


@pytest.fixture
def handler_no_storage():
    """Create handler without storage."""
    ctx: dict[str, Any] = {}
    return DebateStatsHandler(ctx)


@pytest.fixture
def mock_get():
    """Create a mock HTTP GET handler."""
    mock = MagicMock()
    mock.command = "GET"
    mock.headers = {"Authorization": "Bearer test-token"}
    return mock


# ===========================================================================
# Instantiation and Routes
# ===========================================================================


class TestSetup:
    """Tests for handler instantiation and route registration."""

    def test_instantiation(self, handler):
        """Should create handler with context."""
        assert handler is not None

    def test_routes_defined(self):
        """Should define expected ROUTES."""
        assert "/api/v1/debates/stats" in DebateStatsHandler.ROUTES
        assert "/api/v1/debates/stats/agents" in DebateStatsHandler.ROUTES

    def test_routes_count(self):
        """Should have exactly 2 routes."""
        assert len(DebateStatsHandler.ROUTES) == 2


# ===========================================================================
# can_handle
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle route matching."""

    def test_can_handle_stats(self, handler):
        """Should handle /api/v1/debates/stats."""
        assert handler.can_handle("/api/v1/debates/stats") is True

    def test_can_handle_agent_stats(self, handler):
        """Should handle /api/v1/debates/stats/agents."""
        assert handler.can_handle("/api/v1/debates/stats/agents") is True

    def test_cannot_handle_debates_root(self, handler):
        """Should not handle /api/v1/debates without /stats."""
        assert handler.can_handle("/api/v1/debates") is False

    def test_cannot_handle_unknown(self, handler):
        """Should not handle unknown paths."""
        assert handler.can_handle("/api/v1/stats") is False
        assert handler.can_handle("/api/v1/debates/stats/other") is False


# ===========================================================================
# Method Not Allowed
# ===========================================================================


class TestMethodNotAllowed:
    """Tests for non-GET method rejection."""

    def test_post_returns_405(self, handler):
        """Should reject POST with 405."""
        mock = MagicMock()
        mock.command = "POST"
        result = handler.handle("/api/v1/debates/stats", {}, mock)
        assert result.status_code == 405

    def test_put_returns_405(self, handler):
        """Should reject PUT with 405."""
        mock = MagicMock()
        mock.command = "PUT"
        result = handler.handle("/api/v1/debates/stats", {}, mock)
        assert result.status_code == 405

    def test_delete_returns_405(self, handler):
        """Should reject DELETE with 405."""
        mock = MagicMock()
        mock.command = "DELETE"
        result = handler.handle("/api/v1/debates/stats", {}, mock)
        assert result.status_code == 405


# ===========================================================================
# GET /api/v1/debates/stats
# ===========================================================================


class TestGetStats:
    """Tests for GET /api/v1/debates/stats."""

    def test_get_stats_success(self, handler, mock_get):
        """Should return aggregate debate statistics."""
        mock_stats = MagicMock()
        mock_stats.to_dict.return_value = {
            "total_debates": 100,
            "completed_debates": 90,
            "failed_debates": 5,
            "consensus_reached": 72,
            "consensus_rate": 0.8,
            "avg_rounds": 3.2,
            "avg_duration_seconds": 45.5,
        }

        mock_service = MagicMock()
        mock_service.get_debate_stats.return_value = mock_stats

        with patch(
            "aragora.analytics.debate_analytics.DebateAnalytics",
            return_value=mock_service,
        ):
            result = handler.handle("/api/v1/debates/stats", {}, mock_get)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["total_debates"] == 100
        assert data["consensus_rate"] == 0.8

    def test_get_stats_with_period(self, handler, mock_get):
        """Should pass period filter to analytics service."""
        mock_stats = MagicMock()
        mock_stats.to_dict.return_value = {"total_debates": 10}

        mock_service = MagicMock()
        mock_service.get_debate_stats.return_value = mock_stats

        with patch(
            "aragora.analytics.debate_analytics.DebateAnalytics",
            return_value=mock_service,
        ):
            handler.handle("/api/v1/debates/stats", {"period": "week"}, mock_get)
            mock_service.get_debate_stats.assert_called_once_with(period="week")

    def test_get_stats_default_period(self, handler, mock_get):
        """Should default to 'all' period."""
        mock_stats = MagicMock()
        mock_stats.to_dict.return_value = {"total_debates": 10}

        mock_service = MagicMock()
        mock_service.get_debate_stats.return_value = mock_stats

        with patch(
            "aragora.analytics.debate_analytics.DebateAnalytics",
            return_value=mock_service,
        ):
            handler.handle("/api/v1/debates/stats", {}, mock_get)
            mock_service.get_debate_stats.assert_called_once_with(period="all")

    def test_invalid_period_returns_400(self, handler, mock_get):
        """Should return 400 for invalid period value."""
        result = handler.handle(
            "/api/v1/debates/stats", {"period": "century"}, mock_get
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "period" in data.get("error", "")

    def test_no_storage_returns_error(self, handler_no_storage, mock_get):
        """Should return error when storage not available."""
        result = handler_no_storage.handle("/api/v1/debates/stats", {}, mock_get)
        assert result.status_code in (500, 503)

    def test_service_error_returns_500(self, handler, mock_get):
        """Should return 500 when analytics service fails."""
        with patch(
            "aragora.analytics.debate_analytics.DebateAnalytics",
            side_effect=RuntimeError("DB down"),
        ):
            result = handler.handle("/api/v1/debates/stats", {}, mock_get)
        assert result.status_code == 500


# ===========================================================================
# GET /api/v1/debates/stats/agents
# ===========================================================================


class TestGetAgentStats:
    """Tests for GET /api/v1/debates/stats/agents."""

    def test_get_agent_stats_success(self, handler, mock_get):
        """Should return per-agent statistics."""
        mock_agents = [
            {"agent": "claude", "debates": 50, "win_rate": 0.72},
            {"agent": "gpt4", "debates": 45, "win_rate": 0.68},
        ]

        mock_service = MagicMock()
        mock_service.get_agent_stats.return_value = mock_agents

        with patch(
            "aragora.analytics.debate_analytics.DebateAnalytics",
            return_value=mock_service,
        ):
            result = handler.handle(
                "/api/v1/debates/stats/agents", {}, mock_get
            )

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["count"] == 2
        assert len(data["agents"]) == 2

    def test_get_agent_stats_with_limit(self, handler, mock_get):
        """Should pass limit to analytics service."""
        mock_service = MagicMock()
        mock_service.get_agent_stats.return_value = []

        with patch(
            "aragora.analytics.debate_analytics.DebateAnalytics",
            return_value=mock_service,
        ):
            handler.handle(
                "/api/v1/debates/stats/agents", {"limit": "5"}, mock_get
            )
            mock_service.get_agent_stats.assert_called_once_with(limit=5)

    def test_no_storage_returns_error(self, handler_no_storage, mock_get):
        """Should return error when storage not available."""
        result = handler_no_storage.handle(
            "/api/v1/debates/stats/agents", {}, mock_get
        )
        assert result.status_code in (500, 503)

    def test_service_error_returns_500(self, handler, mock_get):
        """Should return 500 when analytics service fails."""
        with patch(
            "aragora.analytics.debate_analytics.DebateAnalytics",
            side_effect=RuntimeError("DB down"),
        ):
            result = handler.handle(
                "/api/v1/debates/stats/agents", {}, mock_get
            )
        assert result.status_code == 500
