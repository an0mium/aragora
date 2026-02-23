"""Comprehensive tests for the DeliberationAnalyticsMixin handler.

Tests all four endpoints provided by the deliberation analytics handler:
- GET /api/v1/analytics/deliberations            - Deliberation summary (cached: 300s)
- GET /api/v1/analytics/deliberations/channels   - Deliberations by channel
- GET /api/v1/analytics/deliberations/consensus  - Consensus rates by team
- GET /api/v1/analytics/deliberations/performance - Performance metrics

Also tests routing, version prefix handling, error handling, and security.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.analytics_dashboard.handler import AnalyticsDashboardHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeUserCtx:
    user_id: str = "test_user"
    org_id: str = "org_123"
    role: str = "admin"
    is_authenticated: bool = True
    error_reason: str | None = None


def _body(result) -> dict:
    """Extract decoded JSON body from a HandlerResult (tuple-style unpacking)."""
    body, _status, _headers = result
    if isinstance(body, dict):
        return body
    return json.loads(body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    _body_val, status, _headers = result
    return status


def _make_handler() -> AnalyticsDashboardHandler:
    """Create a fresh AnalyticsDashboardHandler instance."""
    return AnalyticsDashboardHandler(ctx={})


def _mock_http_handler() -> MagicMock:
    """Create a mock HTTP handler with minimal auth."""
    h = MagicMock()
    h.headers = {"Authorization": "Bearer test-token", "Content-Length": "0"}
    h.command = "GET"
    h.path = "/api/analytics/deliberations"
    return h


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_auth():
    """Patch auth to always succeed for all tests."""
    with patch(
        "aragora.billing.jwt_auth.extract_user_from_request",
        return_value=FakeUserCtx(),
    ):
        yield


@pytest.fixture
def handler():
    return _make_handler()


@pytest.fixture
def mock_http():
    return _mock_http_handler()


def _make_deliberation_stats(
    total=100,
    completed=80,
    consensus_reached=60,
    in_progress=15,
    failed=5,
    avg_rounds=3.2,
    avg_duration_seconds=45.7,
    by_template=None,
    by_priority=None,
):
    """Build a mock deliberation stats dict."""
    return {
        "total": total,
        "completed": completed,
        "consensus_reached": consensus_reached,
        "in_progress": in_progress,
        "failed": failed,
        "avg_rounds": avg_rounds,
        "avg_duration_seconds": avg_duration_seconds,
        "by_template": by_template or {"structured": 50, "freeform": 30, "tournament": 20},
        "by_priority": by_priority or {"high": 20, "medium": 60, "low": 20},
    }


def _make_channel_stats():
    """Build mock channel stats list."""
    return [
        {
            "platform": "web",
            "total_deliberations": 50,
            "consensus_reached": 40,
            "total_duration": 2000,
        },
        {
            "platform": "api",
            "total_deliberations": 30,
            "consensus_reached": 20,
            "total_duration": 1500,
        },
        {
            "platform": "cli",
            "total_deliberations": 20,
            "consensus_reached": 15,
            "total_duration": 800,
        },
    ]


def _make_consensus_stats():
    """Build mock consensus stats."""
    return {
        "overall_consensus_rate": "75.0%",
        "by_team_size": {
            "3": "80.0%",
            "5": "70.0%",
            "7": "65.0%",
        },
        "by_agent": [
            {"agent": "claude", "consensus_rate": "82.0%"},
            {"agent": "gpt-4", "consensus_rate": "75.0%"},
        ],
        "top_teams": [
            {"team": ["claude", "gpt-4"], "rate": "90.0%"},
            {"team": ["claude", "gemini"], "rate": "85.0%"},
        ],
    }


def _make_performance_stats():
    """Build mock performance stats."""
    return {
        "summary": {
            "avg_latency_ms": 1200,
            "p95_latency_ms": 3500,
            "avg_cost_usd": 0.05,
            "efficiency_score": 0.82,
        },
        "by_template": [
            {"template": "structured", "avg_latency_ms": 1000, "avg_cost_usd": 0.04},
            {"template": "freeform", "avg_latency_ms": 1500, "avg_cost_usd": 0.06},
        ],
        "trends": [
            {"date": "2026-02-20", "latency_ms": 1100, "cost_usd": 0.05},
            {"date": "2026-02-21", "latency_ms": 1300, "cost_usd": 0.06},
        ],
        "cost_by_agent": {
            "claude": 0.03,
            "gpt-4": 0.02,
        },
    }


# =========================================================================
# 1. DELIBERATION SUMMARY ENDPOINT - GET /api/analytics/deliberations
# =========================================================================


class TestDeliberationSummary:
    """Tests for _get_deliberation_summary endpoint."""

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_summary_happy_path(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_stats.return_value = _make_deliberation_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_summary(
            {"org_id": "org_123", "days": "30"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["org_id"] == "org_123"
        assert body["total_deliberations"] == 100
        assert body["completed"] == 80
        assert body["consensus_reached"] == 60
        assert body["consensus_rate"] == "75.0%"

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_summary_default_days(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_stats.return_value = _make_deliberation_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_summary(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["period"]["days"] == 30

    def test_summary_missing_org_id(self, handler, mock_http):
        result = handler._get_deliberation_summary({}, handler=mock_http)
        assert _status(result) == 400
        body = _body(result)
        error = body.get("error", {})
        error_text = error if isinstance(error, str) else error.get("message", "")
        assert "org_id" in error_text.lower()

    def test_summary_empty_org_id(self, handler, mock_http):
        result = handler._get_deliberation_summary({"org_id": ""}, handler=mock_http)
        assert _status(result) == 400

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_summary_period_structure(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_stats.return_value = _make_deliberation_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_summary(
            {"org_id": "org_123", "days": "14"},
            handler=mock_http,
        )
        body = _body(result)
        assert "period" in body
        assert "start" in body["period"]
        assert "end" in body["period"]
        assert body["period"]["days"] == 14

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_summary_includes_all_fields(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_stats.return_value = _make_deliberation_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_summary(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        expected_keys = [
            "org_id",
            "period",
            "total_deliberations",
            "completed",
            "in_progress",
            "failed",
            "consensus_reached",
            "consensus_rate",
            "avg_rounds",
            "avg_duration_seconds",
            "by_template",
            "by_priority",
        ]
        for key in expected_keys:
            assert key in body, f"Missing key: {key}"

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_summary_consensus_rate_zero_completed(self, mock_get_store, handler, mock_http):
        """When no deliberations completed, consensus rate should be '0%'."""
        mock_store = MagicMock()
        stats = _make_deliberation_stats(total=10, completed=0, consensus_reached=0)
        mock_store.get_deliberation_stats.return_value = stats
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_summary(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["consensus_rate"] == "0%"

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_summary_consensus_rate_all_consensus(self, mock_get_store, handler, mock_http):
        """100% consensus rate."""
        mock_store = MagicMock()
        stats = _make_deliberation_stats(completed=50, consensus_reached=50)
        mock_store.get_deliberation_stats.return_value = stats
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_summary(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["consensus_rate"] == "100.0%"

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_summary_avg_rounds_rounding(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        stats = _make_deliberation_stats(avg_rounds=3.456)
        mock_store.get_deliberation_stats.return_value = stats
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_summary(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["avg_rounds"] == 3.5

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_summary_avg_duration_rounding(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        stats = _make_deliberation_stats(avg_duration_seconds=45.789)
        mock_store.get_deliberation_stats.return_value = stats
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_summary(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["avg_duration_seconds"] == 45.8

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_summary_by_template_passthrough(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        template_data = {"structured": 50, "freeform": 30}
        stats = _make_deliberation_stats(by_template=template_data)
        mock_store.get_deliberation_stats.return_value = stats
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_summary(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["by_template"] == template_data

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_summary_by_priority_passthrough(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        priority_data = {"high": 20, "medium": 60, "low": 20}
        stats = _make_deliberation_stats(by_priority=priority_data)
        mock_store.get_deliberation_stats.return_value = stats
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_summary(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["by_priority"] == priority_data

    @patch("aragora.memory.debate_store.get_debate_store", side_effect=ImportError("no module"))
    def test_summary_import_error(self, _mock, handler, mock_http):
        result = handler._get_deliberation_summary(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.memory.debate_store.get_debate_store", side_effect=RuntimeError("db down"))
    def test_summary_runtime_error(self, _mock, handler, mock_http):
        result = handler._get_deliberation_summary(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.memory.debate_store.get_debate_store", side_effect=OSError("disk failure"))
    def test_summary_os_error(self, _mock, handler, mock_http):
        result = handler._get_deliberation_summary(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.memory.debate_store.get_debate_store", side_effect=LookupError("not found"))
    def test_summary_lookup_error(self, _mock, handler, mock_http):
        result = handler._get_deliberation_summary(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_summary_days_clamped_min(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_stats.return_value = _make_deliberation_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_summary(
            {"org_id": "org_123", "days": "0"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["period"]["days"] >= 1

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_summary_days_clamped_max(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_stats.return_value = _make_deliberation_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_summary(
            {"org_id": "org_123", "days": "9999"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["period"]["days"] <= 365

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_summary_days_negative(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_stats.return_value = _make_deliberation_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_summary(
            {"org_id": "org_123", "days": "-5"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["period"]["days"] >= 1

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_summary_days_non_numeric(self, mock_get_store, handler, mock_http):
        """Non-numeric days should fall back to default 30."""
        mock_store = MagicMock()
        mock_store.get_deliberation_stats.return_value = _make_deliberation_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_summary(
            {"org_id": "org_123", "days": "abc"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["period"]["days"] == 30

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_summary_days_boundary_1(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_stats.return_value = _make_deliberation_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_summary(
            {"org_id": "org_123", "days": "1"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["period"]["days"] == 1

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_summary_days_boundary_365(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_stats.return_value = _make_deliberation_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_summary(
            {"org_id": "org_123", "days": "365"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["period"]["days"] == 365

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_summary_missing_stat_keys_default_gracefully(self, mock_get_store, handler, mock_http):
        """Stats dict with missing keys should default to 0 or empty."""
        mock_store = MagicMock()
        mock_store.get_deliberation_stats.return_value = {}
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_summary(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["total_deliberations"] == 0
        assert body["completed"] == 0
        assert body["consensus_reached"] == 0
        assert body["consensus_rate"] == "0%"
        assert body["in_progress"] == 0
        assert body["failed"] == 0
        assert body["avg_rounds"] == 0
        assert body["avg_duration_seconds"] == 0
        assert body["by_template"] == {}
        assert body["by_priority"] == {}

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_summary_preserves_org_id(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_stats.return_value = _make_deliberation_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_summary(
            {"org_id": "my-special-org"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["org_id"] == "my-special-org"

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_summary_store_called_with_correct_params(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_stats.return_value = _make_deliberation_stats()
        mock_get_store.return_value = mock_store

        handler._get_deliberation_summary(
            {"org_id": "org_123", "days": "7"},
            handler=mock_http,
        )
        call_kwargs = mock_store.get_deliberation_stats.call_args.kwargs
        assert call_kwargs["org_id"] == "org_123"
        assert "start_time" in call_kwargs
        assert "end_time" in call_kwargs

    def test_summary_none_handler_returns_401(self, handler):
        result = handler._get_deliberation_summary(
            {"org_id": "org_123"},
            handler=None,
        )
        assert _status(result) == 401


# =========================================================================
# 2. CHANNEL BREAKDOWN ENDPOINT - GET /api/analytics/deliberations/channels
# =========================================================================


class TestDeliberationByChannel:
    """Tests for _get_deliberation_by_channel endpoint."""

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_channel_happy_path(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_stats_by_channel.return_value = _make_channel_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_by_channel(
            {"org_id": "org_123", "days": "30"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["org_id"] == "org_123"
        assert "channels" in body
        assert "by_platform" in body

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_channel_platform_aggregation(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_stats_by_channel.return_value = _make_channel_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_by_channel(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        by_platform = body["by_platform"]
        assert "web" in by_platform
        assert "api" in by_platform
        assert "cli" in by_platform
        assert by_platform["web"]["count"] == 50
        assert by_platform["api"]["count"] == 30
        assert by_platform["cli"]["count"] == 20

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_channel_consensus_rate_calculation(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_stats_by_channel.return_value = _make_channel_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_by_channel(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        # web: 40/50 = 80%
        assert body["by_platform"]["web"]["consensus_rate"] == "80%"
        # api: 20/30 = 67%
        assert body["by_platform"]["api"]["consensus_rate"] == "67%"
        # cli: 15/20 = 75%
        assert body["by_platform"]["cli"]["consensus_rate"] == "75%"

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_channel_multiple_entries_same_platform(self, mock_get_store, handler, mock_http):
        """Multiple channel entries for the same platform should be aggregated."""
        mock_store = MagicMock()
        mock_store.get_deliberation_stats_by_channel.return_value = [
            {
                "platform": "web",
                "total_deliberations": 30,
                "consensus_reached": 20,
                "total_duration": 1000,
            },
            {
                "platform": "web",
                "total_deliberations": 20,
                "consensus_reached": 15,
                "total_duration": 800,
            },
        ]
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_by_channel(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["by_platform"]["web"]["count"] == 50
        # 35/50 = 70%
        assert body["by_platform"]["web"]["consensus_rate"] == "70%"

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_channel_missing_platform_defaults_to_api(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_stats_by_channel.return_value = [
            {"total_deliberations": 10, "consensus_reached": 5, "total_duration": 500},
        ]
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_by_channel(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert "api" in body["by_platform"]
        assert body["by_platform"]["api"]["count"] == 10

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_channel_zero_count_platform(self, mock_get_store, handler, mock_http):
        """Platform with zero count should show 0% consensus."""
        mock_store = MagicMock()
        mock_store.get_deliberation_stats_by_channel.return_value = [
            {
                "platform": "web",
                "total_deliberations": 0,
                "consensus_reached": 0,
                "total_duration": 0,
            },
        ]
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_by_channel(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["by_platform"]["web"]["consensus_rate"] == "0%"

    def test_channel_missing_org_id(self, handler, mock_http):
        result = handler._get_deliberation_by_channel({}, handler=mock_http)
        assert _status(result) == 400

    def test_channel_empty_org_id(self, handler, mock_http):
        result = handler._get_deliberation_by_channel({"org_id": ""}, handler=mock_http)
        assert _status(result) == 400

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_channel_default_days(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_stats_by_channel.return_value = []
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_by_channel(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["period"]["days"] == 30

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_channel_custom_days(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_stats_by_channel.return_value = []
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_by_channel(
            {"org_id": "org_123", "days": "7"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["period"]["days"] == 7

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_channel_period_structure(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_stats_by_channel.return_value = []
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_by_channel(
            {"org_id": "org_123", "days": "14"},
            handler=mock_http,
        )
        body = _body(result)
        assert "period" in body
        assert "start" in body["period"]
        assert "end" in body["period"]
        assert body["period"]["days"] == 14

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_channel_empty_results(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_stats_by_channel.return_value = []
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_by_channel(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["channels"] == []
        assert body["by_platform"] == {}

    @patch("aragora.memory.debate_store.get_debate_store", side_effect=ImportError("no module"))
    def test_channel_import_error(self, _mock, handler, mock_http):
        result = handler._get_deliberation_by_channel(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.memory.debate_store.get_debate_store", side_effect=RuntimeError("db error"))
    def test_channel_runtime_error(self, _mock, handler, mock_http):
        result = handler._get_deliberation_by_channel(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.memory.debate_store.get_debate_store", side_effect=OSError("disk failure"))
    def test_channel_os_error(self, _mock, handler, mock_http):
        result = handler._get_deliberation_by_channel(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.memory.debate_store.get_debate_store", side_effect=LookupError("not found"))
    def test_channel_lookup_error(self, _mock, handler, mock_http):
        result = handler._get_deliberation_by_channel(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    def test_channel_none_handler_returns_401(self, handler):
        result = handler._get_deliberation_by_channel(
            {"org_id": "org_123"},
            handler=None,
        )
        assert _status(result) == 401

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_channel_days_clamped_min(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_stats_by_channel.return_value = []
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_by_channel(
            {"org_id": "org_123", "days": "0"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["period"]["days"] >= 1

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_channel_days_clamped_max(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_stats_by_channel.return_value = []
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_by_channel(
            {"org_id": "org_123", "days": "9999"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["period"]["days"] <= 365


# =========================================================================
# 3. CONSENSUS RATES ENDPOINT - GET /api/analytics/deliberations/consensus
# =========================================================================


class TestConsensusRates:
    """Tests for _get_consensus_rates endpoint."""

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_consensus_happy_path(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_consensus_stats.return_value = _make_consensus_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_consensus_rates(
            {"org_id": "org_123", "days": "30"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["org_id"] == "org_123"
        assert body["overall_consensus_rate"] == "75.0%"

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_consensus_by_team_size(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_consensus_stats.return_value = _make_consensus_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_consensus_rates(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert "by_team_size" in body
        assert body["by_team_size"]["3"] == "80.0%"

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_consensus_by_agent(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_consensus_stats.return_value = _make_consensus_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_consensus_rates(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert "by_agent" in body
        assert len(body["by_agent"]) == 2

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_consensus_top_teams(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_consensus_stats.return_value = _make_consensus_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_consensus_rates(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert "top_teams" in body
        assert len(body["top_teams"]) == 2

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_consensus_default_days(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_consensus_stats.return_value = _make_consensus_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_consensus_rates(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["period"]["days"] == 30

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_consensus_period_structure(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_consensus_stats.return_value = _make_consensus_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_consensus_rates(
            {"org_id": "org_123", "days": "14"},
            handler=mock_http,
        )
        body = _body(result)
        assert "period" in body
        assert "start" in body["period"]
        assert "end" in body["period"]
        assert body["period"]["days"] == 14

    def test_consensus_missing_org_id(self, handler, mock_http):
        result = handler._get_consensus_rates({}, handler=mock_http)
        assert _status(result) == 400

    def test_consensus_empty_org_id(self, handler, mock_http):
        result = handler._get_consensus_rates({"org_id": ""}, handler=mock_http)
        assert _status(result) == 400

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_consensus_missing_stat_keys_default(self, mock_get_store, handler, mock_http):
        """Stats dict with missing keys should use defaults."""
        mock_store = MagicMock()
        mock_store.get_consensus_stats.return_value = {}
        mock_get_store.return_value = mock_store

        result = handler._get_consensus_rates(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["overall_consensus_rate"] == "0%"
        assert body["by_team_size"] == {}
        assert body["by_agent"] == []
        assert body["top_teams"] == []

    @patch("aragora.memory.debate_store.get_debate_store", side_effect=ImportError("no module"))
    def test_consensus_import_error(self, _mock, handler, mock_http):
        result = handler._get_consensus_rates(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.memory.debate_store.get_debate_store", side_effect=RuntimeError("db error"))
    def test_consensus_runtime_error(self, _mock, handler, mock_http):
        result = handler._get_consensus_rates(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.memory.debate_store.get_debate_store", side_effect=OSError("disk failure"))
    def test_consensus_os_error(self, _mock, handler, mock_http):
        result = handler._get_consensus_rates(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.memory.debate_store.get_debate_store", side_effect=LookupError("not found"))
    def test_consensus_lookup_error(self, _mock, handler, mock_http):
        result = handler._get_consensus_rates(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    def test_consensus_none_handler_returns_401(self, handler):
        result = handler._get_consensus_rates(
            {"org_id": "org_123"},
            handler=None,
        )
        assert _status(result) == 401

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_consensus_days_clamped_min(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_consensus_stats.return_value = _make_consensus_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_consensus_rates(
            {"org_id": "org_123", "days": "-10"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["period"]["days"] >= 1

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_consensus_days_clamped_max(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_consensus_stats.return_value = _make_consensus_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_consensus_rates(
            {"org_id": "org_123", "days": "10000"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["period"]["days"] <= 365


# =========================================================================
# 4. PERFORMANCE ENDPOINT - GET /api/analytics/deliberations/performance
# =========================================================================


class TestDeliberationPerformance:
    """Tests for _get_deliberation_performance endpoint."""

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_performance_happy_path(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_performance.return_value = _make_performance_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_performance(
            {"org_id": "org_123", "days": "30"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["org_id"] == "org_123"
        assert "summary" in body
        assert "by_template" in body
        assert "trends" in body
        assert "cost_by_agent" in body

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_performance_default_granularity(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_performance.return_value = _make_performance_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_performance(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["granularity"] == "day"

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_performance_granularity_week(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_performance.return_value = _make_performance_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_performance(
            {"org_id": "org_123", "granularity": "week"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["granularity"] == "week"

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_performance_invalid_granularity_defaults_to_day(
        self, mock_get_store, handler, mock_http
    ):
        mock_store = MagicMock()
        mock_store.get_deliberation_performance.return_value = _make_performance_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_performance(
            {"org_id": "org_123", "granularity": "minute"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["granularity"] == "day"

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_performance_empty_granularity_defaults_to_day(
        self, mock_get_store, handler, mock_http
    ):
        mock_store = MagicMock()
        mock_store.get_deliberation_performance.return_value = _make_performance_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_performance(
            {"org_id": "org_123", "granularity": ""},
            handler=mock_http,
        )
        body = _body(result)
        assert body["granularity"] == "day"

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_performance_default_days(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_performance.return_value = _make_performance_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_performance(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["period"]["days"] == 30

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_performance_period_structure(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_performance.return_value = _make_performance_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_performance(
            {"org_id": "org_123", "days": "14"},
            handler=mock_http,
        )
        body = _body(result)
        assert "period" in body
        assert "start" in body["period"]
        assert "end" in body["period"]
        assert body["period"]["days"] == 14

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_performance_summary_passthrough(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        perf = _make_performance_stats()
        mock_store.get_deliberation_performance.return_value = perf
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_performance(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["summary"] == perf["summary"]

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_performance_by_template_passthrough(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        perf = _make_performance_stats()
        mock_store.get_deliberation_performance.return_value = perf
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_performance(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["by_template"] == perf["by_template"]

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_performance_trends_passthrough(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        perf = _make_performance_stats()
        mock_store.get_deliberation_performance.return_value = perf
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_performance(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["trends"] == perf["trends"]

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_performance_cost_by_agent_passthrough(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        perf = _make_performance_stats()
        mock_store.get_deliberation_performance.return_value = perf
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_performance(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["cost_by_agent"] == perf["cost_by_agent"]

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_performance_missing_stat_keys_default(self, mock_get_store, handler, mock_http):
        """Stats dict with missing keys should use defaults."""
        mock_store = MagicMock()
        mock_store.get_deliberation_performance.return_value = {}
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_performance(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["summary"] == {}
        assert body["by_template"] == []
        assert body["trends"] == []
        assert body["cost_by_agent"] == {}

    def test_performance_missing_org_id(self, handler, mock_http):
        result = handler._get_deliberation_performance({}, handler=mock_http)
        assert _status(result) == 400

    def test_performance_empty_org_id(self, handler, mock_http):
        result = handler._get_deliberation_performance({"org_id": ""}, handler=mock_http)
        assert _status(result) == 400

    @patch("aragora.memory.debate_store.get_debate_store", side_effect=ImportError("no module"))
    def test_performance_import_error(self, _mock, handler, mock_http):
        result = handler._get_deliberation_performance(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.memory.debate_store.get_debate_store", side_effect=RuntimeError("db error"))
    def test_performance_runtime_error(self, _mock, handler, mock_http):
        result = handler._get_deliberation_performance(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.memory.debate_store.get_debate_store", side_effect=OSError("disk failure"))
    def test_performance_os_error(self, _mock, handler, mock_http):
        result = handler._get_deliberation_performance(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.memory.debate_store.get_debate_store", side_effect=LookupError("not found"))
    def test_performance_lookup_error(self, _mock, handler, mock_http):
        result = handler._get_deliberation_performance(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_performance_granularity_passed_to_store(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_performance.return_value = _make_performance_stats()
        mock_get_store.return_value = mock_store

        handler._get_deliberation_performance(
            {"org_id": "org_123", "granularity": "week"},
            handler=mock_http,
        )
        call_kwargs = mock_store.get_deliberation_performance.call_args.kwargs
        assert call_kwargs["granularity"] == "week"

    def test_performance_none_handler_returns_401(self, handler):
        result = handler._get_deliberation_performance(
            {"org_id": "org_123"},
            handler=None,
        )
        assert _status(result) == 401

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_performance_days_clamped_min(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_performance.return_value = _make_performance_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_performance(
            {"org_id": "org_123", "days": "0"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["period"]["days"] >= 1

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_performance_days_clamped_max(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_performance.return_value = _make_performance_stats()
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_performance(
            {"org_id": "org_123", "days": "9999"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["period"]["days"] <= 365


# =========================================================================
# 5. ROUTING TESTS
# =========================================================================


class TestRouting:
    """Tests for route registration and can_handle."""

    def test_deliberations_route_registered(self):
        assert "/api/analytics/deliberations" in AnalyticsDashboardHandler.ROUTES

    def test_deliberations_channels_route_registered(self):
        assert "/api/analytics/deliberations/channels" in AnalyticsDashboardHandler.ROUTES

    def test_deliberations_consensus_route_registered(self):
        assert "/api/analytics/deliberations/consensus" in AnalyticsDashboardHandler.ROUTES

    def test_deliberations_performance_route_registered(self):
        assert "/api/analytics/deliberations/performance" in AnalyticsDashboardHandler.ROUTES

    def test_can_handle_deliberations(self, handler):
        assert handler.can_handle("/api/analytics/deliberations") is True

    def test_can_handle_deliberations_channels(self, handler):
        assert handler.can_handle("/api/analytics/deliberations/channels") is True

    def test_can_handle_deliberations_consensus(self, handler):
        assert handler.can_handle("/api/analytics/deliberations/consensus") is True

    def test_can_handle_deliberations_performance(self, handler):
        assert handler.can_handle("/api/analytics/deliberations/performance") is True

    def test_can_handle_versioned_deliberations(self, handler):
        assert handler.can_handle("/api/v1/analytics/deliberations") is True

    def test_can_handle_versioned_channels(self, handler):
        assert handler.can_handle("/api/v1/analytics/deliberations/channels") is True

    def test_can_handle_versioned_consensus(self, handler):
        assert handler.can_handle("/api/v1/analytics/deliberations/consensus") is True

    def test_can_handle_versioned_performance(self, handler):
        assert handler.can_handle("/api/v1/analytics/deliberations/performance") is True

    def test_cannot_handle_unrelated_path(self, handler):
        assert handler.can_handle("/api/debates") is False

    def test_cannot_handle_partial_match(self, handler):
        assert handler.can_handle("/api/analytics/deliberationsextra") is False

    def test_cannot_handle_empty(self, handler):
        assert handler.can_handle("") is False

    def test_has_deliberation_summary_method(self, handler):
        assert hasattr(handler, "_get_deliberation_summary")
        assert callable(handler._get_deliberation_summary)

    def test_has_deliberation_by_channel_method(self, handler):
        assert hasattr(handler, "_get_deliberation_by_channel")
        assert callable(handler._get_deliberation_by_channel)

    def test_has_consensus_rates_method(self, handler):
        assert hasattr(handler, "_get_consensus_rates")
        assert callable(handler._get_consensus_rates)

    def test_has_deliberation_performance_method(self, handler):
        assert hasattr(handler, "_get_deliberation_performance")
        assert callable(handler._get_deliberation_performance)


# =========================================================================
# 6. HANDLE() METHOD ROUTING
# =========================================================================


class TestHandleRouting:
    """Tests that handle() routes to correct method for deliberation endpoints."""

    def test_handle_deliberations_stub_without_user(self, handler):
        """When no user context, stub response is returned."""
        mock_h = MagicMock()
        handler.get_current_user = MagicMock(return_value=None)
        result = handler.handle("/api/analytics/deliberations", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert "summary" in body

    def test_handle_channels_stub_without_user(self, handler):
        mock_h = MagicMock()
        handler.get_current_user = MagicMock(return_value=None)
        result = handler.handle("/api/analytics/deliberations/channels", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert "channels" in body

    def test_handle_consensus_stub_without_user(self, handler):
        mock_h = MagicMock()
        handler.get_current_user = MagicMock(return_value=None)
        result = handler.handle("/api/analytics/deliberations/consensus", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert "consensus" in body

    def test_handle_performance_stub_without_user(self, handler):
        mock_h = MagicMock()
        handler.get_current_user = MagicMock(return_value=None)
        result = handler.handle("/api/analytics/deliberations/performance", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert "performance" in body

    def test_handle_versioned_deliberations(self, handler):
        """Versioned paths should be normalized and routed correctly."""
        mock_h = MagicMock()
        handler.get_current_user = MagicMock(return_value=None)
        result = handler.handle("/api/v1/analytics/deliberations", {}, mock_h)
        assert _status(result) == 200

    def test_handle_versioned_channels(self, handler):
        mock_h = MagicMock()
        handler.get_current_user = MagicMock(return_value=None)
        result = handler.handle("/api/v1/analytics/deliberations/channels", {}, mock_h)
        assert _status(result) == 200

    def test_handle_versioned_consensus(self, handler):
        mock_h = MagicMock()
        handler.get_current_user = MagicMock(return_value=None)
        result = handler.handle("/api/v1/analytics/deliberations/consensus", {}, mock_h)
        assert _status(result) == 200

    def test_handle_versioned_performance(self, handler):
        mock_h = MagicMock()
        handler.get_current_user = MagicMock(return_value=None)
        result = handler.handle("/api/v1/analytics/deliberations/performance", {}, mock_h)
        assert _status(result) == 200

    def test_handle_unknown_route_returns_none(self, handler):
        mock_h = MagicMock()
        result = handler.handle("/api/unknown/route", {}, mock_h)
        assert result is None


# =========================================================================
# 7. SECURITY TESTS
# =========================================================================


class TestSecurityInputs:
    """Tests for input validation and security concerns."""

    def test_org_id_with_path_traversal(self, handler, mock_http):
        result = handler._get_deliberation_summary(
            {"org_id": "../../etc/passwd"},
            handler=mock_http,
        )
        assert _status(result) in (200, 400, 500)

    def test_org_id_with_sql_injection(self, handler, mock_http):
        result = handler._get_deliberation_summary(
            {"org_id": "'; DROP TABLE debates; --"},
            handler=mock_http,
        )
        assert _status(result) in (200, 400, 500)

    def test_very_long_org_id(self, handler, mock_http):
        result = handler._get_deliberation_summary(
            {"org_id": "x" * 10000},
            handler=mock_http,
        )
        assert _status(result) in (200, 400, 500)

    def test_unicode_org_id(self, handler, mock_http):
        result = handler._get_deliberation_summary(
            {"org_id": "\u0000\ud83d\ude80\u2603"},
            handler=mock_http,
        )
        assert _status(result) in (200, 400, 500)

    def test_channel_org_id_with_path_traversal(self, handler, mock_http):
        result = handler._get_deliberation_by_channel(
            {"org_id": "../../../etc/shadow"},
            handler=mock_http,
        )
        assert _status(result) in (200, 400, 500)

    def test_channel_org_id_with_sql_injection(self, handler, mock_http):
        result = handler._get_deliberation_by_channel(
            {"org_id": "1 OR 1=1; --"},
            handler=mock_http,
        )
        assert _status(result) in (200, 400, 500)

    def test_consensus_org_id_with_xss(self, handler, mock_http):
        result = handler._get_consensus_rates(
            {"org_id": "<script>alert('xss')</script>"},
            handler=mock_http,
        )
        assert _status(result) in (200, 400, 500)

    def test_performance_org_id_with_null_bytes(self, handler, mock_http):
        result = handler._get_deliberation_performance(
            {"org_id": "org\x00injected"},
            handler=mock_http,
        )
        assert _status(result) in (200, 400, 500)

    def test_granularity_with_injection(self, handler, mock_http):
        """Granularity injection should fall back to 'day'."""
        result = handler._get_deliberation_performance(
            {"org_id": "org_123", "granularity": "'; DROP TABLE --"},
            handler=mock_http,
        )
        assert _status(result) in (200, 400, 500)

    def test_days_with_float(self, handler, mock_http):
        """Float days should be handled by get_clamped_int_param."""
        result = handler._get_deliberation_summary(
            {"org_id": "org_123", "days": "7.5"},
            handler=mock_http,
        )
        assert _status(result) in (200, 400, 500)

    def test_days_with_xss(self, handler, mock_http):
        result = handler._get_deliberation_summary(
            {"org_id": "org_123", "days": "<img src=x onerror=alert(1)>"},
            handler=mock_http,
        )
        assert _status(result) in (200, 400, 500)

    def test_days_with_very_large_number(self, handler, mock_http):
        result = handler._get_deliberation_summary(
            {"org_id": "org_123", "days": "999999999"},
            handler=mock_http,
        )
        assert _status(result) in (200, 400, 500)


# =========================================================================
# 8. MIXIN ISOLATION TESTS
# =========================================================================


class TestMixinIsolation:
    """Test that DeliberationAnalyticsMixin works as a standalone mixin."""

    def test_mixin_can_be_instantiated_standalone(self):
        from aragora.server.handlers.analytics_dashboard.endpoints import DeliberationAnalyticsMixin

        class TestHandler(DeliberationAnalyticsMixin):
            pass

        h = TestHandler()
        assert hasattr(h, "_get_deliberation_summary")
        assert hasattr(h, "_get_deliberation_by_channel")
        assert hasattr(h, "_get_consensus_rates")
        assert hasattr(h, "_get_deliberation_performance")

    def test_mixin_methods_are_callable(self):
        from aragora.server.handlers.analytics_dashboard.endpoints import DeliberationAnalyticsMixin

        class TestHandler(DeliberationAnalyticsMixin):
            pass

        h = TestHandler()
        assert callable(h._get_deliberation_summary)
        assert callable(h._get_deliberation_by_channel)
        assert callable(h._get_consensus_rates)
        assert callable(h._get_deliberation_performance)


# =========================================================================
# 9. EDGE CASE TESTS
# =========================================================================


class TestEdgeCases:
    """Tests for various edge cases."""

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_summary_all_zeroes(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        stats = _make_deliberation_stats(
            total=0,
            completed=0,
            consensus_reached=0,
            in_progress=0,
            failed=0,
            avg_rounds=0,
            avg_duration_seconds=0,
        )
        mock_store.get_deliberation_stats.return_value = stats
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_summary(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["total_deliberations"] == 0
        assert body["consensus_rate"] == "0%"
        assert body["avg_rounds"] == 0
        assert body["avg_duration_seconds"] == 0

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_summary_large_values(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        stats = _make_deliberation_stats(
            total=1_000_000,
            completed=999_999,
            consensus_reached=800_000,
            in_progress=1,
            failed=0,
        )
        mock_store.get_deliberation_stats.return_value = stats
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_summary(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["total_deliberations"] == 1_000_000
        assert body["completed"] == 999_999

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_channel_many_platforms(self, mock_get_store, handler, mock_http):
        """Many different platforms should all be aggregated."""
        mock_store = MagicMock()
        channels = [
            {
                "platform": f"platform_{i}",
                "total_deliberations": 10,
                "consensus_reached": 5,
                "total_duration": 100,
            }
            for i in range(20)
        ]
        mock_store.get_deliberation_stats_by_channel.return_value = channels
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_by_channel(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert len(body["by_platform"]) == 20

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_performance_empty_stats(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_performance.return_value = {}
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_performance(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["summary"] == {}
        assert body["by_template"] == []
        assert body["trends"] == []
        assert body["cost_by_agent"] == {}

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_consensus_store_called_with_correct_params(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_consensus_stats.return_value = _make_consensus_stats()
        mock_get_store.return_value = mock_store

        handler._get_consensus_rates(
            {"org_id": "org_abc", "days": "7"},
            handler=mock_http,
        )
        call_kwargs = mock_store.get_consensus_stats.call_args.kwargs
        assert call_kwargs["org_id"] == "org_abc"
        assert "start_time" in call_kwargs
        assert "end_time" in call_kwargs

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_channel_store_called_with_correct_params(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_stats_by_channel.return_value = []
        mock_get_store.return_value = mock_store

        handler._get_deliberation_by_channel(
            {"org_id": "org_xyz", "days": "14"},
            handler=mock_http,
        )
        call_kwargs = mock_store.get_deliberation_stats_by_channel.call_args.kwargs
        assert call_kwargs["org_id"] == "org_xyz"
        assert "start_time" in call_kwargs
        assert "end_time" in call_kwargs

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_performance_store_called_with_granularity(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_performance.return_value = _make_performance_stats()
        mock_get_store.return_value = mock_store

        handler._get_deliberation_performance(
            {"org_id": "org_123", "days": "7", "granularity": "week"},
            handler=mock_http,
        )
        call_kwargs = mock_store.get_deliberation_performance.call_args.kwargs
        assert call_kwargs["org_id"] == "org_123"
        assert call_kwargs["granularity"] == "week"

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_summary_consensus_rate_decimal_precision(self, mock_get_store, handler, mock_http):
        """Consensus rate with recurring decimal."""
        mock_store = MagicMock()
        stats = _make_deliberation_stats(completed=3, consensus_reached=1)
        mock_store.get_deliberation_stats.return_value = stats
        mock_get_store.return_value = mock_store

        result = handler._get_deliberation_summary(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        body = _body(result)
        # 1/3 * 100 = 33.3333... -> "33.3%"
        assert body["consensus_rate"] == "33.3%"

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_summary_store_method_called_once(self, mock_get_store, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_deliberation_stats.return_value = _make_deliberation_stats()
        mock_get_store.return_value = mock_store

        handler._get_deliberation_summary(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert mock_store.get_deliberation_stats.call_count == 1

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_summary_error_response_contains_code(self, mock_get_store, handler, mock_http):
        """Internal errors should include error code."""
        mock_get_store.side_effect = RuntimeError("db down")

        result = handler._get_deliberation_summary(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500
        body = _body(result)
        error = body.get("error", {})
        if isinstance(error, dict):
            assert error.get("code") == "INTERNAL_ERROR"

    @patch("aragora.memory.debate_store.get_debate_store")
    def test_channel_error_response_contains_code(self, mock_get_store, handler, mock_http):
        mock_get_store.side_effect = RuntimeError("db down")

        result = handler._get_deliberation_by_channel(
            {"org_id": "org_123"},
            handler=mock_http,
        )
        assert _status(result) == 500
        body = _body(result)
        error = body.get("error", {})
        if isinstance(error, dict):
            assert error.get("code") == "INTERNAL_ERROR"

    def test_channel_missing_org_id_error_code(self, handler, mock_http):
        result = handler._get_deliberation_by_channel({}, handler=mock_http)
        assert _status(result) == 400
        body = _body(result)
        error = body.get("error", {})
        if isinstance(error, dict):
            assert error.get("code") == "MISSING_ORG_ID"

    def test_consensus_missing_org_id_error_code(self, handler, mock_http):
        result = handler._get_consensus_rates({}, handler=mock_http)
        assert _status(result) == 400
        body = _body(result)
        error = body.get("error", {})
        if isinstance(error, dict):
            assert error.get("code") == "MISSING_ORG_ID"

    def test_performance_missing_org_id_error_code(self, handler, mock_http):
        result = handler._get_deliberation_performance({}, handler=mock_http)
        assert _status(result) == 400
        body = _body(result)
        error = body.get("error", {})
        if isinstance(error, dict):
            assert error.get("code") == "MISSING_ORG_ID"
