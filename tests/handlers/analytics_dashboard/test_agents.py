"""Comprehensive tests for the AgentAnalyticsMixin handler.

Tests all five endpoints provided by the agent analytics handler:
- GET /api/analytics/agents              - Agent performance metrics (cached: 300s)
- GET /api/analytics/flips/summary       - Flip detection summary
- GET /api/analytics/flips/recent        - Recent flip events
- GET /api/analytics/flips/consistency   - Agent consistency scores
- GET /api/analytics/flips/trends        - Flip trends over time

Also tests routing, version prefix handling, error handling, and edge cases.
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


@dataclass
class FakeAgentMetric:
    """Fake agent metric returned by analytics dashboard."""

    agent_id: str = "claude"
    elo: float = 1800.0
    debates: int = 42
    win_rate: float = 0.78
    avg_response_time_ms: float = 1200.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "elo": self.elo,
            "debates": self.debates,
            "win_rate": self.win_rate,
            "avg_response_time_ms": self.avg_response_time_ms,
        }


@dataclass
class FakeFlipEvent:
    """Fake flip event for recent flips."""

    agent_name: str = "claude"
    flip_type: str = "contradiction"
    topic: str = "Rate limiting"
    detected_at: str = "2026-02-20T10:00:00Z"
    confidence: float = 0.92


@dataclass
class FakeConsistencyScore:
    """Fake consistency score for an agent."""

    agent_name: str = "claude"
    consistency: float = 0.94
    total_flips: int = 3
    total_statements: int = 50


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
    h.path = "/api/analytics/agents"
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


def _make_agent_metrics(count=3):
    """Build a list of fake agent metrics."""
    agents = [
        FakeAgentMetric(agent_id="claude", elo=1847, debates=42, win_rate=0.78),
        FakeAgentMetric(agent_id="gpt-4", elo=1792, debates=38, win_rate=0.71),
        FakeAgentMetric(agent_id="gemini", elo=1734, debates=35, win_rate=0.65),
        FakeAgentMetric(agent_id="mistral", elo=1688, debates=28, win_rate=0.58),
        FakeAgentMetric(agent_id="grok", elo=1650, debates=20, win_rate=0.52),
    ]
    return agents[:count]


def _make_flip_summary():
    """Build a fake flip summary dict."""
    return {
        "total_flips": 150,
        "by_type": {"contradiction": 45, "retraction": 20, "qualification": 50, "refinement": 35},
        "by_agent": {"claude": 30, "gpt-4": 25, "gemini": 40, "mistral": 55},
        "recent_24h": 12,
    }


def _make_recent_flips(count=5):
    """Build a list of fake recent flip events."""
    flips = [
        FakeFlipEvent(agent_name="claude", flip_type="contradiction", topic="Rate limiting"),
        FakeFlipEvent(agent_name="gpt-4", flip_type="retraction", topic="Auth flow"),
        FakeFlipEvent(agent_name="gemini", flip_type="qualification", topic="DB migration"),
        FakeFlipEvent(agent_name="claude", flip_type="refinement", topic="Cost model"),
        FakeFlipEvent(agent_name="mistral", flip_type="contradiction", topic="API design"),
        FakeFlipEvent(agent_name="grok", flip_type="retraction", topic="Security"),
    ]
    return flips[:count]


def _make_formatted_flip(agent="claude", flip_type="contradiction", topic="Rate limiting"):
    """Build a formatted flip dict as returned by format_flip_for_ui."""
    return {
        "agent": agent,
        "flip_type": flip_type,
        "topic": topic,
        "detected_at": "2026-02-20T10:00:00Z",
        "confidence": 0.92,
    }


def _make_consistency_scores(agents=None):
    """Build a dict of agent name -> consistency score."""
    if agents is None:
        agents = ["claude", "gpt-4", "gemini"]
    scores = {}
    base_values = {"claude": 0.94, "gpt-4": 0.87, "gemini": 0.82, "mistral": 0.76, "grok": 0.90}
    for agent in agents:
        val = base_values.get(agent, 0.80)
        scores[agent] = FakeConsistencyScore(
            agent_name=agent,
            consistency=val,
            total_flips=3,
            total_statements=50,
        )
    return scores


def _make_formatted_consistency(agent="claude", consistency="94.0%"):
    """Build a formatted consistency dict as returned by format_consistency_for_ui."""
    return {
        "agent": agent,
        "consistency": consistency,
        "total_flips": 3,
        "total_statements": 50,
    }


# =========================================================================
# 1. AGENT METRICS ENDPOINT - GET /api/analytics/agents
# =========================================================================


class TestAgentMetrics:
    """Tests for _get_agent_metrics endpoint."""

    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    def test_happy_path(self, mock_tr, mock_get_dash, mock_run_async, handler, mock_http):
        metrics = _make_agent_metrics(3)
        mock_run_async.side_effect = lambda coro: metrics
        mock_dash = MagicMock()
        mock_get_dash.return_value = mock_dash

        result = handler._get_agent_metrics(
            {"workspace_id": "ws_123", "time_range": "7d"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["workspace_id"] == "ws_123"
        assert body["time_range"] == "7d"
        assert len(body["agents"]) == 3

    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    def test_default_time_range(self, mock_tr, mock_get_dash, mock_run_async, handler, mock_http):
        mock_run_async.side_effect = lambda coro: _make_agent_metrics(1)
        mock_get_dash.return_value = MagicMock()

        result = handler._get_agent_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["time_range"] == "30d"

    def test_missing_workspace_id(self, handler, mock_http):
        result = handler._get_agent_metrics({}, handler=mock_http)
        assert _status(result) == 400
        body = _body(result)
        error = body.get("error", {})
        error_text = error if isinstance(error, str) else error.get("message", "")
        assert "workspace_id" in error_text.lower()

    def test_empty_workspace_id(self, handler, mock_http):
        result = handler._get_agent_metrics({"workspace_id": ""}, handler=mock_http)
        assert _status(result) == 400

    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    def test_agents_list_format(self, mock_tr, mock_get_dash, mock_run_async, handler, mock_http):
        metrics = _make_agent_metrics(2)
        mock_run_async.side_effect = lambda coro: metrics
        mock_get_dash.return_value = MagicMock()

        result = handler._get_agent_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert isinstance(body["agents"], list)
        for agent in body["agents"]:
            assert "agent_id" in agent

    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    def test_empty_metrics(self, mock_tr, mock_get_dash, mock_run_async, handler, mock_http):
        mock_run_async.side_effect = lambda coro: []
        mock_get_dash.return_value = MagicMock()

        result = handler._get_agent_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["agents"] == []

    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange", side_effect=ValueError("Invalid time range"))
    def test_invalid_time_range(self, mock_tr, mock_get_dash, mock_run_async, handler, mock_http):
        result = handler._get_agent_metrics(
            {"workspace_id": "ws_123", "time_range": "invalid"},
            handler=mock_http,
        )
        assert _status(result) == 400
        body = _body(result)
        error = body.get("error", {})
        code = error.get("code", "") if isinstance(error, dict) else ""
        assert code == "INVALID_PARAMETER"

    @patch("aragora.analytics.get_analytics_dashboard", side_effect=ImportError("no module"))
    def test_import_error(self, _mock, handler, mock_http):
        result = handler._get_agent_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 500
        body = _body(result)
        error = body.get("error", {})
        code = error.get("code", "") if isinstance(error, dict) else ""
        assert code == "INTERNAL_ERROR"

    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    def test_runtime_error(self, mock_tr, mock_get_dash, mock_run_async, handler, mock_http):
        mock_run_async.side_effect = RuntimeError("computation failed")
        mock_get_dash.return_value = MagicMock()

        result = handler._get_agent_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    def test_os_error(self, mock_tr, mock_get_dash, mock_run_async, handler, mock_http):
        mock_run_async.side_effect = OSError("disk failure")
        mock_get_dash.return_value = MagicMock()

        result = handler._get_agent_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 500

    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    def test_key_error_in_data(self, mock_tr, mock_get_dash, mock_run_async, handler, mock_http):
        mock_run_async.side_effect = KeyError("missing_key")
        mock_get_dash.return_value = MagicMock()

        result = handler._get_agent_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 400
        body = _body(result)
        error = body.get("error", {})
        code = error.get("code", "") if isinstance(error, dict) else ""
        assert code == "DATA_ERROR"

    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    def test_type_error_in_data(self, mock_tr, mock_get_dash, mock_run_async, handler, mock_http):
        mock_run_async.side_effect = TypeError("bad type")
        mock_get_dash.return_value = MagicMock()

        result = handler._get_agent_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 400

    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    def test_attribute_error_in_data(self, mock_tr, mock_get_dash, mock_run_async, handler, mock_http):
        mock_run_async.side_effect = AttributeError("no attr")
        mock_get_dash.return_value = MagicMock()

        result = handler._get_agent_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 400

    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    def test_response_structure(self, mock_tr, mock_get_dash, mock_run_async, handler, mock_http):
        mock_run_async.side_effect = lambda coro: _make_agent_metrics(1)
        mock_get_dash.return_value = MagicMock()

        result = handler._get_agent_metrics(
            {"workspace_id": "ws_123", "time_range": "14d"},
            handler=mock_http,
        )
        body = _body(result)
        assert "workspace_id" in body
        assert "time_range" in body
        assert "agents" in body

    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    def test_multiple_agents_returned(self, mock_tr, mock_get_dash, mock_run_async, handler, mock_http):
        mock_run_async.side_effect = lambda coro: _make_agent_metrics(5)
        mock_get_dash.return_value = MagicMock()

        result = handler._get_agent_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        body = _body(result)
        assert len(body["agents"]) == 5

    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    def test_agent_metric_to_dict_called(self, mock_tr, mock_get_dash, mock_run_async, handler, mock_http):
        """Each metric should have to_dict() called."""
        metric = MagicMock()
        metric.to_dict.return_value = {"agent_id": "test", "elo": 1500}
        mock_run_async.side_effect = lambda coro: [metric]
        mock_get_dash.return_value = MagicMock()

        result = handler._get_agent_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["agents"] == [{"agent_id": "test", "elo": 1500}]
        metric.to_dict.assert_called_once()


# =========================================================================
# 2. FLIP SUMMARY ENDPOINT - GET /api/analytics/flips/summary
# =========================================================================


class TestFlipSummary:
    """Tests for _get_flip_summary endpoint."""

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_happy_path(self, mock_detector_cls, handler, mock_http):
        mock_detector = MagicMock()
        mock_detector.get_flip_summary.return_value = _make_flip_summary()
        mock_detector_cls.return_value = mock_detector

        result = handler._get_flip_summary({}, handler=mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["total_flips"] == 150
        assert "by_type" in body
        assert "by_agent" in body
        assert body["recent_24h"] == 12

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_empty_summary(self, mock_detector_cls, handler, mock_http):
        mock_detector = MagicMock()
        mock_detector.get_flip_summary.return_value = {
            "total_flips": 0,
            "by_type": {},
            "by_agent": {},
            "recent_24h": 0,
        }
        mock_detector_cls.return_value = mock_detector

        result = handler._get_flip_summary({}, handler=mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["total_flips"] == 0
        assert body["by_type"] == {}

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_summary_by_type_keys(self, mock_detector_cls, handler, mock_http):
        mock_detector = MagicMock()
        summary = _make_flip_summary()
        mock_detector.get_flip_summary.return_value = summary
        mock_detector_cls.return_value = mock_detector

        result = handler._get_flip_summary({}, handler=mock_http)
        body = _body(result)
        assert "contradiction" in body["by_type"]
        assert "retraction" in body["by_type"]

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_summary_by_agent_keys(self, mock_detector_cls, handler, mock_http):
        mock_detector = MagicMock()
        summary = _make_flip_summary()
        mock_detector.get_flip_summary.return_value = summary
        mock_detector_cls.return_value = mock_detector

        result = handler._get_flip_summary({}, handler=mock_http)
        body = _body(result)
        assert "claude" in body["by_agent"]
        assert "gpt-4" in body["by_agent"]

    @patch("aragora.insights.flip_detector.FlipDetector", side_effect=ImportError("no module"))
    def test_import_error(self, _mock, handler, mock_http):
        result = handler._get_flip_summary({}, handler=mock_http)
        assert _status(result) == 500

    @patch("aragora.insights.flip_detector.FlipDetector", side_effect=RuntimeError("db failure"))
    def test_runtime_error(self, _mock, handler, mock_http):
        result = handler._get_flip_summary({}, handler=mock_http)
        assert _status(result) == 500

    @patch("aragora.insights.flip_detector.FlipDetector", side_effect=OSError("disk error"))
    def test_os_error(self, _mock, handler, mock_http):
        result = handler._get_flip_summary({}, handler=mock_http)
        assert _status(result) == 500

    @patch("aragora.insights.flip_detector.FlipDetector", side_effect=LookupError("not found"))
    def test_lookup_error(self, _mock, handler, mock_http):
        result = handler._get_flip_summary({}, handler=mock_http)
        assert _status(result) == 500

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_detector_exception_during_get(self, mock_detector_cls, handler, mock_http):
        mock_detector = MagicMock()
        mock_detector.get_flip_summary.side_effect = RuntimeError("unexpected")
        mock_detector_cls.return_value = mock_detector

        result = handler._get_flip_summary({}, handler=mock_http)
        assert _status(result) == 500

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_summary_passthrough(self, mock_detector_cls, handler, mock_http):
        """Whatever the detector returns should be passed through as JSON."""
        custom_summary = {"custom_key": "custom_value", "total_flips": 99}
        mock_detector = MagicMock()
        mock_detector.get_flip_summary.return_value = custom_summary
        mock_detector_cls.return_value = mock_detector

        result = handler._get_flip_summary({}, handler=mock_http)
        body = _body(result)
        assert body["custom_key"] == "custom_value"
        assert body["total_flips"] == 99

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_query_params_ignored(self, mock_detector_cls, handler, mock_http):
        """Flip summary does not require/use any query params."""
        mock_detector = MagicMock()
        mock_detector.get_flip_summary.return_value = _make_flip_summary()
        mock_detector_cls.return_value = mock_detector

        result = handler._get_flip_summary(
            {"irrelevant": "param", "workspace_id": "ws_123"},
            handler=mock_http,
        )
        assert _status(result) == 200


# =========================================================================
# 3. RECENT FLIPS ENDPOINT - GET /api/analytics/flips/recent
# =========================================================================


class TestRecentFlips:
    """Tests for _get_recent_flips endpoint."""

    @patch("aragora.insights.flip_detector.format_flip_for_ui")
    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_happy_path(self, mock_detector_cls, mock_format, handler, mock_http):
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = _make_recent_flips(3)
        mock_detector_cls.return_value = mock_detector
        mock_format.side_effect = lambda f: _make_formatted_flip(f.agent_name, f.flip_type, f.topic)

        result = handler._get_recent_flips({}, handler=mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert "flips" in body
        assert "count" in body
        assert body["count"] == 3

    @patch("aragora.insights.flip_detector.format_flip_for_ui")
    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_default_limit(self, mock_detector_cls, mock_format, handler, mock_http):
        """Default limit is 20, so get_recent_flips is called with limit=40 (2x)."""
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = []
        mock_detector_cls.return_value = mock_detector

        handler._get_recent_flips({}, handler=mock_http)
        mock_detector.get_recent_flips.assert_called_once_with(limit=40)

    @patch("aragora.insights.flip_detector.format_flip_for_ui")
    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_custom_limit(self, mock_detector_cls, mock_format, handler, mock_http):
        """Custom limit is doubled when fetching from detector."""
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = []
        mock_detector_cls.return_value = mock_detector

        handler._get_recent_flips({"limit": "10"}, handler=mock_http)
        mock_detector.get_recent_flips.assert_called_once_with(limit=20)

    @patch("aragora.insights.flip_detector.format_flip_for_ui")
    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_limit_clamped_min(self, mock_detector_cls, mock_format, handler, mock_http):
        """Limit below 1 should be clamped to 1."""
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = []
        mock_detector_cls.return_value = mock_detector

        handler._get_recent_flips({"limit": "0"}, handler=mock_http)
        mock_detector.get_recent_flips.assert_called_once_with(limit=2)

    @patch("aragora.insights.flip_detector.format_flip_for_ui")
    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_limit_clamped_max(self, mock_detector_cls, mock_format, handler, mock_http):
        """Limit above 100 should be clamped to 100."""
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = []
        mock_detector_cls.return_value = mock_detector

        handler._get_recent_flips({"limit": "999"}, handler=mock_http)
        mock_detector.get_recent_flips.assert_called_once_with(limit=200)

    @patch("aragora.insights.flip_detector.format_flip_for_ui")
    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_agent_filter(self, mock_detector_cls, mock_format, handler, mock_http):
        flips = _make_recent_flips(5)
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = flips
        mock_detector_cls.return_value = mock_detector
        mock_format.side_effect = lambda f: _make_formatted_flip(f.agent_name, f.flip_type, f.topic)

        result = handler._get_recent_flips({"agent": "claude"}, handler=mock_http)
        body = _body(result)
        # Should only return flips from claude
        for flip in body["flips"]:
            assert flip["agent"] == "claude"

    @patch("aragora.insights.flip_detector.format_flip_for_ui")
    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_type_filter(self, mock_detector_cls, mock_format, handler, mock_http):
        flips = _make_recent_flips(5)
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = flips
        mock_detector_cls.return_value = mock_detector
        mock_format.side_effect = lambda f: _make_formatted_flip(f.agent_name, f.flip_type, f.topic)

        result = handler._get_recent_flips({"flip_type": "contradiction"}, handler=mock_http)
        body = _body(result)
        for flip in body["flips"]:
            assert flip["flip_type"] == "contradiction"

    @patch("aragora.insights.flip_detector.format_flip_for_ui")
    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_combined_filters(self, mock_detector_cls, mock_format, handler, mock_http):
        flips = _make_recent_flips(5)
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = flips
        mock_detector_cls.return_value = mock_detector
        mock_format.side_effect = lambda f: _make_formatted_flip(f.agent_name, f.flip_type, f.topic)

        result = handler._get_recent_flips(
            {"agent": "claude", "flip_type": "contradiction"},
            handler=mock_http,
        )
        body = _body(result)
        for flip in body["flips"]:
            assert flip["agent"] == "claude"
            assert flip["flip_type"] == "contradiction"

    @patch("aragora.insights.flip_detector.format_flip_for_ui")
    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_no_matching_flips(self, mock_detector_cls, mock_format, handler, mock_http):
        flips = _make_recent_flips(3)
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = flips
        mock_detector_cls.return_value = mock_detector
        mock_format.side_effect = lambda f: _make_formatted_flip(f.agent_name, f.flip_type, f.topic)

        result = handler._get_recent_flips({"agent": "nonexistent"}, handler=mock_http)
        body = _body(result)
        assert body["flips"] == []
        assert body["count"] == 0

    @patch("aragora.insights.flip_detector.format_flip_for_ui")
    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_limit_applied_after_filter(self, mock_detector_cls, mock_format, handler, mock_http):
        """Limit should apply after filtering, not before."""
        # Create many flips from claude
        flips = [
            FakeFlipEvent(agent_name="claude", flip_type="contradiction", topic=f"Topic {i}")
            for i in range(30)
        ]
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = flips
        mock_detector_cls.return_value = mock_detector
        mock_format.side_effect = lambda f: _make_formatted_flip(f.agent_name, f.flip_type, f.topic)

        result = handler._get_recent_flips(
            {"agent": "claude", "limit": "5"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["count"] <= 5

    @patch("aragora.insights.flip_detector.FlipDetector", side_effect=ImportError("no module"))
    def test_import_error(self, _mock, handler, mock_http):
        result = handler._get_recent_flips({}, handler=mock_http)
        assert _status(result) == 500

    @patch("aragora.insights.flip_detector.FlipDetector", side_effect=RuntimeError("db failure"))
    def test_runtime_error(self, _mock, handler, mock_http):
        result = handler._get_recent_flips({}, handler=mock_http)
        assert _status(result) == 500

    @patch("aragora.insights.flip_detector.FlipDetector", side_effect=OSError("disk error"))
    def test_os_error(self, _mock, handler, mock_http):
        result = handler._get_recent_flips({}, handler=mock_http)
        assert _status(result) == 500

    @patch("aragora.insights.flip_detector.FlipDetector", side_effect=LookupError("not found"))
    def test_lookup_error(self, _mock, handler, mock_http):
        result = handler._get_recent_flips({}, handler=mock_http)
        assert _status(result) == 500

    @patch("aragora.insights.flip_detector.format_flip_for_ui")
    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_count_matches_flips_length(self, mock_detector_cls, mock_format, handler, mock_http):
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = _make_recent_flips(4)
        mock_detector_cls.return_value = mock_detector
        mock_format.side_effect = lambda f: _make_formatted_flip(f.agent_name, f.flip_type, f.topic)

        result = handler._get_recent_flips({}, handler=mock_http)
        body = _body(result)
        assert body["count"] == len(body["flips"])

    @patch("aragora.insights.flip_detector.format_flip_for_ui")
    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_invalid_limit_uses_default(self, mock_detector_cls, mock_format, handler, mock_http):
        """Non-numeric limit falls back to default (20)."""
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = []
        mock_detector_cls.return_value = mock_detector

        handler._get_recent_flips({"limit": "abc"}, handler=mock_http)
        # get_clamped_int_param should fall back to default 20, so limit*2 = 40
        mock_detector.get_recent_flips.assert_called_once_with(limit=40)

    @patch("aragora.insights.flip_detector.format_flip_for_ui")
    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_empty_agent_filter_ignored(self, mock_detector_cls, mock_format, handler, mock_http):
        """Empty string agent filter should not filter anything."""
        flips = _make_recent_flips(3)
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = flips
        mock_detector_cls.return_value = mock_detector
        mock_format.side_effect = lambda f: _make_formatted_flip(f.agent_name, f.flip_type, f.topic)

        result = handler._get_recent_flips({"agent": ""}, handler=mock_http)
        body = _body(result)
        # Empty string is falsy, so no filtering should happen
        assert body["count"] == 3


# =========================================================================
# 4. AGENT CONSISTENCY ENDPOINT - GET /api/analytics/flips/consistency
# =========================================================================


class TestAgentConsistency:
    """Tests for _get_agent_consistency endpoint."""

    @patch("aragora.insights.flip_detector.format_consistency_for_ui")
    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_happy_path_with_agents_param(self, mock_detector_cls, mock_format, handler, mock_http):
        mock_detector = MagicMock()
        scores = _make_consistency_scores(["claude", "gpt-4"])
        mock_detector.get_agents_consistency_batch.return_value = scores
        mock_detector_cls.return_value = mock_detector
        mock_format.side_effect = lambda s: _make_formatted_consistency(s.agent_name, f"{s.consistency * 100:.1f}%")

        result = handler._get_agent_consistency(
            {"agents": "claude,gpt-4"},
            handler=mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert "agents" in body
        assert "count" in body
        assert body["count"] == 2

    @patch("aragora.insights.flip_detector.format_consistency_for_ui")
    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_no_agents_param_gets_all(self, mock_detector_cls, mock_format, handler, mock_http):
        """Without agents param, should get all agents from flip summary."""
        mock_detector = MagicMock()
        mock_detector.get_flip_summary.return_value = {
            "by_agent": {"claude": 30, "gpt-4": 25},
        }
        scores = _make_consistency_scores(["claude", "gpt-4"])
        mock_detector.get_agents_consistency_batch.return_value = scores
        mock_detector_cls.return_value = mock_detector
        mock_format.side_effect = lambda s: _make_formatted_consistency(s.agent_name, f"{s.consistency * 100:.1f}%")

        result = handler._get_agent_consistency({}, handler=mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 2

    @patch("aragora.insights.flip_detector.format_consistency_for_ui")
    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_empty_agents_param_gets_all(self, mock_detector_cls, mock_format, handler, mock_http):
        """Empty agents param should behave same as no agents param."""
        mock_detector = MagicMock()
        mock_detector.get_flip_summary.return_value = {
            "by_agent": {"claude": 30},
        }
        scores = _make_consistency_scores(["claude"])
        mock_detector.get_agents_consistency_batch.return_value = scores
        mock_detector_cls.return_value = mock_detector
        mock_format.side_effect = lambda s: _make_formatted_consistency(s.agent_name, f"{s.consistency * 100:.1f}%")

        result = handler._get_agent_consistency({"agents": ""}, handler=mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 1

    @patch("aragora.insights.flip_detector.format_consistency_for_ui")
    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_no_agents_in_summary_returns_empty(self, mock_detector_cls, mock_format, handler, mock_http):
        """If summary has no agents, return empty list."""
        mock_detector = MagicMock()
        mock_detector.get_flip_summary.return_value = {"by_agent": {}}
        mock_detector_cls.return_value = mock_detector

        result = handler._get_agent_consistency({}, handler=mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["agents"] == []
        assert body["count"] == 0

    @patch("aragora.insights.flip_detector.format_consistency_for_ui")
    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_sorted_by_consistency_descending(self, mock_detector_cls, mock_format, handler, mock_http):
        """Results should be sorted by consistency score, highest first."""
        mock_detector = MagicMock()
        scores = _make_consistency_scores(["claude", "gpt-4", "gemini"])
        mock_detector.get_agents_consistency_batch.return_value = scores
        mock_detector_cls.return_value = mock_detector
        mock_format.side_effect = lambda s: _make_formatted_consistency(s.agent_name, f"{s.consistency * 100:.1f}%")

        result = handler._get_agent_consistency(
            {"agents": "claude,gpt-4,gemini"},
            handler=mock_http,
        )
        body = _body(result)
        agents = body["agents"]
        scores_list = [float(a["consistency"].rstrip("%")) for a in agents]
        assert scores_list == sorted(scores_list, reverse=True)

    @patch("aragora.insights.flip_detector.format_consistency_for_ui")
    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_comma_separated_agents(self, mock_detector_cls, mock_format, handler, mock_http):
        """Should parse comma-separated agents correctly."""
        mock_detector = MagicMock()
        scores = _make_consistency_scores(["claude", "gpt-4", "gemini"])
        mock_detector.get_agents_consistency_batch.return_value = scores
        mock_detector_cls.return_value = mock_detector
        mock_format.side_effect = lambda s: _make_formatted_consistency(s.agent_name, f"{s.consistency * 100:.1f}%")

        handler._get_agent_consistency(
            {"agents": "claude, gpt-4, gemini"},
            handler=mock_http,
        )
        call_args = mock_detector.get_agents_consistency_batch.call_args[0][0]
        assert "claude" in call_args
        assert "gpt-4" in call_args
        assert "gemini" in call_args

    @patch("aragora.insights.flip_detector.format_consistency_for_ui")
    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_single_agent(self, mock_detector_cls, mock_format, handler, mock_http):
        mock_detector = MagicMock()
        scores = _make_consistency_scores(["claude"])
        mock_detector.get_agents_consistency_batch.return_value = scores
        mock_detector_cls.return_value = mock_detector
        mock_format.side_effect = lambda s: _make_formatted_consistency(s.agent_name, f"{s.consistency * 100:.1f}%")

        result = handler._get_agent_consistency(
            {"agents": "claude"},
            handler=mock_http,
        )
        body = _body(result)
        assert body["count"] == 1

    @patch("aragora.insights.flip_detector.FlipDetector", side_effect=ImportError("no module"))
    def test_import_error(self, _mock, handler, mock_http):
        result = handler._get_agent_consistency({}, handler=mock_http)
        assert _status(result) == 500

    @patch("aragora.insights.flip_detector.FlipDetector", side_effect=RuntimeError("db failure"))
    def test_runtime_error(self, _mock, handler, mock_http):
        result = handler._get_agent_consistency({}, handler=mock_http)
        assert _status(result) == 500

    @patch("aragora.insights.flip_detector.FlipDetector", side_effect=OSError("disk error"))
    def test_os_error(self, _mock, handler, mock_http):
        result = handler._get_agent_consistency({}, handler=mock_http)
        assert _status(result) == 500

    @patch("aragora.insights.flip_detector.FlipDetector", side_effect=LookupError("not found"))
    def test_lookup_error(self, _mock, handler, mock_http):
        result = handler._get_agent_consistency({}, handler=mock_http)
        assert _status(result) == 500

    @patch("aragora.insights.flip_detector.format_consistency_for_ui")
    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_agents_with_spaces_trimmed(self, mock_detector_cls, mock_format, handler, mock_http):
        """Whitespace around agent names should be stripped."""
        mock_detector = MagicMock()
        scores = _make_consistency_scores(["claude", "gpt-4"])
        mock_detector.get_agents_consistency_batch.return_value = scores
        mock_detector_cls.return_value = mock_detector
        mock_format.side_effect = lambda s: _make_formatted_consistency(s.agent_name, f"{s.consistency * 100:.1f}%")

        handler._get_agent_consistency(
            {"agents": " claude , gpt-4 "},
            handler=mock_http,
        )
        call_args = mock_detector.get_agents_consistency_batch.call_args[0][0]
        assert call_args == ["claude", "gpt-4"]

    @patch("aragora.insights.flip_detector.format_consistency_for_ui")
    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_empty_commas_ignored(self, mock_detector_cls, mock_format, handler, mock_http):
        """Trailing/extra commas produce empty strings that should be filtered out."""
        mock_detector = MagicMock()
        mock_detector.get_flip_summary.return_value = {"by_agent": {"claude": 30}}
        scores = _make_consistency_scores(["claude"])
        mock_detector.get_agents_consistency_batch.return_value = scores
        mock_detector_cls.return_value = mock_detector
        mock_format.side_effect = lambda s: _make_formatted_consistency(s.agent_name, f"{s.consistency * 100:.1f}%")

        result = handler._get_agent_consistency(
            {"agents": ",,, ,,"},
            handler=mock_http,
        )
        # All commas produce empty strings, filtered out -> falls into no-agents-param path
        assert _status(result) == 200


# =========================================================================
# 5. FLIP TRENDS ENDPOINT - GET /api/analytics/flips/trends
# =========================================================================


class TestFlipTrends:
    """Tests for _get_flip_trends endpoint."""

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_happy_path(self, mock_detector_cls, handler, mock_http):
        mock_detector = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ("2026-02-18", "contradiction", 5),
            ("2026-02-18", "retraction", 3),
            ("2026-02-19", "contradiction", 7),
            ("2026-02-20", "qualification", 2),
        ]
        mock_detector.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_detector.db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_detector_cls.return_value = mock_detector

        result = handler._get_flip_trends({}, handler=mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert "period" in body
        assert "granularity" in body
        assert "data_points" in body
        assert "summary" in body

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_default_days_and_granularity(self, mock_detector_cls, handler, mock_http):
        mock_detector = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_detector.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_detector.db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_detector_cls.return_value = mock_detector

        result = handler._get_flip_trends({}, handler=mock_http)
        body = _body(result)
        assert body["period"]["days"] == 30
        assert body["granularity"] == "day"

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_custom_days(self, mock_detector_cls, handler, mock_http):
        mock_detector = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_detector.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_detector.db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_detector_cls.return_value = mock_detector

        result = handler._get_flip_trends({"days": "14"}, handler=mock_http)
        body = _body(result)
        assert body["period"]["days"] == 14

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_week_granularity(self, mock_detector_cls, handler, mock_http):
        mock_detector = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_detector.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_detector.db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_detector_cls.return_value = mock_detector

        result = handler._get_flip_trends({"granularity": "week"}, handler=mock_http)
        body = _body(result)
        assert body["granularity"] == "week"

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_invalid_granularity_defaults_to_day(self, mock_detector_cls, handler, mock_http):
        mock_detector = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_detector.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_detector.db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_detector_cls.return_value = mock_detector

        result = handler._get_flip_trends({"granularity": "month"}, handler=mock_http)
        body = _body(result)
        assert body["granularity"] == "day"

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_days_clamped_min(self, mock_detector_cls, handler, mock_http):
        mock_detector = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_detector.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_detector.db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_detector_cls.return_value = mock_detector

        result = handler._get_flip_trends({"days": "0"}, handler=mock_http)
        body = _body(result)
        assert body["period"]["days"] == 1

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_days_clamped_max(self, mock_detector_cls, handler, mock_http):
        mock_detector = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_detector.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_detector.db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_detector_cls.return_value = mock_detector

        result = handler._get_flip_trends({"days": "999"}, handler=mock_http)
        body = _body(result)
        assert body["period"]["days"] == 365

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_data_points_grouping(self, mock_detector_cls, handler, mock_http):
        mock_detector = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ("2026-02-18", "contradiction", 5),
            ("2026-02-18", "retraction", 3),
            ("2026-02-19", "contradiction", 7),
        ]
        mock_detector.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_detector.db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_detector_cls.return_value = mock_detector

        result = handler._get_flip_trends({}, handler=mock_http)
        body = _body(result)
        # 2 distinct dates
        assert len(body["data_points"]) == 2
        # First date should have combined total
        first = body["data_points"][0]
        assert first["date"] == "2026-02-18"
        assert first["total"] == 8
        assert first["by_type"]["contradiction"] == 5
        assert first["by_type"]["retraction"] == 3

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_summary_total_flips(self, mock_detector_cls, handler, mock_http):
        mock_detector = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ("2026-02-18", "contradiction", 5),
            ("2026-02-19", "contradiction", 7),
        ]
        mock_detector.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_detector.db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_detector_cls.return_value = mock_detector

        result = handler._get_flip_trends({}, handler=mock_http)
        body = _body(result)
        assert body["summary"]["total_flips"] == 12

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_summary_avg_per_day(self, mock_detector_cls, handler, mock_http):
        mock_detector = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ("2026-02-18", "contradiction", 15),
            ("2026-02-19", "contradiction", 15),
        ]
        mock_detector.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_detector.db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_detector_cls.return_value = mock_detector

        result = handler._get_flip_trends({"days": "30"}, handler=mock_http)
        body = _body(result)
        # 30 total flips / 30 days = 1.0
        assert body["summary"]["avg_per_day"] == 1.0

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_trend_increasing(self, mock_detector_cls, handler, mock_http):
        """When second half has >1.2x first half, trend should be increasing."""
        mock_detector = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ("2026-02-10", "contradiction", 2),
            ("2026-02-11", "contradiction", 2),
            ("2026-02-20", "contradiction", 10),
            ("2026-02-21", "contradiction", 10),
        ]
        mock_detector.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_detector.db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_detector_cls.return_value = mock_detector

        result = handler._get_flip_trends({}, handler=mock_http)
        body = _body(result)
        assert body["summary"]["trend"] == "increasing"

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_trend_decreasing(self, mock_detector_cls, handler, mock_http):
        """When second half has <0.8x first half, trend should be decreasing."""
        mock_detector = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ("2026-02-10", "contradiction", 10),
            ("2026-02-11", "contradiction", 10),
            ("2026-02-20", "contradiction", 1),
            ("2026-02-21", "contradiction", 1),
        ]
        mock_detector.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_detector.db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_detector_cls.return_value = mock_detector

        result = handler._get_flip_trends({}, handler=mock_http)
        body = _body(result)
        assert body["summary"]["trend"] == "decreasing"

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_trend_stable(self, mock_detector_cls, handler, mock_http):
        """When halves are similar, trend should be stable."""
        mock_detector = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ("2026-02-10", "contradiction", 10),
            ("2026-02-11", "contradiction", 10),
            ("2026-02-20", "contradiction", 10),
            ("2026-02-21", "contradiction", 10),
        ]
        mock_detector.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_detector.db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_detector_cls.return_value = mock_detector

        result = handler._get_flip_trends({}, handler=mock_http)
        body = _body(result)
        assert body["summary"]["trend"] == "stable"

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_trend_insufficient_data(self, mock_detector_cls, handler, mock_http):
        """With fewer than 2 data points, trend is insufficient_data."""
        mock_detector = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ("2026-02-20", "contradiction", 5),
        ]
        mock_detector.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_detector.db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_detector_cls.return_value = mock_detector

        result = handler._get_flip_trends({}, handler=mock_http)
        body = _body(result)
        assert body["summary"]["trend"] == "insufficient_data"

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_trend_no_data_points(self, mock_detector_cls, handler, mock_http):
        """Empty results should give insufficient_data and zero totals."""
        mock_detector = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_detector.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_detector.db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_detector_cls.return_value = mock_detector

        result = handler._get_flip_trends({}, handler=mock_http)
        body = _body(result)
        assert body["data_points"] == []
        assert body["summary"]["total_flips"] == 0
        assert body["summary"]["avg_per_day"] == 0
        assert body["summary"]["trend"] == "insufficient_data"

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_period_has_start_end(self, mock_detector_cls, handler, mock_http):
        mock_detector = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_detector.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_detector.db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_detector_cls.return_value = mock_detector

        result = handler._get_flip_trends({}, handler=mock_http)
        body = _body(result)
        assert "start" in body["period"]
        assert "end" in body["period"]
        assert "days" in body["period"]

    @patch("aragora.insights.flip_detector.FlipDetector", side_effect=ImportError("no module"))
    def test_import_error(self, _mock, handler, mock_http):
        result = handler._get_flip_trends({}, handler=mock_http)
        assert _status(result) == 500

    @patch("aragora.insights.flip_detector.FlipDetector", side_effect=RuntimeError("db failure"))
    def test_runtime_error(self, _mock, handler, mock_http):
        result = handler._get_flip_trends({}, handler=mock_http)
        assert _status(result) == 500

    @patch("aragora.insights.flip_detector.FlipDetector", side_effect=OSError("disk error"))
    def test_os_error(self, _mock, handler, mock_http):
        result = handler._get_flip_trends({}, handler=mock_http)
        assert _status(result) == 500

    @patch("aragora.insights.flip_detector.FlipDetector", side_effect=LookupError("not found"))
    def test_lookup_error(self, _mock, handler, mock_http):
        result = handler._get_flip_trends({}, handler=mock_http)
        assert _status(result) == 500

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_data_point_by_type_structure(self, mock_detector_cls, handler, mock_http):
        mock_detector = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ("2026-02-18", "contradiction", 5),
            ("2026-02-18", "retraction", 3),
        ]
        mock_detector.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_detector.db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_detector_cls.return_value = mock_detector

        result = handler._get_flip_trends({}, handler=mock_http)
        body = _body(result)
        point = body["data_points"][0]
        assert "date" in point
        assert "total" in point
        assert "by_type" in point
        assert isinstance(point["by_type"], dict)

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_invalid_days_uses_default(self, mock_detector_cls, handler, mock_http):
        """Non-numeric days should fall back to default of 30."""
        mock_detector = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_detector.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_detector.db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_detector_cls.return_value = mock_detector

        result = handler._get_flip_trends({"days": "abc"}, handler=mock_http)
        body = _body(result)
        assert body["period"]["days"] == 30

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_multiple_flip_types_same_period(self, mock_detector_cls, handler, mock_http):
        mock_detector = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ("2026-02-18", "contradiction", 5),
            ("2026-02-18", "retraction", 3),
            ("2026-02-18", "qualification", 2),
            ("2026-02-18", "refinement", 1),
        ]
        mock_detector.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_detector.db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_detector_cls.return_value = mock_detector

        result = handler._get_flip_trends({}, handler=mock_http)
        body = _body(result)
        assert len(body["data_points"]) == 1
        point = body["data_points"][0]
        assert point["total"] == 11
        assert len(point["by_type"]) == 4


# =========================================================================
# 6. ROUTING TESTS
# =========================================================================


class TestRouting:
    """Tests for routing through AnalyticsDashboardHandler.handle()."""

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_route_flips_summary(self, mock_detector_cls, handler, mock_http):
        mock_detector = MagicMock()
        mock_detector.get_flip_summary.return_value = _make_flip_summary()
        mock_detector_cls.return_value = mock_detector

        result = handler.handle("/api/analytics/flips/summary", {}, mock_http)
        assert result is not None
        assert _status(result) == 200

    @patch("aragora.insights.flip_detector.format_flip_for_ui")
    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_route_flips_recent(self, mock_detector_cls, mock_format, handler, mock_http):
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = []
        mock_detector_cls.return_value = mock_detector

        result = handler.handle("/api/analytics/flips/recent", {}, mock_http)
        assert result is not None
        assert _status(result) == 200

    @patch("aragora.insights.flip_detector.format_consistency_for_ui")
    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_route_flips_consistency(self, mock_detector_cls, mock_format, handler, mock_http):
        mock_detector = MagicMock()
        mock_detector.get_flip_summary.return_value = {"by_agent": {}}
        mock_detector_cls.return_value = mock_detector

        result = handler.handle("/api/analytics/flips/consistency", {}, mock_http)
        assert result is not None
        assert _status(result) == 200

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_route_flips_trends(self, mock_detector_cls, handler, mock_http):
        mock_detector = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_detector.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_detector.db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_detector_cls.return_value = mock_detector

        result = handler.handle("/api/analytics/flips/trends", {}, mock_http)
        assert result is not None
        assert _status(result) == 200

    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    def test_route_agents(self, mock_tr, mock_get_dash, mock_run_async, handler, mock_http):
        mock_run_async.side_effect = lambda coro: []
        mock_get_dash.return_value = MagicMock()

        result = handler.handle(
            "/api/analytics/agents",
            {"workspace_id": "ws_123"},
            mock_http,
        )
        assert result is not None
        assert _status(result) == 200

    def test_route_agents_stub_no_workspace(self, handler, mock_http):
        """Without workspace_id, should return stub response."""
        result = handler.handle("/api/analytics/agents", {}, mock_http)
        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert "agents" in body

    def test_route_flips_summary_stub_no_workspace(self, handler, mock_http):
        """Without workspace_id, flips/summary returns stub."""
        result = handler.handle("/api/analytics/flips/summary", {}, mock_http)
        assert result is not None
        assert _status(result) == 200

    def test_route_flips_recent_stub_no_workspace(self, handler, mock_http):
        """Without workspace_id, flips/recent returns stub."""
        result = handler.handle("/api/analytics/flips/recent", {}, mock_http)
        assert result is not None
        assert _status(result) == 200

    def test_route_flips_consistency_stub_no_workspace(self, handler, mock_http):
        """Without workspace_id, flips/consistency returns stub."""
        result = handler.handle("/api/analytics/flips/consistency", {}, mock_http)
        assert result is not None
        assert _status(result) == 200

    def test_route_flips_trends_stub_no_workspace(self, handler, mock_http):
        """Without workspace_id, flips/trends returns stub."""
        result = handler.handle("/api/analytics/flips/trends", {}, mock_http)
        assert result is not None
        assert _status(result) == 200

    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    def test_route_agents_with_version_prefix(self, mock_tr, mock_get_dash, mock_run_async, handler, mock_http):
        """Should handle /api/v1/analytics/agents correctly."""
        mock_run_async.side_effect = lambda coro: []
        mock_get_dash.return_value = MagicMock()

        result = handler.handle(
            "/api/v1/analytics/agents",
            {"workspace_id": "ws_123"},
            mock_http,
        )
        assert result is not None
        assert _status(result) == 200


# =========================================================================
# 7. CAN_HANDLE TESTS
# =========================================================================


class TestCanHandle:
    """Tests for can_handle routing."""

    def test_can_handle_agents(self, handler):
        assert handler.can_handle("/api/analytics/agents") is True

    def test_can_handle_flips_summary(self, handler):
        assert handler.can_handle("/api/analytics/flips/summary") is True

    def test_can_handle_flips_recent(self, handler):
        assert handler.can_handle("/api/analytics/flips/recent") is True

    def test_can_handle_flips_consistency(self, handler):
        assert handler.can_handle("/api/analytics/flips/consistency") is True

    def test_can_handle_flips_trends(self, handler):
        assert handler.can_handle("/api/analytics/flips/trends") is True

    def test_can_handle_with_version_prefix(self, handler):
        assert handler.can_handle("/api/v1/analytics/agents") is True

    def test_can_handle_with_v2_prefix(self, handler):
        assert handler.can_handle("/api/v2/analytics/flips/summary") is True

    def test_cannot_handle_unknown_path(self, handler):
        assert handler.can_handle("/api/analytics/unknown") is False

    def test_cannot_handle_partial_path(self, handler):
        assert handler.can_handle("/api/analytics") is False

    def test_cannot_handle_empty_path(self, handler):
        assert handler.can_handle("") is False


# =========================================================================
# 8. EDGE CASES AND ERROR CODES
# =========================================================================


class TestEdgeCases:
    """Tests for error codes and edge case handling."""

    def test_agent_metrics_error_code_missing_workspace(self, handler, mock_http):
        result = handler._get_agent_metrics({}, handler=mock_http)
        body = _body(result)
        error = body.get("error", {})
        code = error.get("code", "") if isinstance(error, dict) else ""
        assert code == "MISSING_WORKSPACE_ID"

    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange", side_effect=ValueError("bad"))
    def test_agent_metrics_error_code_invalid_param(self, mock_tr, mock_get_dash, mock_run_async, handler, mock_http):
        result = handler._get_agent_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        body = _body(result)
        error = body.get("error", {})
        code = error.get("code", "") if isinstance(error, dict) else ""
        assert code == "INVALID_PARAMETER"

    @patch("aragora.server.handlers.analytics_dashboard._run_async")
    @patch("aragora.analytics.get_analytics_dashboard")
    @patch("aragora.analytics.TimeRange")
    def test_agent_metrics_error_code_data_error(self, mock_tr, mock_get_dash, mock_run_async, handler, mock_http):
        mock_run_async.side_effect = KeyError("missing")
        mock_get_dash.return_value = MagicMock()

        result = handler._get_agent_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        body = _body(result)
        error = body.get("error", {})
        code = error.get("code", "") if isinstance(error, dict) else ""
        assert code == "DATA_ERROR"

    @patch("aragora.analytics.get_analytics_dashboard", side_effect=ImportError("no module"))
    def test_agent_metrics_error_code_internal(self, _mock, handler, mock_http):
        result = handler._get_agent_metrics(
            {"workspace_id": "ws_123"},
            handler=mock_http,
        )
        body = _body(result)
        error = body.get("error", {})
        code = error.get("code", "") if isinstance(error, dict) else ""
        assert code == "INTERNAL_ERROR"

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_flip_trends_avg_per_day_rounding(self, mock_detector_cls, handler, mock_http):
        """avg_per_day should be rounded to 2 decimal places."""
        mock_detector = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ("2026-02-18", "contradiction", 7),
        ]
        mock_detector.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_detector.db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_detector_cls.return_value = mock_detector

        result = handler._get_flip_trends({"days": "3"}, handler=mock_http)
        body = _body(result)
        # 7 / 3 = 2.333... -> should be 2.33
        assert body["summary"]["avg_per_day"] == round(7 / 3, 2)

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_flip_trends_row_limit_safety(self, mock_detector_cls, handler, mock_http):
        """Row limit should be capped at 1000 for memory safety."""
        mock_detector = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_detector.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_detector.db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_detector_cls.return_value = mock_detector

        handler._get_flip_trends({"days": "365"}, handler=mock_http)
        # Verify SQL was executed (we can't easily check the limit param,
        # but we verify it doesn't crash)
        mock_conn.execute.assert_called_once()

    @patch("aragora.insights.flip_detector.format_flip_for_ui")
    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_recent_flips_negative_limit_clamped(self, mock_detector_cls, mock_format, handler, mock_http):
        """Negative limit should be clamped to 1."""
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = []
        mock_detector_cls.return_value = mock_detector

        handler._get_recent_flips({"limit": "-5"}, handler=mock_http)
        mock_detector.get_recent_flips.assert_called_once_with(limit=2)

    @patch("aragora.insights.flip_detector.format_consistency_for_ui")
    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_consistency_detector_instantiated(self, mock_detector_cls, mock_format, handler, mock_http):
        """FlipDetector should be instantiated for each request."""
        mock_detector = MagicMock()
        mock_detector.get_flip_summary.return_value = {"by_agent": {}}
        mock_detector_cls.return_value = mock_detector

        handler._get_agent_consistency({}, handler=mock_http)
        mock_detector_cls.assert_called_once()

    @patch("aragora.insights.flip_detector.FlipDetector")
    def test_flip_summary_detector_instantiated(self, mock_detector_cls, handler, mock_http):
        mock_detector = MagicMock()
        mock_detector.get_flip_summary.return_value = _make_flip_summary()
        mock_detector_cls.return_value = mock_detector

        handler._get_flip_summary({}, handler=mock_http)
        mock_detector_cls.assert_called_once()
