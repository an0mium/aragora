"""Tests for agent calibration endpoint handlers."""

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


@dataclass
class MockCalibrationBucket:
    """Mock calibration bucket."""

    range_start: float
    range_end: float
    total_predictions: int
    correct_predictions: int
    accuracy: float
    brier_score: float


@dataclass
class MockCalibrationSummary:
    """Mock calibration summary."""

    agent: str
    total_predictions: int
    total_correct: int
    accuracy: float
    brier_score: float
    ece: float
    is_overconfident: bool
    is_underconfident: bool


@pytest.fixture(autouse=True)
def clear_module_state():
    """Clear any module-level state between tests."""
    from aragora.server.handlers.agents.calibration import _calibration_limiter

    _calibration_limiter._requests.clear()
    yield


class TestCalibrationHandlerRoutes:
    """Tests for CalibrationHandler route configuration."""

    def test_routes_defined(self):
        """Test CalibrationHandler has expected routes."""
        from aragora.server.handlers.agents.calibration import CalibrationHandler

        routes = CalibrationHandler.ROUTES

        assert "/api/agent/*/calibration-curve" in routes
        assert "/api/agent/*/calibration-summary" in routes
        assert "/api/calibration/leaderboard" in routes

    def test_can_handle_calibration_curve(self):
        """Test can_handle returns True for calibration curve."""
        from aragora.server.handlers.agents.calibration import CalibrationHandler

        handler = CalibrationHandler()

        assert handler.can_handle("/api/agent/claude/calibration-curve") is True
        assert handler.can_handle("/api/agent/gemini/calibration-summary") is True
        assert handler.can_handle("/api/calibration/leaderboard") is True

    def test_can_handle_non_calibration(self):
        """Test can_handle returns False for non-calibration routes."""
        from aragora.server.handlers.agents.calibration import CalibrationHandler

        handler = CalibrationHandler()

        assert handler.can_handle("/api/agent/claude/profile") is False
        assert handler.can_handle("/api/debates") is False


class TestCalibrationHandlerAuth:
    """Tests for CalibrationHandler authentication."""

    @pytest.mark.asyncio
    async def test_requires_authentication(self):
        """Test calibration endpoints require authentication."""
        from aragora.server.handlers.agents.calibration import CalibrationHandler
        from aragora.server.handlers.secure import UnauthorizedError

        handler = CalibrationHandler()
        mock_http_handler = MagicMock()

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.side_effect = UnauthorizedError("Not authenticated")
            result = await handler.handle("/api/calibration/leaderboard", {}, mock_http_handler)

        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_checks_permission(self):
        """Test calibration endpoints check permission."""
        from aragora.server.handlers.agents.calibration import CalibrationHandler
        from aragora.server.handlers.secure import ForbiddenError

        handler = CalibrationHandler()
        mock_http_handler = MagicMock()
        mock_auth_context = MagicMock()

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.side_effect = ForbiddenError("Permission denied")
            result = await handler.handle("/api/calibration/leaderboard", {}, mock_http_handler)

        assert result.status_code == 403


class TestCalibrationHandlerRateLimit:
    """Tests for CalibrationHandler rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self):
        """Test rate limit returns 429."""
        from aragora.server.handlers.agents.calibration import CalibrationHandler

        handler = CalibrationHandler()
        mock_http_handler = MagicMock()

        with (
            patch(
                "aragora.server.handlers.agents.calibration.get_client_ip", return_value="127.0.0.1"
            ),
            patch(
                "aragora.server.handlers.agents.calibration._calibration_limiter.is_allowed",
                return_value=False,
            ),
        ):
            result = await handler.handle("/api/calibration/leaderboard", {}, mock_http_handler)

        assert result.status_code == 429


class TestGetCalibrationCurve:
    """Tests for _get_calibration_curve method."""

    def test_calibration_curve_success(self):
        """Test calibration curve returns data."""
        from aragora.server.handlers.agents.calibration import CalibrationHandler

        handler = CalibrationHandler()

        mock_buckets = [
            MockCalibrationBucket(0.0, 0.1, 10, 1, 0.1, 0.08),
            MockCalibrationBucket(0.1, 0.2, 15, 3, 0.2, 0.09),
        ]

        mock_tracker = MagicMock()
        mock_tracker.get_calibration_curve.return_value = mock_buckets

        with (
            patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ),
            patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True),
        ):
            result = handler._get_calibration_curve("claude", 10, None)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["agent"] == "claude"
        assert len(body["buckets"]) == 2
        assert body["buckets"][0]["total_predictions"] == 10

    def test_calibration_curve_not_available(self):
        """Test calibration curve returns 503 when tracker not available."""
        from aragora.server.handlers.agents.calibration import CalibrationHandler

        handler = CalibrationHandler()

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", False):
            result = handler._get_calibration_curve("claude", 10, None)

        assert result.status_code == 503

    def test_calibration_curve_with_domain(self):
        """Test calibration curve with domain filter."""
        from aragora.server.handlers.agents.calibration import CalibrationHandler

        handler = CalibrationHandler()

        mock_tracker = MagicMock()
        mock_tracker.get_calibration_curve.return_value = []

        with (
            patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ),
            patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True),
        ):
            result = handler._get_calibration_curve("claude", 10, "math")

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["domain"] == "math"


class TestGetCalibrationSummary:
    """Tests for _get_calibration_summary method."""

    def test_calibration_summary_success(self):
        """Test calibration summary returns data."""
        from aragora.server.handlers.agents.calibration import CalibrationHandler

        handler = CalibrationHandler()

        mock_summary = MockCalibrationSummary(
            agent="claude",
            total_predictions=100,
            total_correct=85,
            accuracy=0.85,
            brier_score=0.12,
            ece=0.08,
            is_overconfident=False,
            is_underconfident=True,
        )

        mock_tracker = MagicMock()
        mock_tracker.get_calibration_summary.return_value = mock_summary

        with (
            patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ),
            patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True),
        ):
            result = handler._get_calibration_summary("claude", None)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["agent"] == "claude"
        assert body["total_predictions"] == 100
        assert body["accuracy"] == 0.85
        assert body["brier_score"] == 0.12

    def test_calibration_summary_not_available(self):
        """Test calibration summary returns 503 when tracker not available."""
        from aragora.server.handlers.agents.calibration import CalibrationHandler

        handler = CalibrationHandler()

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", False):
            result = handler._get_calibration_summary("claude", None)

        assert result.status_code == 503


class TestGetCalibrationLeaderboard:
    """Tests for _get_calibration_leaderboard method."""

    def test_leaderboard_success(self):
        """Test leaderboard returns ranked agents."""
        from aragora.server.handlers.agents.calibration import CalibrationHandler

        handler = CalibrationHandler()

        mock_rating = MagicMock()
        mock_rating.calibration_total = 50
        mock_rating.calibration_correct = 45
        mock_rating.calibration_score = 0.9
        mock_rating.calibration_brier_score = 0.08
        mock_rating.calibration_accuracy = 0.9
        mock_rating.elo = 1600

        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [{"agent": "claude"}]
        mock_elo.get_rating.return_value = mock_rating

        with (
            patch("aragora.server.handlers.agents.calibration.EloSystem", return_value=mock_elo),
            patch("aragora.server.handlers.agents.calibration.ELO_AVAILABLE", True),
        ):
            result = handler._get_calibration_leaderboard(10, "brier", 5)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert "agents" in body
        assert body["metric"] == "brier"

    def test_leaderboard_elo_not_available(self):
        """Test leaderboard returns 503 when ELO not available."""
        from aragora.server.handlers.agents.calibration import CalibrationHandler

        handler = CalibrationHandler()

        with patch("aragora.server.handlers.agents.calibration.ELO_AVAILABLE", False):
            result = handler._get_calibration_leaderboard(10, "brier", 5)

        assert result.status_code == 503

    def test_leaderboard_filters_by_min_predictions(self):
        """Test leaderboard filters agents by min predictions."""
        from aragora.server.handlers.agents.calibration import CalibrationHandler

        handler = CalibrationHandler()

        mock_rating_low = MagicMock()
        mock_rating_low.calibration_total = 3  # Below min
        mock_rating_low.calibration_correct = 3

        mock_rating_high = MagicMock()
        mock_rating_high.calibration_total = 50  # Above min
        mock_rating_high.calibration_correct = 45
        mock_rating_high.calibration_score = 0.9
        mock_rating_high.calibration_brier_score = 0.08
        mock_rating_high.calibration_accuracy = 0.9
        mock_rating_high.elo = 1600

        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [
            {"agent": "low_agent"},
            {"agent": "high_agent"},
        ]

        def get_rating(agent):
            if agent == "low_agent":
                return mock_rating_low
            return mock_rating_high

        mock_elo.get_rating.side_effect = get_rating

        with (
            patch("aragora.server.handlers.agents.calibration.EloSystem", return_value=mock_elo),
            patch("aragora.server.handlers.agents.calibration.ELO_AVAILABLE", True),
        ):
            result = handler._get_calibration_leaderboard(10, "brier", 5)

        body = json.loads(result.body.decode("utf-8"))
        # Only high_agent should be included
        assert len(body["agents"]) == 1


class TestGetCalibrationVisualization:
    """Tests for _get_calibration_visualization method."""

    def test_visualization_success(self):
        """Test visualization returns data."""
        from aragora.server.handlers.agents.calibration import CalibrationHandler

        handler = CalibrationHandler()

        mock_summary = MockCalibrationSummary(
            agent="claude",
            total_predictions=100,
            total_correct=85,
            accuracy=0.85,
            brier_score=0.12,
            ece=0.08,
            is_overconfident=False,
            is_underconfident=True,
        )

        mock_tracker = MagicMock()
        mock_tracker.get_all_agents.return_value = ["claude"]
        mock_tracker.get_calibration_summary.return_value = mock_summary
        mock_tracker.get_calibration_curve.return_value = [
            MockCalibrationBucket(0.0, 0.1, 10, 1, 0.1, 0.08),
        ]
        mock_tracker.get_domain_breakdown.return_value = {}

        with (
            patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ),
            patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True),
        ):
            result = handler._get_calibration_visualization(5)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert "calibration_curves" in body
        assert "scatter_data" in body
        assert "confidence_histogram" in body
        assert "summary" in body

    def test_visualization_not_available(self):
        """Test visualization returns 503 when tracker not available."""
        from aragora.server.handlers.agents.calibration import CalibrationHandler

        handler = CalibrationHandler()

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", False):
            result = handler._get_calibration_visualization(5)

        assert result.status_code == 503

    def test_visualization_empty_agents(self):
        """Test visualization handles no agents."""
        from aragora.server.handlers.agents.calibration import CalibrationHandler

        handler = CalibrationHandler()

        mock_tracker = MagicMock()
        mock_tracker.get_all_agents.return_value = []

        with (
            patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ),
            patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True),
        ):
            result = handler._get_calibration_visualization(5)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["summary"]["total_agents"] == 0
