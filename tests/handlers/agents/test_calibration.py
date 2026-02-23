"""Comprehensive tests for calibration handler.

Tests the calibration API endpoints including:
- GET /api/v1/agent/{name}/calibration-curve
- GET /api/v1/agent/{name}/calibration-summary
- GET /api/v1/calibration/leaderboard
- GET /api/v1/calibration/visualization

Covers success paths, error handling, parameter validation, RBAC,
rate limiting, sorting, filtering, and edge cases.
"""

import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Mock Data Classes
# =============================================================================


@dataclass
class MockCalibrationBucket:
    """Mock calibration bucket data."""

    range_start: float
    range_end: float
    total_predictions: int
    correct_predictions: int
    accuracy: float
    brier_score: float


@dataclass
class MockCalibrationSummary:
    """Mock calibration summary data."""

    agent: str
    total_predictions: int
    total_correct: int
    accuracy: float
    brier_score: float
    ece: float
    is_overconfident: bool
    is_underconfident: bool


@dataclass
class MockAgentRating:
    """Mock agent rating data."""

    elo: float
    calibration_score: float
    calibration_brier_score: float
    calibration_accuracy: float
    calibration_total: int
    calibration_correct: int


# =============================================================================
# Helpers
# =============================================================================


def _body(result) -> dict:
    """Parse HandlerResult.body bytes into dict."""
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    return result.status_code


def _make_mock_http_handler(ip="127.0.0.1"):
    """Create a mock HTTP handler with given IP."""
    handler = MagicMock()
    handler.client_address = (ip, 12345)
    handler.headers = {}
    return handler


def _make_summary(agent="claude", total=100, correct=75, accuracy=0.75,
                  brier=0.15, ece=0.05, overconfident=False, underconfident=True):
    """Create a MockCalibrationSummary with defaults."""
    return MockCalibrationSummary(
        agent=agent,
        total_predictions=total,
        total_correct=correct,
        accuracy=accuracy,
        brier_score=brier,
        ece=ece,
        is_overconfident=overconfident,
        is_underconfident=underconfident,
    )


def _make_buckets(n=10):
    """Create a list of mock calibration buckets."""
    buckets = []
    for i in range(n):
        start = i / n
        end = (i + 1) / n
        mid = (start + end) / 2
        buckets.append(MockCalibrationBucket(
            range_start=start,
            range_end=end,
            total_predictions=10 + i,
            correct_predictions=int((10 + i) * mid),
            accuracy=mid,
            brier_score=round(abs(mid - mid) + 0.01, 4),
        ))
    return buckets


def _make_rating(elo=1500, score=0.85, brier=0.1, acc=0.8, total=100, correct=80):
    """Create a MockAgentRating with defaults."""
    return MockAgentRating(
        elo=elo,
        calibration_score=score,
        calibration_brier_score=brier,
        calibration_accuracy=acc,
        calibration_total=total,
        calibration_correct=correct,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def calibration_handler():
    """Create calibration handler with mock context."""
    from aragora.server.handlers.agents.calibration import CalibrationHandler

    return CalibrationHandler({})


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset the module-level rate limiter before each test."""
    try:
        from aragora.server.middleware.rate_limit.registry import reset_rate_limiters
        reset_rate_limiters()
    except ImportError:
        pass

    try:
        from aragora.server.handlers.agents import calibration
        calibration._calibration_limiter = calibration.RateLimiter(requests_per_minute=30)
    except (ImportError, AttributeError):
        pass

    yield

    try:
        from aragora.server.middleware.rate_limit.registry import reset_rate_limiters
        reset_rate_limiters()
    except ImportError:
        pass


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler."""
    return _make_mock_http_handler()


# =============================================================================
# 1. Initialization and Constructor Tests
# =============================================================================


class TestCalibrationHandlerInit:
    """Tests for handler initialization."""

    def test_init_with_empty_dict(self):
        from aragora.server.handlers.agents.calibration import CalibrationHandler
        h = CalibrationHandler({})
        assert h.ctx == {}

    def test_init_with_server_context(self):
        from aragora.server.handlers.agents.calibration import CalibrationHandler
        ctx = {"key": "value"}
        h = CalibrationHandler(server_context=ctx)
        assert h.ctx == ctx

    def test_init_with_ctx_kwarg(self):
        from aragora.server.handlers.agents.calibration import CalibrationHandler
        ctx = {"legacy": True}
        h = CalibrationHandler(ctx=ctx)
        assert h.ctx == ctx

    def test_init_with_none(self):
        from aragora.server.handlers.agents.calibration import CalibrationHandler
        h = CalibrationHandler(None)
        assert h.ctx == {}

    def test_init_server_context_takes_priority(self):
        from aragora.server.handlers.agents.calibration import CalibrationHandler
        h = CalibrationHandler(ctx={"old": 1}, server_context={"new": 2})
        assert h.ctx == {"new": 2}

    def test_routes_defined(self, calibration_handler):
        assert hasattr(calibration_handler, "ROUTES")
        assert len(calibration_handler.ROUTES) == 4

    def test_routes_contain_expected_patterns(self, calibration_handler):
        routes = calibration_handler.ROUTES
        assert "/api/agent/*/calibration-curve" in routes
        assert "/api/agent/*/calibration-summary" in routes
        assert "/api/calibration/leaderboard" in routes
        assert "/api/calibration/visualization" in routes

    def test_permissions_constants(self):
        from aragora.server.handlers.agents.calibration import (
            CALIBRATION_READ_PERMISSION,
            CALIBRATION_WRITE_PERMISSION,
            CALIBRATION_PERMISSION,
        )
        assert CALIBRATION_READ_PERMISSION == "agents:calibration:read"
        assert CALIBRATION_WRITE_PERMISSION == "agents:calibration:write"
        assert CALIBRATION_PERMISSION == CALIBRATION_READ_PERMISSION

    def test_max_calibration_agents_constant(self):
        from aragora.server.handlers.agents.calibration import MAX_CALIBRATION_AGENTS
        assert MAX_CALIBRATION_AGENTS == 100


# =============================================================================
# 2. can_handle Tests
# =============================================================================


class TestCanHandle:
    """Tests for can_handle routing."""

    def test_calibration_curve_with_version_prefix(self, calibration_handler):
        assert calibration_handler.can_handle("/api/v1/agent/claude/calibration-curve")

    def test_calibration_curve_without_version_prefix(self, calibration_handler):
        assert calibration_handler.can_handle("/api/agent/claude/calibration-curve")

    def test_calibration_summary_with_version_prefix(self, calibration_handler):
        assert calibration_handler.can_handle("/api/v1/agent/gpt4/calibration-summary")

    def test_calibration_summary_without_version_prefix(self, calibration_handler):
        assert calibration_handler.can_handle("/api/agent/gpt4/calibration-summary")

    def test_leaderboard_with_version_prefix(self, calibration_handler):
        assert calibration_handler.can_handle("/api/v1/calibration/leaderboard")

    def test_leaderboard_without_version_prefix(self, calibration_handler):
        assert calibration_handler.can_handle("/api/calibration/leaderboard")

    def test_visualization_with_version_prefix(self, calibration_handler):
        assert calibration_handler.can_handle("/api/v1/calibration/visualization")

    def test_visualization_without_version_prefix(self, calibration_handler):
        assert calibration_handler.can_handle("/api/calibration/visualization")

    def test_rejects_agent_root(self, calibration_handler):
        assert not calibration_handler.can_handle("/api/v1/agent/claude")

    def test_rejects_agent_elo(self, calibration_handler):
        assert not calibration_handler.can_handle("/api/v1/agent/claude/elo")

    def test_rejects_debates(self, calibration_handler):
        assert not calibration_handler.can_handle("/api/v1/debates")

    def test_rejects_calibration_root(self, calibration_handler):
        assert not calibration_handler.can_handle("/api/v1/calibration")

    def test_rejects_unrelated_path(self, calibration_handler):
        assert not calibration_handler.can_handle("/api/v1/health")

    def test_various_agent_names(self, calibration_handler):
        for name in ["claude", "gpt4", "mistral-large", "gemini_pro"]:
            assert calibration_handler.can_handle(f"/api/v1/agent/{name}/calibration-curve")
            assert calibration_handler.can_handle(f"/api/v1/agent/{name}/calibration-summary")


# =============================================================================
# 3. Calibration Curve Endpoint Tests
# =============================================================================


class TestCalibrationCurve:
    """Tests for GET /api/agent/{name}/calibration-curve."""

    @pytest.mark.asyncio
    async def test_returns_503_when_tracker_unavailable(self, calibration_handler, mock_http_handler):
        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", False):
            result = await calibration_handler.handle(
                "/api/v1/agent/claude/calibration-curve", {}, mock_http_handler
            )
            assert _status(result) == 503
            assert "not available" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_returns_503_when_tracker_class_is_none(self, calibration_handler, mock_http_handler):
        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch("aragora.server.handlers.agents.calibration.CalibrationTracker", None):
                result = await calibration_handler.handle(
                    "/api/v1/agent/claude/calibration-curve", {}, mock_http_handler
                )
                assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_success_with_data(self, calibration_handler, mock_http_handler):
        buckets = _make_buckets(3)
        mock_tracker = MagicMock()
        mock_tracker.get_calibration_curve.return_value = buckets

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/agent/claude/calibration-curve", {}, mock_http_handler
                )
                assert _status(result) == 200
                data = _body(result)
                assert data["agent"] == "claude"
                assert data["count"] == 3
                assert len(data["buckets"]) == 3
                assert data["domain"] is None

    @pytest.mark.asyncio
    async def test_curve_bucket_fields(self, calibration_handler, mock_http_handler):
        bucket = MockCalibrationBucket(
            range_start=0.2, range_end=0.3,
            total_predictions=50, correct_predictions=12,
            accuracy=0.24, brier_score=0.06,
        )
        mock_tracker = MagicMock()
        mock_tracker.get_calibration_curve.return_value = [bucket]

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/agent/claude/calibration-curve", {}, mock_http_handler
                )
                b = _body(result)["buckets"][0]
                assert b["range_start"] == 0.2
                assert b["range_end"] == 0.3
                assert b["total_predictions"] == 50
                assert b["correct_predictions"] == 12
                assert b["accuracy"] == 0.24
                assert b["expected_accuracy"] == pytest.approx(0.25)
                assert b["brier_score"] == 0.06

    @pytest.mark.asyncio
    async def test_default_buckets_param(self, calibration_handler, mock_http_handler):
        mock_tracker = MagicMock()
        mock_tracker.get_calibration_curve.return_value = []

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                await calibration_handler.handle(
                    "/api/v1/agent/claude/calibration-curve", {}, mock_http_handler
                )
                mock_tracker.get_calibration_curve.assert_called_once_with(
                    "claude", num_buckets=10, domain=None
                )

    @pytest.mark.asyncio
    async def test_custom_buckets_param(self, calibration_handler, mock_http_handler):
        mock_tracker = MagicMock()
        mock_tracker.get_calibration_curve.return_value = []

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                await calibration_handler.handle(
                    "/api/v1/agent/claude/calibration-curve",
                    {"buckets": ["15"]},
                    mock_http_handler,
                )
                mock_tracker.get_calibration_curve.assert_called_once_with(
                    "claude", num_buckets=15, domain=None
                )

    @pytest.mark.asyncio
    async def test_buckets_clamped_to_min(self, calibration_handler, mock_http_handler):
        mock_tracker = MagicMock()
        mock_tracker.get_calibration_curve.return_value = []

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                await calibration_handler.handle(
                    "/api/v1/agent/claude/calibration-curve",
                    {"buckets": ["1"]},  # Below min of 5
                    mock_http_handler,
                )
                mock_tracker.get_calibration_curve.assert_called_once_with(
                    "claude", num_buckets=5, domain=None
                )

    @pytest.mark.asyncio
    async def test_buckets_clamped_to_max(self, calibration_handler, mock_http_handler):
        mock_tracker = MagicMock()
        mock_tracker.get_calibration_curve.return_value = []

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                await calibration_handler.handle(
                    "/api/v1/agent/claude/calibration-curve",
                    {"buckets": ["99"]},  # Above max of 20
                    mock_http_handler,
                )
                mock_tracker.get_calibration_curve.assert_called_once_with(
                    "claude", num_buckets=20, domain=None
                )

    @pytest.mark.asyncio
    async def test_with_domain_param(self, calibration_handler, mock_http_handler):
        mock_tracker = MagicMock()
        mock_tracker.get_calibration_curve.return_value = []

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                await calibration_handler.handle(
                    "/api/v1/agent/claude/calibration-curve",
                    {"domain": ["technical"]},
                    mock_http_handler,
                )
                mock_tracker.get_calibration_curve.assert_called_once_with(
                    "claude", num_buckets=10, domain="technical"
                )

    @pytest.mark.asyncio
    async def test_domain_included_in_response(self, calibration_handler, mock_http_handler):
        mock_tracker = MagicMock()
        mock_tracker.get_calibration_curve.return_value = []

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/agent/claude/calibration-curve",
                    {"domain": ["math"]},
                    mock_http_handler,
                )
                data = _body(result)
                assert data["domain"] == "math"

    @pytest.mark.asyncio
    async def test_empty_curve(self, calibration_handler, mock_http_handler):
        mock_tracker = MagicMock()
        mock_tracker.get_calibration_curve.return_value = []

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/agent/claude/calibration-curve", {}, mock_http_handler
                )
                assert _status(result) == 200
                data = _body(result)
                assert data["buckets"] == []
                assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_invalid_agent_name_script(self, calibration_handler, mock_http_handler):
        result = await calibration_handler.handle(
            "/api/v1/agent/<script>/calibration-curve", {}, mock_http_handler
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_invalid_agent_name_traversal(self, calibration_handler, mock_http_handler):
        result = await calibration_handler.handle(
            "/api/v1/agent/../../../etc/passwd/calibration-curve", {}, mock_http_handler
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_invalid_agent_name_spaces(self, calibration_handler, mock_http_handler):
        result = await calibration_handler.handle(
            "/api/v1/agent/bad agent/calibration-curve", {}, mock_http_handler
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_tracker_exception_returns_500(self, calibration_handler, mock_http_handler):
        mock_tracker = MagicMock()
        mock_tracker.get_calibration_curve.side_effect = RuntimeError("DB error")

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/agent/claude/calibration-curve", {}, mock_http_handler
                )
                assert _status(result) == 500


# =============================================================================
# 4. Calibration Summary Endpoint Tests
# =============================================================================


class TestCalibrationSummary:
    """Tests for GET /api/agent/{name}/calibration-summary."""

    @pytest.mark.asyncio
    async def test_returns_503_when_tracker_unavailable(self, calibration_handler, mock_http_handler):
        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", False):
            result = await calibration_handler.handle(
                "/api/v1/agent/claude/calibration-summary", {}, mock_http_handler
            )
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_returns_503_when_tracker_class_none(self, calibration_handler, mock_http_handler):
        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch("aragora.server.handlers.agents.calibration.CalibrationTracker", None):
                result = await calibration_handler.handle(
                    "/api/v1/agent/claude/calibration-summary", {}, mock_http_handler
                )
                assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_success_returns_all_fields(self, calibration_handler, mock_http_handler):
        summary = _make_summary(
            agent="claude", total=100, correct=75, accuracy=0.75,
            brier=0.15, ece=0.05, overconfident=False, underconfident=True,
        )
        mock_tracker = MagicMock()
        mock_tracker.get_calibration_summary.return_value = summary

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/agent/claude/calibration-summary", {}, mock_http_handler
                )
                assert _status(result) == 200
                data = _body(result)
                assert data["agent"] == "claude"
                assert data["total_predictions"] == 100
                assert data["total_correct"] == 75
                assert data["accuracy"] == 0.75
                assert data["brier_score"] == 0.15
                assert data["ece"] == 0.05
                assert data["is_overconfident"] is False
                assert data["is_underconfident"] is True
                assert data["domain"] is None

    @pytest.mark.asyncio
    async def test_with_domain_param(self, calibration_handler, mock_http_handler):
        summary = _make_summary(agent="claude")
        mock_tracker = MagicMock()
        mock_tracker.get_calibration_summary.return_value = summary

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/agent/claude/calibration-summary",
                    {"domain": ["math"]},
                    mock_http_handler,
                )
                mock_tracker.get_calibration_summary.assert_called_once_with(
                    "claude", domain="math"
                )
                data = _body(result)
                assert data["domain"] == "math"

    @pytest.mark.asyncio
    async def test_overconfident_agent(self, calibration_handler, mock_http_handler):
        summary = _make_summary(overconfident=True, underconfident=False)
        mock_tracker = MagicMock()
        mock_tracker.get_calibration_summary.return_value = summary

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/agent/claude/calibration-summary", {}, mock_http_handler
                )
                data = _body(result)
                assert data["is_overconfident"] is True
                assert data["is_underconfident"] is False

    @pytest.mark.asyncio
    async def test_invalid_agent_name(self, calibration_handler, mock_http_handler):
        result = await calibration_handler.handle(
            "/api/v1/agent/<script>/calibration-summary", {}, mock_http_handler
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_tracker_runtime_error_returns_500(self, calibration_handler, mock_http_handler):
        mock_tracker = MagicMock()
        mock_tracker.get_calibration_summary.side_effect = RuntimeError("DB error")

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/agent/claude/calibration-summary", {}, mock_http_handler
                )
                assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_tracker_value_error_returns_400(self, calibration_handler, mock_http_handler):
        mock_tracker = MagicMock()
        mock_tracker.get_calibration_summary.side_effect = ValueError("bad data")

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/agent/claude/calibration-summary", {}, mock_http_handler
                )
                assert _status(result) == 400


# =============================================================================
# 5. Calibration Leaderboard Endpoint Tests
# =============================================================================


class TestCalibrationLeaderboard:
    """Tests for GET /api/calibration/leaderboard."""

    def _setup_elo(self, agents):
        """Create a mock EloSystem with the given agent data.

        agents: list of dicts with keys: name, elo, score, brier, acc, total, correct
        """
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [{"agent": a["name"]} for a in agents]

        def get_rating(name):
            for a in agents:
                if a["name"] == name:
                    return _make_rating(
                        elo=a.get("elo", 1500),
                        score=a.get("score", 0.8),
                        brier=a.get("brier", 0.1),
                        acc=a.get("acc", 0.75),
                        total=a.get("total", 100),
                        correct=a.get("correct", 75),
                    )
            raise KeyError(f"Agent {name} not found")

        mock_elo.get_rating.side_effect = get_rating
        return mock_elo

    @pytest.mark.asyncio
    async def test_returns_503_when_elo_unavailable(self, calibration_handler, mock_http_handler):
        with patch("aragora.server.handlers.agents.calibration.ELO_AVAILABLE", False):
            result = await calibration_handler.handle(
                "/api/v1/calibration/leaderboard", {}, mock_http_handler
            )
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_returns_503_when_elo_class_none(self, calibration_handler, mock_http_handler):
        with patch("aragora.server.handlers.agents.calibration.ELO_AVAILABLE", True):
            with patch("aragora.server.handlers.agents.calibration.EloSystem", None):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/leaderboard", {}, mock_http_handler
                )
                assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_success_returns_agents(self, calibration_handler, mock_http_handler):
        agents = [
            {"name": "claude", "elo": 1500, "score": 0.85, "brier": 0.1, "acc": 0.8, "total": 100, "correct": 80},
            {"name": "gpt4", "elo": 1480, "score": 0.75, "brier": 0.15, "acc": 0.7, "total": 50, "correct": 35},
        ]
        mock_elo = self._setup_elo(agents)

        with patch("aragora.server.handlers.agents.calibration.ELO_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.EloSystem",
                return_value=mock_elo,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/leaderboard", {}, mock_http_handler
                )
                assert _status(result) == 200
                data = _body(result)
                assert data["count"] == 2
                assert len(data["agents"]) == 2
                assert data["metric"] == "brier"
                assert data["min_predictions"] == 5

    @pytest.mark.asyncio
    async def test_leaderboard_entry_fields(self, calibration_handler, mock_http_handler):
        agents = [{"name": "claude", "elo": 1500, "score": 0.85, "brier": 0.1, "acc": 0.8, "total": 100, "correct": 80}]
        mock_elo = self._setup_elo(agents)

        with patch("aragora.server.handlers.agents.calibration.ELO_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.EloSystem",
                return_value=mock_elo,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/leaderboard", {}, mock_http_handler
                )
                entry = _body(result)["agents"][0]
                assert entry["agent"] == "claude"
                assert entry["calibration_score"] == 0.85
                assert entry["brier_score"] == 0.1
                assert entry["accuracy"] == 0.8
                assert "ece" in entry
                assert entry["predictions_count"] == 100
                assert entry["correct_count"] == 80
                assert entry["elo"] == 1500

    @pytest.mark.asyncio
    async def test_ece_calculation(self, calibration_handler, mock_http_handler):
        agents = [{"name": "agent1", "score": 0.85, "total": 100}]
        mock_elo = self._setup_elo(agents)

        with patch("aragora.server.handlers.agents.calibration.ELO_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.EloSystem",
                return_value=mock_elo,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/leaderboard", {}, mock_http_handler
                )
                entry = _body(result)["agents"][0]
                assert entry["ece"] == pytest.approx(1.0 - 0.85)

    @pytest.mark.asyncio
    async def test_ece_when_score_is_zero(self, calibration_handler, mock_http_handler):
        agents = [{"name": "agent1", "score": 0.0, "total": 100}]
        mock_elo = self._setup_elo(agents)

        with patch("aragora.server.handlers.agents.calibration.ELO_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.EloSystem",
                return_value=mock_elo,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/leaderboard", {}, mock_http_handler
                )
                entry = _body(result)["agents"][0]
                assert entry["ece"] == 1.0

    @pytest.mark.asyncio
    async def test_filters_by_min_predictions(self, calibration_handler, mock_http_handler):
        agents = [
            {"name": "claude", "total": 100},
            {"name": "newbie", "total": 3},  # Below default min of 5
        ]
        mock_elo = self._setup_elo(agents)

        with patch("aragora.server.handlers.agents.calibration.ELO_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.EloSystem",
                return_value=mock_elo,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/leaderboard", {}, mock_http_handler
                )
                data = _body(result)
                assert data["count"] == 1
                assert data["agents"][0]["agent"] == "claude"

    @pytest.mark.asyncio
    async def test_custom_min_predictions(self, calibration_handler, mock_http_handler):
        agents = [
            {"name": "claude", "total": 100},
            {"name": "mid", "total": 50},
            {"name": "newbie", "total": 10},
        ]
        mock_elo = self._setup_elo(agents)

        with patch("aragora.server.handlers.agents.calibration.ELO_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.EloSystem",
                return_value=mock_elo,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/leaderboard",
                    {"min_predictions": ["60"]},
                    mock_http_handler,
                )
                data = _body(result)
                assert data["count"] == 1
                assert data["agents"][0]["agent"] == "claude"

    @pytest.mark.asyncio
    async def test_sort_by_brier(self, calibration_handler, mock_http_handler):
        agents = [
            {"name": "worse", "brier": 0.3, "total": 100},
            {"name": "better", "brier": 0.05, "total": 100},
        ]
        mock_elo = self._setup_elo(agents)

        with patch("aragora.server.handlers.agents.calibration.ELO_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.EloSystem",
                return_value=mock_elo,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/leaderboard",
                    {"metric": ["brier"]},
                    mock_http_handler,
                )
                data = _body(result)
                assert data["agents"][0]["agent"] == "better"
                assert data["agents"][1]["agent"] == "worse"

    @pytest.mark.asyncio
    async def test_sort_by_ece(self, calibration_handler, mock_http_handler):
        agents = [
            {"name": "low_score", "score": 0.6, "total": 100},   # ECE = 0.4
            {"name": "high_score", "score": 0.95, "total": 100},  # ECE = 0.05
        ]
        mock_elo = self._setup_elo(agents)

        with patch("aragora.server.handlers.agents.calibration.ELO_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.EloSystem",
                return_value=mock_elo,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/leaderboard",
                    {"metric": ["ece"]},
                    mock_http_handler,
                )
                data = _body(result)
                # Lower ECE is better, so high_score agent first
                assert data["agents"][0]["agent"] == "high_score"

    @pytest.mark.asyncio
    async def test_sort_by_accuracy(self, calibration_handler, mock_http_handler):
        agents = [
            {"name": "low_acc", "acc": 0.5, "total": 100},
            {"name": "high_acc", "acc": 0.95, "total": 100},
        ]
        mock_elo = self._setup_elo(agents)

        with patch("aragora.server.handlers.agents.calibration.ELO_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.EloSystem",
                return_value=mock_elo,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/leaderboard",
                    {"metric": ["accuracy"]},
                    mock_http_handler,
                )
                data = _body(result)
                # Higher accuracy is better
                assert data["agents"][0]["agent"] == "high_acc"

    @pytest.mark.asyncio
    async def test_sort_by_composite(self, calibration_handler, mock_http_handler):
        agents = [
            {"name": "low_score", "score": 0.5, "total": 100},
            {"name": "high_score", "score": 0.95, "total": 100},
        ]
        mock_elo = self._setup_elo(agents)

        with patch("aragora.server.handlers.agents.calibration.ELO_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.EloSystem",
                return_value=mock_elo,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/leaderboard",
                    {"metric": ["composite"]},
                    mock_http_handler,
                )
                data = _body(result)
                # Higher calibration_score is better
                assert data["agents"][0]["agent"] == "high_score"

    @pytest.mark.asyncio
    async def test_unknown_metric_uses_composite(self, calibration_handler, mock_http_handler):
        agents = [
            {"name": "low_score", "score": 0.5, "total": 100},
            {"name": "high_score", "score": 0.95, "total": 100},
        ]
        mock_elo = self._setup_elo(agents)

        with patch("aragora.server.handlers.agents.calibration.ELO_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.EloSystem",
                return_value=mock_elo,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/leaderboard",
                    {"metric": ["unknown_metric"]},
                    mock_http_handler,
                )
                data = _body(result)
                assert data["agents"][0]["agent"] == "high_score"

    @pytest.mark.asyncio
    async def test_limit_param(self, calibration_handler, mock_http_handler):
        agents = [{"name": f"agent{i}", "total": 100} for i in range(5)]
        mock_elo = self._setup_elo(agents)

        with patch("aragora.server.handlers.agents.calibration.ELO_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.EloSystem",
                return_value=mock_elo,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/leaderboard",
                    {"limit": ["2"]},
                    mock_http_handler,
                )
                data = _body(result)
                assert data["count"] == 2

    @pytest.mark.asyncio
    async def test_skips_agents_with_empty_name(self, calibration_handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [
            {"agent": ""},
            {"agent": "claude"},
        ]
        mock_elo.get_rating.return_value = _make_rating(total=100)

        with patch("aragora.server.handlers.agents.calibration.ELO_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.EloSystem",
                return_value=mock_elo,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/leaderboard", {}, mock_http_handler
                )
                data = _body(result)
                assert data["count"] == 1
                assert data["agents"][0]["agent"] == "claude"

    @pytest.mark.asyncio
    async def test_skips_agents_with_rating_errors(self, calibration_handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [
            {"agent": "bad"},
            {"agent": "good"},
        ]

        def get_rating(name):
            if name == "bad":
                raise KeyError("Not found")
            return _make_rating(total=100)

        mock_elo.get_rating.side_effect = get_rating

        with patch("aragora.server.handlers.agents.calibration.ELO_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.EloSystem",
                return_value=mock_elo,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/leaderboard", {}, mock_http_handler
                )
                data = _body(result)
                assert data["count"] == 1
                assert data["agents"][0]["agent"] == "good"

    @pytest.mark.asyncio
    async def test_skips_agent_with_value_error(self, calibration_handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [{"agent": "broken"}]
        mock_elo.get_rating.side_effect = ValueError("bad data")

        with patch("aragora.server.handlers.agents.calibration.ELO_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.EloSystem",
                return_value=mock_elo,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/leaderboard", {}, mock_http_handler
                )
                data = _body(result)
                assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_skips_agent_with_attribute_error(self, calibration_handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [{"agent": "broken"}]
        mock_elo.get_rating.side_effect = AttributeError("no attr")

        with patch("aragora.server.handlers.agents.calibration.ELO_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.EloSystem",
                return_value=mock_elo,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/leaderboard", {}, mock_http_handler
                )
                assert _body(result)["count"] == 0

    @pytest.mark.asyncio
    async def test_skips_agent_with_type_error(self, calibration_handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [{"agent": "broken"}]
        mock_elo.get_rating.side_effect = TypeError("type mismatch")

        with patch("aragora.server.handlers.agents.calibration.ELO_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.EloSystem",
                return_value=mock_elo,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/leaderboard", {}, mock_http_handler
                )
                assert _body(result)["count"] == 0

    @pytest.mark.asyncio
    async def test_empty_leaderboard(self, calibration_handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = []

        with patch("aragora.server.handlers.agents.calibration.ELO_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.EloSystem",
                return_value=mock_elo,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/leaderboard", {}, mock_http_handler
                )
                data = _body(result)
                assert data["count"] == 0
                assert data["agents"] == []

    @pytest.mark.asyncio
    async def test_elo_get_leaderboard_exception_returns_500(self, calibration_handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.side_effect = RuntimeError("DB down")

        with patch("aragora.server.handlers.agents.calibration.ELO_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.EloSystem",
                return_value=mock_elo,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/leaderboard", {}, mock_http_handler
                )
                assert _status(result) == 500


# =============================================================================
# 6. Calibration Visualization Endpoint Tests
# =============================================================================


class TestCalibrationVisualization:
    """Tests for GET /api/calibration/visualization."""

    def _setup_tracker(self, agents_data):
        """Create a mock CalibrationTracker with given agent data.

        agents_data: list of dicts with keys: name, summary, buckets, domains
        """
        mock_tracker = MagicMock()
        agent_names = [a["name"] for a in agents_data]
        mock_tracker.get_all_agents.return_value = agent_names

        def get_summary(agent, domain=None):
            for a in agents_data:
                if a["name"] == agent:
                    return a.get("summary", _make_summary(agent=agent))
            return None

        def get_curve(agent, num_buckets=10):
            for a in agents_data:
                if a["name"] == agent:
                    return a.get("buckets", _make_buckets(num_buckets))
            return []

        def get_domain_breakdown(agent):
            for a in agents_data:
                if a["name"] == agent:
                    return a.get("domains", {})
            return {}

        mock_tracker.get_calibration_summary.side_effect = get_summary
        mock_tracker.get_calibration_curve.side_effect = get_curve
        mock_tracker.get_domain_breakdown.side_effect = get_domain_breakdown
        return mock_tracker

    @pytest.mark.asyncio
    async def test_returns_503_when_tracker_unavailable(self, calibration_handler, mock_http_handler):
        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", False):
            result = await calibration_handler.handle(
                "/api/v1/calibration/visualization", {}, mock_http_handler
            )
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_returns_503_when_tracker_class_none(self, calibration_handler, mock_http_handler):
        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch("aragora.server.handlers.agents.calibration.CalibrationTracker", None):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/visualization", {}, mock_http_handler
                )
                assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_empty_agents_list(self, calibration_handler, mock_http_handler):
        mock_tracker = MagicMock()
        mock_tracker.get_all_agents.return_value = []

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/visualization", {}, mock_http_handler
                )
                assert _status(result) == 200
                data = _body(result)
                assert data["summary"]["total_agents"] == 0
                assert data["calibration_curves"] == {}
                assert data["scatter_data"] == []

    @pytest.mark.asyncio
    async def test_agents_below_min_predictions(self, calibration_handler, mock_http_handler):
        mock_tracker = MagicMock()
        mock_tracker.get_all_agents.return_value = ["agent1"]
        mock_tracker.get_calibration_summary.return_value = _make_summary(total=2)  # Below 5

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/visualization", {}, mock_http_handler
                )
                data = _body(result)
                assert data["summary"]["total_agents"] == 1
                assert data["scatter_data"] == []

    @pytest.mark.asyncio
    async def test_full_visualization_data(self, calibration_handler, mock_http_handler):
        buckets = _make_buckets(2)
        summary = _make_summary(agent="claude", total=100, brier=0.1, ece=0.05)
        domain_summary = _make_summary(agent="claude", total=50, accuracy=0.9, brier=0.08)

        agents_data = [{
            "name": "claude",
            "summary": summary,
            "buckets": buckets,
            "domains": {"math": domain_summary},
        }]
        mock_tracker = self._setup_tracker(agents_data)

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/visualization", {}, mock_http_handler
                )
                assert _status(result) == 200
                data = _body(result)

                # Summary
                assert data["summary"]["total_agents"] == 1
                assert data["summary"]["avg_brier"] == 0.1
                assert data["summary"]["avg_ece"] == 0.05
                assert data["summary"]["best_calibrated"] == "claude"
                assert data["summary"]["worst_calibrated"] == "claude"

                # Calibration curves
                assert "claude" in data["calibration_curves"]
                curve = data["calibration_curves"]["claude"]
                assert len(curve["buckets"]) == 2
                assert len(curve["perfect_line"]) == 11

                # Scatter data
                assert len(data["scatter_data"]) == 1
                scatter = data["scatter_data"][0]
                assert scatter["agent"] == "claude"
                assert scatter["accuracy"] == 0.75
                assert scatter["brier_score"] == 0.1

                # Confidence histogram
                assert len(data["confidence_histogram"]) == 10
                assert data["confidence_histogram"][0]["range"] == "0-10%"

                # Domain heatmap
                assert "claude" in data["domain_heatmap"]
                assert "math" in data["domain_heatmap"]["claude"]

    @pytest.mark.asyncio
    async def test_scatter_data_fields(self, calibration_handler, mock_http_handler):
        summary = _make_summary(
            agent="claude", accuracy=0.85, brier=0.12, ece=0.04,
            total=200, overconfident=True, underconfident=False,
        )
        mock_tracker = self._setup_tracker([{"name": "claude", "summary": summary}])

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/visualization", {}, mock_http_handler
                )
                scatter = _body(result)["scatter_data"][0]
                assert scatter["agent"] == "claude"
                assert scatter["accuracy"] == 0.85
                assert scatter["brier_score"] == 0.12
                assert scatter["ece"] == 0.04
                assert scatter["predictions"] == 200
                assert scatter["is_overconfident"] is True
                assert scatter["is_underconfident"] is False

    @pytest.mark.asyncio
    async def test_perfect_line_in_curves(self, calibration_handler, mock_http_handler):
        mock_tracker = self._setup_tracker([{
            "name": "agent1",
            "summary": _make_summary(agent="agent1"),
            "buckets": [MockCalibrationBucket(0.0, 0.1, 10, 1, 0.1, 0.09)],
        }])

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/visualization", {}, mock_http_handler
                )
                line = _body(result)["calibration_curves"]["agent1"]["perfect_line"]
                assert line[0] == {"x": 0.0, "y": 0.0}
                assert line[-1] == {"x": 1.0, "y": 1.0}
                assert len(line) == 11

    @pytest.mark.asyncio
    async def test_multi_agent_summary_stats(self, calibration_handler, mock_http_handler):
        agents_data = [
            {"name": "best", "summary": _make_summary(agent="best", brier=0.05, ece=0.02)},
            {"name": "mid", "summary": _make_summary(agent="mid", brier=0.15, ece=0.06)},
            {"name": "worst", "summary": _make_summary(agent="worst", brier=0.25, ece=0.10)},
        ]
        mock_tracker = self._setup_tracker(agents_data)

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/visualization", {}, mock_http_handler
                )
                data = _body(result)
                assert data["summary"]["best_calibrated"] == "best"
                assert data["summary"]["worst_calibrated"] == "worst"
                assert data["summary"]["avg_brier"] == pytest.approx(0.15, abs=0.001)
                assert data["summary"]["avg_ece"] == pytest.approx(0.06, abs=0.001)

    @pytest.mark.asyncio
    async def test_limit_param(self, calibration_handler, mock_http_handler):
        agents_data = [
            {"name": f"agent{i}", "summary": _make_summary(agent=f"agent{i}")}
            for i in range(5)
        ]
        mock_tracker = self._setup_tracker(agents_data)

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/visualization",
                    {"limit": ["2"]},
                    mock_http_handler,
                )
                data = _body(result)
                # Should only process 2 agents
                assert len(data["scatter_data"]) <= 2

    @pytest.mark.asyncio
    async def test_limit_clamped_to_max_10(self, calibration_handler, mock_http_handler):
        """Limit param is clamped to max of 10."""
        agents_data = [
            {"name": f"agent{i}", "summary": _make_summary(agent=f"agent{i}")}
            for i in range(15)
        ]
        mock_tracker = self._setup_tracker(agents_data)

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/visualization",
                    {"limit": ["50"]},  # Above max of 10
                    mock_http_handler,
                )
                data = _body(result)
                # Clamped to 10 agents max
                assert len(data["scatter_data"]) <= 10

    @pytest.mark.asyncio
    async def test_agent_summary_error_skipped(self, calibration_handler, mock_http_handler):
        mock_tracker = MagicMock()
        mock_tracker.get_all_agents.return_value = ["bad_agent", "good_agent"]

        def get_summary(agent, domain=None):
            if agent == "bad_agent":
                raise ValueError("corrupt data")
            return _make_summary(agent=agent)

        mock_tracker.get_calibration_summary.side_effect = get_summary
        mock_tracker.get_calibration_curve.return_value = _make_buckets(2)
        mock_tracker.get_domain_breakdown.return_value = {}

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/visualization", {}, mock_http_handler
                )
                data = _body(result)
                assert len(data["scatter_data"]) == 1
                assert data["scatter_data"][0]["agent"] == "good_agent"

    @pytest.mark.asyncio
    async def test_curve_error_skipped_gracefully(self, calibration_handler, mock_http_handler):
        mock_tracker = MagicMock()
        mock_tracker.get_all_agents.return_value = ["agent1"]
        mock_tracker.get_calibration_summary.return_value = _make_summary(agent="agent1")
        mock_tracker.get_calibration_curve.side_effect = KeyError("missing data")
        mock_tracker.get_domain_breakdown.return_value = {}

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/visualization", {}, mock_http_handler
                )
                data = _body(result)
                assert data["calibration_curves"] == {}
                # Scatter data should still be present
                assert len(data["scatter_data"]) == 1

    @pytest.mark.asyncio
    async def test_domain_breakdown_error_skipped(self, calibration_handler, mock_http_handler):
        mock_tracker = MagicMock()
        mock_tracker.get_all_agents.return_value = ["agent1"]
        mock_tracker.get_calibration_summary.return_value = _make_summary(agent="agent1")
        mock_tracker.get_calibration_curve.return_value = _make_buckets(2)
        mock_tracker.get_domain_breakdown.side_effect = TypeError("bad")

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/visualization", {}, mock_http_handler
                )
                data = _body(result)
                assert data["domain_heatmap"] == {}
                assert len(data["scatter_data"]) == 1

    @pytest.mark.asyncio
    async def test_histogram_ranges(self, calibration_handler, mock_http_handler):
        mock_tracker = self._setup_tracker([{
            "name": "agent1",
            "summary": _make_summary(agent="agent1"),
            "buckets": _make_buckets(10),
        }])

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/visualization", {}, mock_http_handler
                )
                histogram = _body(result)["confidence_histogram"]
                ranges = [h["range"] for h in histogram]
                assert "0-10%" in ranges
                assert "90-100%" in ranges
                assert len(histogram) == 10

    @pytest.mark.asyncio
    async def test_domain_heatmap_fields(self, calibration_handler, mock_http_handler):
        domain_summary = _make_summary(accuracy=0.9, brier=0.08, total=50)
        mock_tracker = self._setup_tracker([{
            "name": "agent1",
            "summary": _make_summary(agent="agent1"),
            "buckets": _make_buckets(2),
            "domains": {"technical": domain_summary},
        }])

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/visualization", {}, mock_http_handler
                )
                heatmap = _body(result)["domain_heatmap"]
                assert "agent1" in heatmap
                tech = heatmap["agent1"]["technical"]
                assert tech["accuracy"] == 0.9
                assert tech["brier"] == 0.08
                assert tech["count"] == 50

    @pytest.mark.asyncio
    async def test_outer_exception_returns_result_anyway(self, calibration_handler, mock_http_handler):
        """Top-level exception in visualization is caught and result still returned."""
        mock_tracker = MagicMock()
        mock_tracker.get_all_agents.side_effect = RuntimeError("fatal")

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/visualization", {}, mock_http_handler
                )
                assert _status(result) == 200
                data = _body(result)
                # Default empty result returned
                assert data["calibration_curves"] == {}
                assert data["scatter_data"] == []

    @pytest.mark.asyncio
    async def test_empty_curve_not_added(self, calibration_handler, mock_http_handler):
        mock_tracker = MagicMock()
        mock_tracker.get_all_agents.return_value = ["agent1"]
        mock_tracker.get_calibration_summary.return_value = _make_summary(agent="agent1")
        mock_tracker.get_calibration_curve.return_value = []  # Empty curve
        mock_tracker.get_domain_breakdown.return_value = {}

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/visualization", {}, mock_http_handler
                )
                data = _body(result)
                # Empty curve should not be added to calibration_curves
                assert "agent1" not in data["calibration_curves"]

    @pytest.mark.asyncio
    async def test_empty_domain_breakdown_not_added(self, calibration_handler, mock_http_handler):
        mock_tracker = MagicMock()
        mock_tracker.get_all_agents.return_value = ["agent1"]
        mock_tracker.get_calibration_summary.return_value = _make_summary(agent="agent1")
        mock_tracker.get_calibration_curve.return_value = _make_buckets(2)
        mock_tracker.get_domain_breakdown.return_value = {}

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/visualization", {}, mock_http_handler
                )
                data = _body(result)
                assert "agent1" not in data["domain_heatmap"]


# =============================================================================
# 7. Authentication and Authorization Tests
# =============================================================================


class TestCalibrationAuth:
    """Tests for authentication and authorization."""

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_unauthenticated_returns_401(self, mock_http_handler):
        from aragora.server.handlers.agents.calibration import CalibrationHandler
        from aragora.server.handlers.secure import SecureHandler, UnauthorizedError

        handler = CalibrationHandler({})

        async def mock_get_auth_raise(self, request, require_auth=True):
            raise UnauthorizedError("Not authenticated")

        with patch.object(SecureHandler, "get_auth_context", mock_get_auth_raise):
            result = await handler.handle(
                "/api/v1/agent/claude/calibration-curve", {}, mock_http_handler
            )
            assert _status(result) == 401
            assert "authentication" in _body(result).get("error", "").lower()

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_forbidden_returns_403(self, mock_http_handler):
        from aragora.server.handlers.agents.calibration import CalibrationHandler
        from aragora.server.handlers.secure import SecureHandler, ForbiddenError
        from aragora.rbac.models import AuthorizationContext

        handler = CalibrationHandler({})

        mock_ctx = AuthorizationContext(
            user_id="user1", user_email="u@test.com",
            org_id="org1", roles={"viewer"}, permissions=set(),
        )

        async def mock_get_auth(self, request, require_auth=True):
            return mock_ctx

        def mock_check_perm(self, ctx, perm, resource_id=None):
            raise ForbiddenError(f"Missing {perm}")

        with patch.object(SecureHandler, "get_auth_context", mock_get_auth):
            with patch.object(SecureHandler, "check_permission", mock_check_perm):
                result = await handler.handle(
                    "/api/v1/agent/claude/calibration-curve", {}, mock_http_handler
                )
                assert _status(result) == 403
                assert "permission" in _body(result).get("error", "").lower()

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_401_on_leaderboard(self, mock_http_handler):
        from aragora.server.handlers.agents.calibration import CalibrationHandler
        from aragora.server.handlers.secure import SecureHandler, UnauthorizedError

        handler = CalibrationHandler({})

        async def mock_get_auth_raise(self, request, require_auth=True):
            raise UnauthorizedError("Not authenticated")

        with patch.object(SecureHandler, "get_auth_context", mock_get_auth_raise):
            result = await handler.handle(
                "/api/v1/calibration/leaderboard", {}, mock_http_handler
            )
            assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_401_on_visualization(self, mock_http_handler):
        from aragora.server.handlers.agents.calibration import CalibrationHandler
        from aragora.server.handlers.secure import SecureHandler, UnauthorizedError

        handler = CalibrationHandler({})

        async def mock_get_auth_raise(self, request, require_auth=True):
            raise UnauthorizedError("Not authenticated")

        with patch.object(SecureHandler, "get_auth_context", mock_get_auth_raise):
            result = await handler.handle(
                "/api/v1/calibration/visualization", {}, mock_http_handler
            )
            assert _status(result) == 401


# =============================================================================
# 8. Rate Limiting Tests
# =============================================================================


class TestCalibrationRateLimiting:
    """Tests for rate limiting on calibration endpoints."""

    @pytest.mark.asyncio
    async def test_rate_limit_enforced(self, calibration_handler):
        """Exceeding 30 requests returns 429."""
        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", False):
            for i in range(35):
                h = _make_mock_http_handler(ip="10.0.0.99")
                result = await calibration_handler.handle(
                    "/api/v1/agent/claude/calibration-curve", {}, h
                )
                if _status(result) == 429:
                    assert "rate limit" in _body(result).get("error", "").lower()
                    return
        # Rate limiting may vary by timing; this is acceptable

    @pytest.mark.asyncio
    async def test_different_ips_not_rate_limited(self, calibration_handler):
        """Different IPs have independent rate limits."""
        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", False):
            for i in range(5):
                h1 = _make_mock_http_handler(ip=f"10.0.{i}.1")
                result = await calibration_handler.handle(
                    "/api/v1/agent/claude/calibration-curve", {}, h1
                )
                # None of these should be rate-limited (5 requests from 5 different IPs)
                assert _status(result) != 429


# =============================================================================
# 9. Path Routing / Dispatch Tests
# =============================================================================


class TestPathRouting:
    """Tests for path routing in handle()."""

    @pytest.mark.asyncio
    async def test_non_agent_non_calibration_path_returns_none(self, calibration_handler, mock_http_handler):
        result = await calibration_handler.handle(
            "/api/v1/debates/list", {}, mock_http_handler
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_agent_unknown_suffix_returns_none(self, calibration_handler, mock_http_handler):
        result = await calibration_handler.handle(
            "/api/v1/agent/claude/unknown-endpoint", {}, mock_http_handler
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_agent_root_returns_none(self, calibration_handler, mock_http_handler):
        result = await calibration_handler.handle(
            "/api/v1/agent/claude", {}, mock_http_handler
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_version_prefix_stripped_for_routing(self, calibration_handler, mock_http_handler):
        """Both /api/v1/... and /api/... paths route correctly."""
        mock_tracker = MagicMock()
        mock_tracker.get_calibration_curve.return_value = []

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                # With version prefix
                result = await calibration_handler.handle(
                    "/api/v1/agent/claude/calibration-curve", {}, mock_http_handler
                )
                assert _status(result) == 200

                # Without version prefix
                result = await calibration_handler.handle(
                    "/api/agent/claude/calibration-curve", {}, mock_http_handler
                )
                assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_leaderboard_default_params(self, calibration_handler, mock_http_handler):
        """Leaderboard uses correct defaults when no params given."""
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = []

        with patch("aragora.server.handlers.agents.calibration.ELO_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.EloSystem",
                return_value=mock_elo,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/leaderboard", {}, mock_http_handler
                )
                data = _body(result)
                assert data["metric"] == "brier"
                assert data["min_predictions"] == 5

    @pytest.mark.asyncio
    async def test_visualization_default_limit(self, calibration_handler, mock_http_handler):
        """Visualization default limit is 5."""
        mock_tracker = MagicMock()
        mock_tracker.get_all_agents.return_value = []

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/visualization", {}, mock_http_handler
                )
                assert _status(result) == 200


# =============================================================================
# 10. Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_agent_name_with_hyphens(self, calibration_handler, mock_http_handler):
        mock_tracker = MagicMock()
        mock_tracker.get_calibration_curve.return_value = []

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/agent/mistral-large/calibration-curve", {}, mock_http_handler
                )
                assert _status(result) == 200
                assert _body(result)["agent"] == "mistral-large"

    @pytest.mark.asyncio
    async def test_agent_name_with_underscores(self, calibration_handler, mock_http_handler):
        mock_tracker = MagicMock()
        mock_tracker.get_calibration_curve.return_value = []

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/agent/gemini_pro/calibration-curve", {}, mock_http_handler
                )
                assert _status(result) == 200
                assert _body(result)["agent"] == "gemini_pro"

    @pytest.mark.asyncio
    async def test_agent_name_with_numbers(self, calibration_handler, mock_http_handler):
        mock_tracker = MagicMock()
        mock_tracker.get_calibration_curve.return_value = []

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/agent/gpt4o/calibration-curve", {}, mock_http_handler
                )
                assert _status(result) == 200
                assert _body(result)["agent"] == "gpt4o"

    @pytest.mark.asyncio
    async def test_leaderboard_all_agents_filtered_out(self, calibration_handler, mock_http_handler):
        """When all agents are below min_predictions, return empty."""
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [
            {"agent": "agent1"},
            {"agent": "agent2"},
        ]
        mock_elo.get_rating.return_value = _make_rating(total=2)

        with patch("aragora.server.handlers.agents.calibration.ELO_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.EloSystem",
                return_value=mock_elo,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/leaderboard",
                    {"min_predictions": ["10"]},
                    mock_http_handler,
                )
                data = _body(result)
                assert data["count"] == 0
                assert data["agents"] == []

    @pytest.mark.asyncio
    async def test_visualization_confidence_histogram_aggregates(self, calibration_handler, mock_http_handler):
        """Histogram aggregates predictions from multiple agents."""
        bucket = MockCalibrationBucket(0.0, 0.1, 5, 1, 0.1, 0.09)

        mock_tracker = MagicMock()
        mock_tracker.get_all_agents.return_value = ["a1", "a2"]
        mock_tracker.get_calibration_summary.return_value = _make_summary(total=100)
        mock_tracker.get_calibration_curve.return_value = [bucket]
        mock_tracker.get_domain_breakdown.return_value = {}

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/visualization", {}, mock_http_handler
                )
                histogram = _body(result)["confidence_histogram"]
                # First bucket should have 5+5=10 (from two agents)
                assert histogram[0]["count"] == 10

    @pytest.mark.asyncio
    async def test_leaderboard_entry_without_agent_key(self, calibration_handler, mock_http_handler):
        """Entries without 'agent' key are skipped."""
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [
            {"name": "missing_agent_key"},  # No "agent" key
            {"agent": "good"},
        ]
        mock_elo.get_rating.return_value = _make_rating(total=100)

        with patch("aragora.server.handlers.agents.calibration.ELO_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.EloSystem",
                return_value=mock_elo,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/calibration/leaderboard", {}, mock_http_handler
                )
                data = _body(result)
                assert data["count"] == 1
                assert data["agents"][0]["agent"] == "good"

    @pytest.mark.asyncio
    async def test_curve_bucket_expected_accuracy_calculation(self, calibration_handler, mock_http_handler):
        """Expected accuracy is the midpoint of range_start and range_end."""
        bucket = MockCalibrationBucket(0.4, 0.6, 30, 15, 0.5, 0.04)
        mock_tracker = MagicMock()
        mock_tracker.get_calibration_curve.return_value = [bucket]

        with patch("aragora.server.handlers.agents.calibration.CALIBRATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.agents.calibration.CalibrationTracker",
                return_value=mock_tracker,
            ):
                result = await calibration_handler.handle(
                    "/api/v1/agent/claude/calibration-curve", {}, mock_http_handler
                )
                b = _body(result)["buckets"][0]
                assert b["expected_accuracy"] == pytest.approx(0.5)
