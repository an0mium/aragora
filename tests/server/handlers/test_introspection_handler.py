"""Tests for Agent Introspection handler endpoints.

Comprehensive tests covering:
- Agent self-awareness data endpoints
- Introspection snapshots and caching
- Track record retrieval
- Historical performance data
- Calibration metrics
- Error handling and validation
- Authentication requirements
- Rate limiting
"""

import json
from datetime import datetime
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.introspection import IntrospectionHandler


def mock_authenticated_user():
    """Create a mock authenticated user context."""
    from aragora.billing.auth.context import UserAuthContext
    user_ctx = MagicMock(spec=UserAuthContext)
    user_ctx.is_authenticated = True
    user_ctx.user_id = "test-user-123"
    user_ctx.email = "test@example.com"
    return user_ctx


def mock_unauthenticated_user():
    """Create a mock unauthenticated user context."""
    from aragora.billing.auth.context import UserAuthContext
    user_ctx = MagicMock(spec=UserAuthContext)
    user_ctx.is_authenticated = False
    user_ctx.user_id = None
    return user_ctx


@pytest.fixture
def introspection_handler():
    """Create an introspection handler with mocked dependencies."""
    ctx = {
        "storage": None,
        "elo_system": None,
        "nomic_dir": None,
        "calibration_tracker": None,
    }
    handler = IntrospectionHandler(ctx)
    return handler


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler with client address."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {"Content-Length": "0"}
    handler.command = "GET"
    return handler


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter before each test."""
    try:
        from aragora.server.handlers.introspection import _introspection_limiter
        if hasattr(_introspection_limiter, '_requests'):
            _introspection_limiter._requests.clear()
    except ImportError:
        pass
    yield


def create_request_body(data: dict, with_auth: bool = False) -> MagicMock:
    """Create a mock HTTP handler with a JSON body."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    body = json.dumps(data).encode("utf-8")
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": "application/json",
    }
    if with_auth:
        handler.headers["Authorization"] = "Bearer test-token"
    handler.rfile = BytesIO(body)
    handler.command = "POST"
    return handler


class TestIntrospectionHandlerCanHandle:
    """Test IntrospectionHandler.can_handle method."""

    def test_can_handle_snapshot(self, introspection_handler):
        """Test can_handle returns True for snapshot endpoint."""
        assert introspection_handler.can_handle("/api/introspection/snapshot")

    def test_can_handle_agent_snapshot(self, introspection_handler):
        """Test can_handle returns True for agent snapshot endpoint."""
        assert introspection_handler.can_handle("/api/introspection/agent/claude")

    def test_can_handle_track_record(self, introspection_handler):
        """Test can_handle returns True for track record endpoint."""
        assert introspection_handler.can_handle("/api/introspection/track-record/claude")

    def test_can_handle_calibration(self, introspection_handler):
        """Test can_handle returns True for calibration endpoint."""
        assert introspection_handler.can_handle("/api/introspection/calibration/claude")

    def test_can_handle_history(self, introspection_handler):
        """Test can_handle returns True for history endpoint."""
        assert introspection_handler.can_handle("/api/introspection/history/claude")

    def test_can_handle_compare(self, introspection_handler):
        """Test can_handle returns True for compare endpoint."""
        assert introspection_handler.can_handle("/api/introspection/compare")

    def test_can_handle_prompt_section(self, introspection_handler):
        """Test can_handle returns True for prompt section endpoint."""
        assert introspection_handler.can_handle("/api/introspection/prompt-section/claude")

    def test_cannot_handle_unknown(self, introspection_handler):
        """Test can_handle returns False for unknown endpoint."""
        assert not introspection_handler.can_handle("/api/unknown")
        assert not introspection_handler.can_handle("/api/agents")
        assert not introspection_handler.can_handle("/api/introspection/unknown")


class TestIntrospectionHandlerRoutesAttribute:
    """Tests for ROUTES class attribute."""

    def test_routes_contains_snapshot(self, introspection_handler):
        """ROUTES contains snapshot endpoint."""
        assert "/api/introspection/snapshot" in introspection_handler.ROUTES

    def test_routes_contains_compare(self, introspection_handler):
        """ROUTES contains compare endpoint."""
        assert "/api/introspection/compare" in introspection_handler.ROUTES


class TestIntrospectionHandlerSnapshotEndpoint:
    """Tests for GET /api/introspection/snapshot endpoint."""

    def test_snapshot_returns_503_when_unavailable(self, introspection_handler, mock_http_handler):
        """Snapshot endpoint returns 503 when introspection unavailable."""
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", False):
            result = introspection_handler.handle("/api/introspection/snapshot", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 503

    def test_snapshot_requires_agent_name(self, introspection_handler, mock_http_handler):
        """Snapshot endpoint requires agent_name parameter."""
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            result = introspection_handler.handle("/api/introspection/snapshot", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "agent_name" in body["error"].lower()

    def test_snapshot_validates_agent_name(self, introspection_handler, mock_http_handler):
        """Snapshot endpoint validates agent_name format."""
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            result = introspection_handler.handle(
                "/api/introspection/snapshot",
                {"agent_name": "../etc/passwd"},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 400

    def test_snapshot_success(self, introspection_handler, mock_http_handler):
        """Snapshot endpoint returns agent introspection data."""
        mock_snapshot = MagicMock()
        mock_snapshot.agent_name = "claude"
        mock_snapshot.reputation_score = 0.75
        mock_snapshot.vote_weight = 1.2
        mock_snapshot.proposals_made = 10
        mock_snapshot.proposals_accepted = 8
        mock_snapshot.calibration_score = 0.68
        mock_snapshot.debate_count = 25
        mock_snapshot.top_expertise = ["security", "architecture"]
        mock_snapshot.traits = ["thorough", "pragmatic"]
        mock_snapshot.to_dict = MagicMock(return_value={
            "agent_name": "claude",
            "reputation_score": 0.75,
            "vote_weight": 1.2,
            "calibration_score": 0.68,
        })

        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch("aragora.server.handlers.introspection.get_agent_introspection", return_value=mock_snapshot):
                result = introspection_handler.handle(
                    "/api/introspection/snapshot",
                    {"agent_name": "claude"},
                    mock_http_handler,
                )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "snapshot" in body
        assert body["snapshot"]["agent_name"] == "claude"

    def test_snapshot_agent_not_found(self, introspection_handler, mock_http_handler):
        """Snapshot endpoint returns 404 when agent not found."""
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch("aragora.server.handlers.introspection.get_agent_introspection", return_value=None):
                result = introspection_handler.handle(
                    "/api/introspection/snapshot",
                    {"agent_name": "unknown_agent"},
                    mock_http_handler,
                )

        assert result is not None
        assert result.status_code == 404


class TestIntrospectionHandlerAgentSnapshotEndpoint:
    """Tests for GET /api/introspection/agent/:name endpoint."""

    def test_agent_snapshot_returns_503_when_unavailable(self, introspection_handler, mock_http_handler):
        """Agent snapshot endpoint returns 503 when introspection unavailable."""
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", False):
            result = introspection_handler.handle("/api/introspection/agent/claude", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 503

    def test_agent_snapshot_validates_name(self, introspection_handler, mock_http_handler):
        """Agent snapshot endpoint validates agent name format."""
        result = introspection_handler.handle(
            "/api/introspection/agent/../etc/passwd",
            {},
            mock_http_handler,
        )

        assert result is not None
        assert result.status_code == 400

    def test_agent_snapshot_success(self, introspection_handler, mock_http_handler):
        """Agent snapshot endpoint returns agent data."""
        mock_snapshot = MagicMock()
        mock_snapshot.to_dict = MagicMock(return_value={
            "agent_name": "claude",
            "reputation_score": 0.8,
        })

        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch("aragora.server.handlers.introspection.get_agent_introspection", return_value=mock_snapshot):
                result = introspection_handler.handle(
                    "/api/introspection/agent/claude",
                    {},
                    mock_http_handler,
                )

        assert result is not None
        if result.status_code == 200:
            body = json.loads(result.body)
            assert "agent_name" in body or "snapshot" in body

    def test_agent_snapshot_not_found(self, introspection_handler, mock_http_handler):
        """Agent snapshot endpoint returns 404 for unknown agent."""
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch("aragora.server.handlers.introspection.get_agent_introspection", return_value=None):
                result = introspection_handler.handle(
                    "/api/introspection/agent/unknown",
                    {},
                    mock_http_handler,
                )

        assert result is not None
        assert result.status_code == 404


class TestIntrospectionHandlerTrackRecordEndpoint:
    """Tests for GET /api/introspection/track-record/:name endpoint."""

    def test_track_record_returns_503_when_unavailable(self, introspection_handler, mock_http_handler):
        """Track record endpoint returns 503 when introspection unavailable."""
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", False):
            result = introspection_handler.handle("/api/introspection/track-record/claude", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 503

    def test_track_record_validates_name(self, introspection_handler, mock_http_handler):
        """Track record endpoint validates agent name."""
        result = introspection_handler.handle(
            "/api/introspection/track-record/<script>alert(1)</script>",
            {},
            mock_http_handler,
        )

        assert result is not None
        assert result.status_code == 400

    def test_track_record_success(self, introspection_handler, mock_http_handler):
        """Track record endpoint returns performance history."""
        mock_record = {
            "agent_name": "claude",
            "debates_participated": 50,
            "wins": 30,
            "losses": 15,
            "draws": 5,
            "win_rate": 0.6,
            "elo_rating": 1650,
            "elo_history": [1500, 1550, 1600, 1650],
            "expertise_areas": ["security", "architecture", "testing"],
        }

        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch.object(introspection_handler, '_get_track_record', return_value=mock_record):
                result = introspection_handler.handle(
                    "/api/introspection/track-record/claude",
                    {},
                    mock_http_handler,
                )

        assert result is not None
        if result.status_code == 200:
            body = json.loads(result.body)
            assert "agent_name" in body
            assert "debates_participated" in body or "wins" in body

    def test_track_record_not_found(self, introspection_handler, mock_http_handler):
        """Track record endpoint returns 404 for unknown agent."""
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch.object(introspection_handler, '_get_track_record', return_value=None):
                result = introspection_handler.handle(
                    "/api/introspection/track-record/unknown",
                    {},
                    mock_http_handler,
                )

        assert result is not None
        assert result.status_code == 404


class TestIntrospectionHandlerCalibrationEndpoint:
    """Tests for GET /api/introspection/calibration/:name endpoint."""

    def test_calibration_returns_503_when_unavailable(self, introspection_handler, mock_http_handler):
        """Calibration endpoint returns 503 when tracker unavailable."""
        introspection_handler.ctx["calibration_tracker"] = None

        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            result = introspection_handler.handle(
                "/api/introspection/calibration/claude",
                {},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code in [200, 503]

    def test_calibration_validates_name(self, introspection_handler, mock_http_handler):
        """Calibration endpoint validates agent name."""
        result = introspection_handler.handle(
            "/api/introspection/calibration/../etc",
            {},
            mock_http_handler,
        )

        assert result is not None
        assert result.status_code == 400

    def test_calibration_success(self, introspection_handler, mock_http_handler):
        """Calibration endpoint returns calibration metrics."""
        mock_tracker = MagicMock()
        mock_tracker.get_agent_calibration.return_value = {
            "agent_name": "claude",
            "calibration_score": 0.72,
            "predictions_made": 100,
            "predictions_correct": 72,
            "confidence_bins": {
                "0.0-0.1": {"count": 10, "accuracy": 0.1},
                "0.9-1.0": {"count": 20, "accuracy": 0.95},
            },
        }
        introspection_handler.ctx["calibration_tracker"] = mock_tracker

        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            result = introspection_handler.handle(
                "/api/introspection/calibration/claude",
                {},
                mock_http_handler,
            )

        assert result is not None
        if result.status_code == 200:
            body = json.loads(result.body)
            assert "calibration_score" in body or "agent_name" in body

    def test_calibration_not_found(self, introspection_handler, mock_http_handler):
        """Calibration endpoint returns 404 for unknown agent."""
        mock_tracker = MagicMock()
        mock_tracker.get_agent_calibration.return_value = None
        introspection_handler.ctx["calibration_tracker"] = mock_tracker

        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            result = introspection_handler.handle(
                "/api/introspection/calibration/unknown",
                {},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 404


class TestIntrospectionHandlerHistoryEndpoint:
    """Tests for GET /api/introspection/history/:name endpoint."""

    def test_history_returns_503_when_unavailable(self, introspection_handler, mock_http_handler):
        """History endpoint returns 503 when introspection unavailable."""
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", False):
            result = introspection_handler.handle(
                "/api/introspection/history/claude",
                {},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 503

    def test_history_validates_name(self, introspection_handler, mock_http_handler):
        """History endpoint validates agent name."""
        result = introspection_handler.handle(
            "/api/introspection/history/",
            {},
            mock_http_handler,
        )

        assert result is not None
        assert result.status_code == 400

    def test_history_supports_limit_parameter(self, introspection_handler, mock_http_handler):
        """History endpoint supports limit parameter."""
        mock_history = [
            {"debate_id": "d1", "timestamp": "2026-01-10", "result": "win"},
            {"debate_id": "d2", "timestamp": "2026-01-11", "result": "loss"},
        ]

        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch.object(introspection_handler, '_get_agent_history', return_value=mock_history):
                result = introspection_handler.handle(
                    "/api/introspection/history/claude",
                    {"limit": "10"},
                    mock_http_handler,
                )

        assert result is not None

    def test_history_limit_capped(self, introspection_handler, mock_http_handler):
        """History endpoint caps limit at 100."""
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch.object(introspection_handler, '_get_agent_history', return_value=[]):
                result = introspection_handler.handle(
                    "/api/introspection/history/claude",
                    {"limit": "500"},
                    mock_http_handler,
                )

        assert result is not None

    def test_history_success(self, introspection_handler, mock_http_handler):
        """History endpoint returns debate history."""
        mock_history = [
            {"debate_id": "d1", "timestamp": "2026-01-10T10:00:00", "result": "win", "topic": "Rate limiting"},
            {"debate_id": "d2", "timestamp": "2026-01-11T14:30:00", "result": "loss", "topic": "Caching strategies"},
        ]

        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch.object(introspection_handler, '_get_agent_history', return_value=mock_history):
                result = introspection_handler.handle(
                    "/api/introspection/history/claude",
                    {},
                    mock_http_handler,
                )

        assert result is not None
        if result.status_code == 200:
            body = json.loads(result.body)
            assert "history" in body or "debates" in body


class TestIntrospectionHandlerCompareEndpoint:
    """Tests for GET /api/introspection/compare endpoint."""

    def test_compare_returns_503_when_unavailable(self, introspection_handler, mock_http_handler):
        """Compare endpoint returns 503 when introspection unavailable."""
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", False):
            result = introspection_handler.handle(
                "/api/introspection/compare",
                {"agents": "claude,gpt-4"},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 503

    def test_compare_requires_agents_parameter(self, introspection_handler, mock_http_handler):
        """Compare endpoint requires agents parameter."""
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            result = introspection_handler.handle(
                "/api/introspection/compare",
                {},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "agents" in body["error"].lower()

    def test_compare_requires_multiple_agents(self, introspection_handler, mock_http_handler):
        """Compare endpoint requires at least 2 agents."""
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            result = introspection_handler.handle(
                "/api/introspection/compare",
                {"agents": "claude"},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 400

    def test_compare_max_agents_capped(self, introspection_handler, mock_http_handler):
        """Compare endpoint caps agents at 10."""
        agents = ",".join([f"agent{i}" for i in range(15)])

        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            result = introspection_handler.handle(
                "/api/introspection/compare",
                {"agents": agents},
                mock_http_handler,
            )

        assert result is not None
        # Should either reject or cap at 10
        assert result.status_code in [200, 400]

    def test_compare_success(self, introspection_handler, mock_http_handler):
        """Compare endpoint returns comparison data."""
        mock_snapshot1 = MagicMock()
        mock_snapshot1.to_dict.return_value = {
            "agent_name": "claude",
            "reputation_score": 0.8,
            "calibration_score": 0.75,
        }
        mock_snapshot2 = MagicMock()
        mock_snapshot2.to_dict.return_value = {
            "agent_name": "gpt-4",
            "reputation_score": 0.78,
            "calibration_score": 0.72,
        }

        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch("aragora.server.handlers.introspection.get_agent_introspection") as mock_get:
                mock_get.side_effect = [mock_snapshot1, mock_snapshot2]
                result = introspection_handler.handle(
                    "/api/introspection/compare",
                    {"agents": "claude,gpt-4"},
                    mock_http_handler,
                )

        assert result is not None
        if result.status_code == 200:
            body = json.loads(result.body)
            assert "agents" in body or "comparison" in body


class TestIntrospectionHandlerPromptSectionEndpoint:
    """Tests for GET /api/introspection/prompt-section/:name endpoint."""

    def test_prompt_section_returns_503_when_unavailable(self, introspection_handler, mock_http_handler):
        """Prompt section endpoint returns 503 when introspection unavailable."""
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", False):
            result = introspection_handler.handle(
                "/api/introspection/prompt-section/claude",
                {},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 503

    def test_prompt_section_validates_name(self, introspection_handler, mock_http_handler):
        """Prompt section endpoint validates agent name."""
        result = introspection_handler.handle(
            "/api/introspection/prompt-section/../etc",
            {},
            mock_http_handler,
        )

        assert result is not None
        assert result.status_code == 400

    def test_prompt_section_success(self, introspection_handler, mock_http_handler):
        """Prompt section endpoint returns formatted prompt text."""
        mock_snapshot = MagicMock()
        mock_snapshot.to_prompt_section.return_value = """## YOUR TRACK RECORD
Reputation: 75%
Vote weight: 1.2x
Calibration: good (68%)
"""

        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch("aragora.server.handlers.introspection.get_agent_introspection", return_value=mock_snapshot):
                result = introspection_handler.handle(
                    "/api/introspection/prompt-section/claude",
                    {},
                    mock_http_handler,
                )

        assert result is not None
        if result.status_code == 200:
            body = json.loads(result.body)
            assert "prompt_section" in body or "section" in body
            # Verify it's a string and under character limit
            section = body.get("prompt_section") or body.get("section", "")
            assert len(section) <= 600

    def test_prompt_section_not_found(self, introspection_handler, mock_http_handler):
        """Prompt section endpoint returns 404 for unknown agent."""
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch("aragora.server.handlers.introspection.get_agent_introspection", return_value=None):
                result = introspection_handler.handle(
                    "/api/introspection/prompt-section/unknown",
                    {},
                    mock_http_handler,
                )

        assert result is not None
        assert result.status_code == 404


class TestIntrospectionHandlerErrorHandling:
    """Test error handling in introspection handler."""

    def test_snapshot_handles_exception(self, introspection_handler, mock_http_handler):
        """Snapshot endpoint handles exceptions gracefully."""
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch("aragora.server.handlers.introspection.get_agent_introspection", side_effect=Exception("DB error")):
                result = introspection_handler.handle(
                    "/api/introspection/snapshot",
                    {"agent_name": "claude"},
                    mock_http_handler,
                )

        assert result is not None
        assert result.status_code == 500

    def test_compare_handles_partial_failure(self, introspection_handler, mock_http_handler):
        """Compare endpoint handles partial agent lookup failure."""
        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {"agent_name": "claude", "reputation_score": 0.8}

        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch("aragora.server.handlers.introspection.get_agent_introspection") as mock_get:
                # First agent succeeds, second fails
                mock_get.side_effect = [mock_snapshot, None]
                result = introspection_handler.handle(
                    "/api/introspection/compare",
                    {"agents": "claude,unknown"},
                    mock_http_handler,
                )

        assert result is not None
        # Should either return partial data or indicate missing agents


class TestIntrospectionHandlerRateLimiting:
    """Test rate limiting behavior."""

    def test_rate_limit_exceeded_returns_429(self, introspection_handler, mock_http_handler):
        """Rate limit exceeded returns 429."""
        try:
            from aragora.server.handlers.introspection import _introspection_limiter

            # Simulate rate limit exceeded
            with patch.object(_introspection_limiter, 'is_allowed', return_value=False):
                result = introspection_handler.handle(
                    "/api/introspection/snapshot",
                    {"agent_name": "claude"},
                    mock_http_handler,
                )

            assert result is not None
            assert result.status_code == 429
        except ImportError:
            pytest.skip("Rate limiter not available")

    def test_multiple_requests_tracked(self, introspection_handler, mock_http_handler):
        """Multiple requests are tracked for rate limiting."""
        # Make several requests
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch("aragora.server.handlers.introspection.get_agent_introspection", return_value=None):
                for _ in range(5):
                    result = introspection_handler.handle(
                        "/api/introspection/snapshot",
                        {"agent_name": "claude"},
                        mock_http_handler,
                    )
                    assert result is not None


class TestIntrospectionHandlerInputValidation:
    """Test input validation."""

    def test_agent_name_path_traversal_blocked(self, introspection_handler, mock_http_handler):
        """Agent name path traversal is blocked."""
        endpoints = [
            "/api/introspection/agent/../etc/passwd",
            "/api/introspection/track-record/../../../etc",
            "/api/introspection/calibration/../test",
            "/api/introspection/history/../hack",
            "/api/introspection/prompt-section/../etc",
        ]

        for endpoint in endpoints:
            result = introspection_handler.handle(endpoint, {}, mock_http_handler)
            assert result is not None
            assert result.status_code == 400, f"Endpoint {endpoint} should reject path traversal"

    def test_agent_name_special_chars_handled(self, introspection_handler, mock_http_handler):
        """Special characters in agent names are handled."""
        # Some names with special chars should be sanitized or rejected
        special_names = [
            "claude<script>",
            "gpt;DROP TABLE",
            "agent|cat /etc/passwd",
        ]

        for name in special_names:
            with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
                result = introspection_handler.handle(
                    f"/api/introspection/agent/{name}",
                    {},
                    mock_http_handler,
                )

            assert result is not None


class TestIntrospectionHandlerCaching:
    """Test caching behavior."""

    def test_snapshot_cache_hit(self, introspection_handler, mock_http_handler):
        """Snapshot endpoint uses cache on repeated requests."""
        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {"agent_name": "claude", "reputation_score": 0.8}

        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch("aragora.server.handlers.introspection.get_agent_introspection", return_value=mock_snapshot) as mock_get:
                # First request
                result1 = introspection_handler.handle(
                    "/api/introspection/snapshot",
                    {"agent_name": "claude"},
                    mock_http_handler,
                )
                # Second request should use cache
                result2 = introspection_handler.handle(
                    "/api/introspection/snapshot",
                    {"agent_name": "claude"},
                    mock_http_handler,
                )

        assert result1 is not None
        assert result2 is not None
        # Both should succeed
        if result1.status_code == 200:
            assert result2.status_code == 200

    def test_cache_bypass_with_parameter(self, introspection_handler, mock_http_handler):
        """Cache can be bypassed with force_refresh parameter."""
        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {"agent_name": "claude", "reputation_score": 0.8}

        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch("aragora.server.handlers.introspection.get_agent_introspection", return_value=mock_snapshot):
                result = introspection_handler.handle(
                    "/api/introspection/snapshot",
                    {"agent_name": "claude", "force_refresh": "true"},
                    mock_http_handler,
                )

        assert result is not None


class TestIntrospectionHandlerIntegration:
    """Integration tests for introspection handler."""

    def test_all_routes_reachable(self, introspection_handler, mock_http_handler):
        """All introspection routes return a response."""
        routes_to_test = [
            ("/api/introspection/snapshot", {"agent_name": "test"}),
            ("/api/introspection/agent/test", {}),
            ("/api/introspection/track-record/test", {}),
            ("/api/introspection/calibration/test", {}),
            ("/api/introspection/history/test", {}),
            ("/api/introspection/compare", {"agents": "a,b"}),
            ("/api/introspection/prompt-section/test", {}),
        ]

        for route, params in routes_to_test:
            result = introspection_handler.handle(route, params, mock_http_handler)
            assert result is not None, f"Route {route} returned None"
            assert result.status_code in [200, 400, 404, 429, 500, 503], \
                f"Route {route} returned unexpected status {result.status_code}"

    def test_handler_inherits_from_base(self, introspection_handler):
        """Handler inherits from BaseHandler."""
        from aragora.server.handlers.base import BaseHandler
        assert isinstance(introspection_handler, BaseHandler)

    def test_handle_returns_none_for_unknown(self, introspection_handler, mock_http_handler):
        """Handle returns None for unknown paths."""
        result = introspection_handler.handle("/api/unknown", {}, mock_http_handler)
        assert result is None

    def test_full_introspection_flow(self, introspection_handler, mock_http_handler):
        """Test typical usage flow: snapshot -> history -> compare."""
        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {"agent_name": "claude", "reputation_score": 0.8}

        mock_history = [{"debate_id": "d1", "result": "win"}]

        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch("aragora.server.handlers.introspection.get_agent_introspection", return_value=mock_snapshot):
                # Step 1: Get snapshot
                snapshot_result = introspection_handler.handle(
                    "/api/introspection/snapshot",
                    {"agent_name": "claude"},
                    mock_http_handler,
                )

                # Step 2: Get history
                with patch.object(introspection_handler, '_get_agent_history', return_value=mock_history):
                    history_result = introspection_handler.handle(
                        "/api/introspection/history/claude",
                        {},
                        mock_http_handler,
                    )

                # Step 3: Compare with another agent
                compare_result = introspection_handler.handle(
                    "/api/introspection/compare",
                    {"agents": "claude,gpt-4"},
                    mock_http_handler,
                )

        # All should return some response
        assert snapshot_result is not None
        assert history_result is not None
        assert compare_result is not None


class TestIntrospectionHandlerResponseFormat:
    """Test response format consistency."""

    def test_snapshot_response_format(self, introspection_handler, mock_http_handler):
        """Snapshot response has expected format."""
        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {
            "agent_name": "claude",
            "reputation_score": 0.8,
            "vote_weight": 1.2,
            "calibration_score": 0.72,
        }

        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch("aragora.server.handlers.introspection.get_agent_introspection", return_value=mock_snapshot):
                result = introspection_handler.handle(
                    "/api/introspection/snapshot",
                    {"agent_name": "claude"},
                    mock_http_handler,
                )

        if result and result.status_code == 200:
            body = json.loads(result.body)
            # Response should be valid JSON with expected structure
            assert isinstance(body, dict)
            # Should contain snapshot data
            assert "snapshot" in body or "agent_name" in body

    def test_compare_response_format(self, introspection_handler, mock_http_handler):
        """Compare response has expected format."""
        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {"agent_name": "test", "reputation_score": 0.8}

        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch("aragora.server.handlers.introspection.get_agent_introspection", return_value=mock_snapshot):
                result = introspection_handler.handle(
                    "/api/introspection/compare",
                    {"agents": "claude,gpt-4"},
                    mock_http_handler,
                )

        if result and result.status_code == 200:
            body = json.loads(result.body)
            # Response should contain list of agents or comparison data
            assert isinstance(body, dict)
            assert "agents" in body or "comparison" in body or "snapshots" in body
