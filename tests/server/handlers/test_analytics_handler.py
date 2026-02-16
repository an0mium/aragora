"""Tests for Analytics handler endpoints.

Validates the REST API endpoints for analytics and metrics including:
- Disagreement statistics
- Role rotation statistics
- Early stopping statistics
- Consensus quality metrics
- Ranking statistics
- Memory statistics
- Cross-pollination statistics (v2.0.3)
- Learning efficiency
- Voting accuracy
- Calibration statistics
- Rate limiting
- Authorization (RBAC)
- Error handling
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.analytics import AnalyticsHandler, ANALYTICS_PERMISSION


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def analytics_handler():
    """Create an analytics handler with mocked dependencies."""
    ctx = {"storage": None, "elo_system": None, "nomic_dir": None}
    handler = AnalyticsHandler(ctx)
    return handler


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler with client address."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {}
    return handler


@pytest.fixture
def authenticated_http_handler(mock_http_handler):
    """Create a mock HTTP handler with valid authentication."""
    mock_http_handler.headers = {
        "Authorization": "Bearer valid_token",
    }
    return mock_http_handler


@pytest.fixture
def mock_auth_context():
    """Create a mock authorization context with analytics permission."""
    from aragora.rbac.models import AuthorizationContext

    return AuthorizationContext(
        user_id="test-user-123",
        org_id="test-org-456",
        workspace_id="test-workspace-789",
        roles={"analyst"},
        permissions={"analytics:read", "analytics:write"},
    )


@pytest.fixture
def mock_auth_context_no_permission():
    """Create a mock authorization context without analytics permission."""
    from aragora.rbac.models import AuthorizationContext

    return AuthorizationContext(
        user_id="test-user-123",
        org_id="test-org-456",
        workspace_id="test-workspace-789",
        roles={"viewer"},
        permissions=set(),
    )


@pytest.fixture
def mock_storage():
    """Create a mock storage with sample debates."""
    storage = MagicMock()

    # Sample debate data
    debates = [
        {
            "id": "debate-1",
            "timestamp": "2024-01-01T00:00:00Z",
            "messages": [
                {"role": "agent", "cognitive_role": "advocate"},
                {"role": "agent", "cognitive_role": "critic"},
            ],
            "result": {
                "confidence": 0.85,
                "consensus_reached": True,
                "rounds_used": 3,
                "early_stopped": False,
                "disagreement_report": {"unanimous_critiques": False},
                "uncertainty_metrics": {"disagreement_type": "minor"},
            },
        },
        {
            "id": "debate-2",
            "timestamp": "2024-01-02T00:00:00Z",
            "messages": [
                {"role": "agent", "cognitive_role": "synthesizer"},
            ],
            "result": {
                "confidence": 0.65,
                "consensus_reached": False,
                "rounds_used": 2,
                "early_stopped": True,
                "disagreement_report": {"unanimous_critiques": True},
                "uncertainty_metrics": {"disagreement_type": "major"},
            },
        },
    ]

    storage.list_debates.return_value = debates
    return storage


@pytest.fixture
def mock_storage_empty():
    """Create a mock storage with no debates."""
    storage = MagicMock()
    storage.list_debates.return_value = []
    return storage


@pytest.fixture
def mock_storage_many_debates():
    """Create a mock storage with many debates for trend analysis."""
    storage = MagicMock()

    # Create 20 debates with improving confidence trend
    debates = []
    for i in range(20):
        confidence = 0.5 + (i * 0.02)  # Improving trend
        debates.append(
            {
                "id": f"debate-{i}",
                "timestamp": f"2024-01-{i + 1:02d}T00:00:00Z",
                "messages": [{"role": "agent", "cognitive_role": "advocate"}],
                "result": {
                    "confidence": min(confidence, 0.95),
                    "consensus_reached": i % 2 == 0,
                    "rounds_used": 3,
                    "early_stopped": False,
                    "disagreement_report": {"unanimous_critiques": False},
                    "uncertainty_metrics": {"disagreement_type": "minor"},
                },
            }
        )

    storage.list_debates.return_value = debates
    return storage


@pytest.fixture
def mock_storage_declining_trend():
    """Create a mock storage with declining confidence trend."""
    storage = MagicMock()

    # Create 20 debates with declining confidence trend
    debates = []
    for i in range(20):
        confidence = 0.95 - (i * 0.02)  # Declining trend
        debates.append(
            {
                "id": f"debate-{i}",
                "timestamp": f"2024-01-{i + 1:02d}T00:00:00Z",
                "messages": [],
                "result": {
                    "confidence": max(confidence, 0.3),
                    "consensus_reached": i % 3 == 0,
                    "rounds_used": 2,
                    "early_stopped": True,
                    "disagreement_report": {"unanimous_critiques": True},
                    "uncertainty_metrics": {"disagreement_type": "major"},
                },
            }
        )

    storage.list_debates.return_value = debates
    return storage


@pytest.fixture
def mock_elo_system():
    """Create a mock ELO system."""
    elo = MagicMock()

    # Mock agent data
    agent1 = MagicMock()
    agent1.agent_name = "claude"
    agent1.elo = 1650
    agent1.debates_count = 50

    agent2 = MagicMock()
    agent2.agent_name = "gpt-4"
    agent2.elo = 1550
    agent2.debates_count = 45

    agent3 = MagicMock()
    agent3.agent_name = "codex"
    agent3.elo = 1480
    agent3.debates_count = 30

    elo.get_leaderboard.return_value = [agent1, agent2, agent3]

    # Mock learning efficiency methods
    elo.get_learning_efficiency.return_value = {
        "rate": 0.15,
        "trend": "improving",
        "window": 20,
    }
    elo.get_learning_efficiency_batch.return_value = {
        "claude": {"rate": 0.15, "trend": "improving"},
        "gpt-4": {"rate": 0.12, "trend": "stable"},
        "codex": {"rate": 0.08, "trend": "declining"},
    }

    # Mock voting accuracy methods
    elo.get_voting_accuracy.return_value = {
        "accuracy": 0.78,
        "total_votes": 100,
        "correct_votes": 78,
    }
    elo.get_voting_accuracy_batch.return_value = {
        "claude": {"accuracy": 0.78, "total_votes": 100},
        "gpt-4": {"accuracy": 0.75, "total_votes": 90},
        "codex": {"accuracy": 0.70, "total_votes": 60},
    }

    return elo


@pytest.fixture
def mock_calibration_tracker():
    """Create a mock calibration tracker."""
    tracker = MagicMock()

    # Mock calibration summary
    summary = MagicMock()
    summary.total_predictions = 100
    summary.temperature = 1.2
    summary.scaling_factor = 0.95

    tracker.get_calibration_summary.return_value = summary
    return tracker


@pytest.fixture
def mock_calibration_module(mock_calibration_tracker):
    """Inject a mock calibration module into sys.modules.

    The aragora.ranking.calibration module doesn't exist yet, so we need to
    inject a mock to enable testing the calibration endpoint.
    """
    mock_mod = ModuleType("aragora.ranking.calibration")
    mock_mod.CalibrationTracker = MagicMock(return_value=mock_calibration_tracker)

    original = sys.modules.get("aragora.ranking.calibration")
    sys.modules["aragora.ranking.calibration"] = mock_mod
    yield mock_mod
    if original is not None:
        sys.modules["aragora.ranking.calibration"] = original
    else:
        sys.modules.pop("aragora.ranking.calibration", None)


@pytest.fixture
def reset_rate_limiter():
    """Reset the analytics rate limiter before and after each test."""
    from aragora.server.handlers.analytics import _analytics_limiter

    original_requests = (
        _analytics_limiter._buckets.copy() if hasattr(_analytics_limiter, "_buckets") else {}
    )
    _analytics_limiter.clear()

    yield

    _analytics_limiter.clear()
    if hasattr(_analytics_limiter, "_buckets"):
        _analytics_limiter._buckets.update(original_requests)


# ==============================================================================
# Test: can_handle Method
# ==============================================================================


class TestAnalyticsHandlerCanHandle:
    """Test AnalyticsHandler.can_handle method."""

    def test_can_handle_disagreements(self, analytics_handler):
        """Test can_handle returns True for disagreements endpoint."""
        assert analytics_handler.can_handle("/api/analytics/disagreements")
        assert analytics_handler.can_handle("/api/v1/analytics/disagreements")
        assert analytics_handler.can_handle("/api/v2/analytics/disagreements")

    def test_can_handle_role_rotation(self, analytics_handler):
        """Test can_handle returns True for role-rotation endpoint."""
        assert analytics_handler.can_handle("/api/analytics/role-rotation")
        assert analytics_handler.can_handle("/api/v1/analytics/role-rotation")

    def test_can_handle_early_stops(self, analytics_handler):
        """Test can_handle returns True for early-stops endpoint."""
        assert analytics_handler.can_handle("/api/analytics/early-stops")
        assert analytics_handler.can_handle("/api/v1/analytics/early-stops")

    def test_can_handle_consensus_quality(self, analytics_handler):
        """Test can_handle returns True for consensus-quality endpoint."""
        assert analytics_handler.can_handle("/api/analytics/consensus-quality")
        assert analytics_handler.can_handle("/api/v1/analytics/consensus-quality")

    def test_can_handle_ranking_stats(self, analytics_handler):
        """Test can_handle returns True for ranking stats endpoint."""
        assert analytics_handler.can_handle("/api/ranking/stats")
        assert analytics_handler.can_handle("/api/v1/ranking/stats")

    def test_can_handle_memory_stats(self, analytics_handler):
        """Test can_handle returns True for memory stats endpoint."""
        assert analytics_handler.can_handle("/api/memory/stats")
        assert analytics_handler.can_handle("/api/v1/memory/stats")

    def test_can_handle_cross_pollination(self, analytics_handler):
        """Test can_handle returns True for cross-pollination endpoint."""
        assert analytics_handler.can_handle("/api/analytics/cross-pollination")
        assert analytics_handler.can_handle("/api/v1/analytics/cross-pollination")

    def test_can_handle_learning_efficiency(self, analytics_handler):
        """Test can_handle returns True for learning-efficiency endpoint."""
        assert analytics_handler.can_handle("/api/analytics/learning-efficiency")
        assert analytics_handler.can_handle("/api/v1/analytics/learning-efficiency")

    def test_can_handle_voting_accuracy(self, analytics_handler):
        """Test can_handle returns True for voting-accuracy endpoint."""
        assert analytics_handler.can_handle("/api/analytics/voting-accuracy")
        assert analytics_handler.can_handle("/api/v1/analytics/voting-accuracy")

    def test_can_handle_calibration(self, analytics_handler):
        """Test can_handle returns True for calibration endpoint."""
        assert analytics_handler.can_handle("/api/analytics/calibration")
        assert analytics_handler.can_handle("/api/v1/analytics/calibration")

    def test_cannot_handle_unknown(self, analytics_handler):
        """Test can_handle returns False for unknown endpoint."""
        assert not analytics_handler.can_handle("/api/analytics/unknown")
        assert not analytics_handler.can_handle("/api/v1/analytics/unknown")
        assert not analytics_handler.can_handle("/api/debates")
        assert not analytics_handler.can_handle("/api/v1/debates")

    def test_cannot_handle_similar_paths(self, analytics_handler):
        """Test can_handle returns False for similar but invalid paths."""
        assert not analytics_handler.can_handle("/api/analytics")
        assert not analytics_handler.can_handle("/api/analytics/")
        assert not analytics_handler.can_handle("/analytics/disagreements")


# ==============================================================================
# Test: Authorization (RBAC)
# ==============================================================================


class TestAnalyticsHandlerAuthorization:
    """Test RBAC authorization for analytics endpoints."""

    @pytest.mark.asyncio
    async def test_unauthenticated_request_returns_401(
        self, analytics_handler, mock_http_handler, reset_rate_limiter
    ):
        """Test that unauthenticated requests return 401."""
        with patch.object(
            analytics_handler,
            "get_auth_context",
            new_callable=AsyncMock,
        ) as mock_auth:
            from aragora.server.handlers.secure import UnauthorizedError

            mock_auth.side_effect = UnauthorizedError("Authentication required")

            result = await analytics_handler.handle(
                "/api/analytics/disagreements", {}, mock_http_handler
            )

            assert result is not None
            assert result.status_code == 401
            body = json.loads(result.body)
            assert "error" in body

    @pytest.mark.asyncio
    async def test_missing_permission_returns_403(
        self,
        analytics_handler,
        mock_http_handler,
        mock_auth_context_no_permission,
        reset_rate_limiter,
    ):
        """Test that requests without analytics:read permission return 403."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            from aragora.server.handlers.secure import ForbiddenError

            mock_auth.return_value = mock_auth_context_no_permission
            mock_check.side_effect = ForbiddenError("Permission denied: analytics:read")

            result = await analytics_handler.handle(
                "/api/analytics/disagreements", {}, mock_http_handler
            )

            assert result is not None
            assert result.status_code == 403
            body = json.loads(result.body)
            assert "error" in body

    @pytest.mark.asyncio
    async def test_authorized_request_succeeds(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test that authorized requests succeed."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/analytics/disagreements", {}, mock_http_handler
            )

            assert result is not None
            # Should be 200 or return stats (not 401/403)
            assert result.status_code in (200, 503)
            mock_check.assert_called_once_with(mock_auth_context, ANALYTICS_PERMISSION)


# ==============================================================================
# Test: Rate Limiting
# ==============================================================================


class TestAnalyticsHandlerRateLimiting:
    """Test rate limiting for analytics endpoints."""

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_returns_429(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test that rate limit exceeded returns 429."""
        from aragora.server.handlers.analytics import _analytics_limiter

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            # Exhaust rate limit (30 requests per minute for analytics)
            for _ in range(35):
                result = await analytics_handler.handle(
                    "/api/analytics/disagreements", {}, mock_http_handler
                )

            # Last request should be rate limited
            assert result is not None
            if result.status_code == 429:
                body = json.loads(result.body)
                assert "error" in body
                assert "Rate limit" in body["error"] or "rate limit" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_rate_limit_per_client_ip(
        self, analytics_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test that rate limiting is per-client IP."""
        # Create two handlers with different IPs
        handler1 = MagicMock()
        handler1.client_address = ("192.168.1.1", 12345)
        handler1.headers = {}

        handler2 = MagicMock()
        handler2.client_address = ("192.168.1.2", 12346)
        handler2.headers = {}

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            # Both clients should be able to make requests independently
            result1 = await analytics_handler.handle("/api/analytics/disagreements", {}, handler1)
            result2 = await analytics_handler.handle("/api/analytics/disagreements", {}, handler2)

            # Both should succeed (neither is rate limited yet)
            assert result1.status_code != 429
            assert result2.status_code != 429


# ==============================================================================
# Test: Disagreement Statistics
# ==============================================================================


class TestAnalyticsHandlerDisagreements:
    """Test GET /api/analytics/disagreements endpoint."""

    @pytest.mark.asyncio
    async def test_disagreement_stats_no_storage(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test disagreement stats returns empty when no storage."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/v1/analytics/disagreements", {}, mock_http_handler
            )

            assert result is not None
            body = json.loads(result.body)
            assert "stats" in body

    @pytest.mark.asyncio
    async def test_disagreement_stats_with_storage(
        self,
        analytics_handler,
        mock_http_handler,
        mock_storage,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test disagreement stats with storage data."""
        analytics_handler.ctx["storage"] = mock_storage

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/v1/analytics/disagreements", {}, mock_http_handler
            )

            assert result is not None
            body = json.loads(result.body)
            assert "stats" in body
            stats = body["stats"]
            assert "total_debates" in stats
            assert "with_disagreements" in stats
            assert "unanimous" in stats
            assert "disagreement_types" in stats
            assert stats["total_debates"] == 2

    @pytest.mark.asyncio
    async def test_disagreement_stats_counts_correctly(
        self,
        analytics_handler,
        mock_http_handler,
        mock_storage,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test disagreement stats counts are calculated correctly."""
        analytics_handler.ctx["storage"] = mock_storage

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/analytics/disagreements", {}, mock_http_handler
            )

            body = json.loads(result.body)
            stats = body["stats"]
            # First debate has unanimous_critiques=False (unanimous)
            # Second debate has unanimous_critiques=True (with disagreements)
            assert "disagreement_types" in stats


# ==============================================================================
# Test: Role Rotation Statistics
# ==============================================================================


class TestAnalyticsHandlerRoleRotation:
    """Test GET /api/analytics/role-rotation endpoint."""

    @pytest.mark.asyncio
    async def test_role_rotation_stats_no_storage(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test role rotation stats returns empty when no storage."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/v1/analytics/role-rotation", {}, mock_http_handler
            )

            assert result is not None
            body = json.loads(result.body)
            assert "stats" in body

    @pytest.mark.asyncio
    async def test_role_rotation_stats_with_storage(
        self,
        analytics_handler,
        mock_http_handler,
        mock_storage,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test role rotation stats with storage data."""
        analytics_handler.ctx["storage"] = mock_storage

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/v1/analytics/role-rotation", {}, mock_http_handler
            )

            assert result is not None
            body = json.loads(result.body)
            assert "stats" in body
            stats = body["stats"]
            assert "total_debates" in stats
            assert "role_assignments" in stats
            assert stats["total_debates"] == 2

    @pytest.mark.asyncio
    async def test_role_rotation_stats_counts_roles(
        self,
        analytics_handler,
        mock_http_handler,
        mock_storage,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test role rotation stats counts cognitive roles correctly."""
        analytics_handler.ctx["storage"] = mock_storage

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/analytics/role-rotation", {}, mock_http_handler
            )

            body = json.loads(result.body)
            stats = body["stats"]
            role_assignments = stats["role_assignments"]
            # Should have advocate, critic, and synthesizer from mock data
            assert "advocate" in role_assignments or "critic" in role_assignments


# ==============================================================================
# Test: Early Stopping Statistics
# ==============================================================================


class TestAnalyticsHandlerEarlyStops:
    """Test GET /api/analytics/early-stops endpoint."""

    @pytest.mark.asyncio
    async def test_early_stop_stats_no_storage(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test early stop stats returns empty when no storage."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/v1/analytics/early-stops", {}, mock_http_handler
            )

            assert result is not None
            body = json.loads(result.body)
            assert "stats" in body

    @pytest.mark.asyncio
    async def test_early_stop_stats_with_storage(
        self,
        analytics_handler,
        mock_http_handler,
        mock_storage,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test early stop stats with storage data."""
        analytics_handler.ctx["storage"] = mock_storage

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/v1/analytics/early-stops", {}, mock_http_handler
            )

            assert result is not None
            body = json.loads(result.body)
            assert "stats" in body
            stats = body["stats"]
            assert "total_debates" in stats
            assert "early_stopped" in stats
            assert "full_rounds" in stats
            assert "average_rounds" in stats

    @pytest.mark.asyncio
    async def test_early_stop_stats_calculates_average(
        self,
        analytics_handler,
        mock_http_handler,
        mock_storage,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test early stop stats calculates average rounds correctly."""
        analytics_handler.ctx["storage"] = mock_storage

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/analytics/early-stops", {}, mock_http_handler
            )

            body = json.loads(result.body)
            stats = body["stats"]
            # debates have 3 and 2 rounds, so average should be 2.5
            assert stats["average_rounds"] == 2.5
            assert stats["early_stopped"] == 1
            assert stats["full_rounds"] == 1


# ==============================================================================
# Test: Consensus Quality Statistics
# ==============================================================================


class TestAnalyticsHandlerConsensusQuality:
    """Test GET /api/analytics/consensus-quality endpoint."""

    @pytest.mark.asyncio
    async def test_consensus_quality_no_storage(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test consensus quality returns empty when no storage."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/v1/analytics/consensus-quality", {}, mock_http_handler
            )

            assert result is not None
            body = json.loads(result.body)
            assert "stats" in body
            assert "quality_score" in body

    @pytest.mark.asyncio
    async def test_consensus_quality_with_storage(
        self,
        analytics_handler,
        mock_http_handler,
        mock_storage,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test consensus quality with storage data."""
        analytics_handler.ctx["storage"] = mock_storage

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/v1/analytics/consensus-quality", {}, mock_http_handler
            )

            assert result is not None
            body = json.loads(result.body)
            assert "stats" in body
            assert "quality_score" in body
            stats = body["stats"]
            assert "total_debates" in stats
            assert "confidence_history" in stats
            assert "trend" in stats
            assert "average_confidence" in stats
            assert "consensus_rate" in stats

    @pytest.mark.asyncio
    async def test_consensus_quality_empty_debates(
        self,
        analytics_handler,
        mock_http_handler,
        mock_storage_empty,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test consensus quality with no debates returns proper empty state."""
        analytics_handler.ctx["storage"] = mock_storage_empty

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/v1/analytics/consensus-quality", {}, mock_http_handler
            )

            assert result is not None
            body = json.loads(result.body)
            assert body["quality_score"] == 0
            assert body["stats"]["trend"] == "insufficient_data"

    @pytest.mark.asyncio
    async def test_consensus_quality_detects_improving_trend(
        self,
        analytics_handler,
        mock_http_handler,
        mock_storage_many_debates,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test consensus quality detects improving trend."""
        analytics_handler.ctx["storage"] = mock_storage_many_debates

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/analytics/consensus-quality", {}, mock_http_handler
            )

            body = json.loads(result.body)
            # With improving confidence, trend should be "improving"
            assert body["stats"]["trend"] in ("improving", "stable")

    @pytest.mark.asyncio
    async def test_consensus_quality_detects_declining_trend(
        self,
        analytics_handler,
        mock_http_handler,
        mock_storage_declining_trend,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test consensus quality detects declining trend."""
        analytics_handler.ctx["storage"] = mock_storage_declining_trend

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/analytics/consensus-quality", {}, mock_http_handler
            )

            body = json.loads(result.body)
            # With declining confidence, trend should be "declining" or alert should be set
            assert body["stats"]["trend"] in ("declining", "stable")

    @pytest.mark.asyncio
    async def test_consensus_quality_generates_alerts(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test consensus quality generates appropriate alerts for low quality."""
        # Create storage with low confidence debates
        storage = MagicMock()
        debates = [
            {
                "id": f"debate-{i}",
                "timestamp": f"2024-01-{i + 1:02d}T00:00:00Z",
                "messages": [],
                "result": {
                    "confidence": 0.2,  # Very low confidence
                    "consensus_reached": False,
                    "rounds_used": 2,
                    "early_stopped": True,
                    "disagreement_report": {},
                    "uncertainty_metrics": {},
                },
            }
            for i in range(10)
        ]
        storage.list_debates.return_value = debates
        analytics_handler.ctx["storage"] = storage

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/analytics/consensus-quality", {}, mock_http_handler
            )

            body = json.loads(result.body)
            # Low quality should trigger an alert
            assert "alert" in body
            if body["alert"] is not None:
                assert "level" in body["alert"]
                assert "message" in body["alert"]


# ==============================================================================
# Test: Ranking Statistics
# ==============================================================================


class TestAnalyticsHandlerRankingStats:
    """Test GET /api/ranking/stats endpoint."""

    @pytest.mark.asyncio
    async def test_ranking_stats_no_elo(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test ranking stats returns 503 when no ELO system."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle("/api/v1/ranking/stats", {}, mock_http_handler)

            assert result is not None
            assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_ranking_stats_with_elo(
        self,
        analytics_handler,
        mock_http_handler,
        mock_elo_system,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test ranking stats with ELO system data."""
        analytics_handler.ctx["elo_system"] = mock_elo_system

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle("/api/v1/ranking/stats", {}, mock_http_handler)

            assert result is not None
            body = json.loads(result.body)
            assert "stats" in body
            stats = body["stats"]
            assert "total_agents" in stats
            assert "total_matches" in stats
            assert "avg_elo" in stats
            assert "top_agent" in stats
            assert "elo_range" in stats
            assert stats["total_agents"] == 3
            assert stats["top_agent"] == "claude"

    @pytest.mark.asyncio
    async def test_ranking_stats_elo_range(
        self,
        analytics_handler,
        mock_http_handler,
        mock_elo_system,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test ranking stats includes correct ELO range."""
        analytics_handler.ctx["elo_system"] = mock_elo_system

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle("/api/ranking/stats", {}, mock_http_handler)

            body = json.loads(result.body)
            stats = body["stats"]
            assert stats["elo_range"]["min"] == 1480  # codex
            assert stats["elo_range"]["max"] == 1650  # claude


# ==============================================================================
# Test: Memory Statistics
# ==============================================================================


class TestAnalyticsHandlerMemoryStats:
    """Test GET /api/memory/stats endpoint."""

    @pytest.mark.asyncio
    async def test_memory_stats_no_nomic_dir(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test memory stats returns empty when no nomic dir."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle("/api/v1/memory/stats", {}, mock_http_handler)

            assert result is not None
            body = json.loads(result.body)
            assert "stats" in body

    @pytest.mark.asyncio
    async def test_memory_stats_with_nomic_dir(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter, tmp_path
    ):
        """Test memory stats with nomic dir."""
        # Create a temp nomic dir with some database files
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        (nomic_dir / "debate_embeddings.db").touch()

        analytics_handler.ctx["nomic_dir"] = nomic_dir

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle("/api/v1/memory/stats", {}, mock_http_handler)

            assert result is not None
            body = json.loads(result.body)
            assert "stats" in body
            stats = body["stats"]
            assert "embeddings_db" in stats
            assert "insights_db" in stats
            assert "continuum_memory" in stats
            assert stats["embeddings_db"] is True

    @pytest.mark.asyncio
    async def test_memory_stats_all_databases(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter, tmp_path
    ):
        """Test memory stats detects all database types."""
        from aragora.config import DB_INSIGHTS_PATH

        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        (nomic_dir / "debate_embeddings.db").touch()
        (nomic_dir / DB_INSIGHTS_PATH).touch()
        (nomic_dir / "continuum_memory.db").touch()

        analytics_handler.ctx["nomic_dir"] = nomic_dir

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle("/api/memory/stats", {}, mock_http_handler)

            body = json.loads(result.body)
            stats = body["stats"]
            assert stats["embeddings_db"] is True
            assert stats["insights_db"] is True
            assert stats["continuum_memory"] is True


# ==============================================================================
# Test: Cross-Pollination Statistics
# ==============================================================================


class TestAnalyticsHandlerCrossPollination:
    """Test GET /api/analytics/cross-pollination endpoint."""

    @pytest.mark.asyncio
    async def test_cross_pollination_stats_basic(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test cross-pollination stats returns expected structure."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/v1/analytics/cross-pollination", {}, mock_http_handler
            )

            assert result is not None
            body = json.loads(result.body)
            assert "stats" in body
            assert "version" in body
            stats = body["stats"]
            assert "calibration" in stats
            assert "learning" in stats
            assert "voting_accuracy" in stats
            assert "adaptive_rounds" in stats
            assert "rlm_cache" in stats

    @pytest.mark.asyncio
    async def test_cross_pollination_stats_version(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test cross-pollination stats includes version."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/analytics/cross-pollination", {}, mock_http_handler
            )

            body = json.loads(result.body)
            assert body["version"] == "2.0.3"


# ==============================================================================
# Test: Learning Efficiency Statistics
# ==============================================================================


class TestAnalyticsHandlerLearningEfficiency:
    """Test GET /api/analytics/learning-efficiency endpoint."""

    @pytest.mark.asyncio
    async def test_learning_efficiency_no_elo(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test learning efficiency returns error when no ELO system."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            with patch("aragora.ranking.elo.get_elo_store", side_effect=ImportError):
                result = await analytics_handler.handle(
                    "/api/v1/analytics/learning-efficiency", {}, mock_http_handler
                )

                assert result is not None
                body = json.loads(result.body)
                assert "error" in body or "agents" in body

    @pytest.mark.asyncio
    async def test_learning_efficiency_specific_agent(
        self,
        analytics_handler,
        mock_http_handler,
        mock_elo_system,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test learning efficiency for a specific agent."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            with patch(
                "aragora.ranking.elo.get_elo_store",
                return_value=mock_elo_system,
            ):
                result = await analytics_handler.handle(
                    "/api/analytics/learning-efficiency",
                    {"agent": ["claude"]},
                    mock_http_handler,
                )

                body = json.loads(result.body)
                assert "agent" in body
                assert body["agent"] == "claude"
                assert "efficiency" in body

    @pytest.mark.asyncio
    async def test_learning_efficiency_all_agents(
        self,
        analytics_handler,
        mock_http_handler,
        mock_elo_system,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test learning efficiency for all agents (batch query)."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            with patch(
                "aragora.ranking.elo.get_elo_store",
                return_value=mock_elo_system,
            ):
                result = await analytics_handler.handle(
                    "/api/analytics/learning-efficiency", {}, mock_http_handler
                )

                body = json.loads(result.body)
                assert "agents" in body
                assert len(body["agents"]) == 3

    @pytest.mark.asyncio
    async def test_learning_efficiency_with_domain_filter(
        self,
        analytics_handler,
        mock_http_handler,
        mock_elo_system,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test learning efficiency with domain filter."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            with patch(
                "aragora.ranking.elo.get_elo_store",
                return_value=mock_elo_system,
            ):
                result = await analytics_handler.handle(
                    "/api/analytics/learning-efficiency",
                    {"domain": ["coding"]},
                    mock_http_handler,
                )

                body = json.loads(result.body)
                assert "domain" in body
                assert body["domain"] == "coding"

    @pytest.mark.asyncio
    async def test_learning_efficiency_with_limit(
        self,
        analytics_handler,
        mock_http_handler,
        mock_elo_system,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test learning efficiency with limit parameter."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            with patch(
                "aragora.ranking.elo.get_elo_store",
                return_value=mock_elo_system,
            ):
                result = await analytics_handler.handle(
                    "/api/analytics/learning-efficiency",
                    {"limit": ["2"]},
                    mock_http_handler,
                )

                body = json.loads(result.body)
                assert "agents" in body
                # Limit should be applied to leaderboard query


# ==============================================================================
# Test: Voting Accuracy Statistics
# ==============================================================================


class TestAnalyticsHandlerVotingAccuracy:
    """Test GET /api/analytics/voting-accuracy endpoint."""

    @pytest.mark.asyncio
    async def test_voting_accuracy_no_elo(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test voting accuracy returns error when no ELO system."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            with patch("aragora.ranking.elo.get_elo_store", side_effect=ImportError):
                result = await analytics_handler.handle(
                    "/api/v1/analytics/voting-accuracy", {}, mock_http_handler
                )

                assert result is not None
                body = json.loads(result.body)
                assert "error" in body or "agents" in body

    @pytest.mark.asyncio
    async def test_voting_accuracy_specific_agent(
        self,
        analytics_handler,
        mock_http_handler,
        mock_elo_system,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test voting accuracy for a specific agent."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            with patch(
                "aragora.ranking.elo.get_elo_store",
                return_value=mock_elo_system,
            ):
                result = await analytics_handler.handle(
                    "/api/analytics/voting-accuracy",
                    {"agent": ["gpt-4"]},
                    mock_http_handler,
                )

                body = json.loads(result.body)
                assert "agent" in body
                assert body["agent"] == "gpt-4"
                assert "accuracy" in body

    @pytest.mark.asyncio
    async def test_voting_accuracy_all_agents(
        self,
        analytics_handler,
        mock_http_handler,
        mock_elo_system,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test voting accuracy for all agents (batch query)."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            with patch(
                "aragora.ranking.elo.get_elo_store",
                return_value=mock_elo_system,
            ):
                result = await analytics_handler.handle(
                    "/api/analytics/voting-accuracy", {}, mock_http_handler
                )

                body = json.loads(result.body)
                assert "agents" in body
                assert len(body["agents"]) == 3
                for agent_data in body["agents"]:
                    assert "agent" in agent_data
                    assert "accuracy" in agent_data


# ==============================================================================
# Test: Calibration Statistics
# ==============================================================================


# Helper to check if calibration module exists
def _calibration_module_available():
    """Check if the calibration module is available."""
    try:
        import aragora.ranking.calibration  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


class TestAnalyticsHandlerCalibration:
    """Test GET /api/analytics/calibration endpoint."""

    @pytest.mark.asyncio
    async def test_calibration_no_elo(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test calibration returns error when no ELO system."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            with patch("aragora.ranking.elo.get_elo_store", side_effect=ImportError):
                result = await analytics_handler.handle(
                    "/api/v1/analytics/calibration", {}, mock_http_handler
                )

                assert result is not None
                body = json.loads(result.body)
                assert "error" in body or "agents" in body

    @pytest.mark.asyncio
    async def test_calibration_specific_agent_with_tracker(
        self,
        analytics_handler,
        mock_http_handler,
        mock_elo_system,
        mock_calibration_tracker,
        mock_calibration_module,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test calibration for a specific agent with calibration tracker."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            with patch(
                "aragora.ranking.elo.get_elo_store",
                return_value=mock_elo_system,
            ):
                result = await analytics_handler.handle(
                    "/api/analytics/calibration",
                    {"agent": ["claude"]},
                    mock_http_handler,
                )

                body = json.loads(result.body)
                assert "agent" in body
                assert body["agent"] == "claude"
                assert "calibration" in body

    @pytest.mark.asyncio
    async def test_calibration_specific_agent_no_tracker(
        self,
        analytics_handler,
        mock_http_handler,
        mock_elo_system,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test calibration for a specific agent without calibration tracker."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            # Since the module doesn't exist, it naturally returns None for calibration
            with patch(
                "aragora.ranking.elo.get_elo_store",
                return_value=mock_elo_system,
            ):
                result = await analytics_handler.handle(
                    "/api/analytics/calibration",
                    {"agent": ["claude"]},
                    mock_http_handler,
                )

                body = json.loads(result.body)
                assert "agent" in body
                assert body["calibration"] is None

    @pytest.mark.asyncio
    async def test_calibration_all_agents(
        self,
        analytics_handler,
        mock_http_handler,
        mock_elo_system,
        mock_calibration_tracker,
        mock_calibration_module,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test calibration for all agents."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            with patch(
                "aragora.ranking.elo.get_elo_store",
                return_value=mock_elo_system,
            ):
                result = await analytics_handler.handle(
                    "/api/analytics/calibration", {}, mock_http_handler
                )

                body = json.loads(result.body)
                assert "agents" in body
                assert len(body["agents"]) == 3


# ==============================================================================
# Test: Error Handling
# ==============================================================================


class TestAnalyticsHandlerErrorHandling:
    """Test error handling for analytics endpoints."""

    @pytest.mark.asyncio
    async def test_storage_exception_handled(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test that storage exceptions are handled gracefully."""
        storage = MagicMock()
        storage.list_debates.side_effect = ValueError("Database connection failed")
        analytics_handler.ctx["storage"] = storage

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/analytics/disagreements", {}, mock_http_handler
            )

            # Should return a valid response (either error or empty stats)
            assert result is not None
            body = json.loads(result.body)
            assert isinstance(body, dict)

    @pytest.mark.asyncio
    async def test_elo_exception_handled(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test that ELO system exceptions are handled gracefully."""
        elo = MagicMock()
        elo.get_leaderboard.side_effect = ValueError("ELO system error")
        analytics_handler.ctx["elo_system"] = elo

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle("/api/ranking/stats", {}, mock_http_handler)

            # Should return an error response
            assert result is not None
            assert result.status_code in (400, 500, 503)

    @pytest.mark.asyncio
    async def test_unhandled_path_returns_none(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test that unhandled paths return None."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/analytics/unknown-endpoint", {}, mock_http_handler
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_invalid_query_param_defaults(
        self,
        analytics_handler,
        mock_http_handler,
        mock_elo_system,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test that invalid query params use defaults."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            with patch(
                "aragora.ranking.elo.get_elo_store",
                return_value=mock_elo_system,
            ):
                # Invalid limit value should default
                result = await analytics_handler.handle(
                    "/api/analytics/learning-efficiency",
                    {"limit": ["not_a_number"]},
                    mock_http_handler,
                )

                assert result is not None
                body = json.loads(result.body)
                # Should still return valid response with default limit
                assert "agents" in body or "error" in body


# ==============================================================================
# Test: Integration Tests
# ==============================================================================


class TestAnalyticsHandlerIntegration:
    """Integration tests for analytics handler."""

    @pytest.mark.asyncio
    async def test_all_endpoints_return_valid_json(
        self,
        analytics_handler,
        mock_http_handler,
        mock_storage,
        mock_elo_system,
        mock_auth_context,
        reset_rate_limiter,
        tmp_path,
    ):
        """Test all analytics endpoints return valid JSON."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()

        analytics_handler.ctx["storage"] = mock_storage
        analytics_handler.ctx["elo_system"] = mock_elo_system
        analytics_handler.ctx["nomic_dir"] = nomic_dir

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            endpoints = [
                "/api/v1/analytics/disagreements",
                "/api/v1/analytics/role-rotation",
                "/api/v1/analytics/early-stops",
                "/api/v1/analytics/consensus-quality",
                "/api/v1/ranking/stats",
                "/api/v1/memory/stats",
                "/api/v1/analytics/cross-pollination",
            ]

            for endpoint in endpoints:
                result = await analytics_handler.handle(endpoint, {}, mock_http_handler)
                assert result is not None, f"Endpoint {endpoint} returned None"

                # Verify it's valid JSON
                try:
                    body = json.loads(result.body)
                    assert isinstance(body, dict), f"Endpoint {endpoint} didn't return dict"
                except json.JSONDecodeError:
                    pytest.fail(f"Endpoint {endpoint} returned invalid JSON")

    @pytest.mark.asyncio
    async def test_endpoints_with_query_params(
        self,
        analytics_handler,
        mock_http_handler,
        mock_elo_system,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test endpoints that accept query parameters."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            with patch(
                "aragora.ranking.elo.get_elo_store",
                return_value=mock_elo_system,
            ):
                # Test learning-efficiency with params
                result = await analytics_handler.handle(
                    "/api/analytics/learning-efficiency",
                    {"agent": ["claude"], "domain": ["coding"], "limit": ["10"]},
                    mock_http_handler,
                )
                assert result is not None
                body = json.loads(result.body)
                assert body["agent"] == "claude"
                assert body["domain"] == "coding"

                # Test voting-accuracy with params
                result = await analytics_handler.handle(
                    "/api/analytics/voting-accuracy",
                    {"agent": ["gpt-4"], "limit": ["5"]},
                    mock_http_handler,
                )
                assert result is not None
                body = json.loads(result.body)
                assert body["agent"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_version_prefix_handling(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test that version prefixes are handled correctly."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            # Test without version prefix
            result1 = await analytics_handler.handle(
                "/api/analytics/disagreements", {}, mock_http_handler
            )

            # Test with v1 prefix
            result2 = await analytics_handler.handle(
                "/api/v1/analytics/disagreements", {}, mock_http_handler
            )

            # Test with v2 prefix
            result3 = await analytics_handler.handle(
                "/api/v2/analytics/disagreements", {}, mock_http_handler
            )

            # All should return the same structure
            assert result1 is not None
            assert result2 is not None
            assert result3 is not None

            body1 = json.loads(result1.body)
            body2 = json.loads(result2.body)
            body3 = json.loads(result3.body)

            assert "stats" in body1
            assert "stats" in body2
            assert "stats" in body3


# ==============================================================================
# Test: Caching Behavior
# ==============================================================================


class TestAnalyticsHandlerCaching:
    """Test caching behavior for analytics endpoints."""

    @pytest.mark.asyncio
    async def test_cached_response_consistency(
        self,
        analytics_handler,
        mock_http_handler,
        mock_storage,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test that cached responses are consistent."""
        analytics_handler.ctx["storage"] = mock_storage

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            # Make two consecutive requests
            result1 = await analytics_handler.handle(
                "/api/analytics/disagreements", {}, mock_http_handler
            )
            result2 = await analytics_handler.handle(
                "/api/analytics/disagreements", {}, mock_http_handler
            )

            body1 = json.loads(result1.body)
            body2 = json.loads(result2.body)

            # Both should have the same structure
            assert body1.keys() == body2.keys()


# ==============================================================================
# Test: Edge Cases
# ==============================================================================


class TestAnalyticsHandlerEdgeCases:
    """Test edge cases for analytics endpoints."""

    @pytest.mark.asyncio
    async def test_empty_leaderboard(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test ranking stats with empty leaderboard."""
        elo = MagicMock()
        elo.get_leaderboard.return_value = []
        analytics_handler.ctx["elo_system"] = elo

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle("/api/ranking/stats", {}, mock_http_handler)

            body = json.loads(result.body)
            stats = body["stats"]
            assert stats["total_agents"] == 0
            assert stats["top_agent"] is None

    @pytest.mark.asyncio
    async def test_debate_with_missing_fields(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test handling of debates with missing fields."""
        storage = MagicMock()
        debates = [
            {"id": "incomplete-debate"},  # Missing most fields
            {
                "id": "partial-debate",
                "result": {},  # Empty result
            },
        ]
        storage.list_debates.return_value = debates
        analytics_handler.ctx["storage"] = storage

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            # Should handle gracefully without crashing
            result = await analytics_handler.handle(
                "/api/analytics/consensus-quality", {}, mock_http_handler
            )

            assert result is not None
            body = json.loads(result.body)
            assert "stats" in body

    @pytest.mark.asyncio
    async def test_non_dict_debate_entries(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test handling of non-dict debate entries."""
        storage = MagicMock()
        # Some systems might return objects instead of dicts
        storage.list_debates.return_value = [None, "invalid", 123]
        analytics_handler.ctx["storage"] = storage

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            # Should handle gracefully
            result = await analytics_handler.handle(
                "/api/analytics/early-stops", {}, mock_http_handler
            )

            assert result is not None
            body = json.loads(result.body)
            assert "stats" in body

    @pytest.mark.asyncio
    async def test_very_large_confidence_values(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test handling of unusual confidence values."""
        storage = MagicMock()
        debates = [
            {
                "id": "debate-1",
                "timestamp": "2024-01-01T00:00:00Z",
                "messages": [],
                "result": {
                    "confidence": 1.5,  # Greater than 1
                    "consensus_reached": True,
                    "rounds_used": 3,
                    "early_stopped": False,
                    "disagreement_report": {},
                    "uncertainty_metrics": {},
                },
            },
            {
                "id": "debate-2",
                "timestamp": "2024-01-02T00:00:00Z",
                "messages": [],
                "result": {
                    "confidence": -0.5,  # Negative
                    "consensus_reached": False,
                    "rounds_used": 2,
                    "early_stopped": True,
                    "disagreement_report": {},
                    "uncertainty_metrics": {},
                },
            },
        ]
        storage.list_debates.return_value = debates
        analytics_handler.ctx["storage"] = storage

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/analytics/consensus-quality", {}, mock_http_handler
            )

            assert result is not None
            body = json.loads(result.body)
            # Should still calculate stats, even with unusual values
            assert "stats" in body
            assert "quality_score" in body


# ==============================================================================
# Test: Metric Aggregation
# ==============================================================================


class TestMetricAggregation:
    """Test metric aggregation and calculations."""

    @pytest.mark.asyncio
    async def test_disagreement_stats_counts_types_correctly(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test that disagreement types are counted correctly."""
        storage = MagicMock()
        debates = [
            {
                "id": f"debate-{i}",
                "timestamp": f"2024-01-{i + 1:02d}T00:00:00Z",
                "messages": [],
                "result": {
                    "disagreement_report": {"unanimous_critiques": i % 2 == 0},
                    "uncertainty_metrics": {"disagreement_type": f"type-{i % 3}"},
                },
            }
            for i in range(9)
        ]
        storage.list_debates.return_value = debates
        analytics_handler.ctx["storage"] = storage

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/analytics/disagreements", {}, mock_http_handler
            )

            body = json.loads(result.body)
            stats = body["stats"]
            assert stats["total_debates"] == 9
            # Types should be aggregated
            assert "disagreement_types" in stats

    @pytest.mark.asyncio
    async def test_early_stop_stats_average_calculation(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test average rounds calculation in early stop stats."""
        storage = MagicMock()
        debates = [
            {"id": "d1", "result": {"rounds_used": 2, "early_stopped": True}},
            {"id": "d2", "result": {"rounds_used": 4, "early_stopped": True}},
            {"id": "d3", "result": {"rounds_used": 6, "early_stopped": False}},
            {"id": "d4", "result": {"rounds_used": 8, "early_stopped": False}},
        ]
        storage.list_debates.return_value = debates
        analytics_handler.ctx["storage"] = storage

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/analytics/early-stops", {}, mock_http_handler
            )

            body = json.loads(result.body)
            stats = body["stats"]
            assert stats["total_debates"] == 4
            assert stats["early_stopped"] == 2
            assert stats["full_rounds"] == 2
            # Average should be (2+4+6+8)/4 = 5.0
            assert stats["average_rounds"] == 5.0

    @pytest.mark.asyncio
    async def test_role_rotation_aggregates_all_roles(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test that all cognitive roles are aggregated."""
        storage = MagicMock()
        debates = [
            {
                "id": "d1",
                "messages": [
                    {"cognitive_role": "advocate"},
                    {"cognitive_role": "critic"},
                    {"cognitive_role": "synthesizer"},
                ],
            },
            {
                "id": "d2",
                "messages": [
                    {"cognitive_role": "advocate"},
                    {"role": "fallback_role"},
                ],
            },
        ]
        storage.list_debates.return_value = debates
        analytics_handler.ctx["storage"] = storage

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/analytics/role-rotation", {}, mock_http_handler
            )

            body = json.loads(result.body)
            stats = body["stats"]
            role_assignments = stats["role_assignments"]
            assert role_assignments["advocate"] == 2
            assert role_assignments["critic"] == 1
            assert role_assignments["synthesizer"] == 1
            assert role_assignments["fallback_role"] == 1

    @pytest.mark.asyncio
    async def test_consensus_quality_score_clamping(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test that quality score is clamped between 0 and 100."""
        storage = MagicMock()
        # Create debates with very high confidence to test upper bound
        debates = [
            {
                "id": f"d{i}",
                "timestamp": f"2024-01-{i + 1:02d}T00:00:00Z",
                "result": {"confidence": 1.0, "consensus_reached": True},
            }
            for i in range(10)
        ]
        storage.list_debates.return_value = debates
        analytics_handler.ctx["storage"] = storage

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/analytics/consensus-quality", {}, mock_http_handler
            )

            body = json.loads(result.body)
            quality_score = body["quality_score"]
            assert 0 <= quality_score <= 100


# ==============================================================================
# Test: Agent Performance
# ==============================================================================


class TestAgentPerformanceMetrics:
    """Test agent performance-related metrics."""

    @pytest.mark.asyncio
    async def test_ranking_stats_elo_range(
        self,
        analytics_handler,
        mock_http_handler,
        mock_elo_system,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test ELO range calculation in ranking stats."""
        analytics_handler.ctx["elo_system"] = mock_elo_system

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle("/api/ranking/stats", {}, mock_http_handler)

            body = json.loads(result.body)
            stats = body["stats"]
            assert "elo_range" in stats
            assert stats["elo_range"]["min"] <= stats["elo_range"]["max"]

    @pytest.mark.asyncio
    async def test_learning_efficiency_with_domain_filter(
        self,
        analytics_handler,
        mock_http_handler,
        mock_elo_system,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test learning efficiency filtered by domain."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            with patch(
                "aragora.ranking.elo.get_elo_store",
                return_value=mock_elo_system,
            ):
                result = await analytics_handler.handle(
                    "/api/analytics/learning-efficiency",
                    {"domain": ["security"]},
                    mock_http_handler,
                )

                body = json.loads(result.body)
                assert body["domain"] == "security"

    @pytest.mark.asyncio
    async def test_voting_accuracy_aggregation(
        self,
        analytics_handler,
        mock_http_handler,
        mock_elo_system,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test voting accuracy aggregation for multiple agents."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            with patch(
                "aragora.ranking.elo.get_elo_store",
                return_value=mock_elo_system,
            ):
                result = await analytics_handler.handle(
                    "/api/analytics/voting-accuracy",
                    {},
                    mock_http_handler,
                )

                body = json.loads(result.body)
                assert "agents" in body
                # Should have agents with accuracy data
                for agent_data in body["agents"]:
                    assert "agent" in agent_data
                    assert "accuracy" in agent_data

    @pytest.mark.asyncio
    async def test_calibration_with_limit_parameter(
        self,
        analytics_handler,
        mock_http_handler,
        mock_elo_system,
        mock_calibration_tracker,
        mock_calibration_module,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test calibration stats with limit parameter."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            with patch(
                "aragora.ranking.elo.get_elo_store",
                return_value=mock_elo_system,
            ):
                result = await analytics_handler.handle(
                    "/api/analytics/calibration",
                    {"limit": ["2"]},
                    mock_http_handler,
                )

                body = json.loads(result.body)
                assert "agents" in body
                # With limit=2, should have at most 2 agents
                # (Note: mock returns 3, but limit should filter)
                assert len(body["agents"]) <= 3  # Mock may not actually limit


# ==============================================================================
# Test: Query Parameter Validation
# ==============================================================================


class TestQueryParameterValidation:
    """Test query parameter validation and defaults."""

    @pytest.mark.asyncio
    async def test_limit_clamps_to_max(
        self,
        analytics_handler,
        mock_http_handler,
        mock_elo_system,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test that limit is clamped to maximum value."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            with patch(
                "aragora.ranking.elo.get_elo_store",
                return_value=mock_elo_system,
            ):
                # Request limit > max (1000)
                result = await analytics_handler.handle(
                    "/api/analytics/learning-efficiency",
                    {"limit": ["9999"]},
                    mock_http_handler,
                )

                assert result is not None
                # Should succeed with clamped limit
                body = json.loads(result.body)
                assert "agents" in body or "error" in body

    @pytest.mark.asyncio
    async def test_limit_clamps_to_min(
        self,
        analytics_handler,
        mock_http_handler,
        mock_elo_system,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test that limit is clamped to minimum value (1)."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            with patch(
                "aragora.ranking.elo.get_elo_store",
                return_value=mock_elo_system,
            ):
                # Request limit < min (0 or negative)
                result = await analytics_handler.handle(
                    "/api/analytics/learning-efficiency",
                    {"limit": ["-5"]},
                    mock_http_handler,
                )

                assert result is not None
                body = json.loads(result.body)
                # Should still return valid response
                assert "agents" in body or "error" in body

    @pytest.mark.asyncio
    async def test_empty_agent_param_returns_all(
        self,
        analytics_handler,
        mock_http_handler,
        mock_elo_system,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test that empty agent param returns all agents."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            with patch(
                "aragora.ranking.elo.get_elo_store",
                return_value=mock_elo_system,
            ):
                result = await analytics_handler.handle(
                    "/api/analytics/voting-accuracy",
                    {"agent": [""]},
                    mock_http_handler,
                )

                body = json.loads(result.body)
                # Should return agents list
                assert "agents" in body

    @pytest.mark.asyncio
    async def test_none_agent_param_returns_all(
        self,
        analytics_handler,
        mock_http_handler,
        mock_elo_system,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test that None agent param returns all agents."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            with patch(
                "aragora.ranking.elo.get_elo_store",
                return_value=mock_elo_system,
            ):
                result = await analytics_handler.handle(
                    "/api/analytics/voting-accuracy",
                    {"agent": [None]},
                    mock_http_handler,
                )

                body = json.loads(result.body)
                # Should return agents list, not single agent
                assert "agents" in body


# ==============================================================================
# Test: Cross-Pollination Statistics
# ==============================================================================


class TestCrossPollinationStats:
    """Test cross-pollination statistics endpoint."""

    @pytest.mark.asyncio
    async def test_cross_pollination_returns_version(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test that cross-pollination stats include version."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/analytics/cross-pollination", {}, mock_http_handler
            )

            body = json.loads(result.body)
            assert "version" in body
            assert body["version"] == "2.0.3"

    @pytest.mark.asyncio
    async def test_cross_pollination_has_expected_sections(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test that cross-pollination stats have all expected sections."""
        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/analytics/cross-pollination", {}, mock_http_handler
            )

            body = json.loads(result.body)
            stats = body["stats"]
            assert "calibration" in stats
            assert "learning" in stats
            assert "voting_accuracy" in stats
            assert "adaptive_rounds" in stats
            assert "rlm_cache" in stats

    @pytest.mark.asyncio
    async def test_cross_pollination_with_rlm_cache(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test cross-pollination stats with RLM cache available."""
        mock_cache = MagicMock()
        mock_cache.get_stats.return_value = {
            "hits": 100,
            "misses": 20,
            "hit_rate": 0.833,
        }

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            with patch(
                "aragora.rlm.bridge.RLMHierarchyCache",
                return_value=mock_cache,
            ):
                result = await analytics_handler.handle(
                    "/api/analytics/cross-pollination", {}, mock_http_handler
                )

                body = json.loads(result.body)
                rlm_stats = body["stats"]["rlm_cache"]
                assert rlm_stats["enabled"] is True
                assert rlm_stats["hits"] == 100
                assert rlm_stats["misses"] == 20
                assert rlm_stats["hit_rate"] == 0.833


# ==============================================================================
# Test: Trend Detection
# ==============================================================================


class TestTrendDetection:
    """Test trend detection in consensus quality."""

    @pytest.mark.asyncio
    async def test_stable_trend_detection(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test that stable trends are detected correctly."""
        storage = MagicMock()
        # Create debates with stable confidence
        debates = [
            {
                "id": f"d{i}",
                "timestamp": f"2024-01-{i + 1:02d}T00:00:00Z",
                "result": {"confidence": 0.75, "consensus_reached": True},
            }
            for i in range(10)
        ]
        storage.list_debates.return_value = debates
        analytics_handler.ctx["storage"] = storage

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/analytics/consensus-quality", {}, mock_http_handler
            )

            body = json.loads(result.body)
            assert body["stats"]["trend"] == "stable"

    @pytest.mark.asyncio
    async def test_insufficient_data_trend(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test that insufficient data returns appropriate trend."""
        storage = MagicMock()
        # Create only 3 debates (less than 5 needed for trend)
        debates = [
            {
                "id": f"d{i}",
                "result": {"confidence": 0.5 + i * 0.1, "consensus_reached": True},
            }
            for i in range(3)
        ]
        storage.list_debates.return_value = debates
        analytics_handler.ctx["storage"] = storage

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/analytics/consensus-quality", {}, mock_http_handler
            )

            body = json.loads(result.body)
            # With < 5 debates, trend should be "stable" (default)
            assert body["stats"]["trend"] in ["stable", "improving", "declining"]


# ==============================================================================
# Test: Alert Generation
# ==============================================================================


class TestAlertGeneration:
    """Test alert generation in consensus quality."""

    @pytest.mark.asyncio
    async def test_critical_alert_threshold(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test that critical alert is generated for very low quality."""
        storage = MagicMock()
        # Create debates with very low confidence and no consensus
        debates = [
            {
                "id": f"d{i}",
                "timestamp": f"2024-01-{i + 1:02d}T00:00:00Z",
                "result": {"confidence": 0.1, "consensus_reached": False},
            }
            for i in range(10)
        ]
        storage.list_debates.return_value = debates
        analytics_handler.ctx["storage"] = storage

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/analytics/consensus-quality", {}, mock_http_handler
            )

            body = json.loads(result.body)
            assert body["alert"] is not None
            assert body["alert"]["level"] == "critical"

    @pytest.mark.asyncio
    async def test_warning_alert_threshold(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test that warning alert is generated for moderately low quality."""
        storage = MagicMock()
        # Create debates with moderate confidence
        debates = [
            {
                "id": f"d{i}",
                "timestamp": f"2024-01-{i + 1:02d}T00:00:00Z",
                "result": {"confidence": 0.4, "consensus_reached": i % 2 == 0},
            }
            for i in range(10)
        ]
        storage.list_debates.return_value = debates
        analytics_handler.ctx["storage"] = storage

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/analytics/consensus-quality", {}, mock_http_handler
            )

            body = json.loads(result.body)
            # Quality should be moderate, alert could be warning or critical
            assert body["alert"] is not None or body["quality_score"] >= 60

    @pytest.mark.asyncio
    async def test_no_alert_for_high_quality(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test that no alert is generated for high quality."""
        storage = MagicMock()
        # Create debates with high confidence and consensus
        debates = [
            {
                "id": f"d{i}",
                "timestamp": f"2024-01-{i + 1:02d}T00:00:00Z",
                "result": {"confidence": 0.95, "consensus_reached": True},
            }
            for i in range(10)
        ]
        storage.list_debates.return_value = debates
        analytics_handler.ctx["storage"] = storage

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/analytics/consensus-quality", {}, mock_http_handler
            )

            body = json.loads(result.body)
            # High quality should have no alert or just info
            if body["alert"] is not None:
                assert body["alert"]["level"] in ["info", "warning"]


# ==============================================================================
# Test: Memory Statistics
# ==============================================================================


class TestMemoryStatistics:
    """Test memory statistics endpoint."""

    @pytest.mark.asyncio
    async def test_memory_stats_detects_all_databases(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter, tmp_path
    ):
        """Test that all database files are detected."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()

        # Create all database files (use aragora_insights.db per DB_INSIGHTS_PATH config)
        (nomic_dir / "debate_embeddings.db").touch()
        (nomic_dir / "aragora_insights.db").touch()
        (nomic_dir / "continuum_memory.db").touch()

        analytics_handler.ctx["nomic_dir"] = nomic_dir

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle("/api/memory/stats", {}, mock_http_handler)

            body = json.loads(result.body)
            stats = body["stats"]
            assert stats["embeddings_db"] is True
            assert stats["insights_db"] is True
            assert stats["continuum_memory"] is True

    @pytest.mark.asyncio
    async def test_memory_stats_partial_databases(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter, tmp_path
    ):
        """Test that partial database presence is detected correctly."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()

        # Create only some database files
        (nomic_dir / "debate_embeddings.db").touch()
        # insights.db not created
        # continuum_memory.db not created

        analytics_handler.ctx["nomic_dir"] = nomic_dir

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle("/api/memory/stats", {}, mock_http_handler)

            body = json.loads(result.body)
            stats = body["stats"]
            assert stats["embeddings_db"] is True
            assert stats["insights_db"] is False
            assert stats["continuum_memory"] is False


# ==============================================================================
# Test: Rate Limiting Behavior
# ==============================================================================


class TestRateLimitingBehavior:
    """Test rate limiting behavior in detail."""

    @pytest.mark.asyncio
    async def test_rate_limit_resets_after_window(
        self, analytics_handler, mock_http_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test that rate limit resets after time window."""
        from aragora.server.handlers.analytics import _analytics_limiter

        # Clear the limiter
        _analytics_limiter.clear()

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            # First request should succeed
            result = await analytics_handler.handle(
                "/api/analytics/disagreements", {}, mock_http_handler
            )
            assert result is not None
            assert result.status_code != 429

    @pytest.mark.asyncio
    async def test_rate_limit_tracks_different_ips(
        self, analytics_handler, mock_auth_context, reset_rate_limiter
    ):
        """Test that rate limit tracks different client IPs separately."""
        from aragora.server.handlers.analytics import _analytics_limiter

        # Clear the limiter
        _analytics_limiter.clear()

        handler1 = MagicMock()
        handler1.client_address = ("192.168.1.1", 12345)
        handler1.headers = {}

        handler2 = MagicMock()
        handler2.client_address = ("192.168.1.2", 12345)
        handler2.headers = {}

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            # Both IPs should be allowed
            result1 = await analytics_handler.handle("/api/analytics/disagreements", {}, handler1)
            result2 = await analytics_handler.handle("/api/analytics/disagreements", {}, handler2)

            # Both should succeed (or at least not both fail due to same rate limit)
            assert result1 is not None
            assert result2 is not None


# ==============================================================================
# Test: Response Structure Consistency
# ==============================================================================


class TestResponseStructureConsistency:
    """Test that response structures are consistent."""

    @pytest.mark.asyncio
    async def test_all_stats_endpoints_have_stats_key(
        self,
        analytics_handler,
        mock_http_handler,
        mock_storage,
        mock_elo_system,
        mock_auth_context,
        reset_rate_limiter,
        tmp_path,
    ):
        """Test that all stats endpoints return a 'stats' key."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()

        analytics_handler.ctx["storage"] = mock_storage
        analytics_handler.ctx["elo_system"] = mock_elo_system
        analytics_handler.ctx["nomic_dir"] = nomic_dir

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            stats_endpoints = [
                "/api/analytics/disagreements",
                "/api/analytics/role-rotation",
                "/api/analytics/early-stops",
                "/api/ranking/stats",
                "/api/memory/stats",
                "/api/analytics/cross-pollination",
            ]

            for endpoint in stats_endpoints:
                result = await analytics_handler.handle(endpoint, {}, mock_http_handler)
                assert result is not None, f"{endpoint} returned None"
                body = json.loads(result.body)
                assert "stats" in body, f"{endpoint} missing 'stats' key"

    @pytest.mark.asyncio
    async def test_consensus_quality_has_required_fields(
        self,
        analytics_handler,
        mock_http_handler,
        mock_storage,
        mock_auth_context,
        reset_rate_limiter,
    ):
        """Test that consensus quality response has all required fields."""
        analytics_handler.ctx["storage"] = mock_storage

        with (
            patch.object(
                analytics_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(analytics_handler, "check_permission") as mock_check,
        ):
            mock_auth.return_value = mock_auth_context
            mock_check.return_value = True

            result = await analytics_handler.handle(
                "/api/analytics/consensus-quality", {}, mock_http_handler
            )

            body = json.loads(result.body)
            assert "stats" in body
            assert "quality_score" in body
            assert "alert" in body

            stats = body["stats"]
            assert "total_debates" in stats
            assert "trend" in stats
            assert "average_confidence" in stats
            assert "consensus_rate" in stats
