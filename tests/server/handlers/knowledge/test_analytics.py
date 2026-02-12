"""
Tests for aragora.server.handlers.knowledge.analytics - Knowledge Analytics handler.

Tests cover:
- Route matching (can_handle)
- GET /api/v1/knowledge/mound/stats - Mound statistics
- GET /api/v1/knowledge/sharing/stats - Sharing statistics
- GET /api/v1/knowledge/federation/stats - Federation statistics
- GET /api/v1/knowledge/analytics/summary - Combined analytics summary
- GET /api/v1/knowledge/learning/stats - Learning statistics
- Rate limiting
- Authentication and RBAC checks
- Error handling
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.server.handlers.knowledge.analytics import AnalyticsHandler


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter between tests."""
    from aragora.server.handlers.knowledge.analytics import _analytics_limiter

    _analytics_limiter.clear()
    yield


@dataclass
class MockUser:
    """Mock user for testing."""

    user_id: str = "user-123"
    email: str = "test@example.com"
    org_id: str | None = "org-123"
    role: str = "admin"


@dataclass
class MockMoundStats:
    """Mock knowledge mound statistics."""

    total_nodes: int = 150
    nodes_by_type: dict = field(default_factory=lambda: {"fact": 50, "insight": 30, "pattern": 70})
    nodes_by_tier: dict = field(default_factory=lambda: {"global": 20, "org": 100, "workspace": 30})
    nodes_by_validation: dict = field(default_factory=lambda: {"validated": 100, "pending": 50})
    total_relationships: int = 200
    relationships_by_type: dict = field(
        default_factory=lambda: {"supports": 100, "contradicts": 50, "relates": 50}
    )
    average_confidence: float = 0.85
    stale_nodes_count: int = 5


class MockHandler:
    """Mock HTTP request handler."""

    def __init__(self, client_ip: str = "127.0.0.1"):
        self.client_ip = client_ip
        self._query_params = {}

    def get_client_ip(self):
        return self.client_ip


@pytest.fixture
def analytics_handler():
    """Create an AnalyticsHandler for testing."""
    from aragora.server.handlers.base import clear_cache

    clear_cache()
    return AnalyticsHandler(server_context={})


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler."""
    return MockHandler()


@pytest.fixture
def mock_user():
    """Create a mock authenticated user."""
    return MockUser()


# ===========================================================================
# Route Matching Tests (can_handle)
# ===========================================================================


class TestAnalyticsHandlerRouting:
    """Tests for AnalyticsHandler route matching."""

    def test_can_handle_mound_stats(self, analytics_handler):
        """Handler recognizes mound stats path."""
        assert analytics_handler.can_handle("/api/v1/knowledge/mound/stats") is True

    def test_can_handle_sharing_stats(self, analytics_handler):
        """Handler recognizes sharing stats path."""
        assert analytics_handler.can_handle("/api/v1/knowledge/sharing/stats") is True

    def test_can_handle_federation_stats(self, analytics_handler):
        """Handler recognizes federation stats path."""
        assert analytics_handler.can_handle("/api/v1/knowledge/federation/stats") is True

    def test_can_handle_analytics_summary(self, analytics_handler):
        """Handler recognizes analytics summary path."""
        assert analytics_handler.can_handle("/api/v1/knowledge/analytics") is True

    def test_can_handle_learning_stats(self, analytics_handler):
        """Handler recognizes learning stats path."""
        assert analytics_handler.can_handle("/api/v1/knowledge/learning") is True

    def test_can_handle_unknown_path(self, analytics_handler):
        """Handler rejects unknown paths."""
        assert analytics_handler.can_handle("/api/v1/unknown/path") is False
        assert analytics_handler.can_handle("/api/v1/debates") is False
        assert analytics_handler.can_handle("/api/v1/knowledge/other") is False


# ===========================================================================
# GET /api/v1/knowledge/mound/stats Tests
# ===========================================================================


class TestMoundStats:
    """Tests for mound statistics endpoint."""

    @pytest.mark.asyncio
    async def test_get_mound_stats_success(self, analytics_handler, mock_http_handler, mock_user):
        """Successfully retrieve mound statistics."""
        mock_stats = MockMoundStats()
        mock_mound = MagicMock()
        mock_mound.get_stats = AsyncMock(return_value=mock_stats)

        with patch.object(
            analytics_handler, "require_auth_or_error", return_value=(mock_user, None)
        ):
            with patch(
                "aragora.knowledge.mound.get_knowledge_mound",
                return_value=mock_mound,
            ):
                result = await analytics_handler.handle(
                    "/api/v1/knowledge/mound/stats",
                    {"workspace_id": "ws-123"},
                    mock_http_handler,
                )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total_nodes"] == 150
        assert body["average_confidence"] == 0.85
        assert body["workspace_id"] == "ws-123"

    @pytest.mark.asyncio
    async def test_get_mound_stats_import_error(
        self, analytics_handler, mock_http_handler, mock_user
    ):
        """Return empty stats when knowledge mound module not available."""
        with patch.object(
            analytics_handler, "require_auth_or_error", return_value=(mock_user, None)
        ):
            with patch(
                "aragora.knowledge.mound.get_knowledge_mound",
                side_effect=ImportError("No module"),
            ):
                result = await analytics_handler._get_mound_stats("ws-123")

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total_nodes"] == 0
        assert body["average_confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_get_mound_stats_error_handling(
        self, analytics_handler, mock_http_handler, mock_user
    ):
        """Handle errors when retrieving mound stats."""
        with patch.object(
            analytics_handler, "require_auth_or_error", return_value=(mock_user, None)
        ):
            with patch(
                "aragora.knowledge.mound.get_knowledge_mound",
                side_effect=Exception("DB error"),
            ):
                result = await analytics_handler._get_mound_stats("ws-123")

        assert result is not None
        assert result.status_code == 500


# ===========================================================================
# GET /api/v1/knowledge/sharing/stats Tests
# ===========================================================================


class TestSharingStats:
    """Tests for sharing statistics endpoint."""

    def test_get_sharing_stats_success(self, analytics_handler, mock_user):
        """Successfully retrieve sharing statistics."""
        mock_notifications = [
            MagicMock(notification_type=MagicMock(value="item_shared")),
            MagicMock(notification_type=MagicMock(value="item_shared")),
        ]
        mock_store = MagicMock()
        mock_store.get_notifications = MagicMock(return_value=mock_notifications)

        with patch(
            "aragora.server.handlers.knowledge.analytics.get_notification_store",
            return_value=mock_store,
        ):
            result = analytics_handler._get_sharing_stats("ws-123", "user-123")

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["items_shared_with_me"] == 2
        assert body["workspace_id"] == "ws-123"

    def test_get_sharing_stats_import_error(self, analytics_handler):
        """Return empty stats when notification store not available."""
        with patch(
            "aragora.server.handlers.knowledge.analytics.get_notification_store",
            side_effect=ImportError,
        ):
            result = analytics_handler._get_sharing_stats("ws-123", "user-123")

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total_shared_items"] == 0


# ===========================================================================
# GET /api/v1/knowledge/federation/stats Tests
# ===========================================================================


class TestFederationStats:
    """Tests for federation statistics endpoint."""

    def test_get_federation_stats_success(self, analytics_handler):
        """Successfully retrieve federation statistics."""
        mock_scheduler = MagicMock()
        mock_scheduler.list_schedules = MagicMock(return_value=["region-us", "region-eu"])
        mock_scheduler.get_stats = MagicMock(
            return_value={
                "schedules": {"active": 2},
                "runs": {"total": 100},
                "recent": {"success_rate": 0.95},
            }
        )
        mock_run = MagicMock()
        mock_run.started_at = datetime.now(timezone.utc)
        mock_run.items_pushed = 50
        mock_run.items_pulled = 30
        mock_scheduler.get_history = MagicMock(return_value=[mock_run])

        with patch(
            "aragora.server.handlers.knowledge.analytics.get_federation_scheduler",
            return_value=mock_scheduler,
        ):
            result = analytics_handler._get_federation_stats("ws-123")

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["registered_regions"] == 2
        assert body["success_rate"] == 0.95

    def test_get_federation_stats_import_error(self, analytics_handler):
        """Return empty stats when federation scheduler not available."""
        with patch(
            "aragora.server.handlers.knowledge.analytics.get_federation_scheduler",
            side_effect=ImportError,
        ):
            result = analytics_handler._get_federation_stats("ws-123")

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["registered_regions"] == 0
        assert body["total_syncs"] == 0


# ===========================================================================
# GET /api/v1/knowledge/analytics/summary Tests
# ===========================================================================


class TestAnalyticsSummary:
    """Tests for combined analytics summary endpoint."""

    @pytest.mark.asyncio
    async def test_get_summary_success(self, analytics_handler, mock_user):
        """Successfully retrieve combined analytics summary."""
        mock_stats = MockMoundStats()
        mock_mound = MagicMock()
        mock_mound.get_stats = AsyncMock(return_value=mock_stats)

        with patch(
            "aragora.knowledge.mound.get_knowledge_mound",
            return_value=mock_mound,
        ):
            with patch(
                "aragora.server.handlers.knowledge.analytics.get_notification_store",
                side_effect=ImportError,
            ):
                with patch(
                    "aragora.server.handlers.knowledge.analytics.get_federation_scheduler",
                    side_effect=ImportError,
                ):
                    result = await analytics_handler._get_summary("ws-123", "user-123")

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "mound" in body
        assert "sharing" in body
        assert "federation" in body

    @pytest.mark.asyncio
    async def test_get_summary_error_handling(self, analytics_handler):
        """Handle errors when retrieving summary."""
        with patch.object(analytics_handler, "_get_mound_stats", side_effect=Exception("Error")):
            result = await analytics_handler._get_summary("ws-123", "user-123")

        assert result is not None
        assert result.status_code == 500


# ===========================================================================
# GET /api/v1/knowledge/learning/stats Tests
# ===========================================================================


class TestLearningStats:
    """Tests for learning statistics endpoint."""

    def test_get_learning_stats_success(self, analytics_handler):
        """Successfully retrieve learning statistics."""
        result = analytics_handler._get_learning_stats("ws-123")

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "knowledge_reuse" in body
        assert "validation" in body
        assert "learning_velocity" in body
        assert "cross_debate_utility" in body
        assert "adapter_activity" in body

    def test_get_learning_stats_with_continuum(self, analytics_handler):
        """Retrieve learning stats with continuum memory available."""
        mock_adapter = MagicMock()
        mock_adapter.get_stats = MagicMock(
            return_value={
                "avg_cross_debate_utility": 0.75,
                "km_validated_entries": 100,
            }
        )
        mock_continuum = MagicMock()
        mock_continuum._km_adapter = mock_adapter

        with patch(
            "aragora.server.handlers.knowledge.analytics.get_continuum_memory",
            return_value=mock_continuum,
        ):
            result = analytics_handler._get_learning_stats("ws-123")

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["cross_debate_utility"]["avg_utility_score"] == 0.75


# ===========================================================================
# Authentication and Rate Limiting Tests
# ===========================================================================


class TestAuthAndRateLimiting:
    """Tests for authentication and rate limiting."""

    @pytest.mark.asyncio
    async def test_unauthenticated_request_rejected(self, analytics_handler, mock_http_handler):
        """Unauthenticated requests are rejected."""
        from aragora.server.handlers.base import error_response

        with patch.object(
            analytics_handler,
            "require_auth_or_error",
            return_value=(None, error_response("Unauthorized", 401)),
        ):
            result = await analytics_handler.handle(
                "/api/v1/knowledge/mound/stats",
                {},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, analytics_handler, mock_http_handler, mock_user):
        """Requests exceeding rate limit are rejected."""
        from aragora.server.handlers.knowledge.analytics import _analytics_limiter

        # Exhaust rate limit
        for _ in range(61):
            _analytics_limiter.is_allowed("127.0.0.1")

        with patch.object(
            analytics_handler, "require_auth_or_error", return_value=(mock_user, None)
        ):
            result = await analytics_handler.handle(
                "/api/v1/knowledge/mound/stats",
                {},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 429


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_handle_returns_none_for_unhandled_path(
        self, analytics_handler, mock_http_handler, mock_user
    ):
        """Handler returns None for paths it recognizes but doesn't match exactly."""
        with patch.object(
            analytics_handler, "require_auth_or_error", return_value=(mock_user, None)
        ):
            result = await analytics_handler.handle(
                "/api/v1/knowledge/unknown/endpoint",
                {},
                mock_http_handler,
            )

        # can_handle returns False for unknown paths, so handle should not be called
        # This test verifies the handler returns None for unmatched routes
        assert result is None
