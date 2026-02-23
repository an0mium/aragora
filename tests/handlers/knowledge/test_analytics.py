"""
Comprehensive tests for AnalyticsHandler.

Tests the Knowledge Mound analytics HTTP handler endpoints:
- GET /api/v1/knowledge/mound/stats - Get mound statistics
- GET /api/v1/knowledge/sharing/stats - Get sharing statistics
- GET /api/v1/knowledge/federation/stats - Get federation statistics
- GET /api/v1/knowledge/analytics/summary - Get combined summary
- GET /api/v1/knowledge/learning/stats - Get learning statistics
- GET /api/v1/knowledge/analytics/learning - Get learning statistics (alias)
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.knowledge.analytics import AnalyticsHandler


# =============================================================================
# Helpers
# =============================================================================


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# =============================================================================
# Mock data classes
# =============================================================================


@dataclass
class MockMoundStats:
    """Mock for Knowledge Mound statistics."""

    total_nodes: int = 150
    nodes_by_type: dict = field(default_factory=lambda: {"fact": 80, "opinion": 50, "evidence": 20})
    nodes_by_tier: dict = field(default_factory=lambda: {"gold": 30, "silver": 70, "bronze": 50})
    nodes_by_validation: dict = field(
        default_factory=lambda: {"validated": 100, "pending": 40, "rejected": 10}
    )
    total_relationships: int = 250
    relationships_by_type: dict = field(
        default_factory=lambda: {"supports": 120, "contradicts": 80, "related": 50}
    )
    average_confidence: float = 0.78
    stale_nodes_count: int = 12


@dataclass
class MockNotification:
    """Mock notification object."""

    notification_type: Any = None
    message: str = ""


@dataclass
class MockNotificationType:
    """Mock notification type enum value."""

    value: str = "item_shared"


@dataclass
class MockFederationRun:
    """Mock federation run history entry."""

    started_at: datetime = field(default_factory=datetime.now)
    items_pushed: int = 5
    items_pulled: int = 3


# =============================================================================
# Mock HTTP handler
# =============================================================================


class _MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to the analytics handler."""

    def __init__(
        self,
        method: str = "GET",
        path: str = "/api/v1/knowledge/mound/stats",
        body: dict[str, Any] | None = None,
    ):
        self.command = method
        self.path = path
        self.client_address = ("127.0.0.1", 12345)
        self.headers = {"Content-Length": "0", "Host": "localhost:8080"}

        if body is not None:
            body_bytes = json.dumps(body).encode("utf-8")
            self.headers["Content-Length"] = str(len(body_bytes))
            self.headers["Content-Type"] = "application/json"
            self.rfile = io.BytesIO(body_bytes)
        else:
            self.rfile = io.BytesIO(b"")

    def get_client_ip(self):
        return "127.0.0.1"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def handler():
    """Create an AnalyticsHandler with empty server context."""
    return AnalyticsHandler(server_context={})


@pytest.fixture
def http_handler():
    """Create a mock HTTP handler."""
    return _MockHTTPHandler()


@pytest.fixture
def mock_mound():
    """Create a mock Knowledge Mound."""
    mound = AsyncMock()
    mound.get_stats = AsyncMock(return_value=MockMoundStats())
    return mound


@pytest.fixture
def mock_notification_store():
    """Create a mock notification store."""
    store = MagicMock()
    store.get_notifications = MagicMock(return_value=[])
    return store


@pytest.fixture
def mock_federation_scheduler():
    """Create a mock federation scheduler."""
    scheduler = MagicMock()
    scheduler.get_stats.return_value = {
        "schedules": {"active": 3},
        "runs": {"total": 42},
        "recent": {"success_rate": 0.95},
    }
    scheduler.get_history.return_value = []
    scheduler.list_schedules.return_value = ["region-a", "region-b"]
    return scheduler


# =============================================================================
# Patch path constants
# =============================================================================

_MOD = "aragora.server.handlers.knowledge.analytics"


# =============================================================================
# Test: Handler initialization
# =============================================================================


class TestHandlerInit:
    """Tests for handler construction."""

    def test_init_with_empty_context(self):
        h = AnalyticsHandler(server_context={})
        assert h.ctx == {}

    def test_init_with_none_context(self):
        h = AnalyticsHandler(server_context=None)
        assert h.ctx == {}

    def test_init_with_ctx_param(self):
        h = AnalyticsHandler(ctx={"key": "val"})
        assert h.ctx == {"key": "val"}

    def test_server_context_overrides_ctx(self):
        h = AnalyticsHandler(ctx={"old": True}, server_context={"new": True})
        assert h.ctx == {"new": True}


# =============================================================================
# Test: can_handle
# =============================================================================


class TestCanHandle:
    """Tests for route matching."""

    def test_handles_mound_stats(self, handler):
        assert handler.can_handle("/api/v1/knowledge/mound/stats") is True

    def test_handles_sharing_stats(self, handler):
        assert handler.can_handle("/api/v1/knowledge/sharing/stats") is True

    def test_handles_federation_stats(self, handler):
        assert handler.can_handle("/api/v1/knowledge/federation/stats") is True

    def test_handles_analytics_summary(self, handler):
        assert handler.can_handle("/api/v1/knowledge/analytics") is True

    def test_handles_analytics_learning(self, handler):
        assert handler.can_handle("/api/v1/knowledge/analytics/learning") is True

    def test_handles_learning_stats(self, handler):
        assert handler.can_handle("/api/v1/knowledge/learning") is True

    def test_rejects_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates/list") is False

    def test_rejects_partial_match(self, handler):
        assert handler.can_handle("/api/v1/knowledge/other") is False


# =============================================================================
# Test: GET /api/v1/knowledge/mound/stats
# =============================================================================


class TestMoundStats:
    """Tests for mound statistics endpoint."""

    @pytest.mark.asyncio
    async def test_mound_stats_success(self, handler, http_handler, mock_mound):
        """Returns real mound stats when knowledge mound is available."""
        with patch(f"{_MOD}.knowledge_mound.get_knowledge_mound", return_value=mock_mound):
            with patch(f"{_MOD}.emit_handler_event"):
                with patch(f"{_MOD}._analytics_limiter") as limiter:
                    limiter.is_allowed.return_value = True
                    result = await handler.handle(
                        "/api/v1/knowledge/mound/stats",
                        {},
                        http_handler,
                    )

        assert _status(result) == 200
        body = _body(result)
        assert body["total_nodes"] == 150
        assert body["total_relationships"] == 250
        assert body["average_confidence"] == 0.78
        assert body["stale_nodes_count"] == 12
        assert "nodes_by_type" in body
        assert "nodes_by_tier" in body
        assert "nodes_by_validation" in body
        assert "relationships_by_type" in body

    @pytest.mark.asyncio
    async def test_mound_stats_with_workspace(self, handler, http_handler, mock_mound):
        """Passes workspace_id through to the response."""
        with patch(f"{_MOD}.knowledge_mound.get_knowledge_mound", return_value=mock_mound):
            with patch(f"{_MOD}.emit_handler_event"):
                with patch(f"{_MOD}._analytics_limiter") as limiter:
                    limiter.is_allowed.return_value = True
                    result = await handler.handle(
                        "/api/v1/knowledge/mound/stats",
                        {"workspace_id": "ws-123"},
                        http_handler,
                    )

        assert _status(result) == 200
        body = _body(result)
        assert body["workspace_id"] == "ws-123"

    @pytest.mark.asyncio
    async def test_mound_stats_fallback_on_import_error(self, handler, http_handler):
        """Returns zero stats when knowledge mound import fails."""
        with patch(
            f"{_MOD}.knowledge_mound.get_knowledge_mound",
            side_effect=ImportError("no module"),
        ):
            with patch(f"{_MOD}.emit_handler_event"):
                with patch(f"{_MOD}._analytics_limiter") as limiter:
                    limiter.is_allowed.return_value = True
                    result = await handler.handle(
                        "/api/v1/knowledge/mound/stats",
                        {},
                        http_handler,
                    )

        assert _status(result) == 200
        body = _body(result)
        assert body["total_nodes"] == 0
        assert body["total_relationships"] == 0
        assert body["average_confidence"] == 0.0
        assert body["stale_nodes_count"] == 0

    @pytest.mark.asyncio
    async def test_mound_stats_error_returns_500(self, handler, http_handler):
        """Returns 500 on unexpected runtime error."""
        with patch(
            f"{_MOD}.knowledge_mound.get_knowledge_mound",
            side_effect=RuntimeError("db down"),
        ):
            with patch(f"{_MOD}.emit_handler_event"):
                with patch(f"{_MOD}._analytics_limiter") as limiter:
                    limiter.is_allowed.return_value = True
                    result = await handler.handle(
                        "/api/v1/knowledge/mound/stats",
                        {},
                        http_handler,
                    )

        assert _status(result) == 500
        body = _body(result)
        assert "error" in body or "message" in body


# =============================================================================
# Test: GET /api/v1/knowledge/sharing/stats
# =============================================================================


class TestSharingStats:
    """Tests for sharing statistics endpoint."""

    @pytest.mark.asyncio
    async def test_sharing_stats_success(self, handler, http_handler, mock_notification_store):
        """Returns sharing stats when notification store is available."""
        with patch(f"{_MOD}.get_notification_store", return_value=mock_notification_store):
            with patch(f"{_MOD}._analytics_limiter") as limiter:
                limiter.is_allowed.return_value = True
                result = await handler.handle(
                    "/api/v1/knowledge/sharing/stats",
                    {},
                    http_handler,
                )

        assert _status(result) == 200
        body = _body(result)
        assert body["total_shared_items"] == 0
        assert body["items_shared_with_me"] == 0
        assert body["items_shared_by_me"] == 0
        assert body["active_grants"] == 0
        assert body["expired_grants"] == 0

    @pytest.mark.asyncio
    async def test_sharing_stats_with_workspace(
        self, handler, http_handler, mock_notification_store
    ):
        """Passes workspace_id through to the response."""
        with patch(f"{_MOD}.get_notification_store", return_value=mock_notification_store):
            with patch(f"{_MOD}._analytics_limiter") as limiter:
                limiter.is_allowed.return_value = True
                result = await handler.handle(
                    "/api/v1/knowledge/sharing/stats",
                    {"workspace_id": "ws-456"},
                    http_handler,
                )

        assert _status(result) == 200
        body = _body(result)
        assert body["workspace_id"] == "ws-456"

    @pytest.mark.asyncio
    async def test_sharing_stats_counts_shared_notifications(
        self, handler, http_handler, mock_notification_store
    ):
        """Counts item_shared notifications for the user."""
        shared_notif = MockNotification(notification_type=MockNotificationType(value="item_shared"))
        other_notif = MockNotification(notification_type=MockNotificationType(value="mention"))
        mock_notification_store.get_notifications.return_value = [
            shared_notif,
            shared_notif,
            other_notif,
        ]

        with patch(f"{_MOD}.get_notification_store", return_value=mock_notification_store):
            with patch(f"{_MOD}._analytics_limiter") as limiter:
                limiter.is_allowed.return_value = True
                result = await handler.handle(
                    "/api/v1/knowledge/sharing/stats",
                    {},
                    http_handler,
                )

        assert _status(result) == 200
        body = _body(result)
        # The user_id comes from auth; shared_with_me counts item_shared types
        assert body["items_shared_with_me"] == 2

    @pytest.mark.asyncio
    async def test_sharing_stats_fallback_on_import_error(self, handler, http_handler):
        """Returns zero stats when notification store import fails."""
        with patch(f"{_MOD}.get_notification_store", side_effect=ImportError("no module")):
            with patch(f"{_MOD}._analytics_limiter") as limiter:
                limiter.is_allowed.return_value = True
                result = await handler.handle(
                    "/api/v1/knowledge/sharing/stats",
                    {},
                    http_handler,
                )

        assert _status(result) == 200
        body = _body(result)
        assert body["total_shared_items"] == 0
        assert body["items_shared_with_me"] == 0

    @pytest.mark.asyncio
    async def test_sharing_stats_error_returns_500(self, handler, http_handler):
        """Returns 500 on unexpected runtime error."""
        with patch(f"{_MOD}.get_notification_store", side_effect=RuntimeError("store down")):
            with patch(f"{_MOD}._analytics_limiter") as limiter:
                limiter.is_allowed.return_value = True
                result = await handler.handle(
                    "/api/v1/knowledge/sharing/stats",
                    {},
                    http_handler,
                )

        assert _status(result) == 500


# =============================================================================
# Test: GET /api/v1/knowledge/federation/stats
# =============================================================================


class TestFederationStats:
    """Tests for federation statistics endpoint."""

    @pytest.mark.asyncio
    async def test_federation_stats_success(self, handler, http_handler, mock_federation_scheduler):
        """Returns federation stats from scheduler."""
        with patch(
            f"{_MOD}.get_federation_scheduler", return_value=mock_federation_scheduler
        ):
            with patch(f"{_MOD}._analytics_limiter") as limiter:
                limiter.is_allowed.return_value = True
                result = await handler.handle(
                    "/api/v1/knowledge/federation/stats",
                    {},
                    http_handler,
                )

        assert _status(result) == 200
        body = _body(result)
        assert body["registered_regions"] == 2
        assert body["active_schedules"] == 3
        assert body["total_syncs"] == 42
        assert body["success_rate"] == 0.95
        assert body["items_pushed_today"] == 0
        assert body["items_pulled_today"] == 0
        assert body["last_sync_at"] is None

    @pytest.mark.asyncio
    async def test_federation_stats_with_history(
        self, handler, http_handler, mock_federation_scheduler
    ):
        """Calculates today's items pushed/pulled from history."""
        now = datetime.now()
        run1 = MockFederationRun(started_at=now, items_pushed=10, items_pulled=7)
        run2 = MockFederationRun(started_at=now, items_pushed=3, items_pulled=2)
        mock_federation_scheduler.get_history.return_value = [run1, run2]

        with patch(
            f"{_MOD}.get_federation_scheduler", return_value=mock_federation_scheduler
        ):
            with patch(f"{_MOD}._analytics_limiter") as limiter:
                limiter.is_allowed.return_value = True
                result = await handler.handle(
                    "/api/v1/knowledge/federation/stats",
                    {},
                    http_handler,
                )

        assert _status(result) == 200
        body = _body(result)
        assert body["items_pushed_today"] == 13
        assert body["items_pulled_today"] == 9
        # last_sync is the first element of history
        assert body["last_sync_at"] is not None

    @pytest.mark.asyncio
    async def test_federation_stats_with_workspace(
        self, handler, http_handler, mock_federation_scheduler
    ):
        """Passes workspace_id through to response."""
        with patch(
            f"{_MOD}.get_federation_scheduler", return_value=mock_federation_scheduler
        ):
            with patch(f"{_MOD}._analytics_limiter") as limiter:
                limiter.is_allowed.return_value = True
                result = await handler.handle(
                    "/api/v1/knowledge/federation/stats",
                    {"workspace_id": "ws-789"},
                    http_handler,
                )

        assert _status(result) == 200
        body = _body(result)
        assert body["workspace_id"] == "ws-789"

    @pytest.mark.asyncio
    async def test_federation_stats_fallback_on_import_error(self, handler, http_handler):
        """Returns zero stats when federation scheduler import fails."""
        with patch(f"{_MOD}.get_federation_scheduler", side_effect=ImportError("no module")):
            with patch(f"{_MOD}._analytics_limiter") as limiter:
                limiter.is_allowed.return_value = True
                result = await handler.handle(
                    "/api/v1/knowledge/federation/stats",
                    {},
                    http_handler,
                )

        assert _status(result) == 200
        body = _body(result)
        assert body["registered_regions"] == 0
        assert body["active_schedules"] == 0
        assert body["total_syncs"] == 0
        assert body["success_rate"] == 0

    @pytest.mark.asyncio
    async def test_federation_stats_error_returns_500(self, handler, http_handler):
        """Returns 500 on unexpected runtime error."""
        with patch(f"{_MOD}.get_federation_scheduler", side_effect=RuntimeError("crash")):
            with patch(f"{_MOD}._analytics_limiter") as limiter:
                limiter.is_allowed.return_value = True
                result = await handler.handle(
                    "/api/v1/knowledge/federation/stats",
                    {},
                    http_handler,
                )

        assert _status(result) == 500


# =============================================================================
# Test: GET /api/v1/knowledge/analytics/summary
# =============================================================================


class TestAnalyticsSummary:
    """Tests for combined analytics summary endpoint."""

    @pytest.mark.asyncio
    async def test_summary_aggregates_all_stats(
        self, handler, http_handler, mock_mound, mock_notification_store, mock_federation_scheduler
    ):
        """Summary combines mound, sharing, and federation stats."""
        with (
            patch(f"{_MOD}.knowledge_mound.get_knowledge_mound", return_value=mock_mound),
            patch(f"{_MOD}.get_notification_store", return_value=mock_notification_store),
            patch(f"{_MOD}.get_federation_scheduler", return_value=mock_federation_scheduler),
            patch(f"{_MOD}.emit_handler_event"),
            patch(f"{_MOD}._analytics_limiter") as limiter,
        ):
            limiter.is_allowed.return_value = True
            result = await handler.handle(
                "/api/v1/knowledge/analytics/summary",
                {},
                http_handler,
            )

        assert _status(result) == 200
        body = _body(result)
        assert "mound" in body
        assert "sharing" in body
        assert "federation" in body
        # Verify mound stats were included
        assert body["mound"]["total_nodes"] == 150
        # Verify sharing stats
        assert body["sharing"]["total_shared_items"] == 0
        # Verify federation stats
        assert body["federation"]["registered_regions"] == 2

    @pytest.mark.asyncio
    async def test_summary_with_workspace_id(
        self, handler, http_handler, mock_mound, mock_notification_store, mock_federation_scheduler
    ):
        """Summary passes workspace_id to all sub-calls."""
        with (
            patch(f"{_MOD}.knowledge_mound.get_knowledge_mound", return_value=mock_mound),
            patch(f"{_MOD}.get_notification_store", return_value=mock_notification_store),
            patch(f"{_MOD}.get_federation_scheduler", return_value=mock_federation_scheduler),
            patch(f"{_MOD}.emit_handler_event"),
            patch(f"{_MOD}._analytics_limiter") as limiter,
        ):
            limiter.is_allowed.return_value = True
            result = await handler.handle(
                "/api/v1/knowledge/analytics/summary",
                {"workspace_id": "ws-sum"},
                http_handler,
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["workspace_id"] == "ws-sum"

    @pytest.mark.asyncio
    async def test_summary_partial_failure_still_works(
        self, handler, http_handler, mock_notification_store, mock_federation_scheduler
    ):
        """Summary still returns data even if mound stats fail (via ImportError fallback)."""
        with (
            patch(
                f"{_MOD}.knowledge_mound.get_knowledge_mound",
                side_effect=ImportError("no mound"),
            ),
            patch(f"{_MOD}.get_notification_store", return_value=mock_notification_store),
            patch(f"{_MOD}.get_federation_scheduler", return_value=mock_federation_scheduler),
            patch(f"{_MOD}.emit_handler_event"),
            patch(f"{_MOD}._analytics_limiter") as limiter,
        ):
            limiter.is_allowed.return_value = True
            result = await handler.handle(
                "/api/v1/knowledge/analytics/summary",
                {},
                http_handler,
            )

        assert _status(result) == 200
        body = _body(result)
        # Mound stats should fall back to zeros
        assert body["mound"]["total_nodes"] == 0
        # Other stats should still be populated
        assert body["federation"]["registered_regions"] == 2


# =============================================================================
# Test: GET /api/v1/knowledge/learning/stats
# =============================================================================


class TestLearningStats:
    """Tests for learning statistics endpoint."""

    @pytest.mark.asyncio
    async def test_learning_stats_defaults(self, handler, http_handler):
        """Returns default zero stats when no adapters are available."""
        with (
            patch(f"{_MOD}.get_continuum_memory", side_effect=ImportError("no module")),
            patch(f"{_MOD}._analytics_limiter") as limiter,
        ):
            limiter.is_allowed.return_value = True
            result = await handler.handle(
                "/api/v1/knowledge/learning/stats",
                {},
                http_handler,
            )

        assert _status(result) == 200
        body = _body(result)
        assert "knowledge_reuse" in body
        assert body["knowledge_reuse"]["total_queries"] == 0
        assert body["knowledge_reuse"]["reuse_rate"] == 0.0
        assert "validation" in body
        assert body["validation"]["total_validations"] == 0
        assert body["validation"]["accuracy_rate"] == 0.0
        assert "learning_velocity" in body
        assert body["learning_velocity"]["new_items_today"] == 0
        assert "cross_debate_utility" in body
        assert body["cross_debate_utility"]["avg_utility_score"] == 0.0
        assert "adapter_activity" in body
        assert "timestamp" in body

    @pytest.mark.asyncio
    async def test_learning_stats_with_workspace(self, handler, http_handler):
        """Includes workspace_id in learning stats response."""
        with (
            patch(f"{_MOD}.get_continuum_memory", side_effect=ImportError("no module")),
            patch(f"{_MOD}._analytics_limiter") as limiter,
        ):
            limiter.is_allowed.return_value = True
            result = await handler.handle(
                "/api/v1/knowledge/learning/stats",
                {"workspace_id": "ws-learn"},
                http_handler,
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["workspace_id"] == "ws-learn"

    @pytest.mark.asyncio
    async def test_learning_stats_with_continuum_adapter(self, handler, http_handler):
        """Pulls cross-debate utility from continuum memory adapter."""
        mock_adapter = MagicMock()
        mock_adapter.get_stats.return_value = {
            "avg_cross_debate_utility": 0.75,
            "km_validated_entries": 100,
        }

        mock_continuum = MagicMock()
        mock_continuum._km_adapter = mock_adapter

        with (
            patch(f"{_MOD}.get_continuum_memory", return_value=mock_continuum),
            patch(f"{_MOD}._analytics_limiter") as limiter,
        ):
            limiter.is_allowed.return_value = True
            result = await handler.handle(
                "/api/v1/knowledge/learning/stats",
                {},
                http_handler,
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["cross_debate_utility"]["avg_utility_score"] == 0.75
        assert body["cross_debate_utility"]["high_utility_items"] == 75
        assert body["cross_debate_utility"]["low_utility_items"] == 25

    @pytest.mark.asyncio
    async def test_learning_stats_with_cross_subscribers(self, handler, http_handler):
        """Pulls validation stats from cross-subscriber manager."""
        mock_manager = MagicMock()
        mock_manager.get_stats.return_value = {
            "handlers": {
                "km_validation_feedback": {"call_count": 42},
            }
        }

        with (
            patch(f"{_MOD}.get_continuum_memory", side_effect=ImportError("no module")),
            patch(
                "aragora.events.cross_subscribers.get_cross_subscriber_manager",
                return_value=mock_manager,
            ),
            patch(f"{_MOD}._analytics_limiter") as limiter,
        ):
            limiter.is_allowed.return_value = True
            result = await handler.handle(
                "/api/v1/knowledge/learning/stats",
                {},
                http_handler,
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["validation"]["total_validations"] == 42

    @pytest.mark.asyncio
    async def test_learning_stats_continuum_error_graceful(self, handler, http_handler):
        """Continuum memory runtime error does not crash learning stats."""
        with (
            patch(f"{_MOD}.get_continuum_memory", side_effect=RuntimeError("connection lost")),
            patch(f"{_MOD}._analytics_limiter") as limiter,
        ):
            limiter.is_allowed.return_value = True
            result = await handler.handle(
                "/api/v1/knowledge/learning/stats",
                {},
                http_handler,
            )

        assert _status(result) == 200
        body = _body(result)
        # Should still return defaults
        assert body["cross_debate_utility"]["avg_utility_score"] == 0.0

    @pytest.mark.asyncio
    async def test_learning_stats_via_analytics_learning_alias(self, handler, http_handler):
        """The /api/v1/knowledge/analytics/learning alias routes to learning stats."""
        with (
            patch(f"{_MOD}.get_continuum_memory", side_effect=ImportError("no module")),
            patch(f"{_MOD}._analytics_limiter") as limiter,
        ):
            limiter.is_allowed.return_value = True
            result = await handler.handle(
                "/api/v1/knowledge/analytics/learning",
                {},
                http_handler,
            )

        assert _status(result) == 200
        body = _body(result)
        assert "knowledge_reuse" in body
        assert "validation" in body
        assert "learning_velocity" in body

    @pytest.mark.asyncio
    async def test_learning_stats_zero_queries_no_division_error(self, handler, http_handler):
        """Reuse rate calculation does not divide by zero when total_queries is 0."""
        with (
            patch(f"{_MOD}.get_continuum_memory", side_effect=ImportError("no module")),
            patch(f"{_MOD}._analytics_limiter") as limiter,
        ):
            limiter.is_allowed.return_value = True
            result = await handler.handle(
                "/api/v1/knowledge/learning/stats",
                {},
                http_handler,
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["knowledge_reuse"]["reuse_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_learning_stats_zero_validations_no_division_error(self, handler, http_handler):
        """Accuracy rate calculation does not divide by zero when total_validations is 0."""
        with (
            patch(f"{_MOD}.get_continuum_memory", side_effect=ImportError("no module")),
            patch(f"{_MOD}._analytics_limiter") as limiter,
        ):
            limiter.is_allowed.return_value = True
            result = await handler.handle(
                "/api/v1/knowledge/learning/stats",
                {},
                http_handler,
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["validation"]["accuracy_rate"] == 0.0


# =============================================================================
# Test: Rate limiting
# =============================================================================


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_returns_429(self, handler, http_handler):
        """Returns 429 when rate limit is exceeded."""
        with patch(f"{_MOD}._analytics_limiter") as limiter:
            limiter.is_allowed.return_value = False
            result = await handler.handle(
                "/api/v1/knowledge/mound/stats",
                {},
                http_handler,
            )

        assert _status(result) == 429
        body = _body(result)
        assert "rate limit" in body.get("error", "").lower() or "rate limit" in body.get(
            "message", ""
        ).lower()

    @pytest.mark.asyncio
    async def test_rate_limit_uses_client_ip_from_handler_method(self, handler):
        """Uses handler.get_client_ip() when available."""
        mock_http = _MockHTTPHandler()
        with patch(f"{_MOD}._analytics_limiter") as limiter:
            limiter.is_allowed.return_value = False
            await handler.handle(
                "/api/v1/knowledge/mound/stats",
                {},
                mock_http,
            )
            limiter.is_allowed.assert_called_with("127.0.0.1")

    @pytest.mark.asyncio
    async def test_rate_limit_uses_get_client_ip_fallback(self, handler):
        """Falls back to get_client_ip() when handler lacks get_client_ip method."""
        mock_http = MagicMock(spec=[])
        mock_http.client_address = ("10.0.0.1", 9999)
        mock_http.headers = {}

        with (
            patch(f"{_MOD}._analytics_limiter") as limiter,
            patch(f"{_MOD}.get_client_ip", return_value="10.0.0.1"),
        ):
            limiter.is_allowed.return_value = False
            await handler.handle(
                "/api/v1/knowledge/mound/stats",
                {},
                mock_http,
            )
            limiter.is_allowed.assert_called_with("10.0.0.1")


# =============================================================================
# Test: Unmatched routes
# =============================================================================


class TestUnmatchedRoutes:
    """Tests for paths that don't match any endpoint."""

    @pytest.mark.asyncio
    async def test_unmatched_path_returns_none(self, handler, http_handler):
        """Returns None for paths not handled by this handler."""
        with patch(f"{_MOD}._analytics_limiter") as limiter:
            limiter.is_allowed.return_value = True
            result = await handler.handle(
                "/api/v1/knowledge/unknown/endpoint",
                {},
                http_handler,
            )

        assert result is None


# =============================================================================
# Test: Federation stats with last_sync_at
# =============================================================================


class TestFederationLastSync:
    """Tests for federation last_sync_at field."""

    @pytest.mark.asyncio
    async def test_last_sync_at_none_when_no_history(
        self, handler, http_handler, mock_federation_scheduler
    ):
        """last_sync_at is None when history is empty."""
        mock_federation_scheduler.get_history.return_value = []

        with (
            patch(f"{_MOD}.get_federation_scheduler", return_value=mock_federation_scheduler),
            patch(f"{_MOD}._analytics_limiter") as limiter,
        ):
            limiter.is_allowed.return_value = True
            result = await handler.handle(
                "/api/v1/knowledge/federation/stats",
                {},
                http_handler,
            )

        body = _body(result)
        assert body["last_sync_at"] is None

    @pytest.mark.asyncio
    async def test_last_sync_at_from_first_history_entry(
        self, handler, http_handler, mock_federation_scheduler
    ):
        """last_sync_at comes from the first history entry."""
        sync_time = datetime(2026, 2, 23, 10, 0, 0)
        run = MockFederationRun(started_at=sync_time, items_pushed=1, items_pulled=1)
        mock_federation_scheduler.get_history.return_value = [run]

        with (
            patch(f"{_MOD}.get_federation_scheduler", return_value=mock_federation_scheduler),
            patch(f"{_MOD}._analytics_limiter") as limiter,
        ):
            limiter.is_allowed.return_value = True
            result = await handler.handle(
                "/api/v1/knowledge/federation/stats",
                {},
                http_handler,
            )

        body = _body(result)
        assert body["last_sync_at"] == sync_time.isoformat()


# =============================================================================
# Test: Learning stats with adapter having zero validated entries
# =============================================================================


class TestLearningAdapterEdgeCases:
    """Edge cases for learning stats adapter integration."""

    @pytest.mark.asyncio
    async def test_adapter_with_zero_validated_entries(self, handler, http_handler):
        """No division or utility calculations when km_validated_entries is 0."""
        mock_adapter = MagicMock()
        mock_adapter.get_stats.return_value = {
            "avg_cross_debate_utility": 0.5,
            "km_validated_entries": 0,
        }

        mock_continuum = MagicMock()
        mock_continuum._km_adapter = mock_adapter

        with (
            patch(f"{_MOD}.get_continuum_memory", return_value=mock_continuum),
            patch(f"{_MOD}._analytics_limiter") as limiter,
        ):
            limiter.is_allowed.return_value = True
            result = await handler.handle(
                "/api/v1/knowledge/learning/stats",
                {},
                http_handler,
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["cross_debate_utility"]["high_utility_items"] == 0
        assert body["cross_debate_utility"]["low_utility_items"] == 0

    @pytest.mark.asyncio
    async def test_continuum_without_km_adapter(self, handler, http_handler):
        """Gracefully handles continuum memory without _km_adapter attribute."""
        mock_continuum = MagicMock(spec=[])  # No _km_adapter attribute

        with (
            patch(f"{_MOD}.get_continuum_memory", return_value=mock_continuum),
            patch(f"{_MOD}._analytics_limiter") as limiter,
        ):
            limiter.is_allowed.return_value = True
            result = await handler.handle(
                "/api/v1/knowledge/learning/stats",
                {},
                http_handler,
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["cross_debate_utility"]["avg_utility_score"] == 0.0

    @pytest.mark.asyncio
    async def test_continuum_with_none_km_adapter(self, handler, http_handler):
        """Gracefully handles continuum memory with _km_adapter = None."""
        mock_continuum = MagicMock()
        mock_continuum._km_adapter = None

        with (
            patch(f"{_MOD}.get_continuum_memory", return_value=mock_continuum),
            patch(f"{_MOD}._analytics_limiter") as limiter,
        ):
            limiter.is_allowed.return_value = True
            result = await handler.handle(
                "/api/v1/knowledge/learning/stats",
                {},
                http_handler,
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["cross_debate_utility"]["avg_utility_score"] == 0.0

    @pytest.mark.asyncio
    async def test_adapter_stats_missing_keys(self, handler, http_handler):
        """Adapter returning partial stats does not crash."""
        mock_adapter = MagicMock()
        mock_adapter.get_stats.return_value = {}  # Empty dict

        mock_continuum = MagicMock()
        mock_continuum._km_adapter = mock_adapter

        with (
            patch(f"{_MOD}.get_continuum_memory", return_value=mock_continuum),
            patch(f"{_MOD}._analytics_limiter") as limiter,
        ):
            limiter.is_allowed.return_value = True
            result = await handler.handle(
                "/api/v1/knowledge/learning/stats",
                {},
                http_handler,
            )

        assert _status(result) == 200
        body = _body(result)
        # Falls back to 0 for missing keys
        assert body["cross_debate_utility"]["avg_utility_score"] == 0.0


# =============================================================================
# Test: Summary error handling
# =============================================================================


class TestSummaryErrors:
    """Tests for analytics summary error paths."""

    @pytest.mark.asyncio
    async def test_summary_error_returns_500(self, handler, http_handler):
        """Returns 500 when summary aggregation fails with unexpected error."""
        with (
            patch.object(handler, "_get_mound_stats", side_effect=TypeError("unexpected")),
            patch(f"{_MOD}._analytics_limiter") as limiter,
        ):
            limiter.is_allowed.return_value = True
            result = await handler.handle(
                "/api/v1/knowledge/analytics/summary",
                {},
                http_handler,
            )

        assert _status(result) == 500


# =============================================================================
# Test: Multiple routes in sequence
# =============================================================================


class TestRouteDispatching:
    """Tests that routes are dispatched to the correct internal methods."""

    @pytest.mark.asyncio
    async def test_mound_stats_dispatches_correctly(self, handler, http_handler):
        """The mound stats path calls _get_mound_stats."""
        with (
            patch.object(
                handler, "_get_mound_stats", new_callable=AsyncMock, return_value=MagicMock(status_code=200, body=b'{}')
            ) as mock_method,
            patch(f"{_MOD}.emit_handler_event"),
            patch(f"{_MOD}._analytics_limiter") as limiter,
        ):
            limiter.is_allowed.return_value = True
            await handler.handle(
                "/api/v1/knowledge/mound/stats",
                {},
                http_handler,
            )
            mock_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_sharing_stats_dispatches_correctly(self, handler, http_handler):
        """The sharing stats path calls _get_sharing_stats."""
        with (
            patch.object(
                handler, "_get_sharing_stats", return_value=MagicMock(status_code=200, body=b'{}')
            ) as mock_method,
            patch(f"{_MOD}._analytics_limiter") as limiter,
        ):
            limiter.is_allowed.return_value = True
            await handler.handle(
                "/api/v1/knowledge/sharing/stats",
                {},
                http_handler,
            )
            mock_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_federation_stats_dispatches_correctly(self, handler, http_handler):
        """The federation stats path calls _get_federation_stats."""
        with (
            patch.object(
                handler, "_get_federation_stats", return_value=MagicMock(status_code=200, body=b'{}')
            ) as mock_method,
            patch(f"{_MOD}._analytics_limiter") as limiter,
        ):
            limiter.is_allowed.return_value = True
            await handler.handle(
                "/api/v1/knowledge/federation/stats",
                {},
                http_handler,
            )
            mock_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_summary_dispatches_correctly(self, handler, http_handler):
        """The analytics summary path calls _get_summary."""
        with (
            patch.object(
                handler, "_get_summary", new_callable=AsyncMock, return_value=MagicMock(status_code=200, body=b'{}')
            ) as mock_method,
            patch(f"{_MOD}._analytics_limiter") as limiter,
        ):
            limiter.is_allowed.return_value = True
            await handler.handle(
                "/api/v1/knowledge/analytics/summary",
                {},
                http_handler,
            )
            mock_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_learning_stats_dispatches_correctly(self, handler, http_handler):
        """The learning stats path calls _get_learning_stats."""
        with (
            patch.object(
                handler, "_get_learning_stats", return_value=MagicMock(status_code=200, body=b'{}')
            ) as mock_method,
            patch(f"{_MOD}._analytics_limiter") as limiter,
        ):
            limiter.is_allowed.return_value = True
            await handler.handle(
                "/api/v1/knowledge/learning/stats",
                {},
                http_handler,
            )
            mock_method.assert_called_once()


# =============================================================================
# Test: Event emission
# =============================================================================


class TestEventEmission:
    """Tests for handler event emission."""

    @pytest.mark.asyncio
    async def test_mound_stats_emits_queried_event(self, handler, http_handler, mock_mound):
        """Mound stats endpoint emits a QUERIED handler event."""
        with (
            patch(f"{_MOD}.knowledge_mound.get_knowledge_mound", return_value=mock_mound),
            patch(f"{_MOD}.emit_handler_event") as mock_emit,
            patch(f"{_MOD}._analytics_limiter") as limiter,
        ):
            limiter.is_allowed.return_value = True
            await handler.handle(
                "/api/v1/knowledge/mound/stats",
                {},
                http_handler,
            )

        mock_emit.assert_called_once()
        args, kwargs = mock_emit.call_args
        assert args[0] == "knowledge"
        assert args[1] is not None  # QUERIED constant
        assert args[2] == {"endpoint": "mound_stats"}
