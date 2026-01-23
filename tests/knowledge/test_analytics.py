"""
Tests for Knowledge Mound Analytics.

Tests the analytics data aggregation and API endpoints.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch


class TestAnalyticsCanHandle:
    """Test analytics handler path matching."""

    def test_can_handle_mound_stats(self):
        """Test that handler matches mound stats path."""
        from aragora.server.handlers.knowledge.analytics import AnalyticsHandler

        # Create minimal mock context
        mock_ctx = MagicMock()
        handler = AnalyticsHandler(mock_ctx)

        assert handler.can_handle("/api/v1/knowledge/mound/stats") is True

    def test_can_handle_sharing_stats(self):
        """Test that handler matches sharing stats path."""
        from aragora.server.handlers.knowledge.analytics import AnalyticsHandler

        mock_ctx = MagicMock()
        handler = AnalyticsHandler(mock_ctx)

        assert handler.can_handle("/api/v1/knowledge/sharing/stats") is True

    def test_can_handle_federation_stats(self):
        """Test that handler matches federation stats path."""
        from aragora.server.handlers.knowledge.analytics import AnalyticsHandler

        mock_ctx = MagicMock()
        handler = AnalyticsHandler(mock_ctx)

        assert handler.can_handle("/api/v1/knowledge/federation/stats") is True

    def test_can_handle_analytics_summary(self):
        """Test that handler matches analytics summary path."""
        from aragora.server.handlers.knowledge.analytics import AnalyticsHandler

        mock_ctx = MagicMock()
        handler = AnalyticsHandler(mock_ctx)

        assert handler.can_handle("/api/v1/knowledge/analytics/summary") is True

    def test_cannot_handle_other_paths(self):
        """Test that handler doesn't match unrelated paths."""
        from aragora.server.handlers.knowledge.analytics import AnalyticsHandler

        mock_ctx = MagicMock()
        handler = AnalyticsHandler(mock_ctx)

        assert handler.can_handle("/api/v1/other") is False


class TestAnalyticsInternalMethods:
    """Test analytics internal methods."""

    def test_get_mound_stats_returns_structure(self):
        """Test mound stats returns valid structure."""
        from aragora.server.handlers.knowledge.analytics import AnalyticsHandler

        mock_ctx = MagicMock()
        handler = AnalyticsHandler(mock_ctx)

        # Mock the knowledge mound and its get_stats method
        mock_stats = MagicMock()
        mock_stats.total_nodes = 10
        mock_stats.nodes_by_type = {"fact": 5, "opinion": 3}
        mock_stats.nodes_by_tier = {"fast": 2, "slow": 8}
        mock_stats.nodes_by_validation = {"verified": 7, "unverified": 3}
        mock_stats.total_relationships = 5
        mock_stats.relationships_by_type = {"supports": 3, "contradicts": 2}
        mock_stats.average_confidence = 0.75
        mock_stats.stale_nodes_count = 1

        mock_mound = MagicMock()
        mock_mound.get_stats = AsyncMock(return_value=mock_stats)

        with patch("aragora.knowledge.mound.get_knowledge_mound", return_value=mock_mound):
            result = handler._get_mound_stats(None)

        assert result is not None
        # HandlerResult is a dataclass with .body attribute
        assert hasattr(result, "body") or isinstance(result, dict)

        import json

        body = result.body if hasattr(result, "body") else result.get("body")
        data = json.loads(body)

        # Check basic structure exists
        assert "total_nodes" in data
        assert "nodes_by_type" in data

    def test_get_sharing_stats_returns_structure(self):
        """Test sharing stats returns valid structure."""
        from aragora.server.handlers.knowledge.analytics import AnalyticsHandler

        mock_ctx = MagicMock()
        handler = AnalyticsHandler(mock_ctx)

        result = handler._get_sharing_stats("ws_123", "user_456")

        assert result is not None

        import json

        body = result.body if hasattr(result, "body") else result.get("body")
        data = json.loads(body)

        assert "total_shared_items" in data
        assert "items_shared_with_me" in data

    def test_get_federation_stats_returns_structure(self):
        """Test federation stats returns valid structure."""
        from aragora.server.handlers.knowledge.analytics import AnalyticsHandler

        mock_ctx = MagicMock()
        handler = AnalyticsHandler(mock_ctx)

        result = handler._get_federation_stats("ws_123")

        assert result is not None

        import json

        body = result.body if hasattr(result, "body") else result.get("body")
        data = json.loads(body)

        assert "registered_regions" in data
        assert "active_schedules" in data


class TestFederationStatsIntegration:
    """Test federation stats integration with scheduler."""

    def test_federation_stats_from_scheduler(self):
        """Test getting stats from federation scheduler."""
        from aragora.knowledge.mound.ops.federation_scheduler import (
            FederationScheduler,
            FederationScheduleConfig,
            SyncMode,
        )

        scheduler = FederationScheduler()

        # Add some schedules
        for i in range(3):
            config = FederationScheduleConfig(
                name=f"sync_{i}",
                region_id=f"region_{i}",
                workspace_id="ws_test",
                sync_mode=SyncMode.BIDIRECTIONAL,
            )
            scheduler.add_schedule(config)

        stats = scheduler.get_stats()

        assert stats["schedules"]["total"] == 3
        assert stats["schedules"]["active"] == 3
        assert stats["running"] is False

    @pytest.mark.asyncio
    async def test_federation_stats_with_history(self):
        """Test stats after running syncs."""
        from aragora.knowledge.mound.ops.federation_scheduler import (
            FederationScheduler,
            FederationScheduleConfig,
            SyncRun,
            SyncMode,
        )

        mock_callback = AsyncMock(
            return_value={
                "items_pushed": 25,
                "items_pulled": 15,
                "items_conflicted": 0,
            }
        )

        scheduler = FederationScheduler(sync_callback=mock_callback)

        config = FederationScheduleConfig(
            name="test_sync",
            region_id="us-west-2",
            workspace_id="ws_test",
        )
        schedule = scheduler.add_schedule(config)

        # Run sync
        await scheduler.trigger_sync(schedule.schedule_id)

        # Check history
        history = scheduler.get_history()
        assert len(history) == 1
        assert history[0].items_pushed == 25
        assert history[0].items_pulled == 15

        # Check stats
        stats = scheduler.get_stats()
        assert stats["runs"]["total"] == 1
        assert stats["recent"]["successful"] == 1


class TestSharingStatsIntegration:
    """Test sharing stats integration with notification store."""

    def test_sharing_stats_from_notifications(self):
        """Test calculating sharing stats from notifications."""
        from aragora.knowledge.mound.notifications import (
            InAppNotificationStore,
            SharingNotification,
            NotificationType,
        )

        store = InAppNotificationStore()

        # Add some sharing notifications
        for i in range(5):
            store.add_notification(
                SharingNotification(
                    id=f"notif_{i}",
                    user_id="user_bob",
                    notification_type=NotificationType.ITEM_SHARED,
                    title=f"Shared {i}",
                    message=f"Item {i} shared",
                )
            )

        notifications = store.get_notifications("user_bob")

        # Count shared items
        shared_count = len(
            [n for n in notifications if n.notification_type == NotificationType.ITEM_SHARED]
        )

        assert shared_count == 5


class TestAnalyticsSummary:
    """Test combined analytics summary."""

    def test_summary_combines_all_stats(self):
        """Test that summary includes all stat types."""
        from aragora.server.handlers.knowledge.analytics import AnalyticsHandler

        mock_ctx = MagicMock()
        handler = AnalyticsHandler(mock_ctx)

        result = handler._get_summary("ws_123", "user_456")

        assert result is not None

        import json

        body = result.body if hasattr(result, "body") else result.get("body")
        data = json.loads(body)

        # Check all sections are present
        assert "mound" in data
        assert "sharing" in data
        assert "federation" in data
        assert data["workspace_id"] == "ws_123"


class TestRateLimiting:
    """Test analytics rate limiting."""

    def test_rate_limiter_exists(self):
        """Test that rate limiter is configured."""
        from aragora.server.handlers.knowledge.analytics import _analytics_limiter

        assert _analytics_limiter is not None
        # Rate limiter stores rate info differently - just check it exists
        assert hasattr(_analytics_limiter, "is_allowed")
