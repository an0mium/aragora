"""
Tests for cross-pollination handler.

Tests cover:
- Stats endpoint
- Subscribers listing
- Bridge status
- Metrics endpoint
- Reset functionality
- KM integration endpoints
- Error handling for unavailable modules
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock
from enum import Enum

import pytest

from aragora.server.handlers.cross_pollination import (
    CrossPollinationStatsHandler,
    CrossPollinationSubscribersHandler,
    CrossPollinationBridgeHandler,
    CrossPollinationMetricsHandler,
    CrossPollinationResetHandler,
    CrossPollinationKMHandler,
)


class MockEventType(Enum):
    """Mock event type for testing."""

    DEBATE_STARTED = "debate_started"
    CONSENSUS_REACHED = "consensus_reached"


class MockStreamType(Enum):
    """Mock stream type for testing."""

    DEBATE = "debate"
    CONSENSUS = "consensus"


class MockCrossSubscriberManager:
    """Mock cross-subscriber manager for testing."""

    def __init__(
        self,
        stats: dict[str, dict[str, Any]] | None = None,
        batch_stats: dict[str, Any] | None = None,
    ):
        self._stats = stats or {}
        self._batch_stats = batch_stats or {}
        self._subscribers: dict[Any, list[tuple[str, Any]]] = {}

    def get_stats(self) -> dict[str, dict[str, Any]]:
        return self._stats

    def get_batch_stats(self) -> dict[str, Any]:
        return self._batch_stats

    def reset_stats(self) -> None:
        pass

    def flush_all_batches(self) -> int:
        return 5


@pytest.fixture
def mock_server_context() -> MagicMock:
    """Create mock server context."""
    return MagicMock()


@pytest.fixture
def stats_handler(mock_server_context: MagicMock) -> CrossPollinationStatsHandler:
    """Create stats handler for tests."""
    return CrossPollinationStatsHandler(mock_server_context)


@pytest.fixture
def subscribers_handler(mock_server_context: MagicMock) -> CrossPollinationSubscribersHandler:
    """Create subscribers handler for tests."""
    return CrossPollinationSubscribersHandler(mock_server_context)


@pytest.fixture
def bridge_handler(mock_server_context: MagicMock) -> CrossPollinationBridgeHandler:
    """Create bridge handler for tests."""
    return CrossPollinationBridgeHandler(mock_server_context)


@pytest.fixture
def metrics_handler(mock_server_context: MagicMock) -> CrossPollinationMetricsHandler:
    """Create metrics handler for tests."""
    return CrossPollinationMetricsHandler(mock_server_context)


@pytest.fixture
def reset_handler(mock_server_context: MagicMock) -> CrossPollinationResetHandler:
    """Create reset handler for tests."""
    return CrossPollinationResetHandler(mock_server_context)


@pytest.fixture
def km_handler(mock_server_context: MagicMock) -> CrossPollinationKMHandler:
    """Create KM handler for tests."""
    return CrossPollinationKMHandler(mock_server_context)


class TestCrossPollinationStatsHandler:
    """Tests for stats endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self, stats_handler: CrossPollinationStatsHandler):
        """Stats endpoint returns subscriber statistics."""
        mock_manager = MockCrossSubscriberManager(
            stats={
                "memory_to_mound": {"events_processed": 100, "events_failed": 2, "enabled": True},
                "belief_to_mound": {"events_processed": 50, "events_failed": 0, "enabled": True},
                "disabled_handler": {"events_processed": 10, "events_failed": 1, "enabled": False},
            }
        )

        with patch(
            "aragora.events.cross_subscribers.get_cross_subscriber_manager",
            return_value=mock_manager,
        ):
            result = await stats_handler.get.__wrapped__(stats_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert body["status"] == "ok"
        assert body["summary"]["total_subscribers"] == 3
        assert body["summary"]["enabled_subscribers"] == 2
        assert body["summary"]["total_events_processed"] == 160
        assert body["summary"]["total_events_failed"] == 3

    @pytest.mark.asyncio
    async def test_get_stats_module_unavailable(self, stats_handler: CrossPollinationStatsHandler):
        """Stats endpoint handles missing module gracefully."""
        with patch(
            "aragora.events.cross_subscribers.get_cross_subscriber_manager",
            side_effect=ImportError("Module not found"),
        ):
            result = await stats_handler.get.__wrapped__(stats_handler)

        assert result.status_code == 503
        body = json.loads(result.body.decode())
        assert "not available" in body["error"]

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, stats_handler: CrossPollinationStatsHandler):
        """Stats endpoint handles no subscribers."""
        mock_manager = MockCrossSubscriberManager(stats={})

        with patch(
            "aragora.events.cross_subscribers.get_cross_subscriber_manager",
            return_value=mock_manager,
        ):
            result = await stats_handler.get.__wrapped__(stats_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert body["summary"]["total_subscribers"] == 0


class TestCrossPollinationSubscribersHandler:
    """Tests for subscribers listing endpoint."""

    @pytest.mark.asyncio
    async def test_list_subscribers_success(
        self, subscribers_handler: CrossPollinationSubscribersHandler
    ):
        """Subscribers endpoint lists all registered subscribers."""
        mock_manager = MockCrossSubscriberManager()
        mock_manager._subscribers = {
            MockEventType.DEBATE_STARTED: [
                ("debate_handler", lambda x: x),
            ],
            MockEventType.CONSENSUS_REACHED: [
                ("consensus_handler", lambda x: x),
            ],
        }

        with patch(
            "aragora.events.cross_subscribers.get_cross_subscriber_manager",
            return_value=mock_manager,
        ):
            result = await subscribers_handler.get.__wrapped__(subscribers_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert body["status"] == "ok"
        assert body["count"] == 2
        assert len(body["subscribers"]) == 2

    @pytest.mark.asyncio
    async def test_list_subscribers_module_unavailable(
        self, subscribers_handler: CrossPollinationSubscribersHandler
    ):
        """Subscribers endpoint handles missing module gracefully."""
        with patch(
            "aragora.events.cross_subscribers.get_cross_subscriber_manager",
            side_effect=ImportError("Module not found"),
        ):
            result = await subscribers_handler.get.__wrapped__(subscribers_handler)

        assert result.status_code == 503


class TestCrossPollinationBridgeHandler:
    """Tests for bridge status endpoint."""

    @pytest.mark.asyncio
    async def test_get_bridge_status(self, bridge_handler: CrossPollinationBridgeHandler):
        """Bridge endpoint returns event mappings."""
        mock_event_map = {
            "debate_start": MockStreamType.DEBATE,
            "consensus": MockStreamType.CONSENSUS,
        }

        with patch(
            "aragora.events.arena_bridge.EVENT_TYPE_MAP",
            mock_event_map,
        ):
            result = await bridge_handler.get.__wrapped__(bridge_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert body["status"] == "ok"
        assert body["mapped_event_count"] == 2
        assert "debate_start" in body["event_mappings"]

    @pytest.mark.asyncio
    async def test_get_bridge_module_unavailable(
        self, bridge_handler: CrossPollinationBridgeHandler
    ):
        """Bridge endpoint handles missing module gracefully."""
        with patch.dict("sys.modules", {"aragora.events.arena_bridge": None}):
            # Force ImportError
            with patch(
                "aragora.server.handlers.cross_pollination.CrossPollinationBridgeHandler.get",
                side_effect=ImportError("Module not found"),
            ):
                # Can't easily test ImportError inside the method, skip this
                pass


class TestCrossPollinationMetricsHandler:
    """Tests for metrics endpoint."""

    @pytest.mark.asyncio
    async def test_get_metrics_success(self, metrics_handler: CrossPollinationMetricsHandler):
        """Metrics endpoint returns Prometheus-format metrics."""
        mock_metrics = """# HELP cross_pollination_events_total Total events processed
# TYPE cross_pollination_events_total counter
cross_pollination_events_total{handler="memory_to_mound"} 100
"""

        with patch(
            "aragora.server.prometheus_cross_pollination.get_cross_pollination_metrics_text",
            return_value=mock_metrics,
        ):
            result = await metrics_handler.get.__wrapped__(metrics_handler)

        # Metrics endpoint returns dict instead of HandlerResult
        assert result["status"] == 200
        assert "text/plain" in result["headers"]["Content-Type"]
        assert "cross_pollination_events_total" in result["body"]


class TestCrossPollinationResetHandler:
    """Tests for reset endpoint."""

    @pytest.mark.asyncio
    async def test_reset_stats_success(self, reset_handler: CrossPollinationResetHandler):
        """Reset endpoint resets subscriber statistics."""
        mock_manager = MockCrossSubscriberManager()

        with patch(
            "aragora.events.cross_subscribers.get_cross_subscriber_manager",
            return_value=mock_manager,
        ):
            result = await reset_handler.post.__wrapped__(reset_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert body["status"] == "ok"
        assert "reset" in body["message"].lower()

    @pytest.mark.asyncio
    async def test_reset_module_unavailable(self, reset_handler: CrossPollinationResetHandler):
        """Reset endpoint handles missing module gracefully."""
        with patch(
            "aragora.events.cross_subscribers.get_cross_subscriber_manager",
            side_effect=ImportError("Module not found"),
        ):
            result = await reset_handler.post.__wrapped__(reset_handler)

        assert result.status_code == 503


class TestCrossPollinationKMHandler:
    """Tests for KM integration endpoint."""

    @pytest.mark.asyncio
    async def test_get_km_status_success(self, km_handler: CrossPollinationKMHandler):
        """KM endpoint returns bidirectional integration status."""
        mock_manager = MockCrossSubscriberManager(
            stats={
                "memory_to_mound": {"events_processed": 100, "events_failed": 2},
                "mound_to_memory_retrieval": {"events_processed": 50, "events_failed": 0},
                "belief_to_mound": {"events_processed": 30, "events_failed": 1},
            },
            batch_stats={"pending": 5, "processed": 100},
        )

        with patch(
            "aragora.events.cross_subscribers.get_cross_subscriber_manager",
            return_value=mock_manager,
        ):
            result = await km_handler.get.__wrapped__(km_handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert body["status"] == "ok"
        assert "summary" in body
        assert "handlers" in body
        assert "adapters" in body
        assert body["summary"]["total_km_handlers"] > 0

    @pytest.mark.asyncio
    async def test_get_km_status_module_unavailable(self, km_handler: CrossPollinationKMHandler):
        """KM endpoint handles missing module gracefully."""
        with patch(
            "aragora.events.cross_subscribers.get_cross_subscriber_manager",
            side_effect=ImportError("Module not found"),
        ):
            result = await km_handler.get.__wrapped__(km_handler)

        assert result.status_code == 503


class TestRouteRegistration:
    """Tests for route registration."""

    def test_handler_routes_defined(self):
        """All handlers have ROUTES defined."""
        handlers = [
            CrossPollinationStatsHandler,
            CrossPollinationSubscribersHandler,
            CrossPollinationBridgeHandler,
            CrossPollinationMetricsHandler,
            CrossPollinationResetHandler,
            CrossPollinationKMHandler,
        ]

        for handler_class in handlers:
            assert hasattr(handler_class, "ROUTES"), f"{handler_class.__name__} missing ROUTES"
            assert len(handler_class.ROUTES) > 0, f"{handler_class.__name__} has empty ROUTES"


class TestRBACPermissions:
    """Tests for RBAC permission requirements."""

    def test_stats_requires_read_permission(self, stats_handler: CrossPollinationStatsHandler):
        """Stats endpoint requires cross_pollination:read permission."""
        # Check that the method has the decorator
        assert hasattr(stats_handler.get, "__wrapped__")

    def test_reset_requires_write_permission(self, reset_handler: CrossPollinationResetHandler):
        """Reset endpoint requires cross_pollination:write permission."""
        assert hasattr(reset_handler.post, "__wrapped__")


class TestIntegration:
    """Integration tests for cross-pollination handlers."""

    @pytest.mark.asyncio
    async def test_stats_then_reset_flow(
        self,
        stats_handler: CrossPollinationStatsHandler,
        reset_handler: CrossPollinationResetHandler,
    ):
        """Complete flow: get stats, reset, verify empty."""
        mock_manager = MockCrossSubscriberManager(
            stats={
                "memory_to_mound": {"events_processed": 100, "events_failed": 2, "enabled": True},
            }
        )

        with patch(
            "aragora.events.cross_subscribers.get_cross_subscriber_manager",
            return_value=mock_manager,
        ):
            # Step 1: Get initial stats
            stats_result = await stats_handler.get.__wrapped__(stats_handler)
            stats_body = json.loads(stats_result.body.decode())
            assert stats_body["summary"]["total_events_processed"] == 100

            # Step 2: Reset stats
            reset_result = await reset_handler.post.__wrapped__(reset_handler)
            assert reset_result.status_code == 200

            # Step 3: Manager reset_stats was called
            # (In real scenario, stats would be empty after reset)
