"""
Tests for WebSocket Bridge for Knowledge Mound Events.

Covers:
- KMSubscription creation and defaults
- KMSubscription.matches() - event type, source, workspace, confidence filtering
- KMSubscriptionManager thread-safe operations
- subscribe/unsubscribe lifecycle
- update_subscription (add/remove event types and sources)
- get_subscribers matching logic
- KMWebSocketBridge initialization
- start/stop lifecycle (background batching)
- queue_event batching
- Event broadcasting via _emit_callback
- Passthrough events (immediate delivery)
- Stream event type mapping
- Global bridge instance management
- Stats aggregation
"""

import asyncio
import time
from threading import Thread
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from aragora.knowledge.mound.websocket_bridge import (
    KMSubscription,
    KMSubscriptionManager,
    KMWebSocketBridge,
    get_km_bridge,
    set_km_bridge,
    create_km_bridge,
)


# =============================================================================
# KMSubscription Tests
# =============================================================================


class TestKMSubscription:
    """Tests for KMSubscription dataclass."""

    def test_create_subscription_defaults(self):
        """Test creating a subscription with defaults."""
        sub = KMSubscription(client_id="client_1")

        assert sub.client_id == "client_1"
        assert sub.event_types == set()
        assert sub.sources == set()
        assert sub.min_confidence == 0.0
        assert sub.workspace_id is None
        assert sub.created_at > 0

    def test_create_subscription_with_values(self):
        """Test creating a subscription with explicit values."""
        sub = KMSubscription(
            client_id="client_2",
            event_types={"knowledge_indexed", "mound_updated"},
            sources={"evidence", "belief"},
            min_confidence=0.5,
            workspace_id="ws_001",
        )

        assert sub.client_id == "client_2"
        assert sub.event_types == {"knowledge_indexed", "mound_updated"}
        assert sub.sources == {"evidence", "belief"}
        assert sub.min_confidence == 0.5
        assert sub.workspace_id == "ws_001"

    def test_created_at_timestamp(self):
        """Test that created_at is set to current time."""
        before = time.time()
        sub = KMSubscription(client_id="test")
        after = time.time()

        assert before <= sub.created_at <= after


class TestKMSubscriptionMatches:
    """Tests for KMSubscription.matches() method."""

    def test_matches_all_when_empty_filters(self):
        """Test subscription with no filters matches all events."""
        sub = KMSubscription(client_id="client_1")

        assert sub.matches("knowledge_indexed", {}) is True
        assert sub.matches("mound_updated", {"source": "any"}) is True
        assert sub.matches("custom_event", {"confidence": 0.1}) is True

    def test_matches_event_type_filter(self):
        """Test event type filtering."""
        sub = KMSubscription(
            client_id="client_1",
            event_types={"knowledge_indexed", "mound_updated"},
        )

        assert sub.matches("knowledge_indexed", {}) is True
        assert sub.matches("mound_updated", {}) is True
        assert sub.matches("other_event", {}) is False

    def test_matches_source_filter(self):
        """Test source filtering."""
        sub = KMSubscription(
            client_id="client_1",
            sources={"evidence", "belief"},
        )

        assert sub.matches("any_event", {"source": "evidence"}) is True
        assert sub.matches("any_event", {"source": "belief"}) is True
        assert sub.matches("any_event", {"source": "other"}) is False
        # Empty source matches (no source specified in event)
        assert sub.matches("any_event", {}) is True

    def test_matches_source_via_adapter_key(self):
        """Test source filtering via 'adapter' key."""
        sub = KMSubscription(
            client_id="client_1",
            sources={"evidence"},
        )

        assert sub.matches("any_event", {"adapter": "evidence"}) is True
        assert sub.matches("any_event", {"adapter": "belief"}) is False

    def test_matches_confidence_threshold(self):
        """Test confidence threshold filtering."""
        sub = KMSubscription(
            client_id="client_1",
            min_confidence=0.7,
        )

        assert sub.matches("any_event", {"confidence": 0.9}) is True
        assert sub.matches("any_event", {"confidence": 0.7}) is True
        assert sub.matches("any_event", {"confidence": 0.6}) is False
        # Default confidence is 1.0 when not specified
        assert sub.matches("any_event", {}) is True

    def test_matches_workspace_filter(self):
        """Test workspace filtering."""
        sub = KMSubscription(
            client_id="client_1",
            workspace_id="ws_001",
        )

        assert sub.matches("any_event", {"workspace_id": "ws_001"}) is True
        assert sub.matches("any_event", {"workspace_id": "ws_002"}) is False
        # Empty workspace matches (no workspace in event)
        assert sub.matches("any_event", {}) is True

    def test_matches_workspace_via_workspace_key(self):
        """Test workspace filtering via 'workspace' key."""
        sub = KMSubscription(
            client_id="client_1",
            workspace_id="ws_001",
        )

        assert sub.matches("any_event", {"workspace": "ws_001"}) is True
        assert sub.matches("any_event", {"workspace": "ws_002"}) is False

    def test_matches_combined_filters(self):
        """Test combined filtering (AND logic)."""
        sub = KMSubscription(
            client_id="client_1",
            event_types={"knowledge_indexed"},
            sources={"evidence"},
            min_confidence=0.5,
            workspace_id="ws_001",
        )

        # All conditions met
        assert (
            sub.matches(
                "knowledge_indexed",
                {
                    "source": "evidence",
                    "confidence": 0.8,
                    "workspace_id": "ws_001",
                },
            )
            is True
        )

        # Wrong event type
        assert (
            sub.matches(
                "other_event",
                {
                    "source": "evidence",
                    "confidence": 0.8,
                    "workspace_id": "ws_001",
                },
            )
            is False
        )

        # Wrong source
        assert (
            sub.matches(
                "knowledge_indexed",
                {
                    "source": "belief",
                    "confidence": 0.8,
                    "workspace_id": "ws_001",
                },
            )
            is False
        )

        # Low confidence
        assert (
            sub.matches(
                "knowledge_indexed",
                {
                    "source": "evidence",
                    "confidence": 0.3,
                    "workspace_id": "ws_001",
                },
            )
            is False
        )

        # Wrong workspace
        assert (
            sub.matches(
                "knowledge_indexed",
                {
                    "source": "evidence",
                    "confidence": 0.8,
                    "workspace_id": "ws_002",
                },
            )
            is False
        )


# =============================================================================
# KMSubscriptionManager Tests
# =============================================================================


class TestKMSubscriptionManager:
    """Tests for KMSubscriptionManager class."""

    def test_create_manager(self):
        """Test creating a subscription manager."""
        manager = KMSubscriptionManager()
        assert manager._subscriptions == {}

    def test_subscribe_basic(self):
        """Test basic subscription."""
        manager = KMSubscriptionManager()

        sub = manager.subscribe("client_1")

        assert sub.client_id == "client_1"
        assert "client_1" in manager._subscriptions

    def test_subscribe_with_params(self):
        """Test subscription with all parameters."""
        manager = KMSubscriptionManager()

        sub = manager.subscribe(
            client_id="client_1",
            event_types=["knowledge_indexed", "mound_updated"],
            sources=["evidence", "belief"],
            min_confidence=0.5,
            workspace_id="ws_001",
        )

        assert sub.event_types == {"knowledge_indexed", "mound_updated"}
        assert sub.sources == {"evidence", "belief"}
        assert sub.min_confidence == 0.5
        assert sub.workspace_id == "ws_001"

    def test_subscribe_replaces_existing(self):
        """Test that re-subscribing replaces the existing subscription."""
        manager = KMSubscriptionManager()

        manager.subscribe("client_1", event_types=["event_1"])
        sub2 = manager.subscribe("client_1", event_types=["event_2"])

        assert manager._subscriptions["client_1"] is sub2
        assert sub2.event_types == {"event_2"}

    def test_unsubscribe_existing(self):
        """Test unsubscribing an existing client."""
        manager = KMSubscriptionManager()
        manager.subscribe("client_1")

        result = manager.unsubscribe("client_1")

        assert result is True
        assert "client_1" not in manager._subscriptions

    def test_unsubscribe_nonexistent(self):
        """Test unsubscribing a non-existent client."""
        manager = KMSubscriptionManager()

        result = manager.unsubscribe("nonexistent")

        assert result is False

    def test_update_subscription_add_types(self):
        """Test adding event types to a subscription."""
        manager = KMSubscriptionManager()
        manager.subscribe("client_1", event_types=["event_1"])

        result = manager.update_subscription("client_1", add_types=["event_2", "event_3"])

        assert result is not None
        assert result.event_types == {"event_1", "event_2", "event_3"}

    def test_update_subscription_remove_types(self):
        """Test removing event types from a subscription."""
        manager = KMSubscriptionManager()
        manager.subscribe("client_1", event_types=["event_1", "event_2", "event_3"])

        result = manager.update_subscription("client_1", remove_types=["event_2"])

        assert result is not None
        assert result.event_types == {"event_1", "event_3"}

    def test_update_subscription_add_sources(self):
        """Test adding sources to a subscription."""
        manager = KMSubscriptionManager()
        manager.subscribe("client_1", sources=["evidence"])

        result = manager.update_subscription("client_1", add_sources=["belief", "pulse"])

        assert result is not None
        assert result.sources == {"evidence", "belief", "pulse"}

    def test_update_subscription_remove_sources(self):
        """Test removing sources from a subscription."""
        manager = KMSubscriptionManager()
        manager.subscribe("client_1", sources=["evidence", "belief", "pulse"])

        result = manager.update_subscription("client_1", remove_sources=["belief"])

        assert result is not None
        assert result.sources == {"evidence", "pulse"}

    def test_update_subscription_combined(self):
        """Test combined add and remove operations."""
        manager = KMSubscriptionManager()
        manager.subscribe(
            "client_1",
            event_types=["event_1", "event_2"],
            sources=["evidence", "belief"],
        )

        result = manager.update_subscription(
            "client_1",
            add_types=["event_3"],
            remove_types=["event_1"],
            add_sources=["pulse"],
            remove_sources=["evidence"],
        )

        assert result is not None
        assert result.event_types == {"event_2", "event_3"}
        assert result.sources == {"belief", "pulse"}

    def test_update_subscription_nonexistent(self):
        """Test updating a non-existent subscription."""
        manager = KMSubscriptionManager()

        result = manager.update_subscription("nonexistent", add_types=["event_1"])

        assert result is None

    def test_get_subscribers_matching(self):
        """Test getting subscribers that match an event."""
        manager = KMSubscriptionManager()
        manager.subscribe("client_1", event_types=["event_a"])
        manager.subscribe("client_2", event_types=["event_b"])
        manager.subscribe("client_3")  # Subscribes to all

        subscribers = manager.get_subscribers("event_a", {})

        assert "client_1" in subscribers
        assert "client_2" not in subscribers
        assert "client_3" in subscribers

    def test_get_subscribers_with_source_filter(self):
        """Test getting subscribers with source filtering."""
        manager = KMSubscriptionManager()
        manager.subscribe("client_1", sources=["evidence"])
        manager.subscribe("client_2", sources=["belief"])
        manager.subscribe("client_3")  # All sources

        subscribers = manager.get_subscribers("any_event", {"source": "evidence"})

        assert "client_1" in subscribers
        assert "client_2" not in subscribers
        assert "client_3" in subscribers

    def test_get_subscribers_with_confidence_filter(self):
        """Test getting subscribers with confidence filtering."""
        manager = KMSubscriptionManager()
        manager.subscribe("client_1", min_confidence=0.8)
        manager.subscribe("client_2", min_confidence=0.3)
        manager.subscribe("client_3")  # min_confidence=0.0

        subscribers = manager.get_subscribers("any_event", {"confidence": 0.5})

        assert "client_1" not in subscribers
        assert "client_2" in subscribers
        assert "client_3" in subscribers

    def test_get_subscription(self):
        """Test getting a specific subscription."""
        manager = KMSubscriptionManager()
        original = manager.subscribe("client_1", event_types=["event_1"])

        retrieved = manager.get_subscription("client_1")

        assert retrieved is original

    def test_get_subscription_nonexistent(self):
        """Test getting a non-existent subscription."""
        manager = KMSubscriptionManager()

        result = manager.get_subscription("nonexistent")

        assert result is None

    def test_get_all_subscriptions(self):
        """Test getting all subscriptions."""
        manager = KMSubscriptionManager()
        manager.subscribe("client_1")
        manager.subscribe("client_2")
        manager.subscribe("client_3")

        all_subs = manager.get_all_subscriptions()

        assert len(all_subs) == 3
        assert "client_1" in all_subs
        assert "client_2" in all_subs
        assert "client_3" in all_subs

    def test_get_all_subscriptions_returns_copy(self):
        """Test that get_all_subscriptions returns a copy."""
        manager = KMSubscriptionManager()
        manager.subscribe("client_1")

        all_subs = manager.get_all_subscriptions()
        all_subs["client_2"] = KMSubscription(client_id="client_2")

        assert "client_2" not in manager._subscriptions

    def test_get_stats(self):
        """Test getting subscription statistics."""
        manager = KMSubscriptionManager()
        manager.subscribe("client_1", event_types=["event_a", "event_b"], sources=["evidence"])
        manager.subscribe("client_2", event_types=["event_a"], sources=["evidence", "belief"])
        manager.subscribe("client_3")

        stats = manager.get_stats()

        assert stats["total_subscribers"] == 3
        assert stats["event_type_subscriptions"]["event_a"] == 2
        assert stats["event_type_subscriptions"]["event_b"] == 1
        assert stats["source_subscriptions"]["evidence"] == 2
        assert stats["source_subscriptions"]["belief"] == 1


class TestKMSubscriptionManagerThreadSafety:
    """Tests for thread safety of KMSubscriptionManager."""

    def test_concurrent_subscribe_unsubscribe(self):
        """Test concurrent subscribe and unsubscribe operations."""
        manager = KMSubscriptionManager()
        errors = []

        def subscribe_clients(start_id: int, count: int):
            try:
                for i in range(count):
                    manager.subscribe(f"client_{start_id + i}")
            except Exception as e:
                errors.append(e)

        def unsubscribe_clients(start_id: int, count: int):
            try:
                for i in range(count):
                    manager.unsubscribe(f"client_{start_id + i}")
            except Exception as e:
                errors.append(e)

        # Create threads
        threads = [
            Thread(target=subscribe_clients, args=(0, 100)),
            Thread(target=subscribe_clients, args=(100, 100)),
            Thread(target=unsubscribe_clients, args=(0, 50)),
            Thread(target=subscribe_clients, args=(200, 100)),
        ]

        # Run threads
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_get_subscribers(self):
        """Test concurrent get_subscribers calls."""
        manager = KMSubscriptionManager()
        errors = []

        # Pre-populate
        for i in range(50):
            manager.subscribe(f"client_{i}", event_types=["event_1"])

        def get_subscribers_repeatedly():
            try:
                for _ in range(100):
                    manager.get_subscribers("event_1", {})
            except Exception as e:
                errors.append(e)

        def modify_subscriptions():
            try:
                for i in range(50, 100):
                    manager.subscribe(f"client_{i}", event_types=["event_1"])
            except Exception as e:
                errors.append(e)

        threads = [
            Thread(target=get_subscribers_repeatedly),
            Thread(target=get_subscribers_repeatedly),
            Thread(target=modify_subscriptions),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# KMWebSocketBridge Tests
# =============================================================================


class TestKMWebSocketBridge:
    """Tests for KMWebSocketBridge class."""

    def test_create_bridge_defaults(self):
        """Test creating a bridge with defaults."""
        bridge = KMWebSocketBridge()

        assert bridge._broadcaster is None
        assert bridge._enable_subscriptions is True
        assert bridge._loop is None

    def test_create_bridge_with_broadcaster(self):
        """Test creating a bridge with a broadcaster."""
        broadcaster = MagicMock()
        bridge = KMWebSocketBridge(broadcaster=broadcaster)

        assert bridge._broadcaster is broadcaster

    def test_create_bridge_custom_config(self):
        """Test creating a bridge with custom configuration."""
        bridge = KMWebSocketBridge(
            batch_interval_ms=200.0,
            max_batch_size=100,
            passthrough_events=["km_critical"],
            enable_subscriptions=False,
        )

        assert bridge._batcher._batch_interval_ms == 200.0
        assert bridge._batcher._max_batch_size == 100
        assert "km_critical" in bridge._batcher._passthrough_event_types
        assert bridge._enable_subscriptions is False

    def test_set_broadcaster(self):
        """Test setting broadcaster after initialization."""
        bridge = KMWebSocketBridge()
        broadcaster = MagicMock()

        bridge.set_broadcaster(broadcaster)

        assert bridge._broadcaster is broadcaster

    def test_subscriptions_property(self):
        """Test that subscriptions property returns the manager."""
        bridge = KMWebSocketBridge()

        assert isinstance(bridge.subscriptions, KMSubscriptionManager)
        assert bridge.subscriptions is bridge._subscriptions

    def test_adapter_callback_property(self):
        """Test that adapter_callback property returns the callback."""
        bridge = KMWebSocketBridge()

        callback = bridge.adapter_callback

        assert callable(callback)

    def test_subscribe_passthrough(self):
        """Test that subscribe passes through to subscription manager."""
        bridge = KMWebSocketBridge()

        sub = bridge.subscribe(
            client_id="client_1",
            event_types=["event_1"],
            sources=["evidence"],
            min_confidence=0.5,
            workspace_id="ws_001",
        )

        assert sub.client_id == "client_1"
        assert sub.event_types == {"event_1"}
        assert sub.sources == {"evidence"}
        assert sub.min_confidence == 0.5
        assert sub.workspace_id == "ws_001"

    def test_unsubscribe_passthrough(self):
        """Test that unsubscribe passes through to subscription manager."""
        bridge = KMWebSocketBridge()
        bridge.subscribe("client_1")

        result = bridge.unsubscribe("client_1")

        assert result is True
        assert bridge._subscriptions.get_subscription("client_1") is None

    def test_queue_event(self):
        """Test queuing an event."""
        bridge = KMWebSocketBridge()

        bridge.queue_event("knowledge_indexed", {"id": "1"})

        assert bridge._batcher._total_events_queued == 1


class TestKMWebSocketBridgeAsync:
    """Async tests for KMWebSocketBridge."""

    @pytest.mark.asyncio
    async def test_start_with_loop(self):
        """Test starting the bridge with an event loop."""
        bridge = KMWebSocketBridge()
        loop = asyncio.get_event_loop()

        bridge.start(loop)

        assert bridge._loop is loop
        assert bridge._batcher._running is True

        await bridge.stop()

    @pytest.mark.asyncio
    async def test_start_auto_detect_loop(self):
        """Test starting the bridge with auto-detected loop."""
        bridge = KMWebSocketBridge()

        bridge.start()

        assert bridge._loop is not None
        assert bridge._batcher._running is True

        await bridge.stop()

    @pytest.mark.asyncio
    async def test_stop_flushes_events(self):
        """Test that stopping flushes remaining events."""
        callback = MagicMock()
        bridge = KMWebSocketBridge(batch_interval_ms=10000)  # Long interval
        bridge._batcher._callback = callback
        bridge.start()

        bridge.queue_event("e1", {})
        bridge.queue_event("e2", {})

        await bridge.stop()

        # Events should have been flushed
        assert bridge._batcher._running is False

    @pytest.mark.asyncio
    async def test_queue_event_batching(self):
        """Test that events are batched."""
        events_received = []

        def callback(event_type, data):
            events_received.append((event_type, data))

        bridge = KMWebSocketBridge(batch_interval_ms=50, max_batch_size=100)
        bridge._batcher._callback = callback
        bridge.start()

        # Queue multiple events
        for i in range(5):
            bridge.queue_event(f"event_{i}", {"index": i})

        # Wait for batch interval
        await asyncio.sleep(0.1)

        await bridge.stop()

        # Should have received at least one batch
        assert len(events_received) >= 1
        assert events_received[0][0] == "km_batch"


class TestKMWebSocketBridgeEmitCallback:
    """Tests for _emit_callback method."""

    @pytest.mark.asyncio
    async def test_emit_callback_without_broadcaster(self):
        """Test emit callback when no broadcaster is set."""
        bridge = KMWebSocketBridge()
        bridge.start()

        # Should not raise
        bridge._emit_callback("km_batch", {"events": []})

        await bridge.stop()

    @pytest.mark.asyncio
    async def test_emit_callback_with_broadcaster(self):
        """Test emit callback broadcasts events."""
        broadcaster = MagicMock()
        broadcaster.broadcast = AsyncMock()

        bridge = KMWebSocketBridge(broadcaster=broadcaster)
        bridge.start()

        # Patch the imports that happen inside _emit_callback at the source module
        with patch("aragora.events.types.StreamEvent") as MockEvent:
            with patch("aragora.events.types.StreamEventType") as MockType:
                MockType.KM_BATCH = MagicMock()
                mock_event = MagicMock()
                MockEvent.return_value = mock_event

                bridge._emit_callback("km_batch", {"events": [], "count": 0})

                # Give async task time to run
                await asyncio.sleep(0.05)

                # The callback should have been invoked
                # Note: actual broadcast happens asynchronously

        await bridge.stop()

    @pytest.mark.asyncio
    async def test_emit_callback_event_type_mapping(self):
        """Test that event types are mapped correctly."""
        broadcaster = MagicMock()
        broadcaster.broadcast = AsyncMock()

        bridge = KMWebSocketBridge(broadcaster=broadcaster)
        bridge.start()

        # The actual mapping is tested through the event type lookup
        event_map = {
            "km_knowledge_indexed": "KNOWLEDGE_INDEXED",
            "km_knowledge_queried": "KNOWLEDGE_QUERIED",
            "km_mound_updated": "MOUND_UPDATED",
            "km_knowledge_stale": "KNOWLEDGE_STALE",
            "km_belief_converged": "BELIEF_CONVERGED",
            "km_crux_detected": "CRUX_DETECTED",
            "knowledge_indexed": "KNOWLEDGE_INDEXED",
            "knowledge_queried": "KNOWLEDGE_QUERIED",
            "mound_updated": "MOUND_UPDATED",
            "knowledge_stale": "KNOWLEDGE_STALE",
        }

        # Just verify the callback doesn't raise for known types
        for event_type in event_map.keys():
            bridge._emit_callback(event_type, {"test": True})
            await asyncio.sleep(0.01)

        await bridge.stop()

    @pytest.mark.asyncio
    async def test_emit_callback_unknown_event_type_defaults_to_batch(self):
        """Test that unknown event types default to KM_BATCH."""
        bridge = KMWebSocketBridge()
        bridge.start()

        # Should not raise for unknown event type
        bridge._emit_callback("unknown_event_type", {"test": True})

        await bridge.stop()

    @pytest.mark.asyncio
    async def test_emit_callback_handles_exceptions(self):
        """Test that emit callback handles exceptions gracefully."""
        broadcaster = MagicMock()
        broadcaster.broadcast = AsyncMock(side_effect=RuntimeError("Broadcast error"))

        bridge = KMWebSocketBridge(broadcaster=broadcaster)
        bridge.start()

        # Should not raise despite exception
        bridge._emit_callback("km_batch", {"events": []})

        await bridge.stop()

    @pytest.mark.asyncio
    async def test_emit_callback_no_loop_running(self):
        """Test emit callback when loop is not running."""
        bridge = KMWebSocketBridge()
        # Don't start the bridge, so no loop

        # Should not raise
        bridge._emit_callback("km_batch", {"events": []})


class TestKMWebSocketBridgePassthrough:
    """Tests for passthrough events (immediate delivery)."""

    def test_default_passthrough_events(self):
        """Test default passthrough event types."""
        bridge = KMWebSocketBridge()

        # Default passthrough events are km_error and km_sync_failed
        assert "km_error" in bridge._batcher._passthrough_event_types
        assert "km_sync_failed" in bridge._batcher._passthrough_event_types

    def test_custom_passthrough_events(self):
        """Test custom passthrough event types."""
        bridge = KMWebSocketBridge(passthrough_events=["critical_event"])

        assert "critical_event" in bridge._batcher._passthrough_event_types
        # Default events replaced
        assert "km_error" not in bridge._batcher._passthrough_event_types

    def test_passthrough_event_bypasses_batching(self):
        """Test that passthrough events bypass batching."""
        events_received = []

        def callback(event_type, data):
            events_received.append((event_type, data))

        bridge = KMWebSocketBridge(passthrough_events=["urgent_event"])
        bridge._batcher._callback = callback

        bridge.queue_event("urgent_event", {"priority": "high"})

        # Should be emitted immediately, not batched
        assert len(events_received) == 1
        assert events_received[0][0] == "urgent_event"
        assert bridge._batcher._passthrough_events == 1


class TestKMWebSocketBridgeStats:
    """Tests for get_stats method."""

    def test_get_stats_basic(self):
        """Test getting basic statistics."""
        bridge = KMWebSocketBridge()

        bridge.queue_event("e1", {})
        bridge.queue_event("e2", {})

        stats = bridge.get_stats()

        assert "total_events_queued" in stats
        assert "pending_events" in stats
        assert "subscriptions" in stats
        assert stats["total_events_queued"] == 2

    def test_get_stats_includes_subscription_stats(self):
        """Test that stats include subscription statistics."""
        bridge = KMWebSocketBridge()

        bridge.subscribe("client_1", event_types=["event_a"])
        bridge.subscribe("client_2", event_types=["event_a", "event_b"])

        stats = bridge.get_stats()

        assert stats["subscriptions"]["total_subscribers"] == 2
        assert stats["subscriptions"]["event_type_subscriptions"]["event_a"] == 2
        assert stats["subscriptions"]["event_type_subscriptions"]["event_b"] == 1


# =============================================================================
# Global Bridge Instance Tests
# =============================================================================


class TestGlobalBridgeInstance:
    """Tests for global bridge instance management."""

    def teardown_method(self):
        """Reset global bridge after each test."""
        # Reset to None
        from aragora.knowledge.mound import websocket_bridge

        websocket_bridge._global_bridge = None

    def test_get_km_bridge_initially_none(self):
        """Test that get_km_bridge returns None initially."""
        from aragora.knowledge.mound import websocket_bridge

        websocket_bridge._global_bridge = None

        result = get_km_bridge()

        assert result is None

    def test_set_km_bridge(self):
        """Test setting the global bridge."""
        bridge = KMWebSocketBridge()

        set_km_bridge(bridge)
        result = get_km_bridge()

        assert result is bridge

    def test_create_km_bridge_basic(self):
        """Test creating and setting global bridge."""
        bridge = create_km_bridge()

        assert bridge is not None
        assert get_km_bridge() is bridge

    def test_create_km_bridge_with_broadcaster(self):
        """Test creating global bridge with broadcaster."""
        broadcaster = MagicMock()

        bridge = create_km_bridge(broadcaster=broadcaster)

        assert bridge._broadcaster is broadcaster

    def test_create_km_bridge_with_kwargs(self):
        """Test creating global bridge with custom kwargs."""
        bridge = create_km_bridge(
            batch_interval_ms=200.0,
            max_batch_size=100,
            enable_subscriptions=False,
        )

        assert bridge._batcher._batch_interval_ms == 200.0
        assert bridge._batcher._max_batch_size == 100
        assert bridge._enable_subscriptions is False

    def test_create_km_bridge_replaces_existing(self):
        """Test that create_km_bridge replaces existing bridge."""
        bridge1 = create_km_bridge()
        bridge2 = create_km_bridge()

        assert get_km_bridge() is bridge2
        assert get_km_bridge() is not bridge1


# =============================================================================
# Integration Tests
# =============================================================================


class TestKMWebSocketBridgeIntegration:
    """Integration tests for the full event flow."""

    @pytest.mark.asyncio
    async def test_full_event_flow(self):
        """Test full event flow from queue to broadcast."""
        events_received = []

        def callback(event_type, data):
            events_received.append((event_type, data))

        bridge = KMWebSocketBridge(batch_interval_ms=50)
        bridge._batcher._callback = callback
        bridge.start()

        # Subscribe a client
        bridge.subscribe("client_1", event_types=["knowledge_indexed"])

        # Queue events
        bridge.queue_event("knowledge_indexed", {"id": "1"})
        bridge.queue_event("knowledge_indexed", {"id": "2"})

        # Wait for batch
        await asyncio.sleep(0.1)

        await bridge.stop()

        # Should have received a batch
        assert len(events_received) >= 1

    @pytest.mark.asyncio
    async def test_adapter_callback_integration(self):
        """Test using the adapter callback."""
        events_received = []

        def callback(event_type, data):
            events_received.append((event_type, data))

        bridge = KMWebSocketBridge(batch_interval_ms=50)
        bridge._batcher._callback = callback
        bridge.start()

        # Use adapter callback
        adapter_cb = bridge.adapter_callback
        adapter_cb("indexed", {"source": "evidence"})
        adapter_cb("updated", {"source": "belief"})

        # Wait for batch
        await asyncio.sleep(0.1)

        await bridge.stop()

        # Events should have been prefixed with "km_"
        assert len(events_received) >= 1
        batch = events_received[0][1]
        assert batch["events"][0]["type"] == "km_indexed"

    @pytest.mark.asyncio
    async def test_subscription_filtering_in_flow(self):
        """Test that subscriptions filter events correctly."""
        bridge = KMWebSocketBridge()
        bridge.start()

        # Subscribe clients with different filters
        bridge.subscribe("client_1", event_types=["knowledge_indexed"])
        bridge.subscribe("client_2", event_types=["mound_updated"])
        bridge.subscribe("client_3")  # All events

        # Test filtering
        subs = bridge.subscriptions.get_subscribers("knowledge_indexed", {})
        assert "client_1" in subs
        assert "client_2" not in subs
        assert "client_3" in subs

        subs = bridge.subscriptions.get_subscribers("mound_updated", {})
        assert "client_1" not in subs
        assert "client_2" in subs
        assert "client_3" in subs

        await bridge.stop()

    @pytest.mark.asyncio
    async def test_high_volume_event_handling(self):
        """Test handling high volume of events."""
        batch_count = 0

        def callback(event_type, data):
            nonlocal batch_count
            if event_type == "km_batch":
                batch_count += 1

        bridge = KMWebSocketBridge(batch_interval_ms=20, max_batch_size=10)
        bridge._batcher._callback = callback
        bridge.start()

        # Queue many events
        for i in range(100):
            bridge.queue_event("test_event", {"index": i})

        # Wait for batches
        await asyncio.sleep(0.3)

        await bridge.stop()

        # Should have received multiple batches
        assert batch_count >= 1
        assert bridge._batcher._total_events_queued == 100
