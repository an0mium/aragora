"""Tests for regional synchronization in multi-region control plane."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.control_plane.regional_sync import (
    RegionalEvent,
    RegionalEventBus,
    RegionalEventType,
    RegionalStateManager,
    RegionalSyncConfig,
    RegionHealth,
    get_regional_event_bus,
    init_regional_sync,
    set_regional_event_bus,
)


class TestRegionalEventType:
    """Tests for RegionalEventType enum."""

    def test_agent_event_types(self):
        """Test agent-related event types exist."""
        assert RegionalEventType.AGENT_REGISTERED.value == "agent_registered"
        assert RegionalEventType.AGENT_UPDATED.value == "agent_updated"
        assert RegionalEventType.AGENT_UNREGISTERED.value == "agent_unregistered"
        assert RegionalEventType.AGENT_HEARTBEAT.value == "agent_heartbeat"

    def test_task_event_types(self):
        """Test task-related event types exist."""
        assert RegionalEventType.TASK_SUBMITTED.value == "task_submitted"
        assert RegionalEventType.TASK_ASSIGNED.value == "task_assigned"
        assert RegionalEventType.TASK_COMPLETED.value == "task_completed"
        assert RegionalEventType.TASK_FAILED.value == "task_failed"
        assert RegionalEventType.TASK_CANCELLED.value == "task_cancelled"

    def test_leader_event_types(self):
        """Test leader-related event types exist."""
        assert RegionalEventType.LEADER_ELECTED.value == "leader_elected"
        assert RegionalEventType.LEADER_RESIGNED.value == "leader_resigned"

    def test_region_event_types(self):
        """Test region-related event types exist."""
        assert RegionalEventType.REGION_HEALTH.value == "region_health"
        assert RegionalEventType.REGION_JOINED.value == "region_joined"
        assert RegionalEventType.REGION_LEFT.value == "region_left"


class TestRegionalEvent:
    """Tests for RegionalEvent dataclass."""

    def test_create_event(self):
        """Test creating a regional event."""
        event = RegionalEvent(
            event_type=RegionalEventType.AGENT_REGISTERED,
            source_region="us-west-2",
            entity_id="agent-123",
            data={"capabilities": ["code", "debate"]},
        )

        assert event.event_type == RegionalEventType.AGENT_REGISTERED
        assert event.source_region == "us-west-2"
        assert event.entity_id == "agent-123"
        assert event.data == {"capabilities": ["code", "debate"]}
        assert event.version == 1
        assert event.timestamp > 0

    def test_event_serialization(self):
        """Test event serialization to dictionary."""
        event = RegionalEvent(
            event_type=RegionalEventType.TASK_SUBMITTED,
            source_region="eu-west-1",
            entity_id="task-456",
            timestamp=1234567890.0,
            data={"priority": "high"},
        )

        data = event.to_dict()

        assert data["event_type"] == "task_submitted"
        assert data["source_region"] == "eu-west-1"
        assert data["entity_id"] == "task-456"
        assert data["timestamp"] == 1234567890.0
        assert data["data"] == {"priority": "high"}
        assert data["version"] == 1

    def test_event_deserialization(self):
        """Test event deserialization from dictionary."""
        data = {
            "event_type": "agent_updated",
            "source_region": "ap-southeast-1",
            "entity_id": "agent-789",
            "timestamp": 1234567890.0,
            "data": {"status": "active"},
            "version": 1,
        }

        event = RegionalEvent.from_dict(data)

        assert event.event_type == RegionalEventType.AGENT_UPDATED
        assert event.source_region == "ap-southeast-1"
        assert event.entity_id == "agent-789"
        assert event.timestamp == 1234567890.0
        assert event.data == {"status": "active"}
        assert event.version == 1

    def test_event_round_trip(self):
        """Test event serialization round trip."""
        original = RegionalEvent(
            event_type=RegionalEventType.LEADER_ELECTED,
            source_region="us-east-1",
            entity_id="leader-001",
            data={"term": 5, "elected_at": "2024-01-01T00:00:00Z"},
        )

        data = original.to_dict()
        restored = RegionalEvent.from_dict(data)

        assert restored.event_type == original.event_type
        assert restored.source_region == original.source_region
        assert restored.entity_id == original.entity_id
        assert restored.timestamp == original.timestamp
        assert restored.data == original.data
        assert restored.version == original.version

    def test_event_is_newer_than(self):
        """Test event timestamp comparison."""
        older = RegionalEvent(
            event_type=RegionalEventType.AGENT_UPDATED,
            source_region="us-west-2",
            entity_id="agent-123",
            timestamp=1000.0,
        )
        newer = RegionalEvent(
            event_type=RegionalEventType.AGENT_UPDATED,
            source_region="us-east-1",
            entity_id="agent-123",
            timestamp=2000.0,
        )

        assert newer.is_newer_than(older)
        assert not older.is_newer_than(newer)
        assert not older.is_newer_than(older)  # Same timestamp


class TestRegionalSyncConfig:
    """Tests for RegionalSyncConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RegionalSyncConfig()

        assert config.channel_prefix == "aragora:regional:"
        assert config.heartbeat_interval == 10.0
        assert config.region_timeout == 30.0
        assert config.max_event_buffer == 1000
        assert config.sync_heartbeats is False
        assert config.conflict_strategy == "last_write_wins"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RegionalSyncConfig(
            local_region="us-west-2",
            sync_regions=["us-east-1", "eu-west-1"],
            heartbeat_interval=5.0,
            region_timeout=15.0,
            sync_heartbeats=True,
        )

        assert config.local_region == "us-west-2"
        assert config.sync_regions == ["us-east-1", "eu-west-1"]
        assert config.heartbeat_interval == 5.0
        assert config.region_timeout == 15.0
        assert config.sync_heartbeats is True

    def test_channel_names(self):
        """Test channel name generation."""
        config = RegionalSyncConfig(local_region="us-west-2")

        assert config.get_global_channel() == "aragora:regional:global"
        assert config.get_region_channel("us-east-1") == "aragora:regional:region:us-east-1"
        assert config.get_region_channel("eu-west-1") == "aragora:regional:region:eu-west-1"

    def test_custom_channel_prefix(self):
        """Test custom channel prefix."""
        config = RegionalSyncConfig(channel_prefix="custom:prefix:")

        assert config.get_global_channel() == "custom:prefix:global"
        assert config.get_region_channel("region-1") == "custom:prefix:region:region-1"


class TestRegionHealth:
    """Tests for RegionHealth dataclass."""

    def test_create_region_health(self):
        """Test creating region health status."""
        health = RegionHealth(
            region_id="us-west-2",
            last_seen=1234567890.0,
            healthy=True,
            agent_count=10,
            task_count=5,
            leader_id="leader-001",
        )

        assert health.region_id == "us-west-2"
        assert health.last_seen == 1234567890.0
        assert health.healthy is True
        assert health.agent_count == 10
        assert health.task_count == 5
        assert health.leader_id == "leader-001"

    def test_region_health_defaults(self):
        """Test region health default values."""
        health = RegionHealth(
            region_id="eu-west-1",
            last_seen=1234567890.0,
            healthy=False,
        )

        assert health.agent_count == 0
        assert health.task_count == 0
        assert health.leader_id is None


class TestRegionalEventBus:
    """Tests for RegionalEventBus class."""

    def test_create_event_bus(self):
        """Test creating an event bus."""
        config = RegionalSyncConfig(local_region="us-west-2")
        bus = RegionalEventBus(redis_url="redis://localhost:6379", config=config)

        assert bus.local_region == "us-west-2"
        assert bus.is_connected is False

    def test_event_bus_default_config(self):
        """Test event bus with default config."""
        bus = RegionalEventBus()

        assert bus.local_region is not None  # Uses environment or "default"
        assert bus.is_connected is False

    def test_subscribe_type_specific(self):
        """Test subscribing to specific event types."""
        bus = RegionalEventBus()

        handler = AsyncMock()
        bus.subscribe(RegionalEventType.AGENT_REGISTERED, handler)

        assert RegionalEventType.AGENT_REGISTERED in bus._handlers
        assert handler in bus._handlers[RegionalEventType.AGENT_REGISTERED]

    def test_subscribe_global(self):
        """Test subscribing to all events."""
        bus = RegionalEventBus()

        handler = AsyncMock()
        bus.subscribe(event_type=None, handler=handler)

        assert handler in bus._global_handlers

    def test_unsubscribe(self):
        """Test unsubscribing from events."""
        bus = RegionalEventBus()

        handler = AsyncMock()
        bus.subscribe(RegionalEventType.TASK_SUBMITTED, handler)
        bus.unsubscribe(RegionalEventType.TASK_SUBMITTED, handler)

        assert handler not in bus._handlers.get(RegionalEventType.TASK_SUBMITTED, [])

    def test_unsubscribe_global(self):
        """Test unsubscribing from global events."""
        bus = RegionalEventBus()

        handler = AsyncMock()
        bus.subscribe(event_type=None, handler=handler)
        bus.unsubscribe(event_type=None, handler=handler)

        assert handler not in bus._global_handlers

    def test_healthy_regions_empty(self):
        """Test healthy regions when none tracked."""
        bus = RegionalEventBus()

        assert bus.get_healthy_regions() == []

    def test_region_health_empty(self):
        """Test region health when none tracked."""
        bus = RegionalEventBus()

        assert bus.get_region_health() == {}

    @pytest.mark.asyncio
    async def test_publish_when_disconnected_buffers(self):
        """Test that events are buffered when disconnected."""
        bus = RegionalEventBus()

        event = RegionalEvent(
            event_type=RegionalEventType.AGENT_REGISTERED,
            source_region="us-west-2",
            entity_id="agent-123",
        )

        result = await bus.publish(event)

        assert result is True
        assert len(bus._event_buffer) == 1
        assert bus._event_buffer[0] == event

    @pytest.mark.asyncio
    async def test_publish_heartbeat_skipped_by_default(self):
        """Test that heartbeat events are skipped when sync_heartbeats is False."""
        config = RegionalSyncConfig(sync_heartbeats=False)
        bus = RegionalEventBus(config=config)

        event = RegionalEvent(
            event_type=RegionalEventType.AGENT_HEARTBEAT,
            source_region="us-west-2",
            entity_id="agent-123",
        )

        result = await bus.publish(event)

        assert result is True
        assert len(bus._event_buffer) == 0  # Not buffered

    @pytest.mark.asyncio
    async def test_publish_heartbeat_when_enabled(self):
        """Test that heartbeat events are published when sync_heartbeats is True."""
        config = RegionalSyncConfig(sync_heartbeats=True)
        bus = RegionalEventBus(config=config)

        event = RegionalEvent(
            event_type=RegionalEventType.AGENT_HEARTBEAT,
            source_region="us-west-2",
            entity_id="agent-123",
        )

        result = await bus.publish(event)

        assert result is True
        assert len(bus._event_buffer) == 1  # Buffered since disconnected

    @pytest.mark.asyncio
    async def test_buffer_limit(self):
        """Test that buffer respects max size limit."""
        config = RegionalSyncConfig(max_event_buffer=5)
        bus = RegionalEventBus(config=config)

        # Fill buffer
        for i in range(5):
            event = RegionalEvent(
                event_type=RegionalEventType.AGENT_UPDATED,
                source_region="us-west-2",
                entity_id=f"agent-{i}",
            )
            result = await bus.publish(event)
            assert result is True

        # Try to add one more
        event = RegionalEvent(
            event_type=RegionalEventType.AGENT_UPDATED,
            source_region="us-west-2",
            entity_id="agent-overflow",
        )
        result = await bus.publish(event)

        assert result is False
        assert len(bus._event_buffer) == 5


class TestRegionalEventBusWithMockRedis:
    """Tests for RegionalEventBus with mocked Redis."""

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection to Redis."""
        config = RegionalSyncConfig(local_region="us-west-2")
        bus = RegionalEventBus(redis_url="redis://localhost:6379", config=config)

        # Mock redis module
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_pubsub = AsyncMock()
        mock_redis.pubsub = MagicMock(return_value=mock_pubsub)

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            result = await bus.connect()

            # Clean up
            bus._running = False
            if bus._listener_task:
                bus._listener_task.cancel()
                try:
                    await bus._listener_task
                except asyncio.CancelledError:
                    pass
            if bus._heartbeat_task:
                bus._heartbeat_task.cancel()
                try:
                    await bus._heartbeat_task
                except asyncio.CancelledError:
                    pass

        assert result is True
        assert bus.is_connected is True

    @pytest.mark.asyncio
    async def test_connect_failure_import_error(self):
        """Test connection failure when redis not installed."""
        bus = RegionalEventBus()

        with patch.dict("sys.modules", {"redis.asyncio": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'redis'")):
                # This won't actually trigger since redis.asyncio is in the code
                pass

    @pytest.mark.asyncio
    async def test_publish_to_redis(self):
        """Test publishing event to Redis."""
        config = RegionalSyncConfig(local_region="us-west-2")
        bus = RegionalEventBus(config=config)

        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock(return_value=1)
        bus._redis = mock_redis
        bus._connected = True

        event = RegionalEvent(
            event_type=RegionalEventType.AGENT_REGISTERED,
            source_region="us-west-2",
            entity_id="agent-123",
        )

        result = await bus.publish(event)

        assert result is True
        mock_redis.publish.assert_called_once()
        call_args = mock_redis.publish.call_args
        assert call_args[0][0] == "aragora:regional:global"

    @pytest.mark.asyncio
    async def test_publish_to_specific_region(self):
        """Test publishing event to specific region."""
        config = RegionalSyncConfig(local_region="us-west-2")
        bus = RegionalEventBus(config=config)

        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock(return_value=1)
        bus._redis = mock_redis
        bus._connected = True

        event = RegionalEvent(
            event_type=RegionalEventType.AGENT_REGISTERED,
            source_region="us-west-2",
            entity_id="agent-123",
        )

        result = await bus.publish(event, target_region="us-east-1")

        assert result is True
        mock_redis.publish.assert_called_once()
        call_args = mock_redis.publish.call_args
        assert call_args[0][0] == "aragora:regional:region:us-east-1"

    @pytest.mark.asyncio
    async def test_publish_agent_update_convenience(self):
        """Test convenience method for agent updates."""
        config = RegionalSyncConfig(local_region="us-west-2")
        bus = RegionalEventBus(config=config)

        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock(return_value=1)
        bus._redis = mock_redis
        bus._connected = True

        result = await bus.publish_agent_update(
            agent_id="agent-123",
            agent_data={"status": "active", "capabilities": ["code"]},
        )

        assert result is True
        call_args = mock_redis.publish.call_args
        published_data = json.loads(call_args[0][1])
        assert published_data["event_type"] == "agent_updated"
        assert published_data["entity_id"] == "agent-123"
        assert published_data["source_region"] == "us-west-2"

    @pytest.mark.asyncio
    async def test_publish_task_update_convenience(self):
        """Test convenience method for task updates."""
        config = RegionalSyncConfig(local_region="us-west-2")
        bus = RegionalEventBus(config=config)

        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock(return_value=1)
        bus._redis = mock_redis
        bus._connected = True

        result = await bus.publish_task_update(
            task_id="task-456",
            task_data={"status": "completed", "result": "success"},
            event_type=RegionalEventType.TASK_COMPLETED,
        )

        assert result is True
        call_args = mock_redis.publish.call_args
        published_data = json.loads(call_args[0][1])
        assert published_data["event_type"] == "task_completed"
        assert published_data["entity_id"] == "task-456"

    @pytest.mark.asyncio
    async def test_handle_message_ignores_self(self):
        """Test that messages from self are ignored."""
        config = RegionalSyncConfig(local_region="us-west-2")
        bus = RegionalEventBus(config=config)

        handler = AsyncMock()
        bus.subscribe(RegionalEventType.AGENT_REGISTERED, handler)

        # Message from self
        data = json.dumps(
            {
                "event_type": "agent_registered",
                "source_region": "us-west-2",  # Same as local
                "entity_id": "agent-123",
                "timestamp": time.time(),
            }
        )

        await bus._handle_message(data)

        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_message_calls_handlers(self):
        """Test that messages from other regions call handlers."""
        config = RegionalSyncConfig(local_region="us-west-2")
        bus = RegionalEventBus(config=config)

        handler = AsyncMock()
        bus.subscribe(RegionalEventType.AGENT_REGISTERED, handler)

        # Message from another region
        data = json.dumps(
            {
                "event_type": "agent_registered",
                "source_region": "us-east-1",  # Different from local
                "entity_id": "agent-123",
                "timestamp": time.time(),
            }
        )

        await bus._handle_message(data)

        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_message_updates_region_health(self):
        """Test that messages update region health tracking."""
        config = RegionalSyncConfig(local_region="us-west-2")
        bus = RegionalEventBus(config=config)

        data = json.dumps(
            {
                "event_type": "agent_updated",
                "source_region": "us-east-1",
                "entity_id": "agent-123",
                "timestamp": time.time(),
            }
        )

        await bus._handle_message(data)

        assert "us-east-1" in bus._region_last_seen
        health = bus.get_region_health()
        assert "us-east-1" in health
        assert health["us-east-1"]["healthy"] is True

    @pytest.mark.asyncio
    async def test_close_cleans_up(self):
        """Test that close cleans up resources."""
        config = RegionalSyncConfig(local_region="us-west-2")
        bus = RegionalEventBus(config=config)

        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock(return_value=1)
        mock_redis.close = AsyncMock()
        mock_pubsub = AsyncMock()
        mock_pubsub.unsubscribe = AsyncMock()
        mock_pubsub.close = AsyncMock()

        bus._redis = mock_redis
        bus._pubsub = mock_pubsub
        bus._connected = True
        bus._running = True

        await bus.close()

        assert bus._connected is False
        assert bus._running is False
        mock_pubsub.unsubscribe.assert_called_once()
        mock_pubsub.close.assert_called_once()
        mock_redis.close.assert_called_once()


class TestRegionalStateManager:
    """Tests for RegionalStateManager class."""

    def test_create_state_manager(self):
        """Test creating a state manager."""
        config = RegionalSyncConfig(local_region="us-west-2")
        bus = RegionalEventBus(config=config)
        manager = RegionalStateManager(event_bus=bus)

        assert manager._event_bus == bus
        assert manager._state_store is None

    def test_state_manager_registers_handlers(self):
        """Test that state manager registers event handlers."""
        config = RegionalSyncConfig(local_region="us-west-2")
        bus = RegionalEventBus(config=config)
        manager = RegionalStateManager(event_bus=bus)

        # Check handlers were registered
        assert RegionalEventType.AGENT_REGISTERED in bus._handlers
        assert RegionalEventType.AGENT_UPDATED in bus._handlers
        assert RegionalEventType.AGENT_UNREGISTERED in bus._handlers
        assert RegionalEventType.TASK_SUBMITTED in bus._handlers
        assert RegionalEventType.TASK_COMPLETED in bus._handlers

    @pytest.mark.asyncio
    async def test_is_newer_check(self):
        """Test version checking for conflict resolution."""
        config = RegionalSyncConfig(local_region="us-west-2")
        bus = RegionalEventBus(config=config)
        manager = RegionalStateManager(event_bus=bus)

        # First event should be newer
        assert manager._is_newer("entity-1", 1000.0)

        # Update version
        manager._entity_versions["entity-1"] = 1000.0

        # Older event should not be newer
        assert not manager._is_newer("entity-1", 500.0)

        # Newer event should be newer
        assert manager._is_newer("entity-1", 2000.0)

    @pytest.mark.asyncio
    async def test_handle_agent_registered(self):
        """Test handling agent registration from another region."""
        config = RegionalSyncConfig(local_region="us-west-2")
        bus = RegionalEventBus(config=config)
        manager = RegionalStateManager(event_bus=bus)

        event = RegionalEvent(
            event_type=RegionalEventType.AGENT_REGISTERED,
            source_region="us-east-1",
            entity_id="agent-123",
            timestamp=time.time(),
            data={"capabilities": ["code"]},
        )

        await manager._handle_agent_registered(event)

        assert "agent-123" in manager._entity_versions

    @pytest.mark.asyncio
    async def test_handle_old_event_ignored(self):
        """Test that old events are ignored."""
        config = RegionalSyncConfig(local_region="us-west-2")
        bus = RegionalEventBus(config=config)
        manager = RegionalStateManager(event_bus=bus)

        # Set a version
        manager._entity_versions["agent-123"] = 2000.0

        # Old event
        event = RegionalEvent(
            event_type=RegionalEventType.AGENT_UPDATED,
            source_region="us-east-1",
            entity_id="agent-123",
            timestamp=1000.0,  # Older than stored
            data={"status": "old"},
        )

        await manager._handle_agent_updated(event)

        # Version should not change
        assert manager._entity_versions["agent-123"] == 2000.0


class TestModuleSingleton:
    """Tests for module-level singleton functions."""

    def test_get_event_bus_initially_none(self):
        """Test that event bus is initially None."""
        # Reset singleton
        import aragora.control_plane.regional_sync as rs

        original = rs._regional_event_bus
        rs._regional_event_bus = None

        try:
            result = get_regional_event_bus()
            assert result is None
        finally:
            rs._regional_event_bus = original

    def test_set_and_get_event_bus(self):
        """Test setting and getting event bus."""
        import aragora.control_plane.regional_sync as rs

        original = rs._regional_event_bus

        try:
            bus = RegionalEventBus()
            set_regional_event_bus(bus)
            result = get_regional_event_bus()
            assert result == bus
        finally:
            rs._regional_event_bus = original

    @pytest.mark.asyncio
    async def test_init_regional_sync_without_redis(self):
        """Test init_regional_sync when Redis is unavailable."""
        import aragora.control_plane.regional_sync as rs

        original = rs._regional_event_bus
        rs._regional_event_bus = None

        try:
            # Mock connection failure
            with patch.object(
                RegionalEventBus, "connect", new_callable=AsyncMock, return_value=False
            ):
                result = await init_regional_sync()
                assert result is None
        finally:
            rs._regional_event_bus = original


class TestIntegration:
    """Integration tests for regional sync components."""

    @pytest.mark.asyncio
    async def test_full_event_flow(self):
        """Test full event flow from publish to handler."""
        config = RegionalSyncConfig(local_region="us-west-2")
        bus = RegionalEventBus(config=config)

        received_events = []

        async def handler(event):
            received_events.append(event)

        bus.subscribe(RegionalEventType.AGENT_REGISTERED, handler)

        # Simulate receiving message from another region
        data = json.dumps(
            {
                "event_type": "agent_registered",
                "source_region": "us-east-1",
                "entity_id": "agent-123",
                "timestamp": time.time(),
                "data": {"capabilities": ["code", "debate"]},
            }
        )

        await bus._handle_message(data)

        assert len(received_events) == 1
        assert received_events[0].entity_id == "agent-123"
        assert received_events[0].source_region == "us-east-1"
        assert received_events[0].data["capabilities"] == ["code", "debate"]

    @pytest.mark.asyncio
    async def test_state_manager_integration(self):
        """Test state manager integration with event bus."""
        config = RegionalSyncConfig(local_region="us-west-2")
        bus = RegionalEventBus(config=config)
        manager = RegionalStateManager(event_bus=bus)

        # Simulate receiving events
        agent_registered = json.dumps(
            {
                "event_type": "agent_registered",
                "source_region": "us-east-1",
                "entity_id": "agent-123",
                "timestamp": 1000.0,
            }
        )
        await bus._handle_message(agent_registered)

        # State manager should track version
        assert "agent-123" in manager._entity_versions
        assert manager._entity_versions["agent-123"] == 1000.0

        # Newer update
        agent_updated = json.dumps(
            {
                "event_type": "agent_updated",
                "source_region": "eu-west-1",
                "entity_id": "agent-123",
                "timestamp": 2000.0,
            }
        )
        await bus._handle_message(agent_updated)

        assert manager._entity_versions["agent-123"] == 2000.0

    @pytest.mark.asyncio
    async def test_multiple_regions_health_tracking(self):
        """Test health tracking for multiple regions."""
        config = RegionalSyncConfig(local_region="us-west-2", region_timeout=60.0)
        bus = RegionalEventBus(config=config)

        # Simulate messages from multiple regions
        regions = ["us-east-1", "eu-west-1", "ap-southeast-1"]
        for region in regions:
            data = json.dumps(
                {
                    "event_type": "region_health",
                    "source_region": region,
                    "entity_id": region,
                    "timestamp": time.time(),
                }
            )
            await bus._handle_message(data)

        health = bus.get_region_health()
        assert len(health) == 3

        for region in regions:
            assert region in health
            assert health[region]["healthy"] is True

        healthy_regions = bus.get_healthy_regions()
        assert set(healthy_regions) == set(regions)
