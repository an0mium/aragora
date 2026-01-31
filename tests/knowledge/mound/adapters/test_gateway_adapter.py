"""
Tests for GatewayAdapter - Bridges LocalGateway to Knowledge Mound.

Tests cover:
- MessageRoutingRecord, ChannelPerformanceSnapshot, DeviceRegistrationRecord dataclasses
- Adapter initialization
- Routing record storage
- Channel snapshot storage
- Device registration storage
- Cache behavior
- Statistics
"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock

from aragora.knowledge.mound.adapters.gateway_adapter import (
    GatewayAdapter,
    MessageRoutingRecord,
    ChannelPerformanceSnapshot,
    DeviceRegistrationRecord,
    RoutingDecisionRecord,
)


# =============================================================================
# MessageRoutingRecord Dataclass Tests
# =============================================================================


class TestMessageRoutingRecord:
    """Tests for MessageRoutingRecord dataclass."""

    def test_create_record(self):
        """Should create a message routing record."""
        record = MessageRoutingRecord(
            message_id="msg-001",
            channel="slack",
            sender="user-123",
            agent_id="agent-456",
            routing_rule="priority_route",
            success=True,
            latency_ms=150.5,
        )

        assert record.message_id == "msg-001"
        assert record.channel == "slack"
        assert record.sender == "user-123"
        assert record.agent_id == "agent-456"
        assert record.success is True
        assert record.latency_ms == 150.5

    def test_record_defaults(self):
        """Should use default values."""
        record = MessageRoutingRecord(
            message_id="msg-002",
            channel="discord",
            sender="user-456",
            agent_id=None,
            routing_rule=None,
            success=False,
            latency_ms=500.0,
        )

        assert record.priority == "normal"
        assert record.thread_id is None
        assert record.error_message is None
        assert record.workspace_id == "default"

    def test_record_with_all_fields(self):
        """Should accept all fields."""
        record = MessageRoutingRecord(
            message_id="msg-003",
            channel="teams",
            sender="user-789",
            agent_id="agent-123",
            routing_rule="fallback_route",
            success=False,
            latency_ms=1000.0,
            priority="high",
            thread_id="thread-001",
            error_message="Timeout error",
            workspace_id="ws-custom",
        )

        assert record.priority == "high"
        assert record.thread_id == "thread-001"
        assert record.error_message == "Timeout error"
        assert record.workspace_id == "ws-custom"


# =============================================================================
# ChannelPerformanceSnapshot Dataclass Tests
# =============================================================================


class TestChannelPerformanceSnapshot:
    """Tests for ChannelPerformanceSnapshot dataclass."""

    def test_create_snapshot(self):
        """Should create a channel performance snapshot."""
        snapshot = ChannelPerformanceSnapshot(
            channel="slack",
            messages_received=100,
            messages_routed=95,
            messages_failed=5,
            avg_latency_ms=120.5,
        )

        assert snapshot.channel == "slack"
        assert snapshot.messages_received == 100
        assert snapshot.messages_routed == 95
        assert snapshot.messages_failed == 5

    def test_snapshot_defaults(self):
        """Should use default values."""
        snapshot = ChannelPerformanceSnapshot(
            channel="discord",
            messages_received=50,
            messages_routed=48,
            messages_failed=2,
            avg_latency_ms=80.0,
        )

        assert snapshot.active_threads == 0
        assert snapshot.unique_senders == 0
        assert snapshot.workspace_id == "default"
        assert snapshot.metadata == {}


# =============================================================================
# DeviceRegistrationRecord Dataclass Tests
# =============================================================================


class TestDeviceRegistrationRecord:
    """Tests for DeviceRegistrationRecord dataclass."""

    def test_create_record(self):
        """Should create a device registration record."""
        record = DeviceRegistrationRecord(
            device_id="device-001",
            device_name="Claude Bot",
            device_type="ai_agent",
            status="online",
            capabilities=["text", "code", "analysis"],
            registered_at=time.time(),
            last_seen=time.time(),
        )

        assert record.device_id == "device-001"
        assert record.device_name == "Claude Bot"
        assert record.device_type == "ai_agent"
        assert "text" in record.capabilities


# =============================================================================
# RoutingDecisionRecord Dataclass Tests
# =============================================================================


class TestRoutingDecisionRecord:
    """Tests for RoutingDecisionRecord dataclass."""

    def test_create_record(self):
        """Should create a routing decision record."""
        record = RoutingDecisionRecord(
            decision_id="decision-001",
            message_id="msg-001",
            channel="slack",
            rule_matched="priority_route",
            agent_selected="agent-456",
            fallback_used=False,
            capabilities_required=["text"],
            capabilities_available=["text", "code"],
            timestamp=time.time(),
        )

        assert record.decision_id == "decision-001"
        assert record.rule_matched == "priority_route"
        assert record.fallback_used is False


# =============================================================================
# GatewayAdapter Initialization Tests
# =============================================================================


class TestGatewayAdapterInit:
    """Tests for GatewayAdapter initialization."""

    def test_default_init(self):
        """Should initialize with default values."""
        adapter = GatewayAdapter()

        assert adapter._gateway is None
        assert adapter._knowledge_mound is None
        assert adapter._workspace_id == "default"
        assert adapter._min_confidence_threshold == 0.6
        assert adapter.adapter_name == "gateway"

    def test_init_with_gateway(self):
        """Should accept gateway instance."""
        mock_gateway = MagicMock()
        adapter = GatewayAdapter(gateway=mock_gateway)

        assert adapter._gateway is mock_gateway

    def test_init_with_knowledge_mound(self):
        """Should accept knowledge mound."""
        mock_km = MagicMock()
        adapter = GatewayAdapter(knowledge_mound=mock_km)

        assert adapter._knowledge_mound is mock_km

    def test_init_with_workspace(self):
        """Should accept workspace ID."""
        adapter = GatewayAdapter(workspace_id="ws-custom")

        assert adapter._workspace_id == "ws-custom"

    def test_init_with_event_callback(self):
        """Should accept event callback."""
        callback = MagicMock()
        adapter = GatewayAdapter(event_callback=callback)

        assert adapter._event_callback is callback


# =============================================================================
# Routing Record Storage Tests
# =============================================================================


class TestStoreRoutingRecord:
    """Tests for storing routing records."""

    @pytest.mark.asyncio
    async def test_store_success_record(self):
        """Should store successful routing record."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-001")

        adapter = GatewayAdapter(knowledge_mound=mock_km)

        record = MessageRoutingRecord(
            message_id="msg-001",
            channel="slack",
            sender="user-123",
            agent_id="agent-456",
            routing_rule="priority_route",
            success=True,
            latency_ms=150.0,
        )

        result = await adapter.store_routing_record(record)

        assert result == "item-001"
        mock_km.ingest.assert_called_once()
        assert adapter._stats["routing_records_stored"] == 1

    @pytest.mark.asyncio
    async def test_store_no_km(self):
        """Should return None without knowledge mound."""
        adapter = GatewayAdapter()

        record = MessageRoutingRecord(
            message_id="msg-001",
            channel="slack",
            sender="user-123",
            agent_id=None,
            routing_rule=None,
            success=True,
            latency_ms=100.0,
        )

        result = await adapter.store_routing_record(record)

        assert result is None

    @pytest.mark.asyncio
    async def test_store_failed_record(self):
        """Should store failed routing record."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-002")

        adapter = GatewayAdapter(knowledge_mound=mock_km)

        record = MessageRoutingRecord(
            message_id="msg-002",
            channel="discord",
            sender="user-456",
            agent_id=None,
            routing_rule=None,
            success=False,
            latency_ms=500.0,
            error_message="Timeout",
        )

        result = await adapter.store_routing_record(record)

        assert result == "item-002"

    @pytest.mark.asyncio
    async def test_store_invalidates_cache(self):
        """Should invalidate routing cache on store."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-003")

        adapter = GatewayAdapter(knowledge_mound=mock_km)
        adapter._routing_patterns_cache["slack"] = [MagicMock()]

        record = MessageRoutingRecord(
            message_id="msg-003",
            channel="slack",
            sender="user-789",
            agent_id="agent-123",
            routing_rule="test",
            success=True,
            latency_ms=100.0,
        )

        await adapter.store_routing_record(record)

        assert "slack" not in adapter._routing_patterns_cache


# =============================================================================
# Channel Snapshot Storage Tests
# =============================================================================


class TestStoreChannelSnapshot:
    """Tests for storing channel snapshots."""

    @pytest.mark.asyncio
    async def test_store_snapshot(self):
        """Should store channel performance snapshot."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-001")

        adapter = GatewayAdapter(knowledge_mound=mock_km)

        snapshot = ChannelPerformanceSnapshot(
            channel="slack",
            messages_received=100,
            messages_routed=95,
            messages_failed=5,
            avg_latency_ms=120.0,
        )

        result = await adapter.store_channel_snapshot(snapshot)

        assert result == "item-001"
        assert adapter._stats["channel_snapshots_stored"] == 1

    @pytest.mark.asyncio
    async def test_store_no_km(self):
        """Should return None without knowledge mound."""
        adapter = GatewayAdapter()

        snapshot = ChannelPerformanceSnapshot(
            channel="slack",
            messages_received=50,
            messages_routed=48,
            messages_failed=2,
            avg_latency_ms=80.0,
        )

        result = await adapter.store_channel_snapshot(snapshot)

        assert result is None


# =============================================================================
# Device Registration Storage Tests
# =============================================================================


class TestStoreDeviceRegistration:
    """Tests for storing device registrations."""

    @pytest.mark.asyncio
    async def test_store_registration(self):
        """Should store device registration."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-001")

        adapter = GatewayAdapter(knowledge_mound=mock_km)

        record = DeviceRegistrationRecord(
            device_id="device-001",
            device_name="Claude Bot",
            device_type="ai_agent",
            status="online",
            capabilities=["text", "code"],
            registered_at=time.time(),
            last_seen=time.time(),
        )

        result = await adapter.store_device_registration(record)

        assert result == "item-001"
        assert adapter._stats["device_records_stored"] == 1

    @pytest.mark.asyncio
    async def test_store_no_km(self):
        """Should return None without knowledge mound."""
        adapter = GatewayAdapter()

        record = DeviceRegistrationRecord(
            device_id="device-002",
            device_name="Test Bot",
            device_type="bot",
            status="offline",
            capabilities=[],
            registered_at=time.time(),
            last_seen=time.time(),
        )

        result = await adapter.store_device_registration(record)

        assert result is None


# =============================================================================
# Routing Decision Storage Tests
# =============================================================================


class TestStoreRoutingDecision:
    """Tests for storing routing decisions."""

    @pytest.mark.asyncio
    async def test_store_decision(self):
        """Should store routing decision."""
        mock_km = AsyncMock()
        mock_km.ingest = AsyncMock(return_value="item-001")

        adapter = GatewayAdapter(knowledge_mound=mock_km)

        record = RoutingDecisionRecord(
            decision_id="decision-001",
            message_id="msg-001",
            channel="slack",
            rule_matched="priority_route",
            agent_selected="agent-456",
            fallback_used=False,
            capabilities_required=["text"],
            capabilities_available=["text", "code"],
            timestamp=time.time(),
        )

        result = await adapter.store_routing_decision(record)

        assert result == "item-001"
        assert adapter._stats["routing_decisions_stored"] == 1


# =============================================================================
# Query Tests
# =============================================================================


class TestQueries:
    """Tests for query methods."""

    @pytest.mark.asyncio
    async def test_get_channel_performance_no_km(self):
        """Should return empty list without KM."""
        adapter = GatewayAdapter()

        result = await adapter.get_channel_performance_history("slack")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_channel_performance_uses_cache(self):
        """Should use cached results."""
        adapter = GatewayAdapter()

        # Pre-populate cache
        cached = ChannelPerformanceSnapshot(
            channel="slack",
            messages_received=100,
            messages_routed=95,
            messages_failed=5,
            avg_latency_ms=120.0,
        )
        adapter._channel_performance_cache["slack"] = [cached]
        adapter._cache_times["channel_slack"] = time.time()

        result = await adapter.get_channel_performance_history("slack")

        assert len(result) == 1
        assert result[0].channel == "slack"
        assert adapter._stats["channel_queries"] == 1

    @pytest.mark.asyncio
    async def test_get_routing_patterns_no_km(self):
        """Should return empty list without KM."""
        adapter = GatewayAdapter()

        result = await adapter.get_routing_patterns("slack")

        assert result == []


# =============================================================================
# Recommendations Tests
# =============================================================================


class TestRecommendations:
    """Tests for routing recommendations."""

    @pytest.mark.asyncio
    async def test_no_history_recommendation(self):
        """Should return no history message."""
        adapter = GatewayAdapter()

        result = await adapter.get_routing_recommendations("slack")

        assert len(result) >= 1
        assert "No routing history" in result[0]["recommendation"]

    @pytest.mark.asyncio
    async def test_recommendations_from_patterns(self):
        """Should generate recommendations from patterns."""
        adapter = GatewayAdapter()

        # Pre-populate cache with patterns
        patterns = [
            MessageRoutingRecord(
                message_id=f"msg-{i}",
                channel="slack",
                sender=f"user-{i}",
                agent_id="agent-best",
                routing_rule="priority_route",
                success=True,
                latency_ms=100.0,
            )
            for i in range(10)
        ]
        adapter._routing_patterns_cache["slack"] = patterns
        adapter._cache_times["routing_slack"] = time.time()

        result = await adapter.get_routing_recommendations("slack")

        assert len(result) >= 1


# =============================================================================
# Stats and Cache Tests
# =============================================================================


class TestStatsAndCache:
    """Tests for statistics and cache operations."""

    def test_get_stats(self):
        """Should return adapter stats."""
        adapter = GatewayAdapter(workspace_id="ws-test")

        stats = adapter.get_stats()

        assert stats["routing_records_stored"] == 0
        assert stats["channel_snapshots_stored"] == 0
        assert stats["workspace_id"] == "ws-test"
        assert stats["has_knowledge_mound"] is False
        assert stats["has_gateway"] is False

    def test_clear_cache(self):
        """Should clear all caches."""
        adapter = GatewayAdapter()

        # Populate caches
        adapter._channel_performance_cache["slack"] = [MagicMock()]
        adapter._routing_patterns_cache["discord"] = [MagicMock()]
        adapter._cache_times["test"] = time.time()

        count = adapter.clear_cache()

        assert count == 2
        assert len(adapter._channel_performance_cache) == 0
        assert len(adapter._routing_patterns_cache) == 0
        assert len(adapter._cache_times) == 0


# =============================================================================
# Sync Tests
# =============================================================================


class TestSync:
    """Tests for sync operations."""

    @pytest.mark.asyncio
    async def test_sync_no_gateway(self):
        """Should return error without gateway."""
        adapter = GatewayAdapter()

        result = await adapter.sync_from_gateway()

        assert "error" in result
        assert "No gateway configured" in result["error"]
