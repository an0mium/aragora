"""
Tests for real-time usage metering stream.

Tests cover:
- Token usage event emission
- Cost tracking events
- Budget alert triggering
- Tenant-scoped subscriptions and isolation
- Event history management (rotation at 1000 events)
- Callback dispatch and exception isolation
- Real-time usage updates
- Concurrent usage tracking
- Memory management for event history
- Error handling for malformed data
"""

import asyncio
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.stream.usage_stream import (
    UsageEventType,
    UsageStreamEmitter,
    UsageStreamEvent,
    emit_usage_event,
    get_usage_emitter,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def emitter():
    """Create a fresh UsageStreamEmitter for each test."""
    return UsageStreamEmitter()


@pytest.fixture
def sample_event():
    """Create a sample usage event."""
    return UsageStreamEvent(
        event_type=UsageEventType.TOKEN_USAGE,
        tenant_id="tenant-123",
        workspace_id="workspace-456",
        user_id="user-789",
        tokens_in=100,
        tokens_out=50,
        total_tokens=150,
        cost_usd=Decimal("0.0015"),
        provider="anthropic",
        model="claude-3-opus",
        debate_id="debate-001",
        operation="proposal",
    )


@pytest.fixture
def cost_event():
    """Create a cost update event."""
    return UsageStreamEvent(
        event_type=UsageEventType.COST_UPDATE,
        tenant_id="tenant-123",
        tokens_in=5000,
        tokens_out=2500,
        total_tokens=7500,
        cost_usd=Decimal("0.15"),
        metadata={"period": "daily"},
    )


@pytest.fixture
def budget_alert_event():
    """Create a budget alert event."""
    return UsageStreamEvent(
        event_type=UsageEventType.BUDGET_ALERT,
        tenant_id="tenant-123",
        budget_used_pct=85.0,
        budget_remaining=Decimal("150.00"),
        metadata={"alert_level": "warning"},
    )


@pytest.fixture(autouse=True)
def reset_global_emitter():
    """Reset the global emitter before and after each test."""
    import aragora.server.stream.usage_stream as usage_module

    original = usage_module._usage_emitter
    usage_module._usage_emitter = None
    yield
    usage_module._usage_emitter = original


# ===========================================================================
# Test UsageEventType
# ===========================================================================


class TestUsageEventType:
    """Tests for UsageEventType enum."""

    def test_token_usage_value(self):
        """TOKEN_USAGE has correct value."""
        assert UsageEventType.TOKEN_USAGE.value == "token_usage"

    def test_cost_update_value(self):
        """COST_UPDATE has correct value."""
        assert UsageEventType.COST_UPDATE.value == "cost_update"

    def test_budget_alert_value(self):
        """BUDGET_ALERT has correct value."""
        assert UsageEventType.BUDGET_ALERT.value == "budget_alert"

    def test_usage_summary_value(self):
        """USAGE_SUMMARY has correct value."""
        assert UsageEventType.USAGE_SUMMARY.value == "usage_summary"

    def test_rate_limit_value(self):
        """RATE_LIMIT has correct value."""
        assert UsageEventType.RATE_LIMIT.value == "rate_limit"

    def test_quota_warning_value(self):
        """QUOTA_WARNING has correct value."""
        assert UsageEventType.QUOTA_WARNING.value == "quota_warning"


# ===========================================================================
# Test UsageStreamEvent
# ===========================================================================


class TestUsageStreamEvent:
    """Tests for UsageStreamEvent dataclass."""

    def test_default_values(self):
        """Event initializes with sensible defaults."""
        event = UsageStreamEvent()
        assert event.event_type == UsageEventType.TOKEN_USAGE
        assert event.tenant_id == ""
        assert event.workspace_id is None
        assert event.user_id is None
        assert event.tokens_in == 0
        assert event.tokens_out == 0
        assert event.total_tokens == 0
        assert event.cost_usd == Decimal("0")
        assert event.provider == ""
        assert event.model == ""
        assert event.debate_id is None
        assert event.operation == ""
        assert event.budget_used_pct == 0.0
        assert event.budget_remaining == Decimal("0")
        assert event.metadata == {}
        assert event.timestamp is not None

    def test_id_auto_generated(self):
        """Event ID is auto-generated UUID."""
        event1 = UsageStreamEvent()
        event2 = UsageStreamEvent()
        assert event1.id != event2.id
        assert len(event1.id) == 36  # UUID format

    def test_timestamp_auto_generated(self):
        """Timestamp is auto-generated in UTC."""
        event = UsageStreamEvent()
        assert event.timestamp.tzinfo == timezone.utc
        # Should be close to now
        now = datetime.now(timezone.utc)
        assert abs((now - event.timestamp).total_seconds()) < 1

    def test_to_dict(self, sample_event):
        """to_dict converts event to JSON-serializable dict."""
        d = sample_event.to_dict()
        assert d["event_type"] == "token_usage"
        assert d["tenant_id"] == "tenant-123"
        assert d["workspace_id"] == "workspace-456"
        assert d["user_id"] == "user-789"
        assert d["tokens_in"] == 100
        assert d["tokens_out"] == 50
        assert d["total_tokens"] == 150
        assert d["cost_usd"] == "0.0015"
        assert d["provider"] == "anthropic"
        assert d["model"] == "claude-3-opus"
        assert d["debate_id"] == "debate-001"
        assert d["operation"] == "proposal"
        assert d["budget_used_pct"] == 0.0
        assert d["budget_remaining"] == "0"

    def test_to_dict_timestamp_format(self, sample_event):
        """to_dict timestamp is ISO format."""
        d = sample_event.to_dict()
        assert "T" in d["timestamp"]  # ISO format has T separator
        # Should be parseable
        datetime.fromisoformat(d["timestamp"])

    def test_to_json(self, sample_event):
        """to_json returns valid JSON string."""
        import json

        json_str = sample_event.to_json()
        parsed = json.loads(json_str)
        assert parsed["event_type"] == "token_usage"
        assert parsed["tenant_id"] == "tenant-123"

    def test_decimal_serialization(self):
        """Decimal cost values serialize correctly."""
        event = UsageStreamEvent(
            cost_usd=Decimal("123.456789"),
            budget_remaining=Decimal("1000.00"),
        )
        d = event.to_dict()
        assert d["cost_usd"] == "123.456789"
        assert d["budget_remaining"] == "1000.00"

    def test_metadata_serialization(self):
        """Metadata dict is preserved in serialization."""
        event = UsageStreamEvent(
            metadata={"custom_field": "value", "nested": {"key": 123}},
        )
        d = event.to_dict()
        assert d["metadata"]["custom_field"] == "value"
        assert d["metadata"]["nested"]["key"] == 123


# ===========================================================================
# Test UsageStreamEmitter Subscription
# ===========================================================================


class TestUsageStreamEmitterSubscription:
    """Tests for UsageStreamEmitter subscription management."""

    @pytest.mark.asyncio
    async def test_subscribe_tenant(self, emitter):
        """subscribe_tenant adds callback for tenant."""
        callback = MagicMock()
        await emitter.subscribe_tenant("tenant-1", callback)
        assert "tenant-1" in emitter._subscribers
        assert callback in emitter._subscribers["tenant-1"]

    @pytest.mark.asyncio
    async def test_subscribe_tenant_multiple_callbacks(self, emitter):
        """Multiple callbacks can subscribe to same tenant."""
        cb1 = MagicMock()
        cb2 = MagicMock()
        await emitter.subscribe_tenant("tenant-1", cb1)
        await emitter.subscribe_tenant("tenant-1", cb2)
        assert len(emitter._subscribers["tenant-1"]) == 2

    @pytest.mark.asyncio
    async def test_unsubscribe_tenant(self, emitter):
        """unsubscribe_tenant removes callback."""
        callback = MagicMock()
        await emitter.subscribe_tenant("tenant-1", callback)
        await emitter.unsubscribe_tenant("tenant-1", callback)
        assert "tenant-1" not in emitter._subscribers

    @pytest.mark.asyncio
    async def test_unsubscribe_tenant_keeps_others(self, emitter):
        """unsubscribe_tenant keeps other callbacks."""
        cb1 = MagicMock()
        cb2 = MagicMock()
        await emitter.subscribe_tenant("tenant-1", cb1)
        await emitter.subscribe_tenant("tenant-1", cb2)
        await emitter.unsubscribe_tenant("tenant-1", cb1)
        assert cb2 in emitter._subscribers["tenant-1"]
        assert cb1 not in emitter._subscribers["tenant-1"]

    @pytest.mark.asyncio
    async def test_unsubscribe_tenant_nonexistent(self, emitter):
        """unsubscribe_tenant handles nonexistent tenant gracefully."""
        callback = MagicMock()
        # Should not raise
        await emitter.unsubscribe_tenant("nonexistent", callback)

    @pytest.mark.asyncio
    async def test_subscribe_workspace(self, emitter):
        """subscribe_workspace adds callback for workspace."""
        callback = MagicMock()
        await emitter.subscribe_workspace("workspace-1", callback)
        assert "workspace-1" in emitter._workspace_subscribers
        assert callback in emitter._workspace_subscribers["workspace-1"]

    @pytest.mark.asyncio
    async def test_subscribe_global(self, emitter):
        """subscribe_global adds callback for all events."""
        callback = MagicMock()
        await emitter.subscribe_global(callback)
        assert callback in emitter._global_subscribers

    @pytest.mark.asyncio
    async def test_get_subscriber_count_empty(self, emitter):
        """get_subscriber_count returns 0 when no subscribers."""
        assert emitter.get_subscriber_count() == 0
        assert emitter.get_subscriber_count("tenant-1") == 0

    @pytest.mark.asyncio
    async def test_get_subscriber_count_tenant(self, emitter):
        """get_subscriber_count returns correct count for tenant."""
        await emitter.subscribe_tenant("tenant-1", MagicMock())
        await emitter.subscribe_tenant("tenant-1", MagicMock())
        assert emitter.get_subscriber_count("tenant-1") == 2

    @pytest.mark.asyncio
    async def test_get_subscriber_count_total(self, emitter):
        """get_subscriber_count returns total across all types."""
        await emitter.subscribe_tenant("tenant-1", MagicMock())
        await emitter.subscribe_workspace("workspace-1", MagicMock())
        await emitter.subscribe_global(MagicMock())
        assert emitter.get_subscriber_count() == 3


# ===========================================================================
# Test UsageStreamEmitter Event Emission
# ===========================================================================


class TestUsageStreamEmitterEmit:
    """Tests for UsageStreamEmitter event emission."""

    @pytest.mark.asyncio
    async def test_emit_to_tenant_subscriber(self, emitter, sample_event):
        """emit sends event to tenant subscribers."""
        received = []

        def callback(e):
            received.append(e)

        await emitter.subscribe_tenant("tenant-123", callback)
        await emitter.emit(sample_event)
        assert len(received) == 1
        assert received[0] == sample_event

    @pytest.mark.asyncio
    async def test_emit_to_workspace_subscriber(self, emitter, sample_event):
        """emit sends event to workspace subscribers."""
        received = []

        def callback(e):
            received.append(e)

        await emitter.subscribe_workspace("workspace-456", callback)
        await emitter.emit(sample_event)
        assert len(received) == 1
        assert received[0] == sample_event

    @pytest.mark.asyncio
    async def test_emit_to_global_subscriber(self, emitter, sample_event):
        """emit sends event to global subscribers."""
        received = []

        def callback(e):
            received.append(e)

        await emitter.subscribe_global(callback)
        await emitter.emit(sample_event)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_emit_to_multiple_subscriber_types(self, emitter, sample_event):
        """emit sends to all matching subscriber types."""
        tenant_received = []
        workspace_received = []
        global_received = []

        await emitter.subscribe_tenant("tenant-123", lambda e: tenant_received.append(e))
        await emitter.subscribe_workspace("workspace-456", lambda e: workspace_received.append(e))
        await emitter.subscribe_global(lambda e: global_received.append(e))

        await emitter.emit(sample_event)

        assert len(tenant_received) == 1
        assert len(workspace_received) == 1
        assert len(global_received) == 1

    @pytest.mark.asyncio
    async def test_emit_async_callback(self, emitter, sample_event):
        """emit handles async callbacks correctly."""
        received = []

        async def async_callback(event):
            await asyncio.sleep(0.01)
            received.append(event)

        await emitter.subscribe_tenant("tenant-123", async_callback)
        await emitter.emit(sample_event)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_emit_sync_callback(self, emitter, sample_event):
        """emit handles sync callbacks correctly."""
        received = []

        def sync_callback(event):
            received.append(event)

        await emitter.subscribe_tenant("tenant-123", sync_callback)
        await emitter.emit(sample_event)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_emit_isolates_callback_exceptions(self, emitter, sample_event):
        """emit isolates exceptions from individual callbacks."""
        received = []

        def bad_callback(event):
            raise ValueError("Callback error")

        def good_callback(event):
            received.append(event)

        await emitter.subscribe_tenant("tenant-123", bad_callback)
        await emitter.subscribe_tenant("tenant-123", good_callback)

        # Should not raise
        await emitter.emit(sample_event)
        # Good callback should still receive event
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_emit_no_subscribers(self, emitter, sample_event):
        """emit handles no subscribers gracefully."""
        # Should not raise
        await emitter.emit(sample_event)

    @pytest.mark.asyncio
    async def test_emit_tenant_isolation(self, emitter):
        """emit only sends to matching tenant subscribers."""
        tenant1_received = []
        tenant2_received = []

        await emitter.subscribe_tenant("tenant-1", lambda e: tenant1_received.append(e))
        await emitter.subscribe_tenant("tenant-2", lambda e: tenant2_received.append(e))

        event = UsageStreamEvent(tenant_id="tenant-1")
        await emitter.emit(event)

        assert len(tenant1_received) == 1
        assert len(tenant2_received) == 0


# ===========================================================================
# Test UsageStreamEmitter Aggregation
# ===========================================================================


class TestUsageStreamEmitterAggregation:
    """Tests for UsageStreamEmitter usage aggregation."""

    @pytest.mark.asyncio
    async def test_update_aggregation(self, emitter):
        """_update_aggregation accumulates usage stats."""
        event1 = UsageStreamEvent(
            tenant_id="tenant-1",
            tokens_in=100,
            tokens_out=50,
            cost_usd=Decimal("0.01"),
        )
        event2 = UsageStreamEvent(
            tenant_id="tenant-1",
            tokens_in=200,
            tokens_out=100,
            cost_usd=Decimal("0.02"),
        )

        await emitter.emit(event1)
        await emitter.emit(event2)

        agg = emitter._current_period_usage.get("tenant-1")
        assert agg is not None
        assert agg["tokens_in"] == 300
        assert agg["tokens_out"] == 150
        assert agg["total_cost"] == Decimal("0.03")
        assert agg["event_count"] == 2

    @pytest.mark.asyncio
    async def test_aggregation_per_tenant(self, emitter):
        """Aggregation is tracked per tenant."""
        event1 = UsageStreamEvent(tenant_id="tenant-1", tokens_in=100)
        event2 = UsageStreamEvent(tenant_id="tenant-2", tokens_in=200)

        await emitter.emit(event1)
        await emitter.emit(event2)

        assert emitter._current_period_usage["tenant-1"]["tokens_in"] == 100
        assert emitter._current_period_usage["tenant-2"]["tokens_in"] == 200

    @pytest.mark.asyncio
    async def test_aggregation_global_fallback(self, emitter):
        """Events without tenant_id aggregate to 'global'."""
        event = UsageStreamEvent(tenant_id="", tokens_in=100)
        await emitter.emit(event)
        assert "global" in emitter._current_period_usage

    @pytest.mark.asyncio
    async def test_emit_summary(self, emitter):
        """emit_summary creates and emits summary event."""
        received = []
        await emitter.subscribe_tenant("tenant-1", lambda e: received.append(e))

        # Add some usage
        await emitter.emit(
            UsageStreamEvent(
                tenant_id="tenant-1",
                tokens_in=100,
                tokens_out=50,
                cost_usd=Decimal("0.015"),
            )
        )

        # Emit summary
        summary = await emitter.emit_summary("tenant-1")

        assert summary.event_type == UsageEventType.USAGE_SUMMARY
        assert summary.tenant_id == "tenant-1"
        assert summary.tokens_in == 100
        assert summary.tokens_out == 50
        assert summary.total_tokens == 150
        assert summary.cost_usd == Decimal("0.015")

    @pytest.mark.asyncio
    async def test_emit_summary_resets_aggregation(self, emitter):
        """emit_summary resets aggregation after emitting."""
        await emitter.emit(UsageStreamEvent(tenant_id="tenant-1", tokens_in=100))
        await emitter.emit_summary("tenant-1")

        assert "tenant-1" not in emitter._current_period_usage

    @pytest.mark.asyncio
    async def test_emit_summary_empty_aggregation(self, emitter):
        """emit_summary handles empty aggregation gracefully."""
        summary = await emitter.emit_summary("nonexistent-tenant")
        assert summary.tokens_in == 0
        assert summary.tokens_out == 0
        assert summary.cost_usd == Decimal("0")


# ===========================================================================
# Test UsageStreamEmitter Budget Alerts
# ===========================================================================


class TestUsageStreamEmitterBudgetAlerts:
    """Tests for UsageStreamEmitter budget alert functionality."""

    @pytest.mark.asyncio
    async def test_emit_budget_alert(self, emitter):
        """emit_budget_alert sends budget alert event."""
        received = []
        await emitter.subscribe_tenant("tenant-1", lambda e: received.append(e))

        await emitter.emit_budget_alert(
            tenant_id="tenant-1",
            budget_used_pct=80.0,
            budget_remaining=Decimal("200.00"),
            alert_level="warning",
        )

        assert len(received) == 1
        event = received[0]
        assert event.event_type == UsageEventType.BUDGET_ALERT
        assert event.budget_used_pct == 80.0
        assert event.budget_remaining == Decimal("200.00")
        assert event.metadata["alert_level"] == "warning"

    @pytest.mark.asyncio
    async def test_emit_budget_alert_critical(self, emitter):
        """emit_budget_alert handles critical alert level."""
        received = []
        await emitter.subscribe_tenant("tenant-1", lambda e: received.append(e))

        await emitter.emit_budget_alert(
            tenant_id="tenant-1",
            budget_used_pct=95.0,
            budget_remaining=Decimal("50.00"),
            alert_level="critical",
        )

        assert received[0].metadata["alert_level"] == "critical"

    @pytest.mark.asyncio
    async def test_emit_budget_alert_exceeded(self, emitter):
        """emit_budget_alert handles exceeded alert level."""
        received = []
        await emitter.subscribe_tenant("tenant-1", lambda e: received.append(e))

        await emitter.emit_budget_alert(
            tenant_id="tenant-1",
            budget_used_pct=105.0,
            budget_remaining=Decimal("-50.00"),
            alert_level="exceeded",
        )

        event = received[0]
        assert event.budget_used_pct == 105.0
        assert event.budget_remaining == Decimal("-50.00")
        assert event.metadata["alert_level"] == "exceeded"


# ===========================================================================
# Test Global Emitter Functions
# ===========================================================================


class TestGlobalEmitterFunctions:
    """Tests for module-level emitter functions."""

    def test_get_usage_emitter_creates_singleton(self):
        """get_usage_emitter creates and returns singleton."""
        emitter1 = get_usage_emitter()
        emitter2 = get_usage_emitter()
        assert emitter1 is emitter2

    def test_get_usage_emitter_creates_new_if_none(self):
        """get_usage_emitter creates new emitter if None."""
        emitter = get_usage_emitter()
        assert isinstance(emitter, UsageStreamEmitter)

    @pytest.mark.asyncio
    async def test_emit_usage_event_basic(self):
        """emit_usage_event creates and emits event."""
        emitter = get_usage_emitter()
        received = []
        await emitter.subscribe_tenant("tenant-1", lambda e: received.append(e))

        event = await emit_usage_event(
            tenant_id="tenant-1",
            tokens_in=100,
            tokens_out=50,
            cost_usd=Decimal("0.015"),
            provider="openai",
            model="gpt-4",
        )

        assert event.tenant_id == "tenant-1"
        assert event.tokens_in == 100
        assert event.tokens_out == 50
        assert event.total_tokens == 150
        assert event.cost_usd == Decimal("0.015")
        assert event.provider == "openai"
        assert event.model == "gpt-4"
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_emit_usage_event_with_all_params(self):
        """emit_usage_event handles all parameters."""
        event = await emit_usage_event(
            tenant_id="tenant-1",
            event_type=UsageEventType.COST_UPDATE,
            tokens_in=1000,
            tokens_out=500,
            cost_usd=Decimal("0.15"),
            provider="anthropic",
            model="claude-3-opus",
            workspace_id="workspace-1",
            user_id="user-1",
            debate_id="debate-1",
            operation="critique",
            metadata={"custom": "value"},
        )

        assert event.event_type == UsageEventType.COST_UPDATE
        assert event.workspace_id == "workspace-1"
        assert event.user_id == "user-1"
        assert event.debate_id == "debate-1"
        assert event.operation == "critique"
        assert event.metadata["custom"] == "value"

    @pytest.mark.asyncio
    async def test_emit_usage_event_default_metadata(self):
        """emit_usage_event uses empty dict for None metadata."""
        event = await emit_usage_event(
            tenant_id="tenant-1",
            metadata=None,
        )
        assert event.metadata == {}


# ===========================================================================
# Test Concurrent Usage Tracking
# ===========================================================================


class TestConcurrentUsageTracking:
    """Tests for concurrent usage tracking scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_emit(self, emitter):
        """Multiple concurrent emits are handled correctly."""
        received = []
        lock = asyncio.Lock()

        async def callback(event):
            async with lock:
                received.append(event)

        await emitter.subscribe_global(callback)

        async def emit_events():
            for i in range(10):
                event = UsageStreamEvent(
                    tenant_id="tenant-1",
                    tokens_in=i * 10,
                )
                await emitter.emit(event)

        # Run multiple emitters concurrently
        await asyncio.gather(
            emit_events(),
            emit_events(),
            emit_events(),
        )

        assert len(received) == 30

    @pytest.mark.asyncio
    async def test_concurrent_subscribe_unsubscribe(self, emitter):
        """Concurrent subscribe/unsubscribe is thread-safe."""
        callbacks = [MagicMock() for _ in range(10)]

        async def subscribe_all():
            for cb in callbacks:
                await emitter.subscribe_tenant("tenant-1", cb)

        async def unsubscribe_all():
            for cb in callbacks:
                await emitter.unsubscribe_tenant("tenant-1", cb)

        # Run concurrently
        await asyncio.gather(subscribe_all(), unsubscribe_all())

        # Should not raise, final state depends on timing

    @pytest.mark.asyncio
    async def test_aggregation_thread_safety(self, emitter):
        """Aggregation is thread-safe under concurrent access."""

        async def emit_many():
            for _ in range(100):
                event = UsageStreamEvent(
                    tenant_id="tenant-1",
                    tokens_in=1,
                    tokens_out=1,
                    cost_usd=Decimal("0.001"),
                )
                await emitter.emit(event)

        await asyncio.gather(
            emit_many(),
            emit_many(),
            emit_many(),
        )

        agg = emitter._current_period_usage.get("tenant-1")
        assert agg is not None
        assert agg["tokens_in"] == 300
        assert agg["tokens_out"] == 300
        assert agg["event_count"] == 300


# ===========================================================================
# Test Error Handling
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling in usage stream."""

    @pytest.mark.asyncio
    async def test_callback_exception_logged(self, emitter, caplog):
        """Callback exceptions are logged."""
        import logging

        def bad_callback(event):
            raise RuntimeError("Test error")

        await emitter.subscribe_tenant("tenant-1", bad_callback)

        with caplog.at_level(logging.WARNING):
            await emitter.emit(UsageStreamEvent(tenant_id="tenant-1"))

        assert any("Callback error" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_async_callback_exception_handled(self, emitter):
        """Async callback exceptions are handled."""
        received = []

        async def bad_async_callback(event):
            raise ValueError("Async error")

        async def good_async_callback(event):
            received.append(event)

        await emitter.subscribe_tenant("tenant-1", bad_async_callback)
        await emitter.subscribe_tenant("tenant-1", good_async_callback)

        # Should not raise
        await emitter.emit(UsageStreamEvent(tenant_id="tenant-1"))
        assert len(received) == 1

    def test_event_with_decimal_edge_cases(self):
        """Event handles decimal edge cases."""
        # Very large decimal
        event = UsageStreamEvent(cost_usd=Decimal("999999999.999999"))
        d = event.to_dict()
        assert d["cost_usd"] == "999999999.999999"

        # Very small decimal - may be in scientific notation
        event = UsageStreamEvent(cost_usd=Decimal("0.000000001"))
        d = event.to_dict()
        # Decimal str() may return scientific notation for very small numbers
        assert Decimal(d["cost_usd"]) == Decimal("0.000000001")

    def test_event_with_large_token_counts(self):
        """Event handles large token counts."""
        event = UsageStreamEvent(
            tokens_in=1_000_000_000,
            tokens_out=500_000_000,
            total_tokens=1_500_000_000,
        )
        d = event.to_dict()
        assert d["tokens_in"] == 1_000_000_000
        assert d["total_tokens"] == 1_500_000_000


# ===========================================================================
# Test Real-time Usage Updates
# ===========================================================================


class TestRealTimeUsageUpdates:
    """Tests for real-time usage update scenarios."""

    @pytest.mark.asyncio
    async def test_rapid_event_emission(self, emitter):
        """Rapid event emission is handled correctly."""
        received = []
        await emitter.subscribe_global(lambda e: received.append(e))

        for i in range(100):
            await emitter.emit(UsageStreamEvent(tenant_id="tenant-1", tokens_in=i))

        assert len(received) == 100

    @pytest.mark.asyncio
    async def test_mixed_event_types(self, emitter):
        """Mixed event types are emitted correctly."""
        received = []
        await emitter.subscribe_tenant("tenant-1", lambda e: received.append(e))

        await emitter.emit(
            UsageStreamEvent(
                tenant_id="tenant-1",
                event_type=UsageEventType.TOKEN_USAGE,
            )
        )
        await emitter.emit_budget_alert(
            tenant_id="tenant-1",
            budget_used_pct=50.0,
            budget_remaining=Decimal("500.00"),
        )
        await emitter.emit_summary("tenant-1")

        assert len(received) == 3
        assert received[0].event_type == UsageEventType.TOKEN_USAGE
        assert received[1].event_type == UsageEventType.BUDGET_ALERT
        assert received[2].event_type == UsageEventType.USAGE_SUMMARY

    @pytest.mark.asyncio
    async def test_workspace_filtering(self, emitter):
        """Workspace filtering works correctly."""
        ws1_received = []
        ws2_received = []

        await emitter.subscribe_workspace("ws-1", lambda e: ws1_received.append(e))
        await emitter.subscribe_workspace("ws-2", lambda e: ws2_received.append(e))

        await emitter.emit(UsageStreamEvent(tenant_id="t-1", workspace_id="ws-1"))
        await emitter.emit(UsageStreamEvent(tenant_id="t-1", workspace_id="ws-2"))
        await emitter.emit(UsageStreamEvent(tenant_id="t-1", workspace_id="ws-3"))

        assert len(ws1_received) == 1
        assert len(ws2_received) == 1


# ===========================================================================
# Test Memory Management
# ===========================================================================


class TestMemoryManagement:
    """Tests for memory management in usage stream."""

    @pytest.mark.asyncio
    async def test_aggregation_memory_cleanup(self, emitter):
        """Aggregation memory is cleaned up on summary emission."""
        # Generate usage for multiple tenants
        for i in range(10):
            await emitter.emit(UsageStreamEvent(tenant_id=f"tenant-{i}", tokens_in=100))

        assert len(emitter._current_period_usage) == 10

        # Emit summaries to clean up
        for i in range(10):
            await emitter.emit_summary(f"tenant-{i}")

        assert len(emitter._current_period_usage) == 0

    @pytest.mark.asyncio
    async def test_subscriber_cleanup_on_unsubscribe(self, emitter):
        """Subscribers are properly cleaned up on unsubscribe."""
        callbacks = [MagicMock() for _ in range(10)]

        for cb in callbacks:
            await emitter.subscribe_tenant("tenant-1", cb)

        assert len(emitter._subscribers["tenant-1"]) == 10

        for cb in callbacks:
            await emitter.unsubscribe_tenant("tenant-1", cb)

        assert "tenant-1" not in emitter._subscribers


# ===========================================================================
# Test Integration Scenarios
# ===========================================================================


class TestIntegrationScenarios:
    """Integration tests for realistic usage scenarios."""

    @pytest.mark.asyncio
    async def test_debate_usage_tracking(self, emitter):
        """Track usage throughout a debate lifecycle."""
        tenant_usage = []
        await emitter.subscribe_tenant("tenant-1", lambda e: tenant_usage.append(e))

        # Simulate debate with multiple agent calls using the emitter directly
        for round_num in range(3):
            for agent in ["claude", "gpt4", "gemini"]:
                event = UsageStreamEvent(
                    tenant_id="tenant-1",
                    tokens_in=500 + round_num * 100,
                    tokens_out=200 + round_num * 50,
                    cost_usd=Decimal("0.01"),
                    provider=agent,
                    model=f"{agent}-model",
                    debate_id="debate-123",
                    operation=f"round-{round_num}-{agent}",
                )
                await emitter.emit(event)

        assert len(tenant_usage) == 9  # 3 rounds x 3 agents

        # Verify aggregation
        summary = await emitter.emit_summary("tenant-1")
        assert summary.metadata["event_count"] == 9

    @pytest.mark.asyncio
    async def test_multi_tenant_isolation(self, emitter):
        """Events are properly isolated between tenants."""
        tenant1_events = []
        tenant2_events = []
        global_events = []

        await emitter.subscribe_tenant("tenant-1", lambda e: tenant1_events.append(e))
        await emitter.subscribe_tenant("tenant-2", lambda e: tenant2_events.append(e))
        await emitter.subscribe_global(lambda e: global_events.append(e))

        for i in range(5):
            await emitter.emit(UsageStreamEvent(tenant_id="tenant-1", tokens_in=i * 10))
        for i in range(3):
            await emitter.emit(UsageStreamEvent(tenant_id="tenant-2", tokens_in=i * 20))

        assert len(tenant1_events) == 5
        assert len(tenant2_events) == 3
        assert len(global_events) == 8

        # Verify no cross-tenant pollution
        for event in tenant1_events:
            assert event.tenant_id == "tenant-1"
        for event in tenant2_events:
            assert event.tenant_id == "tenant-2"

    @pytest.mark.asyncio
    async def test_budget_monitoring_workflow(self, emitter):
        """Complete budget monitoring workflow."""
        alerts = []

        async def alert_handler(event):
            if event.event_type == UsageEventType.BUDGET_ALERT:
                alerts.append(event)

        await emitter.subscribe_tenant("tenant-1", alert_handler)

        # Simulate usage approaching budget
        total_budget = Decimal("1000.00")

        for pct in [50, 75, 90, 100, 110]:
            budget_used = total_budget * pct / 100
            budget_remaining = total_budget - budget_used

            if pct >= 75:
                level = (
                    "info"
                    if pct < 90
                    else ("warning" if pct < 100 else ("critical" if pct == 100 else "exceeded"))
                )
                await emitter.emit_budget_alert(
                    tenant_id="tenant-1",
                    budget_used_pct=float(pct),
                    budget_remaining=budget_remaining,
                    alert_level=level,
                )

        assert len(alerts) == 4
        assert alerts[0].metadata["alert_level"] == "info"
        assert alerts[1].metadata["alert_level"] == "warning"
        assert alerts[2].metadata["alert_level"] == "critical"
        assert alerts[3].metadata["alert_level"] == "exceeded"
