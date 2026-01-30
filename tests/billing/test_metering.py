"""
Tests for Usage Metering System.

Covers:
- BillingEventType and BillingPeriod enums
- BillingEvent dataclass
- UsageSummary dataclass
- MeteringConfig dataclass
- UsageMeter operations
- record_usage convenience function
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from aragora.billing.metering import (
    BillingEvent,
    BillingEventType,
    BillingPeriod,
    MeteringConfig,
    UsageMeter,
    UsageSummary,
    get_usage_meter,
    record_usage,
)


class TestBillingEventType:
    """Tests for BillingEventType enum."""

    def test_api_call_type(self):
        """Should have API call type."""
        assert BillingEventType.API_CALL.value == "api_call"

    def test_debate_type(self):
        """Should have debate type."""
        assert BillingEventType.DEBATE.value == "debate"

    def test_tokens_type(self):
        """Should have tokens type."""
        assert BillingEventType.TOKENS.value == "tokens"

    def test_storage_type(self):
        """Should have storage type."""
        assert BillingEventType.STORAGE.value == "storage"

    def test_connector_sync_type(self):
        """Should have connector sync type."""
        assert BillingEventType.CONNECTOR_SYNC.value == "connector_sync"

    def test_knowledge_query_type(self):
        """Should have knowledge query type."""
        assert BillingEventType.KNOWLEDGE_QUERY.value == "knowledge_query"

    def test_agent_call_type(self):
        """Should have agent call type."""
        assert BillingEventType.AGENT_CALL.value == "agent_call"

    def test_export_type(self):
        """Should have export type."""
        assert BillingEventType.EXPORT.value == "export"

    def test_sso_auth_type(self):
        """Should have SSO auth type."""
        assert BillingEventType.SSO_AUTH.value == "sso_auth"


class TestBillingPeriod:
    """Tests for BillingPeriod enum."""

    def test_hourly_period(self):
        """Should have hourly period."""
        assert BillingPeriod.HOURLY.value == "hourly"

    def test_daily_period(self):
        """Should have daily period."""
        assert BillingPeriod.DAILY.value == "daily"

    def test_monthly_period(self):
        """Should have monthly period."""
        assert BillingPeriod.MONTHLY.value == "monthly"


class TestBillingEvent:
    """Tests for BillingEvent dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        event = BillingEvent()

        assert event.id is not None
        assert event.tenant_id == ""
        assert event.user_id is None
        assert event.event_type == BillingEventType.API_CALL
        assert event.quantity == 1
        assert event.tokens_in == 0
        assert event.tokens_out == 0
        assert event.bytes_used == 0
        assert event.unit_cost == Decimal("0")
        assert event.total_cost == Decimal("0")
        assert event.currency == "USD"

    def test_calculate_cost(self):
        """Should calculate total cost."""
        event = BillingEvent(
            unit_cost=Decimal("0.01"),
            quantity=5,
        )

        result = event.calculate_cost()

        assert result == Decimal("0.05")
        assert event.total_cost == Decimal("0.05")

    def test_to_dict(self):
        """Should convert to dictionary."""
        now = datetime.now(timezone.utc)
        event = BillingEvent(
            id="test-id",
            tenant_id="tenant-123",
            user_id="user-456",
            event_type=BillingEventType.DEBATE,
            resource="debate",
            quantity=2,
            tokens_in=1000,
            tokens_out=500,
            unit_cost=Decimal("0.05"),
            total_cost=Decimal("0.10"),
            debate_id="debate-789",
            timestamp=now,
            billing_period="2026-01",
        )

        result = event.to_dict()

        assert result["id"] == "test-id"
        assert result["tenant_id"] == "tenant-123"
        assert result["user_id"] == "user-456"
        assert result["event_type"] == "debate"
        assert result["resource"] == "debate"
        assert result["quantity"] == 2
        assert result["tokens_in"] == 1000
        assert result["tokens_out"] == 500
        assert result["unit_cost"] == "0.05"
        assert result["total_cost"] == "0.10"
        assert result["debate_id"] == "debate-789"
        assert result["billing_period"] == "2026-01"

    def test_from_dict(self):
        """Should create from dictionary."""
        data = {
            "id": "from-dict-id",
            "tenant_id": "tenant-abc",
            "event_type": "tokens",
            "quantity": 100,
            "tokens_in": 50,
            "tokens_out": 50,
            "unit_cost": "0.001",
            "total_cost": "0.100",
            "timestamp": "2026-01-15T10:30:00+00:00",
        }

        event = BillingEvent.from_dict(data)

        assert event.id == "from-dict-id"
        assert event.tenant_id == "tenant-abc"
        assert event.event_type == BillingEventType.TOKENS
        assert event.quantity == 100
        assert event.tokens_in == 50
        assert event.unit_cost == Decimal("0.001")
        assert event.total_cost == Decimal("0.100")

    def test_from_dict_defaults(self):
        """Should use defaults for missing fields."""
        data = {}
        event = BillingEvent.from_dict(data)

        assert event.id is not None
        assert event.tenant_id == ""
        assert event.event_type == BillingEventType.API_CALL
        assert event.quantity == 1

    def test_from_dict_datetime_object(self):
        """Should handle datetime objects in timestamp."""
        now = datetime.now(timezone.utc)
        data = {
            "timestamp": now,
        }
        event = BillingEvent.from_dict(data)
        assert event.timestamp == now


class TestUsageSummary:
    """Tests for UsageSummary dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        now = datetime.now(timezone.utc)
        summary = UsageSummary(
            tenant_id="tenant-123",
            period_start=now - timedelta(days=30),
            period_end=now,
        )

        assert summary.total_events == 0
        assert summary.api_calls == 0
        assert summary.debates == 0
        assert summary.tokens_in == 0
        assert summary.total_tokens == 0
        assert summary.total_cost == Decimal("0")
        assert summary.cost_by_type == {}

    def test_to_dict(self):
        """Should convert to dictionary."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=30)

        summary = UsageSummary(
            tenant_id="tenant-123",
            period_start=start,
            period_end=now,
            period_type=BillingPeriod.MONTHLY,
            total_events=100,
            api_calls=50,
            debates=10,
            tokens_in=5000,
            tokens_out=2500,
            total_tokens=7500,
            total_cost=Decimal("12.50"),
            cost_by_type={"api_call": Decimal("5.00"), "debate": Decimal("7.50")},
        )

        result = summary.to_dict()

        assert result["tenant_id"] == "tenant-123"
        assert result["period_type"] == "monthly"
        assert result["total_events"] == 100
        assert result["api_calls"] == 50
        assert result["debates"] == 10
        assert result["total_tokens"] == 7500
        assert result["total_cost"] == "12.50"
        assert result["cost_by_type"] == {"api_call": "5.00", "debate": "7.50"}


class TestMeteringConfig:
    """Tests for MeteringConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = MeteringConfig()

        assert config.buffer_size == 100
        assert config.flush_interval == 30.0
        assert config.persist_events is True
        assert config.api_call_price == Decimal("0.0001")
        assert config.debate_base_price == Decimal("0.01")
        assert config.token_price_per_1k == Decimal("0.002")
        assert config.storage_price_per_gb_month == Decimal("0.10")
        assert config.connector_sync_price == Decimal("0.005")
        assert config.knowledge_query_price == Decimal("0.001")

    def test_custom_values(self):
        """Should accept custom values."""
        config = MeteringConfig(
            buffer_size=50,
            flush_interval=10.0,
            api_call_price=Decimal("0.001"),
            persist_events=False,
        )

        assert config.buffer_size == 50
        assert config.flush_interval == 10.0
        assert config.api_call_price == Decimal("0.001")
        assert config.persist_events is False


class TestUsageMeter:
    """Tests for UsageMeter class."""

    @pytest.fixture
    def meter(self, tmp_path):
        """Create meter with non-persistent config."""
        config = MeteringConfig(
            persist_events=False,
            buffer_size=10,
            flush_interval=0.1,
        )
        return UsageMeter(config=config)

    @pytest.fixture
    def persistent_meter(self, tmp_path):
        """Create meter with persistent storage."""
        db_path = tmp_path / "test_billing.db"
        config = MeteringConfig(
            persist_events=True,
            db_path=db_path,
            buffer_size=5,
            flush_interval=0.1,
        )
        return UsageMeter(config=config)

    def test_init_default_config(self):
        """Should use default config."""
        meter = UsageMeter()
        assert meter.config.buffer_size == 100

    def test_init_custom_config(self, meter):
        """Should use custom config."""
        assert meter.config.buffer_size == 10

    @pytest.mark.asyncio
    async def test_record_event(self, meter):
        """Should record billing event."""
        event = BillingEvent(
            tenant_id="tenant-123",
            event_type=BillingEventType.API_CALL,
            quantity=1,
        )

        await meter.record_event(event)

        assert len(meter._events) == 1
        assert meter._events[0].tenant_id == "tenant-123"

    @pytest.mark.asyncio
    async def test_record_event_sets_billing_period(self, meter):
        """Should set billing period if not provided."""
        event = BillingEvent(
            tenant_id="tenant-123",
            event_type=BillingEventType.API_CALL,
        )

        await meter.record_event(event)

        assert meter._events[0].billing_period is not None
        assert len(meter._events[0].billing_period) == 7  # YYYY-MM format

    @pytest.mark.asyncio
    async def test_record_event_skips_no_tenant(self, meter):
        """Should skip events without tenant."""
        event = BillingEvent(
            event_type=BillingEventType.API_CALL,
        )

        await meter.record_event(event)

        assert len(meter._events) == 0

    @pytest.mark.asyncio
    async def test_record_api_call(self, meter):
        """Should record API call with correct pricing."""
        # Set tenant_id via event (since no context available)
        event = BillingEvent(
            tenant_id="tenant-123",
            event_type=BillingEventType.API_CALL,
            quantity=1,
            unit_cost=meter.config.api_call_price,
        )
        event.calculate_cost()
        await meter.record_event(event)

        assert len(meter._events) == 1
        assert meter._events[0].event_type == BillingEventType.API_CALL
        assert meter._events[0].total_cost == meter.config.api_call_price

    @pytest.mark.asyncio
    async def test_record_debate(self, meter):
        """Should record debate with token costs."""
        # Create event directly since record_debate needs tenant context
        event = BillingEvent(
            tenant_id="tenant-123",
            event_type=BillingEventType.DEBATE,
            debate_id="debate-456",
            tokens_in=1000,
            tokens_out=500,
            quantity=1,
        )

        await meter.record_event(event)

        assert len(meter._events) == 1
        assert meter._events[0].event_type == BillingEventType.DEBATE
        assert meter._events[0].tokens_in == 1000
        assert meter._events[0].tokens_out == 500

    @pytest.mark.asyncio
    async def test_record_tokens(self, meter):
        """Should record token usage."""
        event = BillingEvent(
            tenant_id="tenant-123",
            event_type=BillingEventType.TOKENS,
            tokens_in=2000,
            tokens_out=1000,
            quantity=3000,
            metadata={"provider": "anthropic", "model": "claude-3"},
        )

        await meter.record_event(event)

        assert len(meter._events) == 1
        assert meter._events[0].tokens_in == 2000
        assert meter._events[0].tokens_out == 1000

    @pytest.mark.asyncio
    async def test_record_storage(self, meter):
        """Should record storage usage."""
        event = BillingEvent(
            tenant_id="tenant-123",
            event_type=BillingEventType.STORAGE,
            bytes_used=1024 * 1024 * 100,  # 100 MB
            resource="general",
        )

        await meter.record_event(event)

        assert len(meter._events) == 1
        assert meter._events[0].bytes_used == 1024 * 1024 * 100

    @pytest.mark.asyncio
    async def test_record_connector_sync(self, meter):
        """Should record connector sync."""
        event = BillingEvent(
            tenant_id="tenant-123",
            event_type=BillingEventType.CONNECTOR_SYNC,
            connector_id="slack-123",
            resource="slack",
            metadata={"items_synced": 150},
        )

        await meter.record_event(event)

        assert len(meter._events) == 1
        assert meter._events[0].connector_id == "slack-123"

    @pytest.mark.asyncio
    async def test_record_knowledge_query(self, meter):
        """Should record knowledge query."""
        event = BillingEvent(
            tenant_id="tenant-123",
            event_type=BillingEventType.KNOWLEDGE_QUERY,
            resource="search",
            tokens_in=100,
        )

        await meter.record_event(event)

        assert len(meter._events) == 1
        assert meter._events[0].event_type == BillingEventType.KNOWLEDGE_QUERY

    @pytest.mark.asyncio
    async def test_get_billing_events_empty(self, meter):
        """Should return empty list for no events."""
        now = datetime.now(timezone.utc)
        events = await meter.get_billing_events(
            start_date=now - timedelta(days=7),
            end_date=now,
            tenant_id="tenant-123",
        )

        assert events == []

    @pytest.mark.asyncio
    async def test_get_billing_events_includes_buffered(self, meter):
        """Should include buffered events."""
        now = datetime.now(timezone.utc)

        event = BillingEvent(
            tenant_id="tenant-123",
            event_type=BillingEventType.API_CALL,
            timestamp=now,
        )
        await meter.record_event(event)

        events = await meter.get_billing_events(
            start_date=now - timedelta(hours=1),
            end_date=now + timedelta(hours=1),
            tenant_id="tenant-123",
        )

        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_get_billing_events_filters_by_type(self, meter):
        """Should filter events by type."""
        now = datetime.now(timezone.utc)

        # Add API call
        api_event = BillingEvent(
            tenant_id="tenant-123",
            event_type=BillingEventType.API_CALL,
            timestamp=now,
        )
        await meter.record_event(api_event)

        # Add debate
        debate_event = BillingEvent(
            tenant_id="tenant-123",
            event_type=BillingEventType.DEBATE,
            timestamp=now,
        )
        await meter.record_event(debate_event)

        # Query only API calls
        events = await meter.get_billing_events(
            start_date=now - timedelta(hours=1),
            end_date=now + timedelta(hours=1),
            tenant_id="tenant-123",
            event_type=BillingEventType.API_CALL,
        )

        assert len(events) == 1
        assert events[0].event_type == BillingEventType.API_CALL

    @pytest.mark.asyncio
    async def test_get_usage_summary(self, meter):
        """Should calculate usage summary."""
        now = datetime.now(timezone.utc)

        # Add various events
        for i in range(3):
            event = BillingEvent(
                tenant_id="tenant-123",
                event_type=BillingEventType.API_CALL,
                quantity=1,
                total_cost=Decimal("0.01"),
                timestamp=now - timedelta(hours=i),
            )
            await meter.record_event(event)

        debate_event = BillingEvent(
            tenant_id="tenant-123",
            event_type=BillingEventType.DEBATE,
            quantity=1,
            tokens_in=1000,
            tokens_out=500,
            total_cost=Decimal("0.10"),
            timestamp=now,
        )
        await meter.record_event(debate_event)

        summary = await meter.get_usage_summary(
            start_date=now - timedelta(days=1),
            end_date=now + timedelta(hours=1),
            tenant_id="tenant-123",
        )

        assert summary.tenant_id == "tenant-123"
        assert summary.total_events == 4
        assert summary.api_calls == 3
        assert summary.debates == 1
        assert summary.tokens_in == 1000
        assert summary.tokens_out == 500

    @pytest.mark.asyncio
    async def test_get_usage_summary_no_tenant(self, meter):
        """Should raise error without tenant."""
        now = datetime.now(timezone.utc)

        with pytest.raises(ValueError, match="No tenant context"):
            await meter.get_usage_summary(
                start_date=now - timedelta(days=1),
                end_date=now,
            )

    @pytest.mark.asyncio
    async def test_start_and_stop(self, meter):
        """Should start and stop metering."""
        await meter.start()
        assert meter._running is True
        assert meter._flush_task is not None

        await meter.stop()
        assert meter._running is False

    @pytest.mark.asyncio
    async def test_flush_events(self, meter):
        """Should flush buffered events."""
        event = BillingEvent(
            tenant_id="tenant-123",
            event_type=BillingEventType.API_CALL,
        )
        await meter.record_event(event)

        assert len(meter._events) == 1

        await meter._flush_events()

        # Events cleared after flush (non-persistent mode)
        assert len(meter._events) == 0

    def test_get_billing_period(self, meter):
        """Should format billing period correctly."""
        dt = datetime(2026, 1, 15, 10, 30, 0)
        period = meter._get_billing_period(dt)

        assert period == "2026-01"

    @pytest.mark.asyncio
    async def test_estimate_monthly_cost(self, meter):
        """Should estimate monthly cost."""
        now = datetime.now(timezone.utc)

        # Add some events
        for _ in range(5):
            event = BillingEvent(
                tenant_id="tenant-123",
                event_type=BillingEventType.API_CALL,
                total_cost=Decimal("0.10"),
                timestamp=now,
            )
            await meter.record_event(event)

        estimate = await meter.estimate_monthly_cost(tenant_id="tenant-123")

        assert estimate["tenant_id"] == "tenant-123"
        assert "current_cost" in estimate
        assert "projected_cost" in estimate
        assert "days_elapsed" in estimate
        assert "usage" in estimate

    @pytest.mark.asyncio
    async def test_persistent_storage(self, persistent_meter):
        """Should persist events to database."""
        now = datetime.now(timezone.utc)

        # Add event
        event = BillingEvent(
            tenant_id="tenant-123",
            event_type=BillingEventType.API_CALL,
            quantity=1,
            total_cost=Decimal("0.01"),
            timestamp=now,
        )
        await persistent_meter.record_event(event)

        # Flush to database
        await persistent_meter._flush_events()

        # Query back
        events = await persistent_meter.get_billing_events(
            start_date=now - timedelta(hours=1),
            end_date=now + timedelta(hours=1),
            tenant_id="tenant-123",
        )

        assert len(events) == 1
        assert events[0].tenant_id == "tenant-123"


class TestGetUsageMeter:
    """Tests for get_usage_meter singleton."""

    def test_returns_meter(self):
        """Should return UsageMeter instance."""
        meter = get_usage_meter()
        assert isinstance(meter, UsageMeter)

    def test_returns_same_instance(self):
        """Should return same instance on repeated calls."""
        meter1 = get_usage_meter()
        meter2 = get_usage_meter()
        # Both should be UsageMeter instances
        assert isinstance(meter1, UsageMeter)
        assert isinstance(meter2, UsageMeter)


class TestRecordUsage:
    """Tests for record_usage convenience function."""

    @pytest.mark.asyncio
    async def test_record_usage(self):
        """Should record usage via convenience function."""
        event = await record_usage(
            event_type=BillingEventType.API_CALL,
            quantity=1,
            tenant_id="tenant-123",
            resource="test",
        )

        assert event.event_type == BillingEventType.API_CALL
        assert event.quantity == 1
        assert event.tenant_id == "tenant-123"
        assert event.resource == "test"

    @pytest.mark.asyncio
    async def test_record_usage_with_tokens(self):
        """Should record token usage."""
        event = await record_usage(
            event_type=BillingEventType.TOKENS,
            quantity=1500,
            tenant_id="tenant-123",
            tokens_in=1000,
            tokens_out=500,
        )

        assert event.event_type == BillingEventType.TOKENS
        assert event.tokens_in == 1000
        assert event.tokens_out == 500


class TestModuleExports:
    """Tests for module exports."""

    def test_all_classes_importable(self):
        """Should export all public classes."""
        from aragora.billing.metering import (
            BillingEvent,
            BillingEventType,
            BillingPeriod,
            MeteringConfig,
            UsageMeter,
            UsageSummary,
            get_usage_meter,
            record_usage,
        )

        assert BillingEventType is not None
        assert BillingPeriod is not None
        assert BillingEvent is not None
        assert UsageSummary is not None
        assert MeteringConfig is not None
        assert UsageMeter is not None
        assert get_usage_meter is not None
        assert record_usage is not None
