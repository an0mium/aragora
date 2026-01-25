"""
Tests for enterprise metering module.

Tests cover:
- TokenUsageRecord dataclass
- BudgetConfig dataclass
- CostBreakdown dataclass
- Invoice dataclass
- EnterpriseMeter class methods
"""

import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from aragora.billing.enterprise_metering import (
    BudgetAlertLevel,
    BudgetConfig,
    CostBreakdown,
    EnterpriseMeter,
    Invoice,
    InvoiceStatus,
    TokenUsageRecord,
    UsageForecast,
)


class TestBudgetAlertLevel:
    """Tests for BudgetAlertLevel enum."""

    def test_alert_levels_exist(self):
        """All alert levels should be defined."""
        assert BudgetAlertLevel.INFO.value == "info"
        assert BudgetAlertLevel.WARNING.value == "warning"
        assert BudgetAlertLevel.CRITICAL.value == "critical"
        assert BudgetAlertLevel.EXCEEDED.value == "exceeded"


class TestInvoiceStatus:
    """Tests for InvoiceStatus enum."""

    def test_invoice_statuses_exist(self):
        """All invoice statuses should be defined."""
        assert InvoiceStatus.DRAFT.value == "draft"
        assert InvoiceStatus.PENDING.value == "pending"
        assert InvoiceStatus.PAID.value == "paid"
        assert InvoiceStatus.OVERDUE.value == "overdue"
        assert InvoiceStatus.CANCELLED.value == "cancelled"


class TestTokenUsageRecord:
    """Tests for TokenUsageRecord dataclass."""

    def test_default_values(self):
        """Should have sensible default values."""
        record = TokenUsageRecord()
        assert record.id is not None
        assert record.tenant_id == ""
        assert record.tokens_in == 0
        assert record.tokens_out == 0
        assert record.total_cost == Decimal("0")
        assert record.request_type == "chat"

    def test_create_with_values(self):
        """Should accept all parameters."""
        record = TokenUsageRecord(
            tenant_id="tenant_1",
            user_id="user_1",
            provider="anthropic",
            model="claude-opus-4",
            tokens_in=1000,
            tokens_out=500,
            total_tokens=1500,
            input_cost=Decimal("0.015"),
            output_cost=Decimal("0.075"),
            total_cost=Decimal("0.09"),
            debate_id="debate_123",
        )
        assert record.tenant_id == "tenant_1"
        assert record.provider == "anthropic"
        assert record.model == "claude-opus-4"
        assert record.tokens_in == 1000
        assert record.tokens_out == 500
        assert record.total_tokens == 1500

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        record = TokenUsageRecord(
            tenant_id="tenant_1",
            provider="openai",
            model="gpt-4",
            tokens_in=100,
            tokens_out=50,
        )
        data = record.to_dict()
        assert isinstance(data, dict)
        assert data["tenant_id"] == "tenant_1"
        assert data["provider"] == "openai"
        assert data["model"] == "gpt-4"
        assert data["tokens_in"] == 100
        assert data["tokens_out"] == 50

    def test_unique_ids(self):
        """Each record should have a unique ID."""
        record1 = TokenUsageRecord()
        record2 = TokenUsageRecord()
        assert record1.id != record2.id


class TestBudgetConfig:
    """Tests for BudgetConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = BudgetConfig()
        assert config.monthly_limit == Decimal("0")
        assert config.alert_thresholds is not None

    def test_create_with_values(self):
        """Should accept all parameters."""
        config = BudgetConfig(
            tenant_id="tenant_1",
            monthly_limit=Decimal("1000.00"),
            daily_limit=Decimal("50.00"),
            per_user_limit=Decimal("100.00"),
        )
        assert config.tenant_id == "tenant_1"
        assert config.monthly_limit == Decimal("1000.00")
        assert config.daily_limit == Decimal("50.00")
        assert config.per_user_limit == Decimal("100.00")


class TestCostBreakdown:
    """Tests for CostBreakdown dataclass."""

    def test_default_values(self):
        """Should initialize with empty collections."""
        breakdown = CostBreakdown()
        assert breakdown.total_cost == Decimal("0")
        assert len(breakdown.by_provider) == 0
        assert len(breakdown.by_model) == 0

    def test_to_dict(self):
        """Should convert to dictionary."""
        breakdown = CostBreakdown(
            total_cost=Decimal("100.00"),
            by_provider={"anthropic": Decimal("60.00"), "openai": Decimal("40.00")},
            by_model={"claude-opus-4": Decimal("60.00"), "gpt-4": Decimal("40.00")},
        )
        data = breakdown.to_dict()
        assert data["total_cost"] == "100.00"
        assert "anthropic" in data["by_provider"]
        assert "claude-opus-4" in data["by_model"]


class TestInvoice:
    """Tests for Invoice dataclass."""

    def test_default_status(self):
        """Should default to DRAFT status."""
        invoice = Invoice()
        assert invoice.status == InvoiceStatus.DRAFT

    def test_create_invoice(self):
        """Should accept all parameters."""
        invoice = Invoice(
            tenant_id="tenant_1",
            period="2025-01",
            total_amount=Decimal("500.00"),
            status=InvoiceStatus.PENDING,
        )
        assert invoice.tenant_id == "tenant_1"
        assert invoice.period == "2025-01"
        assert invoice.total_amount == Decimal("500.00")
        assert invoice.status == InvoiceStatus.PENDING

    def test_to_dict(self):
        """Should convert to dictionary."""
        invoice = Invoice(
            tenant_id="tenant_1",
            period="2025-01",
            total_amount=Decimal("500.00"),
        )
        data = invoice.to_dict()
        assert data["tenant_id"] == "tenant_1"
        assert data["period"] == "2025-01"
        assert data["total_amount"] == "500.00"


class TestUsageForecast:
    """Tests for UsageForecast dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        forecast = UsageForecast()
        assert forecast.confidence == 0.0
        assert len(forecast.daily_projections) == 0

    def test_to_dict(self):
        """Should convert to dictionary."""
        forecast = UsageForecast(
            tenant_id="tenant_1",
            projected_monthly_cost=Decimal("1000.00"),
            confidence=0.85,
        )
        data = forecast.to_dict()
        assert data["tenant_id"] == "tenant_1"
        assert data["projected_monthly_cost"] == "1000.00"
        assert data["confidence"] == 0.85


class TestEnterpriseMeter:
    """Tests for EnterpriseMeter class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_metering.db"

    @pytest.fixture
    async def meter(self, temp_db):
        """Create an initialized meter for testing."""
        meter = EnterpriseMeter(db_path=temp_db)
        await meter.initialize()
        yield meter
        await meter.close()

    @pytest.mark.asyncio
    async def test_initialization(self, temp_db):
        """Should initialize database schema."""
        meter = EnterpriseMeter(db_path=temp_db)
        await meter.initialize()
        assert meter._initialized is True
        await meter.close()

    @pytest.mark.asyncio
    async def test_record_token_usage(self, meter):
        """Should record token usage correctly."""
        record = await meter.record_token_usage(
            provider="anthropic",
            model="claude-3-opus-20240229",
            tokens_in=1000,
            tokens_out=500,
            tenant_id="test_tenant",
            debate_id="debate_123",
        )
        assert record.provider == "anthropic"
        assert record.tokens_in == 1000
        assert record.tokens_out == 500
        assert record.total_tokens == 1500
        assert record.total_cost >= Decimal("0")

    @pytest.mark.asyncio
    async def test_record_usage_with_cache(self, meter):
        """Should apply cache discount correctly."""
        record = await meter.record_token_usage(
            provider="openai",
            model="gpt-4",
            tokens_in=1000,
            tokens_out=500,
            tenant_id="test_tenant",
            cached_tokens=500,  # 50% cache hit
        )
        assert record.cached_tokens == 500
        assert record.discount_applied >= Decimal("0")

    @pytest.mark.asyncio
    async def test_record_usage_default_tenant(self, meter):
        """Should use default tenant if not specified."""
        record = await meter.record_token_usage(
            provider="anthropic",
            model="claude-3-sonnet-20240229",
            tokens_in=100,
            tokens_out=50,
        )
        assert record.tenant_id  # Should have some tenant ID

    @pytest.mark.asyncio
    async def test_calculate_costs(self, meter):
        """Should calculate costs based on provider pricing."""
        input_cost, output_cost = meter._calculate_costs(
            provider="anthropic",
            model="claude-3-opus-20240229",
            tokens_in=1000,
            tokens_out=500,
        )
        assert input_cost >= Decimal("0")
        assert output_cost >= Decimal("0")

    @pytest.mark.asyncio
    async def test_calculate_costs_unknown_provider(self, meter):
        """Should use default pricing for unknown provider."""
        input_cost, output_cost = meter._calculate_costs(
            provider="unknown_provider",
            model="unknown_model",
            tokens_in=1000,
            tokens_out=500,
        )
        # Should not raise, should use default pricing
        assert input_cost >= Decimal("0")
        assert output_cost >= Decimal("0")

    @pytest.mark.asyncio
    async def test_set_and_get_budget(self, meter):
        """Should set and retrieve budget configuration."""
        await meter.set_budget(
            tenant_id="test_tenant",
            monthly_limit=Decimal("1000.00"),
            daily_limit=Decimal("50.00"),
        )
        budget = await meter.get_budget("test_tenant")
        assert budget is not None
        assert budget.tenant_id == "test_tenant"
        assert budget.monthly_limit == Decimal("1000.00")
        assert budget.daily_limit == Decimal("50.00")

    @pytest.mark.asyncio
    async def test_get_budget_nonexistent(self, meter):
        """Should return None for nonexistent tenant budget."""
        budget = await meter.get_budget("nonexistent_tenant")
        assert budget is None

    @pytest.mark.asyncio
    async def test_get_cost_breakdown(self, meter):
        """Should return cost breakdown for tenant."""
        # Record some usage first
        await meter.record_token_usage(
            provider="anthropic",
            model="claude-3-opus-20240229",
            tokens_in=500,
            tokens_out=250,
            tenant_id="breakdown_test",
        )
        await meter.record_token_usage(
            provider="openai",
            model="gpt-4",
            tokens_in=500,
            tokens_out=250,
            tenant_id="breakdown_test",
        )

        # Flush buffer to persist
        await meter._flush_buffer()

        breakdown = await meter.get_cost_breakdown(tenant_id="breakdown_test")
        assert breakdown is not None
        assert breakdown.total_cost >= Decimal("0")

    @pytest.mark.asyncio
    async def test_get_cost_breakdown_with_dates(self, meter):
        """Should filter cost breakdown by date range."""
        now = datetime.utcnow()
        yesterday = now - timedelta(days=1)
        tomorrow = now + timedelta(days=1)

        await meter.record_token_usage(
            provider="anthropic",
            model="claude-3-sonnet-20240229",
            tokens_in=100,
            tokens_out=50,
            tenant_id="date_filter_test",
        )
        await meter._flush_buffer()

        breakdown = await meter.get_cost_breakdown(
            tenant_id="date_filter_test",
            start_date=yesterday,
            end_date=tomorrow,
        )
        assert breakdown is not None

    @pytest.mark.asyncio
    async def test_forecast_usage(self, meter):
        """Should generate usage forecast."""
        # Record some historical usage
        for _ in range(5):
            await meter.record_token_usage(
                provider="anthropic",
                model="claude-3-sonnet-20240229",
                tokens_in=100,
                tokens_out=50,
                tenant_id="forecast_test",
            )
        await meter._flush_buffer()

        forecast = await meter.forecast_usage(
            tenant_id="forecast_test",
            days_ahead=30,
        )
        assert forecast is not None
        assert forecast.tenant_id == "forecast_test"

    @pytest.mark.asyncio
    async def test_generate_invoice(self, meter):
        """Should generate invoice for period."""
        # Record usage
        await meter.record_token_usage(
            provider="anthropic",
            model="claude-3-opus-20240229",
            tokens_in=1000,
            tokens_out=500,
            tenant_id="invoice_test",
        )
        await meter._flush_buffer()

        period = datetime.utcnow().strftime("%Y-%m")
        invoice = await meter.generate_invoice(
            tenant_id="invoice_test",
            period=period,
        )
        assert invoice is not None
        assert invoice.tenant_id == "invoice_test"
        assert invoice.period == period
        assert invoice.status == InvoiceStatus.DRAFT

    @pytest.mark.asyncio
    async def test_get_invoices(self, meter):
        """Should retrieve invoices for tenant."""
        # Generate an invoice first
        await meter.record_token_usage(
            provider="openai",
            model="gpt-4",
            tokens_in=500,
            tokens_out=250,
            tenant_id="invoices_test",
        )
        await meter._flush_buffer()

        period = datetime.utcnow().strftime("%Y-%m")
        await meter.generate_invoice(tenant_id="invoices_test", period=period)

        invoices = await meter.get_invoices(tenant_id="invoices_test")
        assert len(invoices) >= 1

    @pytest.mark.asyncio
    async def test_buffer_flush_on_threshold(self, meter):
        """Should flush buffer when threshold is reached."""
        # Record many items to trigger buffer flush
        original_buffer_size = meter._buffer_size
        meter._buffer_size = 5  # Small buffer for testing

        for i in range(6):
            await meter.record_token_usage(
                provider="anthropic",
                model="claude-3-sonnet-20240229",
                tokens_in=100,
                tokens_out=50,
                tenant_id="buffer_test",
            )

        # Buffer should have been flushed
        assert len(meter._usage_buffer) < 6

        meter._buffer_size = original_buffer_size

    @pytest.mark.asyncio
    async def test_close(self, temp_db):
        """Should close meter gracefully."""
        meter = EnterpriseMeter(db_path=temp_db)
        await meter.initialize()

        # Record some usage
        await meter.record_token_usage(
            provider="anthropic",
            model="claude-3-sonnet-20240229",
            tokens_in=100,
            tokens_out=50,
            tenant_id="close_test",
        )

        await meter.close()
        # Should not raise


class TestCostCalculation:
    """Tests for cost calculation edge cases."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_costs.db"

    @pytest.fixture
    async def meter(self, temp_db):
        """Create an initialized meter."""
        meter = EnterpriseMeter(db_path=temp_db)
        await meter.initialize()
        yield meter
        await meter.close()

    @pytest.mark.asyncio
    async def test_zero_tokens(self, meter):
        """Should handle zero tokens gracefully."""
        record = await meter.record_token_usage(
            provider="anthropic",
            model="claude-3-opus-20240229",
            tokens_in=0,
            tokens_out=0,
            tenant_id="zero_test",
        )
        assert record.total_cost == Decimal("0")

    @pytest.mark.asyncio
    async def test_large_token_count(self, meter):
        """Should handle large token counts."""
        record = await meter.record_token_usage(
            provider="anthropic",
            model="claude-3-opus-20240229",
            tokens_in=1000000,
            tokens_out=500000,
            tenant_id="large_test",
        )
        assert record.total_tokens == 1500000
        assert record.total_cost > Decimal("0")

    @pytest.mark.asyncio
    async def test_multiple_providers(self, meter):
        """Should track costs across multiple providers."""
        providers = ["anthropic", "openai", "mistral"]
        for provider in providers:
            await meter.record_token_usage(
                provider=provider,
                model="default",
                tokens_in=100,
                tokens_out=50,
                tenant_id="multi_provider_test",
            )
        await meter._flush_buffer()

        breakdown = await meter.get_cost_breakdown(tenant_id="multi_provider_test")
        # Should have tracked usage from all providers
        assert breakdown.total_cost > Decimal("0")
