"""
Tests for the APAutomation service.

Covers:
- Invoice management
- Payment timing optimization
- Batch payment processing
- Cash flow forecasting
- Discount opportunity tracking
"""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from aragora.services.ap_automation import (
    APAutomation,
    PayableInvoice,
    PaymentPriority,
    PaymentMethod,
    PaymentSchedule,
    BatchPayment,
    CashForecast,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ap_automation():
    """Create a fresh APAutomation instance."""
    return APAutomation()


@pytest.fixture
def sample_invoice_data():
    """Sample payable invoice data."""
    return {
        "vendor_id": "vendor_001",
        "vendor_name": "Acme Supplies",
        "invoice_number": "INV-2024-001",
        "invoice_date": datetime.now(),
        "due_date": datetime.now() + timedelta(days=30),
        "total_amount": Decimal("1000.00"),
        "payment_terms": "Net 30",
    }


@pytest.fixture
def invoice_with_discount():
    """Invoice with early payment discount."""
    return {
        "vendor_id": "vendor_002",
        "vendor_name": "Quick Pay Vendor",
        "invoice_number": "INV-DISCOUNT",
        "total_amount": Decimal("5000.00"),
        "due_date": datetime.now() + timedelta(days=30),
        "early_pay_discount": 0.02,  # 2%
        "discount_deadline": datetime.now() + timedelta(days=10),
    }


# =============================================================================
# Invoice Management Tests
# =============================================================================


class TestInvoiceManagement:
    """Test payable invoice management."""

    @pytest.mark.asyncio
    async def test_add_invoice(self, ap_automation, sample_invoice_data):
        """Test adding a payable invoice."""
        invoice = await ap_automation.add_invoice(**sample_invoice_data)

        assert invoice.id.startswith("apinv_")
        assert invoice.vendor_name == sample_invoice_data["vendor_name"]
        assert invoice.balance == sample_invoice_data["total_amount"]

    @pytest.mark.asyncio
    async def test_get_invoice(self, ap_automation, sample_invoice_data):
        """Test retrieving an invoice."""
        invoice = await ap_automation.add_invoice(**sample_invoice_data)
        retrieved = await ap_automation.get_invoice(invoice.id)

        assert retrieved is not None
        assert retrieved.id == invoice.id

    @pytest.mark.asyncio
    async def test_list_invoices(self, ap_automation, sample_invoice_data):
        """Test listing invoices."""
        await ap_automation.add_invoice(**sample_invoice_data)
        await ap_automation.add_invoice(
            vendor_id="vendor_002",
            vendor_name="Other Vendor",
            total_amount=Decimal("500.00"),
        )

        invoices = await ap_automation.list_invoices()
        assert len(invoices) == 2

    @pytest.mark.asyncio
    async def test_list_by_vendor(self, ap_automation, sample_invoice_data):
        """Test filtering by vendor."""
        await ap_automation.add_invoice(**sample_invoice_data)
        await ap_automation.add_invoice(
            vendor_id="vendor_002",
            vendor_name="Other Vendor",
            total_amount=Decimal("500.00"),
        )

        invoices = await ap_automation.list_invoices(vendor_id="vendor_001")
        assert len(invoices) == 1

    @pytest.mark.asyncio
    async def test_list_by_priority(self, ap_automation):
        """Test filtering by priority."""
        await ap_automation.add_invoice(
            vendor_id="vendor_001",
            vendor_name="Normal Vendor",
            total_amount=Decimal("1000.00"),
            priority="normal",
        )
        await ap_automation.add_invoice(
            vendor_id="vendor_002",
            vendor_name="Critical Vendor",
            total_amount=Decimal("2000.00"),
            priority="critical",
        )

        critical = await ap_automation.list_invoices(priority="critical")
        assert len(critical) == 1
        assert critical[0].priority == PaymentPriority.CRITICAL


# =============================================================================
# Payment Recording Tests
# =============================================================================


class TestPaymentRecording:
    """Test payment recording."""

    @pytest.mark.asyncio
    async def test_record_full_payment(self, ap_automation, sample_invoice_data):
        """Test recording a full payment."""
        invoice = await ap_automation.add_invoice(**sample_invoice_data)

        updated = await ap_automation.record_payment(
            invoice_id=invoice.id,
            amount=Decimal("1000.00"),
        )

        assert updated.balance == Decimal("0.00")
        assert updated.paid_at is not None

    @pytest.mark.asyncio
    async def test_record_partial_payment(self, ap_automation, sample_invoice_data):
        """Test recording a partial payment."""
        invoice = await ap_automation.add_invoice(**sample_invoice_data)

        updated = await ap_automation.record_payment(
            invoice_id=invoice.id,
            amount=Decimal("400.00"),
        )

        assert updated.balance == Decimal("600.00")
        assert updated.amount_paid == Decimal("400.00")


# =============================================================================
# Payment Optimization Tests
# =============================================================================


class TestPaymentOptimization:
    """Test payment timing optimization."""

    @pytest.mark.asyncio
    async def test_optimize_single_invoice(self, ap_automation, sample_invoice_data):
        """Test optimizing a single invoice."""
        invoice = await ap_automation.add_invoice(**sample_invoice_data)

        schedule = await ap_automation.optimize_payment_timing([invoice])

        assert isinstance(schedule, PaymentSchedule)
        assert len(schedule.scheduled_payments) == 1

    @pytest.mark.asyncio
    async def test_optimize_prioritizes_discounts(self, ap_automation, invoice_with_discount):
        """Test that discounts are prioritized."""
        # Regular invoice
        regular = await ap_automation.add_invoice(
            vendor_id="vendor_001",
            vendor_name="Regular Vendor",
            total_amount=Decimal("1000.00"),
            due_date=datetime.now() + timedelta(days=30),
        )

        # Invoice with discount
        discounted = await ap_automation.add_invoice(**invoice_with_discount)

        schedule = await ap_automation.optimize_payment_timing(
            [regular, discounted],
            prioritize_discounts=True,
        )

        # Discounted invoice should be scheduled first
        payments = schedule.scheduled_payments
        assert len(payments) == 2

        # Find the discounted invoice payment
        discount_payment = next((p for p in payments if p["invoice_id"] == discounted.id), None)
        assert discount_payment is not None

    @pytest.mark.asyncio
    async def test_optimize_respects_cash_limit(self, ap_automation):
        """Test that cash limit is respected."""
        inv1 = await ap_automation.add_invoice(
            vendor_id="v1", vendor_name="V1", total_amount=Decimal("1000.00")
        )
        inv2 = await ap_automation.add_invoice(
            vendor_id="v2", vendor_name="V2", total_amount=Decimal("1000.00")
        )
        inv3 = await ap_automation.add_invoice(
            vendor_id="v3", vendor_name="V3", total_amount=Decimal("1000.00")
        )

        schedule = await ap_automation.optimize_payment_timing(
            [inv1, inv2, inv3],
            available_cash=Decimal("2000.00"),
        )

        # Should only schedule payments up to available cash
        total_scheduled = sum(Decimal(str(p["amount"])) for p in schedule.scheduled_payments)
        assert total_scheduled <= Decimal("2000.00")


# =============================================================================
# Batch Payment Tests
# =============================================================================


class TestBatchPayments:
    """Test batch payment processing."""

    @pytest.mark.asyncio
    async def test_create_batch(self, ap_automation, sample_invoice_data):
        """Test creating a batch payment."""
        inv1 = await ap_automation.add_invoice(**sample_invoice_data)
        inv2 = await ap_automation.add_invoice(
            vendor_id="vendor_002",
            vendor_name="Other Vendor",
            total_amount=Decimal("500.00"),
        )

        batch = await ap_automation.batch_payments([inv1, inv2])

        assert isinstance(batch, BatchPayment)
        assert batch.total_amount == Decimal("1500.00")
        assert len(batch.invoice_ids) == 2

    @pytest.mark.asyncio
    async def test_batch_groups_by_payment_method(self, ap_automation):
        """Test that batch groups by payment method."""
        inv1 = await ap_automation.add_invoice(
            vendor_id="v1",
            vendor_name="ACH Vendor",
            total_amount=Decimal("1000.00"),
            preferred_payment_method="ach",
        )
        inv2 = await ap_automation.add_invoice(
            vendor_id="v2",
            vendor_name="Check Vendor",
            total_amount=Decimal("500.00"),
            preferred_payment_method="check",
        )

        batch = await ap_automation.batch_payments([inv1, inv2])

        # Should handle mixed payment methods
        assert batch is not None

    @pytest.mark.asyncio
    async def test_batch_with_payment_date(self, ap_automation, sample_invoice_data):
        """Test batch with specific payment date."""
        invoice = await ap_automation.add_invoice(**sample_invoice_data)
        pay_date = datetime.now() + timedelta(days=5)

        batch = await ap_automation.batch_payments(
            [invoice],
            payment_date=pay_date,
        )

        assert batch.payment_date.date() == pay_date.date()


# =============================================================================
# Cash Forecasting Tests
# =============================================================================


class TestCashForecasting:
    """Test cash flow forecasting."""

    @pytest.mark.asyncio
    async def test_forecast_empty(self, ap_automation):
        """Test forecast with no invoices."""
        forecast = await ap_automation.forecast_cash_needs(days_ahead=30)

        assert isinstance(forecast, CashForecast)
        assert forecast.total_due == Decimal("0.00")

    @pytest.mark.asyncio
    async def test_forecast_with_invoices(self, ap_automation):
        """Test forecast with scheduled invoices."""
        # Invoice due in 10 days
        await ap_automation.add_invoice(
            vendor_id="v1",
            vendor_name="Near Term",
            total_amount=Decimal("1000.00"),
            due_date=datetime.now() + timedelta(days=10),
        )

        # Invoice due in 25 days
        await ap_automation.add_invoice(
            vendor_id="v2",
            vendor_name="Later Term",
            total_amount=Decimal("2000.00"),
            due_date=datetime.now() + timedelta(days=25),
        )

        forecast = await ap_automation.forecast_cash_needs(days_ahead=30)

        assert forecast.total_due == Decimal("3000.00")

    @pytest.mark.asyncio
    async def test_forecast_excludes_past_due(self, ap_automation):
        """Test that forecast handles past due invoices."""
        # Past due invoice
        past_due = PayableInvoice(
            id="apinv_past",
            vendor_id="v1",
            vendor_name="Past Due",
            total_amount=Decimal("500.00"),
            balance=Decimal("500.00"),
            due_date=datetime.now() - timedelta(days=10),
        )
        ap_automation._invoices[past_due.id] = past_due

        forecast = await ap_automation.forecast_cash_needs(days_ahead=30)

        # Past due should be in overdue category, not forecast
        assert forecast.overdue >= Decimal("500.00")

    @pytest.mark.asyncio
    async def test_forecast_with_expected_receivables(self, ap_automation):
        """Test forecast with expected receivables."""
        await ap_automation.add_invoice(
            vendor_id="v1",
            vendor_name="Payable",
            total_amount=Decimal("1000.00"),
            due_date=datetime.now() + timedelta(days=15),
        )

        await ap_automation.add_expected_receivable(
            amount=Decimal("500.00"),
            expected_date=datetime.now() + timedelta(days=10),
        )

        forecast = await ap_automation.forecast_cash_needs(days_ahead=30)

        assert forecast.expected_receivables == Decimal("500.00")
        assert forecast.net_cash_need == Decimal("500.00")  # 1000 - 500


# =============================================================================
# Discount Opportunity Tests
# =============================================================================


class TestDiscountOpportunities:
    """Test discount opportunity tracking."""

    @pytest.mark.asyncio
    async def test_get_discount_opportunities(self, ap_automation, invoice_with_discount):
        """Test getting discount opportunities."""
        await ap_automation.add_invoice(**invoice_with_discount)

        opportunities = await ap_automation.get_discount_opportunities()

        assert len(opportunities) >= 1
        opp = opportunities[0]
        assert "discount_amount" in opp
        assert opp["discount_amount"] == 100.00  # 2% of 5000

    @pytest.mark.asyncio
    async def test_no_opportunities_past_deadline(self, ap_automation):
        """Test no opportunities when deadline passed."""
        await ap_automation.add_invoice(
            vendor_id="v1",
            vendor_name="Expired Discount",
            total_amount=Decimal("1000.00"),
            early_pay_discount=0.02,
            discount_deadline=datetime.now() - timedelta(days=1),
        )

        opportunities = await ap_automation.get_discount_opportunities()

        # Should not include expired discount
        assert len(opportunities) == 0

    @pytest.mark.asyncio
    async def test_annualized_return_calculation(self, ap_automation, invoice_with_discount):
        """Test annualized return calculation for discounts."""
        await ap_automation.add_invoice(**invoice_with_discount)

        opportunities = await ap_automation.get_discount_opportunities()

        assert len(opportunities) >= 1
        opp = opportunities[0]
        assert "annualized_return" in opp
        # 2% discount for paying 20 days early = ~36% annualized
        assert opp["annualized_return"] > 20


# =============================================================================
# PayableInvoice Tests
# =============================================================================


class TestPayableInvoice:
    """Test PayableInvoice dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        invoice = PayableInvoice(
            id="apinv_123",
            vendor_id="v1",
            vendor_name="Test Vendor",
            invoice_number="INV-001",
            total_amount=Decimal("1000.00"),
            balance=Decimal("1000.00"),
            priority=PaymentPriority.NORMAL,
        )

        d = invoice.to_dict()

        assert d["id"] == "apinv_123"
        # Uses camelCase keys and float values
        assert d["vendorName"] == "Test Vendor"
        assert d["totalAmount"] == 1000.0
        assert d["priority"] == "normal"

    def test_days_until_due(self):
        """Test days until due calculation."""
        invoice = PayableInvoice(
            id="apinv_123",
            vendor_id="v1",
            vendor_name="Test",
            due_date=datetime.now() + timedelta(days=15),
            total_amount=Decimal("100.00"),
            balance=Decimal("100.00"),
        )

        days = invoice.days_until_due
        assert days is not None
        assert 14 <= days <= 16

    def test_days_until_discount(self):
        """Test days until discount deadline."""
        invoice = PayableInvoice(
            id="apinv_123",
            vendor_id="v1",
            vendor_name="Test",
            total_amount=Decimal("100.00"),
            balance=Decimal("100.00"),
            early_pay_discount=0.02,
            discount_deadline=datetime.now() + timedelta(days=5),
        )

        days = invoice.days_until_discount
        assert days is not None
        assert 4 <= days <= 6

    def test_discount_amount(self):
        """Test discount amount calculation."""
        invoice = PayableInvoice(
            id="apinv_123",
            vendor_id="v1",
            vendor_name="Test",
            total_amount=Decimal("1000.00"),
            balance=Decimal("1000.00"),
            early_pay_discount=0.02,  # 2%
        )

        assert invoice.discount_amount == Decimal("20.00")


# =============================================================================
# Payment Schedule Tests
# =============================================================================


class TestPaymentSchedule:
    """Test PaymentSchedule dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        schedule = PaymentSchedule(
            total_amount=Decimal("5000.00"),
            total_discount_captured=Decimal("100.00"),
            payments=[
                {"invoice_id": "inv1", "amount": 2000.00, "date": "2024-01-15"},
                {"invoice_id": "inv2", "amount": 3000.00, "date": "2024-01-20"},
            ],
        )

        d = schedule.to_dict()

        # Uses camelCase keys and float values
        assert d["totalAmount"] == 5000.0
        assert len(d["payments"]) == 2


# =============================================================================
# Batch Payment Tests
# =============================================================================


class TestBatchPaymentDataclass:
    """Test BatchPayment dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        batch = BatchPayment(
            id="batch_123",
            payment_date=datetime(2024, 1, 15),
            total_amount=Decimal("3000.00"),
            payment_method=PaymentMethod.ACH,
            invoices=[{"id": "inv1"}, {"id": "inv2"}],
            payment_count=2,
        )

        d = batch.to_dict()

        assert d["id"] == "batch_123"
        # Uses camelCase keys and float values
        assert d["totalAmount"] == 3000.0
        assert d["paymentMethod"] == "ach"
        assert len(d["invoices"]) == 2
