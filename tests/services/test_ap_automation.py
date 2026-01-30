"""Tests for the Accounts Payable Automation service."""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

from aragora.services.ap_automation import (
    APAutomation,
    BatchPayment,
    CashForecast,
    PayableInvoice,
    PaymentMethod,
    PaymentPriority,
    PaymentSchedule,
)


def _make_ap(**kw) -> APAutomation:
    kw.setdefault("enable_circuit_breakers", False)
    return APAutomation(**kw)


# ---------------------------------------------------------------------------
# Dataclass / enum tests
# ---------------------------------------------------------------------------


class TestAPDataclasses:
    def test_payment_priority_values(self):
        assert PaymentPriority.CRITICAL.value == "critical"
        assert PaymentPriority.HOLD.value == "hold"

    def test_payment_method_values(self):
        assert PaymentMethod.ACH.value == "ach"
        assert PaymentMethod.VIRTUAL_CARD.value == "virtual_card"

    def test_payable_invoice_discount_amount(self):
        inv = PayableInvoice(
            id="ap_1",
            vendor_id="v1",
            vendor_name="Acme",
            total_amount=Decimal("1000"),
            balance=Decimal("1000"),
            early_pay_discount=0.02,
        )
        assert inv.discount_amount == Decimal("20.00")

    def test_payable_invoice_no_discount(self):
        inv = PayableInvoice(
            id="ap_1",
            vendor_id="v1",
            vendor_name="Acme",
            total_amount=Decimal("1000"),
            balance=Decimal("1000"),
        )
        assert inv.discount_amount == Decimal("0")

    def test_payable_invoice_overdue(self):
        inv = PayableInvoice(
            id="ap_1",
            vendor_id="v1",
            vendor_name="Acme",
            total_amount=Decimal("1000"),
            balance=Decimal("1000"),
            due_date=datetime.now() - timedelta(days=5),
        )
        assert inv.is_overdue is True

    def test_payable_invoice_not_overdue(self):
        inv = PayableInvoice(
            id="ap_1",
            vendor_id="v1",
            vendor_name="Acme",
            total_amount=Decimal("1000"),
            balance=Decimal("1000"),
            due_date=datetime.now() + timedelta(days=5),
        )
        assert inv.is_overdue is False

    def test_payable_invoice_to_dict(self):
        inv = PayableInvoice(
            id="ap_1",
            vendor_id="v1",
            vendor_name="Acme",
            total_amount=Decimal("500"),
            balance=Decimal("500"),
        )
        d = inv.to_dict()
        assert d["id"] == "ap_1"
        assert d["vendorName"] == "Acme"
        assert d["totalAmount"] == 500.0

    def test_payment_schedule_to_dict(self):
        ps = PaymentSchedule(
            total_amount=Decimal("1000"),
            total_discount_captured=Decimal("20"),
        )
        d = ps.to_dict()
        assert d["totalAmount"] == 1000.0
        assert d["totalDiscountCaptured"] == 20.0

    def test_batch_payment_to_dict(self):
        bp = BatchPayment(
            id="batch_1",
            payment_date=datetime(2025, 6, 15),
            total_amount=Decimal("5000"),
            payment_count=3,
        )
        d = bp.to_dict()
        assert d["id"] == "batch_1"
        assert d["paymentCount"] == 3

    def test_cash_forecast_to_dict(self):
        cf = CashForecast(
            current_balance=Decimal("100000"),
            total_payables=Decimal("30000"),
        )
        d = cf.to_dict()
        assert d["currentBalance"] == 100000.0
        assert d["totalPayables"] == 30000.0


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestAPInit:
    def test_init_defaults(self):
        ap = _make_ap()
        assert ap.current_cash_balance == Decimal("100000")
        assert ap.min_cash_reserve == Decimal("20000")

    def test_init_custom_balance(self):
        ap = _make_ap(current_cash_balance=Decimal("50000"))
        assert ap.current_cash_balance == Decimal("50000")


# ---------------------------------------------------------------------------
# add_invoice
# ---------------------------------------------------------------------------


class TestAddInvoice:
    @pytest.mark.asyncio
    async def test_add_basic_invoice(self):
        ap = _make_ap()
        inv = await ap.add_invoice(
            vendor_id="v1",
            vendor_name="Acme Corp",
            total_amount=1000.0,
        )
        assert inv.vendor_name == "Acme Corp"
        assert inv.total_amount == Decimal("1000.0")
        assert inv.balance == Decimal("1000.0")
        assert inv.id.startswith("ap_")

    @pytest.mark.asyncio
    async def test_add_invoice_net30_due_date(self):
        ap = _make_ap()
        inv_date = datetime(2025, 6, 1)
        inv = await ap.add_invoice(
            vendor_id="v1",
            vendor_name="Acme",
            total_amount=1000.0,
            invoice_date=inv_date,
            payment_terms="Net 30",
        )
        assert inv.due_date == inv_date + timedelta(days=30)

    @pytest.mark.asyncio
    async def test_add_invoice_net15_due_date(self):
        ap = _make_ap()
        inv_date = datetime(2025, 6, 1)
        inv = await ap.add_invoice(
            vendor_id="v1",
            vendor_name="Acme",
            total_amount=1000.0,
            invoice_date=inv_date,
            payment_terms="Net 15",
        )
        assert inv.due_date == inv_date + timedelta(days=15)

    @pytest.mark.asyncio
    async def test_add_invoice_with_discount(self):
        ap = _make_ap()
        inv = await ap.add_invoice(
            vendor_id="v1",
            vendor_name="Acme",
            total_amount=1000.0,
            early_pay_discount=0.02,
            discount_days=10,
        )
        assert inv.early_pay_discount == 0.02
        assert inv.discount_deadline is not None

    @pytest.mark.asyncio
    async def test_add_invoice_stored(self):
        ap = _make_ap()
        inv = await ap.add_invoice(vendor_id="v1", vendor_name="X", total_amount=100.0)
        assert inv.id in ap._invoices


# ---------------------------------------------------------------------------
# optimize_payment_timing
# ---------------------------------------------------------------------------


class TestOptimizePaymentTiming:
    @pytest.mark.asyncio
    async def test_optimize_empty(self):
        ap = _make_ap()
        schedule = await ap.optimize_payment_timing([])
        assert schedule.total_amount == Decimal("0")
        assert schedule.payments == []

    @pytest.mark.asyncio
    async def test_optimize_critical_invoice(self):
        ap = _make_ap()
        inv = await ap.add_invoice(
            vendor_id="v1",
            vendor_name="Acme",
            total_amount=1000.0,
            priority=PaymentPriority.CRITICAL,
        )
        schedule = await ap.optimize_payment_timing([inv])
        assert len(schedule.payments) == 1
        assert any(
            "critical" in n.lower() or "immediately" in n.lower()
            for n in schedule.optimization_notes
        )

    @pytest.mark.asyncio
    async def test_optimize_overdue_invoice(self):
        ap = _make_ap()
        inv = await ap.add_invoice(
            vendor_id="v1",
            vendor_name="Acme",
            total_amount=1000.0,
            due_date=datetime.now() - timedelta(days=5),
        )
        schedule = await ap.optimize_payment_timing([inv])
        assert len(schedule.payments) == 1

    @pytest.mark.asyncio
    async def test_optimize_captures_discount(self):
        ap = _make_ap()
        inv_date = datetime.now()
        inv = await ap.add_invoice(
            vendor_id="v1",
            vendor_name="Acme",
            total_amount=10000.0,
            early_pay_discount=0.02,
            discount_days=10,
            invoice_date=inv_date,
            due_date=inv_date + timedelta(days=30),
        )
        schedule = await ap.optimize_payment_timing([inv])
        assert schedule.total_discount_captured >= Decimal("0")


# ---------------------------------------------------------------------------
# batch_payments
# ---------------------------------------------------------------------------


class TestBatchPayments:
    @pytest.mark.asyncio
    async def test_batch_empty(self):
        ap = _make_ap()
        batch = await ap.batch_payments([])
        assert batch.payment_count == 0
        assert batch.total_amount == Decimal("0")

    @pytest.mark.asyncio
    async def test_batch_groups_by_vendor(self):
        ap = _make_ap()
        inv1 = await ap.add_invoice(vendor_id="v1", vendor_name="Acme", total_amount=500.0)
        inv2 = await ap.add_invoice(vendor_id="v1", vendor_name="Acme", total_amount=300.0)
        inv3 = await ap.add_invoice(vendor_id="v2", vendor_name="Other", total_amount=200.0)
        batch = await ap.batch_payments([inv1, inv2, inv3])
        assert batch.payment_count == 3
        assert len(batch.invoices) == 2  # 2 vendors
        assert batch.total_amount == Decimal("1000.0")


# ---------------------------------------------------------------------------
# forecast_cash_needs
# ---------------------------------------------------------------------------


class TestForecastCashNeeds:
    @pytest.mark.asyncio
    async def test_forecast_no_invoices(self):
        ap = _make_ap()
        forecast = await ap.forecast_cash_needs(days_ahead=30)
        assert forecast.total_payables == Decimal("0")
        assert forecast.projected_balance == ap.current_cash_balance

    @pytest.mark.asyncio
    async def test_forecast_with_payables(self):
        ap = _make_ap()
        await ap.add_invoice(
            vendor_id="v1",
            vendor_name="Acme",
            total_amount=5000.0,
            due_date=datetime.now() + timedelta(days=10),
        )
        forecast = await ap.forecast_cash_needs(days_ahead=30)
        assert forecast.total_payables == Decimal("5000.0")

    @pytest.mark.asyncio
    async def test_forecast_with_receivables(self):
        ap = _make_ap()
        await ap.add_expected_receivable(
            expected_date=datetime.now() + timedelta(days=5),
            amount=10000.0,
        )
        forecast = await ap.forecast_cash_needs(days_ahead=30)
        assert forecast.total_receivables == Decimal("10000.0")

    @pytest.mark.asyncio
    async def test_forecast_cash_reserve_warning(self):
        ap = _make_ap(current_cash_balance=Decimal("25000"), min_cash_reserve=Decimal("20000"))
        await ap.add_invoice(
            vendor_id="v1",
            vendor_name="Acme",
            total_amount=10000.0,
            due_date=datetime.now() + timedelta(days=5),
        )
        forecast = await ap.forecast_cash_needs(days_ahead=30)
        assert len(forecast.warnings) >= 1


# ---------------------------------------------------------------------------
# record_payment
# ---------------------------------------------------------------------------


class TestRecordPayment:
    @pytest.mark.asyncio
    async def test_record_partial_payment(self):
        ap = _make_ap()
        inv = await ap.add_invoice(vendor_id="v1", vendor_name="Acme", total_amount=1000.0)
        result = await ap.record_payment(inv.id, amount=500.0)
        assert result is not None
        assert result.amount_paid == Decimal("500.0")
        assert result.balance == Decimal("500.0")
        assert result.paid_at is None

    @pytest.mark.asyncio
    async def test_record_full_payment(self):
        ap = _make_ap()
        inv = await ap.add_invoice(vendor_id="v1", vendor_name="Acme", total_amount=1000.0)
        result = await ap.record_payment(inv.id, amount=1000.0)
        assert result.balance == Decimal("0")
        assert result.paid_at is not None

    @pytest.mark.asyncio
    async def test_record_payment_nonexistent(self):
        ap = _make_ap()
        result = await ap.record_payment("nonexistent", amount=100.0)
        assert result is None


# ---------------------------------------------------------------------------
# list_invoices & filtering
# ---------------------------------------------------------------------------


class TestListInvoices:
    @pytest.mark.asyncio
    async def test_list_all_unpaid(self):
        ap = _make_ap()
        await ap.add_invoice(vendor_id="v1", vendor_name="A", total_amount=100.0)
        await ap.add_invoice(vendor_id="v2", vendor_name="B", total_amount=200.0)
        invoices = await ap.list_invoices()
        assert len(invoices) == 2

    @pytest.mark.asyncio
    async def test_filter_by_vendor(self):
        ap = _make_ap()
        await ap.add_invoice(vendor_id="v1", vendor_name="A", total_amount=100.0)
        await ap.add_invoice(vendor_id="v2", vendor_name="B", total_amount=200.0)
        invoices = await ap.list_invoices(vendor_id="v1")
        assert len(invoices) == 1

    @pytest.mark.asyncio
    async def test_filter_by_priority(self):
        ap = _make_ap()
        await ap.add_invoice(
            vendor_id="v1", vendor_name="A", total_amount=100.0, priority=PaymentPriority.CRITICAL
        )
        await ap.add_invoice(
            vendor_id="v2", vendor_name="B", total_amount=200.0, priority=PaymentPriority.LOW
        )
        invoices = await ap.list_invoices(priority=PaymentPriority.CRITICAL)
        assert len(invoices) == 1


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestAPStats:
    def test_empty_stats(self):
        ap = _make_ap()
        stats = ap.get_stats()
        assert stats["totalPayables"] == 0
        assert stats["invoiceCount"] == 0

    @pytest.mark.asyncio
    async def test_stats_with_invoices(self):
        ap = _make_ap()
        await ap.add_invoice(vendor_id="v1", vendor_name="A", total_amount=1000.0)
        await ap.add_invoice(vendor_id="v2", vendor_name="B", total_amount=2000.0)
        stats = ap.get_stats()
        assert stats["totalPayables"] == 3000.0
        assert stats["invoiceCount"] == 2
        assert stats["vendorCount"] == 2


# ---------------------------------------------------------------------------
# get_discount_opportunities
# ---------------------------------------------------------------------------


class TestDiscountOpportunities:
    @pytest.mark.asyncio
    async def test_no_opportunities(self):
        ap = _make_ap()
        await ap.add_invoice(vendor_id="v1", vendor_name="Acme", total_amount=1000.0)
        opportunities = await ap.get_discount_opportunities()
        assert len(opportunities) == 0

    @pytest.mark.asyncio
    async def test_with_discount_opportunity(self):
        ap = _make_ap()
        await ap.add_invoice(
            vendor_id="v1",
            vendor_name="Acme",
            total_amount=10000.0,
            early_pay_discount=0.02,
            discount_days=10,
        )
        opportunities = await ap.get_discount_opportunities()
        assert len(opportunities) == 1
        assert opportunities[0]["discountPercent"] == 2.0
