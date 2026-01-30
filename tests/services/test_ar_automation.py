"""Tests for the Accounts Receivable Automation service."""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from aragora.services.ar_automation import (
    AgingReport,
    ARAutomation,
    ARInvoice,
    CollectionAction,
    CollectionSuggestion,
    InvoiceStatus,
    ReminderLevel,
    REMINDER_TEMPLATES,
)


def _make_ar(**kw) -> ARAutomation:
    return ARAutomation(**kw)


def _make_invoice(ar: ARAutomation, **overrides) -> ARInvoice:
    defaults = dict(
        id="ar_test",
        customer_id="cust_1",
        customer_name="Test Customer",
        customer_email="test@example.com",
        total_amount=Decimal("1000.00"),
        balance=Decimal("1000.00"),
        due_date=datetime.now() + timedelta(days=30),
        status=InvoiceStatus.SENT,
    )
    defaults.update(overrides)
    inv = ARInvoice(**defaults)
    ar._store_invoice(inv)
    return inv


# ---------------------------------------------------------------------------
# Dataclass / enum tests
# ---------------------------------------------------------------------------


class TestARDataclasses:
    def test_invoice_status_values(self):
        assert InvoiceStatus.DRAFT.value == "draft"
        assert InvoiceStatus.PAID.value == "paid"
        assert InvoiceStatus.WRITTEN_OFF.value == "written_off"

    def test_reminder_level_values(self):
        assert ReminderLevel.FRIENDLY.value == "friendly"
        assert ReminderLevel.FINAL.value == "final"

    def test_invoice_not_overdue_when_due_in_future(self):
        inv = ARInvoice(
            id="t",
            customer_id="c",
            customer_name="C",
            due_date=datetime.now() + timedelta(days=10),
            balance=Decimal("100"),
        )
        assert inv.is_overdue is False
        assert inv.days_overdue == 0

    def test_invoice_overdue(self):
        inv = ARInvoice(
            id="t",
            customer_id="c",
            customer_name="C",
            due_date=datetime.now() - timedelta(days=15),
            balance=Decimal("100"),
        )
        assert inv.is_overdue is True
        assert inv.days_overdue >= 14

    def test_aging_bucket_current(self):
        inv = ARInvoice(
            id="t",
            customer_id="c",
            customer_name="C",
            due_date=datetime.now() + timedelta(days=5),
            balance=Decimal("100"),
        )
        assert inv.aging_bucket == "Current"

    def test_aging_bucket_1_30(self):
        inv = ARInvoice(
            id="t",
            customer_id="c",
            customer_name="C",
            due_date=datetime.now() - timedelta(days=15),
            balance=Decimal("100"),
        )
        assert inv.aging_bucket == "1-30"

    def test_aging_bucket_90_plus(self):
        inv = ARInvoice(
            id="t",
            customer_id="c",
            customer_name="C",
            due_date=datetime.now() - timedelta(days=100),
            balance=Decimal("100"),
        )
        assert inv.aging_bucket == "90+"

    def test_invoice_to_dict(self):
        inv = ARInvoice(
            id="ar_1",
            customer_id="c1",
            customer_name="Customer",
            total_amount=Decimal("1000"),
            balance=Decimal("500"),
        )
        d = inv.to_dict()
        assert d["id"] == "ar_1"
        assert d["totalAmount"] == 1000.0
        assert d["balance"] == 500.0

    def test_aging_report_to_dict(self):
        report = AgingReport(total_receivables=Decimal("5000"))
        d = report.to_dict()
        assert d["totalReceivables"] == 5000.0

    def test_collection_suggestion_to_dict(self):
        cs = CollectionSuggestion(
            invoice_id="ar_1",
            customer_id="c1",
            customer_name="Customer",
            balance=Decimal("500"),
            days_overdue=45,
            action=CollectionAction.PHONE_CALL,
            priority="high",
            reason="Overdue",
        )
        d = cs.to_dict()
        assert d["action"] == "phone_call"

    def test_reminder_templates_exist(self):
        assert ReminderLevel.FRIENDLY in REMINDER_TEMPLATES
        assert ReminderLevel.FINAL in REMINDER_TEMPLATES


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestARInit:
    def test_init_defaults(self):
        ar = _make_ar()
        assert ar.company_name == "Your Company"
        assert ar.default_payment_terms == "Net 30"

    def test_init_custom(self):
        ar = _make_ar(company_name="Aragora Inc", default_payment_terms="Net 15")
        assert ar.company_name == "Aragora Inc"


# ---------------------------------------------------------------------------
# generate_invoice
# ---------------------------------------------------------------------------


class TestGenerateInvoice:
    @pytest.mark.asyncio
    async def test_generate_basic_invoice(self):
        ar = _make_ar()
        inv = await ar.generate_invoice(
            customer_id="cust_1",
            line_items=[
                {"description": "Consulting", "quantity": 10, "unitPrice": 150, "amount": 1500}
            ],
            customer_name="Acme Corp",
        )
        assert inv.id.startswith("ar_")
        assert inv.customer_name == "Acme Corp"
        assert inv.subtotal == Decimal("1500")
        assert inv.total_amount > inv.subtotal  # tax added
        assert inv.status == InvoiceStatus.DRAFT

    @pytest.mark.asyncio
    async def test_generate_invoice_net30_due_date(self):
        ar = _make_ar()
        inv_date = datetime(2025, 6, 1)
        inv = await ar.generate_invoice(
            customer_id="c1",
            line_items=[{"description": "Work", "amount": 100}],
            customer_name="Client",
            invoice_date=inv_date,
            payment_terms="Net 30",
        )
        assert inv.due_date == inv_date + timedelta(days=30)

    @pytest.mark.asyncio
    async def test_generate_invoice_due_on_receipt(self):
        ar = _make_ar()
        inv_date = datetime(2025, 6, 1)
        inv = await ar.generate_invoice(
            customer_id="c1",
            line_items=[{"description": "Work", "amount": 100}],
            customer_name="Client",
            invoice_date=inv_date,
            payment_terms="Due on Receipt",
        )
        assert inv.due_date == inv_date

    @pytest.mark.asyncio
    async def test_generate_invoice_stored(self):
        ar = _make_ar()
        inv = await ar.generate_invoice(
            customer_id="c1",
            line_items=[{"description": "Work", "amount": 100}],
            customer_name="Client",
        )
        assert inv.id in ar._invoices


# ---------------------------------------------------------------------------
# send_invoice
# ---------------------------------------------------------------------------


class TestSendInvoice:
    @pytest.mark.asyncio
    async def test_send_invoice(self):
        ar = _make_ar()
        inv = _make_invoice(ar, status=InvoiceStatus.DRAFT)
        result = await ar.send_invoice(inv.id, send_email=False)
        assert result is True
        assert inv.status == InvoiceStatus.SENT

    @pytest.mark.asyncio
    async def test_send_invoice_no_email(self):
        ar = _make_ar()
        inv = _make_invoice(ar, customer_email=None)
        result = await ar.send_invoice(inv.id)
        assert result is False

    @pytest.mark.asyncio
    async def test_send_invoice_not_found(self):
        ar = _make_ar()
        result = await ar.send_invoice("nonexistent")
        assert result is False


# ---------------------------------------------------------------------------
# send_payment_reminder
# ---------------------------------------------------------------------------


class TestSendReminder:
    @pytest.mark.asyncio
    async def test_send_friendly_reminder(self):
        ar = _make_ar()
        inv = _make_invoice(ar, due_date=datetime.now() - timedelta(days=5))
        reminder = await ar.send_payment_reminder(inv.id, escalation_level=1, send_email=False)
        assert reminder is not None
        assert reminder["level"] == "friendly"
        assert inv.reminder_count == 1

    @pytest.mark.asyncio
    async def test_send_urgent_reminder(self):
        ar = _make_ar()
        inv = _make_invoice(ar, due_date=datetime.now() - timedelta(days=45))
        reminder = await ar.send_payment_reminder(inv.id, escalation_level=3, send_email=False)
        assert reminder["level"] == "urgent"

    @pytest.mark.asyncio
    async def test_auto_escalation(self):
        ar = _make_ar()
        inv = _make_invoice(ar, due_date=datetime.now() - timedelta(days=5), reminder_count=2)
        # Auto-escalation: count+1 = 3 => urgent
        reminder = await ar.send_payment_reminder(inv.id, send_email=False)
        assert reminder["level"] == "urgent"

    @pytest.mark.asyncio
    async def test_reminder_not_found(self):
        ar = _make_ar()
        result = await ar.send_payment_reminder("nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# record_payment
# ---------------------------------------------------------------------------


class TestARRecordPayment:
    @pytest.mark.asyncio
    async def test_record_partial_payment(self):
        ar = _make_ar()
        inv = _make_invoice(ar, total_amount=Decimal("1000"), balance=Decimal("1000"))
        result = await ar.record_payment(inv.id, amount=400.0)
        assert result.status == InvoiceStatus.PARTIAL
        assert result.balance == Decimal("600")

    @pytest.mark.asyncio
    async def test_record_full_payment(self):
        ar = _make_ar()
        inv = _make_invoice(ar, total_amount=Decimal("1000"), balance=Decimal("1000"))
        result = await ar.record_payment(inv.id, amount=1000.0)
        assert result.status == InvoiceStatus.PAID
        assert result.balance <= Decimal("0")

    @pytest.mark.asyncio
    async def test_record_payment_not_found(self):
        ar = _make_ar()
        result = await ar.record_payment("nonexistent", amount=100.0)
        assert result is None


# ---------------------------------------------------------------------------
# track_aging
# ---------------------------------------------------------------------------


class TestTrackAging:
    @pytest.mark.asyncio
    async def test_empty_aging_report(self):
        ar = _make_ar()
        report = await ar.track_aging()
        assert report.total_receivables == Decimal("0")
        assert report.invoice_count == 0

    @pytest.mark.asyncio
    async def test_aging_with_invoices(self):
        ar = _make_ar()
        _make_invoice(
            ar, id="inv_1", balance=Decimal("500"), due_date=datetime.now() + timedelta(days=10)
        )  # Current
        _make_invoice(
            ar, id="inv_2", balance=Decimal("300"), due_date=datetime.now() - timedelta(days=20)
        )  # 1-30
        report = await ar.track_aging()
        assert report.invoice_count == 2
        assert report.total_receivables == Decimal("800")
        assert report.customer_count == 1


# ---------------------------------------------------------------------------
# suggest_collections
# ---------------------------------------------------------------------------


class TestSuggestCollections:
    @pytest.mark.asyncio
    async def test_no_overdue_no_suggestions(self):
        ar = _make_ar()
        _make_invoice(ar, due_date=datetime.now() + timedelta(days=30))
        suggestions = await ar.suggest_collections()
        assert len(suggestions) == 0

    @pytest.mark.asyncio
    async def test_recently_overdue_sends_reminder(self):
        ar = _make_ar()
        _make_invoice(ar, due_date=datetime.now() - timedelta(days=3))
        suggestions = await ar.suggest_collections()
        assert len(suggestions) == 1
        assert suggestions[0].action == CollectionAction.SEND_REMINDER

    @pytest.mark.asyncio
    async def test_90_plus_large_balance_legal(self):
        ar = _make_ar()
        _make_invoice(ar, due_date=datetime.now() - timedelta(days=100), balance=Decimal("10000"))
        suggestions = await ar.suggest_collections()
        assert len(suggestions) == 1
        assert suggestions[0].action == CollectionAction.LEGAL_ACTION

    @pytest.mark.asyncio
    async def test_90_plus_small_balance_write_off(self):
        ar = _make_ar()
        _make_invoice(ar, due_date=datetime.now() - timedelta(days=100), balance=Decimal("50"))
        suggestions = await ar.suggest_collections()
        assert len(suggestions) == 1
        assert suggestions[0].action == CollectionAction.WRITE_OFF


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestARStats:
    def test_empty_stats(self):
        ar = _make_ar()
        stats = ar.get_stats()
        assert stats["totalInvoices"] == 0

    @pytest.mark.asyncio
    async def test_stats_with_invoices(self):
        ar = _make_ar()
        _make_invoice(ar, id="inv_1", balance=Decimal("500"))
        _make_invoice(ar, id="inv_2", balance=Decimal("300"))
        stats = ar.get_stats()
        assert stats["totalInvoices"] == 2
        assert stats["totalReceivables"] == 800.0


# ---------------------------------------------------------------------------
# Customer management
# ---------------------------------------------------------------------------


class TestCustomerManagement:
    @pytest.mark.asyncio
    async def test_add_customer(self):
        ar = _make_ar()
        await ar.add_customer("c1", "Test Customer", "test@example.com")
        assert "c1" in ar._customers
        assert ar._customers["c1"]["name"] == "Test Customer"

    @pytest.mark.asyncio
    async def test_get_customer_balance(self):
        ar = _make_ar()
        _make_invoice(ar, id="inv_1", customer_id="c1", balance=Decimal("500"))
        _make_invoice(ar, id="inv_2", customer_id="c1", balance=Decimal("300"))
        balance = await ar.get_customer_balance("c1")
        assert balance == Decimal("800")

    @pytest.mark.asyncio
    async def test_get_customer_balance_no_invoices(self):
        ar = _make_ar()
        balance = await ar.get_customer_balance("c_unknown")
        assert balance == Decimal("0")
