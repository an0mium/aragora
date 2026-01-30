"""Tests for the Invoice Processor service."""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

from aragora.services.invoice_processor import (
    Anomaly,
    AnomalyType,
    ApprovalLevel,
    APPROVAL_THRESHOLDS,
    InvoiceData,
    InvoiceLineItem,
    InvoiceProcessor,
    InvoiceStatus,
    PaymentSchedule,
    POMatch,
    PurchaseOrder,
)


def _make_processor(**kw) -> InvoiceProcessor:
    kw.setdefault("enable_circuit_breakers", False)
    kw.setdefault("enable_ocr", False)
    kw.setdefault("enable_llm_extraction", False)
    return InvoiceProcessor(**kw)


async def _make_manual_invoice(
    proc: InvoiceProcessor,
    vendor_name: str = "Acme Corp",
    total_amount: float = 1000.0,
    **kw,
) -> InvoiceData:
    return await proc.create_manual_invoice(
        vendor_name=vendor_name,
        total_amount=total_amount,
        **kw,
    )


# ---------------------------------------------------------------------------
# Dataclass / enum tests
# ---------------------------------------------------------------------------


class TestInvoiceProcessorDataclasses:
    def test_invoice_status_values(self):
        assert InvoiceStatus.RECEIVED.value == "received"
        assert InvoiceStatus.APPROVED.value == "approved"
        assert InvoiceStatus.DUPLICATE.value == "duplicate"

    def test_anomaly_type_values(self):
        assert AnomalyType.UNUSUAL_AMOUNT.value == "unusual_amount"
        assert AnomalyType.DUPLICATE.value == "duplicate"
        assert AnomalyType.NEW_VENDOR.value == "new_vendor"

    def test_approval_level_values(self):
        assert ApprovalLevel.AUTO.value == "auto"
        assert ApprovalLevel.EXECUTIVE.value == "executive"

    def test_line_item_to_dict(self):
        item = InvoiceLineItem(
            description="Consulting hours",
            quantity=10.0,
            unit_price=Decimal("150.00"),
            amount=Decimal("1500.00"),
        )
        d = item.to_dict()
        assert d["description"] == "Consulting hours"
        assert d["quantity"] == 10.0
        assert d["unitPrice"] == 150.0
        assert d["amount"] == 1500.0

    def test_invoice_data_hash_key(self):
        inv = InvoiceData(
            id="inv_1",
            vendor_name="Acme",
            invoice_number="INV-001",
            total_amount=Decimal("500"),
        )
        assert isinstance(inv.hash_key, str)
        assert len(inv.hash_key) == 32  # MD5 hex digest

    def test_invoice_data_hash_key_consistent(self):
        inv1 = InvoiceData(
            id="inv_1",
            vendor_name="Acme",
            invoice_number="INV-001",
            total_amount=Decimal("500"),
        )
        inv2 = InvoiceData(
            id="inv_2",
            vendor_name="Acme",
            invoice_number="INV-001",
            total_amount=Decimal("500"),
        )
        assert inv1.hash_key == inv2.hash_key

    def test_invoice_data_days_until_due(self):
        inv = InvoiceData(
            id="inv_1",
            vendor_name="Acme",
            due_date=datetime.now() + timedelta(days=15),
        )
        assert inv.days_until_due is not None
        assert inv.days_until_due >= 14

    def test_invoice_data_days_until_due_none(self):
        inv = InvoiceData(id="inv_1", vendor_name="Acme")
        assert inv.days_until_due is None

    def test_invoice_data_is_overdue(self):
        inv = InvoiceData(
            id="inv_1",
            vendor_name="Acme",
            due_date=datetime.now() - timedelta(days=5),
            status=InvoiceStatus.EXTRACTED,
        )
        assert inv.is_overdue is True

    def test_invoice_data_not_overdue_paid(self):
        inv = InvoiceData(
            id="inv_1",
            vendor_name="Acme",
            due_date=datetime.now() - timedelta(days=5),
            status=InvoiceStatus.PAID,
        )
        assert inv.is_overdue is False

    def test_invoice_data_to_dict(self):
        inv = InvoiceData(
            id="inv_1",
            vendor_name="Acme",
            total_amount=Decimal("1000"),
            status=InvoiceStatus.EXTRACTED,
        )
        d = inv.to_dict()
        assert d["id"] == "inv_1"
        assert d["vendorName"] == "Acme"
        assert d["totalAmount"] == 1000.0
        assert d["status"] == "extracted"

    def test_purchase_order_to_dict(self):
        po = PurchaseOrder(
            id="po_1",
            po_number="PO-001",
            vendor_id="v1",
            vendor_name="Acme",
            total_amount=Decimal("5000"),
            order_date=datetime(2025, 6, 1),
        )
        d = po.to_dict()
        assert d["poNumber"] == "PO-001"
        assert d["totalAmount"] == 5000.0

    def test_po_match_to_dict(self):
        match = POMatch(
            invoice_id="inv_1",
            po_id="po_1",
            match_type="exact",
            match_score=1.0,
        )
        d = match.to_dict()
        assert d["matchType"] == "exact"
        assert d["matchScore"] == 1.0

    def test_anomaly_to_dict(self):
        anomaly = Anomaly(
            type=AnomalyType.NEW_VENDOR,
            severity="medium",
            description="First invoice from vendor",
        )
        d = anomaly.to_dict()
        assert d["type"] == "new_vendor"
        assert d["severity"] == "medium"

    def test_payment_schedule_to_dict(self):
        sched = PaymentSchedule(
            invoice_id="inv_1",
            pay_date=datetime(2025, 7, 1),
            amount=Decimal("1000"),
            vendor_id="v1",
            vendor_name="Acme",
        )
        d = sched.to_dict()
        assert d["invoiceId"] == "inv_1"
        assert d["amount"] == 1000.0

    def test_approval_thresholds_exist(self):
        assert ApprovalLevel.AUTO in APPROVAL_THRESHOLDS
        assert ApprovalLevel.EXECUTIVE in APPROVAL_THRESHOLDS


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInvoiceProcessorInit:
    def test_init_defaults(self):
        proc = _make_processor()
        assert proc.auto_approve_threshold == Decimal("500")
        assert proc.enable_ocr is False
        assert len(proc._invoices) == 0

    def test_init_custom(self):
        proc = _make_processor(auto_approve_threshold=Decimal("1000"))
        assert proc.auto_approve_threshold == Decimal("1000")


# ---------------------------------------------------------------------------
# create_manual_invoice
# ---------------------------------------------------------------------------


class TestCreateManualInvoice:
    @pytest.mark.asyncio
    async def test_create_basic(self):
        proc = _make_processor()
        inv = await _make_manual_invoice(proc)
        assert inv.id.startswith("inv_")
        assert inv.vendor_name == "Acme Corp"
        assert inv.total_amount == Decimal("1000.0")
        assert inv.status == InvoiceStatus.EXTRACTED
        assert inv.confidence_score == 1.0

    @pytest.mark.asyncio
    async def test_create_with_line_items(self):
        proc = _make_processor()
        inv = await _make_manual_invoice(
            proc,
            line_items=[
                {"description": "Widget A", "quantity": 5, "unitPrice": 100, "amount": 500},
                {"description": "Widget B", "quantity": 3, "unitPrice": 100, "amount": 300},
            ],
        )
        assert len(inv.line_items) == 2
        assert inv.subtotal == Decimal("800")

    @pytest.mark.asyncio
    async def test_create_stored(self):
        proc = _make_processor()
        inv = await _make_manual_invoice(proc)
        assert inv.id in proc._invoices

    @pytest.mark.asyncio
    async def test_create_with_po_number(self):
        proc = _make_processor()
        inv = await _make_manual_invoice(proc, po_number="PO-123")
        assert inv.po_number == "PO-123"

    @pytest.mark.asyncio
    async def test_approval_level_auto(self):
        proc = _make_processor()
        inv = await _make_manual_invoice(proc, total_amount=300.0)
        assert inv.approval_level == ApprovalLevel.AUTO

    @pytest.mark.asyncio
    async def test_approval_level_manager(self):
        proc = _make_processor()
        inv = await _make_manual_invoice(proc, total_amount=3000.0)
        assert inv.approval_level == ApprovalLevel.MANAGER

    @pytest.mark.asyncio
    async def test_approval_level_director(self):
        proc = _make_processor()
        inv = await _make_manual_invoice(proc, total_amount=15000.0)
        assert inv.approval_level == ApprovalLevel.DIRECTOR

    @pytest.mark.asyncio
    async def test_approval_level_executive(self):
        proc = _make_processor()
        inv = await _make_manual_invoice(proc, total_amount=50000.0)
        assert inv.approval_level == ApprovalLevel.EXECUTIVE


# ---------------------------------------------------------------------------
# match_to_po
# ---------------------------------------------------------------------------


class TestMatchToPO:
    @pytest.mark.asyncio
    async def test_no_po_found(self):
        proc = _make_processor()
        inv = await _make_manual_invoice(proc, vendor_name="NoMatch")
        match = await proc.match_to_po(inv)
        assert match.match_type == "none"
        assert len(match.issues) > 0
        assert inv.status == InvoiceStatus.UNMATCHED

    @pytest.mark.asyncio
    async def test_match_by_po_number(self):
        proc = _make_processor()
        po = await proc.add_purchase_order(
            po_number="PO-100",
            vendor_name="Acme",
            total_amount=1000.0,
        )
        inv = await _make_manual_invoice(
            proc,
            vendor_name="Acme",
            total_amount=1000.0,
        )
        inv.po_number = "PO-100"
        match = await proc.match_to_po(inv)
        assert match.match_type == "exact"
        assert match.match_score == 1.0
        assert inv.status == InvoiceStatus.MATCHED

    @pytest.mark.asyncio
    async def test_match_by_vendor_and_amount(self):
        proc = _make_processor()
        await proc.add_purchase_order(
            po_number="PO-200",
            vendor_name="Acme Corp",
            total_amount=1000.0,
        )
        inv = await _make_manual_invoice(
            proc,
            vendor_name="Acme Corp",
            total_amount=1020.0,  # ~2% variance
        )
        match = await proc.match_to_po(inv)
        assert match.match_type == "partial"
        assert match.match_score >= 0.7

    @pytest.mark.asyncio
    async def test_match_large_variance(self):
        proc = _make_processor()
        await proc.add_purchase_order(
            po_number="PO-300",
            vendor_name="Widgets Inc",
            total_amount=1000.0,
        )
        inv = await _make_manual_invoice(
            proc,
            vendor_name="Widgets Inc",
            total_amount=1090.0,  # ~9% variance
        )
        match = await proc.match_to_po(inv)
        assert match.match_type == "partial"
        assert match.match_score <= 0.7


# ---------------------------------------------------------------------------
# detect_anomalies
# ---------------------------------------------------------------------------


class TestDetectAnomalies:
    @pytest.mark.asyncio
    async def test_new_vendor_anomaly(self):
        proc = _make_processor()
        inv = await _make_manual_invoice(proc, vendor_name="BrandNewVendor")
        anomalies = await proc.detect_anomalies(inv)
        types = [a.type for a in anomalies]
        assert AnomalyType.NEW_VENDOR in types

    @pytest.mark.asyncio
    async def test_round_amount_anomaly(self):
        proc = _make_processor()
        inv = await _make_manual_invoice(proc, total_amount=5000.0)
        anomalies = await proc.detect_anomalies(inv)
        types = [a.type for a in anomalies]
        assert AnomalyType.ROUND_AMOUNT in types

    @pytest.mark.asyncio
    async def test_high_value_anomaly(self):
        proc = _make_processor()
        inv = await _make_manual_invoice(proc, total_amount=15000.0)
        anomalies = await proc.detect_anomalies(inv)
        types = [a.type for a in anomalies]
        assert AnomalyType.HIGH_VALUE in types

    @pytest.mark.asyncio
    async def test_unusual_amount_high(self):
        proc = _make_processor()
        vendor = "historicvendor"
        proc._known_vendors.add(vendor)
        proc._vendor_history[vendor] = [100.0, 120.0, 110.0]
        inv = await _make_manual_invoice(
            proc,
            vendor_name="HistoricVendor",
            total_amount=500.0,  # >3x average ~110
        )
        anomalies = await proc.detect_anomalies(inv)
        types = [a.type for a in anomalies]
        assert AnomalyType.UNUSUAL_AMOUNT in types

    @pytest.mark.asyncio
    async def test_duplicate_anomaly(self):
        proc = _make_processor()
        inv1 = await _make_manual_invoice(
            proc,
            vendor_name="Acme",
            total_amount=500.0,
            invoice_number="INV-001",
        )
        # Manually create a second invoice with the same hash but different ID
        # The _store_invoice overwrites the hash index, so we restore inv1's entry
        inv2 = await _make_manual_invoice(
            proc,
            vendor_name="Acme",
            total_amount=500.0,
            invoice_number="INV-001",
        )
        # Restore hash index to point to inv1, simulating pre-existing invoice
        proc._hash_index[inv2.hash_key] = inv1.id
        anomalies = await proc.detect_anomalies(inv2)
        types = [a.type for a in anomalies]
        assert AnomalyType.DUPLICATE in types

    @pytest.mark.asyncio
    async def test_missing_po_anomaly(self):
        proc = _make_processor()
        inv = await _make_manual_invoice(proc)
        inv.status = InvoiceStatus.UNMATCHED
        anomalies = await proc.detect_anomalies(inv)
        types = [a.type for a in anomalies]
        assert AnomalyType.MISSING_PO in types


# ---------------------------------------------------------------------------
# schedule_payment
# ---------------------------------------------------------------------------


class TestSchedulePayment:
    @pytest.mark.asyncio
    async def test_schedule_approved_invoice(self):
        proc = _make_processor()
        inv = await _make_manual_invoice(proc)
        inv.status = InvoiceStatus.APPROVED
        pay_date = datetime.now() + timedelta(days=7)
        schedule = await proc.schedule_payment(inv, pay_date=pay_date)
        assert schedule.invoice_id == inv.id
        assert schedule.amount == inv.total_amount
        assert inv.status == InvoiceStatus.SCHEDULED

    @pytest.mark.asyncio
    async def test_schedule_matched_invoice(self):
        proc = _make_processor()
        inv = await _make_manual_invoice(proc)
        inv.status = InvoiceStatus.MATCHED
        schedule = await proc.schedule_payment(inv)
        assert schedule is not None

    @pytest.mark.asyncio
    async def test_schedule_unapproved_raises(self):
        proc = _make_processor()
        inv = await _make_manual_invoice(proc)
        # Status is EXTRACTED (not approved)
        with pytest.raises(ValueError, match="must be approved"):
            await proc.schedule_payment(inv)


# ---------------------------------------------------------------------------
# approve / reject invoice
# ---------------------------------------------------------------------------


class TestApproveReject:
    @pytest.mark.asyncio
    async def test_approve_invoice(self):
        proc = _make_processor()
        inv = await _make_manual_invoice(proc)
        result = await proc.approve_invoice(inv.id, approver_id="user_1")
        assert result.status == InvoiceStatus.APPROVED
        assert result.approver_id == "user_1"
        assert result.approved_at is not None

    @pytest.mark.asyncio
    async def test_approve_updates_vendor_history(self):
        proc = _make_processor()
        inv = await _make_manual_invoice(proc, vendor_name="TestVendor", total_amount=500.0)
        await proc.approve_invoice(inv.id, approver_id="u1")
        assert "testvendor" in proc._known_vendors
        assert 500.0 in proc._vendor_history["testvendor"]

    @pytest.mark.asyncio
    async def test_approve_not_found(self):
        proc = _make_processor()
        result = await proc.approve_invoice("nonexistent", approver_id="u1")
        assert result is None

    @pytest.mark.asyncio
    async def test_reject_invoice(self):
        proc = _make_processor()
        inv = await _make_manual_invoice(proc)
        result = await proc.reject_invoice(inv.id, reason="Invalid")
        assert result.status == InvoiceStatus.REJECTED
        assert result.notes == "Invalid"

    @pytest.mark.asyncio
    async def test_reject_not_found(self):
        proc = _make_processor()
        result = await proc.reject_invoice("nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# get_invoice / list_invoices
# ---------------------------------------------------------------------------


class TestListInvoices:
    @pytest.mark.asyncio
    async def test_get_invoice(self):
        proc = _make_processor()
        inv = await _make_manual_invoice(proc)
        result = await proc.get_invoice(inv.id)
        assert result is not None
        assert result.id == inv.id

    @pytest.mark.asyncio
    async def test_get_invoice_not_found(self):
        proc = _make_processor()
        result = await proc.get_invoice("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_all(self):
        proc = _make_processor()
        await _make_manual_invoice(proc, vendor_name="A")
        await _make_manual_invoice(proc, vendor_name="B")
        invoices, total = await proc.list_invoices()
        assert total == 2
        assert len(invoices) == 2

    @pytest.mark.asyncio
    async def test_list_by_vendor(self):
        proc = _make_processor()
        await _make_manual_invoice(proc, vendor_name="Acme Corp")
        await _make_manual_invoice(proc, vendor_name="Other Inc")
        invoices, total = await proc.list_invoices(vendor="Acme")
        assert total == 1

    @pytest.mark.asyncio
    async def test_list_by_status(self):
        proc = _make_processor()
        inv = await _make_manual_invoice(proc)
        await proc.approve_invoice(inv.id, "u1")
        await _make_manual_invoice(proc)
        invoices, total = await proc.list_invoices(status=InvoiceStatus.APPROVED)
        assert total == 1

    @pytest.mark.asyncio
    async def test_list_pagination(self):
        proc = _make_processor()
        for i in range(5):
            await _make_manual_invoice(proc, vendor_name=f"Vendor_{i}")
        invoices, total = await proc.list_invoices(limit=2, offset=0)
        assert total == 5
        assert len(invoices) == 2


# ---------------------------------------------------------------------------
# Pending approvals / scheduled payments / overdue
# ---------------------------------------------------------------------------


class TestQueryHelpers:
    @pytest.mark.asyncio
    async def test_pending_approvals(self):
        proc = _make_processor()
        inv = await _make_manual_invoice(proc)
        # match it to make it pending_approval-like
        inv.status = InvoiceStatus.MATCHED
        pending = await proc.get_pending_approvals()
        assert len(pending) == 1

    @pytest.mark.asyncio
    async def test_scheduled_payments(self):
        proc = _make_processor()
        inv = await _make_manual_invoice(proc)
        inv.status = InvoiceStatus.APPROVED
        await proc.schedule_payment(inv)
        payments = await proc.get_scheduled_payments()
        assert len(payments) == 1

    @pytest.mark.asyncio
    async def test_overdue_invoices(self):
        proc = _make_processor()
        inv = await _make_manual_invoice(proc)
        inv.due_date = datetime.now() - timedelta(days=5)
        overdue = await proc.get_overdue_invoices()
        assert len(overdue) == 1


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestInvoiceProcessorStats:
    def test_empty_stats(self):
        proc = _make_processor()
        stats = proc.get_stats()
        assert stats["totalInvoices"] == 0

    @pytest.mark.asyncio
    async def test_stats_with_invoices(self):
        proc = _make_processor()
        await _make_manual_invoice(proc, total_amount=500.0)
        await _make_manual_invoice(proc, total_amount=300.0)
        stats = proc.get_stats()
        assert stats["totalInvoices"] == 2
        assert stats["totalAmount"] == 800.0


# ---------------------------------------------------------------------------
# parse_invoice_text
# ---------------------------------------------------------------------------


class TestParseInvoiceText:
    def test_empty_text(self):
        proc = _make_processor()
        result = proc._parse_invoice_text("")
        assert result["vendor"] == ""
        assert result["total"] == 0.0

    def test_finds_invoice_number(self):
        proc = _make_processor()
        text = "Acme Corp\nInvoice #INV-12345\nTotal: $1,500.00"
        result = proc._parse_invoice_text(text)
        assert result["invoice_number"] == "INV-12345"

    def test_finds_subtotal(self):
        proc = _make_processor()
        text = "Items listed\nSubtotal: $1,000.00\nTax: $80.00"
        result = proc._parse_invoice_text(text)
        assert result["subtotal"] == 1000.0
        assert result["tax"] == 80.0

    def test_finds_amount_due(self):
        proc = _make_processor()
        text = "Invoice details\nAmount Due: $2,500.00"
        result = proc._parse_invoice_text(text)
        assert result["total"] == 2500.0

    def test_finds_po_number(self):
        proc = _make_processor()
        text = "Purchase Order #PO-5678\nTotal: $500.00"
        result = proc._parse_invoice_text(text)
        assert result["po_number"] == "PO-5678"
