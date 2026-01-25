"""
Tests for the invoice processing API handler.

Tests cover:
- Invoice upload and extraction
- Invoice CRUD operations (create, list, get)
- Approval workflow (approve, reject, pending)
- PO matching
- Anomaly detection
- Payment scheduling
- Purchase orders
- Statistics and reporting
- InvoiceHandler routing
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from enum import Enum

import pytest

from aragora.server.handlers.invoices import (
    InvoiceHandler,
    handle_upload_invoice,
    handle_create_invoice,
    handle_list_invoices,
    handle_get_invoice,
    handle_approve_invoice,
    handle_reject_invoice,
    handle_get_pending_approvals,
    handle_match_to_po,
    handle_get_anomalies,
    handle_schedule_payment,
    handle_get_scheduled_payments,
    handle_create_purchase_order,
    handle_get_invoice_stats,
    handle_get_overdue_invoices,
)


def parse_result(result):
    """Parse HandlerResult into (data, status_code) tuple.

    Returns the inner 'data' field if present (standard API response envelope),
    otherwise returns the full response body.
    """
    body = json.loads(result.body.decode("utf-8"))
    # Unwrap standard API envelope if present
    if "data" in body and "success" in body:
        return body["data"], result.status_code
    return body, result.status_code


# =============================================================================
# Mock Data Classes
# =============================================================================


class MockInvoiceStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    PAID = "paid"


class MockAnomalySeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class MockLineItem:
    """Mock line item."""

    description: str = "Test Item"
    quantity: float = 1.0
    unit_price: float = 100.00
    total: float = 100.00

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "quantity": self.quantity,
            "unit_price": self.unit_price,
            "total": self.total,
        }


@dataclass
class MockInvoice:
    """Mock invoice object."""

    id: str = "inv_123"
    vendor_name: str = "Test Vendor"
    total_amount: float = 1000.00
    invoice_number: str = "INV-001"
    status: MockInvoiceStatus = MockInvoiceStatus.PENDING
    invoice_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    due_date: Optional[datetime] = None
    po_number: Optional[str] = None
    line_items: List[MockLineItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "vendor_name": self.vendor_name,
            "total_amount": self.total_amount,
            "invoice_number": self.invoice_number,
            "status": self.status.value,
            "invoice_date": self.invoice_date.isoformat(),
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "po_number": self.po_number,
            "line_items": [li.to_dict() for li in self.line_items],
        }


@dataclass
class MockAnomaly:
    """Mock anomaly detection result."""

    id: str = "anom_123"
    description: str = "Duplicate invoice detected"
    severity: MockAnomalySeverity = MockAnomalySeverity.MEDIUM

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "severity": self.severity.value,
        }


@dataclass
class MockPOMatch:
    """Mock PO match result."""

    matched: bool = True
    po_number: str = "PO-001"
    confidence: float = 0.95
    variance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched": self.matched,
            "po_number": self.po_number,
            "confidence": self.confidence,
            "variance": self.variance,
        }


@dataclass
class MockPaymentSchedule:
    """Mock payment schedule."""

    id: str = "sched_123"
    invoice_id: str = "inv_123"
    pay_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    amount: float = 1000.00
    payment_method: str = "ach"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "invoice_id": self.invoice_id,
            "pay_date": self.pay_date.isoformat(),
            "amount": self.amount,
            "payment_method": self.payment_method,
        }


@dataclass
class MockPurchaseOrder:
    """Mock purchase order."""

    id: str = "po_123"
    po_number: str = "PO-001"
    vendor_name: str = "Test Vendor"
    total_amount: float = 1000.00

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "po_number": self.po_number,
            "vendor_name": self.vendor_name,
            "total_amount": self.total_amount,
        }


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_processor():
    """Create mock invoice processor."""
    processor = MagicMock()

    # Upload and extraction
    processor.extract_invoice_data = AsyncMock(return_value=MockInvoice())

    # CRUD
    processor.create_manual_invoice = AsyncMock(return_value=MockInvoice())
    processor.list_invoices = AsyncMock(return_value=([MockInvoice()], 1))
    processor.get_invoice = AsyncMock(return_value=MockInvoice())

    # Approval
    processor.approve_invoice = AsyncMock(return_value=MockInvoice())
    processor.reject_invoice = AsyncMock(return_value=MockInvoice())
    processor.get_pending_approvals = AsyncMock(return_value=[MockInvoice()])

    # Anomaly detection
    processor.detect_anomalies = AsyncMock(return_value=[MockAnomaly()])

    # PO matching
    processor.match_to_po = AsyncMock(return_value=MockPOMatch())

    # Payment scheduling
    processor.schedule_payment = AsyncMock(return_value=MockPaymentSchedule())
    processor.get_scheduled_payments = AsyncMock(return_value=[MockPaymentSchedule()])

    # Purchase orders
    processor.add_purchase_order = AsyncMock(return_value=MockPurchaseOrder())

    # Statistics
    processor.get_stats = MagicMock(
        return_value={
            "total_invoices": 100,
            "total_amount": 50000.00,
            "pending_count": 10,
            "approved_count": 80,
        }
    )
    processor.get_overdue_invoices = AsyncMock(return_value=[MockInvoice()])

    return processor


@pytest.fixture
def handler():
    """Create InvoiceHandler instance."""
    mock_context = {"storage": None}
    return InvoiceHandler(mock_context)


# =============================================================================
# Invoice Upload Tests
# =============================================================================


class TestInvoiceUpload:
    """Tests for invoice upload endpoint."""

    @pytest.mark.asyncio
    async def test_upload_invoice_success(self, mock_processor):
        """Successfully upload and extract invoice."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            # Valid base64 encoded PDF
            pdf_data = b"fake PDF data"
            b64_data = base64.b64encode(pdf_data).decode()

            result = await handle_upload_invoice({"document_data": b64_data})
            data, status = parse_result(result)

            assert status == 200
            assert "invoice" in data
            assert "anomalies" in data
            mock_processor.extract_invoice_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_invoice_missing_document(self, mock_processor):
        """Reject upload without document data."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_upload_invoice({})
            _, status = parse_result(result)

            assert status == 400

    @pytest.mark.asyncio
    async def test_upload_invoice_invalid_base64(self, mock_processor):
        """Reject invalid base64 document."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_upload_invoice({"document_data": "not-valid-base64!!!"})
            _, status = parse_result(result)

            assert status == 400

    @pytest.mark.asyncio
    async def test_upload_invoice_with_vendor_hint(self, mock_processor):
        """Upload invoice with vendor hint."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            pdf_data = b"fake PDF data"
            b64_data = base64.b64encode(pdf_data).decode()

            result = await handle_upload_invoice(
                {"document_data": b64_data, "vendor_hint": "Acme Corp"}
            )
            data, status = parse_result(result)

            assert status == 200
            mock_processor.extract_invoice_data.assert_called_once()
            call_kwargs = mock_processor.extract_invoice_data.call_args[1]
            assert call_kwargs["vendor_hint"] == "Acme Corp"


# =============================================================================
# Invoice CRUD Tests
# =============================================================================


class TestInvoiceCreate:
    """Tests for invoice creation."""

    @pytest.mark.asyncio
    async def test_create_invoice_success(self, mock_processor):
        """Successfully create invoice."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_create_invoice(
                {
                    "vendor_name": "Test Vendor",
                    "total_amount": 1500.00,
                    "invoice_number": "INV-002",
                }
            )
            data, status = parse_result(result)

            assert status == 200
            assert "invoice" in data
            mock_processor.create_manual_invoice.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_invoice_missing_vendor(self, mock_processor):
        """Reject creation without vendor."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_create_invoice({"total_amount": 1000.00})
            data, status = parse_result(result)

            assert status == 400
            assert "vendor_name" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_create_invoice_missing_amount(self, mock_processor):
        """Reject creation without amount."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_create_invoice({"vendor_name": "Test"})
            data, status = parse_result(result)

            assert status == 400
            assert "total_amount" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_create_invoice_invalid_amount(self, mock_processor):
        """Reject invalid amount."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_create_invoice(
                {"vendor_name": "Test", "total_amount": "not-a-number"}
            )
            _, status = parse_result(result)

            assert status == 400

    @pytest.mark.asyncio
    async def test_create_invoice_invalid_date(self, mock_processor):
        """Reject invalid date format."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_create_invoice(
                {
                    "vendor_name": "Test",
                    "total_amount": 1000.00,
                    "invoice_date": "not-a-date",
                }
            )
            _, status = parse_result(result)

            assert status == 400

    @pytest.mark.asyncio
    async def test_create_invoice_invalid_due_date(self, mock_processor):
        """Reject invalid due date format."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_create_invoice(
                {
                    "vendor_name": "Test",
                    "total_amount": 1000.00,
                    "due_date": "not-a-date",
                }
            )
            _, status = parse_result(result)

            assert status == 400


class TestInvoiceList:
    """Tests for invoice listing."""

    @pytest.mark.asyncio
    async def test_list_invoices_success(self, mock_processor):
        """Successfully list invoices."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_list_invoices({})
            data, status = parse_result(result)

            assert status == 200
            assert "invoices" in data
            assert "total" in data
            assert len(data["invoices"]) == 1

    @pytest.mark.asyncio
    async def test_list_invoices_with_filters(self, mock_processor):
        """List invoices with filters."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_list_invoices(
                {
                    "status": "pending",
                    "vendor": "Test Vendor",
                    "start_date": "2026-01-01",
                    "end_date": "2026-01-31",
                    "limit": "50",
                    "offset": "10",
                }
            )
            _, status = parse_result(result)

            assert status == 200


class TestInvoiceGet:
    """Tests for getting single invoice."""

    @pytest.mark.asyncio
    async def test_get_invoice_success(self, mock_processor):
        """Successfully get invoice by ID."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_get_invoice("inv_123")
            data, status = parse_result(result)

            assert status == 200
            assert "invoice" in data

    @pytest.mark.asyncio
    async def test_get_invoice_not_found(self, mock_processor):
        """Return 404 for missing invoice."""
        mock_processor.get_invoice = AsyncMock(return_value=None)

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_get_invoice("nonexistent")
            _, status = parse_result(result)

            assert status == 404


# =============================================================================
# Approval Workflow Tests
# =============================================================================


class TestApprovalWorkflow:
    """Tests for approval workflow."""

    @pytest.mark.asyncio
    async def test_approve_invoice_success(self, mock_processor):
        """Successfully approve invoice."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_approve_invoice("inv_123", {})
            data, status = parse_result(result)

            assert status == 200
            assert "invoice" in data
            mock_processor.approve_invoice.assert_called_once()

    @pytest.mark.asyncio
    async def test_approve_invoice_with_approver(self, mock_processor):
        """Approve invoice with specific approver."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_approve_invoice("inv_123", {"approver_id": "user_456"})
            _, status = parse_result(result)

            assert status == 200
            mock_processor.approve_invoice.assert_called_once_with("inv_123", "user_456")

    @pytest.mark.asyncio
    async def test_approve_invoice_not_found(self, mock_processor):
        """Return 404 for missing invoice."""
        mock_processor.approve_invoice = AsyncMock(return_value=None)

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_approve_invoice("nonexistent", {})
            _, status = parse_result(result)

            assert status == 404

    @pytest.mark.asyncio
    async def test_reject_invoice_success(self, mock_processor):
        """Successfully reject invoice."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_reject_invoice("inv_123", {"reason": "Duplicate"})
            data, status = parse_result(result)

            assert status == 200
            assert "invoice" in data
            mock_processor.reject_invoice.assert_called_once_with("inv_123", "Duplicate")

    @pytest.mark.asyncio
    async def test_reject_invoice_not_found(self, mock_processor):
        """Return 404 for missing invoice."""
        mock_processor.reject_invoice = AsyncMock(return_value=None)

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_reject_invoice("nonexistent", {})
            _, status = parse_result(result)

            assert status == 404

    @pytest.mark.asyncio
    async def test_get_pending_approvals(self, mock_processor):
        """Get pending approval list."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_get_pending_approvals()
            data, status = parse_result(result)

            assert status == 200
            assert "invoices" in data
            assert "count" in data


# =============================================================================
# PO Matching Tests
# =============================================================================


class TestPOMatching:
    """Tests for PO matching."""

    @pytest.mark.asyncio
    async def test_match_to_po_success(self, mock_processor):
        """Successfully match invoice to PO."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_match_to_po("inv_123")
            data, status = parse_result(result)

            assert status == 200
            assert "match" in data
            assert "invoice" in data
            mock_processor.match_to_po.assert_called_once()

    @pytest.mark.asyncio
    async def test_match_to_po_not_found(self, mock_processor):
        """Return 404 for missing invoice."""
        mock_processor.get_invoice = AsyncMock(return_value=None)

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_match_to_po("nonexistent")
            _, status = parse_result(result)

            assert status == 404


# =============================================================================
# Anomaly Detection Tests
# =============================================================================


class TestAnomalyDetection:
    """Tests for anomaly detection."""

    @pytest.mark.asyncio
    async def test_get_anomalies_success(self, mock_processor):
        """Successfully get anomalies."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_get_anomalies("inv_123")
            data, status = parse_result(result)

            assert status == 200
            assert "anomalies" in data
            assert "count" in data
            mock_processor.detect_anomalies.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_anomalies_not_found(self, mock_processor):
        """Return 404 for missing invoice."""
        mock_processor.get_invoice = AsyncMock(return_value=None)

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_get_anomalies("nonexistent")
            _, status = parse_result(result)

            assert status == 404


# =============================================================================
# Payment Scheduling Tests
# =============================================================================


class TestPaymentScheduling:
    """Tests for payment scheduling."""

    @pytest.mark.asyncio
    async def test_schedule_payment_success(self, mock_processor):
        """Successfully schedule payment."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_schedule_payment("inv_123", {})
            data, status = parse_result(result)

            assert status == 200
            assert "schedule" in data
            assert "invoice" in data
            mock_processor.schedule_payment.assert_called_once()

    @pytest.mark.asyncio
    async def test_schedule_payment_with_date(self, mock_processor):
        """Schedule payment with specific date."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_schedule_payment(
                "inv_123",
                {"pay_date": "2026-02-15T00:00:00Z", "payment_method": "wire"},
            )
            _, status = parse_result(result)

            assert status == 200

    @pytest.mark.asyncio
    async def test_schedule_payment_invalid_date(self, mock_processor):
        """Reject invalid pay date."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_schedule_payment("inv_123", {"pay_date": "not-a-date"})
            _, status = parse_result(result)

            assert status == 400

    @pytest.mark.asyncio
    async def test_schedule_payment_not_found(self, mock_processor):
        """Return 404 for missing invoice."""
        mock_processor.get_invoice = AsyncMock(return_value=None)

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_schedule_payment("nonexistent", {})
            _, status = parse_result(result)

            assert status == 404

    @pytest.mark.asyncio
    async def test_get_scheduled_payments(self, mock_processor):
        """Get scheduled payments."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_get_scheduled_payments({})
            data, status = parse_result(result)

            assert status == 200
            assert "payments" in data
            assert "count" in data
            assert "totalAmount" in data

    @pytest.mark.asyncio
    async def test_get_scheduled_payments_with_dates(self, mock_processor):
        """Get scheduled payments with date range."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_get_scheduled_payments(
                {"start_date": "2026-01-01", "end_date": "2026-01-31"}
            )
            _, status = parse_result(result)

            assert status == 200


# =============================================================================
# Purchase Order Tests
# =============================================================================


class TestPurchaseOrders:
    """Tests for purchase orders."""

    @pytest.mark.asyncio
    async def test_create_purchase_order_success(self, mock_processor):
        """Successfully create purchase order."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_create_purchase_order(
                {
                    "po_number": "PO-002",
                    "vendor_name": "Test Vendor",
                    "total_amount": 2000.00,
                }
            )
            data, status = parse_result(result)

            assert status == 200
            assert "purchaseOrder" in data
            mock_processor.add_purchase_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_purchase_order_missing_po_number(self, mock_processor):
        """Reject creation without PO number."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_create_purchase_order(
                {"vendor_name": "Test", "total_amount": 1000.00}
            )
            data, status = parse_result(result)

            assert status == 400
            assert "po_number" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_create_purchase_order_missing_vendor(self, mock_processor):
        """Reject creation without vendor."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_create_purchase_order(
                {"po_number": "PO-002", "total_amount": 1000.00}
            )
            data, status = parse_result(result)

            assert status == 400
            assert "vendor_name" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_create_purchase_order_missing_amount(self, mock_processor):
        """Reject creation without amount."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_create_purchase_order(
                {"po_number": "PO-002", "vendor_name": "Test"}
            )
            data, status = parse_result(result)

            assert status == 400
            assert "total_amount" in data.get("error", "")


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Tests for statistics and reporting."""

    @pytest.mark.asyncio
    async def test_get_invoice_stats(self, mock_processor):
        """Get invoice statistics."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_get_invoice_stats()
            data, status = parse_result(result)

            assert status == 200
            assert "stats" in data

    @pytest.mark.asyncio
    async def test_get_overdue_invoices(self, mock_processor):
        """Get overdue invoices."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handle_get_overdue_invoices()
            data, status = parse_result(result)

            assert status == 200
            assert "invoices" in data
            assert "count" in data
            assert "totalAmount" in data


# =============================================================================
# InvoiceHandler Routing Tests
# =============================================================================


class TestInvoiceHandlerRouting:
    """Tests for InvoiceHandler route handling."""

    def test_can_handle_static_routes(self, handler):
        """Can handle static routes."""
        assert handler.can_handle("/api/v1/accounting/invoices") is True
        assert handler.can_handle("/api/v1/accounting/invoices/upload") is True
        assert handler.can_handle("/api/v1/accounting/invoices/pending") is True
        assert handler.can_handle("/api/v1/accounting/invoices/overdue") is True
        assert handler.can_handle("/api/v1/accounting/invoices/stats") is True
        assert handler.can_handle("/api/v1/accounting/purchase-orders") is True
        assert handler.can_handle("/api/v1/accounting/payments/scheduled") is True

    def test_can_handle_dynamic_routes(self, handler):
        """Can handle dynamic routes."""
        assert handler.can_handle("/api/v1/accounting/invoices/inv_123") is True
        assert handler.can_handle("/api/v1/accounting/invoices/inv_456/approve") is True
        assert handler.can_handle("/api/v1/accounting/invoices/inv_789/reject") is True
        assert handler.can_handle("/api/v1/accounting/invoices/inv_123/match") is True
        assert handler.can_handle("/api/v1/accounting/invoices/inv_123/schedule") is True
        assert handler.can_handle("/api/v1/accounting/invoices/inv_123/anomalies") is True

    def test_cannot_handle_unknown_routes(self, handler):
        """Cannot handle unknown routes."""
        assert handler.can_handle("/api/v1/unknown") is False
        assert handler.can_handle("/api/v1/accounting/expenses") is False

    def test_extract_invoice_id(self, handler):
        """Correctly extract invoice ID from path."""
        assert handler._extract_invoice_id("/api/v1/accounting/invoices/inv_123") == "inv_123"
        assert (
            handler._extract_invoice_id("/api/v1/accounting/invoices/inv_456/approve") == "inv_456"
        )
        assert handler._extract_invoice_id("/api/v1/accounting/invoices") is None

    @pytest.mark.asyncio
    async def test_handle_get_list(self, handler, mock_processor):
        """Handle GET list request."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handler.handle_get("/api/v1/accounting/invoices", {})
            _, status = parse_result(result)

            assert status == 200

    @pytest.mark.asyncio
    async def test_handle_get_single(self, handler, mock_processor):
        """Handle GET single invoice request."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handler.handle_get("/api/v1/accounting/invoices/inv_123", {})
            _, status = parse_result(result)

            assert status == 200

    @pytest.mark.asyncio
    async def test_handle_get_pending(self, handler, mock_processor):
        """Handle GET pending approvals request."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handler.handle_get("/api/v1/accounting/invoices/pending", {})
            _, status = parse_result(result)

            assert status == 200

    @pytest.mark.asyncio
    async def test_handle_get_overdue(self, handler, mock_processor):
        """Handle GET overdue invoices request."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handler.handle_get("/api/v1/accounting/invoices/overdue", {})
            _, status = parse_result(result)

            assert status == 200

    @pytest.mark.asyncio
    async def test_handle_get_stats(self, handler, mock_processor):
        """Handle GET stats request."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handler.handle_get("/api/v1/accounting/invoices/stats", {})
            _, status = parse_result(result)

            assert status == 200

    @pytest.mark.asyncio
    async def test_handle_get_scheduled_payments(self, handler, mock_processor):
        """Handle GET scheduled payments request."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handler.handle_get("/api/v1/accounting/payments/scheduled", {})
            _, status = parse_result(result)

            assert status == 200

    @pytest.mark.asyncio
    async def test_handle_get_anomalies(self, handler, mock_processor):
        """Handle GET anomalies request."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handler.handle_get("/api/v1/accounting/invoices/inv_123/anomalies", {})
            _, status = parse_result(result)

            assert status == 200

    @pytest.mark.asyncio
    async def test_handle_post_upload(self, handler, mock_processor):
        """Handle POST upload request."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            pdf_data = b"fake PDF data"
            b64_data = base64.b64encode(pdf_data).decode()

            result = await handler.handle_post(
                "/api/v1/accounting/invoices/upload",
                {"document_data": b64_data},
            )
            _, status = parse_result(result)

            assert status == 200

    @pytest.mark.asyncio
    async def test_handle_post_create(self, handler, mock_processor):
        """Handle POST create request."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handler.handle_post(
                "/api/v1/accounting/invoices",
                {"vendor_name": "Test", "total_amount": 1000.00},
            )
            _, status = parse_result(result)

            assert status == 200

    @pytest.mark.asyncio
    async def test_handle_post_purchase_order(self, handler, mock_processor):
        """Handle POST purchase order request."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handler.handle_post(
                "/api/v1/accounting/purchase-orders",
                {"po_number": "PO-001", "vendor_name": "Test", "total_amount": 1000.00},
            )
            _, status = parse_result(result)

            assert status == 200

    @pytest.mark.asyncio
    async def test_handle_post_approve(self, handler, mock_processor):
        """Handle POST approve request."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handler.handle_post(
                "/api/v1/accounting/invoices/inv_123/approve",
                {},
            )
            _, status = parse_result(result)

            assert status == 200

    @pytest.mark.asyncio
    async def test_handle_post_reject(self, handler, mock_processor):
        """Handle POST reject request."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handler.handle_post(
                "/api/v1/accounting/invoices/inv_123/reject",
                {"reason": "Invalid"},
            )
            _, status = parse_result(result)

            assert status == 200

    @pytest.mark.asyncio
    async def test_handle_post_match(self, handler, mock_processor):
        """Handle POST match request."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handler.handle_post(
                "/api/v1/accounting/invoices/inv_123/match",
                {},
            )
            _, status = parse_result(result)

            assert status == 200

    @pytest.mark.asyncio
    async def test_handle_post_schedule(self, handler, mock_processor):
        """Handle POST schedule request."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor",
            return_value=mock_processor,
        ):
            result = await handler.handle_post(
                "/api/v1/accounting/invoices/inv_123/schedule",
                {},
            )
            _, status = parse_result(result)

            assert status == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
