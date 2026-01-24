"""Tests for invoice handler endpoints."""

import base64
import json
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.invoices import (
    InvoiceHandler,
    handle_approve_invoice,
    handle_create_invoice,
    handle_create_purchase_order,
    handle_get_anomalies,
    handle_get_invoice,
    handle_get_invoice_stats,
    handle_get_overdue_invoices,
    handle_get_pending_approvals,
    handle_get_scheduled_payments,
    handle_list_invoices,
    handle_match_to_po,
    handle_reject_invoice,
    handle_schedule_payment,
    handle_upload_invoice,
)


def get_data(body: dict) -> dict:
    """Extract data from wrapped response or return body if not wrapped."""
    return body.get("data", body)


@pytest.fixture
def mock_invoice():
    """Create a mock invoice for testing."""
    mock = MagicMock()
    mock.id = "inv_test123"
    mock.vendor_name = "Test Vendor"
    mock.invoice_number = "INV-001"
    mock.total_amount = Decimal("1000.00")
    mock.invoice_date = datetime.now()
    mock.due_date = datetime.now() + timedelta(days=30)
    mock.status = MagicMock(value="extracted")
    mock.to_dict.return_value = {
        "id": "inv_test123",
        "vendorName": "Test Vendor",
        "invoiceNumber": "INV-001",
        "totalAmount": 1000.00,
        "status": "extracted",
    }
    return mock


@pytest.fixture
def mock_anomaly():
    """Create a mock anomaly for testing."""
    mock = MagicMock()
    mock.type = MagicMock(value="new_vendor")
    mock.severity = "medium"
    mock.description = "First invoice from vendor"
    mock.to_dict.return_value = {
        "type": "new_vendor",
        "severity": "medium",
        "description": "First invoice from vendor",
    }
    return mock


@pytest.fixture
def mock_po():
    """Create a mock purchase order for testing."""
    mock = MagicMock()
    mock.id = "po_test123"
    mock.po_number = "PO-001"
    mock.vendor_name = "Test Vendor"
    mock.total_amount = Decimal("1000.00")
    mock.to_dict.return_value = {
        "id": "po_test123",
        "poNumber": "PO-001",
        "vendorName": "Test Vendor",
        "totalAmount": 1000.00,
    }
    return mock


@pytest.fixture
def mock_po_match():
    """Create a mock PO match result for testing."""
    mock = MagicMock()
    mock.invoice_id = "inv_test123"
    mock.po_id = "po_test123"
    mock.po_number = "PO-001"
    mock.match_type = "exact"
    mock.match_score = 1.0
    mock.to_dict.return_value = {
        "invoiceId": "inv_test123",
        "poId": "po_test123",
        "poNumber": "PO-001",
        "matchType": "exact",
        "matchScore": 1.0,
    }
    return mock


@pytest.fixture
def mock_schedule():
    """Create a mock payment schedule for testing."""
    mock = MagicMock()
    mock.invoice_id = "inv_test123"
    mock.pay_date = datetime.now() + timedelta(days=30)
    mock.amount = Decimal("1000.00")
    mock.vendor_name = "Test Vendor"
    mock.to_dict.return_value = {
        "invoiceId": "inv_test123",
        "payDate": (datetime.now() + timedelta(days=30)).isoformat(),
        "amount": 1000.00,
        "vendorName": "Test Vendor",
    }
    return mock


@pytest.fixture
def mock_processor():
    """Create a mock invoice processor."""
    mock = MagicMock()
    mock.extract_invoice_data = AsyncMock()
    mock.detect_anomalies = AsyncMock(return_value=[])
    mock.create_manual_invoice = AsyncMock()
    mock.list_invoices = AsyncMock(return_value=([], 0))
    mock.get_invoice = AsyncMock()
    mock.approve_invoice = AsyncMock()
    mock.reject_invoice = AsyncMock()
    mock.match_to_po = AsyncMock()
    mock.schedule_payment = AsyncMock()
    mock.get_pending_approvals = AsyncMock(return_value=[])
    mock.get_scheduled_payments = AsyncMock(return_value=[])
    mock.get_overdue_invoices = AsyncMock(return_value=[])
    mock.add_purchase_order = AsyncMock()
    mock.get_stats = MagicMock(
        return_value={
            "totalInvoices": 0,
            "totalAmount": 0,
            "pendingApproval": 0,
            "overdue": 0,
            "byStatus": {},
        }
    )
    return mock


class TestUploadInvoice:
    """Tests for invoice upload handler."""

    @pytest.mark.asyncio
    async def test_upload_invoice_success(self, mock_processor, mock_invoice, mock_anomaly):
        """Test successful invoice upload and extraction."""
        mock_processor.extract_invoice_data.return_value = mock_invoice
        mock_processor.detect_anomalies.return_value = [mock_anomaly]

        # Create base64 encoded PDF
        pdf_content = b"%PDF-1.4 test content"
        doc_b64 = base64.b64encode(pdf_content).decode()

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_upload_invoice(
                {
                    "document_data": doc_b64,
                    "vendor_hint": "Test Vendor",
                }
            )

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        data = get_data(body)
        assert "invoice" in data
        assert "anomalies" in data
        assert data["message"] == "Invoice extracted successfully"
        mock_processor.extract_invoice_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_invoice_missing_document(self):
        """Test upload fails without document data."""
        result = await handle_upload_invoice({})

        assert result.status_code == 400
        body = json.loads(result.body.decode())
        assert "document_data is required" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_upload_invoice_invalid_base64(self):
        """Test upload fails with invalid base64."""
        result = await handle_upload_invoice({"document_data": "not-valid-base64!!!"})

        assert result.status_code == 400
        body = json.loads(result.body.decode())
        assert "Invalid base64" in body.get("error", "")


class TestCreateInvoice:
    """Tests for manual invoice creation."""

    @pytest.mark.asyncio
    async def test_create_invoice_success(self, mock_processor, mock_invoice):
        """Test successful manual invoice creation."""
        mock_processor.create_manual_invoice.return_value = mock_invoice

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_create_invoice(
                {
                    "vendor_name": "Test Vendor",
                    "total_amount": 1000.00,
                    "invoice_number": "INV-001",
                }
            )

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        data = get_data(body)
        assert "invoice" in data
        assert data["message"] == "Invoice created successfully"

    @pytest.mark.asyncio
    async def test_create_invoice_missing_vendor(self):
        """Test creation fails without vendor name."""
        result = await handle_create_invoice({"total_amount": 1000.00})

        assert result.status_code == 400
        body = json.loads(result.body.decode())
        assert "vendor_name is required" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_create_invoice_missing_amount(self):
        """Test creation fails without total amount."""
        result = await handle_create_invoice({"vendor_name": "Test Vendor"})

        assert result.status_code == 400
        body = json.loads(result.body.decode())
        assert "total_amount is required" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_create_invoice_invalid_amount(self):
        """Test creation fails with invalid amount."""
        result = await handle_create_invoice(
            {
                "vendor_name": "Test Vendor",
                "total_amount": "not-a-number",
            }
        )

        assert result.status_code == 400
        body = json.loads(result.body.decode())
        assert "total_amount must be a number" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_create_invoice_with_dates(self, mock_processor, mock_invoice):
        """Test invoice creation with dates."""
        mock_processor.create_manual_invoice.return_value = mock_invoice

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_create_invoice(
                {
                    "vendor_name": "Test Vendor",
                    "total_amount": 1000.00,
                    "invoice_date": "2024-01-15T00:00:00Z",
                    "due_date": "2024-02-15T00:00:00Z",
                }
            )

        assert result.status_code == 200
        mock_processor.create_manual_invoice.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_invoice_invalid_date(self):
        """Test creation fails with invalid date format."""
        result = await handle_create_invoice(
            {
                "vendor_name": "Test Vendor",
                "total_amount": 1000.00,
                "invoice_date": "not-a-date",
            }
        )

        assert result.status_code == 400
        body = json.loads(result.body.decode())
        assert "Invalid invoice_date format" in body.get("error", "")


class TestListInvoices:
    """Tests for listing invoices."""

    @pytest.mark.asyncio
    async def test_list_invoices_empty(self, mock_processor):
        """Test listing returns empty list."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_list_invoices({})

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        data = get_data(body)
        assert data["invoices"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_list_invoices_with_results(self, mock_processor, mock_invoice):
        """Test listing returns invoices."""
        mock_processor.list_invoices.return_value = ([mock_invoice], 1)

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_list_invoices({})

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        data = get_data(body)
        assert len(data["invoices"]) == 1
        assert data["total"] == 1

    @pytest.mark.asyncio
    async def test_list_invoices_with_filters(self, mock_processor):
        """Test listing with status and vendor filters."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_list_invoices(
                {
                    "status": "extracted",
                    "vendor": "Test",
                    "start_date": "2024-01-01T00:00:00Z",
                    "end_date": "2024-12-31T23:59:59Z",
                    "limit": 50,
                    "offset": 10,
                }
            )

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        data = get_data(body)
        assert data["limit"] == 50
        assert data["offset"] == 10


class TestGetInvoice:
    """Tests for getting single invoice."""

    @pytest.mark.asyncio
    async def test_get_invoice_success(self, mock_processor, mock_invoice):
        """Test getting invoice by ID."""
        mock_processor.get_invoice.return_value = mock_invoice

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_get_invoice("inv_test123")

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        data = get_data(body)
        assert "invoice" in data

    @pytest.mark.asyncio
    async def test_get_invoice_not_found(self, mock_processor):
        """Test getting non-existent invoice."""
        mock_processor.get_invoice.return_value = None

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_get_invoice("inv_nonexistent")

        assert result.status_code == 404
        body = json.loads(result.body.decode())
        assert "not found" in body.get("error", "").lower()


class TestApproveInvoice:
    """Tests for invoice approval."""

    @pytest.mark.asyncio
    async def test_approve_invoice_success(self, mock_processor, mock_invoice):
        """Test successful invoice approval."""
        mock_processor.approve_invoice.return_value = mock_invoice

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_approve_invoice("inv_test123", {"approver_id": "user123"})

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        data = get_data(body)
        assert data["message"] == "Invoice approved successfully"
        mock_processor.approve_invoice.assert_called_once_with("inv_test123", "user123")

    @pytest.mark.asyncio
    async def test_approve_invoice_not_found(self, mock_processor):
        """Test approving non-existent invoice."""
        mock_processor.approve_invoice.return_value = None

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_approve_invoice("inv_nonexistent", {})

        assert result.status_code == 404


class TestRejectInvoice:
    """Tests for invoice rejection."""

    @pytest.mark.asyncio
    async def test_reject_invoice_success(self, mock_processor, mock_invoice):
        """Test successful invoice rejection."""
        mock_processor.reject_invoice.return_value = mock_invoice

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_reject_invoice("inv_test123", {"reason": "Invalid"})

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        data = get_data(body)
        assert data["message"] == "Invoice rejected"
        mock_processor.reject_invoice.assert_called_once_with("inv_test123", "Invalid")

    @pytest.mark.asyncio
    async def test_reject_invoice_not_found(self, mock_processor):
        """Test rejecting non-existent invoice."""
        mock_processor.reject_invoice.return_value = None

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_reject_invoice("inv_nonexistent", {})

        assert result.status_code == 404


class TestMatchToPO:
    """Tests for PO matching."""

    @pytest.mark.asyncio
    async def test_match_to_po_success(self, mock_processor, mock_invoice, mock_po_match):
        """Test successful PO matching."""
        mock_processor.get_invoice.return_value = mock_invoice
        mock_processor.match_to_po.return_value = mock_po_match

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_match_to_po("inv_test123")

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        data = get_data(body)
        assert "match" in data
        assert "invoice" in data

    @pytest.mark.asyncio
    async def test_match_to_po_invoice_not_found(self, mock_processor):
        """Test matching with non-existent invoice."""
        mock_processor.get_invoice.return_value = None

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_match_to_po("inv_nonexistent")

        assert result.status_code == 404


class TestSchedulePayment:
    """Tests for payment scheduling."""

    @pytest.mark.asyncio
    async def test_schedule_payment_success(self, mock_processor, mock_invoice, mock_schedule):
        """Test successful payment scheduling."""
        mock_processor.get_invoice.return_value = mock_invoice
        mock_processor.schedule_payment.return_value = mock_schedule

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_schedule_payment(
                "inv_test123",
                {
                    "pay_date": "2024-02-15T00:00:00Z",
                    "payment_method": "ach",
                },
            )

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        data = get_data(body)
        assert "schedule" in data
        assert data["message"] == "Payment scheduled successfully"

    @pytest.mark.asyncio
    async def test_schedule_payment_invoice_not_found(self, mock_processor):
        """Test scheduling with non-existent invoice."""
        mock_processor.get_invoice.return_value = None

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_schedule_payment("inv_nonexistent", {})

        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_schedule_payment_invalid_date(self, mock_processor, mock_invoice):
        """Test scheduling with invalid date."""
        mock_processor.get_invoice.return_value = mock_invoice

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_schedule_payment(
                "inv_test123",
                {
                    "pay_date": "not-a-date",
                },
            )

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_schedule_payment_validation_error(self, mock_processor, mock_invoice):
        """Test scheduling returns validation error."""
        mock_processor.get_invoice.return_value = mock_invoice
        mock_processor.schedule_payment.side_effect = ValueError("Invoice must be approved")

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_schedule_payment("inv_test123", {})

        assert result.status_code == 400


class TestGetAnomalies:
    """Tests for anomaly detection."""

    @pytest.mark.asyncio
    async def test_get_anomalies_success(self, mock_processor, mock_invoice, mock_anomaly):
        """Test getting anomalies for invoice."""
        mock_processor.get_invoice.return_value = mock_invoice
        mock_processor.detect_anomalies.return_value = [mock_anomaly]

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_get_anomalies("inv_test123")

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert len(body["anomalies"]) == 1
        assert body["count"] == 1

    @pytest.mark.asyncio
    async def test_get_anomalies_invoice_not_found(self, mock_processor):
        """Test getting anomalies for non-existent invoice."""
        mock_processor.get_invoice.return_value = None

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_get_anomalies("inv_nonexistent")

        assert result.status_code == 404


class TestGetPendingApprovals:
    """Tests for pending approvals."""

    @pytest.mark.asyncio
    async def test_get_pending_approvals_empty(self, mock_processor):
        """Test empty pending approvals."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_get_pending_approvals()

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert body["invoices"] == []
        assert body["count"] == 0

    @pytest.mark.asyncio
    async def test_get_pending_approvals_with_results(self, mock_processor, mock_invoice):
        """Test pending approvals with results."""
        mock_processor.get_pending_approvals.return_value = [mock_invoice]

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_get_pending_approvals()

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert len(body["invoices"]) == 1


class TestGetScheduledPayments:
    """Tests for scheduled payments."""

    @pytest.mark.asyncio
    async def test_get_scheduled_payments_empty(self, mock_processor):
        """Test empty scheduled payments."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_get_scheduled_payments({})

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert body["payments"] == []
        assert body["count"] == 0

    @pytest.mark.asyncio
    async def test_get_scheduled_payments_with_dates(self, mock_processor, mock_schedule):
        """Test scheduled payments with date filters."""
        mock_processor.get_scheduled_payments.return_value = [mock_schedule]

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_get_scheduled_payments(
                {
                    "start_date": "2024-01-01T00:00:00Z",
                    "end_date": "2024-12-31T23:59:59Z",
                }
            )

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert body["count"] == 1


class TestCreatePurchaseOrder:
    """Tests for purchase order creation."""

    @pytest.mark.asyncio
    async def test_create_po_success(self, mock_processor, mock_po):
        """Test successful PO creation."""
        mock_processor.add_purchase_order.return_value = mock_po

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_create_purchase_order(
                {
                    "po_number": "PO-001",
                    "vendor_name": "Test Vendor",
                    "total_amount": 1000.00,
                }
            )

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert "purchaseOrder" in body
        assert body["message"] == "Purchase order created successfully"

    @pytest.mark.asyncio
    async def test_create_po_missing_number(self):
        """Test PO creation fails without number."""
        result = await handle_create_purchase_order(
            {
                "vendor_name": "Test Vendor",
                "total_amount": 1000.00,
            }
        )

        assert result.status_code == 400
        body = json.loads(result.body.decode())
        assert "po_number is required" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_create_po_missing_vendor(self):
        """Test PO creation fails without vendor."""
        result = await handle_create_purchase_order(
            {
                "po_number": "PO-001",
                "total_amount": 1000.00,
            }
        )

        assert result.status_code == 400
        body = json.loads(result.body.decode())
        assert "vendor_name is required" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_create_po_missing_amount(self):
        """Test PO creation fails without amount."""
        result = await handle_create_purchase_order(
            {
                "po_number": "PO-001",
                "vendor_name": "Test Vendor",
            }
        )

        assert result.status_code == 400
        body = json.loads(result.body.decode())
        assert "total_amount is required" in body.get("error", "")


class TestGetInvoiceStats:
    """Tests for invoice statistics."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self, mock_processor):
        """Test getting invoice stats."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_get_invoice_stats()

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert "stats" in body


class TestGetOverdueInvoices:
    """Tests for overdue invoices."""

    @pytest.mark.asyncio
    async def test_get_overdue_empty(self, mock_processor):
        """Test empty overdue invoices."""
        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_get_overdue_invoices()

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert body["invoices"] == []
        assert body["count"] == 0

    @pytest.mark.asyncio
    async def test_get_overdue_with_results(self, mock_processor, mock_invoice):
        """Test overdue invoices with results."""
        mock_processor.get_overdue_invoices.return_value = [mock_invoice]

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handle_get_overdue_invoices()

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert body["count"] == 1


class TestInvoiceHandler:
    """Tests for InvoiceHandler class."""

    def test_can_handle_static_routes(self):
        """Test handler recognizes static routes."""
        handler = InvoiceHandler(server_context={})

        assert handler.can_handle("/api/v1/accounting/invoices")
        assert handler.can_handle("/api/v1/accounting/invoices/upload")
        assert handler.can_handle("/api/v1/accounting/invoices/pending")
        assert handler.can_handle("/api/v1/accounting/invoices/overdue")
        assert handler.can_handle("/api/v1/accounting/invoices/stats")
        assert handler.can_handle("/api/v1/accounting/purchase-orders")
        assert handler.can_handle("/api/v1/accounting/payments/scheduled")

    def test_can_handle_dynamic_routes(self):
        """Test handler recognizes dynamic routes."""
        handler = InvoiceHandler(server_context={})

        assert handler.can_handle("/api/v1/accounting/invoices/inv_123")
        assert handler.can_handle("/api/v1/accounting/invoices/inv_123/approve")
        assert handler.can_handle("/api/v1/accounting/invoices/inv_123/reject")
        assert handler.can_handle("/api/v1/accounting/invoices/inv_123/match")
        assert handler.can_handle("/api/v1/accounting/invoices/inv_123/schedule")
        assert handler.can_handle("/api/v1/accounting/invoices/inv_123/anomalies")

    def test_cannot_handle_unknown_routes(self):
        """Test handler rejects unknown routes."""
        handler = InvoiceHandler(server_context={})

        assert not handler.can_handle("/api/v1/unknown")
        assert not handler.can_handle("/api/v1/accounting/unknown")
        assert not handler.can_handle("/api/v1/accounting/invoices/inv_123/unknown")

    def test_extract_invoice_id(self):
        """Test invoice ID extraction from path."""
        handler = InvoiceHandler(server_context={})

        assert handler._extract_invoice_id("/api/v1/accounting/invoices/inv_123") == "inv_123"
        assert (
            handler._extract_invoice_id("/api/v1/accounting/invoices/inv_123/approve") == "inv_123"
        )
        assert handler._extract_invoice_id("/api/v1/accounting/invoices") is None

    @pytest.mark.asyncio
    async def test_handle_get_list(self, mock_processor):
        """Test GET handler for listing invoices."""
        handler = InvoiceHandler(server_context={})

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handler.handle_get("/api/v1/accounting/invoices")

        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_get_single(self, mock_processor, mock_invoice):
        """Test GET handler for single invoice."""
        handler = InvoiceHandler(server_context={})
        mock_processor.get_invoice.return_value = mock_invoice

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handler.handle_get("/api/v1/accounting/invoices/inv_123")

        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_get_anomalies(self, mock_processor, mock_invoice):
        """Test GET handler for anomalies."""
        handler = InvoiceHandler(server_context={})
        mock_processor.get_invoice.return_value = mock_invoice
        mock_processor.detect_anomalies.return_value = []

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handler.handle_get("/api/v1/accounting/invoices/inv_123/anomalies")

        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_post_create(self, mock_processor, mock_invoice):
        """Test POST handler for creating invoice."""
        handler = InvoiceHandler(server_context={})
        mock_processor.create_manual_invoice.return_value = mock_invoice

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handler.handle_post(
                "/api/v1/accounting/invoices",
                {
                    "vendor_name": "Test",
                    "total_amount": 100,
                },
            )

        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_post_approve(self, mock_processor, mock_invoice):
        """Test POST handler for approving invoice."""
        handler = InvoiceHandler(server_context={})
        mock_processor.approve_invoice.return_value = mock_invoice

        with patch(
            "aragora.server.handlers.invoices.get_invoice_processor", return_value=mock_processor
        ):
            result = await handler.handle_post("/api/v1/accounting/invoices/inv_123/approve", {})

        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_post_unknown_route(self):
        """Test POST handler returns 404 for unknown route."""
        handler = InvoiceHandler(server_context={})

        result = await handler.handle_post("/api/v1/unknown", {})

        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_handle_get_unknown_route(self):
        """Test GET handler returns 404 for unknown route."""
        handler = InvoiceHandler(server_context={})

        result = await handler.handle_get("/api/v1/unknown")

        assert result.status_code == 404
