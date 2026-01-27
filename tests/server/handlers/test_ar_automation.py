"""
Tests for aragora.server.handlers.ar_automation - AR Automation API handler.

Tests cover:
- Invoice Management:
  - Create invoice (POST /api/v1/accounting/ar/invoices)
  - List invoices (GET /api/v1/accounting/ar/invoices)
  - Get invoice (GET /api/v1/accounting/ar/invoices/{id})
  - Send invoice (POST /api/v1/accounting/ar/invoices/{id}/send)
  - Send reminder (POST /api/v1/accounting/ar/invoices/{id}/reminder)
  - Record payment (POST /api/v1/accounting/ar/invoices/{id}/payment)
- AR Reporting:
  - Get aging report (GET /api/v1/accounting/ar/aging)
  - Get collection suggestions (GET /api/v1/accounting/ar/collections)
- Customer Management:
  - Add customer (POST /api/v1/accounting/ar/customers)
  - Get customer balance (GET /api/v1/accounting/ar/customers/{id}/balance)
- RBAC permission checks
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.server.handlers import ar_automation


# ===========================================================================
# Mock Classes
# ===========================================================================


class MockInvoiceStatus(Enum):
    """Mock invoice status enum."""

    DRAFT = "draft"
    SENT = "sent"
    PAID = "paid"
    OVERDUE = "overdue"


@dataclass
class MockLineItem:
    """Mock invoice line item."""

    description: str
    quantity: float = 1.0
    unit_price: float = 0.0
    amount: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "quantity": self.quantity,
            "unit_price": self.unit_price,
            "amount": self.amount,
        }


@dataclass
class MockInvoice:
    """Mock invoice for testing."""

    id: str = "inv-123"
    customer_id: str = "cust-123"
    customer_name: str = "Acme Corp"
    customer_email: str | None = "billing@acme.com"
    line_items: list = field(default_factory=list)
    subtotal: Decimal = Decimal("1000.00")
    tax_amount: Decimal = Decimal("80.00")
    total: Decimal = Decimal("1080.00")
    amount_paid: Decimal = Decimal("0.00")
    balance_due: Decimal = Decimal("1080.00")
    status: MockInvoiceStatus = MockInvoiceStatus.DRAFT
    payment_terms: str = "Net 30"
    memo: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    due_date: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "customer_name": self.customer_name,
            "customer_email": self.customer_email,
            "line_items": [
                item.to_dict() if hasattr(item, "to_dict") else item for item in self.line_items
            ],
            "subtotal": str(self.subtotal),
            "tax_amount": str(self.tax_amount),
            "total": str(self.total),
            "amount_paid": str(self.amount_paid),
            "balance_due": str(self.balance_due),
            "status": self.status.value,
            "payment_terms": self.payment_terms,
            "memo": self.memo,
            "created_at": self.created_at.isoformat(),
            "due_date": self.due_date.isoformat() if self.due_date else None,
        }


@dataclass
class MockAgingReport:
    """Mock AR aging report."""

    current: Decimal = Decimal("5000.00")
    days_1_30: Decimal = Decimal("2500.00")
    days_31_60: Decimal = Decimal("1000.00")
    days_61_90: Decimal = Decimal("500.00")
    over_90: Decimal = Decimal("200.00")
    total_outstanding: Decimal = Decimal("9200.00")

    def to_dict(self) -> dict[str, Any]:
        return {
            "current": str(self.current),
            "days_1_30": str(self.days_1_30),
            "days_31_60": str(self.days_31_60),
            "days_61_90": str(self.days_61_90),
            "over_90": str(self.over_90),
            "total_outstanding": str(self.total_outstanding),
        }


@dataclass
class MockCollectionSuggestion:
    """Mock collection suggestion."""

    invoice_id: str = "inv-123"
    customer_id: str = "cust-123"
    customer_name: str = "Acme Corp"
    amount_due: Decimal = Decimal("1000.00")
    days_overdue: int = 45
    suggested_action: str = "escalate"
    priority: str = "high"

    def to_dict(self) -> dict[str, Any]:
        return {
            "invoice_id": self.invoice_id,
            "customer_id": self.customer_id,
            "customer_name": self.customer_name,
            "amount_due": str(self.amount_due),
            "days_overdue": self.days_overdue,
            "suggested_action": self.suggested_action,
            "priority": self.priority,
        }


class MockARAutomation:
    """Mock AR automation service for testing."""

    def __init__(self):
        self.invoices: dict[str, MockInvoice] = {}
        self.customers: dict[str, dict[str, Any]] = {}

    async def generate_invoice(
        self,
        customer_id: str,
        customer_name: str,
        customer_email: str | None = None,
        line_items: list | None = None,
        payment_terms: str = "Net 30",
        memo: str = "",
        tax_rate: float = 0,
    ) -> MockInvoice:
        invoice = MockInvoice(
            id=f"inv-{len(self.invoices) + 1}",
            customer_id=customer_id,
            customer_name=customer_name,
            customer_email=customer_email,
            line_items=line_items or [],
            payment_terms=payment_terms,
            memo=memo,
        )
        self.invoices[invoice.id] = invoice
        return invoice

    async def list_invoices(
        self,
        customer_id: str | None = None,
        status: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[MockInvoice]:
        invoices = list(self.invoices.values())
        if customer_id:
            invoices = [inv for inv in invoices if inv.customer_id == customer_id]
        if status:
            invoices = [inv for inv in invoices if inv.status.value == status]
        return invoices

    async def get_invoice(self, invoice_id: str) -> MockInvoice | None:
        return self.invoices.get(invoice_id)

    async def send_invoice(self, invoice_id: str) -> bool:
        invoice = self.invoices.get(invoice_id)
        if invoice:
            invoice.status = MockInvoiceStatus.SENT
            return True
        return False

    async def send_payment_reminder(self, invoice_id: str, escalation_level: int = 1) -> bool:
        return invoice_id in self.invoices

    async def record_payment(
        self,
        invoice_id: str,
        amount: Decimal,
        payment_date: datetime | None = None,
        payment_method: str | None = None,
        reference: str | None = None,
    ) -> MockInvoice | None:
        invoice = self.invoices.get(invoice_id)
        if invoice:
            invoice.amount_paid += amount
            invoice.balance_due -= amount
            if invoice.balance_due <= 0:
                invoice.status = MockInvoiceStatus.PAID
            return invoice
        return None

    async def track_aging(self) -> MockAgingReport:
        return MockAgingReport()

    async def suggest_collections(self) -> list[MockCollectionSuggestion]:
        return [MockCollectionSuggestion()]

    async def add_customer(
        self,
        customer_id: str,
        name: str,
        email: str | None = None,
        payment_terms: str = "Net 30",
    ) -> None:
        self.customers[customer_id] = {
            "customer_id": customer_id,
            "name": name,
            "email": email,
            "payment_terms": payment_terms,
        }

    async def get_customer_balance(self, customer_id: str) -> Decimal:
        total = Decimal("0.00")
        for invoice in self.invoices.values():
            if invoice.customer_id == customer_id:
                total += invoice.balance_due
        return total


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def mock_ar_service():
    """Create a mock AR automation service with test data."""
    service = MockARAutomation()

    # Add test invoices
    service.invoices["inv-123"] = MockInvoice(
        id="inv-123",
        customer_id="cust-123",
        customer_name="Acme Corp",
        line_items=[{"description": "Consulting", "amount": 1000.0}],
    )
    service.invoices["inv-456"] = MockInvoice(
        id="inv-456",
        customer_id="cust-456",
        customer_name="Tech Inc",
        status=MockInvoiceStatus.SENT,
    )

    # Add test customers
    service.customers["cust-123"] = {
        "customer_id": "cust-123",
        "name": "Acme Corp",
        "email": "billing@acme.com",
    }

    return service


# ===========================================================================
# Helper Functions
# ===========================================================================


def parse_response(result) -> tuple[dict[str, Any], int]:
    """Parse HandlerResult into (body_dict, status_code).

    Note: success_response wraps data in {"data": ..., "success": True}
    """
    body = json.loads(result.body) if result.body else {}
    # Unwrap data if present (success_response format)
    if "data" in body and body.get("success"):
        return body["data"], result.status_code
    return body, result.status_code


# ===========================================================================
# Test: Create Invoice
# ===========================================================================


class TestCreateInvoice:
    """Tests for POST /api/v1/accounting/ar/invoices."""

    @pytest.mark.asyncio
    async def test_create_invoice_success(self, mock_ar_service):
        """Should create a new invoice."""
        with patch.object(ar_automation, "get_ar_automation", return_value=mock_ar_service):
            result = await ar_automation.handle_create_invoice(
                data={
                    "customer_id": "cust-789",
                    "customer_name": "New Customer",
                    "customer_email": "new@customer.com",
                    "line_items": [
                        {"description": "Service", "amount": 500.0},
                    ],
                    "payment_terms": "Net 15",
                },
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert "invoice" in body
            assert body["invoice"]["customer_name"] == "New Customer"

    @pytest.mark.asyncio
    async def test_create_invoice_missing_customer_id(self, mock_ar_service):
        """Should return 400 when customer_id is missing."""
        with patch.object(ar_automation, "get_ar_automation", return_value=mock_ar_service):
            result = await ar_automation.handle_create_invoice(
                data={
                    "customer_name": "New Customer",
                    "line_items": [{"description": "Service", "amount": 500.0}],
                },
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 400
            assert "customer_id" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_create_invoice_missing_line_items(self, mock_ar_service):
        """Should return 400 when line_items is missing."""
        with patch.object(ar_automation, "get_ar_automation", return_value=mock_ar_service):
            result = await ar_automation.handle_create_invoice(
                data={
                    "customer_id": "cust-789",
                    "customer_name": "New Customer",
                },
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 400
            assert "line_items" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_create_invoice_invalid_line_item(self, mock_ar_service):
        """Should return 400 for invalid line items."""
        with patch.object(ar_automation, "get_ar_automation", return_value=mock_ar_service):
            result = await ar_automation.handle_create_invoice(
                data={
                    "customer_id": "cust-789",
                    "customer_name": "New Customer",
                    "line_items": [{"invalid": "item"}],  # Missing description and amount
                },
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 400


# ===========================================================================
# Test: List Invoices
# ===========================================================================


class TestListInvoices:
    """Tests for GET /api/v1/accounting/ar/invoices."""

    @pytest.mark.asyncio
    async def test_list_invoices_success(self, mock_ar_service):
        """Should return list of invoices."""
        with patch.object(ar_automation, "get_ar_automation", return_value=mock_ar_service):
            result = await ar_automation.handle_list_invoices(
                data={},
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert "invoices" in body
            assert body["total"] == 2

    @pytest.mark.asyncio
    async def test_list_invoices_with_filter(self, mock_ar_service):
        """Should filter invoices by customer_id."""
        with patch.object(ar_automation, "get_ar_automation", return_value=mock_ar_service):
            result = await ar_automation.handle_list_invoices(
                data={"customer_id": "cust-123"},
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert body["total"] == 1
            assert body["invoices"][0]["customer_id"] == "cust-123"


# ===========================================================================
# Test: Get Invoice
# ===========================================================================


class TestGetInvoice:
    """Tests for GET /api/v1/accounting/ar/invoices/{invoice_id}."""

    @pytest.mark.asyncio
    async def test_get_invoice_success(self, mock_ar_service):
        """Should return invoice by ID."""
        with patch.object(ar_automation, "get_ar_automation", return_value=mock_ar_service):
            result = await ar_automation.handle_get_invoice(
                data={},
                invoice_id="inv-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert body["invoice"]["id"] == "inv-123"
            assert body["invoice"]["customer_name"] == "Acme Corp"

    @pytest.mark.asyncio
    async def test_get_invoice_not_found(self, mock_ar_service):
        """Should return 404 for non-existent invoice."""
        with patch.object(ar_automation, "get_ar_automation", return_value=mock_ar_service):
            result = await ar_automation.handle_get_invoice(
                data={},
                invoice_id="inv-nonexistent",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 404


# ===========================================================================
# Test: Send Invoice
# ===========================================================================


class TestSendInvoice:
    """Tests for POST /api/v1/accounting/ar/invoices/{invoice_id}/send."""

    @pytest.mark.asyncio
    async def test_send_invoice_success(self, mock_ar_service):
        """Should send invoice successfully."""
        with patch.object(ar_automation, "get_ar_automation", return_value=mock_ar_service):
            result = await ar_automation.handle_send_invoice(
                data={},
                invoice_id="inv-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert "sent successfully" in body.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_send_invoice_not_found(self, mock_ar_service):
        """Should return 404 for non-existent invoice."""
        with patch.object(ar_automation, "get_ar_automation", return_value=mock_ar_service):
            result = await ar_automation.handle_send_invoice(
                data={},
                invoice_id="inv-nonexistent",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 404


# ===========================================================================
# Test: Send Reminder
# ===========================================================================


class TestSendReminder:
    """Tests for POST /api/v1/accounting/ar/invoices/{invoice_id}/reminder."""

    @pytest.mark.asyncio
    async def test_send_reminder_success(self, mock_ar_service):
        """Should send payment reminder successfully."""
        with patch.object(ar_automation, "get_ar_automation", return_value=mock_ar_service):
            result = await ar_automation.handle_send_reminder(
                data={"escalation_level": 2},
                invoice_id="inv-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert "reminder" in body.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_send_reminder_invalid_level(self, mock_ar_service):
        """Should return 400 for invalid escalation level."""
        with patch.object(ar_automation, "get_ar_automation", return_value=mock_ar_service):
            result = await ar_automation.handle_send_reminder(
                data={"escalation_level": 5},
                invoice_id="inv-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 400
            assert "escalation_level" in body.get("error", "").lower()


# ===========================================================================
# Test: Record Payment
# ===========================================================================


class TestRecordPayment:
    """Tests for POST /api/v1/accounting/ar/invoices/{invoice_id}/payment."""

    @pytest.mark.asyncio
    async def test_record_payment_success(self, mock_ar_service):
        """Should record payment successfully."""
        with patch.object(ar_automation, "get_ar_automation", return_value=mock_ar_service):
            result = await ar_automation.handle_record_payment(
                data={
                    "amount": 500.0,
                    "payment_method": "check",
                    "reference": "CHK-001",
                },
                invoice_id="inv-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert "Payment" in body.get("message", "")

    @pytest.mark.asyncio
    async def test_record_payment_missing_amount(self, mock_ar_service):
        """Should return 400 when amount is missing."""
        with patch.object(ar_automation, "get_ar_automation", return_value=mock_ar_service):
            result = await ar_automation.handle_record_payment(
                data={"payment_method": "check"},
                invoice_id="inv-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 400
            assert "amount" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_record_payment_not_found(self, mock_ar_service):
        """Should return 404 for non-existent invoice."""
        with patch.object(ar_automation, "get_ar_automation", return_value=mock_ar_service):
            result = await ar_automation.handle_record_payment(
                data={"amount": 500.0},
                invoice_id="inv-nonexistent",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 404


# ===========================================================================
# Test: Aging Report
# ===========================================================================


class TestAgingReport:
    """Tests for GET /api/v1/accounting/ar/aging."""

    @pytest.mark.asyncio
    async def test_get_aging_report_success(self, mock_ar_service):
        """Should return AR aging report."""
        with patch.object(ar_automation, "get_ar_automation", return_value=mock_ar_service):
            result = await ar_automation.handle_get_aging_report(
                data={},
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert "aging_report" in body
            assert "total_outstanding" in body["aging_report"]


# ===========================================================================
# Test: Collection Suggestions
# ===========================================================================


class TestCollectionSuggestions:
    """Tests for GET /api/v1/accounting/ar/collections."""

    @pytest.mark.asyncio
    async def test_get_collections_success(self, mock_ar_service):
        """Should return collection suggestions."""
        with patch.object(ar_automation, "get_ar_automation", return_value=mock_ar_service):
            result = await ar_automation.handle_get_collections(
                data={},
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert "suggestions" in body
            assert body["total"] >= 1


# ===========================================================================
# Test: Add Customer
# ===========================================================================


class TestAddCustomer:
    """Tests for POST /api/v1/accounting/ar/customers."""

    @pytest.mark.asyncio
    async def test_add_customer_success(self, mock_ar_service):
        """Should add customer successfully."""
        with patch.object(ar_automation, "get_ar_automation", return_value=mock_ar_service):
            result = await ar_automation.handle_add_customer(
                data={
                    "customer_id": "cust-new",
                    "name": "New Customer Inc",
                    "email": "billing@newcustomer.com",
                },
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert body["customer_id"] == "cust-new"

    @pytest.mark.asyncio
    async def test_add_customer_missing_id(self, mock_ar_service):
        """Should return 400 when customer_id is missing."""
        with patch.object(ar_automation, "get_ar_automation", return_value=mock_ar_service):
            result = await ar_automation.handle_add_customer(
                data={"name": "New Customer Inc"},
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 400
            assert "customer_id" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_add_customer_missing_name(self, mock_ar_service):
        """Should return 400 when name is missing."""
        with patch.object(ar_automation, "get_ar_automation", return_value=mock_ar_service):
            result = await ar_automation.handle_add_customer(
                data={"customer_id": "cust-new"},
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 400
            assert "name" in body.get("error", "").lower()


# ===========================================================================
# Test: Get Customer Balance
# ===========================================================================


class TestGetCustomerBalance:
    """Tests for GET /api/v1/accounting/ar/customers/{customer_id}/balance."""

    @pytest.mark.asyncio
    async def test_get_customer_balance_success(self, mock_ar_service):
        """Should return customer balance."""
        with patch.object(ar_automation, "get_ar_automation", return_value=mock_ar_service):
            result = await ar_automation.handle_get_customer_balance(
                data={},
                customer_id="cust-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert body["customer_id"] == "cust-123"
            assert "outstanding_balance" in body
