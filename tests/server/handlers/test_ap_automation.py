"""
Tests for aragora.server.handlers.ap_automation - AP Automation API handler.

Tests cover:
- Invoice Management:
  - Add invoice (POST /api/v1/accounting/ap/invoices)
  - List invoices (GET /api/v1/accounting/ap/invoices)
  - Get invoice (GET /api/v1/accounting/ap/invoices/{id})
  - Record payment (POST /api/v1/accounting/ap/invoices/{id}/payment)
- Payment Optimization:
  - Optimize payments (POST /api/v1/accounting/ap/optimize)
  - Batch payments (POST /api/v1/accounting/ap/batch)
- Forecasting:
  - Get forecast (GET /api/v1/accounting/ap/forecast)
  - Get discounts (GET /api/v1/accounting/ap/discounts)
- Input validation and error handling
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from unittest.mock import patch

import pytest

from aragora.server.handlers import ap_automation


# ===========================================================================
# Mock Classes
# ===========================================================================


class MockInvoiceStatus(Enum):
    """Mock invoice status enum."""

    UNPAID = "unpaid"
    PARTIAL = "partial"
    PAID = "paid"


class MockPriority(Enum):
    """Mock priority enum."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    HOLD = "hold"


@dataclass
class MockAPInvoice:
    """Mock AP invoice for testing."""

    id: str = "ap-inv-123"
    vendor_id: str = "vendor-123"
    vendor_name: str = "Vendor Corp"
    invoice_number: str = "INV-001"
    invoice_date: datetime | None = None
    due_date: datetime | None = None
    total_amount: Decimal = Decimal("5000.00")
    amount_paid: Decimal = Decimal("0.00")
    balance_due: Decimal = Decimal("5000.00")
    status: MockInvoiceStatus = MockInvoiceStatus.UNPAID
    payment_terms: str = "Net 30"
    early_pay_discount: float = 0.02
    discount_deadline: datetime | None = None
    priority: str = "normal"
    preferred_payment_method: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "vendor_id": self.vendor_id,
            "vendor_name": self.vendor_name,
            "invoice_number": self.invoice_number,
            "invoice_date": self.invoice_date.isoformat() if self.invoice_date else None,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "total_amount": str(self.total_amount),
            "amount_paid": str(self.amount_paid),
            "balance_due": str(self.balance_due),
            "status": self.status.value,
            "payment_terms": self.payment_terms,
            "early_pay_discount": self.early_pay_discount,
            "discount_deadline": (
                self.discount_deadline.isoformat() if self.discount_deadline else None
            ),
            "priority": self.priority,
            "preferred_payment_method": self.preferred_payment_method,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class MockPaymentSchedule:
    """Mock payment schedule."""

    payments: list = field(default_factory=list)
    total_amount: Decimal = Decimal("10000.00")
    total_discount_captured: Decimal = Decimal("200.00")
    recommended_payment_date: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "payments": self.payments,
            "total_amount": str(self.total_amount),
            "total_discount_captured": str(self.total_discount_captured),
            "recommended_payment_date": (
                self.recommended_payment_date.isoformat() if self.recommended_payment_date else None
            ),
        }


@dataclass
class MockBatchPayment:
    """Mock batch payment."""

    id: str = "batch-123"
    invoice_ids: list = field(default_factory=list)
    total_amount: Decimal = Decimal("10000.00")
    payment_date: datetime | None = None
    payment_method: str | None = None
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "invoice_ids": self.invoice_ids,
            "total_amount": str(self.total_amount),
            "payment_date": self.payment_date.isoformat() if self.payment_date else None,
            "payment_method": self.payment_method,
            "status": self.status,
        }


@dataclass
class MockForecast:
    """Mock cash flow forecast."""

    days_ahead: int = 30
    total_due: Decimal = Decimal("50000.00")
    daily_breakdown: list = field(default_factory=list)
    critical_dates: list = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "days_ahead": self.days_ahead,
            "total_due": str(self.total_due),
            "daily_breakdown": self.daily_breakdown,
            "critical_dates": self.critical_dates,
        }


class MockAPAutomation:
    """Mock AP automation service for testing."""

    def __init__(self):
        self.invoices: dict[str, MockAPInvoice] = {}

    async def add_invoice(
        self,
        vendor_id: str,
        vendor_name: str,
        invoice_number: str = "",
        invoice_date: datetime | None = None,
        due_date: datetime | None = None,
        total_amount: Decimal = Decimal("0.00"),
        payment_terms: str = "Net 30",
        early_pay_discount: float = 0,
        discount_deadline: datetime | None = None,
        priority: str | None = None,
        preferred_payment_method: str | None = None,
    ) -> MockAPInvoice:
        invoice = MockAPInvoice(
            id=f"ap-inv-{len(self.invoices) + 1}",
            vendor_id=vendor_id,
            vendor_name=vendor_name,
            invoice_number=invoice_number,
            invoice_date=invoice_date,
            due_date=due_date,
            total_amount=total_amount,
            balance_due=total_amount,
            payment_terms=payment_terms,
            early_pay_discount=early_pay_discount,
            discount_deadline=discount_deadline,
            priority=priority or "normal",
            preferred_payment_method=preferred_payment_method,
        )
        self.invoices[invoice.id] = invoice
        return invoice

    async def list_invoices(
        self,
        vendor_id: str | None = None,
        status: str | None = None,
        priority: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[MockAPInvoice]:
        invoices = list(self.invoices.values())
        if vendor_id:
            invoices = [inv for inv in invoices if inv.vendor_id == vendor_id]
        if status:
            invoices = [inv for inv in invoices if inv.status.value == status]
        if priority:
            invoices = [inv for inv in invoices if inv.priority == priority]
        return invoices

    async def get_invoice(self, invoice_id: str) -> MockAPInvoice | None:
        return self.invoices.get(invoice_id)

    async def record_payment(
        self,
        invoice_id: str,
        amount: Decimal,
        payment_date: datetime | None = None,
        payment_method: str | None = None,
        reference: str | None = None,
    ) -> MockAPInvoice | None:
        invoice = self.invoices.get(invoice_id)
        if invoice:
            invoice.amount_paid += amount
            invoice.balance_due -= amount
            if invoice.balance_due <= 0:
                invoice.status = MockInvoiceStatus.PAID
            elif invoice.amount_paid > 0:
                invoice.status = MockInvoiceStatus.PARTIAL
            return invoice
        return None

    async def optimize_payment_timing(
        self,
        invoices: list[MockAPInvoice],
        available_cash: Decimal | None = None,
        prioritize_discounts: bool = True,
    ) -> MockPaymentSchedule:
        return MockPaymentSchedule(
            payments=[{"invoice_id": inv.id, "amount": str(inv.balance_due)} for inv in invoices],
            total_amount=sum(inv.balance_due for inv in invoices),
        )

    async def batch_payments(
        self,
        invoices: list[MockAPInvoice],
        payment_date: datetime | None = None,
        payment_method: str | None = None,
    ) -> MockBatchPayment:
        return MockBatchPayment(
            invoice_ids=[inv.id for inv in invoices],
            total_amount=sum(inv.balance_due for inv in invoices),
            payment_date=payment_date,
            payment_method=payment_method,
        )

    async def forecast_cash_needs(self, days_ahead: int = 30) -> MockForecast:
        return MockForecast(days_ahead=days_ahead)

    async def get_discount_opportunities(self) -> list[dict[str, Any]]:
        return [
            {
                "invoice_id": "ap-inv-123",
                "vendor_name": "Vendor Corp",
                "discount_percent": 2.0,
                "discount_amount": "100.00",
                "deadline": "2024-02-01",
            }
        ]


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def mock_ap_service():
    """Create a mock AP automation service with test data."""
    service = MockAPAutomation()

    # Add test invoices
    service.invoices["ap-inv-123"] = MockAPInvoice(
        id="ap-inv-123",
        vendor_id="vendor-123",
        vendor_name="Vendor Corp",
        total_amount=Decimal("5000.00"),
        balance_due=Decimal("5000.00"),
    )
    service.invoices["ap-inv-456"] = MockAPInvoice(
        id="ap-inv-456",
        vendor_id="vendor-456",
        vendor_name="Supplies Inc",
        total_amount=Decimal("2500.00"),
        balance_due=Decimal("2500.00"),
        status=MockInvoiceStatus.PARTIAL,
    )

    return service


# ===========================================================================
# Helper Functions
# ===========================================================================


def parse_response(result) -> tuple[dict[str, Any], int]:
    """Parse HandlerResult into (body_dict, status_code)."""
    body = json.loads(result.body) if result.body else {}
    # Unwrap data if present (success_response format)
    if "data" in body and body.get("success"):
        return body["data"], result.status_code
    return body, result.status_code


# ===========================================================================
# Test: Add Invoice
# ===========================================================================


class TestAddInvoice:
    """Tests for POST /api/v1/accounting/ap/invoices."""

    @pytest.mark.asyncio
    async def test_add_invoice_success(self, mock_ap_service):
        """Should add a new payable invoice."""
        with patch.object(ap_automation, "get_ap_automation", return_value=mock_ap_service):
            result = await ap_automation.handle_add_invoice(
                data={
                    "vendor_id": "vendor-new",
                    "vendor_name": "New Vendor LLC",
                    "total_amount": 3000.00,
                    "payment_terms": "Net 15",
                },
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert "invoice" in body
            assert body["invoice"]["vendor_name"] == "New Vendor LLC"

    @pytest.mark.asyncio
    async def test_add_invoice_missing_vendor_id(self, mock_ap_service):
        """Should return 400 when vendor_id is missing."""
        with patch.object(ap_automation, "get_ap_automation", return_value=mock_ap_service):
            result = await ap_automation.handle_add_invoice(
                data={
                    "vendor_name": "New Vendor",
                    "total_amount": 1000.00,
                },
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 400
            assert "vendor_id" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_add_invoice_missing_amount(self, mock_ap_service):
        """Should return 400 when total_amount is missing."""
        with patch.object(ap_automation, "get_ap_automation", return_value=mock_ap_service):
            result = await ap_automation.handle_add_invoice(
                data={
                    "vendor_id": "vendor-123",
                    "vendor_name": "Vendor Corp",
                },
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 400
            assert "total_amount" in body.get("error", "").lower()


# ===========================================================================
# Test: List Invoices
# ===========================================================================


class TestListInvoices:
    """Tests for GET /api/v1/accounting/ap/invoices."""

    @pytest.mark.asyncio
    async def test_list_invoices_success(self, mock_ap_service):
        """Should return list of invoices."""
        with patch.object(ap_automation, "get_ap_automation", return_value=mock_ap_service):
            result = await ap_automation.handle_list_invoices(
                data={},
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert "invoices" in body
            assert body["total"] == 2

    @pytest.mark.asyncio
    async def test_list_invoices_with_vendor_filter(self, mock_ap_service):
        """Should filter invoices by vendor_id."""
        with patch.object(ap_automation, "get_ap_automation", return_value=mock_ap_service):
            result = await ap_automation.handle_list_invoices(
                data={"vendor_id": "vendor-123"},
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert body["total"] == 1


# ===========================================================================
# Test: Get Invoice
# ===========================================================================


class TestGetInvoice:
    """Tests for GET /api/v1/accounting/ap/invoices/{invoice_id}."""

    @pytest.mark.asyncio
    async def test_get_invoice_success(self, mock_ap_service):
        """Should return invoice by ID."""
        with patch.object(ap_automation, "get_ap_automation", return_value=mock_ap_service):
            result = await ap_automation.handle_get_invoice(
                data={},
                invoice_id="ap-inv-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert body["invoice"]["id"] == "ap-inv-123"

    @pytest.mark.asyncio
    async def test_get_invoice_not_found(self, mock_ap_service):
        """Should return 404 for non-existent invoice."""
        with patch.object(ap_automation, "get_ap_automation", return_value=mock_ap_service):
            result = await ap_automation.handle_get_invoice(
                data={},
                invoice_id="nonexistent",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 404


# ===========================================================================
# Test: Record Payment
# ===========================================================================


class TestRecordPayment:
    """Tests for POST /api/v1/accounting/ap/invoices/{invoice_id}/payment."""

    @pytest.mark.asyncio
    async def test_record_payment_success(self, mock_ap_service):
        """Should record payment successfully."""
        with patch.object(ap_automation, "get_ap_automation", return_value=mock_ap_service):
            result = await ap_automation.handle_record_payment(
                data={
                    "amount": 2500.00,
                    "payment_method": "ach",
                },
                invoice_id="ap-inv-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert "Payment" in body.get("message", "")

    @pytest.mark.asyncio
    async def test_record_payment_missing_amount(self, mock_ap_service):
        """Should return 400 when amount is missing."""
        with patch.object(ap_automation, "get_ap_automation", return_value=mock_ap_service):
            result = await ap_automation.handle_record_payment(
                data={"payment_method": "check"},
                invoice_id="ap-inv-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 400
            assert "amount" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_record_payment_not_found(self, mock_ap_service):
        """Should return 404 for non-existent invoice."""
        with patch.object(ap_automation, "get_ap_automation", return_value=mock_ap_service):
            result = await ap_automation.handle_record_payment(
                data={"amount": 1000.00},
                invoice_id="nonexistent",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 404


# ===========================================================================
# Test: Optimize Payments
# ===========================================================================


class TestOptimizePayments:
    """Tests for POST /api/v1/accounting/ap/optimize."""

    @pytest.mark.asyncio
    async def test_optimize_payments_success(self, mock_ap_service):
        """Should return optimized payment schedule."""
        with patch.object(ap_automation, "get_ap_automation", return_value=mock_ap_service):
            result = await ap_automation.handle_optimize_payments(
                data={
                    "invoice_ids": ["ap-inv-123", "ap-inv-456"],
                    "available_cash": 10000.00,
                },
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert "schedule" in body

    @pytest.mark.asyncio
    async def test_optimize_payments_no_invoices(self, mock_ap_service):
        """Should handle empty invoice list."""
        # Clear all invoices
        mock_ap_service.invoices.clear()

        with patch.object(ap_automation, "get_ap_automation", return_value=mock_ap_service):
            result = await ap_automation.handle_optimize_payments(
                data={},
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert "No invoices to optimize" in body.get("message", "")


# ===========================================================================
# Test: Batch Payments
# ===========================================================================


class TestBatchPayments:
    """Tests for POST /api/v1/accounting/ap/batch."""

    @pytest.mark.asyncio
    async def test_batch_payments_success(self, mock_ap_service):
        """Should create batch payment."""
        with patch.object(ap_automation, "get_ap_automation", return_value=mock_ap_service):
            result = await ap_automation.handle_batch_payments(
                data={
                    "invoice_ids": ["ap-inv-123", "ap-inv-456"],
                    "payment_method": "ach",
                },
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert "batch" in body

    @pytest.mark.asyncio
    async def test_batch_payments_missing_ids(self, mock_ap_service):
        """Should return 400 when invoice_ids is missing."""
        with patch.object(ap_automation, "get_ap_automation", return_value=mock_ap_service):
            result = await ap_automation.handle_batch_payments(
                data={"payment_method": "check"},
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 400
            assert "invoice_ids" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_batch_payments_invoice_not_found(self, mock_ap_service):
        """Should return 404 when invoice not found."""
        with patch.object(ap_automation, "get_ap_automation", return_value=mock_ap_service):
            result = await ap_automation.handle_batch_payments(
                data={"invoice_ids": ["ap-inv-123", "nonexistent"]},
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 404


# ===========================================================================
# Test: Get Forecast
# ===========================================================================


class TestGetForecast:
    """Tests for GET /api/v1/accounting/ap/forecast."""

    @pytest.mark.asyncio
    async def test_get_forecast_success(self, mock_ap_service):
        """Should return cash flow forecast."""
        with patch.object(ap_automation, "get_ap_automation", return_value=mock_ap_service):
            result = await ap_automation.handle_get_forecast(
                data={"days_ahead": 30},
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert "forecast" in body
            assert body["days_ahead"] == 30

    @pytest.mark.asyncio
    async def test_get_forecast_invalid_days(self, mock_ap_service):
        """Should return 400 for invalid days_ahead."""
        with patch.object(ap_automation, "get_ap_automation", return_value=mock_ap_service):
            result = await ap_automation.handle_get_forecast(
                data={"days_ahead": 500},
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 400
            assert "days_ahead" in body.get("error", "").lower()


# ===========================================================================
# Test: Get Discounts
# ===========================================================================


class TestGetDiscounts:
    """Tests for GET /api/v1/accounting/ap/discounts."""

    @pytest.mark.asyncio
    async def test_get_discounts_success(self, mock_ap_service):
        """Should return discount opportunities."""
        with patch.object(ap_automation, "get_ap_automation", return_value=mock_ap_service):
            result = await ap_automation.handle_get_discounts(
                data={},
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert "opportunities" in body
            assert body["total"] >= 1
