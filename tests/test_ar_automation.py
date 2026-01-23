"""
Tests for the ARAutomation service.

Covers:
- Invoice generation
- Payment reminders
- AR aging reports
- Collection suggestions
- Customer management
"""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from aragora.services.ar_automation import (
    ARAutomation,
    ARInvoice,
    InvoiceStatus,
    ReminderLevel,
    CollectionAction,
    AgingReport,
    CollectionSuggestion,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ar_automation():
    """Create a fresh ARAutomation instance."""
    return ARAutomation()


@pytest.fixture
def sample_customer_data():
    """Sample customer data."""
    return {
        "customer_id": "cust_001",
        "name": "Acme Corporation",
        "email": "billing@acme.com",
        "payment_terms": "Net 30",
    }


@pytest.fixture
def sample_line_items():
    """Sample invoice line items."""
    return [
        {
            "description": "Consulting Services",
            "quantity": 10,
            "unit_price": 150.00,
            "amount": 1500.00,
        },
        {"description": "Travel Expenses", "quantity": 1, "unit_price": 500.00, "amount": 500.00},
    ]


# =============================================================================
# Invoice Generation Tests
# =============================================================================


class TestInvoiceGeneration:
    """Test AR invoice generation."""

    @pytest.mark.asyncio
    async def test_generate_invoice(self, ar_automation, sample_line_items):
        """Test generating an invoice."""
        invoice = await ar_automation.generate_invoice(
            customer_id="cust_001",
            customer_name="Acme Corporation",
            customer_email="billing@acme.com",
            line_items=sample_line_items,
            payment_terms="Net 30",
        )

        assert invoice.id.startswith("ar_")
        assert invoice.customer_id == "cust_001"
        assert invoice.status == InvoiceStatus.DRAFT
        assert float(invoice.total_amount) == 2000.00

    @pytest.mark.asyncio
    async def test_generate_invoice_with_tax(self, ar_automation, sample_line_items):
        """Test generating an invoice with tax."""
        invoice = await ar_automation.generate_invoice(
            customer_id="cust_001",
            customer_name="Acme Corporation",
            line_items=sample_line_items,
            tax_rate=0.08,
        )

        assert float(invoice.tax_amount) == 160.00  # 8% of 2000
        assert float(invoice.total_amount) == 2160.00

    @pytest.mark.asyncio
    async def test_generate_invoice_number(self, ar_automation, sample_line_items):
        """Test that invoice numbers are unique and sequential."""
        inv1 = await ar_automation.generate_invoice(
            customer_id="cust_001",
            customer_name="Customer 1",
            line_items=sample_line_items,
        )
        inv2 = await ar_automation.generate_invoice(
            customer_id="cust_002",
            customer_name="Customer 2",
            line_items=sample_line_items,
        )

        assert inv1.invoice_number != inv2.invoice_number


# =============================================================================
# Invoice Sending Tests
# =============================================================================


class TestInvoiceSending:
    """Test invoice sending."""

    @pytest.mark.asyncio
    async def test_send_invoice(self, ar_automation, sample_line_items):
        """Test sending an invoice."""
        invoice = await ar_automation.generate_invoice(
            customer_id="cust_001",
            customer_name="Acme Corp",
            customer_email="billing@acme.com",
            line_items=sample_line_items,
        )

        success = await ar_automation.send_invoice(invoice.id)

        assert success is True

        # Check invoice status updated
        updated = await ar_automation.get_invoice(invoice.id)
        assert updated.status == InvoiceStatus.SENT
        assert updated.sent_at is not None

    @pytest.mark.asyncio
    async def test_send_nonexistent_invoice(self, ar_automation):
        """Test sending a non-existent invoice."""
        success = await ar_automation.send_invoice("ar_nonexistent")
        assert success is False


# =============================================================================
# Payment Reminder Tests
# =============================================================================


class TestPaymentReminders:
    """Test payment reminder functionality."""

    @pytest.mark.asyncio
    async def test_send_friendly_reminder(self, ar_automation, sample_line_items):
        """Test sending a friendly reminder."""
        invoice = await ar_automation.generate_invoice(
            customer_id="cust_001",
            customer_name="Acme Corp",
            customer_email="billing@acme.com",
            line_items=sample_line_items,
        )
        await ar_automation.send_invoice(invoice.id)

        success = await ar_automation.send_payment_reminder(
            invoice_id=invoice.id,
            escalation_level=1,
        )

        assert success is True

        updated = await ar_automation.get_invoice(invoice.id)
        assert updated.reminder_count == 1

    @pytest.mark.asyncio
    async def test_reminder_escalation(self, ar_automation, sample_line_items):
        """Test reminder escalation levels."""
        invoice = await ar_automation.generate_invoice(
            customer_id="cust_001",
            customer_name="Acme Corp",
            customer_email="billing@acme.com",
            line_items=sample_line_items,
        )
        await ar_automation.send_invoice(invoice.id)

        # Send reminders at increasing escalation levels
        for level in range(1, 5):
            await ar_automation.send_payment_reminder(
                invoice_id=invoice.id,
                escalation_level=level,
            )

        updated = await ar_automation.get_invoice(invoice.id)
        assert updated.reminder_count == 4


# =============================================================================
# Payment Recording Tests
# =============================================================================


class TestPaymentRecording:
    """Test payment recording."""

    @pytest.mark.asyncio
    async def test_record_full_payment(self, ar_automation, sample_line_items):
        """Test recording a full payment."""
        invoice = await ar_automation.generate_invoice(
            customer_id="cust_001",
            customer_name="Acme Corp",
            line_items=sample_line_items,
        )

        updated = await ar_automation.record_payment(
            invoice_id=invoice.id,
            amount=Decimal("2000.00"),
        )

        assert updated.status == InvoiceStatus.PAID
        assert updated.balance == Decimal("0.00")

    @pytest.mark.asyncio
    async def test_record_partial_payment(self, ar_automation, sample_line_items):
        """Test recording a partial payment."""
        invoice = await ar_automation.generate_invoice(
            customer_id="cust_001",
            customer_name="Acme Corp",
            line_items=sample_line_items,
        )

        updated = await ar_automation.record_payment(
            invoice_id=invoice.id,
            amount=Decimal("1000.00"),
        )

        assert updated.status == InvoiceStatus.PARTIAL
        assert updated.balance == Decimal("1000.00")

    @pytest.mark.asyncio
    async def test_record_overpayment(self, ar_automation, sample_line_items):
        """Test that overpayment is capped at balance."""
        invoice = await ar_automation.generate_invoice(
            customer_id="cust_001",
            customer_name="Acme Corp",
            line_items=sample_line_items,
        )

        updated = await ar_automation.record_payment(
            invoice_id=invoice.id,
            amount=Decimal("5000.00"),  # More than invoice total
        )

        assert updated.status == InvoiceStatus.PAID
        assert updated.amount_paid == invoice.total_amount


# =============================================================================
# AR Aging Tests
# =============================================================================


class TestARAging:
    """Test AR aging report."""

    @pytest.mark.asyncio
    async def test_track_aging_empty(self, ar_automation):
        """Test aging report with no invoices."""
        report = await ar_automation.track_aging()

        assert isinstance(report, AgingReport)
        assert report.total_outstanding == Decimal("0.00")

    @pytest.mark.asyncio
    async def test_track_aging_current(self, ar_automation, sample_line_items):
        """Test aging report with current invoices."""
        await ar_automation.generate_invoice(
            customer_id="cust_001",
            customer_name="Acme Corp",
            line_items=sample_line_items,
        )

        report = await ar_automation.track_aging()

        assert report.current > Decimal("0.00")
        assert report.days_1_30 == Decimal("0.00")

    @pytest.mark.asyncio
    async def test_track_aging_overdue(self, ar_automation):
        """Test aging report with overdue invoices."""
        # Create invoice with past due date
        invoice = ARInvoice(
            id="ar_old",
            customer_id="cust_001",
            customer_name="Slow Payer",
            invoice_date=datetime.now() - timedelta(days=60),
            due_date=datetime.now() - timedelta(days=30),
            total_amount=Decimal("1000.00"),
            balance=Decimal("1000.00"),
            status=InvoiceStatus.SENT,
        )
        ar_automation._invoices[invoice.id] = invoice

        report = await ar_automation.track_aging()

        assert report.days_1_30 > Decimal("0.00") or report.days_31_60 > Decimal("0.00")


# =============================================================================
# Collection Suggestions Tests
# =============================================================================


class TestCollectionSuggestions:
    """Test collection action suggestions."""

    @pytest.mark.asyncio
    async def test_suggest_no_collections_needed(self, ar_automation, sample_line_items):
        """Test no suggestions for current invoices."""
        await ar_automation.generate_invoice(
            customer_id="cust_001",
            customer_name="Good Payer",
            line_items=sample_line_items,
        )

        suggestions = await ar_automation.suggest_collections()

        # Current invoices shouldn't need collection actions
        assert len(suggestions) == 0

    @pytest.mark.asyncio
    async def test_suggest_reminder_for_overdue(self, ar_automation):
        """Test reminder suggestion for overdue invoice."""
        invoice = ARInvoice(
            id="ar_overdue",
            customer_id="cust_001",
            customer_name="Late Payer",
            invoice_date=datetime.now() - timedelta(days=45),
            due_date=datetime.now() - timedelta(days=15),
            total_amount=Decimal("1000.00"),
            balance=Decimal("1000.00"),
            status=InvoiceStatus.OVERDUE,
            reminder_count=0,
        )
        ar_automation._invoices[invoice.id] = invoice

        suggestions = await ar_automation.suggest_collections()

        assert len(suggestions) >= 1
        assert any(s.action == CollectionAction.SEND_REMINDER for s in suggestions)

    @pytest.mark.asyncio
    async def test_suggest_escalation_for_very_overdue(self, ar_automation):
        """Test escalated actions for very overdue invoices."""
        invoice = ARInvoice(
            id="ar_very_overdue",
            customer_id="cust_001",
            customer_name="Non Payer",
            invoice_date=datetime.now() - timedelta(days=120),
            due_date=datetime.now() - timedelta(days=90),
            total_amount=Decimal("5000.00"),
            balance=Decimal("5000.00"),
            status=InvoiceStatus.OVERDUE,
            reminder_count=4,  # All reminders sent
        )
        ar_automation._invoices[invoice.id] = invoice

        suggestions = await ar_automation.suggest_collections()

        # Should suggest more aggressive action
        assert len(suggestions) >= 1
        actions = [s.action for s in suggestions]
        assert any(
            a
            in [
                CollectionAction.PHONE_CALL,
                CollectionAction.COLLECTION_AGENCY,
                CollectionAction.LEGAL_ACTION,
            ]
            for a in actions
        )


# =============================================================================
# Customer Management Tests
# =============================================================================


class TestCustomerManagement:
    """Test customer management."""

    @pytest.mark.asyncio
    async def test_add_customer(self, ar_automation, sample_customer_data):
        """Test adding a customer."""
        await ar_automation.add_customer(**sample_customer_data)

        # Verify customer was added by checking balance
        balance = await ar_automation.get_customer_balance(sample_customer_data["customer_id"])
        assert balance == Decimal("0.00")

    @pytest.mark.asyncio
    async def test_get_customer_balance(self, ar_automation, sample_line_items):
        """Test getting customer balance."""
        await ar_automation.add_customer(
            customer_id="cust_001",
            name="Acme Corp",
        )

        # Create unpaid invoice
        await ar_automation.generate_invoice(
            customer_id="cust_001",
            customer_name="Acme Corp",
            line_items=sample_line_items,
        )

        balance = await ar_automation.get_customer_balance("cust_001")
        assert balance == Decimal("2000.00")

    @pytest.mark.asyncio
    async def test_customer_balance_after_payment(self, ar_automation, sample_line_items):
        """Test customer balance after partial payment."""
        invoice = await ar_automation.generate_invoice(
            customer_id="cust_001",
            customer_name="Acme Corp",
            line_items=sample_line_items,
        )

        await ar_automation.record_payment(
            invoice_id=invoice.id,
            amount=Decimal("500.00"),
        )

        balance = await ar_automation.get_customer_balance("cust_001")
        # Invoice is $2000 + 8.25% tax = $2165, minus $500 payment = $1665
        assert balance == Decimal("1665.00")


# =============================================================================
# Invoice Listing Tests
# =============================================================================


class TestInvoiceListing:
    """Test invoice listing and filtering."""

    @pytest.mark.asyncio
    async def test_list_all_invoices(self, ar_automation, sample_line_items):
        """Test listing all invoices."""
        for i in range(3):
            await ar_automation.generate_invoice(
                customer_id=f"cust_{i}",
                customer_name=f"Customer {i}",
                line_items=sample_line_items,
            )

        invoices = await ar_automation.list_invoices()
        assert len(invoices) == 3

    @pytest.mark.asyncio
    async def test_list_by_customer(self, ar_automation, sample_line_items):
        """Test listing invoices by customer."""
        await ar_automation.generate_invoice(
            customer_id="cust_001",
            customer_name="Customer A",
            line_items=sample_line_items,
        )
        await ar_automation.generate_invoice(
            customer_id="cust_002",
            customer_name="Customer B",
            line_items=sample_line_items,
        )

        invoices = await ar_automation.list_invoices(customer_id="cust_001")
        assert len(invoices) == 1
        assert invoices[0].customer_id == "cust_001"

    @pytest.mark.asyncio
    async def test_list_by_status(self, ar_automation, sample_line_items):
        """Test listing invoices by status."""
        inv1 = await ar_automation.generate_invoice(
            customer_id="cust_001",
            customer_name="Customer A",
            customer_email="a@example.com",  # Email required for send to work
            line_items=sample_line_items,
        )
        await ar_automation.send_invoice(inv1.id)

        await ar_automation.generate_invoice(
            customer_id="cust_002",
            customer_name="Customer B",
            line_items=sample_line_items,
        )

        sent_invoices = await ar_automation.list_invoices(status=InvoiceStatus.SENT)
        draft_invoices = await ar_automation.list_invoices(status=InvoiceStatus.DRAFT)

        assert len(sent_invoices) == 1
        assert len(draft_invoices) == 1


# =============================================================================
# ARInvoice Tests
# =============================================================================


class TestARInvoice:
    """Test ARInvoice dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        invoice = ARInvoice(
            id="ar_123",
            customer_id="cust_001",
            customer_name="Test Customer",
            invoice_number="INV-001",
            total_amount=Decimal("1000.00"),
            balance=Decimal("1000.00"),
            status=InvoiceStatus.DRAFT,
        )

        d = invoice.to_dict()

        assert d["id"] == "ar_123"
        # Uses camelCase keys and float values
        assert d["customerName"] == "Test Customer"
        assert d["totalAmount"] == 1000.0
        assert d["status"] == "draft"

    def test_days_overdue_calculation(self):
        """Test days overdue calculation."""
        invoice = ARInvoice(
            id="ar_123",
            customer_id="cust_001",
            customer_name="Test",
            due_date=datetime.now() - timedelta(days=10),
            total_amount=Decimal("100.00"),
            balance=Decimal("100.00"),
            status=InvoiceStatus.OVERDUE,
        )

        days = invoice.days_overdue
        assert days is not None
        assert 9 <= days <= 11
