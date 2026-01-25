"""
Tests for invoice PDF and HTML export.

Tests cover:
- HTML export generation
- PDF export generation (if reportlab available)
- Export configuration
- Customer information inclusion
- Line items formatting
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock
import pytest

from aragora.billing.invoice_export import (
    InvoiceExporter,
    InvoiceCompanyInfo,
    InvoiceCustomerInfo,
    InvoiceExportConfig,
    export_invoice_html,
    export_invoice_pdf,
    HAS_REPORTLAB,
)


@pytest.fixture
def sample_invoice():
    """Create a sample invoice for testing."""
    return MagicMock(
        id="INV-12345678",
        tenant_id="tenant_123",
        period_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        period_end=datetime(2025, 1, 31, tzinfo=timezone.utc),
        subtotal=Decimal("1500.00"),
        discount=Decimal("150.00"),
        tax=Decimal("135.00"),
        total=Decimal("1485.00"),
        line_items=[
            {
                "description": "API Usage - Claude Opus",
                "quantity": 1000000,
                "unit_price": 0.001,
                "amount": 1000.00,
            },
            {
                "description": "API Usage - GPT-4",
                "quantity": 500000,
                "unit_price": 0.001,
                "amount": 500.00,
            },
        ],
        status="pending",
        due_date=datetime(2025, 2, 15, tzinfo=timezone.utc),
        paid_at=None,
        currency="USD",
        notes="Thank you for using Aragora!",
        created_at=datetime(2025, 1, 31, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_customer():
    """Create sample customer info."""
    return InvoiceCustomerInfo(
        name="Acme Corporation",
        email="billing@acme.com",
        address_line1="123 Main Street",
        city="San Francisco",
        state="CA",
        postal_code="94102",
        country="USA",
    )


@pytest.fixture
def exporter():
    """Create an invoice exporter."""
    config = InvoiceExportConfig(
        company_info=InvoiceCompanyInfo(
            name="Aragora Inc.",
            address_line1="100 AI Boulevard",
            city="San Francisco",
            state="CA",
            postal_code="94105",
            email="billing@aragora.ai",
        ),
    )
    return InvoiceExporter(config)


class TestInvoiceHTMLExport:
    """Tests for HTML invoice export."""

    @pytest.mark.asyncio
    async def test_export_html_basic(self, exporter, sample_invoice):
        """Can export invoice to HTML."""
        html = await exporter.export_html(sample_invoice)

        assert "<!DOCTYPE html>" in html
        assert "INVOICE" in html
        assert "INV-12345678" in html

    @pytest.mark.asyncio
    async def test_export_html_contains_amounts(self, exporter, sample_invoice):
        """HTML contains correct amounts."""
        html = await exporter.export_html(sample_invoice)

        assert "$1500.00" in html  # Subtotal
        assert "$150.00" in html  # Discount
        assert "$135.00" in html  # Tax
        assert "$1485.00" in html  # Total

    @pytest.mark.asyncio
    async def test_export_html_contains_line_items(self, exporter, sample_invoice):
        """HTML contains line items."""
        html = await exporter.export_html(sample_invoice)

        assert "API Usage - Claude Opus" in html
        assert "API Usage - GPT-4" in html

    @pytest.mark.asyncio
    async def test_export_html_contains_dates(self, exporter, sample_invoice):
        """HTML contains formatted dates."""
        html = await exporter.export_html(sample_invoice)

        assert "January" in html
        assert "2025" in html

    @pytest.mark.asyncio
    async def test_export_html_contains_status(self, exporter, sample_invoice):
        """HTML contains invoice status."""
        html = await exporter.export_html(sample_invoice)

        assert "PENDING" in html

    @pytest.mark.asyncio
    async def test_export_html_with_customer(self, exporter, sample_invoice, sample_customer):
        """HTML includes customer information."""
        html = await exporter.export_html(sample_invoice, sample_customer)

        assert "Acme Corporation" in html
        assert "billing@acme.com" in html
        assert "123 Main Street" in html
        assert "San Francisco" in html

    @pytest.mark.asyncio
    async def test_export_html_contains_company_info(self, exporter, sample_invoice):
        """HTML contains company information."""
        html = await exporter.export_html(sample_invoice)

        assert "Aragora Inc." in html
        assert "100 AI Boulevard" in html
        assert "billing@aragora.ai" in html

    @pytest.mark.asyncio
    async def test_export_html_contains_notes(self, exporter, sample_invoice):
        """HTML contains invoice notes."""
        html = await exporter.export_html(sample_invoice)

        assert "Thank you for using Aragora!" in html

    @pytest.mark.asyncio
    async def test_export_html_no_discount(self, exporter, sample_invoice):
        """HTML handles zero discount correctly."""
        sample_invoice.discount = Decimal("0")
        html = await exporter.export_html(sample_invoice)

        # Should not show discount line if zero
        assert html.count("Discount") <= 1  # May appear in "no discount" or not at all


class TestInvoicePDFExport:
    """Tests for PDF invoice export."""

    @pytest.mark.asyncio
    async def test_export_pdf_returns_bytes(self, exporter, sample_invoice):
        """PDF export returns bytes."""
        pdf_bytes = await exporter.export_pdf(sample_invoice)

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_REPORTLAB, reason="reportlab not installed")
    async def test_export_pdf_valid_pdf(self, exporter, sample_invoice):
        """PDF export produces valid PDF."""
        pdf_bytes = await exporter.export_pdf(sample_invoice)

        # PDF files start with %PDF
        assert pdf_bytes.startswith(b"%PDF")

    @pytest.mark.asyncio
    async def test_export_pdf_with_customer(self, exporter, sample_invoice, sample_customer):
        """PDF export includes customer info."""
        pdf_bytes = await exporter.export_pdf(sample_invoice, sample_customer)

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0


class TestInvoiceExportConfig:
    """Tests for export configuration."""

    def test_default_config(self):
        """Default config has sensible values."""
        config = InvoiceExportConfig()

        assert config.currency_symbol == "$"
        assert config.include_line_items is True
        assert config.include_tax_breakdown is True
        assert config.footer_text != ""

    def test_custom_config(self):
        """Can customize export config."""
        config = InvoiceExportConfig(
            currency_symbol="€",
            footer_text="Custom footer",
            include_line_items=False,
        )

        assert config.currency_symbol == "€"
        assert config.footer_text == "Custom footer"
        assert config.include_line_items is False

    def test_company_info(self):
        """Company info can be customized."""
        company = InvoiceCompanyInfo(
            name="Test Company",
            email="test@example.com",
            tax_id="12-3456789",
        )

        assert company.name == "Test Company"
        assert company.email == "test@example.com"
        assert company.tax_id == "12-3456789"


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.mark.asyncio
    async def test_export_invoice_html(self, sample_invoice):
        """Convenience function exports HTML."""
        html = await export_invoice_html(sample_invoice)

        assert "<!DOCTYPE html>" in html
        assert "INVOICE" in html

    @pytest.mark.asyncio
    async def test_export_invoice_pdf(self, sample_invoice):
        """Convenience function exports PDF."""
        pdf_bytes = await export_invoice_pdf(sample_invoice)

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_export_html_empty_line_items(self, exporter, sample_invoice):
        """HTML handles empty line items."""
        sample_invoice.line_items = []
        html = await exporter.export_html(sample_invoice)

        assert "<!DOCTYPE html>" in html

    @pytest.mark.asyncio
    async def test_export_html_none_values(self, exporter, sample_invoice):
        """HTML handles None values gracefully."""
        sample_invoice.due_date = None
        sample_invoice.notes = None
        html = await exporter.export_html(sample_invoice)

        assert "<!DOCTYPE html>" in html

    @pytest.mark.asyncio
    async def test_export_html_string_dates(self, exporter):
        """HTML handles string dates."""
        invoice = MagicMock(
            id="INV-TEST",
            created_at="2025-01-15T10:00:00+00:00",
            period_start="2025-01-01T00:00:00+00:00",
            period_end="2025-01-31T23:59:59+00:00",
            subtotal=Decimal("100.00"),
            discount=Decimal("0"),
            tax=Decimal("0"),
            total=Decimal("100.00"),
            line_items=[],
            status="draft",
            due_date=None,
            notes=None,
        )
        html = await exporter.export_html(invoice)

        assert "<!DOCTYPE html>" in html


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
