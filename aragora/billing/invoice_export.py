"""
Invoice Export Module for Aragora Billing.

Provides PDF and HTML export functionality for invoices.

Usage:
    from aragora.billing.invoice_export import InvoiceExporter

    exporter = InvoiceExporter()

    # Export to PDF bytes
    pdf_bytes = await exporter.export_pdf(invoice)

    # Export to HTML string
    html = await exporter.export_html(invoice)

    # Save PDF to file
    await exporter.save_pdf(invoice, "/path/to/invoice.pdf")
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Try to import PDF generation library
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    logger.debug("reportlab not installed - PDF export will use HTML fallback")


@dataclass
class InvoiceCompanyInfo:
    """Company information for invoice header."""

    name: str = "Aragora Inc."
    address_line1: str = ""
    address_line2: str = ""
    city: str = ""
    state: str = ""
    postal_code: str = ""
    country: str = ""
    email: str = ""
    phone: str = ""
    tax_id: str = ""
    logo_path: Optional[str] = None


@dataclass
class InvoiceCustomerInfo:
    """Customer information for invoice."""

    name: str = ""
    email: str = ""
    address_line1: str = ""
    address_line2: str = ""
    city: str = ""
    state: str = ""
    postal_code: str = ""
    country: str = ""
    tax_id: str = ""


@dataclass
class InvoiceExportConfig:
    """Configuration for invoice export."""

    company_info: InvoiceCompanyInfo = field(default_factory=InvoiceCompanyInfo)
    include_logo: bool = True
    include_line_items: bool = True
    include_tax_breakdown: bool = True
    currency_symbol: str = "$"
    date_format: str = "%B %d, %Y"
    footer_text: str = "Thank you for your business!"


class InvoiceExporter:
    """
    Exports invoices to PDF and HTML formats.

    Supports:
    - PDF generation via ReportLab (optional dependency)
    - HTML export for web display or browser printing
    - Customizable company/customer information
    - Line item details with subtotals
    """

    def __init__(self, config: Optional[InvoiceExportConfig] = None):
        """
        Initialize invoice exporter.

        Args:
            config: Export configuration
        """
        self.config = config or InvoiceExportConfig()

    async def export_pdf(
        self,
        invoice: Any,
        customer_info: Optional[InvoiceCustomerInfo] = None,
    ) -> bytes:
        """
        Export invoice to PDF bytes.

        Args:
            invoice: Invoice object to export
            customer_info: Optional customer information

        Returns:
            PDF document as bytes

        Raises:
            ImportError: If reportlab is not installed
        """
        if not HAS_REPORTLAB:
            # Fallback to HTML which can be printed to PDF
            html = await self.export_html(invoice, customer_info)
            return html.encode("utf-8")

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        elements = []
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            "InvoiceTitle",
            parent=styles["Heading1"],
            fontSize=24,
            spaceAfter=20,
        )
        header_style = ParagraphStyle(
            "HeaderText",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.grey,
        )
        body_style = ParagraphStyle(
            "BodyText",
            parent=styles["Normal"],
            fontSize=10,
        )

        # Header
        elements.append(Paragraph("INVOICE", title_style))
        elements.append(Spacer(1, 0.25 * inch))

        # Company info
        company = self.config.company_info
        company_text = f"""
        <b>{company.name}</b><br/>
        {company.address_line1 or ""}<br/>
        {company.city}, {company.state} {company.postal_code}<br/>
        {company.email or ""}<br/>
        {f"Tax ID: {company.tax_id}" if company.tax_id else ""}
        """
        elements.append(Paragraph(company_text.strip(), header_style))
        elements.append(Spacer(1, 0.25 * inch))

        # Invoice details table
        invoice_data = [
            ["Invoice Number:", str(getattr(invoice, "id", "N/A"))],
            [
                "Invoice Date:",
                self._format_date(getattr(invoice, "created_at", datetime.now())),
            ],
            [
                "Period:",
                f"{self._format_date(getattr(invoice, 'period_start', datetime.now()))} - "
                f"{self._format_date(getattr(invoice, 'period_end', datetime.now()))}",
            ],
            ["Status:", str(getattr(invoice, "status", "draft")).upper()],
        ]

        if due_date := getattr(invoice, "due_date", None):
            invoice_data.append(["Due Date:", self._format_date(due_date)])

        details_table = Table(invoice_data, colWidths=[1.5 * inch, 3 * inch])
        details_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        elements.append(details_table)
        elements.append(Spacer(1, 0.5 * inch))

        # Customer info
        if customer_info:
            customer_text = f"""
            <b>Bill To:</b><br/>
            {customer_info.name}<br/>
            {customer_info.email}<br/>
            {customer_info.address_line1 or ""}<br/>
            {customer_info.city}, {customer_info.state} {customer_info.postal_code}
            """
            elements.append(Paragraph(customer_text.strip(), body_style))
            elements.append(Spacer(1, 0.25 * inch))

        # Line items table
        if self.config.include_line_items:
            line_items = getattr(invoice, "line_items", []) or []
            if line_items:
                header_row = ["Description", "Quantity", "Unit Price", "Amount"]
                data = [header_row]

                for item in line_items:
                    if isinstance(item, dict):
                        data.append(
                            [
                                item.get("description", ""),
                                str(item.get("quantity", 1)),
                                f"{self.config.currency_symbol}{item.get('unit_price', 0):.2f}",
                                f"{self.config.currency_symbol}{item.get('amount', 0):.2f}",
                            ]
                        )

                items_table = Table(
                    data,
                    colWidths=[3.5 * inch, 1 * inch, 1.25 * inch, 1.25 * inch],
                )
                items_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                            ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ]
                    )
                )
                elements.append(items_table)
                elements.append(Spacer(1, 0.25 * inch))

        # Totals
        subtotal = Decimal(str(getattr(invoice, "subtotal", 0)))
        discount = Decimal(str(getattr(invoice, "discount", 0)))
        tax = Decimal(str(getattr(invoice, "tax", 0)))
        total = Decimal(str(getattr(invoice, "total", 0)))

        totals_data = [
            ["Subtotal:", f"{self.config.currency_symbol}{subtotal:.2f}"],
        ]

        if discount > 0:
            totals_data.append(["Discount:", f"-{self.config.currency_symbol}{discount:.2f}"])

        if tax > 0:
            totals_data.append(["Tax:", f"{self.config.currency_symbol}{tax:.2f}"])

        totals_data.append(["Total:", f"{self.config.currency_symbol}{total:.2f}"])

        totals_table = Table(totals_data, colWidths=[5.5 * inch, 1.5 * inch])
        totals_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "RIGHT"),
                    ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("LINEABOVE", (0, -1), (-1, -1), 1, colors.black),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        elements.append(totals_table)
        elements.append(Spacer(1, 0.5 * inch))

        # Footer
        if self.config.footer_text:
            elements.append(Paragraph(self.config.footer_text, header_style))

        # Notes
        if notes := getattr(invoice, "notes", None):
            elements.append(Spacer(1, 0.25 * inch))
            elements.append(Paragraph(f"<b>Notes:</b> {notes}", body_style))

        doc.build(elements)
        return buffer.getvalue()

    async def export_html(
        self,
        invoice: Any,
        customer_info: Optional[InvoiceCustomerInfo] = None,
    ) -> str:
        """
        Export invoice to HTML string.

        Args:
            invoice: Invoice object to export
            customer_info: Optional customer information

        Returns:
            HTML document as string
        """
        company = self.config.company_info
        subtotal = Decimal(str(getattr(invoice, "subtotal", 0)))
        discount = Decimal(str(getattr(invoice, "discount", 0)))
        tax = Decimal(str(getattr(invoice, "tax", 0)))
        total = Decimal(str(getattr(invoice, "total", 0)))

        # Build line items HTML
        line_items_html = ""
        if self.config.include_line_items:
            line_items = getattr(invoice, "line_items", []) or []
            if line_items:
                rows = []
                for item in line_items:
                    if isinstance(item, dict):
                        rows.append(
                            f"""
                            <tr>
                                <td>{item.get("description", "")}</td>
                                <td class="right">{item.get("quantity", 1)}</td>
                                <td class="right">{self.config.currency_symbol}{item.get("unit_price", 0):.2f}</td>
                                <td class="right">{self.config.currency_symbol}{item.get("amount", 0):.2f}</td>
                            </tr>
                            """
                        )
                line_items_html = f"""
                <table class="line-items">
                    <thead>
                        <tr>
                            <th>Description</th>
                            <th class="right">Quantity</th>
                            <th class="right">Unit Price</th>
                            <th class="right">Amount</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(rows)}
                    </tbody>
                </table>
                """

        # Build customer info HTML
        customer_html = ""
        if customer_info:
            customer_html = f"""
            <div class="customer-info">
                <h3>Bill To:</h3>
                <p>
                    {customer_info.name}<br>
                    {customer_info.email}<br>
                    {customer_info.address_line1 or ""}<br>
                    {customer_info.city}, {customer_info.state} {customer_info.postal_code}
                </p>
            </div>
            """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Invoice {getattr(invoice, "id", "N/A")}</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 40px;
                    color: #333;
                }}
                h1 {{
                    color: #1a1a1a;
                    margin-bottom: 30px;
                }}
                .header {{
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 40px;
                }}
                .company-info {{
                    color: #666;
                    font-size: 14px;
                }}
                .invoice-details {{
                    text-align: right;
                }}
                .invoice-details dt {{
                    font-weight: bold;
                    display: inline;
                }}
                .invoice-details dd {{
                    display: inline;
                    margin: 0 0 0 10px;
                }}
                .customer-info {{
                    margin: 30px 0;
                    padding: 20px;
                    background: #f5f5f5;
                    border-radius: 8px;
                }}
                .customer-info h3 {{
                    margin-top: 0;
                    color: #666;
                    font-size: 14px;
                }}
                .line-items {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 30px 0;
                }}
                .line-items th, .line-items td {{
                    padding: 12px;
                    border-bottom: 1px solid #ddd;
                    text-align: left;
                }}
                .line-items th {{
                    background: #f5f5f5;
                    font-weight: 600;
                }}
                .right {{
                    text-align: right;
                }}
                .totals {{
                    margin-top: 30px;
                    text-align: right;
                }}
                .totals table {{
                    margin-left: auto;
                }}
                .totals td {{
                    padding: 8px 15px;
                }}
                .totals .total-row {{
                    font-weight: bold;
                    font-size: 18px;
                    border-top: 2px solid #333;
                }}
                .footer {{
                    margin-top: 50px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                    color: #666;
                    font-size: 14px;
                }}
                .status {{
                    display: inline-block;
                    padding: 4px 12px;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: 600;
                    text-transform: uppercase;
                }}
                .status-draft {{ background: #ffeaa7; color: #d63031; }}
                .status-pending {{ background: #74b9ff; color: #0984e3; }}
                .status-paid {{ background: #55efc4; color: #00b894; }}
                .status-overdue {{ background: #fab1a0; color: #d63031; }}
                @media print {{
                    body {{ padding: 20px; }}
                    .no-print {{ display: none; }}
                }}
            </style>
        </head>
        <body>
            <h1>INVOICE</h1>

            <div class="header">
                <div class="company-info">
                    <strong>{company.name}</strong><br>
                    {company.address_line1 or ""}<br>
                    {company.city}, {company.state} {company.postal_code}<br>
                    {company.email or ""}<br>
                    {f"Tax ID: {company.tax_id}" if company.tax_id else ""}
                </div>
                <div class="invoice-details">
                    <p><dt>Invoice #:</dt><dd>{getattr(invoice, "id", "N/A")}</dd></p>
                    <p><dt>Date:</dt><dd>{self._format_date(getattr(invoice, "created_at", datetime.now()))}</dd></p>
                    <p><dt>Period:</dt><dd>{self._format_date(getattr(invoice, "period_start", datetime.now()))} - {self._format_date(getattr(invoice, "period_end", datetime.now()))}</dd></p>
                    <p><dt>Status:</dt><dd><span class="status status-{str(getattr(invoice, "status", "draft")).lower()}">{str(getattr(invoice, "status", "draft")).upper()}</span></dd></p>
                    {f"<p><dt>Due:</dt><dd>{self._format_date(getattr(invoice, 'due_date', None))}</dd></p>" if getattr(invoice, "due_date", None) else ""}
                </div>
            </div>

            {customer_html}

            {line_items_html}

            <div class="totals">
                <table>
                    <tr>
                        <td>Subtotal:</td>
                        <td class="right">{self.config.currency_symbol}{subtotal:.2f}</td>
                    </tr>
                    {'<tr><td>Discount:</td><td class="right">-' + self.config.currency_symbol + f"{discount:.2f}</td></tr>" if discount > 0 else ""}
                    {'<tr><td>Tax:</td><td class="right">' + self.config.currency_symbol + f"{tax:.2f}</td></tr>" if tax > 0 else ""}
                    <tr class="total-row">
                        <td>Total:</td>
                        <td class="right">{self.config.currency_symbol}{total:.2f}</td>
                    </tr>
                </table>
            </div>

            <div class="footer">
                {self.config.footer_text or ""}
                {f"<p><strong>Notes:</strong> {getattr(invoice, 'notes', '')}</p>" if getattr(invoice, "notes", None) else ""}
            </div>
        </body>
        </html>
        """

        return html

    async def save_pdf(
        self,
        invoice: Any,
        output_path: str,
        customer_info: Optional[InvoiceCustomerInfo] = None,
    ) -> str:
        """
        Save invoice as PDF file.

        Args:
            invoice: Invoice object to export
            output_path: File path for output PDF
            customer_info: Optional customer information

        Returns:
            Path to saved file
        """
        pdf_bytes = await self.export_pdf(invoice, customer_info)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pdf_bytes)
        logger.info(f"Invoice saved to {output_path}")
        return str(path)

    def _format_date(self, dt: Optional[datetime]) -> str:
        """Format a datetime using the configured format."""
        if dt is None:
            return "N/A"
        if isinstance(dt, str):
            try:
                dt = datetime.fromisoformat(dt)
            except ValueError:
                return dt
        return dt.strftime(self.config.date_format)


# Convenience functions
async def export_invoice_pdf(
    invoice: Any,
    customer_info: Optional[InvoiceCustomerInfo] = None,
    config: Optional[InvoiceExportConfig] = None,
) -> bytes:
    """
    Export an invoice to PDF bytes.

    Args:
        invoice: Invoice object
        customer_info: Optional customer information
        config: Optional export configuration

    Returns:
        PDF document as bytes
    """
    exporter = InvoiceExporter(config)
    return await exporter.export_pdf(invoice, customer_info)


async def export_invoice_html(
    invoice: Any,
    customer_info: Optional[InvoiceCustomerInfo] = None,
    config: Optional[InvoiceExportConfig] = None,
) -> str:
    """
    Export an invoice to HTML.

    Args:
        invoice: Invoice object
        customer_info: Optional customer information
        config: Optional export configuration

    Returns:
        HTML document as string
    """
    exporter = InvoiceExporter(config)
    return await exporter.export_html(invoice, customer_info)


__all__ = [
    "InvoiceExporter",
    "InvoiceCompanyInfo",
    "InvoiceCustomerInfo",
    "InvoiceExportConfig",
    "export_invoice_pdf",
    "export_invoice_html",
    "HAS_REPORTLAB",
]
