"""
HTTP API Handlers for Accounts Receivable Automation.

Provides REST APIs for AR workflows:
- Invoice generation and management
- Payment reminder scheduling
- AR aging reports
- Collection action suggestions
- Customer management

Endpoints:
- POST /api/v1/accounting/ar/invoices - Create invoice
- GET /api/v1/accounting/ar/invoices - List invoices
- GET /api/v1/accounting/ar/invoices/{id} - Get invoice by ID
- POST /api/v1/accounting/ar/invoices/{id}/send - Send invoice to customer
- POST /api/v1/accounting/ar/invoices/{id}/reminder - Send payment reminder
- POST /api/v1/accounting/ar/invoices/{id}/payment - Record payment
- GET /api/v1/accounting/ar/aging - Get AR aging report
- GET /api/v1/accounting/ar/collections - Get collection suggestions
- POST /api/v1/accounting/ar/customers - Add customer
- GET /api/v1/accounting/ar/customers/{id}/balance - Get customer balance
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)

# Thread-safe service instance
_ar_automation: Optional[Any] = None
_ar_automation_lock = threading.Lock()


def get_ar_automation():
    """Get or create AR automation service (thread-safe singleton)."""
    global _ar_automation
    if _ar_automation is not None:
        return _ar_automation

    with _ar_automation_lock:
        if _ar_automation is None:
            from aragora.services.ar_automation import ARAutomation

            _ar_automation = ARAutomation()
        return _ar_automation


# =============================================================================
# Invoice Management
# =============================================================================


async def handle_create_invoice(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Create a new AR invoice.

    POST /api/v1/accounting/ar/invoices
    Body: {
        customer_id: str (required),
        customer_name: str (required),
        customer_email: str (optional),
        line_items: list[{description, quantity, unit_price, amount}] (required),
        payment_terms: str (optional, default "Net 30"),
        memo: str (optional),
        tax_rate: float (optional, default 0)
    }
    """
    try:
        ar = get_ar_automation()

        customer_id = data.get("customer_id")
        customer_name = data.get("customer_name")
        line_items = data.get("line_items", [])

        if not customer_id:
            return error_response("customer_id is required", status=400)
        if not customer_name:
            return error_response("customer_name is required", status=400)
        if not line_items:
            return error_response("line_items is required", status=400)

        # Validate line items
        for item in line_items:
            if "description" not in item or "amount" not in item:
                return error_response("Each line item must have description and amount", status=400)

        invoice = await ar.generate_invoice(
            customer_id=customer_id,
            customer_name=customer_name,
            customer_email=data.get("customer_email"),
            line_items=line_items,
            payment_terms=data.get("payment_terms", "Net 30"),
            memo=data.get("memo", ""),
            tax_rate=data.get("tax_rate", 0),
        )

        return success_response(
            {
                "invoice": invoice.to_dict(),
                "message": "Invoice created successfully",
            }
        )

    except Exception as e:
        logger.exception("Error creating invoice")
        return error_response(f"Failed to create invoice: {e}", status=500)


async def handle_list_invoices(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    List AR invoices with filters.

    GET /api/v1/accounting/ar/invoices
    Query params: {
        customer_id: str (optional),
        status: str (optional - draft, sent, paid, overdue),
        start_date: str (optional, ISO format),
        end_date: str (optional, ISO format),
        limit: int (optional, default 100),
        offset: int (optional, default 0)
    }
    """
    try:
        ar = get_ar_automation()

        # Parse filters
        customer_id = data.get("customer_id")
        status = data.get("status")
        start_date = None
        end_date = None

        if data.get("start_date"):
            start_date = datetime.fromisoformat(data["start_date"])
        if data.get("end_date"):
            end_date = datetime.fromisoformat(data["end_date"])

        limit = int(data.get("limit", 100))
        offset = int(data.get("offset", 0))

        invoices = await ar.list_invoices(
            customer_id=customer_id,
            status=status,
            start_date=start_date,
            end_date=end_date,
        )

        # Apply pagination
        paginated = invoices[offset : offset + limit]

        return success_response(
            {
                "invoices": [inv.to_dict() for inv in paginated],
                "total": len(invoices),
                "limit": limit,
                "offset": offset,
            }
        )

    except Exception as e:
        logger.exception("Error listing invoices")
        return error_response(f"Failed to list invoices: {e}", status=500)


async def handle_get_invoice(
    data: Dict[str, Any],
    invoice_id: str,
    user_id: str = "default",
) -> HandlerResult:
    """
    Get an invoice by ID.

    GET /api/v1/accounting/ar/invoices/{invoice_id}
    """
    try:
        ar = get_ar_automation()

        invoice = await ar.get_invoice(invoice_id)
        if not invoice:
            return error_response(f"Invoice {invoice_id} not found", status=404)

        return success_response({"invoice": invoice.to_dict()})

    except Exception as e:
        logger.exception(f"Error getting invoice {invoice_id}")
        return error_response(f"Failed to get invoice: {e}", status=500)


async def handle_send_invoice(
    data: Dict[str, Any],
    invoice_id: str,
    user_id: str = "default",
) -> HandlerResult:
    """
    Send an invoice to the customer.

    POST /api/v1/accounting/ar/invoices/{invoice_id}/send
    """
    try:
        ar = get_ar_automation()

        invoice = await ar.get_invoice(invoice_id)
        if not invoice:
            return error_response(f"Invoice {invoice_id} not found", status=404)

        success = await ar.send_invoice(invoice_id)

        if success:
            return success_response(
                {
                    "message": "Invoice sent successfully",
                    "invoice_id": invoice_id,
                }
            )
        else:
            return error_response("Failed to send invoice", status=500)

    except Exception as e:
        logger.exception(f"Error sending invoice {invoice_id}")
        return error_response(f"Failed to send invoice: {e}", status=500)


async def handle_send_reminder(
    data: Dict[str, Any],
    invoice_id: str,
    user_id: str = "default",
) -> HandlerResult:
    """
    Send a payment reminder for an invoice.

    POST /api/v1/accounting/ar/invoices/{invoice_id}/reminder
    Body: {
        escalation_level: int (optional, 1-4)
    }
    """
    try:
        ar = get_ar_automation()

        invoice = await ar.get_invoice(invoice_id)
        if not invoice:
            return error_response(f"Invoice {invoice_id} not found", status=404)

        escalation_level = int(data.get("escalation_level", 1))
        if escalation_level < 1 or escalation_level > 4:
            return error_response("escalation_level must be 1-4", status=400)

        success = await ar.send_payment_reminder(
            invoice_id=invoice_id,
            escalation_level=escalation_level,
        )

        if success:
            return success_response(
                {
                    "message": f"Reminder (level {escalation_level}) sent successfully",
                    "invoice_id": invoice_id,
                }
            )
        else:
            return error_response("Failed to send reminder", status=500)

    except Exception as e:
        logger.exception(f"Error sending reminder for invoice {invoice_id}")
        return error_response(f"Failed to send reminder: {e}", status=500)


async def handle_record_payment(
    data: Dict[str, Any],
    invoice_id: str,
    user_id: str = "default",
) -> HandlerResult:
    """
    Record a payment against an invoice.

    POST /api/v1/accounting/ar/invoices/{invoice_id}/payment
    Body: {
        amount: float (required),
        payment_date: str (optional, ISO format),
        payment_method: str (optional),
        reference: str (optional)
    }
    """
    try:
        ar = get_ar_automation()

        invoice = await ar.get_invoice(invoice_id)
        if not invoice:
            return error_response(f"Invoice {invoice_id} not found", status=404)

        amount = data.get("amount")
        if amount is None:
            return error_response("amount is required", status=400)

        payment_date = None
        if data.get("payment_date"):
            payment_date = datetime.fromisoformat(data["payment_date"])

        updated_invoice = await ar.record_payment(
            invoice_id=invoice_id,
            amount=Decimal(str(amount)),
            payment_date=payment_date,
            payment_method=data.get("payment_method"),
            reference=data.get("reference"),
        )

        return success_response(
            {
                "invoice": updated_invoice.to_dict(),
                "message": f"Payment of {amount} recorded successfully",
            }
        )

    except Exception as e:
        logger.exception(f"Error recording payment for invoice {invoice_id}")
        return error_response(f"Failed to record payment: {e}", status=500)


# =============================================================================
# AR Reporting
# =============================================================================


async def handle_get_aging_report(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Get AR aging report.

    GET /api/v1/accounting/ar/aging
    """
    try:
        ar = get_ar_automation()

        report = await ar.track_aging()

        return success_response(
            {
                "aging_report": report.to_dict(),
                "generated_at": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.exception("Error generating aging report")
        return error_response(f"Failed to generate aging report: {e}", status=500)


async def handle_get_collections(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Get collection action suggestions.

    GET /api/v1/accounting/ar/collections
    """
    try:
        ar = get_ar_automation()

        suggestions = await ar.suggest_collections()

        return success_response(
            {
                "suggestions": [s.to_dict() for s in suggestions],
                "total": len(suggestions),
            }
        )

    except Exception as e:
        logger.exception("Error getting collection suggestions")
        return error_response(f"Failed to get suggestions: {e}", status=500)


# =============================================================================
# Customer Management
# =============================================================================


async def handle_add_customer(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Add a new customer.

    POST /api/v1/accounting/ar/customers
    Body: {
        customer_id: str (required),
        name: str (required),
        email: str (optional),
        phone: str (optional),
        address: str (optional),
        payment_terms: str (optional, default "Net 30")
    }
    """
    try:
        ar = get_ar_automation()

        customer_id = data.get("customer_id")
        name = data.get("name")

        if not customer_id:
            return error_response("customer_id is required", status=400)
        if not name:
            return error_response("name is required", status=400)

        await ar.add_customer(
            customer_id=customer_id,
            name=name,
            email=data.get("email"),
            payment_terms=data.get("payment_terms", "Net 30"),
        )

        return success_response(
            {
                "message": "Customer added successfully",
                "customer_id": customer_id,
            }
        )

    except Exception as e:
        logger.exception("Error adding customer")
        return error_response(f"Failed to add customer: {e}", status=500)


async def handle_get_customer_balance(
    data: Dict[str, Any],
    customer_id: str,
    user_id: str = "default",
) -> HandlerResult:
    """
    Get outstanding balance for a customer.

    GET /api/v1/accounting/ar/customers/{customer_id}/balance
    """
    try:
        ar = get_ar_automation()

        balance = await ar.get_customer_balance(customer_id)

        return success_response(
            {
                "customer_id": customer_id,
                "outstanding_balance": str(balance),
            }
        )

    except Exception as e:
        logger.exception(f"Error getting balance for customer {customer_id}")
        return error_response(f"Failed to get balance: {e}", status=500)


# =============================================================================
# Handler Registration
# =============================================================================


class ARAutomationHandler(BaseHandler):
    """Handler class for AR automation endpoints."""

    ROUTES: Dict[str, Any] = {
        "POST /api/v1/accounting/ar/invoices": handle_create_invoice,
        "GET /api/v1/accounting/ar/invoices": handle_list_invoices,
        "GET /api/v1/accounting/ar/aging": handle_get_aging_report,
        "GET /api/v1/accounting/ar/collections": handle_get_collections,
        "POST /api/v1/accounting/ar/customers": handle_add_customer,
    }

    DYNAMIC_ROUTES: Dict[str, Any] = {
        "GET /api/v1/accounting/ar/invoices/{invoice_id}": handle_get_invoice,
        "POST /api/v1/accounting/ar/invoices/{invoice_id}/send": handle_send_invoice,
        "POST /api/v1/accounting/ar/invoices/{invoice_id}/reminder": handle_send_reminder,
        "POST /api/v1/accounting/ar/invoices/{invoice_id}/payment": handle_record_payment,
        "GET /api/v1/accounting/ar/customers/{customer_id}/balance": handle_get_customer_balance,
    }
