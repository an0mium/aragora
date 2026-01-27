"""
HTTP API Handlers for Accounts Payable Automation.

Provides REST APIs for AP workflows:
- Invoice management
- Payment optimization
- Batch payment processing
- Cash flow forecasting
- Discount opportunity tracking

Endpoints:
- POST /api/v1/accounting/ap/invoices - Add payable invoice
- GET /api/v1/accounting/ap/invoices - List payable invoices
- GET /api/v1/accounting/ap/invoices/{id} - Get invoice by ID
- POST /api/v1/accounting/ap/invoices/{id}/payment - Record payment
- POST /api/v1/accounting/ap/optimize - Optimize payment timing
- POST /api/v1/accounting/ap/batch - Create batch payment
- GET /api/v1/accounting/ap/forecast - Get cash flow forecast
- GET /api/v1/accounting/ap/discounts - Get discount opportunities
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
from aragora.server.handlers.utils.decorators import require_permission

logger = logging.getLogger(__name__)

# Thread-safe service instance
_ap_automation: Optional[Any] = None
_ap_automation_lock = threading.Lock()


def get_ap_automation():
    """Get or create AP automation service (thread-safe singleton)."""
    global _ap_automation
    if _ap_automation is not None:
        return _ap_automation

    with _ap_automation_lock:
        if _ap_automation is None:
            from aragora.services.ap_automation import APAutomation

            _ap_automation = APAutomation()
        return _ap_automation


# =============================================================================
# Invoice Management
# =============================================================================


@require_permission("finance:write")
async def handle_add_invoice(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Add a payable invoice.

    POST /api/v1/accounting/ap/invoices
    Body: {
        vendor_id: str (required),
        vendor_name: str (required),
        invoice_number: str (optional),
        invoice_date: str (optional, ISO format),
        due_date: str (optional, ISO format),
        total_amount: float (required),
        payment_terms: str (optional, default "Net 30"),
        early_pay_discount: float (optional, e.g. 0.02 for 2%),
        discount_deadline: str (optional, ISO format),
        priority: str (optional - critical, high, normal, low, hold),
        preferred_payment_method: str (optional - ach, wire, check, credit_card)
    }
    """
    try:
        ap = get_ap_automation()

        vendor_id = data.get("vendor_id")
        vendor_name = data.get("vendor_name")
        total_amount = data.get("total_amount")

        if not vendor_id:
            return error_response("vendor_id is required", status=400)
        if not vendor_name:
            return error_response("vendor_name is required", status=400)
        if total_amount is None:
            return error_response("total_amount is required", status=400)

        # Parse dates
        invoice_date = None
        due_date = None
        discount_deadline = None

        if data.get("invoice_date"):
            invoice_date = datetime.fromisoformat(data["invoice_date"])
        if data.get("due_date"):
            due_date = datetime.fromisoformat(data["due_date"])
        if data.get("discount_deadline"):
            discount_deadline = datetime.fromisoformat(data["discount_deadline"])

        invoice = await ap.add_invoice(
            vendor_id=vendor_id,
            vendor_name=vendor_name,
            invoice_number=data.get("invoice_number", ""),
            invoice_date=invoice_date,
            due_date=due_date,
            total_amount=Decimal(str(total_amount)),
            payment_terms=data.get("payment_terms", "Net 30"),
            early_pay_discount=data.get("early_pay_discount", 0),
            discount_deadline=discount_deadline,
            priority=data.get("priority"),
            preferred_payment_method=data.get("preferred_payment_method"),
        )

        return success_response(
            {
                "invoice": invoice.to_dict(),
                "message": "Invoice added successfully",
            }
        )

    except Exception as e:
        logger.exception("Error adding invoice")
        return error_response(f"Failed to add invoice: {e}", status=500)


@require_permission("ap:read")
async def handle_list_invoices(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    List payable invoices with filters.

    GET /api/v1/accounting/ap/invoices
    Query params: {
        vendor_id: str (optional),
        status: str (optional - unpaid, partial, paid),
        priority: str (optional),
        start_date: str (optional, ISO format),
        end_date: str (optional, ISO format),
        limit: int (optional, default 100),
        offset: int (optional, default 0)
    }
    """
    try:
        ap = get_ap_automation()

        # Parse filters
        vendor_id = data.get("vendor_id")
        status = data.get("status")
        priority = data.get("priority")
        start_date = None
        end_date = None

        if data.get("start_date"):
            start_date = datetime.fromisoformat(data["start_date"])
        if data.get("end_date"):
            end_date = datetime.fromisoformat(data["end_date"])

        limit = int(data.get("limit", 100))
        offset = int(data.get("offset", 0))

        invoices = await ap.list_invoices(
            vendor_id=vendor_id,
            status=status,
            priority=priority,
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


@require_permission("ap:read")
async def handle_get_invoice(
    data: Dict[str, Any],
    invoice_id: str,
    user_id: str = "default",
) -> HandlerResult:
    """
    Get a payable invoice by ID.

    GET /api/v1/accounting/ap/invoices/{invoice_id}
    """
    try:
        ap = get_ap_automation()

        invoice = await ap.get_invoice(invoice_id)
        if not invoice:
            return error_response(f"Invoice {invoice_id} not found", status=404)

        return success_response({"invoice": invoice.to_dict()})

    except Exception as e:
        logger.exception(f"Error getting invoice {invoice_id}")
        return error_response(f"Failed to get invoice: {e}", status=500)


@require_permission("finance:write")
async def handle_record_payment(
    data: Dict[str, Any],
    invoice_id: str,
    user_id: str = "default",
) -> HandlerResult:
    """
    Record a payment for a payable invoice.

    POST /api/v1/accounting/ap/invoices/{invoice_id}/payment
    Body: {
        amount: float (required),
        payment_date: str (optional, ISO format),
        payment_method: str (optional),
        reference: str (optional)
    }
    """
    try:
        ap = get_ap_automation()

        invoice = await ap.get_invoice(invoice_id)
        if not invoice:
            return error_response(f"Invoice {invoice_id} not found", status=404)

        amount = data.get("amount")
        if amount is None:
            return error_response("amount is required", status=400)

        payment_date = None
        if data.get("payment_date"):
            payment_date = datetime.fromisoformat(data["payment_date"])

        updated_invoice = await ap.record_payment(
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
# Payment Optimization
# =============================================================================


@require_permission("finance:approve")
async def handle_optimize_payments(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Optimize payment timing for outstanding invoices.

    POST /api/v1/accounting/ap/optimize
    Body: {
        invoice_ids: list[str] (optional, defaults to all unpaid),
        available_cash: float (optional),
        prioritize_discounts: bool (optional, default true)
    }
    """
    try:
        ap = get_ap_automation()

        invoice_ids = data.get("invoice_ids")
        available_cash = data.get("available_cash")
        prioritize_discounts = data.get("prioritize_discounts", True)

        # Get invoices to optimize
        if invoice_ids:
            invoices = []
            for inv_id in invoice_ids:
                inv = await ap.get_invoice(inv_id)
                if inv:
                    invoices.append(inv)
        else:
            # Get all unpaid invoices
            invoices = await ap.list_invoices(status="unpaid")

        if not invoices:
            return success_response(
                {
                    "schedule": [],
                    "message": "No invoices to optimize",
                }
            )

        schedule = await ap.optimize_payment_timing(
            invoices=invoices,
            available_cash=Decimal(str(available_cash)) if available_cash else None,
            prioritize_discounts=prioritize_discounts,
        )

        return success_response(
            {
                "schedule": schedule.to_dict(),
                "total_invoices": len(invoices),
                "message": "Payment schedule optimized",
            }
        )

    except Exception as e:
        logger.exception("Error optimizing payments")
        return error_response(f"Failed to optimize payments: {e}", status=500)


@require_permission("finance:approve")
async def handle_batch_payments(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Create a batch payment for multiple invoices.

    POST /api/v1/accounting/ap/batch
    Body: {
        invoice_ids: list[str] (required),
        payment_date: str (optional, ISO format),
        payment_method: str (optional)
    }
    """
    try:
        ap = get_ap_automation()

        invoice_ids = data.get("invoice_ids")
        if not invoice_ids:
            return error_response("invoice_ids is required", status=400)

        # Get invoices
        invoices = []
        for inv_id in invoice_ids:
            inv = await ap.get_invoice(inv_id)
            if inv:
                invoices.append(inv)
            else:
                return error_response(f"Invoice {inv_id} not found", status=404)

        payment_date = None
        if data.get("payment_date"):
            payment_date = datetime.fromisoformat(data["payment_date"])

        batch = await ap.batch_payments(
            invoices=invoices,
            payment_date=payment_date,
            payment_method=data.get("payment_method"),
        )

        return success_response(
            {
                "batch": batch.to_dict(),
                "message": f"Batch payment created for {len(invoices)} invoices",
            }
        )

    except Exception as e:
        logger.exception("Error creating batch payment")
        return error_response(f"Failed to create batch: {e}", status=500)


# =============================================================================
# Forecasting and Analysis
# =============================================================================


@require_permission("ap:read")
async def handle_get_forecast(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Get cash flow forecast.

    GET /api/v1/accounting/ap/forecast
    Query params: {
        days_ahead: int (optional, default 30)
    }
    """
    try:
        ap = get_ap_automation()

        days_ahead = int(data.get("days_ahead", 30))
        if days_ahead < 1 or days_ahead > 365:
            return error_response("days_ahead must be 1-365", status=400)

        forecast = await ap.forecast_cash_needs(days_ahead=days_ahead)

        return success_response(
            {
                "forecast": forecast.to_dict(),
                "days_ahead": days_ahead,
                "generated_at": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.exception("Error generating forecast")
        return error_response(f"Failed to generate forecast: {e}", status=500)


@require_permission("ap:read")
async def handle_get_discounts(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Get early payment discount opportunities.

    GET /api/v1/accounting/ap/discounts
    """
    try:
        ap = get_ap_automation()

        opportunities = await ap.get_discount_opportunities()

        return success_response(
            {
                "opportunities": opportunities,
                "total": len(opportunities),
            }
        )

    except Exception as e:
        logger.exception("Error getting discount opportunities")
        return error_response(f"Failed to get discounts: {e}", status=500)


# =============================================================================
# Handler Registration
# =============================================================================


class APAutomationHandler(BaseHandler):
    """Handler class for AP automation endpoints."""

    ROUTES: Dict[str, Any] = {
        "POST /api/v1/accounting/ap/invoices": handle_add_invoice,
        "GET /api/v1/accounting/ap/invoices": handle_list_invoices,
        "POST /api/v1/accounting/ap/optimize": handle_optimize_payments,
        "POST /api/v1/accounting/ap/batch": handle_batch_payments,
        "GET /api/v1/accounting/ap/forecast": handle_get_forecast,
        "GET /api/v1/accounting/ap/discounts": handle_get_discounts,
    }

    DYNAMIC_ROUTES: Dict[str, Any] = {
        "GET /api/v1/accounting/ap/invoices/{invoice_id}": handle_get_invoice,
        "POST /api/v1/accounting/ap/invoices/{invoice_id}/payment": handle_record_payment,
    }
