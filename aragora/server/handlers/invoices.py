"""
HTTP API Handlers for Invoice Processing.

Provides REST APIs for accounts payable invoice management:
- Invoice upload and extraction
- Invoice CRUD operations
- Purchase order matching
- Anomaly detection
- Approval workflow
- Payment scheduling
- Statistics and reporting

Endpoints:
- POST /api/v1/accounting/invoices/upload - Upload and extract invoice
- POST /api/v1/accounting/invoices - Create invoice manually
- GET /api/v1/accounting/invoices - List invoices with filters
- GET /api/v1/accounting/invoices/{id} - Get invoice by ID
- PUT /api/v1/accounting/invoices/{id} - Update invoice
- POST /api/v1/accounting/invoices/{id}/approve - Approve invoice
- POST /api/v1/accounting/invoices/{id}/reject - Reject invoice
- POST /api/v1/accounting/invoices/{id}/match - Match to PO
- POST /api/v1/accounting/invoices/{id}/schedule - Schedule payment
- GET /api/v1/accounting/invoices/{id}/anomalies - Get anomalies
- GET /api/v1/accounting/invoices/pending - Get pending approvals
- GET /api/v1/accounting/invoices/overdue - Get overdue invoices
- GET /api/v1/accounting/invoices/stats - Get statistics
- POST /api/v1/accounting/purchase-orders - Add purchase order
- GET /api/v1/accounting/payments/scheduled - Get scheduled payments
"""

from __future__ import annotations

import base64
import logging
import threading
from datetime import datetime
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
_invoice_processor: Optional[Any] = None
_invoice_processor_lock = threading.Lock()


def get_invoice_processor():
    """Get or create invoice processor (thread-safe singleton)."""
    global _invoice_processor
    if _invoice_processor is not None:
        return _invoice_processor

    with _invoice_processor_lock:
        if _invoice_processor is None:
            from aragora.services.invoice_processor import InvoiceProcessor

            _invoice_processor = InvoiceProcessor()
        return _invoice_processor


# =============================================================================
# Invoice Upload and Extraction
# =============================================================================


async def handle_upload_invoice(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Upload and extract data from an invoice document.

    POST /api/v1/accounting/invoices/upload
    Body: {
        document_data: str (base64 encoded PDF/image),
        content_type: str (optional),
        vendor_hint: str (optional)
    }
    """
    try:
        processor = get_invoice_processor()

        doc_b64 = data.get("document_data")
        if not doc_b64:
            return error_response("document_data is required", status=400)

        # Decode base64
        try:
            document_bytes = base64.b64decode(doc_b64)
        except Exception:
            return error_response("Invalid base64 document_data", status=400)

        vendor_hint = data.get("vendor_hint")

        # Extract invoice data
        invoice = await processor.extract_invoice_data(
            document_bytes=document_bytes,
            vendor_hint=vendor_hint,
        )

        # Auto-detect anomalies
        anomalies = await processor.detect_anomalies(invoice)

        return success_response(
            {
                "invoice": invoice.to_dict(),
                "anomalies": [a.to_dict() for a in anomalies],
                "message": "Invoice extracted successfully",
            }
        )

    except Exception as e:
        logger.exception("Error processing invoice")
        return error_response(f"Failed to process invoice: {e}", status=500)


# =============================================================================
# Invoice CRUD
# =============================================================================


async def handle_create_invoice(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Create an invoice manually.

    POST /api/v1/accounting/invoices
    Body: {
        vendor_name: str (required),
        total_amount: float (required),
        invoice_number: str (optional),
        invoice_date: str (ISO format, optional),
        due_date: str (ISO format, optional),
        line_items: list (optional),
        po_number: str (optional)
    }
    """
    try:
        processor = get_invoice_processor()

        vendor_name = data.get("vendor_name")
        total_amount = data.get("total_amount")

        if not vendor_name:
            return error_response("vendor_name is required", status=400)
        if total_amount is None:
            return error_response("total_amount is required", status=400)

        try:
            total_amount = float(total_amount)
        except (TypeError, ValueError):
            return error_response("total_amount must be a number", status=400)

        # Parse dates
        invoice_date = None
        if data.get("invoice_date"):
            try:
                invoice_date = datetime.fromisoformat(data["invoice_date"].replace("Z", "+00:00"))
            except ValueError:
                return error_response("Invalid invoice_date format", status=400)

        due_date = None
        if data.get("due_date"):
            try:
                due_date = datetime.fromisoformat(data["due_date"].replace("Z", "+00:00"))
            except ValueError:
                return error_response("Invalid due_date format", status=400)

        invoice = await processor.create_manual_invoice(
            vendor_name=vendor_name,
            total_amount=total_amount,
            invoice_number=data.get("invoice_number", ""),
            invoice_date=invoice_date,
            due_date=due_date,
            line_items=data.get("line_items"),
            po_number=data.get("po_number"),
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
    query_params: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    List invoices with filters.

    GET /api/v1/accounting/invoices
    Query params:
        status: str
        vendor: str
        start_date: str (ISO format)
        end_date: str (ISO format)
        limit: int (default 100)
        offset: int (default 0)
    """
    try:
        processor = get_invoice_processor()

        from aragora.services.invoice_processor import InvoiceStatus

        # Parse status filter
        status = None
        status_str = query_params.get("status")
        if status_str:
            try:
                status = InvoiceStatus(status_str)
            except ValueError:
                pass

        # Parse date filters
        start_date = None
        if query_params.get("start_date"):
            try:
                start_date = datetime.fromisoformat(
                    query_params["start_date"].replace("Z", "+00:00")
                )
            except ValueError:
                pass

        end_date = None
        if query_params.get("end_date"):
            try:
                end_date = datetime.fromisoformat(query_params["end_date"].replace("Z", "+00:00"))
            except ValueError:
                pass

        limit = int(query_params.get("limit", 100))
        offset = int(query_params.get("offset", 0))

        invoices, total = await processor.list_invoices(
            status=status,
            vendor=query_params.get("vendor"),
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
        )

        return success_response(
            {
                "invoices": [i.to_dict() for i in invoices],
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        )

    except Exception as e:
        logger.exception("Error listing invoices")
        return error_response(f"Failed to list invoices: {e}", status=500)


async def handle_get_invoice(
    invoice_id: str,
    user_id: str = "default",
) -> HandlerResult:
    """
    Get invoice by ID.

    GET /api/v1/accounting/invoices/{id}
    """
    try:
        processor = get_invoice_processor()

        invoice = await processor.get_invoice(invoice_id)
        if not invoice:
            return error_response("Invoice not found", status=404)

        return success_response({"invoice": invoice.to_dict()})

    except Exception as e:
        logger.exception("Error getting invoice")
        return error_response(f"Failed to get invoice: {e}", status=500)


# =============================================================================
# Approval Workflow
# =============================================================================


@require_permission("finance:approve")
async def handle_approve_invoice(
    invoice_id: str,
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Approve an invoice for payment.

    POST /api/v1/accounting/invoices/{id}/approve
    Body: {
        approver_id: str (optional, defaults to user_id)
    }
    """
    try:
        processor = get_invoice_processor()

        approver_id = data.get("approver_id", user_id)
        invoice = await processor.approve_invoice(invoice_id, approver_id)

        if not invoice:
            return error_response("Invoice not found", status=404)

        return success_response(
            {
                "invoice": invoice.to_dict(),
                "message": "Invoice approved successfully",
            }
        )

    except Exception as e:
        logger.exception("Error approving invoice")
        return error_response(f"Failed to approve invoice: {e}", status=500)


@require_permission("finance:approve")
async def handle_reject_invoice(
    invoice_id: str,
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Reject an invoice.

    POST /api/v1/accounting/invoices/{id}/reject
    Body: {
        reason: str (optional)
    }
    """
    try:
        processor = get_invoice_processor()

        reason = data.get("reason", "")
        invoice = await processor.reject_invoice(invoice_id, reason)

        if not invoice:
            return error_response("Invoice not found", status=404)

        return success_response(
            {
                "invoice": invoice.to_dict(),
                "message": "Invoice rejected",
            }
        )

    except Exception as e:
        logger.exception("Error rejecting invoice")
        return error_response(f"Failed to reject invoice: {e}", status=500)


async def handle_get_pending_approvals(
    user_id: str = "default",
) -> HandlerResult:
    """
    Get invoices pending approval.

    GET /api/v1/accounting/invoices/pending
    """
    try:
        processor = get_invoice_processor()

        invoices = await processor.get_pending_approvals()

        return success_response(
            {
                "invoices": [i.to_dict() for i in invoices],
                "count": len(invoices),
            }
        )

    except Exception as e:
        logger.exception("Error getting pending approvals")
        return error_response(f"Failed to get pending approvals: {e}", status=500)


# =============================================================================
# PO Matching
# =============================================================================


async def handle_match_to_po(
    invoice_id: str,
    user_id: str = "default",
) -> HandlerResult:
    """
    Match invoice to purchase order.

    POST /api/v1/accounting/invoices/{id}/match
    """
    try:
        processor = get_invoice_processor()

        invoice = await processor.get_invoice(invoice_id)
        if not invoice:
            return error_response("Invoice not found", status=404)

        match = await processor.match_to_po(invoice)

        return success_response(
            {
                "match": match.to_dict(),
                "invoice": invoice.to_dict(),
            }
        )

    except Exception as e:
        logger.exception("Error matching invoice to PO")
        return error_response(f"Failed to match invoice: {e}", status=500)


# =============================================================================
# Anomaly Detection
# =============================================================================


async def handle_get_anomalies(
    invoice_id: str,
    user_id: str = "default",
) -> HandlerResult:
    """
    Get anomalies for an invoice.

    GET /api/v1/accounting/invoices/{id}/anomalies
    """
    try:
        processor = get_invoice_processor()

        invoice = await processor.get_invoice(invoice_id)
        if not invoice:
            return error_response("Invoice not found", status=404)

        anomalies = await processor.detect_anomalies(invoice)

        return success_response(
            {
                "anomalies": [a.to_dict() for a in anomalies],
                "count": len(anomalies),
            }
        )

    except Exception as e:
        logger.exception("Error detecting anomalies")
        return error_response(f"Failed to detect anomalies: {e}", status=500)


# =============================================================================
# Payment Scheduling
# =============================================================================


@require_permission("finance:approve")
async def handle_schedule_payment(
    invoice_id: str,
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Schedule payment for an invoice.

    POST /api/v1/accounting/invoices/{id}/schedule
    Body: {
        pay_date: str (ISO format, optional),
        payment_method: str (optional, default 'ach')
    }
    """
    try:
        processor = get_invoice_processor()

        invoice = await processor.get_invoice(invoice_id)
        if not invoice:
            return error_response("Invoice not found", status=404)

        # Parse pay date
        pay_date = None
        if data.get("pay_date"):
            try:
                pay_date = datetime.fromisoformat(data["pay_date"].replace("Z", "+00:00"))
            except ValueError:
                return error_response("Invalid pay_date format", status=400)

        payment_method = data.get("payment_method", "ach")

        schedule = await processor.schedule_payment(
            invoice=invoice,
            pay_date=pay_date,
            payment_method=payment_method,
        )

        return success_response(
            {
                "schedule": schedule.to_dict(),
                "invoice": invoice.to_dict(),
                "message": "Payment scheduled successfully",
            }
        )

    except ValueError as e:
        return error_response(str(e), status=400)
    except Exception as e:
        logger.exception("Error scheduling payment")
        return error_response(f"Failed to schedule payment: {e}", status=500)


async def handle_get_scheduled_payments(
    query_params: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Get scheduled payments.

    GET /api/v1/accounting/payments/scheduled
    Query params:
        start_date: str (ISO format)
        end_date: str (ISO format)
    """
    try:
        processor = get_invoice_processor()

        start_date = None
        if query_params.get("start_date"):
            try:
                start_date = datetime.fromisoformat(
                    query_params["start_date"].replace("Z", "+00:00")
                )
            except ValueError:
                pass

        end_date = None
        if query_params.get("end_date"):
            try:
                end_date = datetime.fromisoformat(query_params["end_date"].replace("Z", "+00:00"))
            except ValueError:
                pass

        payments = await processor.get_scheduled_payments(
            start_date=start_date,
            end_date=end_date,
        )

        total_amount = sum(p.amount for p in payments)

        return success_response(
            {
                "payments": [p.to_dict() for p in payments],
                "count": len(payments),
                "totalAmount": float(total_amount),
            }
        )

    except Exception as e:
        logger.exception("Error getting scheduled payments")
        return error_response(f"Failed to get scheduled payments: {e}", status=500)


# =============================================================================
# Purchase Orders
# =============================================================================


@require_permission("finance:write")
async def handle_create_purchase_order(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Add a purchase order for matching.

    POST /api/v1/accounting/purchase-orders
    Body: {
        po_number: str (required),
        vendor_name: str (required),
        total_amount: float (required),
        order_date: str (ISO format, optional),
        expected_delivery: str (ISO format, optional),
        line_items: list (optional)
    }
    """
    try:
        processor = get_invoice_processor()

        po_number = data.get("po_number")
        vendor_name = data.get("vendor_name")
        total_amount = data.get("total_amount")

        if not po_number:
            return error_response("po_number is required", status=400)
        if not vendor_name:
            return error_response("vendor_name is required", status=400)
        if total_amount is None:
            return error_response("total_amount is required", status=400)

        # Parse dates
        order_date = None
        if data.get("order_date"):
            try:
                order_date = datetime.fromisoformat(data["order_date"].replace("Z", "+00:00"))
            except ValueError:
                pass

        expected_delivery = None
        if data.get("expected_delivery"):
            try:
                expected_delivery = datetime.fromisoformat(
                    data["expected_delivery"].replace("Z", "+00:00")
                )
            except ValueError:
                pass

        po = await processor.add_purchase_order(
            po_number=po_number,
            vendor_name=vendor_name,
            total_amount=float(total_amount),
            order_date=order_date,
            expected_delivery=expected_delivery,
            line_items=data.get("line_items"),
        )

        return success_response(
            {
                "purchaseOrder": po.to_dict(),
                "message": "Purchase order created successfully",
            }
        )

    except Exception as e:
        logger.exception("Error creating purchase order")
        return error_response(f"Failed to create purchase order: {e}", status=500)


# =============================================================================
# Statistics
# =============================================================================


async def handle_get_invoice_stats(
    user_id: str = "default",
) -> HandlerResult:
    """
    Get invoice processing statistics.

    GET /api/v1/accounting/invoices/stats
    """
    try:
        processor = get_invoice_processor()

        stats = processor.get_stats()

        return success_response({"stats": stats})

    except Exception as e:
        logger.exception("Error getting invoice stats")
        return error_response(f"Failed to get invoice stats: {e}", status=500)


async def handle_get_overdue_invoices(
    user_id: str = "default",
) -> HandlerResult:
    """
    Get overdue invoices.

    GET /api/v1/accounting/invoices/overdue
    """
    try:
        processor = get_invoice_processor()

        invoices = await processor.get_overdue_invoices()

        return success_response(
            {
                "invoices": [i.to_dict() for i in invoices],
                "count": len(invoices),
                "totalAmount": sum(float(i.total_amount) for i in invoices),
            }
        )

    except Exception as e:
        logger.exception("Error getting overdue invoices")
        return error_response(f"Failed to get overdue invoices: {e}", status=500)


# =============================================================================
# Handler Class for Router Registration
# =============================================================================


class InvoiceHandler(BaseHandler):
    """Handler for invoice-related routes."""

    ROUTES = {
        "/api/v1/accounting/invoices/upload": ["POST"],
        "/api/v1/accounting/invoices": ["GET", "POST"],
        "/api/v1/accounting/invoices/pending": ["GET"],
        "/api/v1/accounting/invoices/overdue": ["GET"],
        "/api/v1/accounting/invoices/stats": ["GET"],
        "/api/v1/accounting/purchase-orders": ["POST"],
        "/api/v1/accounting/payments/scheduled": ["GET"],
    }

    DYNAMIC_ROUTES = {
        "/api/v1/accounting/invoices/{invoice_id}": ["GET"],
        "/api/v1/accounting/invoices/{invoice_id}/approve": ["POST"],
        "/api/v1/accounting/invoices/{invoice_id}/reject": ["POST"],
        "/api/v1/accounting/invoices/{invoice_id}/match": ["POST"],
        "/api/v1/accounting/invoices/{invoice_id}/schedule": ["POST"],
        "/api/v1/accounting/invoices/{invoice_id}/anomalies": ["GET"],
    }

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        if path in self.ROUTES:
            return True
        for route_pattern in self.DYNAMIC_ROUTES:
            if self._matches_pattern(path, route_pattern):
                return True
        return False

    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches a route pattern with {param}."""
        pattern_parts = pattern.split("/")
        path_parts = path.split("/")

        if len(pattern_parts) != len(path_parts):
            return False

        for pattern_part, path_part in zip(pattern_parts, path_parts):
            if pattern_part.startswith("{") and pattern_part.endswith("}"):
                continue
            if pattern_part != path_part:
                return False

        return True

    def _extract_invoice_id(self, path: str) -> Optional[str]:
        """Extract invoice_id from path."""
        parts = path.split("/")
        # parts[0]="", [1]="api", [2]="v1", [3]="accounting", [4]="invoices", [5]=invoice_id
        if len(parts) >= 6:
            return parts[5]
        return None

    async def handle_get(
        self,
        path: str,
        query_params: Optional[Dict[str, Any]] = None,
    ) -> HandlerResult:
        """Handle GET requests."""
        query_params = query_params or {}

        if path == "/api/v1/accounting/invoices":
            return await handle_list_invoices(query_params)

        if path == "/api/v1/accounting/invoices/pending":
            return await handle_get_pending_approvals()

        if path == "/api/v1/accounting/invoices/overdue":
            return await handle_get_overdue_invoices()

        if path == "/api/v1/accounting/invoices/stats":
            return await handle_get_invoice_stats()

        if path == "/api/v1/accounting/payments/scheduled":
            return await handle_get_scheduled_payments(query_params)

        # Dynamic routes
        invoice_id = self._extract_invoice_id(path)
        if invoice_id:
            if "/anomalies" in path:
                return await handle_get_anomalies(invoice_id)
            else:
                return await handle_get_invoice(invoice_id)

        return error_response("Route not found", status=404)

    async def handle_post(  # type: ignore[override]
        self,
        path: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> HandlerResult:
        """Handle POST requests."""
        data = data or {}

        if path == "/api/v1/accounting/invoices/upload":
            return await handle_upload_invoice(data)

        if path == "/api/v1/accounting/invoices":
            return await handle_create_invoice(data)

        if path == "/api/v1/accounting/purchase-orders":
            return await handle_create_purchase_order(data)

        # Dynamic routes
        invoice_id = self._extract_invoice_id(path)
        if invoice_id:
            if "/approve" in path:
                return await handle_approve_invoice(invoice_id, data)
            if "/reject" in path:
                return await handle_reject_invoice(invoice_id, data)
            if "/match" in path:
                return await handle_match_to_po(invoice_id)
            if "/schedule" in path:
                return await handle_schedule_payment(invoice_id, data)

        return error_response("Route not found", status=404)
