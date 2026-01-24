"""
HTTP API Handlers for Expense Tracking.

Provides REST APIs for expense management:
- Receipt upload and processing
- Expense CRUD operations
- Auto-categorization
- Duplicate detection
- QBO sync integration
- Expense reporting and stats

Endpoints:
- POST /api/v1/accounting/expenses/upload - Upload and process receipt
- POST /api/v1/accounting/expenses - Create expense manually
- GET /api/v1/accounting/expenses - List expenses with filters
- GET /api/v1/accounting/expenses/{id} - Get expense by ID
- PUT /api/v1/accounting/expenses/{id} - Update expense
- DELETE /api/v1/accounting/expenses/{id} - Delete expense
- POST /api/v1/accounting/expenses/{id}/approve - Approve expense
- POST /api/v1/accounting/expenses/{id}/reject - Reject expense
- POST /api/v1/accounting/expenses/categorize - Auto-categorize expenses
- POST /api/v1/accounting/expenses/sync - Sync expenses to QBO
- GET /api/v1/accounting/expenses/stats - Get expense statistics
- GET /api/v1/accounting/expenses/pending - Get pending approvals
- GET /api/v1/accounting/expenses/export - Export expenses
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

logger = logging.getLogger(__name__)

# Thread-safe service instance
_expense_tracker: Optional[Any] = None
_expense_tracker_lock = threading.Lock()


def get_expense_tracker():
    """Get or create expense tracker (thread-safe singleton)."""
    global _expense_tracker
    if _expense_tracker is not None:
        return _expense_tracker

    with _expense_tracker_lock:
        if _expense_tracker is None:
            from aragora.services.expense_tracker import ExpenseTracker

            _expense_tracker = ExpenseTracker()
        return _expense_tracker


# =============================================================================
# Receipt Upload and Processing
# =============================================================================


async def handle_upload_receipt(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Upload and process a receipt image.

    POST /api/v1/accounting/expenses/upload
    Body: {
        receipt_data: str (base64 encoded image),
        content_type: str (image/png, image/jpeg, application/pdf),
        employee_id: str (optional),
        payment_method: str (optional, default credit_card)
    }
    """
    try:
        tracker = get_expense_tracker()

        receipt_b64 = data.get("receipt_data")
        if not receipt_b64:
            return error_response("receipt_data is required", status=400)

        # Decode base64 image
        try:
            image_data = base64.b64decode(receipt_b64)
        except Exception:
            return error_response("Invalid base64 receipt_data", status=400)

        employee_id = data.get("employee_id")
        payment_method_str = data.get("payment_method", "credit_card")

        # Parse payment method
        from aragora.services.expense_tracker import PaymentMethod

        try:
            payment_method = PaymentMethod(payment_method_str)
        except ValueError:
            payment_method = PaymentMethod.CREDIT_CARD

        # Process receipt
        expense = await tracker.process_receipt(
            image_data=image_data,
            employee_id=employee_id,
            payment_method=payment_method,
        )

        return success_response(
            {
                "expense": expense.to_dict(),
                "message": "Receipt processed successfully",
            }
        )

    except Exception as e:
        logger.exception("Error processing receipt")
        return error_response(f"Failed to process receipt: {e}", status=500)


# =============================================================================
# Expense CRUD Operations
# =============================================================================


async def handle_create_expense(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Create an expense manually.

    POST /api/v1/accounting/expenses
    Body: {
        vendor_name: str (required),
        amount: float (required),
        date: str (ISO format, optional),
        category: str (optional),
        payment_method: str (optional),
        description: str (optional),
        employee_id: str (optional),
        is_reimbursable: bool (optional),
        tags: list[str] (optional)
    }
    """
    try:
        tracker = get_expense_tracker()

        vendor_name = data.get("vendor_name")
        amount = data.get("amount")

        if not vendor_name:
            return error_response("vendor_name is required", status=400)
        if amount is None:
            return error_response("amount is required", status=400)

        try:
            amount = float(amount)
        except (TypeError, ValueError):
            return error_response("amount must be a number", status=400)

        # Parse date
        date = None
        date_str = data.get("date")
        if date_str:
            try:
                date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except ValueError:
                return error_response("Invalid date format", status=400)

        # Parse category
        from aragora.services.expense_tracker import ExpenseCategory, PaymentMethod

        category = None
        category_str = data.get("category")
        if category_str:
            try:
                category = ExpenseCategory(category_str)
            except ValueError:
                pass  # Will auto-categorize

        # Parse payment method
        payment_method = PaymentMethod.CREDIT_CARD
        payment_method_str = data.get("payment_method")
        if payment_method_str:
            try:
                payment_method = PaymentMethod(payment_method_str)
            except ValueError:
                pass

        expense = await tracker.create_expense(
            vendor_name=vendor_name,
            amount=amount,
            date=date,
            category=category,
            payment_method=payment_method,
            description=data.get("description", ""),
            employee_id=data.get("employee_id"),
            is_reimbursable=data.get("is_reimbursable", False),
            tags=data.get("tags"),
        )

        return success_response(
            {
                "expense": expense.to_dict(),
                "message": "Expense created successfully",
            }
        )

    except Exception as e:
        logger.exception("Error creating expense")
        return error_response(f"Failed to create expense: {e}", status=500)


async def handle_list_expenses(
    query_params: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    List expenses with filters.

    GET /api/v1/accounting/expenses
    Query params:
        category: str
        vendor: str
        start_date: str (ISO format)
        end_date: str (ISO format)
        status: str
        employee_id: str
        limit: int (default 100)
        offset: int (default 0)
    """
    try:
        tracker = get_expense_tracker()

        # Parse filters
        from aragora.services.expense_tracker import ExpenseCategory, ExpenseStatus

        category = None
        category_str = query_params.get("category")
        if category_str:
            try:
                category = ExpenseCategory(category_str)
            except ValueError:
                pass

        status = None
        status_str = query_params.get("status")
        if status_str:
            try:
                status = ExpenseStatus(status_str)
            except ValueError:
                pass

        start_date = None
        start_date_str = query_params.get("start_date")
        if start_date_str:
            try:
                start_date = datetime.fromisoformat(start_date_str.replace("Z", "+00:00"))
            except ValueError:
                pass

        end_date = None
        end_date_str = query_params.get("end_date")
        if end_date_str:
            try:
                end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            except ValueError:
                pass

        limit = int(query_params.get("limit", 100))
        offset = int(query_params.get("offset", 0))

        expenses, total = await tracker.list_expenses(
            category=category,
            vendor=query_params.get("vendor"),
            start_date=start_date,
            end_date=end_date,
            status=status,
            employee_id=query_params.get("employee_id"),
            limit=limit,
            offset=offset,
        )

        return success_response(
            {
                "expenses": [e.to_dict() for e in expenses],
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        )

    except Exception as e:
        logger.exception("Error listing expenses")
        return error_response(f"Failed to list expenses: {e}", status=500)


async def handle_get_expense(
    expense_id: str,
    user_id: str = "default",
) -> HandlerResult:
    """
    Get expense by ID.

    GET /api/v1/accounting/expenses/{id}
    """
    try:
        tracker = get_expense_tracker()

        expense = await tracker.get_expense(expense_id)
        if not expense:
            return error_response("Expense not found", status=404)

        return success_response({"expense": expense.to_dict()})

    except Exception as e:
        logger.exception("Error getting expense")
        return error_response(f"Failed to get expense: {e}", status=500)


async def handle_update_expense(
    expense_id: str,
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Update an expense.

    PUT /api/v1/accounting/expenses/{id}
    Body: {
        vendor_name: str (optional),
        amount: float (optional),
        category: str (optional),
        description: str (optional),
        status: str (optional),
        is_reimbursable: bool (optional),
        tags: list[str] (optional)
    }
    """
    try:
        tracker = get_expense_tracker()

        # Parse category
        from aragora.services.expense_tracker import ExpenseCategory, ExpenseStatus

        category = None
        category_str = data.get("category")
        if category_str:
            try:
                category = ExpenseCategory(category_str)
            except ValueError:
                pass

        status = None
        status_str = data.get("status")
        if status_str:
            try:
                status = ExpenseStatus(status_str)
            except ValueError:
                pass

        expense = await tracker.update_expense(
            expense_id=expense_id,
            vendor_name=data.get("vendor_name"),
            amount=data.get("amount"),
            category=category,
            description=data.get("description"),
            status=status,
            is_reimbursable=data.get("is_reimbursable"),
            tags=data.get("tags"),
        )

        if not expense:
            return error_response("Expense not found", status=404)

        return success_response(
            {
                "expense": expense.to_dict(),
                "message": "Expense updated successfully",
            }
        )

    except Exception as e:
        logger.exception("Error updating expense")
        return error_response(f"Failed to update expense: {e}", status=500)


async def handle_delete_expense(
    expense_id: str,
    user_id: str = "default",
) -> HandlerResult:
    """
    Delete an expense.

    DELETE /api/v1/accounting/expenses/{id}
    """
    try:
        tracker = get_expense_tracker()

        deleted = await tracker.delete_expense(expense_id)
        if not deleted:
            return error_response("Expense not found", status=404)

        return success_response({"message": "Expense deleted successfully"})

    except Exception as e:
        logger.exception("Error deleting expense")
        return error_response(f"Failed to delete expense: {e}", status=500)


# =============================================================================
# Approval Workflow
# =============================================================================


async def handle_approve_expense(
    expense_id: str,
    user_id: str = "default",
) -> HandlerResult:
    """
    Approve an expense for sync.

    POST /api/v1/accounting/expenses/{id}/approve
    """
    try:
        tracker = get_expense_tracker()

        expense = await tracker.approve_expense(expense_id)
        if not expense:
            return error_response("Expense not found", status=404)

        return success_response(
            {
                "expense": expense.to_dict(),
                "message": "Expense approved successfully",
            }
        )

    except Exception as e:
        logger.exception("Error approving expense")
        return error_response(f"Failed to approve expense: {e}", status=500)


async def handle_reject_expense(
    expense_id: str,
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Reject an expense.

    POST /api/v1/accounting/expenses/{id}/reject
    Body: {
        reason: str (optional)
    }
    """
    try:
        tracker = get_expense_tracker()

        reason = data.get("reason", "")
        expense = await tracker.reject_expense(expense_id, reason)
        if not expense:
            return error_response("Expense not found", status=404)

        return success_response(
            {
                "expense": expense.to_dict(),
                "message": "Expense rejected",
            }
        )

    except Exception as e:
        logger.exception("Error rejecting expense")
        return error_response(f"Failed to reject expense: {e}", status=500)


async def handle_get_pending_approvals(
    user_id: str = "default",
) -> HandlerResult:
    """
    Get expenses pending approval.

    GET /api/v1/accounting/expenses/pending
    """
    try:
        tracker = get_expense_tracker()

        expenses = await tracker.get_pending_approval()

        return success_response(
            {
                "expenses": [e.to_dict() for e in expenses],
                "count": len(expenses),
            }
        )

    except Exception as e:
        logger.exception("Error getting pending approvals")
        return error_response(f"Failed to get pending approvals: {e}", status=500)


# =============================================================================
# Categorization
# =============================================================================


async def handle_categorize_expenses(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Auto-categorize expenses.

    POST /api/v1/accounting/expenses/categorize
    Body: {
        expense_ids: list[str] (optional, categorize all uncategorized if empty)
    }
    """
    try:
        tracker = get_expense_tracker()

        expense_ids = data.get("expense_ids")
        results = await tracker.bulk_categorize(expense_ids)

        return success_response(
            {
                "categorized": {eid: cat.value for eid, cat in results.items()},
                "count": len(results),
                "message": f"Categorized {len(results)} expenses",
            }
        )

    except Exception as e:
        logger.exception("Error categorizing expenses")
        return error_response(f"Failed to categorize expenses: {e}", status=500)


# =============================================================================
# QBO Sync
# =============================================================================


async def handle_sync_to_qbo(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Sync expenses to QuickBooks Online.

    POST /api/v1/accounting/expenses/sync
    Body: {
        expense_ids: list[str] (optional, sync all approved if empty)
    }
    """
    try:
        tracker = get_expense_tracker()

        expense_ids = data.get("expense_ids")
        result = await tracker.sync_to_qbo(expense_ids=expense_ids)

        return success_response(
            {
                "result": result.to_dict(),
                "message": f"Synced {result.success_count} expenses to QBO",
            }
        )

    except Exception as e:
        logger.exception("Error syncing to QBO")
        return error_response(f"Failed to sync to QBO: {e}", status=500)


# =============================================================================
# Statistics and Export
# =============================================================================


async def handle_get_expense_stats(
    query_params: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Get expense statistics.

    GET /api/v1/accounting/expenses/stats
    Query params:
        start_date: str (ISO format)
        end_date: str (ISO format)
    """
    try:
        tracker = get_expense_tracker()

        start_date = None
        start_date_str = query_params.get("start_date")
        if start_date_str:
            try:
                start_date = datetime.fromisoformat(start_date_str.replace("Z", "+00:00"))
            except ValueError:
                pass

        end_date = None
        end_date_str = query_params.get("end_date")
        if end_date_str:
            try:
                end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            except ValueError:
                pass

        stats = tracker.get_stats(start_date=start_date, end_date=end_date)

        return success_response({"stats": stats.to_dict()})

    except Exception as e:
        logger.exception("Error getting expense stats")
        return error_response(f"Failed to get expense stats: {e}", status=500)


async def handle_export_expenses(
    query_params: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Export expenses to CSV or JSON.

    GET /api/v1/accounting/expenses/export
    Query params:
        format: str (csv or json, default csv)
        start_date: str (ISO format)
        end_date: str (ISO format)
    """
    try:
        tracker = get_expense_tracker()

        export_format = query_params.get("format", "csv")
        if export_format not in ["csv", "json"]:
            return error_response("format must be 'csv' or 'json'", status=400)

        start_date = None
        start_date_str = query_params.get("start_date")
        if start_date_str:
            try:
                start_date = datetime.fromisoformat(start_date_str.replace("Z", "+00:00"))
            except ValueError:
                pass

        end_date = None
        end_date_str = query_params.get("end_date")
        if end_date_str:
            try:
                end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            except ValueError:
                pass

        data = await tracker.export_expenses(
            format=export_format,
            start_date=start_date,
            end_date=end_date,
        )

        return success_response(
            {
                "data": data,
                "format": export_format,
            }
        )

    except Exception as e:
        logger.exception("Error exporting expenses")
        return error_response(f"Failed to export expenses: {e}", status=500)


# =============================================================================
# Handler Class for Router Registration
# =============================================================================


class ExpenseHandler(BaseHandler):
    """Handler for expense-related routes."""

    ROUTES = {
        "/api/v1/accounting/expenses/upload": ["POST"],
        "/api/v1/accounting/expenses": ["GET", "POST"],
        "/api/v1/accounting/expenses/categorize": ["POST"],
        "/api/v1/accounting/expenses/sync": ["POST"],
        "/api/v1/accounting/expenses/stats": ["GET"],
        "/api/v1/accounting/expenses/pending": ["GET"],
        "/api/v1/accounting/expenses/export": ["GET"],
    }

    # Dynamic routes with path params
    DYNAMIC_ROUTES = {
        "/api/v1/accounting/expenses/{expense_id}": ["GET", "PUT", "DELETE"],
        "/api/v1/accounting/expenses/{expense_id}/approve": ["POST"],
        "/api/v1/accounting/expenses/{expense_id}/reject": ["POST"],
    }

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        if path in self.ROUTES:
            return True
        # Check dynamic routes
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

    def _extract_expense_id(self, path: str) -> Optional[str]:
        """Extract expense_id from path."""
        parts = path.split("/")
        # /api/v1/accounting/expenses/{expense_id}/...
        # parts[0]="", [1]="api", [2]="v1", [3]="accounting", [4]="expenses", [5]=expense_id
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

        if path == "/api/v1/accounting/expenses":
            return await handle_list_expenses(query_params)

        if path == "/api/v1/accounting/expenses/stats":
            return await handle_get_expense_stats(query_params)

        if path == "/api/v1/accounting/expenses/pending":
            return await handle_get_pending_approvals()

        if path == "/api/v1/accounting/expenses/export":
            return await handle_export_expenses(query_params)

        # Dynamic: /api/v1/accounting/expenses/{expense_id}
        expense_id = self._extract_expense_id(path)
        if expense_id and "/approve" not in path and "/reject" not in path:
            return await handle_get_expense(expense_id)

        return error_response("Route not found", status=404)

    async def handle_post(  # type: ignore[override]
        self,
        path: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> HandlerResult:
        """Handle POST requests."""
        data = data or {}

        if path == "/api/v1/accounting/expenses/upload":
            return await handle_upload_receipt(data)

        if path == "/api/v1/accounting/expenses":
            return await handle_create_expense(data)

        if path == "/api/v1/accounting/expenses/categorize":
            return await handle_categorize_expenses(data)

        if path == "/api/v1/accounting/expenses/sync":
            return await handle_sync_to_qbo(data)

        # Dynamic routes
        expense_id = self._extract_expense_id(path)
        if expense_id:
            if "/approve" in path:
                return await handle_approve_expense(expense_id)
            if "/reject" in path:
                return await handle_reject_expense(expense_id, data)

        return error_response("Route not found", status=404)

    async def handle_put(  # type: ignore[override]
        self,
        path: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> HandlerResult:
        """Handle PUT requests."""
        data = data or {}

        expense_id = self._extract_expense_id(path)
        if expense_id:
            return await handle_update_expense(expense_id, data)

        return error_response("Route not found", status=404)

    async def handle_delete(  # type: ignore[override]
        self,
        path: str,
    ) -> HandlerResult:
        """Handle DELETE requests."""
        expense_id = self._extract_expense_id(path)
        if expense_id:
            return await handle_delete_expense(expense_id)

        return error_response("Route not found", status=404)
