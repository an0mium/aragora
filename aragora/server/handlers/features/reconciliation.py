"""
Bank Reconciliation API Handler.

Provides REST APIs for bank reconciliation functionality:
- Run reconciliation between bank and book transactions
- View and manage discrepancies
- Resolve discrepancies with AI suggestions
- Generate reconciliation reports

Endpoints:
- POST /api/v1/reconciliation/run          - Run new reconciliation
- GET  /api/v1/reconciliation/list         - List past reconciliations
- GET  /api/v1/reconciliation/{id}         - Get reconciliation details
- GET  /api/v1/reconciliation/{id}/report  - Generate PDF report
- POST /api/v1/reconciliation/{id}/resolve - Resolve a discrepancy
- POST /api/v1/reconciliation/{id}/approve - Approve reconciliation
- GET  /api/v1/reconciliation/discrepancies - Get all pending discrepancies
- POST /api/v1/reconciliation/discrepancies/bulk-resolve - Bulk resolve
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Service Instance Management
# =============================================================================

_service_instances: Dict[str, Any] = {}  # tenant_id -> ReconciliationService


def get_reconciliation_service(tenant_id: str):
    """Get or create reconciliation service for tenant."""
    if tenant_id not in _service_instances:
        try:
            from aragora.services.accounting.reconciliation import ReconciliationService

            _service_instances[tenant_id] = ReconciliationService()
        except ImportError:
            return None
    return _service_instances[tenant_id]


# =============================================================================
# Handler Class
# =============================================================================


class ReconciliationHandler(BaseHandler):
    """Handler for bank reconciliation API endpoints."""

    ROUTES = [
        "/api/v1/reconciliation/run",
        "/api/v1/reconciliation/list",
        "/api/v1/reconciliation/{reconciliation_id}",
        "/api/v1/reconciliation/{reconciliation_id}/report",
        "/api/v1/reconciliation/{reconciliation_id}/resolve",
        "/api/v1/reconciliation/{reconciliation_id}/approve",
        "/api/v1/reconciliation/discrepancies",
        "/api/v1/reconciliation/discrepancies/bulk-resolve",
        "/api/v1/reconciliation/demo",
    ]

    def __init__(self, server_context: Optional[Dict[str, Any]] = None):
        """Initialize handler with optional server context."""
        super().__init__(server_context or {})  # type: ignore[arg-type]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return path.startswith("/api/v1/reconciliation")

    async def handle(self, request: Any, path: str, method: str) -> HandlerResult:  # type: ignore[override]
        """Route requests to appropriate handler methods."""
        try:
            tenant_id = self._get_tenant_id(request)

            # Run reconciliation
            if path == "/api/v1/reconciliation/run" and method == "POST":
                return await self._handle_run(request, tenant_id)

            # List reconciliations
            elif path == "/api/v1/reconciliation/list" and method == "GET":
                return await self._handle_list(request, tenant_id)

            # Demo data
            elif path == "/api/v1/reconciliation/demo" and method == "GET":
                return await self._handle_demo(request, tenant_id)

            # Get discrepancies
            elif path == "/api/v1/reconciliation/discrepancies" and method == "GET":
                return await self._handle_discrepancies(request, tenant_id)

            # Bulk resolve discrepancies
            elif path == "/api/v1/reconciliation/discrepancies/bulk-resolve" and method == "POST":
                return await self._handle_bulk_resolve(request, tenant_id)

            # Reconciliation-specific paths
            elif path.startswith("/api/v1/reconciliation/"):
                parts = path.split("/")
                if len(parts) >= 5:
                    reconciliation_id = parts[4]

                    if len(parts) == 5 and method == "GET":
                        return await self._handle_get(request, tenant_id, reconciliation_id)

                    elif len(parts) == 6:
                        action = parts[5]
                        if action == "report" and method == "GET":
                            return await self._handle_report(request, tenant_id, reconciliation_id)
                        elif action == "resolve" and method == "POST":
                            return await self._handle_resolve(request, tenant_id, reconciliation_id)
                        elif action == "approve" and method == "POST":
                            return await self._handle_approve(request, tenant_id, reconciliation_id)

            return error_response("Not found", 404)

        except Exception as e:
            logger.exception(f"Error in reconciliation handler: {e}")
            return error_response(f"Internal error: {str(e)}", 500)

    def _get_tenant_id(self, request: Any) -> str:
        """Extract tenant ID from request context."""
        return getattr(request, "tenant_id", "default")

    # =========================================================================
    # Run Reconciliation
    # =========================================================================

    async def _handle_run(self, request: Any, tenant_id: str) -> HandlerResult:
        """Run a new bank reconciliation.

        Request body:
        {
            "account_id": "...",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "use_agents": true,
            "plaid_access_token": "..."  // Optional, use stored if not provided
        }
        """
        try:
            body = await self._get_json_body(request)

            # Parse dates
            start_date_str = body.get("start_date")
            end_date_str = body.get("end_date")

            if not start_date_str or not end_date_str:
                return error_response("start_date and end_date are required", 400)

            try:
                start_date = date.fromisoformat(start_date_str)
                end_date = date.fromisoformat(end_date_str)
            except ValueError:
                return error_response("Invalid date format. Use YYYY-MM-DD", 400)

            if end_date < start_date:
                return error_response("end_date must be after start_date", 400)

            account_id = body.get("account_id")
            use_agents = body.get("use_agents", True)

            # Get service
            service = get_reconciliation_service(tenant_id)
            if not service:
                return error_response("Reconciliation service not available", 503)

            # For demo purposes, return mock data if no plaid credentials
            plaid_token = body.get("plaid_access_token")
            if not plaid_token:
                # Return mock result
                from aragora.services.accounting.reconciliation import (
                    get_mock_reconciliation_result,
                )

                mock_result = get_mock_reconciliation_result()
                mock_result.start_date = start_date
                mock_result.end_date = end_date

                return success_response(
                    {
                        "reconciliation": mock_result.to_dict(),
                        "discrepancies": [d.to_dict() for d in mock_result.discrepancies],
                        "matched_transactions": [
                            m.to_dict() for m in mock_result.matched_transactions[:5]
                        ],
                        "is_demo": True,
                    }
                )

            # Run actual reconciliation
            from aragora.connectors.accounting.plaid import PlaidCredentials

            credentials = PlaidCredentials(
                access_token=plaid_token,
                item_id="",
                institution_id=body.get("institution_id", ""),
                institution_name=body.get("institution_name", ""),
                user_id=getattr(request, "user_id", "api_user"),
                tenant_id=tenant_id,
            )

            result = await service.reconcile(
                plaid_credentials=credentials,
                start_date=start_date,
                end_date=end_date,
                account_id=account_id,
                use_agents=use_agents,
            )

            return success_response(
                {
                    "reconciliation": result.to_dict(),
                    "discrepancies": [d.to_dict() for d in result.discrepancies],
                    "matched_count": result.matched_count,
                }
            )

        except ImportError as e:
            return error_response(f"Required module not available: {e}", 503)
        except Exception as e:
            logger.exception(f"Error running reconciliation: {e}")
            return error_response(f"Reconciliation failed: {str(e)}", 500)

    # =========================================================================
    # List/Get Reconciliations
    # =========================================================================

    async def _handle_list(self, request: Any, tenant_id: str) -> HandlerResult:
        """List past reconciliations."""
        params = self._get_query_params(request)
        account_id = params.get("account_id")
        limit = int(params.get("limit", 20))

        service = get_reconciliation_service(tenant_id)
        if not service:
            return success_response({"reconciliations": [], "total": 0})

        results = service.list_reconciliations(account_id=account_id, limit=limit)

        return success_response(
            {
                "reconciliations": [r.to_dict() for r in results],
                "total": len(results),
            }
        )

    async def _handle_get(
        self, request: Any, tenant_id: str, reconciliation_id: str
    ) -> HandlerResult:
        """Get reconciliation details."""
        service = get_reconciliation_service(tenant_id)
        if not service:
            return error_response("Service not available", 503)

        result = service.get_reconciliation(reconciliation_id)
        if not result:
            return error_response("Reconciliation not found", 404)

        return success_response(
            {
                "reconciliation": result.to_dict(),
                "discrepancies": [d.to_dict() for d in result.discrepancies],
                "matched_transactions": [m.to_dict() for m in result.matched_transactions],
            }
        )

    async def _handle_demo(self, request: Any, tenant_id: str) -> HandlerResult:
        """Get demo reconciliation data."""
        try:
            from aragora.services.accounting.reconciliation import get_mock_reconciliation_result

            result = get_mock_reconciliation_result()

            return success_response(
                {
                    "reconciliation": result.to_dict(),
                    "discrepancies": [d.to_dict() for d in result.discrepancies],
                    "matched_transactions": [m.to_dict() for m in result.matched_transactions],
                    "is_demo": True,
                }
            )
        except ImportError:
            return error_response("Demo data not available", 503)

    # =========================================================================
    # Resolve Discrepancies
    # =========================================================================

    async def _handle_resolve(
        self, request: Any, tenant_id: str, reconciliation_id: str
    ) -> HandlerResult:
        """Resolve a discrepancy.

        Request body:
        {
            "discrepancy_id": "disc_001",
            "resolution": "Created expense entry for office supplies",
            "action": "create_entry" | "ignore" | "match_manual"
        }
        """
        try:
            body = await self._get_json_body(request)

            discrepancy_id = body.get("discrepancy_id")
            resolution = body.get("resolution", "")
            action = body.get("action", "resolve")

            if not discrepancy_id:
                return error_response("discrepancy_id is required", 400)

            service = get_reconciliation_service(tenant_id)
            if not service:
                return error_response("Service not available", 503)

            # Get user info
            user_id = getattr(request, "user_id", "api_user")

            success = await service.resolve_discrepancy(
                discrepancy_id=discrepancy_id,
                reconciliation_id=reconciliation_id,
                resolution=f"[{action}] {resolution}",
                resolved_by=user_id,
            )

            if success:
                # Get updated reconciliation
                result = service.get_reconciliation(reconciliation_id)
                return success_response(
                    {
                        "status": "resolved",
                        "discrepancy_id": discrepancy_id,
                        "reconciliation_status": {
                            "is_reconciled": result.is_reconciled if result else False,
                            "pending_discrepancies": len(
                                [
                                    d
                                    for d in (result.discrepancies if result else [])
                                    if d.resolution_status.value == "pending"
                                ]
                            ),
                        },
                    }
                )
            else:
                return error_response("Failed to resolve discrepancy", 400)

        except Exception as e:
            logger.exception(f"Error resolving discrepancy: {e}")
            return error_response(f"Resolution failed: {str(e)}", 500)

    async def _handle_bulk_resolve(self, request: Any, tenant_id: str) -> HandlerResult:
        """Bulk resolve discrepancies.

        Request body:
        {
            "reconciliation_id": "...",
            "resolutions": [
                {"discrepancy_id": "disc_001", "resolution": "...", "action": "..."},
                ...
            ]
        }
        """
        try:
            body = await self._get_json_body(request)

            reconciliation_id = body.get("reconciliation_id")
            resolutions = body.get("resolutions", [])

            if not reconciliation_id:
                return error_response("reconciliation_id is required", 400)

            service = get_reconciliation_service(tenant_id)
            if not service:
                return error_response("Service not available", 503)

            user_id = getattr(request, "user_id", "api_user")
            success_count = 0
            errors = []

            for item in resolutions:
                discrepancy_id = item.get("discrepancy_id")
                resolution = item.get("resolution", "")
                action = item.get("action", "resolve")

                success = await service.resolve_discrepancy(
                    discrepancy_id=discrepancy_id,
                    reconciliation_id=reconciliation_id,
                    resolution=f"[{action}] {resolution}",
                    resolved_by=user_id,
                )

                if success:
                    success_count += 1
                else:
                    errors.append({"discrepancy_id": discrepancy_id, "error": "Resolution failed"})

            result = service.get_reconciliation(reconciliation_id)

            return success_response(
                {
                    "resolved_count": success_count,
                    "error_count": len(errors),
                    "errors": errors if errors else None,
                    "reconciliation_status": {
                        "is_reconciled": result.is_reconciled if result else False,
                    },
                }
            )

        except Exception as e:
            logger.exception(f"Error in bulk resolve: {e}")
            return error_response(f"Bulk resolution failed: {str(e)}", 500)

    # =========================================================================
    # Approve Reconciliation
    # =========================================================================

    async def _handle_approve(
        self, request: Any, tenant_id: str, reconciliation_id: str
    ) -> HandlerResult:
        """Approve/finalize a reconciliation.

        Request body:
        {
            "notes": "Reviewed and approved by finance team"
        }
        """
        try:
            body = await self._get_json_body(request)
            notes = body.get("notes", "")

            service = get_reconciliation_service(tenant_id)
            if not service:
                return error_response("Service not available", 503)

            result = service.get_reconciliation(reconciliation_id)
            if not result:
                return error_response("Reconciliation not found", 404)

            # Check if there are unresolved discrepancies
            pending = [d for d in result.discrepancies if d.resolution_status.value == "pending"]

            if pending:
                return error_response(
                    f"Cannot approve: {len(pending)} unresolved discrepancies", 400
                )

            # Mark as reconciled
            user_id = getattr(request, "user_id", "api_user")
            result.is_reconciled = True
            result.reconciled_at = datetime.now(timezone.utc)
            result.reconciled_by = user_id

            logger.info(f"[Reconciliation] Approved {reconciliation_id} by {user_id}: {notes}")

            return success_response(
                {
                    "status": "approved",
                    "reconciliation_id": reconciliation_id,
                    "approved_by": user_id,
                    "approved_at": result.reconciled_at.isoformat(),
                }
            )

        except Exception as e:
            logger.exception(f"Error approving reconciliation: {e}")
            return error_response(f"Approval failed: {str(e)}", 500)

    # =========================================================================
    # Get Discrepancies
    # =========================================================================

    async def _handle_discrepancies(self, request: Any, tenant_id: str) -> HandlerResult:
        """Get all pending discrepancies across reconciliations."""
        params = self._get_query_params(request)
        status_filter = params.get("status", "pending")
        severity_filter = params.get("severity")
        limit = int(params.get("limit", 50))

        service = get_reconciliation_service(tenant_id)
        if not service:
            return success_response({"discrepancies": [], "total": 0})

        all_discrepancies = []

        for result in service.list_reconciliations(limit=100):
            for disc in result.discrepancies:
                # Filter by status
                if status_filter and disc.resolution_status.value != status_filter:
                    continue

                # Filter by severity
                if severity_filter and disc.severity.value != severity_filter:
                    continue

                all_discrepancies.append(
                    {
                        **disc.to_dict(),
                        "reconciliation_id": result.reconciliation_id,
                        "account_name": result.account_name,
                        "period": f"{result.start_date} to {result.end_date}",
                    }
                )

        # Sort by severity (critical first)
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        all_discrepancies.sort(key=lambda d: severity_order.get(d["severity"], 4))

        return success_response(
            {
                "discrepancies": all_discrepancies[:limit],
                "total": len(all_discrepancies),
            }
        )

    # =========================================================================
    # Generate Report
    # =========================================================================

    async def _handle_report(
        self, request: Any, tenant_id: str, reconciliation_id: str
    ) -> HandlerResult:
        """Generate reconciliation report."""
        params = self._get_query_params(request)
        format_type = params.get("format", "json")

        service = get_reconciliation_service(tenant_id)
        if not service:
            return error_response("Service not available", 503)

        result = service.get_reconciliation(reconciliation_id)
        if not result:
            return error_response("Reconciliation not found", 404)

        if format_type == "json":
            report = {
                "title": f"Bank Reconciliation Report - {result.account_name}",
                "period": {
                    "start_date": result.start_date.isoformat(),
                    "end_date": result.end_date.isoformat(),
                },
                "summary": {
                    "bank_balance": float(result.bank_total),
                    "book_balance": float(result.book_total),
                    "difference": float(result.difference),
                    "transactions_matched": result.matched_count,
                    "discrepancies_found": result.discrepancy_count,
                    "match_rate": f"{result.match_rate * 100:.1f}%",
                    "is_reconciled": result.is_reconciled,
                },
                "discrepancies": [d.to_dict() for d in result.discrepancies],
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

            return success_response({"report": report})

        elif format_type == "csv":
            # Generate CSV format
            lines = [
                "Type,Description,Bank Amount,Book Amount,Bank Date,Book Date,Status",
            ]

            for disc in result.discrepancies:
                lines.append(
                    f"{disc.discrepancy_type.value},"
                    f'"{disc.description}",'
                    f"{disc.bank_amount or ''},"
                    f"{disc.book_amount or ''},"
                    f"{disc.bank_date or ''},"
                    f"{disc.book_date or ''},"
                    f"{disc.resolution_status.value}"
                )

            csv_content = "\n".join(lines)

            return HandlerResult(
                body=csv_content.encode("utf-8"),
                status_code=200,
                content_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=reconciliation_{reconciliation_id}.csv"
                },
            )

        else:
            return error_response(f"Unsupported format: {format_type}", 400)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def _get_json_body(self, request: Any) -> Dict[str, Any]:
        """Extract JSON body from request."""
        if hasattr(request, "json"):
            if callable(request.json):
                return await request.json()
            return request.json
        return {}

    def _get_query_params(self, request: Any) -> Dict[str, str]:
        """Extract query parameters from request."""
        if hasattr(request, "query"):
            return dict(request.query)
        if hasattr(request, "args"):
            return dict(request.args)
        return {}


# =============================================================================
# Handler Registration
# =============================================================================

_handler_instance: Optional[ReconciliationHandler] = None


def get_reconciliation_handler() -> ReconciliationHandler:
    """Get or create handler instance."""
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = ReconciliationHandler()
    return _handler_instance


async def handle_reconciliation(request: Any, path: str, method: str) -> HandlerResult:
    """Entry point for reconciliation requests."""
    handler = get_reconciliation_handler()
    return await handler.handle(request, path, method)


__all__ = [
    "ReconciliationHandler",
    "handle_reconciliation",
    "get_reconciliation_handler",
]
