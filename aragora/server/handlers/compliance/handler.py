"""
Compliance HTTP Handlers for Aragora.

Provides REST API endpoints for compliance and audit operations:
- SOC 2 Type II report generation
- GDPR data export requests
- GDPR Right-to-be-Forgotten workflow
- Audit trail verification
- SIEM-compatible event export
- Legal hold management

Endpoints:
    GET  /api/v2/compliance/status               - Overall compliance status
    GET  /api/v2/compliance/soc2-report          - Generate SOC 2 compliance summary
    GET  /api/v2/compliance/gdpr-export          - Export user data for GDPR
    POST /api/v2/compliance/gdpr/right-to-be-forgotten - Execute GDPR right to erasure
    POST /api/v2/compliance/audit-verify         - Verify audit trail integrity
    GET  /api/v2/compliance/audit-events         - Export audit events (Elasticsearch/SIEM)
    GET  /api/v2/compliance/gdpr/deletions       - List scheduled deletions
    GET  /api/v2/compliance/gdpr/deletions/:id   - Get deletion request
    POST /api/v2/compliance/gdpr/deletions/:id/cancel - Cancel deletion
    GET  /api/v2/compliance/gdpr/legal-holds     - List legal holds
    POST /api/v2/compliance/gdpr/legal-holds     - Create legal hold
    DELETE /api/v2/compliance/gdpr/legal-holds/:id - Release legal hold
    POST /api/v2/compliance/gdpr/coordinated-deletion - Backup-aware deletion
    POST /api/v2/compliance/gdpr/execute-pending - Execute pending deletions
    GET  /api/v2/compliance/gdpr/backup-exclusions - List backup exclusions
    POST /api/v2/compliance/gdpr/backup-exclusions - Add backup exclusion

These endpoints support enterprise compliance requirements.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from aragora.server.handlers.base import BaseHandler, HandlerResult, error_response
from aragora.server.handlers.utils.rate_limit import rate_limit
from aragora.rbac.decorators import PermissionDeniedError
from aragora.observability.metrics import track_handler

from .soc2 import SOC2Mixin
from .gdpr import GDPRMixin
from .legal_hold import LegalHoldMixin
from .audit_verify import AuditVerifyMixin, parse_timestamp

logger = logging.getLogger(__name__)


class ComplianceHandler(
    BaseHandler,
    SOC2Mixin,
    GDPRMixin,
    LegalHoldMixin,
    AuditVerifyMixin,
):
    """
    HTTP handler for compliance and audit operations.

    Provides REST API access to compliance reports, GDPR exports,
    and audit verification.

    Uses mixins to organize functionality:
    - SOC2Mixin: SOC 2 Type II report generation and control evaluation
    - GDPRMixin: GDPR data export, right-to-be-forgotten, deletion management
    - LegalHoldMixin: Legal hold creation, listing, and release
    - AuditVerifyMixin: Audit trail verification and SIEM event export
    """

    ROUTES = [
        "/api/v2/compliance",
        "/api/v2/compliance/*",
    ]

    def __init__(self, server_context: dict[str, Any]):
        """Initialize with server context."""
        super().__init__(server_context)

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the request."""
        if path.startswith("/api/v2/compliance"):
            return method in ("GET", "POST", "DELETE")
        return False

    @track_handler("compliance/main", method="GET")
    @rate_limit(requests_per_minute=20)
    async def handle(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult:
        """Route request to appropriate handler method."""
        method: str = getattr(handler, "command", "GET") if handler else "GET"
        body: dict[str, Any] = (self.read_json_body(handler) or {}) if handler else {}
        headers: Optional[dict[str, str]] = (
            dict(handler.headers) if handler and hasattr(handler, "headers") else None
        )
        query_params = query_params or {}

        try:
            # Status endpoint
            if path == "/api/v2/compliance/status" and method == "GET":
                return await self._get_status()

            # SOC 2 report endpoint
            if path == "/api/v2/compliance/soc2-report" and method == "GET":
                return await self._get_soc2_report(query_params)

            # GDPR export endpoint
            if path == "/api/v2/compliance/gdpr-export" and method == "GET":
                return await self._gdpr_export(query_params)

            # Audit verify endpoint
            if path == "/api/v2/compliance/audit-verify" and method == "POST":
                return await self._verify_audit(body)

            # Audit events endpoint (SIEM)
            if path == "/api/v2/compliance/audit-events" and method == "GET":
                return await self._get_audit_events(query_params)

            # GDPR Right-to-be-Forgotten endpoint
            if path == "/api/v2/compliance/gdpr/right-to-be-forgotten" and method == "POST":
                return await self._right_to_be_forgotten(body)

            # GDPR Deletion Management endpoints
            if path == "/api/v2/compliance/gdpr/deletions" and method == "GET":
                return await self._list_deletions(query_params)

            if path.startswith("/api/v2/compliance/gdpr/deletions/") and path.endswith("/cancel"):
                if method == "POST":
                    request_id = path.split("/")[-2]
                    return await self._cancel_deletion(request_id, body)

            if path.startswith("/api/v2/compliance/gdpr/deletions/") and method == "GET":
                request_id = path.split("/")[-1]
                return await self._get_deletion(request_id)

            # Legal Hold Management endpoints
            if path == "/api/v2/compliance/gdpr/legal-holds" and method == "GET":
                return await self._list_legal_holds(query_params)

            if path == "/api/v2/compliance/gdpr/legal-holds" and method == "POST":
                return await self._create_legal_hold(body, headers)

            if path.startswith("/api/v2/compliance/gdpr/legal-holds/") and method == "DELETE":
                hold_id = path.split("/")[-1]
                return await self._release_legal_hold(hold_id, body)

            # Coordinated deletion endpoint (backup-aware)
            if path == "/api/v2/compliance/gdpr/coordinated-deletion" and method == "POST":
                return await self._coordinated_deletion(body)

            # Execute pending deletions (for background job or manual trigger)
            if path == "/api/v2/compliance/gdpr/execute-pending" and method == "POST":
                return await self._execute_pending_deletions(body)

            # Backup exclusion management
            if path == "/api/v2/compliance/gdpr/backup-exclusions" and method == "GET":
                return await self._list_backup_exclusions(query_params)

            if path == "/api/v2/compliance/gdpr/backup-exclusions" and method == "POST":
                return await self._add_backup_exclusion(body)

            return error_response("Not found", 404)

        except PermissionDeniedError as e:
            logger.warning(f"Permission denied for compliance request: {e}")
            return error_response(str(e), 403)

        except Exception as e:
            logger.exception(f"Error handling compliance request: {e}")
            return error_response(f"Internal error: {str(e)}", 500)

    # Backward compatible timestamp parser (used in tests)
    _parse_timestamp = staticmethod(parse_timestamp)


def create_compliance_handler(server_context: dict[str, Any]) -> ComplianceHandler:
    """Factory function for handler registration."""
    return ComplianceHandler(server_context)


__all__ = ["ComplianceHandler", "create_compliance_handler"]
