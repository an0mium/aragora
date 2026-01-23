"""
Audit Trail HTTP Handlers for Aragora.

Provides REST API endpoints for audit trail access and verification:
- List and retrieve audit trails
- Export audit trails in multiple formats
- Verify audit trail integrity
- List and retrieve decision receipts

Endpoints:
    GET  /api/v1/audit-trails                    - List recent audit trails
    GET  /api/v1/audit-trails/:trail_id          - Get specific audit trail
    GET  /api/v1/audit-trails/:trail_id/export   - Export (format=json|csv|md)
    POST /api/v1/audit-trails/:trail_id/verify   - Verify integrity checksum

    GET  /api/v1/receipts                        - List recent decision receipts
    GET  /api/v1/receipts/:receipt_id            - Get specific receipt
    POST /api/v1/receipts/:receipt_id/verify     - Verify receipt integrity

These endpoints surface the "defensible decisions" pillar of Aragora's
control plane positioning, providing full audit trails with cryptographic
integrity verification for compliance documentation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    ServerContext,
    error_response,
    json_response,
)
from aragora.server.handlers.utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)


class AuditTrailHandler(BaseHandler):
    """
    HTTP handler for audit trail operations.

    Provides REST API access to audit trails and decision receipts
    for compliance documentation and integrity verification.
    """

    # In-memory storage for demo purposes
    # In production, this would be backed by a database
    _trails: Dict[str, Dict[str, Any]] = {}
    _receipts: Dict[str, Dict[str, Any]] = {}

    def __init__(self, server_context: ServerContext):
        """Initialize with server context."""
        super().__init__(server_context)

    def can_handle(self, method: str, path: str) -> bool:
        """Check if this handler can process the request."""
        if path.startswith("/api/v1/audit-trails"):
            return method in ("GET", "POST")
        if path.startswith("/api/v1/receipts"):
            return method in ("GET", "POST")
        return False

    @rate_limit(requests_per_minute=60)
    async def handle(  # type: ignore[override]
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        query_params: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> HandlerResult:
        """Route request to appropriate handler method."""
        query_params = query_params or {}

        try:
            # Audit Trail Routes
            if path == "/api/v1/audit-trails" and method == "GET":
                return await self._list_audit_trails(query_params)

            if path.startswith("/api/v1/audit-trails/") and "/export" in path:
                trail_id = path.split("/")[4]
                return await self._export_audit_trail(trail_id, query_params)

            if path.startswith("/api/v1/audit-trails/") and "/verify" in path:
                trail_id = path.split("/")[4]
                return await self._verify_audit_trail(trail_id)

            if path.startswith("/api/v1/audit-trails/"):
                trail_id = path.split("/")[4]
                return await self._get_audit_trail(trail_id)

            # Receipt Routes
            if path == "/api/v1/receipts" and method == "GET":
                return await self._list_receipts(query_params)

            if path.startswith("/api/v1/receipts/") and "/verify" in path:
                receipt_id = path.split("/")[4]
                return await self._verify_receipt(receipt_id)

            if path.startswith("/api/v1/receipts/"):
                receipt_id = path.split("/")[4]
                return await self._get_receipt(receipt_id)

            return error_response("Not found", 404)

        except Exception as e:
            logger.exception(f"Error handling audit trail request: {e}")
            return error_response(f"Internal error: {str(e)}", 500)

    async def _list_audit_trails(self, query_params: Dict[str, str]) -> HandlerResult:
        """List recent audit trails with pagination."""
        limit = int(query_params.get("limit", "20"))
        offset = int(query_params.get("offset", "0"))

        # Get trails from storage (would be database in production)
        trails = list(self._trails.values())

        # Sort by created_at descending
        trails.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        # Paginate
        paginated = trails[offset : offset + limit]

        # Return summary for each trail (not full event list)
        summaries = [
            {
                "trail_id": t.get("trail_id"),
                "gauntlet_id": t.get("gauntlet_id"),
                "created_at": t.get("created_at"),
                "verdict": t.get("verdict"),
                "confidence": t.get("confidence"),
                "total_findings": t.get("total_findings"),
                "duration_seconds": t.get("duration_seconds"),
                "checksum": t.get("checksum"),
            }
            for t in paginated
        ]

        return json_response(
            {
                "trails": summaries,
                "total": len(trails),
                "limit": limit,
                "offset": offset,
            }
        )

    async def _get_audit_trail(self, trail_id: str) -> HandlerResult:
        """Get a specific audit trail by ID."""
        trail = self._trails.get(trail_id)

        if not trail:
            # Try to load from gauntlet results if available
            trail = await self._load_trail_from_gauntlet(trail_id)

        if not trail:
            return error_response(f"Audit trail not found: {trail_id}", 404)

        return json_response(trail)

    async def _export_audit_trail(
        self, trail_id: str, query_params: Dict[str, str]
    ) -> HandlerResult:
        """Export audit trail in specified format."""
        format_type = query_params.get("format", "json")

        trail = self._trails.get(trail_id)
        if not trail:
            trail = await self._load_trail_from_gauntlet(trail_id)

        if not trail:
            return error_response(f"Audit trail not found: {trail_id}", 404)

        try:
            from aragora.export.audit_trail import AuditTrail

            # Reconstruct AuditTrail object
            audit_trail = AuditTrail.from_json(__import__("json").dumps(trail))

            if format_type == "json":
                return HandlerResult(
                    status_code=200,
                    content_type="application/json",
                    body=audit_trail.to_json().encode(),
                    headers={
                        "Content-Disposition": f'attachment; filename="{trail_id}.json"',
                    },
                )
            elif format_type == "csv":
                return HandlerResult(
                    status_code=200,
                    content_type="text/csv",
                    body=audit_trail.to_csv().encode(),
                    headers={
                        "Content-Disposition": f'attachment; filename="{trail_id}.csv"',
                    },
                )
            elif format_type in ("md", "markdown"):
                return HandlerResult(
                    status_code=200,
                    content_type="text/markdown",
                    body=audit_trail.to_markdown().encode(),
                    headers={
                        "Content-Disposition": f'attachment; filename="{trail_id}.md"',
                    },
                )
            else:
                return error_response(f"Unknown format: {format_type}. Use json, csv, or md.", 400)

        except ImportError:
            # Fallback if export module not available
            import json

            return HandlerResult(
                status_code=200,
                content_type="application/json",
                body=json.dumps(trail, indent=2).encode(),
                headers={
                    "Content-Disposition": f'attachment; filename="{trail_id}.json"',
                },
            )

    async def _verify_audit_trail(self, trail_id: str) -> HandlerResult:
        """Verify audit trail integrity checksum."""
        trail = self._trails.get(trail_id)
        if not trail:
            trail = await self._load_trail_from_gauntlet(trail_id)

        if not trail:
            return error_response(f"Audit trail not found: {trail_id}", 404)

        try:
            from aragora.export.audit_trail import AuditTrail

            audit_trail = AuditTrail.from_json(__import__("json").dumps(trail))

            is_valid = audit_trail.verify_integrity()
            stored_checksum = trail.get("checksum", "")
            computed_checksum = audit_trail.checksum

            return json_response(
                {
                    "trail_id": trail_id,
                    "valid": is_valid,
                    "stored_checksum": stored_checksum,
                    "computed_checksum": computed_checksum,
                    "match": stored_checksum == computed_checksum,
                }
            )

        except Exception as e:
            logger.error(f"Error verifying audit trail {trail_id}: {e}")
            return json_response(
                {
                    "trail_id": trail_id,
                    "valid": False,
                    "error": str(e),
                }
            )

    async def _list_receipts(self, query_params: Dict[str, str]) -> HandlerResult:
        """List recent decision receipts with pagination."""
        limit = int(query_params.get("limit", "20"))
        offset = int(query_params.get("offset", "0"))

        receipts = list(self._receipts.values())
        receipts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        paginated = receipts[offset : offset + limit]

        summaries = [
            {
                "receipt_id": r.get("receipt_id"),
                "gauntlet_id": r.get("gauntlet_id"),
                "timestamp": r.get("timestamp"),
                "verdict": r.get("verdict"),
                "confidence": r.get("confidence"),
                "risk_level": r.get("risk_level"),
                "findings_count": len(r.get("findings", [])),
                "checksum": r.get("checksum"),
            }
            for r in paginated
        ]

        return json_response(
            {
                "receipts": summaries,
                "total": len(receipts),
                "limit": limit,
                "offset": offset,
            }
        )

    async def _get_receipt(self, receipt_id: str) -> HandlerResult:
        """Get a specific decision receipt by ID."""
        receipt = self._receipts.get(receipt_id)

        if not receipt:
            receipt = await self._load_receipt_from_gauntlet(receipt_id)

        if not receipt:
            return error_response(f"Receipt not found: {receipt_id}", 404)

        return json_response(receipt)

    async def _verify_receipt(self, receipt_id: str) -> HandlerResult:
        """Verify decision receipt integrity."""
        receipt = self._receipts.get(receipt_id)
        if not receipt:
            receipt = await self._load_receipt_from_gauntlet(receipt_id)

        if not receipt:
            return error_response(f"Receipt not found: {receipt_id}", 404)

        try:
            # Compute checksum and verify
            import hashlib
            import json

            content = json.dumps(
                {
                    "receipt_id": receipt.get("receipt_id"),
                    "gauntlet_id": receipt.get("gauntlet_id"),
                    "verdict": receipt.get("verdict"),
                    "confidence": receipt.get("confidence"),
                },
                sort_keys=True,
            )
            computed = hashlib.sha256(content.encode()).hexdigest()[:16]
            stored = receipt.get("checksum", "")

            return json_response(
                {
                    "receipt_id": receipt_id,
                    "valid": computed == stored,
                    "stored_checksum": stored,
                    "computed_checksum": computed,
                    "match": computed == stored,
                }
            )

        except Exception as e:
            logger.error(f"Error verifying receipt {receipt_id}: {e}")
            return json_response(
                {
                    "receipt_id": receipt_id,
                    "valid": False,
                    "error": str(e),
                }
            )

    async def _load_trail_from_gauntlet(self, trail_id: str) -> Optional[Dict[str, Any]]:
        """
        Try to load audit trail from gauntlet results.

        In production, this would query the database for historical results.
        """
        # Extract gauntlet_id from trail_id (format: trail-{gauntlet_id})
        if trail_id.startswith("trail-"):
            gauntlet_id = trail_id[6:]
        else:
            gauntlet_id = trail_id

        # Try to get from gauntlet handler's result cache
        gauntlet_handler = self.ctx.get("gauntlet_handler")
        if gauntlet_handler and hasattr(gauntlet_handler, "_results"):
            result = gauntlet_handler._results.get(gauntlet_id)
            if result:
                try:
                    from aragora.export.audit_trail import generate_audit_trail

                    trail = generate_audit_trail(result)
                    trail_dict = trail.to_dict()
                    # Cache it
                    self._trails[trail_id] = trail_dict
                    return trail_dict
                except Exception as e:
                    logger.debug(f"Could not generate audit trail: {e}")

        return None

    async def _load_receipt_from_gauntlet(self, receipt_id: str) -> Optional[Dict[str, Any]]:
        """
        Try to load decision receipt from gauntlet results.

        In production, this would query the database for historical results.
        """
        # Extract gauntlet_id from receipt_id (format: receipt-{gauntlet_id})
        if receipt_id.startswith("receipt-"):
            gauntlet_id = receipt_id[8:]
        else:
            gauntlet_id = receipt_id

        gauntlet_handler = self.ctx.get("gauntlet_handler")
        if gauntlet_handler and hasattr(gauntlet_handler, "_results"):
            result = gauntlet_handler._results.get(gauntlet_id)
            if result:
                try:
                    from aragora.export.decision_receipt import (
                        generate_decision_receipt,
                    )

                    receipt = generate_decision_receipt(result)
                    receipt_dict = receipt.to_dict()
                    self._receipts[receipt_id] = receipt_dict
                    return receipt_dict
                except Exception as e:
                    logger.debug(f"Could not generate receipt: {e}")

        return None

    @classmethod
    def store_trail(cls, trail_id: str, trail_data: Dict[str, Any]) -> None:
        """Store an audit trail (called from gauntlet handler)."""
        cls._trails[trail_id] = trail_data

    @classmethod
    def store_receipt(cls, receipt_id: str, receipt_data: Dict[str, Any]) -> None:
        """Store a decision receipt (called from gauntlet handler)."""
        cls._receipts[receipt_id] = receipt_data
