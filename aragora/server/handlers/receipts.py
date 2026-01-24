"""
Decision Receipt HTTP Handlers for Aragora.

Provides REST API endpoints for decision receipt management:
- List and retrieve receipts with filtering
- Verify receipt integrity and signatures
- Export receipts in multiple formats
- Batch verification operations

Endpoints:
    GET  /api/v2/receipts                              - List receipts with filters
    GET  /api/v2/receipts/:receipt_id                  - Get specific receipt
    GET  /api/v2/receipts/:receipt_id/export           - Export (format=json|html|md|pdf)
    POST /api/v2/receipts/:receipt_id/verify           - Verify integrity checksum
    POST /api/v2/receipts/:receipt_id/verify-signature - Verify cryptographic signature
    POST /api/v2/receipts/verify-batch                 - Batch signature verification
    GET  /api/v2/receipts/stats                        - Receipt statistics

These endpoints support the "defensible decisions" pillar with:
- Cryptographic signature verification
- 7-year retention for compliance
- Full audit trail integration
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
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


class ReceiptsHandler(BaseHandler):
    """
    HTTP handler for decision receipt operations.

    Provides REST API access to decision receipts with signature
    verification and export capabilities.
    """

    ROUTES = [
        "/api/v2/receipts",
        "/api/v2/receipts/*",
    ]

    def __init__(self, server_context: ServerContext):
        """Initialize with server context."""
        super().__init__(server_context)
        self._store = None  # Lazy initialization

    def _get_store(self):
        """Get or create receipt store (lazy initialization)."""
        if self._store is None:
            from aragora.storage.receipt_store import get_receipt_store

            self._store = get_receipt_store()
        return self._store

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the request."""
        if path.startswith("/api/v2/receipts"):
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
        body = body or {}

        try:
            # Stats endpoint
            if path == "/api/v2/receipts/stats" and method == "GET":
                return await self._get_stats()

            # Batch verification
            if path == "/api/v2/receipts/verify-batch" and method == "POST":
                return await self._verify_batch(body)

            # List receipts
            if path == "/api/v2/receipts" and method == "GET":
                return await self._list_receipts(query_params)

            # Receipt-specific routes
            if path.startswith("/api/v2/receipts/"):
                parts = path.split("/")
                if len(parts) < 5:
                    return error_response("Invalid receipt path", 400)

                receipt_id = parts[4]

                # Export endpoint
                if len(parts) > 5 and parts[5] == "export":
                    return await self._export_receipt(receipt_id, query_params)

                # Integrity verification
                if len(parts) > 5 and parts[5] == "verify" and method == "POST":
                    return await self._verify_integrity(receipt_id)

                # Signature verification
                if len(parts) > 5 and parts[5] == "verify-signature" and method == "POST":
                    return await self._verify_signature(receipt_id)

                # Get single receipt
                if method == "GET":
                    return await self._get_receipt(receipt_id)

            return error_response("Not found", 404)

        except Exception as e:
            logger.exception(f"Error handling receipt request: {e}")
            return error_response(f"Internal error: {str(e)}", 500)

    async def _list_receipts(self, query_params: Dict[str, str]) -> HandlerResult:
        """
        List receipts with filtering and pagination.

        Query params:
            limit: Max results (default 20, max 100)
            offset: Pagination offset
            verdict: Filter by verdict (APPROVED, REJECTED, etc.)
            risk_level: Filter by risk (LOW, MEDIUM, HIGH, CRITICAL)
            date_from: ISO date/timestamp for start
            date_to: ISO date/timestamp for end
            signed_only: Only return signed receipts (true/false)
            sort_by: Sort field (created_at, confidence, risk_score)
            order: Sort order (asc, desc)
        """
        store = self._get_store()

        # Parse pagination
        limit = min(int(query_params.get("limit", "20")), 100)
        offset = int(query_params.get("offset", "0"))

        # Parse filters
        verdict = query_params.get("verdict")
        risk_level = query_params.get("risk_level")
        signed_only = query_params.get("signed_only", "").lower() == "true"

        # Parse date range
        date_from = self._parse_timestamp(query_params.get("date_from"))
        date_to = self._parse_timestamp(query_params.get("date_to"))

        # Parse sorting
        sort_by = query_params.get("sort_by", "created_at")
        order = query_params.get("order", "desc")

        # Query store
        receipts = store.list(
            limit=limit,
            offset=offset,
            verdict=verdict,
            risk_level=risk_level,
            date_from=date_from,
            date_to=date_to,
            signed_only=signed_only,
            sort_by=sort_by,
            order=order,
        )

        total = store.count(
            verdict=verdict,
            risk_level=risk_level,
            date_from=date_from,
            date_to=date_to,
            signed_only=signed_only,
        )

        return json_response(
            {
                "receipts": [r.to_dict() for r in receipts],
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": total,
                    "has_more": offset + len(receipts) < total,
                },
                "filters": {
                    "verdict": verdict,
                    "risk_level": risk_level,
                    "date_from": date_from,
                    "date_to": date_to,
                    "signed_only": signed_only,
                },
            }
        )

    async def _get_receipt(self, receipt_id: str) -> HandlerResult:
        """Get a specific receipt by ID."""
        store = self._get_store()
        receipt = store.get(receipt_id)

        if not receipt:
            # Try by gauntlet_id
            receipt = store.get_by_gauntlet(receipt_id)

        if not receipt:
            return error_response("Receipt not found", 404)

        return json_response(receipt.to_full_dict())

    async def _export_receipt(self, receipt_id: str, query_params: Dict[str, str]) -> HandlerResult:
        """
        Export receipt in specified format.

        Query params:
            format: Export format (json, html, md, pdf, sarif, csv)
            signed: Include signature if available (true/false)
        """
        store = self._get_store()
        receipt = store.get(receipt_id)

        if not receipt:
            return error_response("Receipt not found", 404)

        export_format = query_params.get("format", "json").lower()
        _include_signature = query_params.get("signed", "true").lower() == "true"  # noqa: F841 - Future: signed exports

        try:
            from aragora.export.decision_receipt import DecisionReceipt

            # Reconstruct DecisionReceipt from stored data
            decision_receipt = DecisionReceipt.from_dict(receipt.data)

            if export_format == "json":
                content = decision_receipt.to_json(indent=2)
                body = content.encode("utf-8") if isinstance(content, str) else content
                return HandlerResult(
                    status_code=200,
                    content_type="application/json",
                    body=body,
                )

            elif export_format == "html":
                content = decision_receipt.to_html()
                body = content.encode("utf-8") if isinstance(content, str) else content
                return HandlerResult(
                    status_code=200,
                    content_type="text/html",
                    body=body,
                )

            elif export_format == "md" or export_format == "markdown":
                content = decision_receipt.to_markdown()
                body = content.encode("utf-8") if isinstance(content, str) else content
                return HandlerResult(
                    status_code=200,
                    content_type="text/markdown",
                    body=body,
                )

            elif export_format == "pdf":
                try:
                    pdf_bytes = decision_receipt.to_pdf()
                    return HandlerResult(
                        status_code=200,
                        content_type="application/pdf",
                        body=pdf_bytes,
                        headers={
                            "Content-Disposition": f"attachment; filename=receipt-{receipt_id}.pdf",
                        },
                    )
                except ImportError:
                    return error_response("PDF export requires weasyprint package", 501)

            elif export_format == "sarif":
                from aragora.gauntlet.api.export import export_receipt, ReceiptExportFormat

                sarif_content = export_receipt(decision_receipt, ReceiptExportFormat.SARIF)  # type: ignore[arg-type]
                body = (
                    sarif_content.encode("utf-8")
                    if isinstance(sarif_content, str)
                    else sarif_content
                )
                return HandlerResult(
                    status_code=200,
                    content_type="application/json",
                    body=body,
                )

            elif export_format == "csv":
                content = decision_receipt.to_csv()
                body = content.encode("utf-8") if isinstance(content, str) else content
                return HandlerResult(
                    status_code=200,
                    content_type="text/csv",
                    body=body,
                    headers={
                        "Content-Disposition": f"attachment; filename=receipt-{receipt_id}.csv",
                    },
                )

            else:
                return error_response(
                    f"Unsupported format: {export_format}. "
                    "Supported: json, html, md, pdf, sarif, csv",
                    400,
                )

        except Exception as e:
            logger.exception(f"Export failed: {e}")
            return error_response(f"Export failed: {str(e)}", 500)

    async def _verify_integrity(self, receipt_id: str) -> HandlerResult:
        """Verify receipt integrity checksum."""
        store = self._get_store()
        result = store.verify_integrity(receipt_id)

        if "error" in result and result.get("integrity_valid") is False:
            if "not found" in result.get("error", "").lower():
                return error_response("Receipt not found", 404)

        return json_response(result)

    async def _verify_signature(self, receipt_id: str) -> HandlerResult:
        """Verify receipt cryptographic signature."""
        store = self._get_store()
        result = store.verify_signature(receipt_id)

        if result.error and "not found" in result.error.lower():
            return error_response("Receipt not found", 404)

        return json_response(result.to_dict())

    async def _verify_batch(self, body: Dict[str, Any]) -> HandlerResult:
        """
        Batch verify multiple receipt signatures.

        Body:
            receipt_ids: List of receipt IDs to verify
        """
        receipt_ids = body.get("receipt_ids", [])

        if not receipt_ids:
            return error_response("receipt_ids required", 400)

        if len(receipt_ids) > 100:
            return error_response("Maximum 100 receipts per batch", 400)

        store = self._get_store()
        results, summary = store.verify_batch(receipt_ids)

        return json_response(
            {
                "results": [r.to_dict() for r in results],
                "summary": summary,
            }
        )

    async def _get_stats(self) -> HandlerResult:
        """Get receipt statistics."""
        store = self._get_store()
        stats = store.get_stats()

        return json_response(
            {
                "stats": stats,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _parse_timestamp(self, value: Optional[str]) -> Optional[float]:
        """Parse timestamp from string (ISO date or unix timestamp)."""
        if not value:
            return None

        try:
            # Try as unix timestamp
            return float(value)
        except ValueError:
            pass

        try:
            # Try as ISO date
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.timestamp()
        except (ValueError, AttributeError):
            pass

        return None


# Handler factory function for registration
def create_receipts_handler(server_context: ServerContext) -> ReceiptsHandler:
    """Factory function for handler registration."""
    return ReceiptsHandler(server_context)
