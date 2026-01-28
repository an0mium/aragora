"""
Decision Receipt HTTP Handlers for Aragora.

Provides REST API endpoints for decision receipt management:
- List and retrieve receipts with filtering
- Verify receipt integrity and signatures
- Export receipts in multiple formats
- Batch verification operations

Endpoints:
    GET  /api/v2/receipts                              - List receipts with filters
    GET  /api/v2/receipts/search                       - Full-text search receipts
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
from aragora.rbac.decorators import require_permission

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

            # Retention status endpoint (GDPR compliance)
            if path == "/api/v2/receipts/retention-status" and method == "GET":
                return await self._get_retention_status()

            # DSAR endpoint (GDPR Data Subject Access Request)
            if path.startswith("/api/v2/receipts/dsar/") and method == "GET":
                parts = path.split("/")
                if len(parts) >= 6:
                    user_id = parts[5]
                    return await self._get_dsar(user_id, query_params)
                return error_response("User ID required for DSAR request", 400)

            # Search endpoint
            if path == "/api/v2/receipts/search" and method == "GET":
                return await self._search_receipts(query_params)

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

                # Send to channel
                if len(parts) > 5 and parts[5] == "send-to-channel" and method == "POST":
                    return await self._send_to_channel(receipt_id, body)

                # Get formatted for channel
                if len(parts) > 5 and parts[5] == "formatted" and method == "GET":
                    channel_type = parts[6] if len(parts) > 6 else "slack"
                    return await self._get_formatted(receipt_id, channel_type, query_params)

                # Get single receipt
                if method == "GET":
                    return await self._get_receipt(receipt_id)

            return error_response("Not found", 404)

        except Exception as e:
            logger.exception(f"Error handling receipt request: {e}")
            return error_response(f"Internal error: {str(e)}", 500)

    @require_permission("receipts:read")
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

    @require_permission("receipts:read")
    async def _search_receipts(self, query_params: Dict[str, str]) -> HandlerResult:
        """
        Full-text search across receipt content.

        Query params:
            q: Search query (required, minimum 3 characters)
            limit: Max results (default 50, max 100)
            offset: Pagination offset
            verdict: Optional filter by verdict (APPROVED, REJECTED, etc.)
            risk_level: Optional filter by risk (LOW, MEDIUM, HIGH, CRITICAL)
        """
        query = query_params.get("q", "").strip()

        if not query:
            return error_response("Query parameter 'q' is required", 400)

        if len(query) < 3:
            return error_response("Search query must be at least 3 characters", 400)

        store = self._get_store()

        # Parse pagination
        limit = min(int(query_params.get("limit", "50")), 100)
        offset = int(query_params.get("offset", "0"))

        # Optional filters
        verdict = query_params.get("verdict")
        risk_level = query_params.get("risk_level")

        # Perform search
        receipts = store.search(
            query=query,
            limit=limit,
            offset=offset,
            verdict=verdict,
            risk_level=risk_level,
        )

        total = store.search_count(
            query=query,
            verdict=verdict,
            risk_level=risk_level,
        )

        return json_response(
            {
                "receipts": [r.to_dict() for r in receipts],
                "query": query,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": total,
                    "has_more": offset + len(receipts) < total,
                },
                "filters": {
                    "verdict": verdict,
                    "risk_level": risk_level,
                },
            }
        )

    @require_permission("receipts:read")
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

    @require_permission("receipts:read")
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

    @require_permission("receipts:verify")
    async def _verify_integrity(self, receipt_id: str) -> HandlerResult:
        """Verify receipt integrity checksum."""
        store = self._get_store()
        result = store.verify_integrity(receipt_id)

        if "error" in result and result.get("integrity_valid") is False:
            if "not found" in result.get("error", "").lower():
                return error_response("Receipt not found", 404)

        return json_response(result)

    @require_permission("receipts:verify")
    async def _verify_signature(self, receipt_id: str) -> HandlerResult:
        """Verify receipt cryptographic signature."""
        store = self._get_store()
        result = store.verify_signature(receipt_id)

        if result.error and "not found" in result.error.lower():
            return error_response("Receipt not found", 404)

        return json_response(result.to_dict())

    @require_permission("receipts:verify")
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

    @require_permission("receipts:read")
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

    @require_permission("receipts:send")
    async def _send_to_channel(self, receipt_id: str, body: Dict[str, Any]) -> HandlerResult:
        """
        Send a decision receipt to a specified channel.

        Body:
            channel_type: Channel type (slack, teams, email, discord)
            channel_id: Target channel/conversation ID
            workspace_id: Workspace/tenant ID (for Slack/Teams)
            options: Optional formatting options (compact, etc.)
        """
        channel_type = body.get("channel_type")
        channel_id = body.get("channel_id")
        workspace_id = body.get("workspace_id")
        options = body.get("options", {})

        if not channel_type:
            return error_response("channel_type is required", 400)
        if not channel_id:
            return error_response("channel_id is required", 400)

        # Get the receipt
        store = self._get_store()
        receipt = store.get(receipt_id)
        if not receipt:
            return error_response("Receipt not found", 404)

        try:
            from aragora.channels.formatter import format_receipt_for_channel
            from aragora.export.decision_receipt import DecisionReceipt

            # Reconstruct DecisionReceipt from stored data
            decision_receipt = DecisionReceipt.from_dict(receipt.data)

            # Format the receipt for the channel
            formatted = format_receipt_for_channel(decision_receipt, channel_type, options)

            # Send to the channel based on type
            if channel_type == "slack":
                result = await self._send_to_slack(formatted, channel_id, workspace_id)
            elif channel_type == "teams":
                result = await self._send_to_teams(formatted, channel_id, workspace_id)
            elif channel_type == "email":
                result = await self._send_to_email(formatted, channel_id, options)
            elif channel_type == "discord":
                result = await self._send_to_discord(formatted, channel_id, options)
            else:
                return error_response(
                    f"Unsupported channel type: {channel_type}. "
                    "Supported: slack, teams, email, discord",
                    400,
                )

            return json_response(
                {
                    "sent": True,
                    "receipt_id": receipt_id,
                    "channel_type": channel_type,
                    "channel_id": channel_id,
                    **result,
                }
            )

        except ImportError as e:
            logger.exception(f"Missing dependency for channel {channel_type}: {e}")
            return error_response(f"Channel {channel_type} not available: {str(e)}", 501)
        except Exception as e:
            logger.exception(f"Failed to send receipt to channel: {e}")
            return error_response(f"Failed to send: {str(e)}", 500)

    async def _send_to_slack(
        self,
        formatted: Dict[str, Any],
        channel_id: str,
        workspace_id: Optional[str],
    ) -> Dict[str, Any]:
        """Send formatted receipt to Slack channel."""
        from aragora.storage.slack_workspace_store import get_slack_workspace_store

        if not workspace_id:
            raise ValueError("workspace_id is required for Slack")

        store = get_slack_workspace_store()
        workspace = store.get(workspace_id)
        if not workspace:
            raise ValueError(f"Slack workspace not found: {workspace_id}")

        # Use Slack connector to send
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(
            token=workspace.access_token,
            signing_secret=workspace.signing_secret,
        )

        blocks = formatted.get("blocks", [])
        result = await connector.send_message(
            channel_id=channel_id,
            text="Decision Receipt",
            blocks=blocks,
        )

        return {"message_ts": result.timestamp, "channel": result.channel_id}

    async def _send_to_teams(
        self,
        formatted: Dict[str, Any],
        channel_id: str,
        workspace_id: Optional[str],
    ) -> Dict[str, Any]:
        """Send formatted receipt to Teams channel."""
        from aragora.storage.teams_workspace_store import get_teams_workspace_store

        if not workspace_id:
            raise ValueError("workspace_id (tenant_id) is required for Teams")

        store = get_teams_workspace_store()
        workspace = store.get(workspace_id)
        if not workspace:
            raise ValueError(f"Teams workspace not found: {workspace_id}")

        # Use Teams connector to send
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(
            app_id=workspace.bot_id,
            app_password="",  # Bot Framework uses different auth flow
            service_url=workspace.service_url or "https://smba.trafficmanager.net/amer/",
        )

        # Send Adaptive Card via send_message with blocks
        card_body = formatted.get("body", [])
        result = await connector.send_message(
            channel_id=channel_id,
            text="Decision Receipt",
            blocks=card_body,
            conversation_id=channel_id,
        )

        return {"message_id": result.message_id}

    async def _send_to_email(
        self,
        formatted: Dict[str, Any],
        email_address: str,
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Send formatted receipt via email."""
        import os
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        smtp_host = os.environ.get("SMTP_HOST", "localhost")
        smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        smtp_user = os.environ.get("SMTP_USER", "")
        smtp_password = os.environ.get("SMTP_PASSWORD", "")
        from_email = os.environ.get("SMTP_FROM", "aragora@localhost")

        msg = MIMEMultipart("alternative")
        msg["Subject"] = formatted.get("subject", "Decision Receipt")
        msg["From"] = from_email
        msg["To"] = email_address

        # Add plain text and HTML parts
        if "plain_text" in formatted:
            msg.attach(MIMEText(formatted["plain_text"], "plain"))
        if "html" in formatted:
            msg.attach(MIMEText(formatted["html"], "html"))

        # Send email
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            if smtp_user and smtp_password:
                server.starttls()
                server.login(smtp_user, smtp_password)
            server.send_message(msg)

        return {"email_sent_to": email_address}

    async def _send_to_discord(
        self,
        formatted: Dict[str, Any],
        channel_id: str,
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Send formatted receipt to Discord channel."""
        import os
        import urllib.request
        import json

        bot_token = os.environ.get("DISCORD_BOT_TOKEN")
        if not bot_token:
            raise ValueError("DISCORD_BOT_TOKEN environment variable required")

        url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
        headers = {
            "Authorization": f"Bot {bot_token}",
            "Content-Type": "application/json",
        }

        data = json.dumps(formatted).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers)

        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode())

        return {"message_id": result.get("id")}

    async def _get_formatted(
        self,
        receipt_id: str,
        channel_type: str,
        query_params: Dict[str, str],
    ) -> HandlerResult:
        """
        Get receipt formatted for a specific channel type.

        Returns the formatted payload without sending it.
        """
        store = self._get_store()
        receipt = store.get(receipt_id)

        if not receipt:
            return error_response("Receipt not found", 404)

        options = {
            "compact": query_params.get("compact", "").lower() == "true",
        }

        try:
            from aragora.channels.formatter import format_receipt_for_channel
            from aragora.export.decision_receipt import DecisionReceipt

            decision_receipt = DecisionReceipt.from_dict(receipt.data)
            formatted = format_receipt_for_channel(decision_receipt, channel_type, options)

            return json_response(
                {
                    "receipt_id": receipt_id,
                    "channel_type": channel_type,
                    "formatted": formatted,
                }
            )

        except ValueError as e:
            return error_response(str(e), 400)
        except Exception as e:
            logger.exception(f"Failed to format receipt: {e}")
            return error_response(f"Formatting failed: {str(e)}", 500)

    @require_permission("receipts:read")
    async def _get_retention_status(self) -> HandlerResult:
        """Get retention status for GDPR compliance. Endpoint: GET /api/v2/receipts/retention-status"""
        store = self._get_store()
        status = store.get_retention_status()
        return json_response(status)

    @require_permission("receipts:read")
    async def _get_dsar(self, user_id: str, query_params: Dict[str, str]) -> HandlerResult:
        """Handle GDPR DSAR. Endpoint: GET /api/v2/receipts/dsar/{user_id}"""
        if not user_id or len(user_id) < 3:
            return error_response("Valid user_id required (minimum 3 characters)", 400)

        store = self._get_store()
        limit = min(int(query_params.get("limit", "100")), 1000)
        offset = int(query_params.get("offset", "0"))

        receipts, total = store.get_by_user(user_id=user_id, limit=limit, offset=offset)
        receipt_data = [r.to_full_dict() for r in receipts]

        return json_response(
            {
                "dsar_request": {
                    "user_id": user_id,
                    "request_type": "data_subject_access_request",
                    "gdpr_article": "Article 15 - Right of access",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
                "receipts": receipt_data,
                "pagination": {"limit": limit, "offset": offset, "total": total},
                "summary": {"total_receipts": total, "returned_receipts": len(receipts)},
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
