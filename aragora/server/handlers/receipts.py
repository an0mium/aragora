"""
Decision Receipt HTTP Handlers for Aragora.

Provides REST API endpoints for decision receipt management:
- List and retrieve receipts with filtering
- Verify receipt integrity and signatures
- Export receipts in multiple formats
- Batch verification and signing operations
- Shareable links for receipts

Endpoints:
    GET  /api/v2/receipts                              - List receipts with filters
    GET  /api/v2/receipts/search                       - Full-text search receipts
    GET  /api/v2/receipts/:receipt_id                  - Get specific receipt
    GET  /api/v2/receipts/:receipt_id/export           - Export (format=json|html|md|pdf)
    POST /api/v2/receipts/:receipt_id/verify           - Verify integrity checksum
    POST /api/v2/receipts/:receipt_id/verify-signature - Verify cryptographic signature
    POST /api/v2/receipts/verify-batch                 - Batch signature verification
    POST /api/v2/receipts/sign-batch                   - Batch signing
    POST /api/v2/receipts/batch-export                 - Batch export to ZIP
    GET  /api/v2/receipts/stats                        - Receipt statistics
    POST /api/v2/receipts/:receipt_id/share            - Create shareable link
    GET  /api/v2/receipts/share/:token                 - Access receipt via share token

These endpoints support the "defensible decisions" pillar with:
- Cryptographic signature verification
- 7-year retention for compliance
- Full audit trail integration
- Time-limited shareable links
"""

from __future__ import annotations

import io
import logging
import secrets
import zipfile
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
        self._share_store = None  # Lazy initialization for share tokens

    def _get_store(self):
        """Get or create receipt store (lazy initialization)."""
        if self._store is None:
            from aragora.storage.receipt_store import get_receipt_store

            self._store = get_receipt_store()
        return self._store

    def _get_share_store(self):
        """Get or create receipt share store (lazy initialization)."""
        if self._share_store is None:
            from aragora.storage.receipt_share_store import get_receipt_share_store

            self._share_store = get_receipt_share_store()
        return self._share_store

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

            # Batch signing
            if path == "/api/v2/receipts/sign-batch" and method == "POST":
                return await self._sign_batch(body)

            # Batch export
            if path == "/api/v2/receipts/batch-export" and method == "POST":
                return await self._batch_export(body)

            # Access shared receipt (public endpoint)
            if path.startswith("/api/v2/receipts/share/") and method == "GET":
                token = path.split("/api/v2/receipts/share/")[1].rstrip("/")
                return await self._get_shared_receipt(token)

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

                # Share receipt
                if len(parts) > 5 and parts[5] == "share" and method == "POST":
                    return await self._share_receipt(receipt_id, body)

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

    @require_permission("receipts:share")
    async def _share_receipt(self, receipt_id: str, body: Dict[str, Any]) -> HandlerResult:
        """
        Create a shareable link for a receipt.

        Body:
            expires_in_hours: Hours until link expires (default 24, max 720 = 30 days)
            max_accesses: Maximum number of accesses (optional, None = unlimited)

        Returns:
            Share URL and token details
        """
        store = self._get_store()
        receipt = store.get(receipt_id)

        if not receipt:
            return error_response("Receipt not found", 404)

        # Parse options
        expires_in_hours = min(int(body.get("expires_in_hours", 24)), 720)
        max_accesses = body.get("max_accesses")

        # Generate share token
        token = secrets.token_urlsafe(24)
        expires_at = datetime.now(timezone.utc).timestamp() + (expires_in_hours * 3600)

        # Store share link
        share_store = self._get_share_store()
        share_store.save(
            token=token,
            receipt_id=receipt_id,
            expires_at=expires_at,
            max_accesses=max_accesses,
        )

        # Emit webhook notification
        share_url = f"/api/v2/receipts/share/{token}"
        try:
            from aragora.integrations.receipt_webhooks import ReceiptWebhookNotifier

            notifier = ReceiptWebhookNotifier()
            debate_id = getattr(receipt, "debate_id", "") or ""
            notifier.notify_receipt_shared(
                receipt_id=receipt_id,
                debate_id=debate_id,
                share_url=share_url,
                expires_at=datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat(),
            )
        except ImportError:
            logger.debug("Receipt webhooks not available")

        return json_response(
            {
                "success": True,
                "receipt_id": receipt_id,
                "share_url": share_url,
                "token": token,
                "expires_at": datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat(),
                "max_accesses": max_accesses,
            }
        )

    async def _get_shared_receipt(self, token: str) -> HandlerResult:
        """
        Access a receipt via share token.

        This is a public endpoint - no authentication required.
        """
        share_store = self._get_share_store()
        share_info = share_store.get_by_token(token)

        if not share_info:
            return error_response("Share link not found", 404)

        # Check expiration
        if (
            share_info.get("expires_at")
            and share_info["expires_at"] < datetime.now(timezone.utc).timestamp()
        ):
            return error_response("Share link has expired", 410)

        # Check access limit
        if share_info.get("max_accesses"):
            if share_info.get("access_count", 0) >= share_info["max_accesses"]:
                return error_response("Share link access limit reached", 410)

        # Increment access count
        share_store.increment_access(token)

        # Get receipt
        store = self._get_store()
        receipt = store.get(share_info["receipt_id"])

        if not receipt:
            return error_response("Receipt not found", 404)

        return json_response(
            {
                "receipt": receipt.to_full_dict(),
                "shared": True,
                "access_count": share_info.get("access_count", 0) + 1,
            }
        )

    @require_permission("receipts:sign")
    async def _sign_batch(self, body: Dict[str, Any]) -> HandlerResult:
        """
        Batch sign multiple receipts.

        Body:
            receipt_ids: List of receipt IDs to sign (max 100)
            algorithm: Signing algorithm (hmac-sha256, rsa-sha256, ed25519)
        """
        receipt_ids = body.get("receipt_ids", [])
        algorithm = body.get("algorithm", "hmac-sha256")

        if not receipt_ids:
            return error_response("receipt_ids required", 400)

        if len(receipt_ids) > 100:
            return error_response("Maximum 100 receipts per batch", 400)

        store = self._get_store()
        results = []
        signed_count = 0
        failed_count = 0
        skipped_count = 0

        try:
            from aragora.gauntlet.signing import (
                Ed25519Signer,
                HMACSigner,
                ReceiptSigner,
                RSASigner,
                SigningBackend,
            )

            # Create backend based on algorithm
            backend: SigningBackend
            if algorithm == "rsa-sha256":
                backend = RSASigner.generate_keypair()
            elif algorithm == "ed25519":
                backend = Ed25519Signer.generate_keypair()
            else:
                # Default to HMAC-SHA256
                backend = HMACSigner.from_env()

            signer = ReceiptSigner(backend=backend)

            for receipt_id in receipt_ids:
                receipt = store.get(receipt_id)

                if not receipt:
                    results.append({"receipt_id": receipt_id, "status": "not_found"})
                    failed_count += 1
                    continue

                # Check if already signed
                if store.get_signature(receipt_id):
                    results.append({"receipt_id": receipt_id, "status": "already_signed"})
                    skipped_count += 1
                    continue

                try:
                    # Sign the receipt
                    signature = signer.sign(receipt.data)
                    store.store_signature(receipt_id, signature, algorithm)
                    results.append({"receipt_id": receipt_id, "status": "signed"})
                    signed_count += 1
                except Exception as e:
                    results.append({"receipt_id": receipt_id, "status": "error", "error": str(e)})
                    failed_count += 1

        except ImportError:
            return error_response("Signing module not available", 501)

        return json_response(
            {
                "results": results,
                "summary": {
                    "total": len(receipt_ids),
                    "signed": signed_count,
                    "skipped": skipped_count,
                    "failed": failed_count,
                },
            }
        )

    @require_permission("receipts:export")
    async def _batch_export(self, body: Dict[str, Any]) -> HandlerResult:
        """
        Batch export multiple receipts to a ZIP file.

        Body:
            receipt_ids: List of receipt IDs to export (max 100)
            format: Export format (json, html, markdown, csv)
        """
        receipt_ids = body.get("receipt_ids", [])
        export_format = body.get("format", "json").lower()

        if not receipt_ids:
            return error_response("receipt_ids required", 400)

        if len(receipt_ids) > 100:
            return error_response("Maximum 100 receipts per batch", 400)

        if export_format not in ("json", "html", "markdown", "md", "csv"):
            return error_response(
                f"Unsupported format: {export_format}. Supported: json, html, markdown, csv",
                400,
            )

        store = self._get_store()

        # Create ZIP in memory
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            exported_count = 0
            failed_ids = []

            for receipt_id in receipt_ids:
                receipt = store.get(receipt_id)

                if not receipt:
                    failed_ids.append(receipt_id)
                    continue

                try:
                    from aragora.export.decision_receipt import DecisionReceipt

                    decision_receipt = DecisionReceipt.from_dict(receipt.data)

                    # Determine file extension
                    if export_format == "json":
                        content = decision_receipt.to_json(indent=2)
                        ext = "json"
                    elif export_format in ("html",):
                        content = decision_receipt.to_html()
                        ext = "html"
                    elif export_format in ("markdown", "md"):
                        content = decision_receipt.to_markdown()
                        ext = "md"
                    elif export_format == "csv":
                        content = decision_receipt.to_csv()
                        ext = "csv"
                    else:
                        content = decision_receipt.to_json(indent=2)
                        ext = "json"

                    filename = f"receipt-{receipt_id}.{ext}"
                    zip_file.writestr(filename, content)
                    exported_count += 1

                except Exception as e:
                    logger.warning(f"Failed to export receipt {receipt_id}: {e}")
                    failed_ids.append(receipt_id)

            # Add manifest
            manifest = {
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "format": export_format,
                "total_requested": len(receipt_ids),
                "exported": exported_count,
                "failed": failed_ids,
            }
            import json

            zip_file.writestr("manifest.json", json.dumps(manifest, indent=2))

        zip_buffer.seek(0)
        zip_bytes = zip_buffer.read()

        return HandlerResult(
            status_code=200,
            content_type="application/zip",
            body=zip_bytes,
            headers={
                "Content-Disposition": "attachment; filename=receipts-export.zip",
            },
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
