"""
Receipt Delivery API Handlers.

Provides management APIs for receipt delivery configuration:
- GET /api/v1/sme/receipts/delivery/config - Get delivery config
- POST /api/v1/sme/receipts/delivery/config - Update config
- GET /api/v1/sme/receipts/delivery/history - Delivery history
- POST /api/v1/sme/receipts/delivery/test - Test delivery
- GET /api/v1/sme/receipts/delivery/stats - Delivery stats
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from ..base import (
    error_response,
    get_string_param,
    handle_errors,
    json_response,
)
from ..utils.responses import HandlerResult
from ..utils.url_security import validate_webhook_url
from ..secure import SecureHandler
from aragora.rbac.decorators import require_permission
from ..utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiter for receipt delivery APIs (60 requests per minute)
_delivery_limiter = RateLimiter(requests_per_minute=60)


class ReceiptDeliveryHandler(SecureHandler):
    """Handler for receipt delivery configuration endpoints.

    Provides APIs for managing automatic receipt delivery to
    channels (Slack, Teams, email, webhooks).
    """

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    RESOURCE_TYPE = "receipt_delivery"

    ROUTES = [
        "/api/v1/sme/receipts/delivery/config",
        "/api/v1/sme/receipts/delivery/history",
        "/api/v1/sme/receipts/delivery/test",
        "/api/v1/sme/receipts/delivery/stats",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(
        self,
        path: str,
        query_params: dict,
        handler,
        method: str = "GET",
    ) -> HandlerResult | None:
        """Route receipt delivery requests to appropriate methods."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _delivery_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for receipt delivery: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # Determine HTTP method from handler if not provided
        if hasattr(handler, "command"):
            method = handler.command

        # Route to appropriate handler
        if path == "/api/v1/sme/receipts/delivery/config":
            if method == "GET":
                return self._get_config(handler, query_params)
            elif method == "POST":
                return self._update_config(handler, query_params)
            return error_response("Method not allowed", 405)

        if path == "/api/v1/sme/receipts/delivery/history":
            if method == "GET":
                return self._get_history(handler, query_params)
            return error_response("Method not allowed", 405)

        if path == "/api/v1/sme/receipts/delivery/test":
            if method == "POST":
                return self._test_delivery(handler, query_params)
            return error_response("Method not allowed", 405)

        if path == "/api/v1/sme/receipts/delivery/stats":
            if method == "GET":
                return self._get_stats(handler, query_params)
            return error_response("Method not allowed", 405)

        return error_response("Not found", 404)

    def _get_subscription_store(self):
        """Get channel subscription store instance."""
        from aragora.storage.channel_subscription_store import get_channel_subscription_store

        return get_channel_subscription_store()

    def _get_delivery_history_store(self):
        """Get delivery history store instance."""
        # Use a simple in-memory store for delivery history
        # In production, this would be backed by a database
        if not hasattr(self, "_delivery_history"):
            self._delivery_history: list[dict[str, Any]] = []
        return self._delivery_history

    def _get_user_and_org(self, handler, user):
        """Get user and organization from context."""
        user_store = self.ctx.get("user_store")
        if not user_store:
            return None, None, error_response("Service unavailable", 503)

        db_user = user_store.get_user_by_id(user.user_id)
        if not db_user:
            return None, None, error_response("User not found", 404)

        org = None
        if db_user.org_id:
            org = user_store.get_organization_by_id(db_user.org_id)

        if not org:
            return None, None, error_response("No organization found", 404)

        return db_user, org, None

    @handle_errors("get delivery config")
    @require_permission("sme:receipts:deliver")
    def _get_config(
        self,
        handler,
        query_params: dict,
        user=None,
    ) -> HandlerResult:
        """
        Get receipt delivery configuration for the organization.

        Returns:
            JSON response with delivery configuration:
            {
                "auto_delivery_enabled": true,
                "subscriptions": [...],
                "default_format": "compact",
                "include_full_receipt": false
            }
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        store = self._get_subscription_store()

        # Get subscriptions for 'receipt' events
        from aragora.storage.channel_subscription_store import EventType

        all_subscriptions = store.get_by_org(org.id)
        receipt_subscriptions = [
            sub
            for sub in all_subscriptions
            if EventType.RECEIPT in sub.event_types or EventType.RECEIPT.value in sub.event_types
        ]

        # Get org-level config (would be stored in org settings in production)
        config = {
            "auto_delivery_enabled": len(receipt_subscriptions) > 0,
            "subscriptions": [sub.to_dict() for sub in receipt_subscriptions],
            "default_format": "compact",  # compact, full, summary
            "include_full_receipt": False,
            "include_signature": True,
            "notification_delay_seconds": 0,  # Delay before sending
        }

        return json_response(
            {
                "config": config,
                "subscription_count": len(receipt_subscriptions),
            }
        )

    @handle_errors("update delivery config")
    @require_permission("sme:receipts:deliver")
    def _update_config(
        self,
        handler,
        query_params: dict,
        user=None,
    ) -> HandlerResult:
        """
        Update receipt delivery configuration.

        Request Body:
            {
                "subscriptions": [
                    {
                        "channel_type": "slack",
                        "channel_id": "C123456",
                        "workspace_id": "T123456",
                        "channel_name": "#decisions"
                    }
                ],
                "default_format": "compact",
                "include_full_receipt": false
            }

        Returns:
            JSON response with updated configuration
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        # Parse request body
        import json as json_lib

        try:
            body = handler.rfile.read(int(handler.headers.get("Content-Length", 0)))
            data = json_lib.loads(body.decode("utf-8")) if body else {}
        except (json_lib.JSONDecodeError, ValueError):
            return error_response("Invalid JSON body", 400)

        store = self._get_subscription_store()
        from aragora.storage.channel_subscription_store import ChannelType, EventType

        # Handle subscriptions update
        new_subscriptions = data.get("subscriptions", [])
        created_subscriptions = []
        errors = []

        for sub_data in new_subscriptions:
            channel_type = sub_data.get("channel_type")
            channel_id = sub_data.get("channel_id")

            if not channel_type or not channel_id:
                errors.append({"error": "channel_type and channel_id are required"})
                continue

            try:
                channel_type_enum = ChannelType(channel_type)
            except ValueError:
                errors.append({"error": f"Invalid channel_type: {channel_type}"})
                continue

            # Check if subscription already exists
            existing = store.get_by_org_and_channel(org.id, channel_type, channel_id)
            if existing:
                # Update existing subscription to include receipt events
                if EventType.RECEIPT not in existing.event_types:
                    existing.event_types.append(EventType.RECEIPT)
                    store.update(existing.id, event_types=existing.event_types)
                created_subscriptions.append(existing.to_dict())
            else:
                # Create new subscription
                try:
                    subscription = store.create(
                        org_id=org.id,
                        channel_type=channel_type_enum,
                        channel_id=channel_id,
                        event_types=[EventType.RECEIPT],
                        workspace_id=sub_data.get("workspace_id"),
                        channel_name=sub_data.get("channel_name"),
                        created_by=db_user.id,
                    )
                    created_subscriptions.append(subscription.to_dict())
                except (KeyError, ValueError, OSError) as e:
                    logger.warning("Subscription creation failed for %s/%s: %s", channel_type, channel_id, e)
                    errors.append(
                        {
                            "channel_type": channel_type,
                            "channel_id": channel_id,
                            "error": "Internal server error",
                        }
                    )

        logger.info(
            f"Updated delivery config for org {org.id}: "
            f"{len(created_subscriptions)} subscriptions, {len(errors)} errors"
        )

        return json_response(
            {
                "updated": True,
                "subscriptions": created_subscriptions,
                "errors": errors if errors else None,
            }
        )

    @handle_errors("get delivery history")
    @require_permission("sme:receipts:deliver")
    def _get_history(
        self,
        handler,
        query_params: dict,
        user=None,
    ) -> HandlerResult:
        """
        Get receipt delivery history for the organization.

        Query Parameters:
            limit: Maximum results (default: 50)
            offset: Pagination offset (default: 0)
            channel_type: Filter by channel type
            status: Filter by status (success, failed)

        Returns:
            JSON response with delivery history
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        limit = int(get_string_param(handler, "limit", "50"))
        offset = int(get_string_param(handler, "offset", "0"))
        channel_type_filter = get_string_param(handler, "channel_type", "")
        status_filter = get_string_param(handler, "status", "")

        # In production, this would query a delivery history table
        # For now, use in-memory history for demonstration
        history = self._get_delivery_history_store()

        # Filter history
        filtered = [
            h
            for h in history
            if h.get("org_id") == org.id
            and (not channel_type_filter or h.get("channel_type") == channel_type_filter)
            and (not status_filter or h.get("status") == status_filter)
        ]

        # Sort by timestamp descending
        filtered.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

        # Apply pagination
        paginated = filtered[offset : offset + limit]

        return json_response(
            {
                "history": paginated,
                "total": len(filtered),
                "limit": limit,
                "offset": offset,
            }
        )

    @handle_errors("test delivery")
    @require_permission("sme:receipts:deliver")
    def _test_delivery(
        self,
        handler,
        query_params: dict,
        user=None,
    ) -> HandlerResult:
        """
        Test receipt delivery to a channel.

        Request Body:
            {
                "channel_type": "slack",
                "channel_id": "C123456",
                "workspace_id": "T123456",
                "test_message": "This is a test delivery"
            }

        Returns:
            JSON response with test result
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        # Parse request body
        import json as json_lib

        try:
            body = handler.rfile.read(int(handler.headers.get("Content-Length", 0)))
            data = json_lib.loads(body.decode("utf-8")) if body else {}
        except (json_lib.JSONDecodeError, ValueError):
            return error_response("Invalid JSON body", 400)

        channel_type = data.get("channel_type")
        channel_id = data.get("channel_id")
        workspace_id = data.get("workspace_id")
        test_message = data.get("test_message", "Test receipt delivery from Aragora")

        if not channel_type:
            return error_response("channel_type is required", 400)
        if not channel_id:
            return error_response("channel_id is required", 400)

        # Validate channel type
        valid_types = ["slack", "teams", "email", "webhook"]
        if channel_type not in valid_types:
            return error_response(f"Invalid channel_type. Valid: {valid_types}", 400)

        # For Slack/Teams, workspace_id is required
        if channel_type in ["slack", "teams"] and not workspace_id:
            return error_response(f"workspace_id is required for {channel_type}", 400)

        # For webhook, validate the URL for SSRF protection
        if channel_type == "webhook":
            is_valid, url_error = validate_webhook_url(channel_id, allow_localhost=False)
            if not is_valid:
                return error_response(f"Invalid webhook URL: {url_error}", 400)

        # Attempt test delivery
        try:
            result = self._send_test_message(
                channel_type=channel_type,
                channel_id=channel_id,
                workspace_id=workspace_id,
                message=test_message,
                org_id=org.id,
            )

            # Record in history
            history_entry = {
                "id": f"test-{int(time.time() * 1000)}",
                "org_id": org.id,
                "receipt_id": None,
                "channel_type": channel_type,
                "channel_id": channel_id,
                "workspace_id": workspace_id,
                "status": "success" if result.get("success") else "failed",
                "timestamp": time.time(),
                "is_test": True,
                "error": result.get("error"),
                "message_id": result.get("message_id"),
            }
            self._get_delivery_history_store().append(history_entry)

            return json_response(
                {
                    "test_successful": result.get("success", False),
                    "channel_type": channel_type,
                    "channel_id": channel_id,
                    "message_id": result.get("message_id"),
                    "error": result.get("error"),
                }
            )

        except (ConnectionError, TimeoutError, OSError, ValueError) as e:
            logger.exception(f"Test delivery failed: {e}")

            # Record failure in history
            history_entry = {
                "id": f"test-{int(time.time() * 1000)}",
                "org_id": org.id,
                "receipt_id": None,
                "channel_type": channel_type,
                "channel_id": channel_id,
                "workspace_id": workspace_id,
                "status": "failed",
                "timestamp": time.time(),
                "is_test": True,
                "error": "Internal server error",
            }
            self._get_delivery_history_store().append(history_entry)

            return json_response(
                {
                    "test_successful": False,
                    "channel_type": channel_type,
                    "channel_id": channel_id,
                    "error": "Internal server error",
                }
            )

    def _send_test_message(
        self,
        channel_type: str,
        channel_id: str,
        workspace_id: str | None,
        message: str,
        org_id: str,
    ) -> dict[str, Any]:
        """Send a test message to the specified channel."""
        if channel_type == "slack":
            return self._send_test_to_slack(channel_id, workspace_id, message)
        elif channel_type == "teams":
            return self._send_test_to_teams(channel_id, workspace_id, message)
        elif channel_type == "email":
            return self._send_test_to_email(channel_id, message)
        elif channel_type == "webhook":
            return self._send_test_to_webhook(channel_id, message, org_id)
        else:
            return {"success": False, "error": f"Unsupported channel type: {channel_type}"}

    def _send_test_to_slack(
        self, channel_id: str, workspace_id: str | None, message: str
    ) -> dict[str, Any]:
        """Send test message to Slack."""
        try:
            from aragora.storage.slack_workspace_store import get_slack_workspace_store

            store = get_slack_workspace_store()
            workspace = store.get(workspace_id)
            if not workspace:
                return {"success": False, "error": f"Slack workspace not found: {workspace_id}"}

            from aragora.connectors.chat.slack import SlackConnector

            connector = SlackConnector(
                token=workspace.access_token,
                signing_secret=workspace.signing_secret or "",
            )

            # Send test message with blocks
            import asyncio

            async def send():
                return await connector.send_message(
                    channel_id=channel_id,
                    text=message,
                    blocks=[
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f":white_check_mark: *Test Receipt Delivery*\n{message}",
                            },
                        },
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"Sent from Aragora at {datetime.now(timezone.utc).isoformat()}",
                                }
                            ],
                        },
                    ],
                )

            result = asyncio.run(send())
            return {
                "success": True,
                "message_id": result.timestamp,
                "channel": result.channel_id,
            }
        except (ImportError, ConnectionError, TimeoutError, OSError, ValueError) as e:
            logger.warning("%s receipt delivery failed: %s", "Slack", e)
            return {"success": False, "error": "Internal server error"}

    def _send_test_to_teams(
        self, channel_id: str, workspace_id: str | None, message: str
    ) -> dict[str, Any]:
        """Send test message to Teams."""
        try:
            from aragora.storage.teams_workspace_store import get_teams_workspace_store

            store = get_teams_workspace_store()
            workspace = store.get(workspace_id)
            if not workspace:
                return {"success": False, "error": f"Teams workspace not found: {workspace_id}"}

            from aragora.connectors.chat.teams import TeamsConnector

            connector = TeamsConnector(
                app_id=workspace.bot_id,
                app_password="",
                service_url=workspace.service_url or "https://smba.trafficmanager.net/amer/",
            )

            import asyncio

            async def send():
                return await connector.send_message(
                    channel_id=channel_id,
                    conversation_id=channel_id,
                    text=message,
                    blocks=[
                        {
                            "type": "TextBlock",
                            "text": "Test Receipt Delivery",
                            "weight": "bolder",
                            "size": "medium",
                        },
                        {"type": "TextBlock", "text": message, "wrap": True},
                        {
                            "type": "TextBlock",
                            "text": f"Sent from Aragora at {datetime.now(timezone.utc).isoformat()}",
                            "isSubtle": True,
                            "size": "small",
                        },
                    ],
                )

            result = asyncio.run(send())
            return {"success": True, "message_id": result.message_id}
        except (ImportError, ConnectionError, TimeoutError, OSError, ValueError) as e:
            logger.warning("%s receipt delivery failed: %s", "Teams", e)
            return {"success": False, "error": "Internal server error"}

    def _send_test_to_email(self, email_address: str, message: str) -> dict[str, Any]:
        """Send test message via email."""
        try:
            import os
            import smtplib
            from email.mime.text import MIMEText

            smtp_host = os.environ.get("SMTP_HOST", "localhost")
            smtp_port = int(os.environ.get("SMTP_PORT", "587"))
            smtp_user = os.environ.get("SMTP_USER", "")
            smtp_password = os.environ.get("SMTP_PASSWORD", "")
            from_email = os.environ.get("SMTP_FROM", "aragora@localhost")

            msg = MIMEText(f"Test Receipt Delivery\n\n{message}\n\nSent from Aragora")
            msg["Subject"] = "Aragora Test Receipt Delivery"
            msg["From"] = from_email
            msg["To"] = email_address

            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                if smtp_user and smtp_password:
                    server.starttls()
                    server.login(smtp_user, smtp_password)
                server.send_message(msg)

            return {"success": True, "email_sent_to": email_address}
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.warning("%s receipt delivery failed: %s", "Email", e)
            return {"success": False, "error": "Internal server error"}

    def _send_test_to_webhook(self, webhook_url: str, message: str, org_id: str) -> dict[str, Any]:
        """Send test message to webhook."""
        try:
            import asyncio

            import httpx

            payload = {
                "type": "test_delivery",
                "message": message,
                "org_id": org_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            async def send():
                async with httpx.AsyncClient(timeout=10.0) as client:
                    return await client.post(
                        webhook_url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )

            resp = asyncio.run(send())

            if resp.status_code >= 200 and resp.status_code < 300:
                return {"success": True, "status_code": resp.status_code}
            return {"success": False, "error": f"HTTP {resp.status_code}"}
        except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
            logger.warning("%s receipt delivery failed: %s", "Webhook", e)
            return {"success": False, "error": "Internal server error"}

    @handle_errors("get delivery stats")
    @require_permission("sme:receipts:deliver")
    def _get_stats(
        self,
        handler,
        query_params: dict,
        user=None,
    ) -> HandlerResult:
        """
        Get receipt delivery statistics for the organization.

        Returns:
            JSON response with delivery statistics
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        history = self._get_delivery_history_store()
        org_history = [h for h in history if h.get("org_id") == org.id]

        # Calculate stats
        total_deliveries = len([h for h in org_history if not h.get("is_test")])
        successful = len(
            [h for h in org_history if h.get("status") == "success" and not h.get("is_test")]
        )
        failed = len(
            [h for h in org_history if h.get("status") == "failed" and not h.get("is_test")]
        )
        test_deliveries = len([h for h in org_history if h.get("is_test")])

        # Stats by channel type
        by_channel: dict[str, dict[str, int]] = {}
        for h in org_history:
            if h.get("is_test"):
                continue
            ct = h.get("channel_type", "unknown")
            if ct not in by_channel:
                by_channel[ct] = {"total": 0, "success": 0, "failed": 0}
            by_channel[ct]["total"] += 1
            if h.get("status") == "success":
                by_channel[ct]["success"] += 1
            else:
                by_channel[ct]["failed"] += 1

        # Get subscription count
        store = self._get_subscription_store()
        from aragora.storage.channel_subscription_store import EventType

        all_subs = store.get_by_org(org.id)
        receipt_subs = [
            s
            for s in all_subs
            if EventType.RECEIPT in s.event_types or EventType.RECEIPT.value in s.event_types
        ]

        return json_response(
            {
                "stats": {
                    "total_deliveries": total_deliveries,
                    "successful_deliveries": successful,
                    "failed_deliveries": failed,
                    "test_deliveries": test_deliveries,
                    "success_rate": (successful / total_deliveries * 100)
                    if total_deliveries > 0
                    else 0,
                    "active_subscriptions": len(receipt_subs),
                    "by_channel_type": by_channel,
                },
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )


__all__ = ["ReceiptDeliveryHandler"]
