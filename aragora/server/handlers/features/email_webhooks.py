"""
Email Webhook Handlers.

Endpoints for receiving real-time notifications from email providers:
- Gmail Push Notifications (Google Pub/Sub)
- Outlook Change Notifications (Microsoft Graph webhooks)

Endpoints:
- POST /api/v1/webhooks/gmail              - Handle Gmail Pub/Sub notifications
- POST /api/v1/webhooks/outlook            - Handle Outlook Graph notifications
- POST /api/v1/webhooks/outlook/validate   - Handle Outlook subscription validation
- GET  /api/v1/webhooks/status             - Get webhook subscription status
- POST /api/v1/webhooks/subscribe          - Create new webhook subscription
- DELETE /api/v1/webhooks/unsubscribe      - Remove webhook subscription
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    success_response,
    json_response,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class WebhookProvider(Enum):
    """Supported webhook providers."""

    GMAIL = "gmail"
    OUTLOOK = "outlook"


class WebhookStatus(Enum):
    """Webhook subscription status."""

    ACTIVE = "active"
    PENDING = "pending"
    EXPIRED = "expired"
    ERROR = "error"


class NotificationType(Enum):
    """Types of email notifications."""

    MESSAGE_CREATED = "message_created"
    MESSAGE_UPDATED = "message_updated"
    MESSAGE_DELETED = "message_deleted"
    LABEL_CHANGED = "label_changed"
    SYNC_REQUESTED = "sync_requested"


@dataclass
class WebhookSubscription:
    """Webhook subscription record."""

    id: str
    tenant_id: str
    account_id: str
    provider: WebhookProvider
    status: WebhookStatus
    created_at: datetime
    expires_at: Optional[datetime] = None
    notification_url: str = ""
    client_state: str = ""
    last_notification: Optional[datetime] = None
    notification_count: int = 0
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "account_id": self.account_id,
            "provider": self.provider.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "notification_url": self.notification_url,
            "last_notification": self.last_notification.isoformat() if self.last_notification else None,
            "notification_count": self.notification_count,
            "error_count": self.error_count,
        }


@dataclass
class WebhookNotification:
    """Parsed webhook notification."""

    provider: WebhookProvider
    notification_type: NotificationType
    account_id: str
    resource_id: str
    tenant_id: str
    timestamp: datetime
    raw_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "provider": self.provider.value,
            "notification_type": self.notification_type.value,
            "account_id": self.account_id,
            "resource_id": self.resource_id,
            "tenant_id": self.tenant_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


# =============================================================================
# In-Memory Storage (replace with database in production)
# =============================================================================

_subscriptions: Dict[str, WebhookSubscription] = {}  # subscription_id -> subscription
_tenant_subscriptions: Dict[str, List[str]] = {}  # tenant_id -> [subscription_ids]
_notification_history: Dict[str, List[WebhookNotification]] = {}  # tenant_id -> notifications
_pending_validations: Dict[str, str] = {}  # state -> subscription_id


# =============================================================================
# Webhook Processing
# =============================================================================


async def process_gmail_notification(
    notification_data: Dict[str, Any],
    tenant_id: str,
) -> Optional[WebhookNotification]:
    """Process Gmail Pub/Sub notification.

    Gmail notifications come as base64-encoded messages with structure:
    {
        "message": {
            "data": "base64-encoded-data",
            "messageId": "...",
            "publishTime": "..."
        },
        "subscription": "projects/.../subscriptions/..."
    }

    The decoded data contains:
    {
        "emailAddress": "user@gmail.com",
        "historyId": "12345"
    }
    """
    try:
        message = notification_data.get("message", {})
        data_b64 = message.get("data", "")

        if not data_b64:
            logger.warning("Gmail notification missing data field")
            return None

        # Decode base64 data
        try:
            data_bytes = base64.b64decode(data_b64)
            data = json.loads(data_bytes.decode("utf-8"))
        except Exception as e:
            logger.error(f"Failed to decode Gmail notification data: {e}")
            return None

        email_address = data.get("emailAddress", "")
        history_id = data.get("historyId", "")

        if not email_address:
            logger.warning("Gmail notification missing emailAddress")
            return None

        # Find the account and subscription
        account_id = _find_account_by_email(email_address, tenant_id)

        notification = WebhookNotification(
            provider=WebhookProvider.GMAIL,
            notification_type=NotificationType.SYNC_REQUESTED,
            account_id=account_id or "",
            resource_id=history_id,
            tenant_id=tenant_id,
            timestamp=datetime.now(timezone.utc),
            raw_data=notification_data,
            metadata={
                "email_address": email_address,
                "history_id": history_id,
            },
        )

        # Queue for processing
        await _queue_notification(notification)

        logger.info(
            f"Processed Gmail notification for {email_address}, history_id={history_id}"
        )

        return notification

    except Exception as e:
        logger.exception(f"Error processing Gmail notification: {e}")
        return None


async def process_outlook_notification(
    notification_data: Dict[str, Any],
    tenant_id: str,
    client_state: Optional[str] = None,
) -> List[WebhookNotification]:
    """Process Outlook Graph change notification.

    Outlook notifications have structure:
    {
        "value": [
            {
                "subscriptionId": "...",
                "changeType": "created|updated|deleted",
                "resource": "Users/{user-id}/Messages/{message-id}",
                "clientState": "...",
                "tenantId": "...",
                "resourceData": {...}
            }
        ]
    }
    """
    notifications = []

    try:
        changes = notification_data.get("value", [])

        for change in changes:
            # Verify client state if provided
            change_client_state = change.get("clientState")
            if client_state and change_client_state != client_state:
                logger.warning(
                    f"Client state mismatch: expected={client_state}, "
                    f"got={change_client_state}"
                )
                continue

            # Parse change type
            change_type = change.get("changeType", "").lower()
            if change_type == "created":
                notification_type = NotificationType.MESSAGE_CREATED
            elif change_type == "updated":
                notification_type = NotificationType.MESSAGE_UPDATED
            elif change_type == "deleted":
                notification_type = NotificationType.MESSAGE_DELETED
            else:
                notification_type = NotificationType.SYNC_REQUESTED

            # Extract resource info
            resource = change.get("resource", "")
            subscription_id = change.get("subscriptionId", "")

            # Find account from subscription
            subscription = _subscriptions.get(subscription_id)
            account_id = subscription.account_id if subscription else ""

            notification = WebhookNotification(
                provider=WebhookProvider.OUTLOOK,
                notification_type=notification_type,
                account_id=account_id,
                resource_id=resource,
                tenant_id=tenant_id,
                timestamp=datetime.now(timezone.utc),
                raw_data=change,
                metadata={
                    "subscription_id": subscription_id,
                    "change_type": change_type,
                    "tenant_id": change.get("tenantId", ""),
                },
            )

            notifications.append(notification)
            await _queue_notification(notification)

            logger.info(
                f"Processed Outlook notification: {change_type} for {resource}"
            )

        return notifications

    except Exception as e:
        logger.exception(f"Error processing Outlook notification: {e}")
        return []


async def _queue_notification(notification: WebhookNotification) -> None:
    """Queue notification for async processing."""
    tenant_id = notification.tenant_id

    if tenant_id not in _notification_history:
        _notification_history[tenant_id] = []

    # Keep last 100 notifications per tenant
    history = _notification_history[tenant_id]
    history.append(notification)
    if len(history) > 100:
        _notification_history[tenant_id] = history[-100:]

    # Update subscription stats
    for sub in _subscriptions.values():
        if sub.account_id == notification.account_id:
            sub.last_notification = notification.timestamp
            sub.notification_count += 1
            break

    # Trigger sync service (would integrate with GmailSyncService/OutlookSyncService)
    try:
        if notification.provider == WebhookProvider.GMAIL:
            await _trigger_gmail_sync(notification)
        else:
            await _trigger_outlook_sync(notification)
    except Exception as e:
        logger.warning(f"Failed to trigger sync: {e}")


async def _trigger_gmail_sync(notification: WebhookNotification) -> None:
    """Trigger Gmail sync for new notification."""
    try:
        from aragora.connectors.email import GmailSyncService

        # In production, get the sync service instance for this account
        # and call sync_from_history_id(notification.metadata["history_id"])
        logger.debug(f"Would trigger Gmail sync for history_id={notification.resource_id}")

    except ImportError:
        pass


async def _trigger_outlook_sync(notification: WebhookNotification) -> None:
    """Trigger Outlook sync for new notification."""
    try:
        from aragora.connectors.email import OutlookSyncService

        # In production, get the sync service instance and fetch the message
        logger.debug(f"Would trigger Outlook sync for resource={notification.resource_id}")

    except ImportError:
        pass


def _find_account_by_email(email: str, tenant_id: str) -> Optional[str]:
    """Find account ID by email address."""
    # In production, look up in database
    return None


# =============================================================================
# Handler Class
# =============================================================================


class EmailWebhooksHandler(BaseHandler):
    """Handler for email webhook endpoints."""

    ROUTES = [
        "/api/v1/webhooks/gmail",
        "/api/v1/webhooks/outlook",
        "/api/v1/webhooks/outlook/validate",
        "/api/v1/webhooks/status",
        "/api/v1/webhooks/subscribe",
        "/api/v1/webhooks/unsubscribe",
        "/api/v1/webhooks/history",
    ]

    def __init__(self, server_context: Optional[Dict[str, Any]] = None):
        """Initialize handler with optional server context."""
        super().__init__(server_context or {})

    async def handle(self, request: Any, path: str, method: str) -> HandlerResult:
        """Route requests to appropriate handler methods."""
        try:
            tenant_id = self._get_tenant_id(request)

            # Gmail webhook
            if path == "/api/v1/webhooks/gmail" and method == "POST":
                return await self._handle_gmail_webhook(request, tenant_id)

            # Outlook webhook
            elif path == "/api/v1/webhooks/outlook" and method == "POST":
                return await self._handle_outlook_webhook(request, tenant_id)

            # Outlook validation
            elif path == "/api/v1/webhooks/outlook/validate" and method == "POST":
                return await self._handle_outlook_validation(request)

            # Status
            elif path == "/api/v1/webhooks/status" and method == "GET":
                return await self._handle_status(request, tenant_id)

            # Subscribe
            elif path == "/api/v1/webhooks/subscribe" and method == "POST":
                return await self._handle_subscribe(request, tenant_id)

            # Unsubscribe
            elif path == "/api/v1/webhooks/unsubscribe" and method in ("POST", "DELETE"):
                return await self._handle_unsubscribe(request, tenant_id)

            # History
            elif path == "/api/v1/webhooks/history" and method == "GET":
                return await self._handle_history(request, tenant_id)

            return error_response("Not found", 404)

        except Exception as e:
            logger.exception(f"Error in webhook handler: {e}")
            return error_response(f"Internal error: {str(e)}", 500)

    def _get_tenant_id(self, request: Any) -> str:
        """Extract tenant ID from request context."""
        return getattr(request, "tenant_id", "default")

    # =========================================================================
    # Gmail Webhook
    # =========================================================================

    async def _handle_gmail_webhook(
        self, request: Any, tenant_id: str
    ) -> HandlerResult:
        """Handle Gmail Pub/Sub push notification.

        Google sends notifications to this endpoint when there are
        changes to the user's mailbox.
        """
        try:
            body = await self._get_json_body(request)

            # Process the notification
            notification = await process_gmail_notification(body, tenant_id)

            if notification:
                return success_response({
                    "status": "processed",
                    "notification": notification.to_dict(),
                })
            else:
                # Return 200 to acknowledge receipt even if processing failed
                # (to prevent Google from retrying)
                return success_response({
                    "status": "acknowledged",
                    "message": "Notification received but not processed",
                })

        except Exception as e:
            logger.exception(f"Error handling Gmail webhook: {e}")
            # Return 200 to acknowledge
            return success_response({"status": "error", "message": str(e)})

    # =========================================================================
    # Outlook Webhook
    # =========================================================================

    async def _handle_outlook_webhook(
        self, request: Any, tenant_id: str
    ) -> HandlerResult:
        """Handle Outlook Graph change notification.

        Microsoft sends change notifications when subscribed resources change.
        """
        try:
            # Check for validation request
            params = self._get_query_params(request)
            validation_token = params.get("validationToken")

            if validation_token:
                # This is a subscription validation request
                # Must respond with the token in plain text
                return HandlerResult(
                    body=validation_token.encode("utf-8"),
                    status_code=200,
                    content_type="text/plain",
                )

            body = await self._get_json_body(request)

            # Process notifications
            notifications = await process_outlook_notification(body, tenant_id)

            return success_response({
                "status": "processed",
                "count": len(notifications),
                "notifications": [n.to_dict() for n in notifications],
            })

        except Exception as e:
            logger.exception(f"Error handling Outlook webhook: {e}")
            return success_response({"status": "error", "message": str(e)})

    async def _handle_outlook_validation(self, request: Any) -> HandlerResult:
        """Handle Outlook subscription validation.

        When creating a subscription, Microsoft sends a validation request
        that must echo back the validationToken.
        """
        params = self._get_query_params(request)
        validation_token = params.get("validationToken", "")

        if not validation_token:
            return error_response("Missing validationToken", 400)

        # Return token as plain text
        return HandlerResult(
            body=validation_token.encode("utf-8"),
            status_code=200,
            content_type="text/plain",
        )

    # =========================================================================
    # Subscription Management
    # =========================================================================

    async def _handle_status(self, request: Any, tenant_id: str) -> HandlerResult:
        """Get webhook subscription status."""
        subscription_ids = _tenant_subscriptions.get(tenant_id, [])
        subscriptions = [
            _subscriptions[sid].to_dict()
            for sid in subscription_ids
            if sid in _subscriptions
        ]

        # Calculate summary
        active_count = sum(
            1 for s in subscriptions if s["status"] == "active"
        )
        total_notifications = sum(
            s["notification_count"] for s in subscriptions
        )

        return success_response({
            "subscriptions": subscriptions,
            "summary": {
                "total": len(subscriptions),
                "active": active_count,
                "total_notifications": total_notifications,
            },
        })

    async def _handle_subscribe(
        self, request: Any, tenant_id: str
    ) -> HandlerResult:
        """Create new webhook subscription.

        Request body:
        {
            "provider": "gmail" | "outlook",
            "account_id": "...",
            "notification_url": "https://...",
            "expiration_hours": 72
        }
        """
        try:
            body = await self._get_json_body(request)

            provider_str = body.get("provider", "").lower()
            if provider_str not in ["gmail", "outlook"]:
                return error_response("Invalid provider", 400)

            provider = WebhookProvider(provider_str)
            account_id = body.get("account_id", "")
            notification_url = body.get("notification_url", "")
            expiration_hours = body.get("expiration_hours", 72)

            if not account_id:
                return error_response("Missing account_id", 400)

            # Create subscription
            subscription_id = str(uuid4())
            client_state = hashlib.sha256(
                f"{tenant_id}:{account_id}:{subscription_id}".encode()
            ).hexdigest()[:32]

            now = datetime.now(timezone.utc)
            subscription = WebhookSubscription(
                id=subscription_id,
                tenant_id=tenant_id,
                account_id=account_id,
                provider=provider,
                status=WebhookStatus.PENDING,
                created_at=now,
                expires_at=now + timedelta(hours=expiration_hours),
                notification_url=notification_url,
                client_state=client_state,
            )

            # Create actual subscription with provider
            if provider == WebhookProvider.GMAIL:
                result = await self._create_gmail_subscription(subscription)
            else:
                result = await self._create_outlook_subscription(subscription)

            if result.get("success"):
                subscription.status = WebhookStatus.ACTIVE
                _subscriptions[subscription_id] = subscription

                if tenant_id not in _tenant_subscriptions:
                    _tenant_subscriptions[tenant_id] = []
                _tenant_subscriptions[tenant_id].append(subscription_id)

                logger.info(
                    f"Created {provider.value} webhook subscription: {subscription_id}"
                )

                return success_response({
                    "subscription": subscription.to_dict(),
                    "client_state": client_state,
                })
            else:
                return error_response(
                    result.get("error", "Failed to create subscription"), 400
                )

        except Exception as e:
            logger.exception(f"Error creating subscription: {e}")
            return error_response(f"Failed to create subscription: {str(e)}", 500)

    async def _create_gmail_subscription(
        self, subscription: WebhookSubscription
    ) -> Dict[str, Any]:
        """Create Gmail Pub/Sub watch."""
        try:
            from aragora.connectors.email import GmailSyncService

            # In production, call Gmail API to create push notification watch
            # This requires Pub/Sub topic and project configuration
            return {"success": True}

        except ImportError:
            return {"success": True}  # Simulate success for testing
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _create_outlook_subscription(
        self, subscription: WebhookSubscription
    ) -> Dict[str, Any]:
        """Create Outlook Graph subscription."""
        try:
            from aragora.connectors.email import OutlookSyncService

            # In production, call Microsoft Graph API to create subscription
            return {"success": True}

        except ImportError:
            return {"success": True}  # Simulate success for testing
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_unsubscribe(
        self, request: Any, tenant_id: str
    ) -> HandlerResult:
        """Remove webhook subscription.

        Request body:
        {
            "subscription_id": "..."
        }
        """
        try:
            body = await self._get_json_body(request)
            subscription_id = body.get("subscription_id", "")

            if not subscription_id:
                return error_response("Missing subscription_id", 400)

            if subscription_id not in _subscriptions:
                return error_response("Subscription not found", 404)

            subscription = _subscriptions[subscription_id]

            # Verify tenant access
            if subscription.tenant_id != tenant_id:
                return error_response("Not authorized", 403)

            # Remove from provider
            if subscription.provider == WebhookProvider.GMAIL:
                await self._delete_gmail_subscription(subscription)
            else:
                await self._delete_outlook_subscription(subscription)

            # Remove from storage
            del _subscriptions[subscription_id]
            if tenant_id in _tenant_subscriptions:
                _tenant_subscriptions[tenant_id] = [
                    sid for sid in _tenant_subscriptions[tenant_id]
                    if sid != subscription_id
                ]

            logger.info(f"Deleted webhook subscription: {subscription_id}")

            return success_response({
                "status": "deleted",
                "subscription_id": subscription_id,
            })

        except Exception as e:
            logger.exception(f"Error deleting subscription: {e}")
            return error_response(f"Failed to delete subscription: {str(e)}", 500)

    async def _delete_gmail_subscription(
        self, subscription: WebhookSubscription
    ) -> None:
        """Delete Gmail Pub/Sub watch."""
        # In production, call Gmail API to stop the watch
        pass

    async def _delete_outlook_subscription(
        self, subscription: WebhookSubscription
    ) -> None:
        """Delete Outlook Graph subscription."""
        # In production, call Microsoft Graph API to delete subscription
        pass

    async def _handle_history(
        self, request: Any, tenant_id: str
    ) -> HandlerResult:
        """Get notification history."""
        params = self._get_query_params(request)
        limit = int(params.get("limit", 50))

        history = _notification_history.get(tenant_id, [])
        history = history[-limit:]  # Get last N

        return success_response({
            "notifications": [n.to_dict() for n in reversed(history)],
            "total": len(history),
        })

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

_handler_instance: Optional[EmailWebhooksHandler] = None


def get_email_webhooks_handler() -> EmailWebhooksHandler:
    """Get or create handler instance."""
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = EmailWebhooksHandler()
    return _handler_instance


async def handle_email_webhooks(
    request: Any, path: str, method: str
) -> HandlerResult:
    """Entry point for email webhook requests."""
    handler = get_email_webhooks_handler()
    return await handler.handle(request, path, method)


__all__ = [
    "EmailWebhooksHandler",
    "handle_email_webhooks",
    "get_email_webhooks_handler",
    "WebhookProvider",
    "WebhookStatus",
    "NotificationType",
    "WebhookSubscription",
    "WebhookNotification",
    "process_gmail_notification",
    "process_outlook_notification",
]
