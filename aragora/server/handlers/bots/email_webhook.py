"""
Email Webhook Handler for Inbound Email Processing.

Handles inbound emails from:
- SendGrid Inbound Parse Webhook
- AWS SES SNS Notifications

Endpoints:
- POST /api/bots/email/webhook/sendgrid - SendGrid Inbound Parse
- POST /api/bots/email/webhook/ses - AWS SES SNS notifications
- GET  /api/bots/email/status - Integration status

Environment Variables:
- EMAIL_INBOUND_SECRET - Secret for webhook signature verification
- SENDGRID_INBOUND_SECRET - SendGrid-specific secret (falls back to EMAIL_INBOUND_SECRET)
- SES_NOTIFICATION_SECRET - SES-specific secret (falls back to EMAIL_INBOUND_SECRET)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from aragora.audit.unified import audit_data, audit_security
from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    json_response,
)
from aragora.server.handlers.secure import SecureHandler, ForbiddenError, UnauthorizedError
from aragora.server.handlers.utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# RBAC permission for bot configuration endpoints
BOTS_READ_PERMISSION = "bots:read"

# Configuration
EMAIL_INBOUND_ENABLED = os.environ.get("EMAIL_INBOUND_ENABLED", "true").lower() == "true"


class EmailWebhookHandler(SecureHandler):
    """Handler for email inbound webhook endpoints.

    RBAC Protected:
    - bots:read - required for status endpoint

    Note: Webhook endpoints are authenticated via platform-specific signatures,
    not RBAC, since they are called by SendGrid/AWS SES directly.
    """

    ROUTES = [
        "/api/v1/bots/email/webhook/sendgrid",
        "/api/v1/bots/email/webhook/ses",
        "/api/v1/bots/email/status",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    @rate_limit(rpm=30)
    async def handle(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route email GET requests with RBAC for status endpoint."""
        if path == "/api/v1/bots/email/status":
            # RBAC: Require authentication and bots:read permission
            try:
                auth_context = await self.get_auth_context(handler, require_auth=True)
                self.check_permission(auth_context, BOTS_READ_PERMISSION)
            except UnauthorizedError:
                return error_response("Authentication required", 401)
            except ForbiddenError as e:
                logger.warning(f"Email status access denied: {e}")
                return error_response(str(e), 403)
            return self._get_status()
        return None

    @rate_limit(rpm=120)
    def handle_post(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle POST requests (webhooks)."""
        if not EMAIL_INBOUND_ENABLED:
            return error_response("Email inbound processing disabled", 503)

        if path == "/api/v1/bots/email/webhook/sendgrid":
            return self._handle_sendgrid_webhook(handler)
        elif path == "/api/v1/bots/email/webhook/ses":
            return self._handle_ses_webhook(handler)

        return None

    def _get_status(self) -> HandlerResult:
        """Get email integration status."""
        sendgrid_configured = bool(os.environ.get("SENDGRID_INBOUND_SECRET"))
        ses_configured = bool(os.environ.get("SES_NOTIFICATION_SECRET"))

        return json_response(
            {
                "platform": "email",
                "inbound_enabled": EMAIL_INBOUND_ENABLED,
                "providers": {
                    "sendgrid": {
                        "configured": sendgrid_configured,
                        "webhook_url": "/api/v1/bots/email/webhook/sendgrid",
                    },
                    "ses": {
                        "configured": ses_configured,
                        "webhook_url": "/api/v1/bots/email/webhook/ses",
                    },
                },
            }
        )

    def _handle_sendgrid_webhook(self, handler: Any) -> HandlerResult:
        """
        Handle SendGrid Inbound Parse webhook.

        SendGrid sends POST with multipart/form-data containing:
        - headers: Email headers as text
        - text: Plain text body
        - html: HTML body
        - from: Sender address
        - to: Recipient address
        - subject: Email subject
        - envelope: JSON envelope data
        """
        try:
            from aragora.integrations.email_reply_loop import (
                parse_sendgrid_webhook,
                verify_sendgrid_signature,
                handle_email_reply,
            )

            # Verify signature if configured
            timestamp = handler.headers.get("X-Twilio-Email-Event-Webhook-Timestamp", "")
            signature = handler.headers.get("X-Twilio-Email-Event-Webhook-Signature", "")
            content_length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(content_length)

            if not verify_sendgrid_signature(body, timestamp, signature):
                logger.warning("SendGrid signature verification failed")
                audit_security(
                    event_type="email_webhook_auth_failed",
                    actor_id="unknown",
                    resource_type="sendgrid_webhook",
                    resource_id="signature",
                    reason="signature_verification_failed",
                )
                return error_response("Invalid signature", 401)

            # Parse form data
            content_type = handler.headers.get("Content-Type", "")
            form_data = self._parse_form_data(body, content_type)

            # Parse email
            email_data = parse_sendgrid_webhook(form_data)

            logger.info(
                f"SendGrid inbound email from {email_data.from_email}: "
                f"subject='{email_data.subject[:50]}'"
            )

            audit_data(
                user_id=f"email:{email_data.from_email}",
                resource_type="inbound_email",
                resource_id=email_data.message_id,
                action="create",
                provider="sendgrid",
                subject_preview=email_data.subject[:50],
            )

            # Process asynchronously
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(handle_email_reply(email_data))
                else:
                    loop.run_until_complete(handle_email_reply(email_data))
            except RuntimeError:
                asyncio.run(handle_email_reply(email_data))

            return json_response({"status": "ok", "message_id": email_data.message_id})

        except ImportError as e:
            logger.error(f"Email reply loop module not available: {e}")
            return error_response("Email processing not available", 503)
        except Exception as e:
            logger.exception(f"SendGrid webhook error: {e}")
            # Return 200 to prevent retries
            return json_response({"status": "error", "message": str(e)[:100]})

    def _handle_ses_webhook(self, handler: Any) -> HandlerResult:
        """
        Handle AWS SES SNS notification.

        SES sends JSON notifications via SNS:
        - SubscriptionConfirmation: Needs to be confirmed
        - Notification: Contains email receipt data
        """
        try:
            from aragora.integrations.email_reply_loop import (
                parse_ses_notification,
                verify_ses_signature,
                handle_email_reply,
            )

            content_length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(content_length)

            try:
                notification = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in SES webhook: {e}")
                return error_response("Invalid JSON", 400)

            # Verify signature
            if not verify_ses_signature(notification):
                logger.warning("SES signature verification failed")
                audit_security(
                    event_type="email_webhook_auth_failed",
                    actor_id="unknown",
                    resource_type="ses_webhook",
                    resource_id="signature",
                    reason="signature_verification_failed",
                )
                return error_response("Invalid signature", 401)

            # Handle subscription confirmation
            msg_type = notification.get("Type", "")
            if msg_type == "SubscriptionConfirmation":
                subscribe_url = notification.get("SubscribeURL")
                if subscribe_url:
                    logger.info(f"SES subscription confirmation needed: {subscribe_url}")
                    # In production, auto-confirm by fetching the URL
                    return json_response(
                        {
                            "status": "subscription_pending",
                            "subscribe_url": subscribe_url,
                        }
                    )

            # Parse notification
            email_data = parse_ses_notification(notification)
            if email_data is None:
                logger.debug("SES notification is not an email receipt")
                return json_response({"status": "ignored"})

            logger.info(
                f"SES inbound email from {email_data.from_email}: "
                f"subject='{email_data.subject[:50]}'"
            )

            audit_data(
                user_id=f"email:{email_data.from_email}",
                resource_type="inbound_email",
                resource_id=email_data.message_id,
                action="create",
                provider="ses",
                subject_preview=email_data.subject[:50],
            )

            # Process asynchronously
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(handle_email_reply(email_data))
                else:
                    loop.run_until_complete(handle_email_reply(email_data))
            except RuntimeError:
                asyncio.run(handle_email_reply(email_data))

            return json_response({"status": "ok", "message_id": email_data.message_id})

        except ImportError as e:
            logger.error(f"Email reply loop module not available: {e}")
            return error_response("Email processing not available", 503)
        except Exception as e:
            logger.exception(f"SES webhook error: {e}")
            return json_response({"status": "error", "message": str(e)[:100]})

    def _parse_form_data(self, body: bytes, content_type: str) -> Dict[str, Any]:
        """Parse multipart form data or urlencoded data."""
        import cgi
        from io import BytesIO

        form_data = {}

        if "multipart/form-data" in content_type:
            # Parse boundary from content type
            _, params = cgi.parse_header(content_type)
            boundary = params.get("boundary")

            if boundary:
                # Use cgi.parse_multipart for parsing
                {
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                    "CONTENT_LENGTH": str(len(body)),
                }
                fp = BytesIO(body)

                try:
                    # Python 3.11+ changed parse_multipart signature
                    import sys

                    if sys.version_info >= (3, 11):
                        from email.parser import BytesParser
                        from email.policy import default

                        # Reconstruct as email message
                        msg_bytes = b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body
                        msg = BytesParser(policy=default).parsebytes(msg_bytes)

                        if msg.is_multipart():
                            for part in msg.walk():
                                name = part.get_param("name", header="Content-Disposition")
                                if name:
                                    payload = part.get_payload(decode=True)
                                    if payload:
                                        form_data[name] = payload.decode("utf-8", errors="replace")
                    else:
                        parsed = cgi.parse_multipart(fp, {"boundary": boundary.encode()})
                        for key, values in parsed.items():
                            if values:
                                form_data[key] = values[0] if len(values) == 1 else values
                except Exception as e:
                    logger.warning(f"Multipart parse error: {e}")

        elif "application/x-www-form-urlencoded" in content_type:
            from urllib.parse import parse_qs

            parsed = parse_qs(body.decode("utf-8"))
            for key, values in parsed.items():
                form_data[key] = values[0] if len(values) == 1 else values

        return form_data


__all__ = ["EmailWebhookHandler"]
