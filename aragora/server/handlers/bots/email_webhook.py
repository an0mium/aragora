"""
Email Webhook Handler for Inbound Email Processing.

Handles inbound emails from:
- SendGrid Inbound Parse Webhook (HMAC-SHA256 signature verification)
- Mailgun Webhook (HMAC-SHA256 signature verification)
- AWS SES SNS Notifications

Security (Phase 3.1):
- SendGrid: Verifies X-Twilio-Email-Event-Webhook-Signature header using
  HMAC-SHA256 with base64-encoded digest. Uses timing-safe comparison via
  hmac.compare_digest() to prevent timing attacks.
- Mailgun: Verifies signature field in POST body using HMAC-SHA256 hex digest
  over (timestamp + token). Uses timing-safe comparison via hmac.compare_digest().
- SES: Validates SNS message structure and TopicArn format.
- All providers return 401 on verification failure.

Endpoints:
- POST /api/v1/bots/email/webhook/sendgrid - SendGrid Inbound Parse
- POST /api/v1/bots/email/webhook/mailgun  - Mailgun Inbound Webhook
- POST /api/v1/bots/email/webhook/ses      - AWS SES SNS notifications
- GET  /api/v1/bots/email/status           - Integration status

Environment Variables:
- EMAIL_INBOUND_SECRET - Secret for webhook signature verification
- SENDGRID_INBOUND_SECRET - SendGrid webhook signing key (falls back to EMAIL_INBOUND_SECRET)
- MAILGUN_WEBHOOK_SIGNING_KEY - Mailgun webhook signing key
- SES_NOTIFICATION_SECRET - SES-specific secret (falls back to EMAIL_INBOUND_SECRET)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from aragora.audit.unified import audit_data, audit_security

from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    json_response,
    handle_errors,
)
from aragora.server.handlers.bots.base import BotHandlerMixin
from aragora.server.handlers.secure import SecureHandler
from aragora.server.handlers.utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# Configuration
EMAIL_INBOUND_ENABLED = os.environ.get("EMAIL_INBOUND_ENABLED", "true").lower() == "true"


class EmailWebhookHandler(BotHandlerMixin, SecureHandler):
    """Handler for email inbound webhook endpoints.

    Uses BotHandlerMixin for shared auth/status patterns.

    RBAC Protected:
    - bots.read - required for status endpoint

    Note: Webhook endpoints are authenticated via platform-specific signatures,
    not RBAC, since they are called by SendGrid/AWS SES directly.
    """

    # BotHandlerMixin configuration
    bot_platform = "email"

    ROUTES = [
        "/api/v1/bots/email/webhook/sendgrid",
        "/api/v1/bots/email/webhook/mailgun",
        "/api/v1/bots/email/webhook/ses",
        "/api/v1/bots/email/status",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def _is_bot_enabled(self) -> bool:
        """Check if email inbound is enabled."""
        return EMAIL_INBOUND_ENABLED

    def _get_platform_config_status(self) -> dict[str, Any]:
        """Return email-specific config fields for status response."""
        sendgrid_configured = bool(os.environ.get("SENDGRID_INBOUND_SECRET"))
        mailgun_configured = bool(os.environ.get("MAILGUN_WEBHOOK_SIGNING_KEY"))
        ses_configured = bool(os.environ.get("SES_NOTIFICATION_SECRET"))

        return {
            "inbound_enabled": EMAIL_INBOUND_ENABLED,
            "providers": {
                "sendgrid": {
                    "configured": sendgrid_configured,
                    "webhook_url": "/api/v1/bots/email/webhook/sendgrid",
                    "signature_verification": "hmac-sha256",
                },
                "mailgun": {
                    "configured": mailgun_configured,
                    "webhook_url": "/api/v1/bots/email/webhook/mailgun",
                    "signature_verification": "hmac-sha256",
                },
                "ses": {
                    "configured": ses_configured,
                    "webhook_url": "/api/v1/bots/email/webhook/ses",
                },
            },
        }

    @rate_limit(requests_per_minute=30)
    async def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Route email GET requests with RBAC for status endpoint."""
        if path == "/api/v1/bots/email/status":
            # Use BotHandlerMixin's RBAC-protected status handler
            return await self.handle_status_request(handler)
        return None

    @handle_errors("email webhook creation")
    @rate_limit(requests_per_minute=120)
    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle POST requests (webhooks)."""
        if not EMAIL_INBOUND_ENABLED:
            return error_response("Email inbound processing disabled", 503)

        if path == "/api/v1/bots/email/webhook/sendgrid":
            return self._handle_sendgrid_webhook(handler)
        elif path == "/api/v1/bots/email/webhook/mailgun":
            return self._handle_mailgun_webhook(handler)
        elif path == "/api/v1/bots/email/webhook/ses":
            return self._handle_ses_webhook(handler)

        return None

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
            try:
                content_length = int(handler.headers.get("Content-Length", 0))
            except (ValueError, TypeError):
                return error_response("Invalid Content-Length", 400)
            if content_length > 10 * 1024 * 1024:
                return error_response("Request body too large", 413)
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
                asyncio.get_running_loop()
                _task = asyncio.create_task(handle_email_reply(email_data))
                _task.add_done_callback(
                    lambda t: logger.error("Email reply processing failed: %s", t.exception())
                    if not t.cancelled() and t.exception() else None
                )
            except RuntimeError:
                asyncio.run(handle_email_reply(email_data))

            return json_response({"status": "ok", "message_id": email_data.message_id})

        except ImportError as e:
            logger.error(f"Email reply loop module not available: {e}")
            return error_response("Email processing not available", 503)
        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as e:
            logger.exception("SendGrid webhook error: %s", e)
            # Return 200 to prevent retries
            return json_response({"status": "error", "message": "An error occurred processing the webhook"})

    def _handle_mailgun_webhook(self, handler: Any) -> HandlerResult:
        """
        Handle Mailgun inbound webhook.

        Mailgun sends POST with either multipart/form-data or JSON containing:
        - sender: Sender email address
        - recipient: Recipient email address
        - subject: Email subject
        - body-plain: Plain text body
        - body-html: HTML body
        - stripped-text: Text body with quoted parts removed
        - message-headers: JSON array of headers

        Signature verification uses three fields from the POST body:
        - signature.timestamp: Unix epoch seconds
        - signature.token: Random 50-char string
        - signature.signature: HMAC-SHA256 hex digest

        Security:
            Verifies HMAC-SHA256 signature using MAILGUN_WEBHOOK_SIGNING_KEY.
            Returns 401 if verification fails. Uses timing-safe comparison
            via hmac.compare_digest() to prevent timing attacks.
        """
        try:
            from aragora.integrations.email_reply_loop import (
                verify_mailgun_signature,
                handle_email_reply,
                InboundEmail,
            )

            try:
                content_length = int(handler.headers.get("Content-Length", 0))
            except (ValueError, TypeError):
                return error_response("Invalid Content-Length", 400)
            if content_length > 10 * 1024 * 1024:
                return error_response("Request body too large", 413)
            body = handler.rfile.read(content_length)

            # Parse body to extract signature fields and email data
            content_type = handler.headers.get("Content-Type", "")

            if "application/json" in content_type:
                try:
                    payload = json.loads(body.decode("utf-8"))
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in Mailgun webhook: {e}")
                    return error_response("Invalid JSON", 400)
            else:
                # Parse form data (multipart or urlencoded)
                payload = self._parse_form_data(body, content_type)

            # Extract signature fields
            # Mailgun sends signature data either at top level or nested under "signature"
            sig_data = payload.get("signature", {})
            if isinstance(sig_data, dict):
                mg_timestamp = str(sig_data.get("timestamp", ""))
                mg_token = sig_data.get("token", "")
                mg_signature = sig_data.get("signature", "")
            else:
                # Flat form data layout
                mg_timestamp = str(payload.get("timestamp", ""))
                mg_token = payload.get("token", "")
                mg_signature = payload.get("signature", "")

            # Verify HMAC-SHA256 signature
            if not verify_mailgun_signature(mg_timestamp, mg_token, mg_signature):
                logger.warning("Mailgun signature verification failed")
                audit_security(
                    event_type="email_webhook_auth_failed",
                    actor_id="unknown",
                    resource_type="mailgun_webhook",
                    resource_id="signature",
                    reason="hmac_sha256_verification_failed",
                )
                return error_response("Invalid signature", 401)

            # Extract email data from payload
            # Mailgun nests event data under "event-data" for event webhooks,
            # or provides fields directly for inbound routes
            event_data = payload.get("event-data", payload)

            from_email = (
                event_data.get("sender", "")
                or event_data.get("from", "")
                or event_data.get("From", "")
            )
            to_email = (
                event_data.get("recipient", "")
                or event_data.get("to", "")
                or event_data.get("To", "")
            )
            subject = event_data.get("subject", event_data.get("Subject", ""))
            body_plain = event_data.get("body-plain", event_data.get("stripped-text", ""))
            body_html = event_data.get("body-html", "")
            message_id = event_data.get("Message-Id", event_data.get("message-id", ""))

            # Parse headers if available
            headers = {}
            raw_headers = event_data.get("message-headers", "")
            if isinstance(raw_headers, str) and raw_headers:
                try:
                    header_list = json.loads(raw_headers)
                    for header_pair in header_list:
                        if isinstance(header_pair, (list, tuple)) and len(header_pair) >= 2:
                            headers[header_pair[0]] = header_pair[1]
                except (json.JSONDecodeError, TypeError):
                    pass
            elif isinstance(raw_headers, list):
                for header_pair in raw_headers:
                    if isinstance(header_pair, (list, tuple)) and len(header_pair) >= 2:
                        headers[header_pair[0]] = header_pair[1]

            if not message_id:
                import time as _time

                message_id = f"mailgun-{_time.time()}"

            email_data = InboundEmail(
                message_id=message_id,
                from_email=from_email,
                to_email=to_email,
                subject=subject,
                body_plain=body_plain,
                body_html=body_html,
                in_reply_to=headers.get("In-Reply-To", ""),
                references=(
                    headers.get("References", "").split() if headers.get("References") else []
                ),
                headers=headers,
            )

            logger.info(
                f"Mailgun inbound email from {email_data.from_email}: "
                f"subject='{email_data.subject[:50]}'"
            )

            audit_data(
                user_id=f"email:{email_data.from_email}",
                resource_type="inbound_email",
                resource_id=email_data.message_id,
                action="create",
                provider="mailgun",
                subject_preview=email_data.subject[:50],
            )

            # Process asynchronously
            import asyncio

            try:
                asyncio.get_running_loop()
                _task = asyncio.create_task(handle_email_reply(email_data))
                _task.add_done_callback(
                    lambda t: logger.error("Email reply processing failed: %s", t.exception())
                    if not t.cancelled() and t.exception() else None
                )
            except RuntimeError:
                asyncio.run(handle_email_reply(email_data))

            return json_response({"status": "ok", "message_id": email_data.message_id})

        except ImportError as e:
            logger.error(f"Email reply loop module not available: {e}")
            return error_response("Email processing not available", 503)
        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as e:
            logger.exception("Mailgun webhook error: %s", e)
            # Return 200 to prevent retries
            return json_response({"status": "error", "message": "An error occurred processing the webhook"})

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

            try:
                content_length = int(handler.headers.get("Content-Length", 0))
            except (ValueError, TypeError):
                return error_response("Invalid Content-Length", 400)
            if content_length > 10 * 1024 * 1024:
                return error_response("Request body too large", 413)
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
                asyncio.get_running_loop()
                _task = asyncio.create_task(handle_email_reply(email_data))
                _task.add_done_callback(
                    lambda t: logger.error("Email reply processing failed: %s", t.exception())
                    if not t.cancelled() and t.exception() else None
                )
            except RuntimeError:
                asyncio.run(handle_email_reply(email_data))

            return json_response({"status": "ok", "message_id": email_data.message_id})

        except ImportError as e:
            logger.error(f"Email reply loop module not available: {e}")
            return error_response("Email processing not available", 503)
        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as e:
            logger.exception("SES webhook error: %s", e)
            return json_response({"status": "error", "message": "An error occurred processing the webhook"})

    def _parse_form_data(self, body: bytes, content_type: str) -> dict[str, Any]:
        """Parse multipart form data or urlencoded data."""
        form_data: dict[str, Any] = {}

        if "multipart/form-data" in content_type:
            # Parse boundary from content type using email module (cgi is deprecated)
            from email.message import EmailMessage

            header_msg = EmailMessage()
            header_msg["Content-Type"] = content_type
            boundary = header_msg.get_param("boundary")

            if boundary:
                try:
                    from email.parser import BytesParser
                    from email.policy import default

                    # Reconstruct as email message for parsing
                    msg_bytes = b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body
                    msg = BytesParser(policy=default).parsebytes(msg_bytes)

                    if msg.is_multipart():
                        for part in msg.walk():
                            param = part.get_param("name", header="Content-Disposition")
                            if param and isinstance(param, str):
                                payload = part.get_payload(decode=True)
                                if isinstance(payload, bytes):
                                    form_data[param] = payload.decode("utf-8", errors="replace")
                except (ValueError, KeyError, TypeError, UnicodeDecodeError) as e:
                    logger.warning(f"Multipart parse error: {e}")

        elif "application/x-www-form-urlencoded" in content_type:
            from urllib.parse import parse_qs

            parsed = parse_qs(body.decode("utf-8"))
            for key, values in parsed.items():
                form_data[key] = values[0] if len(values) == 1 else values

        return form_data


__all__ = ["EmailWebhookHandler"]
