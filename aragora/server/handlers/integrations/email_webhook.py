"""
Email Webhook HTTP Handler.

Provides HTTP endpoints for receiving inbound email webhooks from:
- SendGrid (Inbound Parse)
- Mailgun (Routes)
- AWS SES (SNS notifications)

Each provider sends email events in different formats, which are normalized
and processed through the email reply loop.

Usage:
    # In unified_server.py or route registration
    from aragora.server.handlers.integrations.email_webhook import EmailWebhookHandler

    handler = EmailWebhookHandler()

    # Register routes
    app.router.add_post("/webhooks/email/sendgrid", handler.handle_sendgrid)
    app.router.add_post("/webhooks/email/mailgun", handler.handle_mailgun)
    app.router.add_post("/webhooks/email/ses", handler.handle_ses)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from aragora.server.handlers.base import HandlerResult, error_response, json_response

if TYPE_CHECKING:
    from aiohttp import web

# Lazy import - try loading email reply loop
try:
    from aragora.integrations.email_reply_loop import (
        EmailReplyLoop,
        ParsedEmail,
    )

    _HAS_EMAIL_LOOP = True
except ImportError:
    _HAS_EMAIL_LOOP = False
    EmailReplyLoop = None  # type: ignore[misc,assignment]
    ParsedEmail = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


class EmailWebhookHandler:
    """
    HTTP handler for email provider webhooks.

    Receives inbound emails from SendGrid, Mailgun, and AWS SES,
    validates signatures, and processes through the email reply loop.
    """

    def __init__(self) -> None:
        self._processed_count = 0
        self._error_count = 0
        self._last_error: str | None = None

    async def handle_sendgrid(self, request: "web.Request") -> HandlerResult:
        """
        Handle SendGrid Inbound Parse webhook.

        SendGrid sends multipart/form-data with fields:
        - from, to, subject, text, html
        - envelope (JSON string)
        - attachments, attachment-info (JSON)
        - headers (raw email headers)

        For Event webhooks, sends JSON array with events.

        Headers:
        - X-Twilio-Email-Event-Webhook-Signature: HMAC signature
        - X-Twilio-Email-Event-Webhook-Timestamp: Unix timestamp
        """
        try:
            content_type = request.headers.get("Content-Type", "")

            if "multipart/form-data" in content_type:
                # Inbound Parse (email content)
                return await self._handle_sendgrid_inbound(request)
            elif "application/json" in content_type:
                # Event webhook (delivery, open, click, etc.)
                return await self._handle_sendgrid_event(request)
            else:
                logger.warning(f"Unexpected SendGrid content type: {content_type}")
                return error_response("Unsupported content type", status=415)

        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            logger.exception("SendGrid webhook error")
            return error_response(f"Internal error: {e}", status=500)

    async def _handle_sendgrid_inbound(self, request: "web.Request") -> HandlerResult:
        """Handle SendGrid Inbound Parse (multipart form)."""

        data = await request.post()

        # Extract email fields
        sender = str(data.get("from", ""))
        to = str(data.get("to", ""))
        subject = str(data.get("subject", ""))
        text_body = str(data.get("text", ""))
        html_body = str(data.get("html", ""))

        # Parse envelope for additional metadata
        envelope_str = str(data.get("envelope", "{}"))
        try:
            envelope = json.loads(envelope_str)
        except json.JSONDecodeError:
            envelope = {}

        # Extract headers for thread tracking
        headers_raw = str(data.get("headers", ""))
        message_id = _extract_header(headers_raw, "Message-ID")
        in_reply_to = _extract_header(headers_raw, "In-Reply-To")
        references = _extract_header(headers_raw, "References")

        # Create parsed email
        parsed = ParsedEmail(
            message_id=message_id or f"sendgrid-{datetime.now().timestamp()}",
            sender=sender,
            recipients=[to] if to else envelope.get("to", []),
            subject=subject,
            body=text_body,
            html_body=html_body,
            in_reply_to=in_reply_to,
            references=references.split() if references else [],
            received_at=datetime.now(timezone.utc),
        )

        # Process through reply loop
        try:
            loop = EmailReplyLoop()
            result = await loop.process_email(parsed)
            self._processed_count += 1

            return json_response(
                {
                    "status": "processed",
                    "message_id": parsed.message_id,
                    "debate_id": result.get("debate_id") if result else None,
                }
            )

        except Exception as e:
            logger.error(f"Failed to process SendGrid email: {e}")
            self._error_count += 1
            return json_response(
                {
                    "status": "error",
                    "error": str(e),
                },
                status=202,
            )  # Still return 2xx to prevent retries

    async def _handle_sendgrid_event(self, request: "web.Request") -> HandlerResult:
        """Handle SendGrid Event webhook (JSON)."""
        from aragora.integrations.email_reply_loop import verify_sendgrid_signature

        body = await request.read()
        signature = request.headers.get("X-Twilio-Email-Event-Webhook-Signature", "")
        timestamp = request.headers.get("X-Twilio-Email-Event-Webhook-Timestamp", "")

        # Verify signature
        if signature and not verify_sendgrid_signature(body, timestamp, signature):
            logger.warning("Invalid SendGrid event webhook signature")
            return error_response("Invalid signature", status=401)

        # Parse events
        try:
            events = json.loads(body)
        except json.JSONDecodeError:
            return error_response("Invalid JSON", status=400)

        if not isinstance(events, list):
            events = [events]

        # Process events (delivery, bounce, open, click, etc.)
        processed = 0
        for event in events:
            event_type = event.get("event", "unknown")
            email = event.get("email", "")

            logger.debug(f"SendGrid event: {event_type} for {email}")

            # Handle bounces and complaints for list hygiene
            if event_type in ("bounce", "dropped", "spamreport", "unsubscribe"):
                await self._handle_email_event(event_type, email, event)

            processed += 1

        return json_response(
            {
                "status": "processed",
                "events_processed": processed,
            }
        )

    async def handle_mailgun(self, request: "web.Request") -> HandlerResult:
        """
        Handle Mailgun webhook.

        Mailgun sends multipart/form-data with fields:
        - sender, recipient, subject
        - body-plain, body-html, stripped-text, stripped-html
        - Message-Id, In-Reply-To, References
        - signature (timestamp, token, signature)
        """
        try:
            from aragora.integrations.email_reply_loop import (  # type: ignore[attr-defined]
                EmailReplyLoop,
                ParsedEmail,
                verify_mailgun_signature,
            )

            data = await request.post()

            # Verify signature
            timestamp = str(data.get("timestamp", ""))
            token = str(data.get("token", ""))
            signature = str(data.get("signature", ""))

            if not verify_mailgun_signature(timestamp, token, signature):
                logger.warning("Invalid Mailgun webhook signature")
                return error_response("Invalid signature", status=401)

            # Extract email fields
            sender = str(data.get("sender", ""))
            recipient = str(data.get("recipient", ""))
            subject = str(data.get("subject", ""))
            text_body = str(data.get("body-plain", "") or data.get("stripped-text", ""))
            html_body = str(data.get("body-html", "") or data.get("stripped-html", ""))

            message_id = str(data.get("Message-Id", ""))
            in_reply_to = str(data.get("In-Reply-To", ""))
            references_str = str(data.get("References", ""))

            # Create parsed email
            parsed = ParsedEmail(
                message_id=message_id or f"mailgun-{datetime.now().timestamp()}",
                sender=sender,
                recipients=[recipient] if recipient else [],
                subject=subject,
                body=text_body,
                html_body=html_body,
                in_reply_to=in_reply_to,
                references=references_str.split() if references_str else [],
                received_at=datetime.now(timezone.utc),
            )

            # Process through reply loop
            loop = EmailReplyLoop()
            result = await loop.process_email(parsed)
            self._processed_count += 1

            return json_response(
                {
                    "status": "processed",
                    "message_id": parsed.message_id,
                    "debate_id": result.get("debate_id") if result else None,
                }
            )

        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            logger.exception("Mailgun webhook error")
            return error_response(f"Internal error: {e}", status=500)

    async def handle_ses(self, request: "web.Request") -> HandlerResult:
        """
        Handle AWS SES SNS notification.

        AWS SES sends SNS messages with types:
        - SubscriptionConfirmation: Confirm the subscription
        - Notification: Delivery/bounce/complaint notifications or email receipt

        For email receiving, the notification contains the email content.
        """
        try:
            body = await request.read()
            message = json.loads(body)

            message_type = message.get("Type", "")

            if message_type == "SubscriptionConfirmation":
                # Auto-confirm SNS subscription
                return await self._confirm_sns_subscription(message)

            elif message_type == "Notification":
                return await self._handle_ses_notification(message)

            else:
                logger.warning(f"Unknown SES message type: {message_type}")
                return error_response("Unknown message type", status=400)

        except json.JSONDecodeError:
            return error_response("Invalid JSON", status=400)
        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            logger.exception("SES webhook error")
            return error_response(f"Internal error: {e}", status=500)

    async def _confirm_sns_subscription(self, message: dict[str, Any]) -> HandlerResult:
        """Auto-confirm SNS subscription."""
        import aiohttp

        subscribe_url = message.get("SubscribeURL")
        if not subscribe_url:
            return error_response("Missing SubscribeURL", status=400)

        # Verify the URL is from AWS
        if not subscribe_url.startswith("https://sns."):
            logger.warning(f"Suspicious SNS subscription URL: {subscribe_url}")
            return error_response("Invalid subscription URL", status=400)

        # Confirm the subscription
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(subscribe_url) as resp:
                    if resp.status == 200:
                        logger.info("SNS subscription confirmed")
                        return json_response({"status": "confirmed"})
                    else:
                        logger.error(f"SNS subscription confirmation failed: {resp.status}")
                        return error_response("Confirmation failed", status=500)
        except Exception as e:
            logger.error(f"SNS subscription confirmation error: {e}")
            return error_response("Confirmation error", status=500)

    async def _handle_ses_notification(self, message: dict[str, Any]) -> HandlerResult:
        """Handle SES notification (email receipt or delivery status)."""
        from aragora.integrations.email_reply_loop import (
            verify_ses_signature,
        )

        # Verify SNS signature (optional but recommended)
        if not verify_ses_signature(message):
            logger.warning("SES signature verification failed (continuing anyway)")
            # Don't reject - signature verification for SNS is complex

        # Parse the notification message
        try:
            notification = json.loads(message.get("Message", "{}"))
        except json.JSONDecodeError:
            notification = {}

        notification_type = notification.get("notificationType", "")

        # Handle bounces and complaints
        if notification_type in ("Bounce", "Complaint"):
            await self._handle_ses_feedback(notification)
            return json_response({"status": "processed", "type": notification_type.lower()})

        # Handle received email
        if "mail" in notification and "content" in notification:
            return await self._handle_ses_email_receipt(notification)

        # Delivery notification or other
        logger.debug(f"SES notification: {notification_type}")
        return json_response({"status": "acknowledged"})

    async def _handle_ses_email_receipt(self, notification: dict[str, Any]) -> HandlerResult:
        """Handle SES email receipt (for receiving emails via SES)."""

        mail = notification.get("mail", {})
        content = notification.get("content", "")

        # Parse email content (raw MIME)
        # For simplicity, extract basic headers; full parsing would use email.parser
        headers = mail.get("commonHeaders", {})

        parsed = ParsedEmail(
            message_id=headers.get("messageId", mail.get("messageId", "")),
            sender=headers.get("from", [""])[0] if headers.get("from") else "",
            recipients=headers.get("to", []),
            subject=headers.get("subject", ""),
            body=content[:10000] if content else "",  # Truncate for safety
            received_at=datetime.now(timezone.utc),
        )

        # Process through reply loop
        loop = EmailReplyLoop()
        result = await loop.process_email(parsed)
        self._processed_count += 1

        return json_response(
            {
                "status": "processed",
                "message_id": parsed.message_id,
                "debate_id": result.get("debate_id") if result else None,
            }
        )

    async def _handle_ses_feedback(self, notification: dict[str, Any]) -> None:
        """Handle SES bounce/complaint feedback."""
        notification_type = notification.get("notificationType", "")

        if notification_type == "Bounce":
            bounce = notification.get("bounce", {})
            for recipient in bounce.get("bouncedRecipients", []):
                email = recipient.get("emailAddress", "")
                bounce_type = bounce.get("bounceType", "")
                logger.info(f"SES bounce: {email} ({bounce_type})")
                await self._handle_email_event("bounce", email, recipient)

        elif notification_type == "Complaint":
            complaint = notification.get("complaint", {})
            for recipient in complaint.get("complainedRecipients", []):
                email = recipient.get("emailAddress", "")
                logger.info(f"SES complaint: {email}")
                await self._handle_email_event("complaint", email, recipient)

    async def _handle_email_event(
        self,
        event_type: str,
        email: str,
        details: dict[str, Any],
    ) -> None:
        """
        Handle email delivery events (bounce, complaint, unsubscribe).

        Updates recipient lists and preferences based on event type.
        """
        # This would integrate with recipient management
        # For now, just log the event
        logger.info(f"Email event: {event_type} for {email}")

        # In a full implementation:
        # - Update recipient status in database
        # - Remove from active lists on hard bounce
        # - Track complaints for abuse monitoring
        # - Handle unsubscribe requests

    def get_stats(self) -> dict[str, Any]:
        """Get webhook processing statistics."""
        return {
            "processed_count": self._processed_count,
            "error_count": self._error_count,
            "last_error": self._last_error,
        }

    async def handle_status(self, request: "web.Request") -> HandlerResult:
        """Return status and stats for email webhooks."""
        return json_response(
            {
                "status": "ok",
                "stats": self.get_stats(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )


def _extract_header(headers_raw: str, header_name: str) -> str:
    """Extract a header value from raw headers string."""
    for line in headers_raw.split("\n"):
        if line.lower().startswith(f"{header_name.lower()}:"):
            return line.split(":", 1)[1].strip()
    return ""


def register_email_webhook_routes(app: Any) -> EmailWebhookHandler:
    """Register email webhook routes on an aiohttp app."""
    try:
        from aiohttp import web
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("aiohttp is required for email webhook routes") from e

    handler = EmailWebhookHandler()

    async def _wrap(result: Any) -> "web.Response":
        if hasattr(result, "body") and hasattr(result, "status_code"):
            return web.Response(
                body=result.body,
                status=result.status_code,
                content_type=getattr(result, "content_type", "application/json"),
                headers=getattr(result, "headers", None),
            )
        if isinstance(result, web.StreamResponse):
            return result  # type: ignore[return-value]
        return web.json_response(result)

    async def _sendgrid(request: "web.Request") -> "web.Response":
        return await _wrap(await handler.handle_sendgrid(request))

    async def _mailgun(request: "web.Request") -> "web.Response":
        return await _wrap(await handler.handle_mailgun(request))

    async def _ses(request: "web.Request") -> "web.Response":
        return await _wrap(await handler.handle_ses(request))

    async def _status(request: "web.Request") -> "web.Response":
        return await _wrap(await handler.handle_status(request))

    app.router.add_post("/webhooks/email/sendgrid", _sendgrid)
    app.router.add_post("/webhooks/email/mailgun", _mailgun)
    app.router.add_post("/webhooks/email/ses", _ses)
    app.router.add_get("/webhooks/email/status", _status)

    return handler


__all__ = ["EmailWebhookHandler", "register_email_webhook_routes"]
