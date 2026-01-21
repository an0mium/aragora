"""
Email Reply Loop for Bidirectional Email Communication.

Enables users to reply to debate emails and have their input routed
back to the appropriate debate. Works with:
- SendGrid Inbound Parse Webhook
- AWS SES with SNS notifications
- Direct IMAP polling (fallback)

Architecture:
    1. User receives debate notification email
    2. User replies to email
    3. Reply is routed to webhook endpoint
    4. Webhook extracts debate_id from email headers/subject
    5. Reply content is added to debate as user input
    6. Result router sends updated debate results back

Usage:
    # At server startup
    from aragora.integrations.email_reply_loop import setup_email_reply_loop
    setup_email_reply_loop()

    # Or manually process an inbound email
    from aragora.integrations.email_reply_loop import process_inbound_email
    await process_inbound_email(email_data)
"""

from __future__ import annotations

import asyncio
import email
import hashlib
import hmac
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from email import policy
from email.parser import BytesParser, Parser
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Configuration from environment
EMAIL_INBOUND_SECRET = os.environ.get("EMAIL_INBOUND_SECRET", "")
SENDGRID_INBOUND_SECRET = os.environ.get("SENDGRID_INBOUND_SECRET", EMAIL_INBOUND_SECRET)
SES_NOTIFICATION_SECRET = os.environ.get("SES_NOTIFICATION_SECRET", EMAIL_INBOUND_SECRET)

# Email parsing patterns
DEBATE_ID_PATTERN = re.compile(r"debate[_-]?id[:=\s]*([a-zA-Z0-9_-]+)", re.IGNORECASE)
REPLY_MARKER_PATTERN = re.compile(r"^>.*$|^On .+ wrote:$", re.MULTILINE)
SIGNATURE_PATTERN = re.compile(r"^--\s*$.*", re.DOTALL | re.MULTILINE)


@dataclass
class InboundEmail:
    """Represents a parsed inbound email."""

    message_id: str
    from_email: str
    from_name: str = ""
    to_email: str = ""
    subject: str = ""
    body_plain: str = ""
    body_html: str = ""
    in_reply_to: str = ""
    references: List[str] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    received_at: datetime = field(default_factory=datetime.utcnow)
    raw_data: Optional[bytes] = None

    @property
    def debate_id(self) -> Optional[str]:
        """Extract debate ID from email headers or subject."""
        # Check X-Aragora-Debate-Id header
        if "X-Aragora-Debate-Id" in self.headers:
            return self.headers["X-Aragora-Debate-Id"]

        # Check In-Reply-To header for debate reference
        if self.in_reply_to:
            match = DEBATE_ID_PATTERN.search(self.in_reply_to)
            if match:
                return match.group(1)

        # Check References header
        for ref in self.references:
            match = DEBATE_ID_PATTERN.search(ref)
            if match:
                return match.group(1)

        # Check subject line
        if self.subject:
            match = DEBATE_ID_PATTERN.search(self.subject)
            if match:
                return match.group(1)

        return None

    @property
    def cleaned_body(self) -> str:
        """Get body with quoted text and signature removed."""
        body = self.body_plain or ""

        # Remove signature
        body = SIGNATURE_PATTERN.sub("", body).strip()

        # Remove quoted reply text
        lines = body.split("\n")
        cleaned_lines = []
        in_quote = False

        for line in lines:
            # Check if this is a quote marker
            if line.strip().startswith(">") or re.match(r"^On .+ wrote:$", line.strip()):
                in_quote = True
                continue

            if in_quote and line.strip() == "":
                # Empty line after quote - might be end of quoted section
                continue

            if not line.strip().startswith(">"):
                in_quote = False
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "from_email": self.from_email,
            "from_name": self.from_name,
            "to_email": self.to_email,
            "subject": self.subject,
            "body_plain": self.body_plain,
            "debate_id": self.debate_id,
            "cleaned_body": self.cleaned_body,
            "received_at": self.received_at.isoformat(),
        }


@dataclass
class EmailReplyOrigin:
    """Tracks an email that expects a reply."""

    debate_id: str
    message_id: str
    recipient_email: str
    recipient_name: str = ""
    sent_at: datetime = field(default_factory=datetime.utcnow)
    reply_received: bool = False
    reply_received_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "debate_id": self.debate_id,
            "message_id": self.message_id,
            "recipient_email": self.recipient_email,
            "recipient_name": self.recipient_name,
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "reply_received": self.reply_received,
            "reply_received_at": self.reply_received_at.isoformat() if self.reply_received_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmailReplyOrigin":
        """Create from dictionary."""
        sent_at = data.get("sent_at")
        reply_received_at = data.get("reply_received_at")
        return cls(
            debate_id=data["debate_id"],
            message_id=data["message_id"],
            recipient_email=data["recipient_email"],
            recipient_name=data.get("recipient_name", ""),
            sent_at=datetime.fromisoformat(sent_at) if sent_at else datetime.utcnow(),
            reply_received=data.get("reply_received", False),
            reply_received_at=datetime.fromisoformat(reply_received_at) if reply_received_at else None,
            metadata=data.get("metadata", {}),
        )


# TTL for email origins in Redis (24 hours)
EMAIL_ORIGIN_TTL_SECONDS = 86400

# In-memory store for tracking reply origins (with Redis persistence)
_reply_origins: Dict[str, EmailReplyOrigin] = {}
_reply_origins_lock = asyncio.Lock()


def _store_email_origin_redis(origin: EmailReplyOrigin) -> None:
    """Store email origin in Redis with TTL."""
    try:
        import redis

        r = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
        key = f"email_origin:{origin.message_id}"
        r.setex(key, EMAIL_ORIGIN_TTL_SECONDS, json.dumps(origin.to_dict()))
    except ImportError:
        raise
    except Exception as e:
        logger.debug(f"Redis email origin store failed: {e}")
        raise


def _load_email_origin_redis(message_id: str) -> Optional[EmailReplyOrigin]:
    """Load email origin from Redis."""
    try:
        import redis

        r = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
        key = f"email_origin:{message_id}"
        data = r.get(key)
        if data:
            return EmailReplyOrigin.from_dict(json.loads(data))
        return None
    except ImportError:
        raise
    except Exception as e:
        logger.debug(f"Redis email origin load failed: {e}")
        raise


def register_email_origin(
    debate_id: str,
    message_id: str,
    recipient_email: str,
    recipient_name: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> EmailReplyOrigin:
    """
    Register an email as expecting a reply.

    Call this after sending a debate notification email.

    Args:
        debate_id: The debate ID
        message_id: Email Message-ID header
        recipient_email: Recipient email address
        recipient_name: Recipient name
        metadata: Additional metadata

    Returns:
        EmailReplyOrigin tracking object
    """
    origin = EmailReplyOrigin(
        debate_id=debate_id,
        message_id=message_id,
        recipient_email=recipient_email,
        recipient_name=recipient_name,
        metadata=metadata or {},
    )
    _reply_origins[message_id] = origin

    # Persist to Redis for durability across restarts
    try:
        _store_email_origin_redis(origin)
    except Exception as e:
        logger.debug(f"Redis email origin storage not available: {e}")

    logger.debug(f"Registered email origin for debate {debate_id}: {message_id}")
    return origin


async def get_origin_by_reply(in_reply_to: str) -> Optional[EmailReplyOrigin]:
    """Get the origin for an email reply.

    Checks in-memory cache first, then falls back to Redis.
    """
    async with _reply_origins_lock:
        # Check in-memory first
        origin = _reply_origins.get(in_reply_to)
        if origin:
            return origin

        # Fall back to Redis
        try:
            origin = _load_email_origin_redis(in_reply_to)
            if origin:
                _reply_origins[in_reply_to] = origin  # Cache locally
                return origin
        except Exception as e:
            logger.debug(f"Redis email origin lookup not available: {e}")

        return None


def parse_raw_email(raw_data: bytes) -> InboundEmail:
    """
    Parse a raw email (RFC 2822 format).

    Args:
        raw_data: Raw email bytes

    Returns:
        InboundEmail object
    """
    msg = BytesParser(policy=policy.default).parsebytes(raw_data)

    # Extract headers
    headers = {}
    for key in msg.keys():
        headers[key] = str(msg[key])

    # Extract body
    body_plain = ""
    body_html = ""
    attachments = []

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))

            if "attachment" in content_disposition:
                attachments.append({
                    "filename": part.get_filename(),
                    "content_type": content_type,
                    "size": len(part.get_payload(decode=True) or b""),
                })
            elif content_type == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    body_plain = payload.decode("utf-8", errors="replace")
            elif content_type == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    body_html = payload.decode("utf-8", errors="replace")
    else:
        content_type = msg.get_content_type()
        payload = msg.get_payload(decode=True)
        if payload:
            if content_type == "text/html":
                body_html = payload.decode("utf-8", errors="replace")
            else:
                body_plain = payload.decode("utf-8", errors="replace")

    # Parse From header
    from_header = msg.get("From", "")
    from_email, from_name = _parse_email_address(from_header)

    # Parse references
    references = []
    ref_header = msg.get("References", "")
    if ref_header:
        references = [r.strip() for r in ref_header.split() if r.strip()]

    return InboundEmail(
        message_id=msg.get("Message-ID", f"unknown-{time.time()}"),
        from_email=from_email,
        from_name=from_name,
        to_email=msg.get("To", ""),
        subject=msg.get("Subject", ""),
        body_plain=body_plain,
        body_html=body_html,
        in_reply_to=msg.get("In-Reply-To", ""),
        references=references,
        headers=headers,
        attachments=attachments,
        raw_data=raw_data,
    )


def parse_sendgrid_webhook(data: Dict[str, Any]) -> InboundEmail:
    """
    Parse SendGrid Inbound Parse webhook data.

    SendGrid sends form data with email components.

    Args:
        data: Webhook POST data (form fields)

    Returns:
        InboundEmail object
    """
    headers = {}
    if "headers" in data:
        # Parse headers from string
        for line in data["headers"].split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip()] = value.strip()

    envelope = data.get("envelope", "{}")
    if isinstance(envelope, str):
        try:
            envelope = json.loads(envelope)
        except json.JSONDecodeError:
            envelope = {}

    from_email = data.get("from", envelope.get("from", ""))
    from_email, from_name = _parse_email_address(from_email)

    return InboundEmail(
        message_id=headers.get("Message-ID", f"sendgrid-{time.time()}"),
        from_email=from_email,
        from_name=from_name,
        to_email=data.get("to", ""),
        subject=data.get("subject", ""),
        body_plain=data.get("text", ""),
        body_html=data.get("html", ""),
        in_reply_to=headers.get("In-Reply-To", ""),
        references=headers.get("References", "").split() if headers.get("References") else [],
        headers=headers,
        attachments=[],  # Attachments require separate handling
    )


def parse_ses_notification(data: Dict[str, Any]) -> Optional[InboundEmail]:
    """
    Parse AWS SES SNS notification.

    SES sends notifications via SNS, which can contain:
    - Bounce/complaint notifications (type: Notification, notificationType: Bounce/Complaint)
    - Delivery notifications (notificationType: Delivery)
    - Email receipt (action: SNS)

    Args:
        data: SNS notification data

    Returns:
        InboundEmail object or None if not an email notification
    """
    # Check message type
    msg_type = data.get("Type", "")
    if msg_type == "SubscriptionConfirmation":
        logger.info("SES SNS subscription confirmation received")
        return None

    # Parse the Message field (it's JSON-encoded)
    message = data.get("Message", "{}")
    if isinstance(message, str):
        try:
            message = json.loads(message)
        except json.JSONDecodeError:
            logger.error("Failed to parse SES notification message")
            return None

    notification_type = message.get("notificationType", "")

    # Handle receipt notification
    if notification_type == "Received" or "mail" in message:
        mail = message.get("mail", {})
        content = message.get("content", "")

        if content:
            # Full email content available, parse it
            return parse_raw_email(content.encode("utf-8"))

        # Otherwise use mail headers
        headers = {}
        for header in mail.get("headers", []):
            headers[header["name"]] = header["value"]

        common_headers = mail.get("commonHeaders", {})

        return InboundEmail(
            message_id=mail.get("messageId", headers.get("Message-ID", "")),
            from_email=common_headers.get("from", [""])[0] if common_headers.get("from") else "",
            to_email=common_headers.get("to", [""])[0] if common_headers.get("to") else "",
            subject=common_headers.get("subject", ""),
            body_plain="",  # Content not included in basic notification
            headers=headers,
            in_reply_to=headers.get("In-Reply-To", ""),
        )

    return None


def _parse_email_address(address: str) -> tuple[str, str]:
    """Parse email address to (email, name)."""
    if not address:
        return "", ""

    # Format: "Name <email@example.com>" or "email@example.com"
    match = re.match(r'"?([^"<]+)"?\s*<([^>]+)>', address)
    if match:
        return match.group(2).strip(), match.group(1).strip()

    # Plain email
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', address)
    if email_match:
        return email_match.group(0), ""

    return address, ""


async def process_inbound_email(email_data: InboundEmail) -> bool:
    """
    Process an inbound email and route it to the appropriate debate.

    Args:
        email_data: Parsed inbound email

    Returns:
        True if email was successfully processed
    """
    debate_id = email_data.debate_id
    if not debate_id:
        logger.debug(f"No debate_id found in email from {email_data.from_email}")
        return False

    # Check if we have an origin for this reply
    origin = await get_origin_by_reply(email_data.in_reply_to)
    if origin and origin.debate_id != debate_id:
        # In-Reply-To doesn't match, use the origin's debate_id
        debate_id = origin.debate_id

    # Get the cleaned reply content
    content = email_data.cleaned_body
    if not content:
        logger.debug(f"Empty reply content from {email_data.from_email}")
        return False

    logger.info(
        f"Processing email reply for debate {debate_id} from {email_data.from_email}: "
        f"{len(content)} chars"
    )

    # Route the reply to the debate system
    try:
        from aragora.server.debate_origin import register_debate_origin

        # Register email as the reply origin
        register_debate_origin(
            debate_id=debate_id,
            platform="email",
            channel_id=email_data.from_email,
            user_id=email_data.from_email,
            metadata={
                "message_id": email_data.message_id,
                "subject": email_data.subject,
                "from_name": email_data.from_name,
                "is_reply": True,
            },
        )
    except ImportError:
        logger.debug("debate_origin not available")

    # Submit as user input to the debate
    try:
        from aragora.debate.event_bus import get_event_bus

        event_bus = get_event_bus()
        await event_bus.emit(
            "user_input",
            {
                "debate_id": debate_id,
                "source": "email",
                "user_id": email_data.from_email,
                "user_name": email_data.from_name or email_data.from_email,
                "content": content,
                "metadata": {
                    "message_id": email_data.message_id,
                    "subject": email_data.subject,
                },
            },
        )
        logger.info(f"Email reply routed to debate {debate_id}")
        return True

    except ImportError:
        logger.warning("Event bus not available for email reply routing")
    except Exception as e:
        logger.error(f"Failed to route email reply: {e}")

    # Alternative: Try queue-based submission
    try:
        from aragora.queue import create_user_input_job

        job = create_user_input_job(
            debate_id=debate_id,
            user_id=email_data.from_email,
            content=content,
            source="email",
        )

        # Fire and forget
        async def enqueue():
            from aragora.queue import create_redis_queue
            q = await create_redis_queue()
            await q.enqueue(job)

        asyncio.create_task(enqueue())
        return True

    except ImportError:
        logger.debug("Queue system not available for email routing")
    except Exception as e:
        logger.error(f"Failed to queue email input: {e}")

    return False


def verify_sendgrid_signature(payload: bytes, timestamp: str, signature: str) -> bool:
    """
    Verify SendGrid webhook signature.

    Args:
        payload: Raw request body
        timestamp: X-Twilio-Email-Event-Webhook-Timestamp header
        signature: X-Twilio-Email-Event-Webhook-Signature header

    Returns:
        True if signature is valid
    """
    if not SENDGRID_INBOUND_SECRET:
        logger.warning("SENDGRID_INBOUND_SECRET not configured, skipping verification")
        return True

    # SendGrid signature = SHA256(timestamp + payload + secret)
    to_sign = timestamp.encode() + payload
    expected = hmac.new(
        SENDGRID_INBOUND_SECRET.encode(),
        to_sign,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(signature, expected)


def verify_ses_signature(message: Dict[str, Any]) -> bool:
    """
    Verify AWS SNS message signature.

    AWS SNS messages are signed with the certificate URL provided in the message.
    For simplicity, this implementation validates message structure.

    In production, fetch the certificate and validate the signature.

    Args:
        message: SNS notification

    Returns:
        True if message appears valid
    """
    # Basic validation - check required fields
    required = ["Type", "MessageId", "TopicArn", "Timestamp"]
    for field_name in required:
        if field_name not in message:
            return False

    # Validate TopicArn format
    topic_arn = message.get("TopicArn", "")
    if not topic_arn.startswith("arn:aws:sns:"):
        return False

    return True


# Reply handlers registry
_reply_handlers: List[Callable[[InboundEmail], bool]] = []


def register_reply_handler(handler: Callable[[InboundEmail], bool]) -> None:
    """
    Register a custom handler for email replies.

    Handlers are called in order until one returns True.

    Args:
        handler: Callable that takes InboundEmail and returns bool
    """
    _reply_handlers.append(handler)


async def handle_email_reply(email_data: InboundEmail) -> bool:
    """
    Handle an email reply through registered handlers.

    Args:
        email_data: Parsed inbound email

    Returns:
        True if handled
    """
    # Try custom handlers first
    for handler in _reply_handlers:
        try:
            if handler(email_data):
                return True
        except Exception as e:
            logger.warning(f"Reply handler error: {e}")

    # Default processing
    return await process_inbound_email(email_data)


def setup_email_reply_loop() -> None:
    """
    Set up the email reply loop.

    Call this at server startup to enable email reply processing.
    This registers the necessary webhook handlers with the server.
    """
    logger.info("Email reply loop initialized")

    # Note: Actual webhook route registration happens in the handler module
    # This function initializes any required state


__all__ = [
    "InboundEmail",
    "EmailReplyOrigin",
    "parse_raw_email",
    "parse_sendgrid_webhook",
    "parse_ses_notification",
    "process_inbound_email",
    "verify_sendgrid_signature",
    "verify_ses_signature",
    "register_email_origin",
    "register_reply_handler",
    "handle_email_reply",
    "setup_email_reply_loop",
]
