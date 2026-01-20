"""
Debate Origin Tracking for Bidirectional Chat.

Tracks which chat channel/platform originated each debate so results
can be routed back to the user on the same channel.

Usage:
    from aragora.server.debate_origin import (
        register_debate_origin,
        get_debate_origin,
        route_debate_result,
    )

    # When starting a debate from Telegram
    register_debate_origin(
        debate_id="abc123",
        platform="telegram",
        channel_id="12345678",
        user_id="87654321",
        metadata={"username": "john_doe"},
    )

    # When debate completes, route result back
    await route_debate_result(debate_id, result)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# In-memory store with optional Redis backend
_origin_store: Dict[str, "DebateOrigin"] = {}

# TTL for origin records (24 hours)
ORIGIN_TTL_SECONDS = int(os.environ.get("DEBATE_ORIGIN_TTL", 86400))


@dataclass
class DebateOrigin:
    """Origin information for a debate."""

    debate_id: str
    platform: str  # telegram, whatsapp, slack, discord, teams, email, web
    channel_id: str  # Chat ID, channel ID, thread ID, etc.
    user_id: str  # User who initiated the debate
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional threading info
    thread_id: Optional[str] = None
    message_id: Optional[str] = None

    # Result routing
    result_sent: bool = False
    result_sent_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "debate_id": self.debate_id,
            "platform": self.platform,
            "channel_id": self.channel_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "thread_id": self.thread_id,
            "message_id": self.message_id,
            "result_sent": self.result_sent,
            "result_sent_at": self.result_sent_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DebateOrigin":
        return cls(
            debate_id=data["debate_id"],
            platform=data["platform"],
            channel_id=data["channel_id"],
            user_id=data["user_id"],
            created_at=data.get("created_at", time.time()),
            metadata=data.get("metadata", {}),
            thread_id=data.get("thread_id"),
            message_id=data.get("message_id"),
            result_sent=data.get("result_sent", False),
            result_sent_at=data.get("result_sent_at"),
        )


def register_debate_origin(
    debate_id: str,
    platform: str,
    channel_id: str,
    user_id: str,
    thread_id: Optional[str] = None,
    message_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DebateOrigin:
    """Register the origin of a debate for result routing.

    Args:
        debate_id: Unique debate identifier
        platform: Platform name (telegram, whatsapp, slack, discord, etc.)
        channel_id: Channel/chat ID on the platform
        user_id: User ID who initiated the debate
        thread_id: Optional thread ID for threaded conversations
        message_id: Optional message ID that started the debate
        metadata: Optional additional metadata (username, etc.)

    Returns:
        DebateOrigin instance
    """
    origin = DebateOrigin(
        debate_id=debate_id,
        platform=platform,
        channel_id=channel_id,
        user_id=user_id,
        thread_id=thread_id,
        message_id=message_id,
        metadata=metadata or {},
    )

    _origin_store[debate_id] = origin

    # Also try Redis if available
    try:
        _store_origin_redis(origin)
    except Exception as e:
        logger.debug(f"Redis origin storage not available: {e}")

    logger.info(f"Registered debate origin: {debate_id} from {platform}:{channel_id}")
    return origin


def get_debate_origin(debate_id: str) -> Optional[DebateOrigin]:
    """Get the origin of a debate.

    Args:
        debate_id: Debate identifier

    Returns:
        DebateOrigin if found, None otherwise
    """
    # Check in-memory first
    origin = _origin_store.get(debate_id)
    if origin:
        return origin

    # Try Redis
    try:
        origin = _load_origin_redis(debate_id)
        if origin:
            _origin_store[debate_id] = origin  # Cache locally
            return origin
    except Exception as e:
        logger.debug(f"Redis origin lookup not available: {e}")

    return None


def mark_result_sent(debate_id: str) -> None:
    """Mark that the result has been sent for a debate."""
    origin = get_debate_origin(debate_id)
    if origin:
        origin.result_sent = True
        origin.result_sent_at = time.time()

        # Update Redis if available
        try:
            _store_origin_redis(origin)
        except Exception:
            pass


async def route_debate_result(
    debate_id: str,
    result: Dict[str, Any],
) -> bool:
    """Route a debate result back to its originating channel.

    Args:
        debate_id: Debate identifier
        result: Debate result dict with keys like:
            - consensus_reached: bool
            - final_answer: str
            - confidence: float
            - participants: List[str]

    Returns:
        True if result was successfully routed, False otherwise
    """
    origin = get_debate_origin(debate_id)
    if not origin:
        logger.warning(f"No origin found for debate {debate_id}")
        return False

    if origin.result_sent:
        logger.debug(f"Result already sent for debate {debate_id}")
        return True

    platform = origin.platform.lower()
    logger.info(f"Routing result for {debate_id} to {platform}:{origin.channel_id}")

    try:
        # Route to appropriate platform
        if platform == "telegram":
            success = await _send_telegram_result(origin, result)
        elif platform == "whatsapp":
            success = await _send_whatsapp_result(origin, result)
        elif platform == "slack":
            success = await _send_slack_result(origin, result)
        elif platform == "discord":
            success = await _send_discord_result(origin, result)
        elif platform == "teams":
            success = await _send_teams_result(origin, result)
        elif platform == "email":
            success = await _send_email_result(origin, result)
        else:
            logger.warning(f"Unknown platform: {platform}")
            return False

        if success:
            mark_result_sent(debate_id)

        return success

    except Exception as e:
        logger.error(f"Failed to route result for {debate_id}: {e}")
        return False


# Platform-specific result senders


async def _send_telegram_result(origin: DebateOrigin, result: Dict[str, Any]) -> bool:
    """Send result to Telegram."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        logger.warning("TELEGRAM_BOT_TOKEN not configured")
        return False

    chat_id = origin.channel_id
    message = _format_result_message(result, origin)

    try:
        import httpx

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }

        # Reply to original message if we have it
        if origin.message_id:
            data["reply_to_message_id"] = origin.message_id

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=data)
            if response.is_success:
                logger.info(f"Telegram result sent to {chat_id}")
                return True
            else:
                logger.warning(f"Telegram send failed: {response.status_code}")
                return False

    except Exception as e:
        logger.error(f"Telegram result send error: {e}")
        return False


async def _send_whatsapp_result(origin: DebateOrigin, result: Dict[str, Any]) -> bool:
    """Send result to WhatsApp."""
    token = os.environ.get("WHATSAPP_ACCESS_TOKEN", "")
    phone_id = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")

    if not token or not phone_id:
        logger.warning("WhatsApp credentials not configured")
        return False

    to_number = origin.channel_id
    message = _format_result_message(result, origin, markdown=False)

    try:
        import httpx

        url = f"https://graph.facebook.com/v18.0/{phone_id}/messages"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        data = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to_number,
            "type": "text",
            "text": {"preview_url": False, "body": message},
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=data, headers=headers)
            if response.is_success:
                logger.info(f"WhatsApp result sent to {to_number}")
                return True
            else:
                logger.warning(f"WhatsApp send failed: {response.status_code}")
                return False

    except Exception as e:
        logger.error(f"WhatsApp result send error: {e}")
        return False


async def _send_slack_result(origin: DebateOrigin, result: Dict[str, Any]) -> bool:
    """Send result to Slack."""
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        logger.warning("SLACK_BOT_TOKEN not configured")
        return False

    channel = origin.channel_id
    message = _format_result_message(result, origin, markdown=True)

    try:
        import httpx

        url = "https://slack.com/api/chat.postMessage"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        data = {
            "channel": channel,
            "text": message,
            "mrkdwn": True,
        }

        # Reply in thread if we have thread_ts
        if origin.thread_id:
            data["thread_ts"] = origin.thread_id

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=data, headers=headers)
            if response.is_success:
                resp_data = response.json()
                if resp_data.get("ok"):
                    logger.info(f"Slack result sent to {channel}")
                    return True
                else:
                    logger.warning(f"Slack API error: {resp_data.get('error')}")
                    return False
            else:
                logger.warning(f"Slack send failed: {response.status_code}")
                return False

    except Exception as e:
        logger.error(f"Slack result send error: {e}")
        return False


async def _send_discord_result(origin: DebateOrigin, result: Dict[str, Any]) -> bool:
    """Send result to Discord."""
    token = os.environ.get("DISCORD_BOT_TOKEN", "")
    if not token:
        logger.warning("DISCORD_BOT_TOKEN not configured")
        return False

    channel_id = origin.channel_id
    message = _format_result_message(result, origin, markdown=True)

    try:
        import httpx

        url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
        headers = {
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
        }
        data = {"content": message}

        # Reply to original message if we have it
        if origin.message_id:
            data["message_reference"] = {"message_id": origin.message_id}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=data, headers=headers)
            if response.is_success:
                logger.info(f"Discord result sent to {channel_id}")
                return True
            else:
                logger.warning(f"Discord send failed: {response.status_code}")
                return False

    except Exception as e:
        logger.error(f"Discord result send error: {e}")
        return False


async def _send_teams_result(origin: DebateOrigin, result: Dict[str, Any]) -> bool:
    """Send result to Microsoft Teams."""
    # Teams uses webhook URLs stored in metadata
    webhook_url = origin.metadata.get("webhook_url")
    if not webhook_url:
        logger.warning("Teams webhook URL not in origin metadata")
        return False

    message = _format_result_message(result, origin, markdown=False)

    try:
        import httpx

        # Teams Adaptive Card format
        card = {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": {
                        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                        "type": "AdaptiveCard",
                        "version": "1.4",
                        "body": [
                            {
                                "type": "TextBlock",
                                "text": "Aragora Debate Complete",
                                "weight": "Bolder",
                                "size": "Large",
                            },
                            {"type": "TextBlock", "text": message, "wrap": True},
                        ],
                    },
                }
            ],
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(webhook_url, json=card)
            if response.is_success:
                logger.info(f"Teams result sent via webhook")
                return True
            else:
                logger.warning(f"Teams send failed: {response.status_code}")
                return False

    except Exception as e:
        logger.error(f"Teams result send error: {e}")
        return False


async def _send_email_result(origin: DebateOrigin, result: Dict[str, Any]) -> bool:
    """Send result via email."""
    # Use existing email notification system
    try:
        from aragora.server.handlers.social.notifications import send_email_notification

        email = origin.metadata.get("email")
        if not email:
            email = origin.channel_id  # channel_id is email for email platform

        subject = "Aragora Debate Complete"
        message = _format_result_message(result, origin, markdown=False, html=True)

        # Fire-and-forget email
        await send_email_notification(
            to_email=email,
            subject=subject,
            body=message,
        )
        logger.info(f"Email result sent to {email}")
        return True

    except ImportError:
        logger.warning("Email notification system not available")
        return False
    except Exception as e:
        logger.error(f"Email result send error: {e}")
        return False


def _format_result_message(
    result: Dict[str, Any],
    origin: DebateOrigin,
    markdown: bool = True,
    html: bool = False,
) -> str:
    """Format debate result as a message."""
    consensus = result.get("consensus_reached", False)
    answer = result.get("final_answer", "No conclusion reached.")
    confidence = result.get("confidence", 0)
    participants = result.get("participants", [])
    topic = result.get("task", origin.metadata.get("topic", "Unknown topic"))

    # Truncate long answers
    if len(answer) > 800:
        answer = answer[:800] + "..."

    if html:
        return f"""
<h2>Debate Complete!</h2>
<p><strong>Topic:</strong> {topic[:200]}</p>
<p><strong>Consensus:</strong> {'Yes' if consensus else 'No'}</p>
<p><strong>Confidence:</strong> {confidence:.0%}</p>
<p><strong>Agents:</strong> {', '.join(participants[:5])}</p>
<hr>
<p><strong>Conclusion:</strong></p>
<p>{answer}</p>
"""

    if markdown:
        return f"""**Debate Complete!**

**Topic:** {topic[:200]}

**Consensus:** {'Yes' if consensus else 'No'}
**Confidence:** {confidence:.0%}
**Agents:** {', '.join(participants[:5])}

---

**Conclusion:**
{answer}
"""

    # Plain text
    return f"""Debate Complete!

Topic: {topic[:200]}

Consensus: {'Yes' if consensus else 'No'}
Confidence: {confidence:.0%}
Agents: {', '.join(participants[:5])}

---

Conclusion:
{answer}
"""


# Redis backend functions


def _store_origin_redis(origin: DebateOrigin) -> None:
    """Store origin in Redis."""
    import json

    try:
        import redis

        r = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
        key = f"debate_origin:{origin.debate_id}"
        r.setex(key, ORIGIN_TTL_SECONDS, json.dumps(origin.to_dict()))
    except ImportError:
        raise
    except Exception as e:
        logger.debug(f"Redis store failed: {e}")
        raise


def _load_origin_redis(debate_id: str) -> Optional[DebateOrigin]:
    """Load origin from Redis."""
    import json

    try:
        import redis

        r = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
        key = f"debate_origin:{debate_id}"
        data = r.get(key)
        if data:
            return DebateOrigin.from_dict(json.loads(data))
        return None
    except ImportError:
        raise
    except Exception as e:
        logger.debug(f"Redis load failed: {e}")
        raise


# Cleanup function for TTL


def cleanup_expired_origins() -> int:
    """Remove expired origin records from in-memory store."""
    now = time.time()
    expired = [
        k for k, v in _origin_store.items() if now - v.created_at > ORIGIN_TTL_SECONDS
    ]

    for k in expired:
        del _origin_store[k]

    if expired:
        logger.info(f"Cleaned up {len(expired)} expired debate origins")

    return len(expired)


__all__ = [
    "DebateOrigin",
    "register_debate_origin",
    "get_debate_origin",
    "mark_result_sent",
    "route_debate_result",
    "cleanup_expired_origins",
]
