"""
Chat Platform Webhook Event Emission.

Emits events to the unified webhook system when chat-related activities occur.
This allows external systems to subscribe to Telegram/WhatsApp events.

Event Types:
- chat.message_received - New message from user
- chat.command_received - Bot command executed
- chat.debate_started - Debate initiated via chat
- chat.debate_completed - Debate finished
- chat.gauntlet_started - Gauntlet initiated via chat
- chat.gauntlet_completed - Gauntlet finished
- chat.vote_received - User vote on debate result
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _dispatch_chat_event(event_type: str, data: Dict[str, Any]) -> None:
    """
    Dispatch a chat event to the webhook system.

    Args:
        event_type: Event type (e.g., "chat.debate_started")
        data: Event data
    """
    try:
        from aragora.events import dispatch_event

        dispatch_event(event_type, data)
        logger.debug(f"Dispatched chat event: {event_type}")
    except ImportError:
        logger.debug("Event dispatcher not available")
    except Exception as e:
        logger.warning(f"Failed to dispatch chat event {event_type}: {e}")


def emit_message_received(
    platform: str,
    chat_id: str,
    user_id: str,
    username: str,
    message_text: str,
    message_type: str = "text",
) -> None:
    """
    Emit event when a message is received from a chat platform.

    Args:
        platform: Chat platform (telegram, whatsapp)
        chat_id: Chat/conversation ID
        user_id: User ID on the platform
        username: Username or display name
        message_text: Message content (truncated for privacy)
        message_type: Type of message (text, command, callback)
    """
    _dispatch_chat_event(
        "chat.message_received",
        {
            "platform": platform,
            "chat_id": str(chat_id),
            "user_id": str(user_id),
            "username": username,
            "message_type": message_type,
            "message_preview": message_text[:100] if message_text else "",
            "timestamp": time.time(),
        },
    )


def emit_command_received(
    platform: str,
    chat_id: str,
    user_id: str,
    username: str,
    command: str,
    args: str = "",
) -> None:
    """
    Emit event when a bot command is received.

    Args:
        platform: Chat platform
        chat_id: Chat/conversation ID
        user_id: User ID on the platform
        username: Username or display name
        command: Command name (without /)
        args: Command arguments
    """
    _dispatch_chat_event(
        "chat.command_received",
        {
            "platform": platform,
            "chat_id": str(chat_id),
            "user_id": str(user_id),
            "username": username,
            "command": command,
            "args_preview": args[:100] if args else "",
            "timestamp": time.time(),
        },
    )


def emit_debate_started(
    platform: str,
    chat_id: str,
    user_id: str,
    username: str,
    topic: str,
    debate_id: Optional[str] = None,
) -> None:
    """
    Emit event when a debate is started via chat.

    Args:
        platform: Chat platform
        chat_id: Chat/conversation ID
        user_id: User ID on the platform
        username: Username or display name
        topic: Debate topic
        debate_id: Optional debate ID if registered
    """
    _dispatch_chat_event(
        "chat.debate_started",
        {
            "platform": platform,
            "chat_id": str(chat_id),
            "user_id": str(user_id),
            "username": username,
            "topic": topic[:200],
            "debate_id": debate_id,
            "timestamp": time.time(),
        },
    )


def emit_debate_completed(
    platform: str,
    chat_id: str,
    debate_id: str,
    topic: str,
    consensus_reached: bool,
    confidence: float,
    rounds_used: int,
    final_answer: Optional[str] = None,
) -> None:
    """
    Emit event when a debate completes.

    Args:
        platform: Chat platform
        chat_id: Chat/conversation ID
        debate_id: Debate ID
        topic: Debate topic
        consensus_reached: Whether consensus was achieved
        confidence: Confidence score
        rounds_used: Number of rounds
        final_answer: Final conclusion (truncated)
    """
    _dispatch_chat_event(
        "chat.debate_completed",
        {
            "platform": platform,
            "chat_id": str(chat_id),
            "debate_id": debate_id,
            "topic": topic[:200],
            "consensus_reached": consensus_reached,
            "confidence": confidence,
            "rounds_used": rounds_used,
            "final_answer_preview": final_answer[:300] if final_answer else None,
            "timestamp": time.time(),
        },
    )


def emit_gauntlet_started(
    platform: str,
    chat_id: str,
    user_id: str,
    username: str,
    statement: str,
    gauntlet_id: Optional[str] = None,
) -> None:
    """
    Emit event when a gauntlet is started via chat.

    Args:
        platform: Chat platform
        chat_id: Chat/conversation ID
        user_id: User ID on the platform
        username: Username or display name
        statement: Statement being tested
        gauntlet_id: Optional gauntlet ID
    """
    _dispatch_chat_event(
        "chat.gauntlet_started",
        {
            "platform": platform,
            "chat_id": str(chat_id),
            "user_id": str(user_id),
            "username": username,
            "statement": statement[:200],
            "gauntlet_id": gauntlet_id,
            "timestamp": time.time(),
        },
    )


def emit_gauntlet_completed(
    platform: str,
    chat_id: str,
    gauntlet_id: str,
    statement: str,
    verdict: str,
    confidence: float,
    challenges_passed: int,
    challenges_total: int,
) -> None:
    """
    Emit event when a gauntlet completes.

    Args:
        platform: Chat platform
        chat_id: Chat/conversation ID
        gauntlet_id: Gauntlet ID
        statement: Statement tested
        verdict: Final verdict
        confidence: Confidence score
        challenges_passed: Number of challenges passed
        challenges_total: Total number of challenges
    """
    _dispatch_chat_event(
        "chat.gauntlet_completed",
        {
            "platform": platform,
            "chat_id": str(chat_id),
            "gauntlet_id": gauntlet_id,
            "statement": statement[:200],
            "verdict": verdict,
            "confidence": confidence,
            "challenges_passed": challenges_passed,
            "challenges_total": challenges_total,
            "timestamp": time.time(),
        },
    )


def emit_vote_received(
    platform: str,
    chat_id: str,
    user_id: str,
    username: str,
    debate_id: str,
    vote: str,
) -> None:
    """
    Emit event when a user votes on a debate result.

    Args:
        platform: Chat platform
        chat_id: Chat/conversation ID
        user_id: User ID on the platform
        username: Username or display name
        debate_id: ID of the debate being voted on
        vote: Vote value (agree, disagree)
    """
    _dispatch_chat_event(
        "chat.vote_received",
        {
            "platform": platform,
            "chat_id": str(chat_id),
            "user_id": str(user_id),
            "username": username,
            "debate_id": debate_id,
            "vote": vote,
            "timestamp": time.time(),
        },
    )


__all__ = [
    "emit_message_received",
    "emit_command_received",
    "emit_debate_started",
    "emit_debate_completed",
    "emit_gauntlet_started",
    "emit_gauntlet_completed",
    "emit_vote_received",
]
