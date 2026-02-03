"""Debate result routing to originating chat channels.

Dispatches debate results, receipts, and error messages to the
appropriate platform sender based on the debate's origin.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .models import DebateOrigin
from .formatting import _format_receipt_summary, format_error_for_chat
from .senders import (
    _send_telegram_result,
    _send_telegram_voice,
    _send_whatsapp_result,
    _send_whatsapp_voice,
    _send_slack_result,
    _send_discord_result,
    _send_discord_voice,
    _send_teams_result,
    _send_email_result,
    _send_google_chat_result,
    _send_slack_receipt,
    _send_teams_receipt,
    _send_telegram_receipt,
    _send_discord_receipt,
    _send_google_chat_receipt,
    _send_slack_error,
    _send_teams_error,
    _send_telegram_error,
    _send_discord_error,
)
from .sessions import get_sessions_for_debate


def get_debate_origin(debate_id: str):
    """Wrapper for registry.get_debate_origin (supports test patching)."""
    from .registry import get_debate_origin as _get_debate_origin

    return _get_debate_origin(debate_id)


def mark_result_sent(debate_id: str) -> None:
    """Wrapper for registry.mark_result_sent (supports test patching)."""
    from .registry import mark_result_sent as _mark_result_sent

    _mark_result_sent(debate_id)


logger = logging.getLogger(__name__)

# Feature flag for dock-based routing (default: enabled)
# Set DEBATE_ORIGIN_USE_DOCKS=0 to disable
USE_DOCK_ROUTING = os.environ.get("DEBATE_ORIGIN_USE_DOCKS", "1").lower() not in (
    "0",
    "false",
    "no",
)


async def route_debate_result(
    debate_id: str,
    result: dict[str, Any],
    include_voice: bool = False,
    receipt: Any | None = None,
    receipt_url: str | None = None,
) -> bool:
    """Route a debate result back to its originating channel.

    Args:
        debate_id: Debate identifier
        result: Debate result dict with keys like:
            - consensus_reached: bool
            - final_answer: str
            - confidence: float
            - participants: list[str]
        include_voice: Whether to include TTS voice message (default: False)

    Returns:
        True if result was successfully routed, False otherwise
    """
    origin = get_debate_origin(debate_id)
    if not origin:
        logger.warning(f"No origin found for debate {debate_id}")
        return False

    event_type = result.get("event")
    aux_events = {
        "decision_integrity",
        "decision_plan",
        "execution_progress",
        "execution_complete",
    }
    is_aux_event = event_type in aux_events or "package" in result or "progress" in result

    if origin.result_sent and not is_aux_event:
        logger.debug(f"Result already sent for debate {debate_id}")
        return True

    platform = origin.platform.lower()
    logger.info(f"Routing result for {debate_id} to {platform}:{origin.channel_id}")

    # Use dock-based routing if enabled
    if USE_DOCK_ROUTING:
        try:
            from aragora.channels.router import get_channel_router

            router = get_channel_router()
            send_result = await router.route_result(
                platform=platform,
                channel_id=origin.channel_id,
                result=result,
                thread_id=origin.thread_id,
                message_id=origin.message_id,
                include_voice=include_voice,
                webhook_url=origin.metadata.get("webhook_url"),  # Teams
            )

            if send_result.success:
                if not is_aux_event:
                    mark_result_sent(debate_id)
                # Post receipt if provided
                if receipt and receipt_url:
                    try:
                        await post_receipt_to_channel(origin, receipt, receipt_url)
                    except (OSError, RuntimeError) as e:
                        logger.warning(f"Failed to post receipt for {debate_id}: {e}")
                return True
            else:
                logger.warning(f"Dock routing failed for {platform}: {send_result.error}")
                # Fall through to legacy routing
        except (ImportError, OSError, RuntimeError) as e:
            logger.warning(f"Dock routing error, falling back to legacy: {e}")
            # Fall through to legacy routing

    # Legacy platform-specific routing (fallback)
    try:
        # Route to appropriate platform
        if platform == "telegram":
            success = await _send_telegram_result(origin, result)
            if success and include_voice:
                await _send_telegram_voice(origin, result)
        elif platform == "whatsapp":
            success = await _send_whatsapp_result(origin, result)
            if success and include_voice:
                await _send_whatsapp_voice(origin, result)
        elif platform == "slack":
            success = await _send_slack_result(origin, result)
        elif platform == "discord":
            success = await _send_discord_result(origin, result)
            if success and include_voice:
                await _send_discord_voice(origin, result)
        elif platform == "teams":
            success = await _send_teams_result(origin, result)
        elif platform == "email":
            success = await _send_email_result(origin, result)
        elif platform in ("google_chat", "gchat"):
            success = await _send_google_chat_result(origin, result)
        else:
            logger.warning(f"Unknown platform: {platform}")
            return False

        if success:
            if not is_aux_event:
                mark_result_sent(debate_id)
            # Post receipt if provided
            if receipt and receipt_url:
                try:
                    await post_receipt_to_channel(origin, receipt, receipt_url)
                except (OSError, RuntimeError) as e:
                    logger.warning(f"Failed to post receipt for {debate_id}: {e}")

        return success

    except (OSError, RuntimeError, ValueError) as e:
        logger.error(f"Failed to route result for {debate_id}: {e}")
        return False


async def post_receipt_to_channel(
    origin: DebateOrigin,
    receipt: Any,
    receipt_url: str,
) -> bool:
    """Post receipt summary with link to originating channel.

    Args:
        origin: The debate origin containing platform/channel info
        receipt: DecisionReceipt object
        receipt_url: URL to view full receipt

    Returns:
        True if receipt was posted successfully
    """
    platform = origin.platform.lower()
    logger.info(f"Posting receipt to {platform}:{origin.channel_id}")

    # Use dock-based routing if enabled
    if USE_DOCK_ROUTING:
        try:
            from aragora.channels.router import get_channel_router

            router = get_channel_router()
            send_result = await router.route_receipt(
                platform=platform,
                channel_id=origin.channel_id,
                receipt=receipt,
                receipt_url=receipt_url,
                thread_id=origin.thread_id,
                webhook_url=origin.metadata.get("webhook_url"),  # Teams
            )

            if send_result.success:
                return True
            else:
                logger.warning(f"Dock receipt routing failed: {send_result.error}")
                # Fall through to legacy routing
        except (ImportError, OSError, RuntimeError) as e:
            logger.warning(f"Dock receipt routing error, falling back: {e}")
            # Fall through to legacy routing

    # Legacy routing (fallback)
    summary = _format_receipt_summary(receipt, receipt_url)

    try:
        if platform == "slack":
            return await _send_slack_receipt(origin, summary, receipt_url)
        elif platform == "teams":
            return await _send_teams_receipt(origin, summary, receipt_url)
        elif platform == "telegram":
            return await _send_telegram_receipt(origin, summary)
        elif platform == "discord":
            return await _send_discord_receipt(origin, summary)
        elif platform in ("google_chat", "gchat"):
            return await _send_google_chat_receipt(origin, summary)
        else:
            logger.debug(f"Receipt posting not supported for {platform}")
            return False
    except (OSError, RuntimeError, ValueError) as e:
        logger.error(f"Receipt post error for {platform}: {e}")
        return False


async def send_error_to_channel(
    origin: DebateOrigin,
    error: str,
    debate_id: str,
) -> bool:
    """Send a user-friendly error message to the originating channel.

    Args:
        origin: The debate origin with platform/channel info
        error: The technical error message
        debate_id: The debate ID for reference

    Returns:
        True if the message was sent successfully
    """
    platform = origin.platform.lower()
    logger.info(f"Sending error to {platform}:{origin.channel_id}")

    # Use dock-based routing if enabled
    if USE_DOCK_ROUTING:
        try:
            from aragora.channels.router import get_channel_router

            router = get_channel_router()
            send_result = await router.route_error(
                platform=platform,
                channel_id=origin.channel_id,
                error_message=error,
                debate_id=debate_id,
                thread_id=origin.thread_id,
                message_id=origin.message_id,
                webhook_url=origin.metadata.get("webhook_url"),  # Teams
            )

            if send_result.success:
                return True
            else:
                logger.warning(f"Dock error routing failed: {send_result.error}")
                # Fall through to legacy routing
        except (ImportError, OSError, RuntimeError) as e:
            logger.warning(f"Dock error routing error, falling back: {e}")
            # Fall through to legacy routing

    # Legacy routing (fallback)
    friendly_message = format_error_for_chat(error, debate_id)

    try:
        if platform == "slack":
            return await _send_slack_error(origin, friendly_message)
        elif platform == "teams":
            return await _send_teams_error(origin, friendly_message)
        elif platform == "telegram":
            return await _send_telegram_error(origin, friendly_message)
        elif platform == "discord":
            return await _send_discord_error(origin, friendly_message)
        else:
            logger.debug(f"Error notification not supported for {platform}")
            return False
    except (OSError, RuntimeError, ValueError) as e:
        logger.error(f"Failed to send error to {platform}: {e}")
        return False


async def route_result_to_all_sessions(
    debate_id: str,
    result: dict[str, Any],
    include_voice: bool = False,
) -> int:
    """Route debate result to all sessions linked to the debate.

    This extends route_debate_result to support multi-channel scenarios
    where a user may have started a debate on one channel but wants
    results on multiple channels.

    Args:
        debate_id: The debate ID
        result: The debate result dict
        include_voice: Whether to include TTS voice message

    Returns:
        Number of channels successfully notified
    """
    success_count = 0

    # First, route via origin (primary channel)
    origin_success = await route_debate_result(debate_id, result, include_voice)
    if origin_success:
        success_count += 1

    # Then route to any additional sessions
    try:
        sessions = await get_sessions_for_debate(debate_id)
        origin = get_debate_origin(debate_id)
        origin_session_id = origin.session_id if origin else None

        for session in sessions:
            # Skip if this is the same session as the origin
            if session.session_id == origin_session_id:
                continue

            # Create a temporary origin for this session
            temp_origin = DebateOrigin(
                debate_id=debate_id,
                platform=session.channel,
                channel_id=session.context.get("channel_id", session.user_id),
                user_id=session.user_id,
                session_id=session.session_id,
                metadata=session.context,
            )

            # Route to this session's channel
            try:
                platform = temp_origin.platform.lower()
                if platform == "telegram":
                    success = await _send_telegram_result(temp_origin, result)
                elif platform == "slack":
                    success = await _send_slack_result(temp_origin, result)
                elif platform == "discord":
                    success = await _send_discord_result(temp_origin, result)
                elif platform == "teams":
                    success = await _send_teams_result(temp_origin, result)
                elif platform == "whatsapp":
                    success = await _send_whatsapp_result(temp_origin, result)
                else:
                    success = False

                if success:
                    success_count += 1
                    logger.info(f"Routed result to session {session.session_id[:8]}")
            except (OSError, RuntimeError) as e:
                logger.warning(f"Failed to route to session {session.session_id[:8]}: {e}")

    except (OSError, RuntimeError, KeyError, AttributeError) as e:
        logger.debug(f"Multi-session routing failed: {e}")

    return success_count


async def route_plan_result(
    debate_id: str,
    outcome: dict[str, Any],
) -> bool:
    """Route a decision plan outcome back to its originating channel.

    Args:
        debate_id: Original debate identifier (plans link to debates)
        outcome: Plan outcome dict with keys like:
            - plan_id: str
            - success: bool
            - tasks_completed: int
            - tasks_total: int
            - formatted_message: str (pre-formatted notification)

    Returns:
        True if outcome was successfully routed, False otherwise
    """
    origin = get_debate_origin(debate_id)
    if not origin:
        logger.warning(f"No origin found for debate {debate_id} (plan routing)")
        return False

    plan_id = outcome.get("plan_id", "unknown")
    platform = origin.platform.lower()
    logger.info(f"Routing plan {plan_id} outcome to {platform}:{origin.channel_id}")

    # Get pre-formatted message or format one
    message = outcome.get("formatted_message")
    if not message:
        success = outcome.get("success", False)
        tasks_completed = outcome.get("tasks_completed", 0)
        tasks_total = outcome.get("tasks_total", 0)
        message = f"**Decision Plan {'Completed' if success else 'Failed'}**\n"
        message += f"Plan: `{plan_id[:12]}...`\n"
        message += f"Tasks: {tasks_completed}/{tasks_total} completed"

    # Use dock-based routing if enabled
    if USE_DOCK_ROUTING:
        try:
            from aragora.channels.router import get_channel_router

            router = get_channel_router()
            send_result = await router.route_result(
                platform=platform,
                channel_id=origin.channel_id,
                result=outcome,
                thread_id=origin.thread_id,
                webhook_url=origin.metadata.get("webhook_url"),  # Teams
            )

            if send_result.success:
                logger.info(f"Plan outcome routed via dock to {platform}")
                return True
            else:
                logger.warning(f"Dock plan routing failed: {send_result.error}")
                # Fall through to legacy routing
        except (ImportError, OSError, RuntimeError) as e:
            logger.warning(f"Dock plan routing error, falling back: {e}")
            # Fall through to legacy routing

    # Legacy routing - use result senders with plan outcome as "result"
    try:
        # Create a result-like dict for existing senders
        result_like = {
            "consensus_reached": outcome.get("success", False),
            "final_answer": message,
            "confidence": 1.0 if outcome.get("success") else 0.0,
            "task": outcome.get("task", "Decision plan execution"),
        }

        if platform == "telegram":
            success = await _send_telegram_result(origin, result_like)
        elif platform == "slack":
            success = await _send_slack_result(origin, result_like)
        elif platform == "discord":
            success = await _send_discord_result(origin, result_like)
        elif platform == "teams":
            success = await _send_teams_result(origin, result_like)
        elif platform == "whatsapp":
            success = await _send_whatsapp_result(origin, result_like)
        elif platform == "email":
            success = await _send_email_result(origin, result_like)
        elif platform in ("google_chat", "gchat"):
            success = await _send_google_chat_result(origin, result_like)
        else:
            logger.warning(f"Unknown platform for plan routing: {platform}")
            return False

        return success

    except (OSError, RuntimeError, ValueError) as e:
        logger.error(f"Failed to route plan outcome for {plan_id}: {e}")
        return False
