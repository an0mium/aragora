"""Microsoft Teams sender for debate origin result routing.

Sends debate results, receipts, and error messages to Microsoft Teams
conversations. Supports two delivery strategies:

1. **Bot Framework proactive messaging** (preferred): Uses stored conversation
   references and the TeamsBot.send_proactive_message() method to deliver
   rich Adaptive Cards through the bot's identity. This is the recommended
   approach as it supports full card interactivity (voting, view details, etc.).

2. **Webhook fallback**: Posts to an incoming webhook URL stored in the origin
   metadata. Used when the Bot Framework connector is not available or when
   the conversation reference has not been stored.

Result messages use the ``create_consensus_card`` template from
``aragora.server.handlers.bots.teams_cards`` for a rich, interactive
presentation including confidence scores, agent breakdown, key points,
and action buttons.
"""

from __future__ import annotations

import logging
from typing import Any

from ..models import DebateOrigin

logger = logging.getLogger(__name__)


def _build_result_card(result: dict[str, Any], origin: DebateOrigin) -> dict[str, Any]:
    """Build a rich Adaptive Card for a debate result.

    Tries to use the consensus card template from teams_cards for a polished
    presentation. Falls back to a simpler card if the template module is not
    available.

    Args:
        result: Debate result dict with consensus info, participants, etc.
        origin: Debate origin with metadata.

    Returns:
        Adaptive Card dict ready for Bot Framework or webhook delivery.
    """
    debate_id = origin.debate_id
    topic = result.get("task", origin.metadata.get("topic", "Unknown topic"))
    consensus = result.get("consensus_reached", False)
    confidence = result.get("confidence", 0.0)
    final_answer = result.get("final_answer", "No conclusion reached.")
    participants = result.get("participants", [])

    # Try using the rich consensus card template
    try:
        from aragora.server.handlers.bots.teams_cards import create_consensus_card

        # Split participants into supporting/dissenting based on result
        supporting = participants[:5] if participants else ["Unknown"]
        dissenting = result.get("dissenting_agents", [])

        # Extract key points if available
        key_points = result.get("key_points", [])
        if not key_points and final_answer:
            # Try to extract bullet points from the answer
            lines = final_answer.split("\n")
            key_points = [
                line.lstrip("- *").strip()
                for line in lines
                if line.strip().startswith(("-", "*")) and len(line.strip()) > 5
            ][:5]

        consensus_type = result.get("consensus_type", "")
        if not consensus_type:
            consensus_type = "unanimous" if consensus else "no_consensus"

        card = create_consensus_card(
            debate_id=debate_id,
            topic=topic[:200],
            consensus_type=consensus_type,
            final_answer=final_answer[:500],
            confidence=confidence,
            supporting_agents=supporting,
            dissenting_agents=dissenting if dissenting else None,
            key_points=key_points if key_points else None,
            vote_summary=result.get("vote_summary"),
        )
        return card

    except ImportError:
        pass

    # Fallback: build a simple but complete card
    body: list[dict[str, Any]] = [
        {
            "type": "TextBlock",
            "text": "Consensus Reached" if consensus else "Debate Complete",
            "weight": "Bolder",
            "size": "Large",
            "color": "Good" if consensus else "Warning",
        },
        {
            "type": "TextBlock",
            "text": topic[:200],
            "isSubtle": True,
            "wrap": True,
        },
    ]

    # Facts section
    facts = [
        {"title": "Confidence", "value": f"{confidence:.0%}"},
    ]
    if participants:
        facts.append({"title": "Agents", "value": ", ".join(participants[:5])})
    body.append({"type": "FactSet", "facts": facts})

    # Decision text
    if final_answer:
        preview = final_answer[:500]
        if len(final_answer) > 500:
            preview += "..."
        body.append(
            {
                "type": "Container",
                "separator": True,
                "spacing": "Medium",
                "items": [
                    {
                        "type": "TextBlock",
                        "text": "Decision",
                        "weight": "Bolder",
                        "size": "Medium",
                    },
                    {"type": "TextBlock", "text": preview, "wrap": True},
                ],
            }
        )

    # Actions
    actions: list[dict[str, Any]] = [
        {
            "type": "Action.OpenUrl",
            "title": "View Full Report",
            "url": f"https://aragora.ai/debate/{debate_id}",
        },
        {
            "type": "Action.Submit",
            "title": "Share Result",
            "data": {"action": "share", "debate_id": debate_id},
        },
    ]

    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": body,
        "actions": actions,
    }


def _wrap_card_as_message(card: dict[str, Any]) -> dict[str, Any]:
    """Wrap an Adaptive Card dict into a Bot Framework message activity.

    Args:
        card: The Adaptive Card dict.

    Returns:
        Message activity dict suitable for webhook POST.
    """
    return {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": card,
            }
        ],
    }


async def _send_via_proactive(
    origin: DebateOrigin,
    text: str | None = None,
    card: dict[str, Any] | None = None,
    fallback_text: str = "",
) -> bool | None:
    """Try to send a message via Bot Framework proactive messaging.

    Uses the TeamsBot.send_proactive_message() which sends through the
    Bot Framework service URL using stored conversation references.

    Args:
        origin: Debate origin with channel_id as conversation_id.
        text: Plain text to send (if no card).
        card: Adaptive Card dict to send.
        fallback_text: Fallback text for card accessibility.

    Returns:
        True if sent successfully, False if send failed, None if proactive
        messaging is not available (caller should fall back to webhook).
    """
    try:
        from aragora.server.handlers.bots.teams import TeamsBot, get_conversation_reference

        # Check if we have a conversation reference for this conversation
        ref = get_conversation_reference(origin.channel_id)
        if not ref:
            logger.debug(
                f"No conversation reference for {origin.channel_id}, falling back to webhook"
            )
            return None

        bot = TeamsBot()
        success = await bot.send_proactive_message(
            conversation_id=origin.channel_id,
            text=text,
            card=card,
            fallback_text=fallback_text,
        )
        return success

    except ImportError:
        logger.debug("TeamsBot not available for proactive messaging")
        return None
    except (RuntimeError, OSError, ValueError, AttributeError) as e:
        logger.warning(f"Proactive messaging failed: {e}")
        return None


async def _send_via_webhook(
    webhook_url: str, payload: dict[str, Any], context: str = "message"
) -> bool:
    """Send a message via incoming webhook URL.

    Args:
        webhook_url: Teams incoming webhook URL.
        payload: Message payload (typically a wrapped Adaptive Card).
        context: Description for logging (e.g., "result", "receipt", "error").

    Returns:
        True if sent successfully.
    """
    try:
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(webhook_url, json=payload)
            if response.is_success:
                logger.info(f"Teams {context} sent via webhook")
                return True
            else:
                logger.warning(f"Teams {context} webhook failed: {response.status_code}")
                return False

    except ImportError:
        logger.error("httpx not available for Teams webhook delivery")
        return False
    except (Exception,) as e:
        # Broad catch here because httpx.HTTPError is conditionally imported
        logger.error(f"Teams {context} send error: {e}")
        return False


async def _send_teams_result(origin: DebateOrigin, result: dict[str, Any]) -> bool:
    """Send debate result to Microsoft Teams.

    Tries Bot Framework proactive messaging first (rich Adaptive Card with
    interactivity), then falls back to webhook delivery.

    Args:
        origin: Debate origin with Teams conversation info.
        result: Debate result dict.

    Returns:
        True if the result was delivered successfully.
    """
    # Build a rich result card
    card = _build_result_card(result, origin)
    topic = result.get("task", origin.metadata.get("topic", ""))
    fallback_text = f"Debate complete: {topic[:100]}"

    # Strategy 1: Proactive messaging via Bot Framework
    proactive_result = await _send_via_proactive(origin, card=card, fallback_text=fallback_text)
    if proactive_result is True:
        logger.info(f"Teams result sent via proactive messaging for debate {origin.debate_id}")
        return True
    elif proactive_result is False:
        logger.warning("Proactive messaging returned False, trying webhook fallback")

    # Strategy 2: Webhook fallback
    webhook_url = origin.metadata.get("webhook_url")
    if not webhook_url:
        logger.warning("Teams webhook URL not in origin metadata and proactive unavailable")
        return False

    payload = _wrap_card_as_message(card)
    return await _send_via_webhook(webhook_url, payload, "result")


async def _send_teams_receipt(origin: DebateOrigin, summary: str, receipt_url: str) -> bool:
    """Post receipt to Teams with link button.

    Args:
        origin: Debate origin with Teams conversation info.
        summary: Receipt summary text.
        receipt_url: URL to view the full receipt.

    Returns:
        True if the receipt was delivered successfully.
    """
    card = {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": [
            {
                "type": "TextBlock",
                "text": "Decision Receipt",
                "weight": "Bolder",
                "size": "Large",
            },
            {"type": "TextBlock", "text": summary, "wrap": True},
        ],
        "actions": [
            {
                "type": "Action.OpenUrl",
                "title": "View Full Receipt",
                "url": receipt_url,
            }
        ],
    }

    # Try proactive first
    proactive_result = await _send_via_proactive(
        origin, card=card, fallback_text="Decision Receipt"
    )
    if proactive_result is True:
        logger.info(f"Teams receipt sent via proactive messaging for debate {origin.debate_id}")
        return True

    # Webhook fallback
    webhook_url = origin.metadata.get("webhook_url")
    if not webhook_url:
        return False

    payload = _wrap_card_as_message(card)
    return await _send_via_webhook(webhook_url, payload, "receipt")


async def _send_teams_error(origin: DebateOrigin, message: str) -> bool:
    """Send error message to Teams.

    Uses the error card template if available for a consistent error
    presentation with optional retry action.

    Args:
        origin: Debate origin with Teams conversation info.
        message: Error message to display.

    Returns:
        True if the error message was delivered successfully.
    """
    # Try using the error card template
    try:
        from aragora.server.handlers.bots.teams_cards import create_error_card

        card = create_error_card(
            title="Aragora Notice",
            message=message,
            retry_action={
                "action": "start_debate_prompt",
            },
        )
    except ImportError:
        card = {
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": [
                {
                    "type": "TextBlock",
                    "text": "Aragora Notice",
                    "weight": "Bolder",
                    "color": "Attention",
                },
                {"type": "TextBlock", "text": message, "wrap": True},
            ],
        }

    # Try proactive first
    proactive_result = await _send_via_proactive(
        origin, card=card, fallback_text=f"Aragora Notice: {message[:100]}"
    )
    if proactive_result is True:
        return True

    # Webhook fallback
    webhook_url = origin.metadata.get("webhook_url")
    if not webhook_url:
        return False

    payload = _wrap_card_as_message(card)
    return await _send_via_webhook(webhook_url, payload, "error")
