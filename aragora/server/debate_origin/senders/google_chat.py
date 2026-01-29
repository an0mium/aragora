"""Google Chat sender for debate origin result routing."""

from __future__ import annotations

import logging
from typing import Any

from ..models import DebateOrigin

logger = logging.getLogger(__name__)


async def _send_google_chat_result(origin: DebateOrigin, result: dict[str, Any]) -> bool:
    """Send result to Google Chat."""
    try:
        from aragora.server.handlers.bots.google_chat import get_google_chat_connector

        connector = get_google_chat_connector()
        if not connector:
            logger.warning("Google Chat connector not configured")
            return False

        space_name = origin.channel_id  # Space name in format "spaces/XXXXX"
        thread_name = origin.thread_id or origin.metadata.get("thread_name")

        # Format result data
        consensus = result.get("consensus_reached", False)
        answer = result.get("final_answer", "No conclusion reached.")
        confidence = result.get("confidence", 0)
        topic = result.get("task", origin.metadata.get("topic", "Unknown topic"))

        # Truncate long answers
        if len(answer) > 500:
            answer = answer[:500] + "..."

        # Build Card v2 sections
        consensus_emoji = "\u2705" if consensus else "\u274c"
        confidence_bar = "\u2588" * int(confidence * 5) + "\u2591" * (5 - int(confidence * 5))

        sections = [
            {"header": f"{consensus_emoji} Debate Complete"},
            {"widgets": [{"textParagraph": {"text": f"<b>Topic:</b> {topic[:200]}"}}]},
            {
                "widgets": [
                    {
                        "decoratedText": {
                            "topLabel": "Consensus",
                            "text": "Yes" if consensus else "No",
                        }
                    },
                    {
                        "decoratedText": {
                            "topLabel": "Confidence",
                            "text": f"{confidence_bar} {confidence:.0%}",
                        }
                    },
                ]
            },
            {"widgets": [{"textParagraph": {"text": f"<b>Conclusion:</b>\n{answer}"}}]},
        ]

        # Add vote buttons
        debate_id = result.get("id", origin.debate_id)
        sections.append(
            {
                "widgets": [
                    {
                        "buttonList": {
                            "buttons": [
                                {
                                    "text": "\U0001f44d Agree",
                                    "onClick": {
                                        "action": {
                                            "function": "vote_agree",
                                            "parameters": [
                                                {"key": "debate_id", "value": debate_id}
                                            ],
                                        }
                                    },
                                },
                                {
                                    "text": "\U0001f44e Disagree",
                                    "onClick": {
                                        "action": {
                                            "function": "vote_disagree",
                                            "parameters": [
                                                {"key": "debate_id", "value": debate_id}
                                            ],
                                        }
                                    },
                                },
                            ]
                        }
                    }
                ]
            }
        )

        # Send message with card
        response = await connector.send_message(
            space_name,
            f"Debate complete: {topic[:50]}...",
            blocks=sections,
            thread_id=thread_name,
        )

        if response.success:
            logger.info(f"Google Chat result sent to {space_name}")
            return True
        else:
            logger.warning(f"Google Chat send failed: {response.error}")
            return False

    except ImportError as e:
        logger.warning(f"Google Chat connector not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Google Chat result send error: {e}")
        return False


async def _send_google_chat_receipt(origin: DebateOrigin, summary: str) -> bool:
    """Post receipt summary to Google Chat."""
    try:
        from aragora.server.handlers.bots.google_chat import get_google_chat_connector

        connector = get_google_chat_connector()
        if not connector:
            return False

        space_name = origin.channel_id
        thread_name = origin.thread_id or origin.metadata.get("thread_name")

        response = await connector.send_message(
            space_name,
            summary,
            thread_id=thread_name,
        )

        if response.success:
            logger.info(f"Google Chat receipt posted to {space_name}")
            return True
        return False

    except ImportError:
        return False
    except Exception as e:
        logger.error(f"Google Chat receipt post error: {e}")
        return False
