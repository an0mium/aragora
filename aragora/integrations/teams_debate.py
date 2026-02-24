"""
Microsoft Teams thread debate lifecycle management.

Enables starting debates from Teams messages/threads, routing round-by-round
progress updates back to the originating thread, and posting the final
consensus/receipt as a threaded Adaptive Card reply.

Integrates with:
- ``aragora.integrations.teams.TeamsIntegration`` for webhook delivery
- ``aragora.connectors.chat.teams_conversations`` for conversation references
- ``aragora.connectors.chat.teams_adaptive_cards`` for rich card templates
- ``aragora.server.debate_origin`` for bidirectional result routing

Usage:
    lifecycle = TeamsDebateLifecycle(teams_integration)

    # Start a debate from a Teams thread
    debate_id = await lifecycle.start_debate_from_thread(
        channel_id="19:abc@thread.tacv2",
        message_id="1677012345678",
        topic="Should we adopt microservices?",
    )

    # Post round progress
    await lifecycle.post_round_update(channel_id, message_id, round_data)

    # Post final consensus
    await lifecycle.post_consensus(channel_id, message_id, result)

    # Handle bot commands from Teams activities
    response = await lifecycle.handle_bot_command(activity)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TeamsDebateConfig:
    """Configuration for a Teams-initiated debate."""

    rounds: int = 3
    agents: list[str] = field(default_factory=lambda: ["claude", "gpt4", "gemini"])
    consensus_threshold: float = 0.7
    timeout_seconds: float = 300.0
    enable_voting: bool = True


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _build_round_update_card(
    topic: str,
    round_number: int,
    total_rounds: int,
    agent_messages: list[dict[str, str]] | None = None,
    current_consensus: str | None = None,
    debate_id: str | None = None,
) -> dict[str, Any]:
    """Build an Adaptive Card for a round progress update.

    Delegates to ``TeamsAdaptiveCards.progress_card`` when available,
    falling back to a simpler inline card.

    Args:
        topic: The debate topic.
        round_number: Current round number.
        total_rounds: Total number of rounds.
        agent_messages: List of dicts with ``agent`` and ``summary`` keys.
        current_consensus: Emerging consensus text, if any.
        debate_id: Optional debate identifier.

    Returns:
        Adaptive Card dict.
    """
    try:
        from aragora.connectors.chat.teams_adaptive_cards import (
            RoundProgress,
            TeamsAdaptiveCards,
        )

        progress = RoundProgress(
            round_number=round_number,
            total_rounds=total_rounds,
            agent_messages=agent_messages or [],
            current_consensus=current_consensus,
        )
        return TeamsAdaptiveCards.progress_card(
            topic=topic,
            progress=progress,
            debate_id=debate_id,
        )
    except ImportError:
        pass

    # Fallback: minimal card
    pct = int((round_number / total_rounds) * 100) if total_rounds else 0
    body: list[dict[str, Any]] = [
        {
            "type": "TextBlock",
            "text": f"Round {round_number}/{total_rounds} ({pct}%)",
            "weight": "Bolder",
            "size": "Medium",
        },
        {
            "type": "TextBlock",
            "text": topic,
            "wrap": True,
            "isSubtle": True,
        },
    ]

    if agent_messages:
        for msg in agent_messages[-3:]:
            agent = msg.get("agent", "Agent")
            summary = msg.get("summary", "")
            if len(summary) > 200:
                summary = summary[:200] + "..."
            body.append(
                {
                    "type": "TextBlock",
                    "text": f"**{agent}:** {summary}",
                    "wrap": True,
                    "size": "Small",
                }
            )

    if current_consensus:
        body.append(
            {
                "type": "TextBlock",
                "text": f"Emerging consensus: {current_consensus}",
                "wrap": True,
                "isSubtle": True,
                "spacing": "Medium",
            }
        )

    return {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.4",
        "body": body,
    }


def _build_consensus_card(
    topic: str,
    result: dict[str, Any],
    debate_id: str,
) -> dict[str, Any]:
    """Build an Adaptive Card for the final consensus/result.

    Delegates to ``TeamsAdaptiveCards.verdict_card`` when available,
    falling back to a simpler inline card.

    Args:
        topic: The debate topic.
        result: Debate result dict (or DebateResult-like object attributes).
        debate_id: Debate identifier.

    Returns:
        Adaptive Card dict.
    """
    consensus_reached = result.get("consensus_reached", False)
    confidence = result.get("confidence", 0.0)
    final_answer = result.get("final_answer", "No conclusion reached.")
    participants = result.get("participants", [])
    rounds_used = result.get("rounds_used", 0)
    receipt_id = result.get("receipt_id")

    try:
        from aragora.connectors.chat.teams_adaptive_cards import (
            AgentContribution,
            TeamsAdaptiveCards,
        )

        agents = []
        for name in participants:
            position = "for" if consensus_reached else "neutral"
            agents.append(
                AgentContribution(
                    name=name,
                    position=position,
                    key_point="",
                    confidence=confidence,
                )
            )

        return TeamsAdaptiveCards.verdict_card(
            topic=topic,
            verdict=final_answer[:500],
            confidence=confidence,
            agents=agents,
            rounds_completed=rounds_used,
            receipt_id=receipt_id,
            debate_id=debate_id,
        )
    except ImportError:
        pass

    # Fallback card
    status = "Consensus Reached" if consensus_reached else "Debate Complete"
    status_color = "Good" if consensus_reached else "Warning"

    body: list[dict[str, Any]] = [
        {
            "type": "TextBlock",
            "text": status,
            "weight": "Bolder",
            "size": "Large",
            "color": status_color,
        },
        {
            "type": "TextBlock",
            "text": topic,
            "wrap": True,
            "isSubtle": True,
        },
    ]

    facts = [
        {"title": "Confidence", "value": f"{confidence:.0%}"},
        {"title": "Rounds", "value": str(rounds_used)},
    ]
    if participants:
        facts.append({"title": "Agents", "value": ", ".join(participants[:5])})
    body.append({"type": "FactSet", "facts": facts})

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

    actions: list[dict[str, Any]] = [
        {
            "type": "Action.OpenUrl",
            "title": "View Full Report",
            "url": f"https://aragora.ai/debate/{debate_id}",
        },
    ]

    return {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.4",
        "body": body,
        "actions": actions,
    }


def _wrap_card_payload(card: dict[str, Any]) -> dict[str, Any]:
    """Wrap an Adaptive Card dict as a Teams message attachment payload."""
    return {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": card,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Main lifecycle class
# ---------------------------------------------------------------------------


class TeamsDebateLifecycle:
    """Manages the full lifecycle of a debate started from a Teams thread.

    Coordinates between the Teams webhook/bot integration and the debate
    engine to provide a seamless in-thread experience:

    1. Parses a Teams message or bot command to initiate a debate.
    2. Registers the debate origin for bidirectional routing.
    3. Posts progress cards for each debate round.
    4. Posts the final consensus card with receipt link.

    Args:
        teams_integration: An optional ``TeamsIntegration`` instance for
            webhook delivery. Created lazily if not provided.
    """

    # Recognised bot command prefixes
    COMMAND_PREFIX = "/aragora"
    DEBATE_COMMAND = "debate"
    STATUS_COMMAND = "status"
    HELP_COMMAND = "help"

    def __init__(self, teams_integration: Any | None = None) -> None:
        self._integration = teams_integration
        # In-flight debate tracking: debate_id -> metadata
        self._active_debates: dict[str, dict[str, Any]] = {}

    @property
    def integration(self) -> Any:
        """Lazily initialise the TeamsIntegration."""
        if self._integration is None:
            from aragora.integrations.teams import TeamsIntegration

            self._integration = TeamsIntegration()
        return self._integration

    # -----------------------------------------------------------------
    # Debate lifecycle methods
    # -----------------------------------------------------------------

    async def start_debate_from_thread(
        self,
        channel_id: str,
        message_id: str,
        topic: str,
        config: TeamsDebateConfig | None = None,
        user_id: str = "",
        tenant_id: str = "",
    ) -> str:
        """Start a new debate originating from a Teams thread.

        Registers the debate origin, stores a conversation reference for
        proactive reply routing, and posts an initial "Debate Starting"
        card to the thread.

        Args:
            channel_id: Teams channel or conversation ID.
            message_id: The message ID to thread replies under.
            topic: The debate topic / question.
            config: Optional debate configuration overrides.
            user_id: Teams user ID of the initiator.
            tenant_id: Azure AD tenant ID.

        Returns:
            The generated debate ID.
        """
        config = config or TeamsDebateConfig()
        debate_id = f"teams-{uuid.uuid4().hex[:12]}"

        # Track locally
        self._active_debates[debate_id] = {
            "channel_id": channel_id,
            "message_id": message_id,
            "topic": topic,
            "config": config,
            "user_id": user_id,
            "tenant_id": tenant_id,
        }

        # Register with debate origin system for bidirectional routing
        try:
            from aragora.server.debate_origin import register_debate_origin

            register_debate_origin(
                debate_id=debate_id,
                platform="teams",
                channel_id=channel_id,
                user_id=user_id,
                metadata={
                    "message_id": message_id,
                    "tenant_id": tenant_id,
                    "topic": topic,
                },
                thread_id=message_id,
                message_id=message_id,
            )
            logger.info("Registered debate origin for %s on Teams", debate_id)
        except ImportError:
            logger.debug("debate_origin module not available, skipping registration")
        except Exception as exc:
            logger.warning("Failed to register debate origin: %s", exc)

        # Post starting card
        try:
            from aragora.connectors.chat.teams_adaptive_cards import TeamsAdaptiveCards

            card = TeamsAdaptiveCards.starting_card(
                topic=topic,
                initiated_by=user_id or "Teams User",
                agents=config.agents,
                debate_id=debate_id,
            )
        except ImportError:
            card = {
                "type": "AdaptiveCard",
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "version": "1.4",
                "body": [
                    {
                        "type": "TextBlock",
                        "text": "Debate Starting",
                        "weight": "Bolder",
                        "size": "Large",
                    },
                    {"type": "TextBlock", "text": topic, "wrap": True},
                    {
                        "type": "TextBlock",
                        "text": f"Agents: {', '.join(config.agents)}",
                        "isSubtle": True,
                    },
                ],
            }

        await self._send_card_to_thread(channel_id, message_id, card)
        logger.info("Started Teams debate %s in channel %s", debate_id, channel_id)
        return debate_id

    async def post_round_update(
        self,
        channel_id: str,
        message_id: str,
        round_data: dict[str, Any],
    ) -> bool:
        """Post a round progress update to the originating thread.

        Args:
            channel_id: Teams channel or conversation ID.
            message_id: The message ID to thread replies under.
            round_data: Round information with keys:
                - debate_id (str): Debate identifier.
                - topic (str): The debate topic.
                - round_number (int): Current round number.
                - total_rounds (int): Total rounds.
                - agent_messages (list[dict]): Agent name/summary pairs.
                - current_consensus (str|None): Emerging consensus.

        Returns:
            True if the update was posted successfully.
        """
        debate_id = round_data.get("debate_id", "")
        topic = round_data.get("topic", "")
        round_number = round_data.get("round_number", 0)
        total_rounds = round_data.get("total_rounds", 0)
        agent_messages = round_data.get("agent_messages")
        current_consensus = round_data.get("current_consensus")

        card = _build_round_update_card(
            topic=topic,
            round_number=round_number,
            total_rounds=total_rounds,
            agent_messages=agent_messages,
            current_consensus=current_consensus,
            debate_id=debate_id,
        )

        success = await self._send_card_to_thread(channel_id, message_id, card)
        if success:
            logger.debug(
                "Posted round %s/%s update for debate %s",
                round_number,
                total_rounds,
                debate_id,
            )
        return success

    async def post_consensus(
        self,
        channel_id: str,
        message_id: str,
        result: dict[str, Any],
    ) -> bool:
        """Post the final consensus/result card to the originating thread.

        Args:
            channel_id: Teams channel or conversation ID.
            message_id: The message ID to thread replies under.
            result: Debate result dict with keys like consensus_reached,
                final_answer, confidence, participants, rounds_used,
                receipt_id, debate_id.

        Returns:
            True if the consensus was posted successfully.
        """
        debate_id = result.get("debate_id", "")
        topic = result.get("topic", result.get("task", ""))

        card = _build_consensus_card(
            topic=topic,
            result=result,
            debate_id=debate_id,
        )

        success = await self._send_card_to_thread(channel_id, message_id, card)

        # Mark result sent via debate origin
        if success and debate_id:
            try:
                from aragora.server.debate_origin import mark_result_sent

                mark_result_sent(debate_id)
            except ImportError:
                pass
            except Exception as exc:
                logger.warning("Failed to mark result sent: %s", exc)

            # Clean up local tracking
            self._active_debates.pop(debate_id, None)

        if success:
            logger.info("Posted consensus for debate %s", debate_id)
        return success

    # -----------------------------------------------------------------
    # Bot command handling
    # -----------------------------------------------------------------

    async def handle_bot_command(self, activity: dict[str, Any]) -> dict[str, Any] | None:
        """Handle an incoming Teams bot command activity.

        Parses the activity text for recognised commands and dispatches
        accordingly.

        Supported commands:
            ``/aragora debate <topic>`` - Start a debate
            ``/aragora status <debate_id>`` - Check debate status
            ``/aragora help`` - Show available commands

        Args:
            activity: A Bot Framework activity dict with at minimum
                ``text``, ``conversation.id``, and optional ``replyToId``.

        Returns:
            A response dict with ``text`` and/or ``card`` to send back,
            or None if the activity is not a recognised command.
        """
        text = (activity.get("text") or "").strip()

        # Strip bot mention (Teams includes @mention in text)
        if "<at>" in text:
            # Remove <at>...</at> mention tag
            import re

            text = re.sub(r"<at>.*?</at>\s*", "", text).strip()

        if not text.lower().startswith(self.COMMAND_PREFIX):
            return None

        parts = text[len(self.COMMAND_PREFIX) :].strip().split(maxsplit=1)
        command = parts[0].lower() if parts else self.HELP_COMMAND
        argument = parts[1] if len(parts) > 1 else ""

        conversation = activity.get("conversation", {})
        channel_id = conversation.get("id", "")
        message_id = activity.get("replyToId") or activity.get("id", "")
        user_id = activity.get("from", {}).get("id", "")
        tenant_id = conversation.get("tenantId", "")

        if command == self.DEBATE_COMMAND:
            if not argument:
                return {"text": "Please provide a debate topic. Usage: /aragora debate <topic>"}
            debate_id = await self.start_debate_from_thread(
                channel_id=channel_id,
                message_id=message_id,
                topic=argument,
                user_id=user_id,
                tenant_id=tenant_id,
            )
            return {"text": f"Debate started: {debate_id}", "debate_id": debate_id}

        elif command == self.STATUS_COMMAND:
            return self._get_debate_status(argument)

        elif command == self.HELP_COMMAND:
            return self._build_help_response()

        else:
            return {
                "text": f"Unknown command: {command}. Type /aragora help for available commands.",
            }

    def _get_debate_status(self, debate_id: str) -> dict[str, Any]:
        """Return status information for a debate.

        Args:
            debate_id: The debate to look up.

        Returns:
            Response dict with text and optional card.
        """
        if not debate_id:
            return {"text": "Please provide a debate ID. Usage: /aragora status <debate_id>"}

        info = self._active_debates.get(debate_id)
        if info:
            return {
                "text": (
                    f"Debate {debate_id} is active.\n"
                    f"Topic: {info['topic']}\n"
                    f"Channel: {info['channel_id']}"
                ),
            }

        return {"text": f"Debate {debate_id} not found in active debates."}

    @staticmethod
    def _build_help_response() -> dict[str, Any]:
        """Build a help message listing available commands.

        Returns:
            Response dict with help text.
        """
        return {
            "text": (
                "**Aragora Bot Commands:**\n"
                "- `/aragora debate <topic>` - Start a new debate\n"
                "- `/aragora status <debate_id>` - Check debate status\n"
                "- `/aragora help` - Show this help message"
            ),
        }

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    async def _send_card_to_thread(
        self,
        channel_id: str,
        message_id: str,
        card: dict[str, Any],
    ) -> bool:
        """Send an Adaptive Card as a threaded reply.

        Attempts proactive Bot Framework messaging first, then falls back
        to the webhook-based ``TeamsIntegration._send_card``.

        Args:
            channel_id: Teams channel or conversation ID.
            message_id: The message ID to reply to (thread anchor).
            card: Adaptive Card dict.

        Returns:
            True if the card was sent successfully.
        """
        # Strategy 1: Proactive messaging via Bot Framework
        try:
            from aragora.server.debate_origin.senders.teams import _send_via_proactive
            from aragora.server.debate_origin.models import DebateOrigin

            origin = DebateOrigin(
                debate_id="",
                platform="teams",
                channel_id=channel_id,
                user_id="",
                thread_id=message_id,
                message_id=message_id,
            )
            proactive_result = await _send_via_proactive(origin, card=card)
            if proactive_result is True:
                return True
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("Proactive messaging unavailable: %s", exc)

        # Strategy 2: Webhook via TeamsIntegration
        try:
            from aragora.integrations.teams import AdaptiveCard

            wrapped = AdaptiveCard(title="")
            # Override the card payload entirely
            wrapped_payload = _wrap_card_payload(card)
            return await self.integration._send_card(
                AdaptiveCard(
                    title=card.get("body", [{}])[0].get("text", "Aragora"),
                    body=card.get("body", [])[1:] if len(card.get("body", [])) > 1 else [],
                    actions=card.get("actions", []),
                )
            )
        except ImportError:
            logger.warning("TeamsIntegration not available for card delivery")
            return False
        except Exception as exc:
            logger.warning("Failed to send card via webhook: %s", exc)
            return False


__all__ = [
    "TeamsDebateLifecycle",
    "TeamsDebateConfig",
]
