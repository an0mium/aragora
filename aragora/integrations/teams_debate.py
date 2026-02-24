"""
Microsoft Teams thread debate lifecycle management.

Enables starting debates from Teams messages/threads, routing round-by-round
progress updates back to the originating thread, and posting the final
consensus/receipt as a threaded Adaptive Card reply.

Mirrors the SlackDebateLifecycle pattern for consistent cross-platform behavior.

Integrates with:
- ``aragora.integrations.teams.TeamsIntegration`` for webhook delivery
- ``aragora.connectors.chat.teams_adaptive_cards`` for rich card templates
- ``aragora.server.debate_origin`` for bidirectional result routing

Usage:
    lifecycle = TeamsDebateLifecycle(bot_token="...", service_url="...")

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

    # Stop a debate
    stopped = stop_debate_in_thread(channel_id, message_id)
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Active debate tracking (in-memory, per-process)
# ---------------------------------------------------------------------------

# Maps debate_id -> TeamsActiveDebateState for debates currently running
_active_debates: dict[str, "TeamsActiveDebateState"] = {}


@dataclass
class TeamsActiveDebateState:
    """In-memory tracking for a running Teams-thread debate.

    This enables stop commands, user voting, and suggestion collection
    while the debate is in progress.
    """

    debate_id: str
    channel_id: str
    message_id: str
    topic: str
    user_id: str
    tenant_id: str = ""
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    user_votes: dict[str, str] = field(default_factory=dict)
    user_suggestions: list[dict[str, str]] = field(default_factory=list)
    status: str = "running"  # running, stopping, completed, failed

    def record_vote(self, voter_id: str, vote: str) -> None:
        """Record a vote from a Teams user.

        Args:
            voter_id: Teams user ID of the voter.
            vote: The vote value (e.g. ``agree``, ``disagree``, ``abstain``).
        """
        self.user_votes[voter_id] = vote

    def add_suggestion(self, user_id: str, text: str) -> None:
        """Add a user suggestion from a thread reply.

        Args:
            user_id: Teams user ID.
            text: The suggestion text.
        """
        self.user_suggestions.append({"user_id": user_id, "text": text})

    def request_stop(self) -> None:
        """Request that this debate stop early."""
        self.status = "stopping"
        self.cancel_event.set()

    @property
    def vote_summary(self) -> dict[str, int]:
        """Return a summary of votes as {vote_value: count}."""
        counts: dict[str, int] = {}
        for vote in self.user_votes.values():
            counts[vote] = counts.get(vote, 0) + 1
        return counts


def get_active_debate(debate_id: str) -> TeamsActiveDebateState | None:
    """Get the active debate state for a debate ID."""
    return _active_debates.get(debate_id)


def get_active_debate_for_thread(
    channel_id: str, message_id: str
) -> TeamsActiveDebateState | None:
    """Find the active debate for a given Teams thread.

    Args:
        channel_id: Teams channel or conversation ID.
        message_id: Message ID anchoring the thread.

    Returns:
        The active debate state, or None if no debate is running in this thread.
    """
    for state in _active_debates.values():
        if state.channel_id == channel_id and state.message_id == message_id:
            return state
    return None


def stop_debate(debate_id: str) -> bool:
    """Request that a running debate stop early.

    Args:
        debate_id: The debate to stop.

    Returns:
        True if a running debate was found and stop was requested.
    """
    state = _active_debates.get(debate_id)
    if state and state.status == "running":
        state.request_stop()
        return True
    return False


def stop_debate_in_thread(channel_id: str, message_id: str) -> str | None:
    """Stop the debate running in a given Teams thread.

    Args:
        channel_id: Teams channel or conversation ID.
        message_id: Message ID anchoring the thread.

    Returns:
        The debate_id if stopped, None if no running debate found.
    """
    state = get_active_debate_for_thread(channel_id, message_id)
    if state and state.status == "running":
        state.request_stop()
        return state.debate_id
    return None


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
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Adaptive Card formatting helpers
# ---------------------------------------------------------------------------


def _build_debate_started_card(
    debate_id: str,
    topic: str,
    config: TeamsDebateConfig,
) -> dict[str, Any]:
    """Build an Adaptive Card for the 'debate started' announcement."""
    try:
        from aragora.connectors.chat.teams_adaptive_cards import TeamsAdaptiveCards

        return TeamsAdaptiveCards.starting_card(
            topic=topic,
            initiated_by="Teams User",
            agents=config.agents,
            debate_id=debate_id,
        )
    except ImportError:
        pass

    body: list[dict[str, Any]] = [
        {
            "type": "TextBlock",
            "text": "Debate Started",
            "weight": "Bolder",
            "size": "Large",
            "color": "Accent",
        },
        {
            "type": "TextBlock",
            "text": f"**Topic:** {topic}",
            "wrap": True,
        },
        {
            "type": "FactSet",
            "facts": [
                {"title": "Agents", "value": ", ".join(config.agents)},
                {"title": "Rounds", "value": str(config.rounds)},
                {"title": "Debate ID", "value": f"{debate_id[:12]}..."},
            ],
        },
        {
            "type": "TextBlock",
            "text": f"Aragora | {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
            "size": "Small",
            "isSubtle": True,
        },
    ]

    return {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.4",
        "body": body,
        "actions": [
            {
                "type": "Action.Submit",
                "title": "Cancel Debate",
                "style": "destructive",
                "data": {"action": "cancel_debate", "debate_id": debate_id},
            }
        ],
    }


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


def _build_receipt_card(
    receipt: Any,
    debate_id: str = "",
    receipt_url: str = "",
) -> dict[str, Any]:
    """Build an Adaptive Card for a decision receipt message.

    Args:
        receipt: A DecisionReceipt (or duck-typed object) with attributes
                 verdict, confidence, findings, key_arguments,
                 dissenting_views/dissents, receipt_id.
        debate_id: Optional debate ID for context.
        receipt_url: Optional URL to the full receipt.

    Returns:
        Adaptive Card dict.
    """
    verdict = getattr(receipt, "verdict", "UNKNOWN")
    confidence = getattr(receipt, "confidence", 0.0) or 0.0
    receipt_id = getattr(receipt, "receipt_id", "")
    findings = getattr(receipt, "findings", []) or []

    critical_count = sum(
        1
        for f in findings
        if getattr(f, "severity", getattr(f, "level", "")).lower() == "critical"
    )
    high_count = sum(
        1
        for f in findings
        if getattr(f, "severity", getattr(f, "level", "")).lower() == "high"
    )

    findings_text = f"{len(findings)} total"
    if critical_count:
        findings_text += f" ({critical_count} critical)"
    if high_count:
        findings_text += f" ({high_count} high)"

    body: list[dict[str, Any]] = [
        {
            "type": "TextBlock",
            "text": f"Decision Receipt: {verdict}",
            "weight": "Bolder",
            "size": "Large",
        },
        {
            "type": "FactSet",
            "facts": [
                {"title": "Verdict", "value": str(verdict)},
                {"title": "Confidence", "value": f"{confidence:.0%}"},
                {"title": "Findings", "value": findings_text},
            ],
        },
    ]

    # Key arguments
    key_arguments = getattr(receipt, "key_arguments", None)
    if key_arguments is None:
        key_arguments = [
            getattr(f, "description", str(f))
            for f in findings[:5]
            if getattr(f, "description", None)
        ]
    if key_arguments:
        arg_lines = "\n".join(f"  {i + 1}. {a}" for i, a in enumerate(key_arguments[:5]))
        body.append(
            {
                "type": "TextBlock",
                "text": f"**Key Arguments:**\n{arg_lines}",
                "wrap": True,
                "spacing": "Medium",
            }
        )

    # Dissenting views
    dissenting_views = getattr(receipt, "dissenting_views", None) or getattr(
        receipt, "dissents", None
    )
    if dissenting_views:
        dissent_lines = "\n".join(
            f"  - {d}"
            for d in (
                dissenting_views[:5]
                if isinstance(dissenting_views, list)
                else [str(dissenting_views)]
            )
        )
        body.append(
            {
                "type": "TextBlock",
                "text": f"**Dissenting Views:**\n{dissent_lines}",
                "wrap": True,
                "spacing": "Medium",
            }
        )

    # Footer
    body.append(
        {
            "type": "TextBlock",
            "text": (
                f"Receipt {receipt_id[:12]}... | "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
            ),
            "size": "Small",
            "isSubtle": True,
        }
    )

    # Actions
    actions: list[dict[str, Any]] = []
    if receipt_url:
        actions.append(
            {
                "type": "Action.OpenUrl",
                "title": "View Full Receipt",
                "url": receipt_url,
            }
        )
    actions.append(
        {
            "type": "Action.Submit",
            "title": "Audit Trail",
            "data": {"action": "view_audit_trail", "debate_id": debate_id},
        }
    )

    return {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.4",
        "body": body,
        "actions": actions,
    }


def _build_stop_card(
    debate_id: str,
    stopped_by: str = "",
) -> dict[str, Any]:
    """Build an Adaptive Card for a 'debate stopped' notification.

    Args:
        debate_id: The stopped debate ID.
        stopped_by: User ID or name who requested the stop.

    Returns:
        Adaptive Card dict.
    """
    body: list[dict[str, Any]] = [
        {
            "type": "TextBlock",
            "text": "Debate Stopped",
            "weight": "Bolder",
            "size": "Large",
            "color": "Attention",
        },
    ]
    if stopped_by:
        body.append(
            {
                "type": "TextBlock",
                "text": f"Stopped by {stopped_by} | Debate {debate_id[:12]}...",
                "size": "Small",
                "isSubtle": True,
            }
        )

    return {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.4",
        "body": body,
    }


def _build_error_card(
    error_message: str,
    debate_id: str = "",
) -> dict[str, Any]:
    """Build an Adaptive Card for an error notification.

    Args:
        error_message: Human-readable error description.
        debate_id: Optional debate ID for context.

    Returns:
        Adaptive Card dict.
    """
    body: list[dict[str, Any]] = [
        {
            "type": "TextBlock",
            "text": f"**Error:** {error_message}",
            "wrap": True,
            "color": "Attention",
        },
    ]
    if debate_id:
        body.append(
            {
                "type": "TextBlock",
                "text": f"Debate {debate_id[:12]}...",
                "size": "Small",
                "isSubtle": True,
            }
        )

    return {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.4",
        "body": body,
    }


def _build_vote_summary_card(
    vote_summary: dict[str, int],
    suggestions_count: int = 0,
) -> dict[str, Any] | None:
    """Build an Adaptive Card for a user participation summary.

    Args:
        vote_summary: Vote value -> count mapping.
        suggestions_count: Number of user suggestions received.

    Returns:
        Adaptive Card dict, or None if no participation data.
    """
    if not vote_summary and suggestions_count == 0:
        return None

    body: list[dict[str, Any]] = [
        {
            "type": "TextBlock",
            "text": "User Participation",
            "weight": "Bolder",
            "size": "Medium",
        },
    ]

    if vote_summary:
        facts = [
            {"title": vote.capitalize(), "value": str(count)}
            for vote, count in sorted(vote_summary.items(), key=lambda x: -x[1])
        ]
        body.append({"type": "FactSet", "facts": facts})

    if suggestions_count > 0:
        body.append(
            {
                "type": "TextBlock",
                "text": f"**User Suggestions:** {suggestions_count} received",
                "size": "Small",
            }
        )

    return {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.4",
        "body": body,
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


def parse_command_text(text: str) -> tuple[str, str]:
    """Parse a Teams message text to extract a command and argument.

    Strips ``<at>...</at>`` mention tags and looks for ``debate``, ``decide``,
    or ``stop`` keywords.

    Args:
        text: Raw activity text from Teams (may contain ``<at>...</at>`` mentions).

    Returns:
        A ``(command, argument)`` tuple.  ``command`` is one of ``"debate"``,
        ``"decide"``, ``"stop"`` if a keyword was found, ``""`` otherwise.
        ``argument`` is the remaining text after the keyword, or ``""``.
    """
    # Strip bot mention
    clean = re.sub(r"<at>.*?</at>\s*", "", text).strip()
    clean = re.sub(r"\s+", " ", clean)  # collapse whitespace

    if not clean:
        return ("", "")

    lower = clean.lower()
    for keyword in ("debate", "decide", "stop"):
        if lower.startswith(keyword):
            rest = clean[len(keyword):].strip().strip("\"'")
            return (keyword, rest)

    return ("", "")


# ---------------------------------------------------------------------------
# Main lifecycle class
# ---------------------------------------------------------------------------


class TeamsDebateLifecycle:
    """Manages the full lifecycle of a debate within a Teams thread.

    Coordinates debate initiation, progress updates, and result delivery,
    keeping all messages within the originating Teams conversation thread
    for context.

    The class uses httpx for Bot Framework API calls and Adaptive Cards
    for rich formatting.  It lazily imports heavy dependencies (debate
    orchestrator, origin registry) to keep module import fast.

    Args:
        bot_token: Bot Framework access token for proactive messaging.
        service_url: Bot Framework service URL for the conversation.
        teams_integration: Optional ``TeamsIntegration`` instance for
            webhook delivery fallback.
    """

    BOT_FRAMEWORK_BASE = "https://smba.trafficmanager.net"

    # Recognised bot command prefixes
    COMMAND_PREFIX = "/aragora"
    DEBATE_COMMAND = "debate"
    STOP_COMMAND = "stop"
    STATUS_COMMAND = "status"
    HELP_COMMAND = "help"

    def __init__(
        self,
        bot_token: str = "",
        service_url: str = "",
        teams_integration: Any | None = None,
    ) -> None:
        self._bot_token = bot_token
        self._service_url = service_url
        self._integration = teams_integration
        self._client: Any = None  # httpx.AsyncClient, created lazily

    @property
    def integration(self) -> Any:
        """Lazily initialise the TeamsIntegration."""
        if self._integration is None:
            from aragora.integrations.teams import TeamsIntegration

            self._integration = TeamsIntegration()
        return self._integration

    # -- HTTP helpers -------------------------------------------------------

    async def _get_client(self) -> Any:
        """Get or create an httpx async client."""
        import httpx

        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

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
        if self._bot_token and self._service_url:
            try:
                client = await self._get_client()
                url = f"{self._service_url}/v3/conversations/{channel_id}/activities"
                payload = {
                    "type": "message",
                    "replyToId": message_id,
                    "attachments": [
                        {
                            "contentType": "application/vnd.microsoft.card.adaptive",
                            "content": card,
                        }
                    ],
                }
                headers = {
                    "Authorization": f"Bearer {self._bot_token}",
                    "Content-Type": "application/json",
                }
                response = await client.post(url, json=payload, headers=headers)
                if response.is_success:
                    return True
                logger.warning(
                    "Bot Framework API returned %s for thread reply", response.status_code
                )
            except (ImportError, OSError, ValueError) as exc:
                logger.debug("Proactive messaging unavailable: %s", exc)

        # Strategy 2: Via debate_origin sender
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
        except (RuntimeError, OSError, ValueError) as exc:
            logger.debug("Proactive sender unavailable: %s", exc)

        # Strategy 3: Webhook via TeamsIntegration
        try:
            from aragora.integrations.teams import AdaptiveCard

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
        except (RuntimeError, OSError, ValueError) as exc:
            logger.warning("Failed to send card via webhook: %s", exc)
            return False

    # -- Lifecycle methods --------------------------------------------------

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

        Registers the debate origin, stores active debate state for
        stop/vote handling, and posts an initial "Debate Starting"
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

        # Track in module-level active debates for stop/vote support
        state = TeamsActiveDebateState(
            debate_id=debate_id,
            channel_id=channel_id,
            message_id=message_id,
            topic=topic,
            user_id=user_id,
            tenant_id=tenant_id,
        )
        _active_debates[debate_id] = state

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
                    "agents": config.agents,
                    "rounds": config.rounds,
                },
                thread_id=message_id,
                message_id=message_id,
            )
            logger.info("Registered debate origin for %s on Teams", debate_id)
        except ImportError:
            logger.debug("debate_origin module not available, skipping registration")
        except (RuntimeError, OSError, ValueError) as exc:
            logger.warning("Failed to register debate origin: %s", exc)

        # Post starting card
        card = _build_debate_started_card(debate_id, topic, config)
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
        round_number = round_data.get("round_number", round_data.get("round", 0))
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
        result: dict[str, Any] | Any,
    ) -> bool:
        """Post the final consensus/result card to the originating thread.

        Args:
            channel_id: Teams channel or conversation ID.
            message_id: The message ID to thread replies under.
            result: Debate result dict with keys like consensus_reached,
                final_answer, confidence, participants, rounds_used,
                receipt_id, debate_id.  Also accepts DebateResult objects.

        Returns:
            True if the consensus was posted successfully.
        """
        # Normalize result to dict
        if isinstance(result, dict):
            result_dict = result
        else:
            result_dict = {
                "debate_id": getattr(result, "debate_id", ""),
                "topic": getattr(result, "task", getattr(result, "topic", "")),
                "consensus_reached": getattr(result, "consensus_reached", False),
                "confidence": getattr(result, "confidence", 0.0),
                "final_answer": getattr(result, "final_answer", ""),
                "participants": getattr(result, "participants", []),
                "rounds_used": getattr(result, "rounds_used", 0),
                "receipt_id": getattr(result, "receipt_id", None),
            }

        debate_id = result_dict.get("debate_id", "")
        topic = result_dict.get("topic", result_dict.get("task", ""))

        card = _build_consensus_card(
            topic=topic,
            result=result_dict,
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
            except (RuntimeError, OSError, ValueError) as exc:
                logger.warning("Failed to mark result sent: %s", exc)

            # Clean up module-level active debate tracking
            _active_debates.pop(debate_id, None)

        if success:
            logger.info("Posted consensus for debate %s", debate_id)
        return success

    async def post_receipt(
        self,
        channel_id: str,
        message_id: str,
        receipt: Any,
        debate_id: str = "",
        receipt_url: str = "",
    ) -> bool:
        """Post a decision receipt to the thread.

        Args:
            channel_id: Teams channel or conversation ID.
            message_id: The message ID to thread replies under.
            receipt: A DecisionReceipt or duck-typed object.
            debate_id: Optional debate ID for context.
            receipt_url: Optional URL to the full receipt page.

        Returns:
            True if posted successfully.
        """
        card = _build_receipt_card(receipt, debate_id, receipt_url)
        return await self._send_card_to_thread(channel_id, message_id, card)

    async def post_error(
        self,
        channel_id: str,
        message_id: str,
        error_message: str,
        debate_id: str = "",
    ) -> bool:
        """Post an error notification to the thread.

        Args:
            channel_id: Teams channel or conversation ID.
            message_id: The message ID to thread replies under.
            error_message: Human-readable error description.
            debate_id: Optional debate ID for context.

        Returns:
            True if posted successfully.
        """
        card = _build_error_card(error_message, debate_id)
        return await self._send_card_to_thread(channel_id, message_id, card)

    async def post_stop(
        self,
        channel_id: str,
        message_id: str,
        debate_id: str,
        stopped_by: str = "",
    ) -> bool:
        """Post a 'debate stopped' notification to the thread.

        Args:
            channel_id: Teams channel or conversation ID.
            message_id: The message ID to thread replies under.
            debate_id: The stopped debate ID.
            stopped_by: User who requested the stop.

        Returns:
            True if posted successfully.
        """
        card = _build_stop_card(debate_id, stopped_by)
        return await self._send_card_to_thread(channel_id, message_id, card)

    async def run_debate(
        self,
        channel_id: str,
        message_id: str,
        debate_id: str,
        topic: str,
        config: TeamsDebateConfig | None = None,
    ) -> Any:
        """Run a debate and stream progress updates to the Teams thread.

        Lazily imports the debate engine.  Posts round updates, the final
        consensus, and optionally the decision receipt.

        Args:
            channel_id: Teams channel or conversation ID.
            message_id: The message ID to thread replies under.
            debate_id: Previously-generated debate ID.
            topic: The debate topic.
            config: Optional debate configuration.

        Returns:
            The DebateResult if the debate completed, None otherwise.
        """
        config = config or TeamsDebateConfig()

        try:
            from aragora import Arena, DebateProtocol, Environment
        except ImportError:
            logger.error("Debate engine not available (aragora core not installed)")
            await self.post_error(
                channel_id, message_id, "Debate engine is not available.", debate_id
            )
            return None

        env = Environment(task=topic)
        protocol = DebateProtocol(
            rounds=config.rounds,
            consensus=config.consensus_threshold,
        )
        arena = Arena(env, config.agents, protocol)

        result = None
        try:
            result = await asyncio.wait_for(
                arena.run(),
                timeout=config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning("Debate %s timed out after %ss", debate_id, config.timeout_seconds)
            await self.post_error(
                channel_id, message_id, "Debate timed out.", debate_id
            )
            return None
        except (RuntimeError, OSError, ValueError) as exc:
            logger.error("Debate %s failed: %s", debate_id, exc)
            await self.post_error(
                channel_id, message_id, "Debate execution failed.", debate_id
            )
            return None

        # Post round updates from result data if available
        rounds = getattr(result, "rounds", None) or []
        for rd in rounds:
            round_data: dict[str, Any] = {}
            if isinstance(rd, dict):
                round_data = rd
            else:
                round_data = {
                    "round_number": getattr(rd, "round_number", 0),
                    "total_rounds": config.rounds,
                    "topic": topic,
                    "debate_id": debate_id,
                    "agent_messages": [
                        {
                            "agent": getattr(rd, "agent", ""),
                            "summary": getattr(rd, "proposal", ""),
                        }
                    ],
                }
            await self.post_round_update(channel_id, message_id, round_data)

        # Post user participation summary if any
        state = _active_debates.get(debate_id)
        if state:
            summary_card = _build_vote_summary_card(
                state.vote_summary, len(state.user_suggestions)
            )
            if summary_card:
                await self._send_card_to_thread(channel_id, message_id, summary_card)

        # Post consensus
        await self.post_consensus(channel_id, message_id, result)

        # Post receipt if available
        receipt = getattr(result, "receipt", None)
        if receipt:
            await self.post_receipt(
                channel_id, message_id, receipt, debate_id=debate_id
            )

        # Mark result as sent via debate_origin
        try:
            from aragora.server.debate_origin import mark_result_sent

            mark_result_sent(debate_id)
        except (ImportError, RuntimeError, OSError):
            pass

        # Update state
        if state:
            state.status = "completed"

        return result

    async def start_and_run_debate(
        self,
        channel_id: str,
        message_id: str,
        topic: str,
        config: TeamsDebateConfig | None = None,
        user_id: str = "",
        tenant_id: str = "",
    ) -> Any:
        """Convenience: start a debate and immediately run it.

        Combines ``start_debate_from_thread`` and ``run_debate``.

        Returns:
            The DebateResult if the debate completed, None otherwise.
        """
        debate_id = await self.start_debate_from_thread(
            channel_id=channel_id,
            message_id=message_id,
            topic=topic,
            config=config,
            user_id=user_id,
            tenant_id=tenant_id,
        )
        return await self.run_debate(
            channel_id=channel_id,
            message_id=message_id,
            debate_id=debate_id,
            topic=topic,
            config=config,
        )

    # -- Bot command handling -----------------------------------------------

    async def handle_bot_command(self, activity: dict[str, Any]) -> dict[str, Any] | None:
        """Handle an incoming Teams bot command activity.

        Parses the activity text for recognised commands and dispatches
        accordingly.

        Supported commands:
            ``/aragora debate <topic>`` - Start a debate
            ``/aragora stop [debate_id]`` - Stop a running debate
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
            text = re.sub(r"<at>.*?</at>\s*", "", text).strip()

        if not text.lower().startswith(self.COMMAND_PREFIX):
            return None

        parts = text[len(self.COMMAND_PREFIX):].strip().split(maxsplit=1)
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
            # Schedule debate execution in the background
            asyncio.create_task(
                self._run_debate_background(channel_id, message_id, debate_id, argument),
                name=f"teams-debate-{debate_id[:12]}",
            )
            return {"text": f"Debate started: {debate_id}", "debate_id": debate_id}

        elif command == self.STOP_COMMAND:
            return await self._handle_stop_command(
                argument, channel_id, message_id, user_id
            )

        elif command == self.STATUS_COMMAND:
            return self._get_debate_status(argument)

        elif command == self.HELP_COMMAND:
            return self._build_help_response()

        else:
            return {
                "text": f"Unknown command: {command}. Type /aragora help for available commands.",
            }

    async def handle_adaptive_card_action(
        self, activity: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Handle an Adaptive Card action submission (e.g., vote, cancel).

        Args:
            activity: Bot Framework activity with ``value`` containing the action data.

        Returns:
            A response dict, or None if the action is not recognised.
        """
        value = activity.get("value", {})
        action = value.get("action", "")
        conversation = activity.get("conversation", {})
        channel_id = conversation.get("id", "")
        user_id = activity.get("from", {}).get("id", "")

        if action == "vote":
            return self._handle_vote_action(value, user_id)
        elif action == "cancel_debate":
            debate_id = value.get("debate_id", "")
            return await self._handle_stop_command(
                debate_id, channel_id, "", user_id
            )
        elif action == "suggest":
            return self._handle_suggestion_action(value, user_id)

        return None

    async def handle_thread_reply(
        self, activity: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Handle a thread reply that may contain a vote, suggestion, or stop command.

        This handles free-text replies in a debate thread from Teams users.

        Args:
            activity: Bot Framework activity dict.

        Returns:
            Response dict, or None if the reply is not debate-related.
        """
        text = (activity.get("text") or "").strip()
        conversation = activity.get("conversation", {})
        channel_id = conversation.get("id", "")
        reply_to = activity.get("replyToId", "")
        user_id = activity.get("from", {}).get("id", "")

        if not text or not reply_to:
            return None

        # Find the active debate for this thread
        state = get_active_debate_for_thread(channel_id, reply_to)
        if not state:
            return None

        lower = text.lower().strip()

        # Stop command
        if lower in ("stop", "cancel", "/stop", "/cancel"):
            stopped = stop_debate(state.debate_id)
            if stopped:
                await self.post_stop(
                    channel_id, state.message_id, state.debate_id, stopped_by=user_id
                )
                return {"text": f"Debate {state.debate_id} stop requested."}
            return {"text": "No running debate to stop."}

        # Vote keywords
        if lower in ("agree", "yes", "+1", "for"):
            state.record_vote(user_id, "agree")
            return {"text": "Vote recorded: agree"}
        if lower in ("disagree", "no", "-1", "against"):
            state.record_vote(user_id, "disagree")
            return {"text": "Vote recorded: disagree"}
        if lower in ("abstain", "pass", "skip"):
            state.record_vote(user_id, "abstain")
            return {"text": "Vote recorded: abstain"}

        # Otherwise treat as a suggestion
        if lower.startswith("suggest:") or lower.startswith("suggestion:"):
            suggestion_text = text.split(":", 1)[1].strip()
        else:
            suggestion_text = text

        if suggestion_text:
            state.add_suggestion(user_id, suggestion_text)
            return {"text": "Suggestion recorded."}

        return None

    # -- Internal helpers ---------------------------------------------------

    async def _run_debate_background(
        self,
        channel_id: str,
        message_id: str,
        debate_id: str,
        topic: str,
    ) -> None:
        """Run a debate in the background, handling errors gracefully."""
        try:
            await self.run_debate(
                channel_id=channel_id,
                message_id=message_id,
                debate_id=debate_id,
                topic=topic,
            )
        except (RuntimeError, OSError, ValueError, asyncio.CancelledError) as exc:
            logger.error("Background debate %s failed: %s", debate_id, exc)
            await self.post_error(
                channel_id, message_id, "Debate execution failed.", debate_id
            )
            state = _active_debates.get(debate_id)
            if state:
                state.status = "failed"

    async def _handle_stop_command(
        self,
        argument: str,
        channel_id: str,
        message_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        """Handle the stop command.

        Args:
            argument: Debate ID or empty string (stop by thread).
            channel_id: Conversation ID.
            message_id: Message ID for thread context.
            user_id: User who issued the stop.

        Returns:
            Response dict.
        """
        if argument:
            stopped = stop_debate(argument)
            if stopped:
                state = _active_debates.get(argument)
                if state:
                    await self.post_stop(
                        state.channel_id, state.message_id, argument,
                        stopped_by=user_id,
                    )
                return {"text": f"Debate {argument} stop requested."}
            return {"text": f"Debate {argument} not found or not running."}

        # Try to stop by thread
        if channel_id and message_id:
            debate_id = stop_debate_in_thread(channel_id, message_id)
            if debate_id:
                await self.post_stop(channel_id, message_id, debate_id, stopped_by=user_id)
                return {"text": f"Debate {debate_id} stop requested."}

        return {"text": "No running debate found to stop. Usage: /aragora stop <debate_id>"}

    def _handle_vote_action(self, value: dict[str, Any], user_id: str) -> dict[str, Any]:
        """Handle a vote Adaptive Card action.

        Args:
            value: Action data with ``vote`` and ``debate_id``.
            user_id: Teams user ID of the voter.

        Returns:
            Response dict.
        """
        vote = value.get("vote", "")
        debate_id = value.get("debate_id", "")

        state = _active_debates.get(debate_id)
        if state:
            state.record_vote(user_id, vote)
            logger.info("Vote recorded: %s -> %s for debate %s", user_id, vote, debate_id)
            return {"text": f"Vote recorded: {vote}"}

        return {"text": "Debate not found or already completed."}

    def _handle_suggestion_action(
        self, value: dict[str, Any], user_id: str
    ) -> dict[str, Any]:
        """Handle a suggestion Adaptive Card action.

        Args:
            value: Action data with ``suggestion`` and ``debate_id``.
            user_id: Teams user ID.

        Returns:
            Response dict.
        """
        suggestion = value.get("suggestion", "")
        debate_id = value.get("debate_id", "")

        state = _active_debates.get(debate_id)
        if state and suggestion:
            state.add_suggestion(user_id, suggestion)
            return {"text": "Suggestion recorded."}

        return {"text": "Could not record suggestion."}

    def _get_debate_status(self, debate_id: str) -> dict[str, Any]:
        """Return status information for a debate.

        Args:
            debate_id: The debate to look up.

        Returns:
            Response dict with text and optional card.
        """
        if not debate_id:
            return {"text": "Please provide a debate ID. Usage: /aragora status <debate_id>"}

        state = _active_debates.get(debate_id)
        if state:
            return {
                "text": (
                    f"Debate {debate_id} is active.\n"
                    f"Topic: {state.topic}\n"
                    f"Channel: {state.channel_id}\n"
                    f"Status: {state.status}\n"
                    f"Votes: {len(state.user_votes)}\n"
                    f"Suggestions: {len(state.user_suggestions)}"
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
                "- `/aragora stop [debate_id]` - Stop a running debate\n"
                "- `/aragora status <debate_id>` - Check debate status\n"
                "- `/aragora help` - Show this help message\n"
                "\n"
                "**In-thread participation:**\n"
                "- Reply `agree` / `disagree` / `abstain` to vote\n"
                "- Reply `suggest: <your suggestion>` to add input\n"
                "- Reply `stop` to cancel the debate"
            ),
        }


__all__ = [
    "TeamsActiveDebateState",
    "TeamsDebateConfig",
    "TeamsDebateLifecycle",
    "get_active_debate",
    "get_active_debate_for_thread",
    "parse_command_text",
    "stop_debate",
    "stop_debate_in_thread",
    "_build_consensus_card",
    "_build_debate_started_card",
    "_build_error_card",
    "_build_receipt_card",
    "_build_round_update_card",
    "_build_stop_card",
    "_build_vote_summary_card",
    "_wrap_card_payload",
]
