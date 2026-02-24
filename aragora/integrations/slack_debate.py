"""
Slack thread debate lifecycle management.

Enables starting debates from Slack threads, posting round-by-round progress
updates back to the originating thread, and delivering final consensus/receipt
summaries. Integrates with the debate origin registry for bidirectional routing.

Usage:
    lifecycle = SlackDebateLifecycle(bot_token="xoxb-...")

    # Start debate from a Slack thread
    debate_id = await lifecycle.start_debate_from_thread(
        channel_id="C01ABC",
        thread_ts="1234567890.123456",
        topic="Should we adopt microservices?",
    )

    # Post round updates
    await lifecycle.post_round_update(
        channel_id="C01ABC",
        thread_ts="1234567890.123456",
        round_data={"round": 1, "total_rounds": 3, "agent": "claude", ...},
    )

    # Post final consensus
    await lifecycle.post_consensus(
        channel_id="C01ABC",
        thread_ts="1234567890.123456",
        result=debate_result,
    )
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SlackDebateConfig:
    """Configuration for a Slack-initiated debate."""

    rounds: int = 3
    agents: list[str] = field(default_factory=lambda: ["claude", "gpt4"])
    consensus_threshold: float = 0.7
    timeout_seconds: float = 300.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Block Kit formatting helpers
# ---------------------------------------------------------------------------


def _build_debate_started_blocks(
    debate_id: str,
    topic: str,
    config: SlackDebateConfig,
) -> list[dict[str, Any]]:
    """Build Block Kit blocks for the 'debate started' announcement."""
    return [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": ":speech_balloon: Debate Started",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Topic:* {topic}"},
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Agents:*\n{', '.join(config.agents)}",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Rounds:*\n{config.rounds}",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Debate ID:*\n`{debate_id[:12]}...`",
                },
            ],
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f":robot_face: Aragora | "
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
                    ),
                }
            ],
        },
    ]


def _build_round_update_blocks(round_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Build Block Kit blocks for a round progress update."""
    current_round = round_data.get("round", 0)
    total_rounds = round_data.get("total_rounds", 0)
    agent_name = round_data.get("agent")
    proposal_preview = round_data.get("proposal", "")
    phase = round_data.get("phase", "proposal")

    phase_emoji = {
        "proposal": ":pencil:",
        "critique": ":mag:",
        "revision": ":arrows_counterclockwise:",
        "vote": ":ballot_box:",
    }.get(phase, ":arrows_counterclockwise:")

    blocks: list[dict[str, Any]] = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"{phase_emoji} *Round {current_round}/{total_rounds}* "
                    f"- _{phase.capitalize()}_"
                ),
            },
        },
    ]

    if agent_name and proposal_preview:
        preview = proposal_preview[:300]
        if len(proposal_preview) > 300:
            preview += "..."
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{agent_name}:*\n>{preview}",
                },
            }
        )

    return blocks


def _build_consensus_blocks(result: Any) -> list[dict[str, Any]]:
    """Build Block Kit blocks for the final consensus message.

    Args:
        result: A DebateResult (or compatible duck-typed object) with attributes
                consensus_reached, task, final_answer, confidence, rounds_used, winner.
    """
    consensus_reached = getattr(result, "consensus_reached", False)
    status_emoji = ":white_check_mark:" if consensus_reached else ":x:"
    status_text = "Consensus Reached" if consensus_reached else "No Consensus"

    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{status_emoji} {status_text}",
                "emoji": True,
            },
        },
    ]

    task = getattr(result, "task", "")
    if task:
        blocks.append(
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Task:* {task}"}}
        )

    # Results fields
    confidence = getattr(result, "confidence", 0.0) or 0.0
    rounds_used = getattr(result, "rounds_used", 0)
    winner = getattr(result, "winner", None)

    fields = [
        {"type": "mrkdwn", "text": f"*Confidence:*\n{confidence:.0%}"},
        {"type": "mrkdwn", "text": f"*Rounds:*\n{rounds_used}"},
    ]
    if winner:
        fields.append({"type": "mrkdwn", "text": f"*Winner:*\n{winner}"})

    blocks.append({"type": "section", "fields": fields})

    # Final answer preview
    final_answer = getattr(result, "final_answer", None)
    if consensus_reached and final_answer:
        preview = final_answer[:500]
        if len(final_answer) > 500:
            preview += "..."
        blocks.append({"type": "divider"})
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Final Decision:*\n```{preview}```",
                },
            }
        )

    # Footer
    debate_id = getattr(result, "debate_id", "")
    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f":robot_face: Aragora | "
                        f"Debate `{debate_id[:8]}...` | "
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
                    ),
                }
            ],
        }
    )

    return blocks


# ---------------------------------------------------------------------------
# Main lifecycle class
# ---------------------------------------------------------------------------


class SlackDebateLifecycle:
    """Manages the full lifecycle of a debate within a Slack thread.

    Coordinates debate initiation, progress updates, and result delivery,
    keeping all messages within the originating Slack thread for context.

    The class uses the Slack Web API (chat.postMessage) with ``thread_ts``
    to ensure all messages are threaded.  It lazily imports heavy
    dependencies (debate orchestrator, origin registry) to keep module
    import fast.
    """

    SLACK_API_BASE = "https://slack.com/api"

    def __init__(self, bot_token: str) -> None:
        if not bot_token:
            raise ValueError("Slack bot token is required")
        self._bot_token = bot_token
        self._session: Any = None  # aiohttp.ClientSession, created lazily

    # -- HTTP helpers -------------------------------------------------------

    async def _get_session(self) -> Any:
        """Get or create an aiohttp session."""
        import aiohttp

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
            )
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def _post_to_thread(
        self,
        channel_id: str,
        thread_ts: str,
        text: str,
        blocks: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Post a message to a Slack thread via chat.postMessage.

        Args:
            channel_id: The Slack channel ID.
            thread_ts: The thread timestamp (parent message ts).
            text: Fallback text for notifications.
            blocks: Optional Block Kit blocks for rich formatting.

        Returns:
            True if the message was posted successfully.
        """
        import json as _json

        session = await self._get_session()
        url = f"{self.SLACK_API_BASE}/chat.postMessage"

        payload: dict[str, Any] = {
            "channel": channel_id,
            "thread_ts": thread_ts,
            "text": text,
        }
        if blocks:
            payload["blocks"] = blocks

        headers = {
            "Authorization": f"Bearer {self._bot_token}",
            "Content-Type": "application/json; charset=utf-8",
        }

        try:
            async with session.post(
                url, data=_json.dumps(payload), headers=headers
            ) as resp:
                if resp.status != 200:
                    logger.error(
                        "Slack API returned HTTP %s for chat.postMessage", resp.status
                    )
                    return False
                body = await resp.json()
                if not body.get("ok"):
                    logger.error("Slack API error: %s", body.get("error", "unknown"))
                    return False
                return True
        except OSError as exc:
            logger.error("Network error posting to Slack thread: %s", exc)
            return False

    # -- Lifecycle methods --------------------------------------------------

    async def start_debate_from_thread(
        self,
        channel_id: str,
        thread_ts: str,
        topic: str,
        config: SlackDebateConfig | None = None,
        user_id: str = "",
    ) -> str:
        """Start a debate from a Slack message/thread.

        Registers the debate origin for result routing, posts a
        'debate started' announcement to the thread, and kicks off the
        debate asynchronously.

        Args:
            channel_id: Slack channel ID where the thread lives.
            thread_ts: Timestamp of the parent message (thread root).
            topic: The debate topic / question.
            config: Optional debate configuration.
            user_id: Slack user ID who requested the debate.

        Returns:
            The generated debate_id.
        """
        config = config or SlackDebateConfig()
        debate_id = str(uuid.uuid4())

        # Register origin so results route back to this thread
        try:
            from aragora.server.debate_origin import register_debate_origin

            register_debate_origin(
                debate_id=debate_id,
                platform="slack",
                channel_id=channel_id,
                user_id=user_id,
                thread_id=thread_ts,
                metadata={
                    "topic": topic,
                    "agents": config.agents,
                    "rounds": config.rounds,
                },
            )
        except (ImportError, RuntimeError, OSError) as exc:
            logger.warning("Failed to register debate origin: %s", exc)

        # Post 'debate started' announcement to the thread
        blocks = _build_debate_started_blocks(debate_id, topic, config)
        text = f"Debate started: {topic[:80]}"
        await self._post_to_thread(channel_id, thread_ts, text, blocks)

        logger.info(
            "Started Slack thread debate %s in %s (thread=%s)",
            debate_id,
            channel_id,
            thread_ts,
        )
        return debate_id

    async def post_round_update(
        self,
        channel_id: str,
        thread_ts: str,
        round_data: dict[str, Any],
    ) -> bool:
        """Post a round progress update to the originating thread.

        Args:
            channel_id: Slack channel ID.
            thread_ts: Thread timestamp.
            round_data: Round information dict with keys:
                - round: Current round number
                - total_rounds: Total rounds
                - agent: Agent name (optional)
                - proposal: Proposal preview text (optional)
                - phase: Phase name (optional, default 'proposal')

        Returns:
            True if the update was posted successfully.
        """
        current = round_data.get("round", 0)
        total = round_data.get("total_rounds", 0)
        blocks = _build_round_update_blocks(round_data)
        text = f"Round {current}/{total}"
        return await self._post_to_thread(channel_id, thread_ts, text, blocks)

    async def post_consensus(
        self,
        channel_id: str,
        thread_ts: str,
        result: Any,
    ) -> bool:
        """Post the final consensus / debate result to the thread.

        Args:
            channel_id: Slack channel ID.
            thread_ts: Thread timestamp.
            result: A DebateResult (or duck-typed object with consensus_reached,
                    task, final_answer, confidence, rounds_used, winner).

        Returns:
            True if the message was posted successfully.
        """
        blocks = _build_consensus_blocks(result)
        consensus_reached = getattr(result, "consensus_reached", False)
        status = "reached" if consensus_reached else "not reached"
        task_preview = getattr(result, "task", "")[:60]
        text = f"Debate complete: {task_preview} - Consensus {status}"
        return await self._post_to_thread(channel_id, thread_ts, text, blocks)

    async def handle_slash_command(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle the ``/aragora-debate`` slash command from Slack.

        Parses the command text as the debate topic and starts a debate
        in the channel (or thread) from which the command was invoked.

        Args:
            payload: Parsed slash command payload with keys:
                - text (list[str] or str): Command arguments (the topic).
                - channel_id (list[str] or str): Channel ID.
                - user_id (list[str] or str): User ID.
                - thread_ts (str, optional): Thread timestamp if in a thread.

        Returns:
            A Slack-compatible response dict (ephemeral acknowledgement).
        """

        def _first(val: Any) -> str:
            """Extract first element if list, else str."""
            if isinstance(val, list):
                return str(val[0]) if val else ""
            return str(val) if val else ""

        topic = _first(payload.get("text", ""))
        channel_id = _first(payload.get("channel_id", ""))
        user_id = _first(payload.get("user_id", ""))
        thread_ts = _first(payload.get("thread_ts", ""))

        if not topic:
            return {
                "response_type": "ephemeral",
                "text": "Usage: `/aragora-debate <topic>`\nExample: `/aragora-debate Should we adopt microservices?`",
            }

        if not channel_id:
            return {
                "response_type": "ephemeral",
                "text": "Could not determine channel. Please try again.",
            }

        # Use the channel timestamp as thread root if not already in a thread
        if not thread_ts:
            thread_ts = ""

        try:
            debate_id = await self.start_debate_from_thread(
                channel_id=channel_id,
                thread_ts=thread_ts,
                topic=topic,
                user_id=user_id,
            )
            return {
                "response_type": "in_channel",
                "text": f":speech_balloon: Debate `{debate_id[:12]}...` started: _{topic}_",
            }
        except (RuntimeError, OSError, ValueError) as exc:
            logger.error("Slash command debate start failed: %s", exc)
            return {
                "response_type": "ephemeral",
                "text": "Failed to start debate. Please try again later.",
            }


__all__ = [
    "SlackDebateConfig",
    "SlackDebateLifecycle",
]
