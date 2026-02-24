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

import asyncio
import logging
import re
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


def _build_receipt_blocks(
    receipt: Any,
    debate_id: str = "",
    receipt_url: str = "",
) -> list[dict[str, Any]]:
    """Build Block Kit blocks for a decision receipt message.

    Args:
        receipt: A DecisionReceipt (or duck-typed object) with attributes
                 verdict, confidence, findings, key_arguments,
                 dissenting_views/dissents, receipt_id.
        debate_id: Optional debate ID for context.
        receipt_url: Optional URL to the full receipt.
    """
    verdict = getattr(receipt, "verdict", "UNKNOWN")
    verdict_emoji = {
        "APPROVED": ":white_check_mark:",
        "REJECTED": ":x:",
        "NEEDS_REVIEW": ":warning:",
        "APPROVED_WITH_CONDITIONS": ":large_yellow_circle:",
    }.get(str(verdict).upper(), ":question:")

    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{verdict_emoji} Decision Receipt: {verdict}",
                "emoji": True,
            },
        },
    ]

    # Verdict, confidence, and finding counts
    confidence = getattr(receipt, "confidence", 0.0) or 0.0
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

    fields = [
        {"type": "mrkdwn", "text": f"*Verdict:*\n{verdict}"},
        {"type": "mrkdwn", "text": f"*Confidence:*\n{confidence:.0%}"},
        {
            "type": "mrkdwn",
            "text": (
                f"*Findings:*\n"
                f"{len(findings)} total"
                f"{f' ({critical_count} critical)' if critical_count else ''}"
                f"{f' ({high_count} high)' if high_count else ''}"
            ),
        },
    ]
    blocks.append({"type": "section", "fields": fields})

    # Key arguments
    key_arguments = getattr(receipt, "key_arguments", None)
    if key_arguments is None:
        # Fallback: extract from findings descriptions
        key_arguments = [
            getattr(f, "description", str(f))
            for f in findings[:5]
            if getattr(f, "description", None)
        ]
    if key_arguments:
        arg_lines = "\n".join(f"  {i + 1}. {a}" for i, a in enumerate(key_arguments[:5]))
        blocks.append({"type": "divider"})
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Key Arguments:*\n{arg_lines}",
                },
            }
        )

    # Dissenting views
    dissenting_views = getattr(receipt, "dissenting_views", None) or getattr(
        receipt, "dissents", None
    )
    if dissenting_views:
        dissent_lines = "\n".join(
            f"  - {d}" for d in (dissenting_views[:5] if isinstance(dissenting_views, list) else [str(dissenting_views)])
        )
        blocks.append({"type": "divider"})
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Dissenting Views:*\n{dissent_lines}",
                },
            }
        )

    # Action buttons
    action_elements: list[dict[str, Any]] = []
    if receipt_url:
        action_elements.append(
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "View Full Receipt"},
                "url": receipt_url,
            }
        )
    action_elements.append(
        {
            "type": "button",
            "text": {"type": "plain_text", "text": "Audit Trail"},
            "action_id": "view_audit_trail",
        }
    )
    blocks.append({"type": "actions", "elements": action_elements})

    # Footer
    receipt_id = getattr(receipt, "receipt_id", "")
    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f":receipt: Receipt `{receipt_id[:12]}...` | "
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
                    ),
                }
            ],
        }
    )

    return blocks


def _build_error_blocks(
    error_message: str,
    debate_id: str = "",
) -> list[dict[str, Any]]:
    """Build Block Kit blocks for an error notification.

    Args:
        error_message: Human-readable error description.
        debate_id: Optional debate ID for context.
    """
    blocks: list[dict[str, Any]] = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f":warning: *Error:* {error_message}",
            },
        },
    ]
    if debate_id:
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Debate `{debate_id[:12]}...`",
                    }
                ],
            }
        )
    return blocks


def parse_mention_text(text: str) -> tuple[str, str]:
    """Parse an @mention text to extract a debate/decide command and topic.

    Strips the ``<@BOTID>`` mention pattern, then looks for ``debate`` or
    ``decide`` keywords.

    Args:
        text: Raw event text from Slack (may contain ``<@U...>`` mention).

    Returns:
        A ``(command, topic)`` tuple.  ``command`` is ``"debate"`` or
        ``"decide"`` if a keyword was found, ``""`` otherwise.  ``topic``
        is the remaining text after the keyword (stripped of quotes), or
        ``""`` if not found.
    """
    # Strip bot mention
    clean = re.sub(r"<@[A-Z0-9]+>", "", text).strip()
    clean = re.sub(r"\s+", " ", clean)  # collapse whitespace

    if not clean:
        return ("", "")

    lower = clean.lower()
    for keyword in ("debate", "decide"):
        if lower.startswith(keyword):
            rest = clean[len(keyword):].strip().strip("\"'")
            return (keyword, rest)

    return ("", "")


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

    async def post_receipt(
        self,
        channel_id: str,
        thread_ts: str,
        receipt: Any,
        debate_id: str = "",
        receipt_url: str = "",
    ) -> bool:
        """Post a decision receipt to the thread.

        Args:
            channel_id: Slack channel ID.
            thread_ts: Thread timestamp.
            receipt: A DecisionReceipt or duck-typed object.
            debate_id: Optional debate ID for context.
            receipt_url: Optional URL to the full receipt page.

        Returns:
            True if posted successfully.
        """
        blocks = _build_receipt_blocks(receipt, debate_id, receipt_url)
        verdict = getattr(receipt, "verdict", "UNKNOWN")
        text = f"Decision receipt: {verdict}"
        return await self._post_to_thread(channel_id, thread_ts, text, blocks)

    async def post_error(
        self,
        channel_id: str,
        thread_ts: str,
        error_message: str,
        debate_id: str = "",
    ) -> bool:
        """Post an error notification to the thread.

        Args:
            channel_id: Slack channel ID.
            thread_ts: Thread timestamp.
            error_message: Human-readable error description.
            debate_id: Optional debate ID for context.

        Returns:
            True if posted successfully.
        """
        blocks = _build_error_blocks(error_message, debate_id)
        text = f"Error: {error_message}"
        return await self._post_to_thread(channel_id, thread_ts, text, blocks)

    async def run_debate(
        self,
        channel_id: str,
        thread_ts: str,
        debate_id: str,
        topic: str,
        config: SlackDebateConfig | None = None,
    ) -> Any:
        """Run a debate and stream progress updates to the Slack thread.

        Lazily imports the debate engine.  Posts round updates, the final
        consensus, and optionally the decision receipt.

        Args:
            channel_id: Slack channel ID.
            thread_ts: Thread timestamp.
            debate_id: Previously-generated debate ID.
            topic: The debate topic.
            config: Optional debate configuration.

        Returns:
            The DebateResult if the debate completed, None otherwise.
        """
        config = config or SlackDebateConfig()

        try:
            from aragora import Arena, DebateProtocol, Environment
        except ImportError:
            logger.error("Debate engine not available (aragora core not installed)")
            await self.post_error(
                channel_id, thread_ts, "Debate engine is not available.", debate_id
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
                channel_id, thread_ts, "Debate timed out.", debate_id
            )
            return None
        except (RuntimeError, OSError, ValueError) as exc:
            logger.error("Debate %s failed: %s", debate_id, exc)
            await self.post_error(
                channel_id, thread_ts, f"Debate failed: {exc}", debate_id
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
                    "round": getattr(rd, "round_number", 0),
                    "total_rounds": config.rounds,
                    "agent": getattr(rd, "agent", ""),
                    "proposal": getattr(rd, "proposal", ""),
                    "phase": getattr(rd, "phase", "proposal"),
                }
            await self.post_round_update(channel_id, thread_ts, round_data)

        # Post consensus
        await self.post_consensus(channel_id, thread_ts, result)

        # Post receipt if available
        receipt = getattr(result, "receipt", None)
        if receipt:
            await self.post_receipt(
                channel_id, thread_ts, receipt, debate_id=debate_id
            )

        # Mark result as sent via debate_origin
        try:
            from aragora.server.debate_origin import mark_result_sent

            mark_result_sent(debate_id)
        except (ImportError, RuntimeError, OSError):
            pass

        return result

    async def start_and_run_debate(
        self,
        channel_id: str,
        thread_ts: str,
        topic: str,
        config: SlackDebateConfig | None = None,
        user_id: str = "",
    ) -> Any:
        """Convenience: start a debate and immediately run it.

        Combines ``start_debate_from_thread`` and ``run_debate``.

        Returns:
            The DebateResult if the debate completed, None otherwise.
        """
        debate_id = await self.start_debate_from_thread(
            channel_id=channel_id,
            thread_ts=thread_ts,
            topic=topic,
            config=config,
            user_id=user_id,
        )
        return await self.run_debate(
            channel_id=channel_id,
            thread_ts=thread_ts,
            debate_id=debate_id,
            topic=topic,
            config=config,
        )

    async def handle_app_mention(self, event: dict[str, Any]) -> str | None:
        """Handle an @mention event that may contain a debate trigger.

        Parses the mention text for ``debate`` or ``decide`` keywords
        and starts a debate lifecycle if found.

        Args:
            event: Slack event payload.

        Returns:
            The debate_id if a debate was started, None otherwise.
        """
        text = event.get("text", "")
        channel_id = event.get("channel", "")
        user_id = event.get("user", "")
        thread_ts = event.get("thread_ts", "") or event.get("ts", "")

        command, topic = parse_mention_text(text)
        if not command:
            return None

        if not topic:
            await self._post_to_thread(
                channel_id,
                thread_ts,
                f"Usage: `@aragora {command} \"Your topic here\"`",
            )
            return None

        # Schedule the debate as a background task
        debate_id = await self.start_debate_from_thread(
            channel_id=channel_id,
            thread_ts=thread_ts,
            topic=topic,
            user_id=user_id,
        )
        asyncio.create_task(
            self._run_debate_background(channel_id, thread_ts, debate_id, topic),
            name=f"slack-debate-{debate_id[:12]}",
        )
        return debate_id

    async def _run_debate_background(
        self,
        channel_id: str,
        thread_ts: str,
        debate_id: str,
        topic: str,
    ) -> None:
        """Run a debate in the background, handling errors gracefully."""
        try:
            await self.run_debate(
                channel_id=channel_id,
                thread_ts=thread_ts,
                debate_id=debate_id,
                topic=topic,
            )
        except (RuntimeError, OSError, ValueError, asyncio.CancelledError) as exc:
            logger.error("Background debate %s failed: %s", debate_id, exc)
            await self.post_error(
                channel_id, thread_ts, f"Debate failed: {exc}", debate_id
            )

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
    "parse_mention_text",
    "_build_receipt_blocks",
    "_build_error_blocks",
]
