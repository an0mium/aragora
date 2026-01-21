"""
Slack integration for aragora debates.

Posts debate summaries, consensus alerts, and error notifications to Slack channels.
Uses Slack's Block Kit for rich message formatting.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import aiohttp

from aragora.core import DebateResult

try:
    from aragora.observability.tracing import build_trace_headers
except ImportError:

    def build_trace_headers() -> dict[str, str]:
        return {}


logger = logging.getLogger(__name__)


@dataclass
class SlackConfig:
    """Configuration for Slack integration."""

    webhook_url: str
    channel: str = "#debates"
    bot_name: str = "Aragora"
    icon_emoji: str = ":speech_balloon:"

    # Notification settings
    notify_on_consensus: bool = True
    notify_on_debate_end: bool = True
    notify_on_error: bool = True

    # Minimum confidence for consensus alerts
    min_consensus_confidence: float = 0.7

    # Rate limiting
    max_messages_per_minute: int = 10

    def __post_init__(self) -> None:
        if not self.webhook_url:
            raise ValueError("Slack webhook URL is required")


@dataclass
class SlackMessage:
    """A Slack message with optional blocks."""

    text: str
    blocks: list[dict[str, Any]] = field(default_factory=list)
    attachments: list[dict[str, Any]] = field(default_factory=list)

    def to_payload(self, config: SlackConfig) -> dict[str, Any]:
        """Convert to Slack webhook payload."""
        payload: dict[str, Any] = {
            "text": self.text,
            "username": config.bot_name,
            "icon_emoji": config.icon_emoji,
        }
        if self.blocks:
            payload["blocks"] = self.blocks
        if self.attachments:
            payload["attachments"] = self.attachments
        return payload


class SlackIntegration:
    """
    Slack integration for posting debate events.

    Usage:
        slack = SlackIntegration(SlackConfig(
            webhook_url="https://hooks.slack.com/services/...",
            channel="#debates"
        ))

        # Post debate summary
        await slack.post_debate_summary(debate_result)

        # Post consensus alert
        await slack.send_consensus_alert(debate_id, confidence=0.85)
    """

    def __init__(self, config: SlackConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._message_count = 0
        self._last_reset = datetime.now()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = datetime.now()
        elapsed = (now - self._last_reset).total_seconds()

        if elapsed >= 60:
            self._message_count = 0
            self._last_reset = now

        if self._message_count >= self.config.max_messages_per_minute:
            logger.warning("Slack rate limit reached, skipping message")
            return False

        self._message_count += 1
        return True

    async def _send_message(self, message: SlackMessage) -> bool:
        """Send a message to Slack."""
        if not self._check_rate_limit():
            return False

        try:
            session = await self._get_session()
            payload = message.to_payload(self.config)
            # Include trace headers for distributed tracing
            headers = build_trace_headers()

            async with session.post(
                self.config.webhook_url,
                json=payload,
                headers=headers if headers else None,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == 200:
                    logger.debug("Slack message sent successfully")
                    return True
                else:
                    text = await response.text()
                    logger.error(f"Slack API error: {response.status} - {text}")
                    return False

        except aiohttp.ClientError as e:
            logger.error(f"Slack connection error: {e}")
            return False
        except Exception as e:
            logger.error(f"Slack send failed: {e}")
            return False

    async def post_debate_summary(self, result: DebateResult) -> bool:
        """Post a debate summary to Slack.

        Args:
            result: The debate result to summarize

        Returns:
            True if message was sent successfully
        """
        if not self.config.notify_on_debate_end:
            return True

        # Build blocks for rich formatting
        blocks = self._build_debate_summary_blocks(result)

        # Fallback text
        consensus_text = "reached" if result.consensus_reached else "not reached"
        text = f"Debate completed: {result.task[:50]}... - Consensus {consensus_text}"

        message = SlackMessage(text=text, blocks=blocks)
        return await self._send_message(message)

    def _build_debate_summary_blocks(self, result: DebateResult) -> list[dict[str, Any]]:
        """Build Block Kit blocks for debate summary."""
        blocks: list[dict[str, Any]] = []

        # Header
        status_emoji = ":white_check_mark:" if result.consensus_reached else ":x:"
        blocks.append(
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{status_emoji} Debate Completed",
                    "emoji": True,
                },
            }
        )

        # Task description
        blocks.append(
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Task:* {result.task}"}}
        )

        # Divider
        blocks.append({"type": "divider"})

        # Results section
        consensus_status = "Reached" if result.consensus_reached else "Not Reached"
        winner_text = result.winner or "No clear winner"
        confidence = getattr(result, "confidence", 0.0)

        blocks.append(
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Consensus:*\n{consensus_status}"},
                    {"type": "mrkdwn", "text": f"*Winner:*\n{winner_text}"},
                    {"type": "mrkdwn", "text": f"*Confidence:*\n{confidence:.0%}"},
                    {"type": "mrkdwn", "text": f"*Rounds:*\n{result.rounds_used}"},
                ],
            }
        )

        # Final answer preview if consensus reached
        if result.consensus_reached and result.final_answer:
            preview = result.final_answer[:500]
            if len(result.final_answer) > 500:
                preview += "..."

            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Final Proposal:*\n```{preview}```"},
                }
            )

        # Context footer
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f":robot_face: Aragora AI Debate System | {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
                    }
                ],
            }
        )

        return blocks

    async def send_consensus_alert(
        self,
        debate_id: str,
        confidence: float,
        winner: Optional[str] = None,
        task: Optional[str] = None,
    ) -> bool:
        """Send a consensus reached notification.

        Args:
            debate_id: ID of the debate
            confidence: Consensus confidence score
            winner: Winning agent name
            task: Task description

        Returns:
            True if message was sent successfully
        """
        if not self.config.notify_on_consensus:
            return True

        if confidence < self.config.min_consensus_confidence:
            logger.debug(
                f"Skipping consensus alert: confidence {confidence} < {self.config.min_consensus_confidence}"
            )
            return True

        blocks: list[dict[str, Any]] = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": ":tada: Consensus Reached!", "emoji": True},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Debate:*\n{debate_id[:8]}..."},
                    {"type": "mrkdwn", "text": f"*Confidence:*\n{confidence:.0%}"},
                ],
            },
        ]

        if winner:
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Winning Position:* {winner}"},
                }
            )

        if task:
            blocks.append(
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"Task: _{task[:100]}{'...' if len(task) > 100 else ''}_",
                        }
                    ],
                }
            )

        text = f"Consensus reached in debate {debate_id[:8]}... with {confidence:.0%} confidence"
        message = SlackMessage(text=text, blocks=blocks)
        return await self._send_message(message)

    async def send_error_alert(
        self,
        error_type: str,
        error_message: str,
        debate_id: Optional[str] = None,
        severity: str = "warning",
    ) -> bool:
        """Send an error notification.

        Args:
            error_type: Type of error
            error_message: Error details
            debate_id: Optional debate ID
            severity: One of "info", "warning", "error", "critical"

        Returns:
            True if message was sent successfully
        """
        if not self.config.notify_on_error:
            return True

        severity_emojis = {
            "info": ":information_source:",
            "warning": ":warning:",
            "error": ":x:",
            "critical": ":rotating_light:",
        }
        emoji = severity_emojis.get(severity, ":warning:")

        blocks: list[dict[str, Any]] = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"{emoji} {error_type}", "emoji": True},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"```{error_message[:1000]}```"},
            },
        ]

        if debate_id:
            blocks.append(
                {
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": f"Debate: `{debate_id}`"}],
                }
            )

        text = f"{error_type}: {error_message[:100]}"
        message = SlackMessage(text=text, blocks=blocks)
        return await self._send_message(message)

    async def send_leaderboard_update(
        self,
        rankings: list[dict[str, Any]],
        top_n: int = 5,
    ) -> bool:
        """Send a leaderboard update.

        Args:
            rankings: List of agent rankings with name, elo, wins, etc.
            top_n: Number of top agents to show

        Returns:
            True if message was sent successfully
        """
        top_agents = rankings[:top_n]

        blocks: list[dict[str, Any]] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": ":trophy: Agent Leaderboard Update",
                    "emoji": True,
                },
            }
        ]

        for i, agent in enumerate(top_agents):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i + 1}"
            name = agent.get("name", "Unknown")
            elo = agent.get("elo", 1500)
            wins = agent.get("wins", 0)

            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"{medal} *{name}* - ELO: {elo:.0f} | Wins: {wins}",
                    },
                }
            )

        text = f"Leaderboard update: {top_agents[0].get('name', 'Unknown') if top_agents else 'No agents'} leads"
        message = SlackMessage(text=text, blocks=blocks)
        return await self._send_message(message)


__all__ = [
    "SlackConfig",
    "SlackMessage",
    "SlackIntegration",
]
