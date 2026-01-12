"""
Telegram Bot integration for aragora debates.

Posts debate summaries, consensus alerts, and error notifications to Telegram chats.
Uses Telegram's HTML formatting and inline keyboards for rich messages.
"""

import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import aiohttp

from aragora.core import DebateResult

logger = logging.getLogger(__name__)


@dataclass
class TelegramConfig:
    """Configuration for Telegram integration."""

    bot_token: str
    chat_id: str

    # Notification settings
    notify_on_consensus: bool = True
    notify_on_debate_end: bool = True
    notify_on_error: bool = True

    # Minimum confidence for consensus alerts
    min_consensus_confidence: float = 0.7

    # Rate limiting (Telegram allows 30 msg/sec, we use conservative default)
    max_messages_per_minute: int = 20

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0

    # Parse mode for message formatting
    parse_mode: str = "HTML"

    def __post_init__(self) -> None:
        if not self.bot_token:
            raise ValueError("Telegram bot token is required")
        if not self.chat_id:
            raise ValueError("Telegram chat ID is required")

    @property
    def api_base(self) -> str:
        """Get the Telegram Bot API base URL."""
        return f"https://api.telegram.org/bot{self.bot_token}"


@dataclass
class InlineButton:
    """An inline keyboard button."""

    text: str
    url: Optional[str] = None
    callback_data: Optional[str] = None

    def to_dict(self) -> dict[str, str]:
        """Convert to Telegram button dict."""
        button: dict[str, str] = {"text": self.text}
        if self.url:
            button["url"] = self.url
        elif self.callback_data:
            button["callback_data"] = self.callback_data
        return button


@dataclass
class TelegramMessage:
    """A Telegram message with optional inline keyboard."""

    text: str
    reply_markup: list[list[InlineButton]] = field(default_factory=list)
    disable_web_page_preview: bool = True
    disable_notification: bool = False

    def to_payload(self, config: TelegramConfig) -> dict[str, Any]:
        """Convert to Telegram API payload."""
        payload: dict[str, Any] = {
            "chat_id": config.chat_id,
            "text": self.text,
            "parse_mode": config.parse_mode,
            "disable_web_page_preview": self.disable_web_page_preview,
            "disable_notification": self.disable_notification,
        }

        if self.reply_markup:
            payload["reply_markup"] = {
                "inline_keyboard": [
                    [button.to_dict() for button in row]
                    for row in self.reply_markup
                ]
            }

        return payload


class TelegramIntegration:
    """
    Telegram integration for posting debate events.

    Usage:
        telegram = TelegramIntegration(TelegramConfig(
            bot_token="123456:ABC-DEF1234...",
            chat_id="-1001234567890"
        ))

        # Post debate summary
        await telegram.post_debate_summary(debate_result)

        # Post consensus alert
        await telegram.send_consensus_alert(debate_id, confidence=0.85)
    """

    def __init__(self, config: TelegramConfig):
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
            logger.warning("Telegram rate limit reached, skipping message")
            return False

        self._message_count += 1
        return True

    async def _send_message(self, message: TelegramMessage) -> bool:
        """Send a message to Telegram with retry logic."""
        if not self._check_rate_limit():
            return False

        for attempt in range(self.config.max_retries):
            try:
                session = await self._get_session()
                payload = message.to_payload(self.config)

                url = f"{self.config.api_base}/sendMessage"
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    result = await response.json()

                    if response.status == 200 and result.get("ok"):
                        logger.debug("Telegram message sent successfully")
                        return True
                    elif response.status == 429:
                        # Rate limited by Telegram
                        retry_after = result.get("parameters", {}).get("retry_after", 30)
                        logger.warning(f"Telegram rate limited, retry after {retry_after}s")
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(retry_after)
                            continue
                    else:
                        error_desc = result.get("description", "Unknown error")
                        logger.error(f"Telegram API error: {response.status} - {error_desc}")
                        return False

            except aiohttp.ClientError as e:
                logger.error(f"Telegram connection error: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                    continue
                return False
            except Exception as e:
                logger.error(f"Telegram send failed: {e}")
                return False

        return False

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    async def post_debate_summary(self, result: DebateResult) -> bool:
        """Post a debate summary to Telegram.

        Args:
            result: The debate result to summarize

        Returns:
            True if message was sent successfully
        """
        if not self.config.notify_on_debate_end:
            return True

        # Build HTML message
        text = self._build_debate_summary_html(result)

        # Add inline keyboard buttons
        buttons: list[list[InlineButton]] = []
        if hasattr(result, "debate_id"):
            buttons.append([
                InlineButton(
                    text="View Details",
                    url=f"https://aragora.ai/debate/{result.debate_id}"
                ),
                InlineButton(
                    text="Share",
                    callback_data=f"share_{result.debate_id}"
                )
            ])

        message = TelegramMessage(text=text, reply_markup=buttons)
        return await self._send_message(message)

    def _build_debate_summary_html(self, result: DebateResult) -> str:
        """Build HTML formatted debate summary."""
        status_emoji = "\u2705" if result.consensus_reached else "\u274c"  # checkmark or X
        consensus_status = "Reached" if result.consensus_reached else "Not Reached"
        winner_text = result.winner or "No clear winner"
        confidence = getattr(result, 'confidence', 0.0)

        task_escaped = self._escape_html(result.task)

        lines = [
            f"<b>{status_emoji} Debate Completed</b>",
            "",
            f"<b>Task:</b> {task_escaped}",
            "",
            f"\u2022 <b>Consensus:</b> {consensus_status}",
            f"\u2022 <b>Winner:</b> {winner_text}",
            f"\u2022 <b>Confidence:</b> {confidence:.0%}",
            f"\u2022 <b>Rounds:</b> {result.rounds_used}",
        ]

        # Add final answer preview if consensus reached
        if result.consensus_reached and result.final_answer:
            preview = self._escape_html(result.final_answer[:400])
            if len(result.final_answer) > 400:
                preview += "..."
            lines.extend([
                "",
                f"<b>Final Proposal:</b>",
                f"<pre>{preview}</pre>",
            ])

        lines.extend([
            "",
            f"<i>\U0001F916 Aragora AI | {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</i>",
        ])

        return "\n".join(lines)

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
            logger.debug(f"Skipping consensus alert: confidence {confidence} < {self.config.min_consensus_confidence}")
            return True

        lines = [
            "\U0001F389 <b>Consensus Reached!</b>",
            "",
            f"\u2022 <b>Debate:</b> <code>{debate_id[:8]}...</code>",
            f"\u2022 <b>Confidence:</b> {confidence:.0%}",
        ]

        if winner:
            lines.append(f"\u2022 <b>Winner:</b> {self._escape_html(winner)}")

        if task:
            task_preview = self._escape_html(task[:100])
            if len(task) > 100:
                task_preview += "..."
            lines.extend(["", f"<i>Task: {task_preview}</i>"])

        text = "\n".join(lines)

        buttons: list[list[InlineButton]] = [[
            InlineButton(
                text="View Debate",
                url=f"https://aragora.ai/debate/{debate_id}"
            )
        ]]

        message = TelegramMessage(text=text, reply_markup=buttons)
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
            "info": "\u2139\ufe0f",       # info
            "warning": "\u26a0\ufe0f",    # warning
            "error": "\u274c",            # X
            "critical": "\U0001F6A8",     # rotating light
        }
        emoji = severity_emojis.get(severity, "\u26a0\ufe0f")

        error_escaped = self._escape_html(error_message[:800])

        lines = [
            f"{emoji} <b>{self._escape_html(error_type)}</b>",
            "",
            f"<pre>{error_escaped}</pre>",
        ]

        if debate_id:
            lines.extend(["", f"<i>Debate: <code>{debate_id}</code></i>"])

        text = "\n".join(lines)
        message = TelegramMessage(text=text)
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

        lines = ["\U0001F3C6 <b>Agent Leaderboard Update</b>", ""]

        medals = ["\U0001F947", "\U0001F948", "\U0001F949"]  # gold, silver, bronze
        for i, agent in enumerate(top_agents):
            medal = medals[i] if i < 3 else f"#{i+1}"
            name = self._escape_html(agent.get("name", "Unknown"))
            elo = agent.get("elo", 1500)
            wins = agent.get("wins", 0)
            lines.append(f"{medal} <b>{name}</b> - ELO: {elo:.0f} | Wins: {wins}")

        text = "\n".join(lines)
        message = TelegramMessage(text=text)
        return await self._send_message(message)

    async def send_debate_started(
        self,
        debate_id: str,
        task: str,
        agents: list[str],
    ) -> bool:
        """Send a notification when a debate starts.

        Args:
            debate_id: ID of the debate
            task: The debate task/question
            agents: List of participating agents

        Returns:
            True if message was sent successfully
        """
        task_escaped = self._escape_html(task)
        agents_text = ", ".join(self._escape_html(a) for a in agents[:4])
        if len(agents) > 4:
            agents_text += f" +{len(agents) - 4} more"

        lines = [
            "\U0001F4AC <b>Debate Started</b>",
            "",
            f"<b>Task:</b> {task_escaped}",
            "",
            f"<b>Agents:</b> {agents_text}",
            "",
            f"<i>ID: <code>{debate_id[:8]}...</code></i>",
        ]

        text = "\n".join(lines)

        buttons: list[list[InlineButton]] = [[
            InlineButton(
                text="Watch Live",
                url=f"https://aragora.ai/debate/{debate_id}"
            )
        ]]

        message = TelegramMessage(text=text, reply_markup=buttons)
        return await self._send_message(message)


__all__ = [
    "TelegramConfig",
    "TelegramMessage",
    "InlineButton",
    "TelegramIntegration",
]
