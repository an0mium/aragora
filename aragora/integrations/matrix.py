"""
Matrix/Element integration for aragora debates.

Posts debate summaries, consensus alerts, and responds to room commands
via the Matrix Client-Server API.

Requires:
    MATRIX_HOMESERVER_URL - Matrix server URL (e.g., https://matrix.org)
    MATRIX_ACCESS_TOKEN - Bot access token
    MATRIX_USER_ID - Bot user ID (e.g., @aragora-bot:matrix.org)
    MATRIX_ROOM_ID - Room to post to (e.g., !abc123:matrix.org)

Usage:
    matrix = MatrixIntegration(MatrixConfig(
        homeserver_url="https://matrix.org",
        access_token="syt_xxxxx",
        room_id="!abc123:matrix.org"
    ))
    await matrix.post_debate_summary(debate_result)
"""

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import aiohttp

from aragora.core import DebateResult
from aragora.http_client import DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)


@dataclass
class MatrixConfig:
    """Configuration for Matrix integration."""

    homeserver_url: str = ""
    access_token: str = ""
    user_id: str = ""
    room_id: str = ""

    # Notification settings
    notify_on_consensus: bool = True
    notify_on_debate_end: bool = True
    notify_on_error: bool = True
    notify_on_leaderboard: bool = False

    # Respond to room commands
    enable_commands: bool = True

    # Minimum confidence for consensus alerts
    min_consensus_confidence: float = 0.7

    # Rate limiting
    max_messages_per_minute: int = 10

    # Message format
    use_html: bool = True

    def __post_init__(self) -> None:
        # Load from environment if not provided
        if not self.homeserver_url:
            self.homeserver_url = os.environ.get("MATRIX_HOMESERVER_URL", "")
        if not self.access_token:
            self.access_token = os.environ.get("MATRIX_ACCESS_TOKEN", "")
        if not self.user_id:
            self.user_id = os.environ.get("MATRIX_USER_ID", "")
        if not self.room_id:
            self.room_id = os.environ.get("MATRIX_ROOM_ID", "")

        # Ensure homeserver URL doesn't end with /
        self.homeserver_url = self.homeserver_url.rstrip("/")


class MatrixIntegration:
    """
    Matrix/Element integration for posting debate events.

    Uses the Matrix Client-Server API to send messages to rooms.
    Supports both plain text and HTML formatted messages.

    Usage:
        matrix = MatrixIntegration(MatrixConfig(
            homeserver_url="https://matrix.org",
            access_token="syt_xxxxx",
            user_id="@aragora-bot:matrix.org",
            room_id="!abc123:matrix.org"
        ))

        # Post debate summary
        await matrix.post_debate_summary(debate_result)

        # Post consensus alert
        await matrix.send_consensus_alert(debate_id, answer="...", confidence=0.85)

        # Listen for commands (requires sync loop)
        await matrix.start_command_listener()
    """

    # Matrix API version
    API_VERSION = "v3"

    def __init__(self, config: Optional[MatrixConfig] = None):
        self.config = config or MatrixConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._message_count = 0
        self._last_reset = datetime.now()
        self._sync_token: Optional[str] = None

    @property
    def is_configured(self) -> bool:
        """Check if Matrix integration is configured."""
        return bool(self.config.homeserver_url and self.config.access_token and self.config.room_id)

    def _api_url(self, path: str) -> str:
        """Build Matrix API URL."""
        return f"{self.config.homeserver_url}/_matrix/client/{self.API_VERSION}{path}"

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with timeout protection."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=DEFAULT_TIMEOUT)
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
            logger.warning("Matrix rate limit reached, skipping message")
            return False

        self._message_count += 1
        return True

    def _get_headers(self) -> dict[str, str]:
        """Get API request headers."""
        return {
            "Authorization": f"Bearer {self.config.access_token}",
            "Content-Type": "application/json",
        }

    async def send_message(
        self,
        text: str,
        html: Optional[str] = None,
        room_id: Optional[str] = None,
    ) -> bool:
        """Send a message to a Matrix room.

        Args:
            text: Plain text message
            html: Optional HTML formatted message
            room_id: Room to send to (defaults to config room)

        Returns:
            True if message was sent successfully
        """
        if not self.is_configured:
            logger.warning("Matrix not configured, skipping message")
            return False

        if not self._check_rate_limit():
            return False

        target_room = room_id or self.config.room_id
        txn_id = str(uuid.uuid4())
        url = self._api_url(f"/rooms/{target_room}/send/m.room.message/{txn_id}")

        # Build message content
        content: dict[str, Any] = {
            "msgtype": "m.text",
            "body": text,
        }

        if html and self.config.use_html:
            content["format"] = "org.matrix.custom.html"
            content["formatted_body"] = html

        try:
            session = await self._get_session()

            async with session.put(
                url,
                json=content,
                headers=self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == 200:
                    logger.debug("Matrix message sent successfully")
                    return True
                else:
                    text_response = await response.text()
                    logger.error(f"Matrix API error: {response.status} - {text_response}")
                    return False

        except aiohttp.ClientError as e:
            logger.error(f"Matrix connection error: {e}")
            return False
        except asyncio.TimeoutError:
            logger.error("Matrix request timed out")
            return False

    async def post_debate_summary(self, result: DebateResult) -> bool:
        """Post a debate summary to Matrix.

        Args:
            result: The debate result to summarize

        Returns:
            True if message was sent successfully
        """
        if not self.config.notify_on_debate_end:
            return False

        # Plain text version
        lines = [
            "ARAGORA DEBATE COMPLETE",
            "",
            f"Question: {result.task}",
        ]

        if result.final_answer:
            lines.extend(["", f"Answer: {result.final_answer[:500]}"])

        stats = [f"Rounds: {result.rounds_used}"]
        if result.confidence:
            stats.append(f"Confidence: {result.confidence:.0%}")
        if result.participants:
            stats.append(f"Agents: {len(result.participants)}")

        lines.extend(["", " | ".join(stats)])

        if result.participants:
            lines.append(f"\nParticipants: {', '.join(result.participants[:5])}")

        lines.append(f"\nhttps://aragora.ai/debate/{result.debate_id}")

        text = "\n".join(lines)

        # HTML version
        html = f"""
<h3>ARAGORA DEBATE COMPLETE</h3>
<p><strong>Question:</strong> {self._escape_html(result.task)}</p>
"""

        if result.final_answer:
            answer_preview = result.final_answer[:500]
            if len(result.final_answer) > 500:
                answer_preview += "..."
            html += f"<p><strong>Answer:</strong> {self._escape_html(answer_preview)}</p>"

        html += f"<p><code>{' | '.join(stats)}</code></p>"

        if result.participants:
            agents_list = ", ".join(result.participants[:5])
            if len(result.participants) > 5:
                agents_list += f" +{len(result.participants) - 5} more"
            html += f"<p><em>Participants: {agents_list}</em></p>"

        html += f'<p><a href="https://aragora.ai/debate/{result.debate_id}">View Details</a></p>'

        return await self.send_message(text, html)

    async def send_consensus_alert(
        self,
        debate_id: str,
        answer: str,
        confidence: float,
        agents: Optional[list[str]] = None,
    ) -> bool:
        """Send a consensus reached alert.

        Args:
            debate_id: ID of the debate
            answer: The consensus answer
            confidence: Confidence level (0-1)
            agents: List of participating agents

        Returns:
            True if message was sent successfully
        """
        if not self.config.notify_on_consensus:
            return False

        if confidence < self.config.min_consensus_confidence:
            return False

        answer_preview = answer[:500]
        if len(answer) > 500:
            answer_preview += "..."

        # Plain text
        text = f"CONSENSUS REACHED\n\n{answer_preview}\n\nConfidence: {confidence:.0%}\n"
        if agents:
            text += f"Agreement from: {', '.join(agents[:5])}\n"
        text += f"\nhttps://aragora.ai/debate/{debate_id}"

        # HTML
        confidence_color = "green" if confidence >= 0.8 else "orange"
        html = f"""
<h3>CONSENSUS REACHED</h3>
<p>{self._escape_html(answer_preview)}</p>
<p>Confidence: <font color="{confidence_color}"><strong>{confidence:.0%}</strong></font></p>
"""
        if agents:
            html += f"<p><em>Agreement from: {', '.join(agents[:5])}</em></p>"
        html += f'<p><a href="https://aragora.ai/debate/{debate_id}">View Debate</a></p>'

        return await self.send_message(text, html)

    async def send_error_alert(
        self,
        debate_id: str,
        error: str,
        phase: Optional[str] = None,
    ) -> bool:
        """Send an error notification.

        Args:
            debate_id: ID of the debate
            error: Error message
            phase: Current phase when error occurred

        Returns:
            True if message was sent successfully
        """
        if not self.config.notify_on_error:
            return False

        text = f"ARAGORA ERROR\n\nDebate: {debate_id}\nError: {error}"
        if phase:
            text += f"\nPhase: {phase}"

        html = f"""
<h3><font color="red">ARAGORA ERROR</font></h3>
<p><strong>Debate:</strong> {debate_id}</p>
<p><strong>Error:</strong> {self._escape_html(error)}</p>
"""
        if phase:
            html += f"<p><em>Phase: {phase}</em></p>"

        return await self.send_message(text, html)

    async def send_leaderboard_update(
        self,
        rankings: list[dict[str, Any]],
        domain: Optional[str] = None,
    ) -> bool:
        """Send a leaderboard update.

        Args:
            rankings: List of agent rankings with name, elo, wins, losses
            domain: Optional domain for domain-specific rankings

        Returns:
            True if message was sent successfully
        """
        if not self.config.notify_on_leaderboard:
            return False

        title = f"LEADERBOARD UPDATE{f' ({domain})' if domain else ''}"

        # Plain text
        lines = [title, ""]
        for i, agent in enumerate(rankings[:10], 1):
            medal = {1: "1st", 2: "2nd", 3: "3rd"}.get(i, f"{i}th")
            lines.append(
                f"{medal}. {agent['name']} - ELO: {agent.get('elo', 1500):.0f} "
                f"({agent.get('wins', 0)}W/{agent.get('losses', 0)}L)"
            )

        lines.append("\nhttps://aragora.ai/leaderboard")
        text = "\n".join(lines)

        # HTML
        html = f"<h3>{title}</h3><ol>"
        for agent in rankings[:10]:
            html += (
                f"<li><strong>{agent['name']}</strong> - "
                f"ELO: {agent.get('elo', 1500):.0f} "
                f"({agent.get('wins', 0)}W/{agent.get('losses', 0)}L)</li>"
            )
        html += '</ol><p><a href="https://aragora.ai/leaderboard">Full Leaderboard</a></p>'

        return await self.send_message(text, html)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    async def verify_connection(self) -> bool:
        """Verify the Matrix connection and permissions.

        Returns:
            True if connection is valid and bot can send to room
        """
        if not self.is_configured:
            return False

        try:
            session = await self._get_session()
            url = self._api_url("/account/whoami")

            async with session.get(
                url,
                headers=self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Matrix connected as: {data.get('user_id')}")
                    return True
                else:
                    logger.error(f"Matrix auth failed: {response.status}")
                    return False

        except aiohttp.ClientError as e:
            logger.error(f"Matrix connection failed: {e}")
            return False
        except asyncio.TimeoutError:
            logger.error("Matrix connection verification timed out")
            return False

    async def join_room(self, room_id: str) -> bool:
        """Join a Matrix room.

        Args:
            room_id: Room ID to join

        Returns:
            True if join was successful
        """
        if not self.is_configured:
            return False

        try:
            session = await self._get_session()
            url = self._api_url(f"/rooms/{room_id}/join")

            async with session.post(
                url,
                json={},
                headers=self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == 200:
                    logger.info(f"Joined Matrix room: {room_id}")
                    return True
                else:
                    text = await response.text()
                    logger.error(f"Failed to join room: {response.status} - {text}")
                    return False

        except aiohttp.ClientError as e:
            logger.error(f"Matrix join room failed: {e}")
            return False
        except asyncio.TimeoutError:
            logger.error("Matrix join room request timed out")
            return False

    async def __aenter__(self) -> "MatrixIntegration":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
