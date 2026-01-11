"""
Discord integration for aragora debates.
Posts debate summaries and consensus alerts using Discord webhooks.
Uses Discord's rich embed format for message formatting.
"""
import logging
import json
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime, timezone
import asyncio
import hashlib
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class DiscordConfig:
    """Configuration for Discord integration."""
    webhook_url: str
    username: str = "Aragora Debates"
    avatar_url: str = ""
    enabled: bool = True
    # Message settings
    include_agent_details: bool = True
    include_vote_breakdown: bool = True
    max_summary_length: int = 1900  # Discord embeds max at 2048
    # Rate limiting
    rate_limit_per_minute: int = 30
    retry_count: int = 3
    retry_delay: float = 1.0


@dataclass
class DiscordEmbed:
    """Discord embed structure."""
    title: str = ""
    description: str = ""
    color: int = 0x5865F2  # Discord blurple
    url: str = ""
    timestamp: str = ""
    footer: Optional[dict[str, str]] = None
    author: Optional[dict[str, str]] = None
    fields: list[dict[str, Any]] = field(default_factory=list)
    thumbnail: Optional[dict[str, str]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert embed to Discord API format."""
        result: dict[str, Any] = {}
        if self.title:
            result["title"] = self.title
        if self.description:
            result["description"] = self.description[:2048]  # Discord limit
        if self.color:
            result["color"] = self.color
        if self.url:
            result["url"] = self.url
        if self.timestamp:
            result["timestamp"] = self.timestamp
        if self.footer:
            result["footer"] = self.footer
        if self.author:
            result["author"] = self.author
        if self.fields:
            result["fields"] = self.fields[:25]  # Discord limit
        if self.thumbnail:
            result["thumbnail"] = self.thumbnail
        return result


class DiscordIntegration:
    """Discord webhook integration for debate notifications."""

    # Colors for different event types
    COLORS = {
        "debate_start": 0x57F287,   # Green
        "consensus": 0x5865F2,       # Blurple
        "no_consensus": 0xFEE75C,    # Yellow
        "error": 0xED4245,           # Red
        "agent_join": 0x3BA55C,      # Lighter green
        "round_complete": 0xEB459E,  # Pink
    }

    def __init__(self, config: DiscordConfig):
        self.config = config
        self._request_times: list[float] = []
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _check_rate_limit(self) -> None:
        """Simple rate limiting check."""
        now = asyncio.get_event_loop().time()
        # Remove requests older than 1 minute
        self._request_times = [
            t for t in self._request_times
            if now - t < 60
        ]
        if len(self._request_times) >= self.config.rate_limit_per_minute:
            wait_time = 60 - (now - self._request_times[0])
            if wait_time > 0:
                logger.warning(f"Discord rate limit hit, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        self._request_times.append(now)

    async def _send_webhook(
        self,
        embeds: list[DiscordEmbed],
        content: str = "",
    ) -> bool:
        """Send message via Discord webhook."""
        if not self.config.enabled:
            return False

        await self._check_rate_limit()

        payload: dict[str, Any] = {
            "username": self.config.username,
            "embeds": [e.to_dict() for e in embeds],
        }
        if content:
            payload["content"] = content[:2000]  # Discord limit
        if self.config.avatar_url:
            payload["avatar_url"] = self.config.avatar_url

        session = await self._get_session()

        for attempt in range(self.config.retry_count):
            try:
                async with session.post(
                    self.config.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 204:
                        return True
                    elif response.status == 429:
                        # Rate limited by Discord
                        retry_after = float(response.headers.get("Retry-After", 5))
                        logger.warning(f"Discord rate limited, retrying after {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    else:
                        text = await response.text()
                        logger.error(f"Discord webhook failed: {response.status} - {text}")

            except asyncio.TimeoutError:
                logger.warning(f"Discord webhook timeout (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"Discord webhook error: {e}")

            if attempt < self.config.retry_count - 1:
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        return False

    def _truncate(self, text: str, max_length: int = 1024) -> str:
        """Truncate text to Discord's field limit."""
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."

    async def send_debate_start(
        self,
        debate_id: str,
        topic: str,
        agents: list[str],
        config: dict[str, Any],
    ) -> bool:
        """Send debate start notification."""
        embed = DiscordEmbed(
            title="Debate Started",
            description=self._truncate(topic, self.config.max_summary_length),
            color=self.COLORS["debate_start"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            footer={"text": f"Debate ID: {debate_id[:8]}..."},
        )

        if self.config.include_agent_details and agents:
            embed.fields.append({
                "name": "Participating Agents",
                "value": ", ".join(agents[:10]) + ("..." if len(agents) > 10 else ""),
                "inline": True,
            })

        rounds = config.get("rounds", "N/A")
        consensus = config.get("consensus_mode", "majority")
        embed.fields.append({
            "name": "Configuration",
            "value": f"Rounds: {rounds} | Mode: {consensus}",
            "inline": True,
        })

        return await self._send_webhook([embed])

    async def send_consensus_reached(
        self,
        debate_id: str,
        topic: str,
        consensus_type: str,
        result: dict[str, Any],
    ) -> bool:
        """Send consensus reached notification."""
        winner = result.get("winner", "Unknown")
        confidence = result.get("confidence", 0)

        embed = DiscordEmbed(
            title="Consensus Reached",
            description=f"**Winner:** {winner}\n**Type:** {consensus_type.title()}",
            color=self.COLORS["consensus"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            footer={"text": f"Debate: {debate_id[:8]}..."},
        )

        embed.fields.append({
            "name": "Confidence",
            "value": f"{confidence:.1%}" if isinstance(confidence, float) else str(confidence),
            "inline": True,
        })

        if self.config.include_vote_breakdown:
            votes = result.get("votes", {})
            if votes:
                vote_text = "\n".join(
                    f"{choice}: {count}" for choice, count in votes.items()
                )
                embed.fields.append({
                    "name": "Vote Breakdown",
                    "value": self._truncate(vote_text, 1024),
                    "inline": True,
                })

        return await self._send_webhook([embed])

    async def send_no_consensus(
        self,
        debate_id: str,
        topic: str,
        final_state: dict[str, Any],
    ) -> bool:
        """Send notification when debate ends without consensus."""
        embed = DiscordEmbed(
            title="Debate Ended - No Consensus",
            description=self._truncate(topic, self.config.max_summary_length),
            color=self.COLORS["no_consensus"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            footer={"text": f"Debate: {debate_id[:8]}..."},
        )

        rounds_completed = final_state.get("rounds_completed", "N/A")
        embed.fields.append({
            "name": "Rounds Completed",
            "value": str(rounds_completed),
            "inline": True,
        })

        if self.config.include_vote_breakdown:
            votes = final_state.get("final_votes", {})
            if votes:
                vote_text = "\n".join(
                    f"{choice}: {count}" for choice, count in votes.items()
                )
                embed.fields.append({
                    "name": "Final Votes",
                    "value": self._truncate(vote_text, 1024),
                    "inline": True,
                })

        return await self._send_webhook([embed])

    async def send_error(
        self,
        error_type: str,
        message: str,
        debate_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Send error notification."""
        embed = DiscordEmbed(
            title=f"Error: {error_type}",
            description=self._truncate(message, self.config.max_summary_length),
            color=self.COLORS["error"],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        if debate_id:
            embed.footer = {"text": f"Debate: {debate_id[:8]}..."}

        if details:
            for key, value in list(details.items())[:5]:  # Max 5 detail fields
                embed.fields.append({
                    "name": key,
                    "value": self._truncate(str(value), 256),
                    "inline": True,
                })

        return await self._send_webhook([embed])

    async def send_round_summary(
        self,
        debate_id: str,
        round_number: int,
        total_rounds: int,
        summary: str,
        agent_positions: dict[str, str],
    ) -> bool:
        """Send round completion summary."""
        embed = DiscordEmbed(
            title=f"Round {round_number}/{total_rounds} Complete",
            description=self._truncate(summary, self.config.max_summary_length),
            color=self.COLORS["round_complete"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            footer={"text": f"Debate: {debate_id[:8]}..."},
        )

        if self.config.include_agent_details and agent_positions:
            positions_text = "\n".join(
                f"**{agent}:** {self._truncate(position, 100)}"
                for agent, position in list(agent_positions.items())[:5]
            )
            embed.fields.append({
                "name": "Agent Positions",
                "value": self._truncate(positions_text, 1024),
                "inline": False,
            })

        return await self._send_webhook([embed])


class DiscordWebhookManager:
    """Manager for multiple Discord webhook targets."""

    def __init__(self):
        self._integrations: dict[str, DiscordIntegration] = {}

    def register(self, name: str, config: DiscordConfig) -> None:
        """Register a Discord integration."""
        self._integrations[name] = DiscordIntegration(config)

    def unregister(self, name: str) -> None:
        """Unregister a Discord integration."""
        if name in self._integrations:
            del self._integrations[name]

    async def broadcast(
        self,
        method: str,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, bool]:
        """Broadcast to all registered integrations."""
        results: dict[str, bool] = {}

        for name, integration in self._integrations.items():
            handler = getattr(integration, method, None)
            if handler and callable(handler):
                try:
                    results[name] = await handler(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Discord broadcast to {name} failed: {e}")
                    results[name] = False

        return results

    async def close_all(self) -> None:
        """Close all integrations."""
        for integration in self._integrations.values():
            await integration.close()


# Global manager instance
discord_manager = DiscordWebhookManager()


def create_discord_integration(webhook_url: str, **kwargs: Any) -> DiscordIntegration:
    """Factory function to create a Discord integration."""
    config = DiscordConfig(webhook_url=webhook_url, **kwargs)
    return DiscordIntegration(config)
