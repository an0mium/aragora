"""
Slack event handler implementations.

Provides mixin classes for handling Slack Events API callbacks.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from aragora.server.handlers.base import HandlerResult, json_response

logger = logging.getLogger(__name__)


class EventsMixin:
    """Mixin providing Slack event handling implementations.

    Should be mixed into a handler class that provides:
    - SLACK_BOT_TOKEN configuration
    - create_tracked_task utility
    - _post_message_async method
    """

    # Subclass should provide this
    SLACK_BOT_TOKEN: str = ""

    def handle_app_mention(self, event: Dict[str, Any]) -> HandlerResult:
        """Handle @mentions of the app.

        Args:
            event: Slack event payload

        Returns:
            HandlerResult acknowledging the event
        """
        from .._slack_impl import SLACK_BOT_TOKEN, create_tracked_task

        text = event.get("text", "")
        channel = event.get("channel", "")
        user = event.get("user", "")

        logger.info(f"App mention from {user} in {channel}: {text[:50]}...")

        # Parse the mention to extract command/question
        # Remove the bot mention from the text
        clean_text = re.sub(r"<@[A-Z0-9]+>", "", text).strip()

        if not clean_text:
            # Just mentioned with no text - show help
            response_text = (
                "Hi! You can ask me to:\n"
                '• Debate a topic: `@aragora debate "Should AI be regulated?"`\n'
                "• Show status: `@aragora status`\n"
                "• List agents: `@aragora agents`"
            )
        elif clean_text.lower().startswith("debate "):
            topic = clean_text[7:].strip().strip("\"'")
            response_text = f'To start a debate, use the slash command: `/aragora debate "{topic}"`'
        elif clean_text.lower() == "status":
            response_text = "Use `/aragora status` to check the system status."
        elif clean_text.lower() == "agents":
            response_text = "Use `/aragora agents` to list available agents."
        elif clean_text.lower() == "help":
            response_text = "Use `/aragora help` for available commands."
        else:
            response_text = (
                f"I don't understand: `{clean_text[:50]}`. "
                "Try `/aragora help` for available commands."
            )

        # Post reply using Web API if bot token is available
        if SLACK_BOT_TOKEN:
            create_tracked_task(
                self._post_message_async(channel, response_text, thread_ts=event.get("ts")),
                name=f"slack-reply-{channel}",
            )

        return json_response({"ok": True})

    async def _post_message_async(
        self,
        channel: str,
        text: str,
        thread_ts: Optional[str] = None,
        blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """Post a message to Slack using the Web API.

        Args:
            channel: Channel ID to post to
            text: Message text
            thread_ts: Optional thread timestamp to reply to
            blocks: Optional Block Kit blocks for rich formatting

        Returns:
            Message timestamp (ts) if successful, None otherwise
        """
        import aiohttp

        from .._slack_impl import SLACK_BOT_TOKEN

        if not SLACK_BOT_TOKEN:
            logger.warning("Cannot post message: SLACK_BOT_TOKEN not configured")
            return None

        try:
            payload: Dict[str, Any] = {
                "channel": channel,
                "text": text,
            }
            if thread_ts:
                payload["thread_ts"] = thread_ts
            if blocks:
                payload["blocks"] = blocks

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://slack.com/api/chat.postMessage",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
                        "Content-Type": "application/json",
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    result = await response.json()
                    if not result.get("ok"):
                        logger.warning(f"Slack API error: {result.get('error')}")
                        return None
                    # Return message timestamp for thread tracking
                    return result.get("ts")
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Connection error posting Slack message: {e}")
            return None
        except Exception as e:
            logger.exception(f"Unexpected error posting Slack message: {e}")
            return None

    def handle_message_event(self, event: Dict[str, Any]) -> HandlerResult:
        """Handle direct messages to the app.

        Args:
            event: Slack event payload

        Returns:
            HandlerResult acknowledging the event
        """
        from .._slack_impl import SLACK_BOT_TOKEN, create_tracked_task

        # Only handle DMs (channel type is "im")
        channel_type = event.get("channel_type")
        if channel_type != "im":
            return json_response({"ok": True})

        # Ignore bot messages to prevent loops
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return json_response({"ok": True})

        text = event.get("text", "").strip()
        user = event.get("user", "")
        channel = event.get("channel", "")

        logger.info(f"DM from {user}: {text[:50]}...")

        # Parse DM commands
        response_text = self._parse_dm_command(text)

        # Post reply
        if SLACK_BOT_TOKEN:
            create_tracked_task(
                self._post_message_async(channel, response_text),
                name=f"slack-dm-reply-{user}",
            )

        return json_response({"ok": True})

    def _parse_dm_command(self, text: str) -> str:
        """Parse a DM command and return the response text.

        Args:
            text: User's message text

        Returns:
            Response text to send back
        """
        if not text:
            return (
                "Hi! Send me a command:\n"
                "• `help` - Show available commands\n"
                "• `status` - Check system status\n"
                "• `agents` - List available agents\n"
                '• `debate "topic"` - Start a debate'
            )

        text_lower = text.lower()

        if text_lower == "help":
            return (
                "*Aragora Direct Message Commands*\n\n"
                "• `help` - Show this message\n"
                "• `status` - Check system status\n"
                "• `agents` - List available agents\n"
                '• `debate "Your topic here"` - Start a debate on a topic\n'
                "• `recent` - Show recent debates\n\n"
                "_You can also use `/aragora` commands in any channel._"
            )

        if text_lower == "status":
            return self._get_status_response()

        if text_lower == "agents":
            return self._get_agents_response()

        if text_lower.startswith("debate "):
            topic = text[7:].strip().strip("\"'")
            return (
                f'To start a full debate on "{topic}", '
                f'use `/aragora debate "{topic}"` in any channel.\n\n'
                "_Debates require slash commands for proper formatting._"
            )

        # Unknown command
        return f"I don't understand: `{text[:50]}`\n\nSend `help` for available commands."

    def _get_status_response(self) -> str:
        """Get status response for DM."""
        try:
            from aragora.ranking.elo import EloSystem

            store = EloSystem()
            agents = store.get_all_ratings()
            return f"*Aragora Status*\n• Status: Online\n• Agents: {len(agents)} registered"
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.debug(f"Failed to fetch status: {e}")
            return "*Aragora Status*\n• Status: Online\n• Agents: Unknown"

    def _get_agents_response(self) -> str:
        """Get agents list response for DM."""
        try:
            from aragora.ranking.elo import EloSystem

            store = EloSystem()
            agents = store.get_all_ratings()
            if agents:
                agents = sorted(agents, key=lambda a: getattr(a, "rating", 1500), reverse=True)
                lines = ["*Top Agents*"]
                for i, agent in enumerate(agents[:5]):
                    name = getattr(agent, "agent_id", "Unknown")
                    elo = getattr(agent, "rating", 1500)
                    lines.append(f"{i + 1}. {name} (ELO: {elo:.0f})")
                return "\n".join(lines)
            else:
                return "No agents registered yet."
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.debug(f"Failed to fetch agents: {e}")
            return "Agent list temporarily unavailable."


__all__ = ["EventsMixin"]
