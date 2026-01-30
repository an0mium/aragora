"""
Slack Events API handlers.

Handles event callbacks including app_mention, message (DM), and URL verification.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from aragora.config import DEFAULT_ROUNDS

from .config import (
    SLACK_BOT_TOKEN,
    HandlerResult,
    auto_error_response,
    _get_audit_logger,
    create_tracked_task,
    json_response,
)
from .config import rate_limit
from .messaging import MessagingMixin

logger = logging.getLogger(__name__)


class EventsMixin(MessagingMixin):
    """Mixin providing Slack Events API handling."""

    @auto_error_response("handle slack events")
    @rate_limit(requests_per_minute=100, limiter_name="slack_events")
    def _handle_events(self, handler: Any) -> HandlerResult:
        """Handle Slack Events API callbacks.

        This handles events like app_mention, message, etc.
        """
        team_id = ""
        event_type = ""
        inner_type = ""
        user_id = ""
        channel_id = ""

        try:
            # Use pre-read body from handle()
            body = getattr(handler, "_slack_body", "")
            # Workspace context available for future use
            _workspace = getattr(handler, "_slack_workspace", None)  # noqa: F841
            team_id = getattr(handler, "_slack_team_id", None) or ""
            event = json.loads(body)

            event_type = event.get("type", "")

            # Handle URL verification challenge
            if event_type == "url_verification":
                challenge = event.get("challenge", "")
                return json_response({"challenge": challenge})

            # Handle event callbacks
            if event_type == "event_callback":
                inner_event = event.get("event", {})
                inner_type = inner_event.get("type", "")
                user_id = inner_event.get("user", "")
                channel_id = inner_event.get("channel", "")

                # Audit log the event
                audit = _get_audit_logger()
                if audit:
                    audit.log_event(
                        workspace_id=team_id,
                        event_type=inner_type,
                        payload_summary={"event_type": inner_type},
                        user_id=user_id,
                        channel_id=channel_id,
                        success=True,
                    )

                if inner_type == "app_mention":
                    return self._handle_app_mention(inner_event)
                elif inner_type == "message":
                    return self._handle_message_event(inner_event)

            # Acknowledge unknown events
            return json_response({"ok": True})

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in Slack event: {e}")
            self._audit_event_error(team_id, event_type or "unknown", str(e))
            return json_response({"ok": True})  # Always 200 for events
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Invalid event data: {e}")
            self._audit_event_error(team_id, event_type or inner_type or "unknown", str(e))
            return json_response({"ok": True})  # Always 200 for events
        except Exception as e:
            logger.exception(f"Unexpected events handler error: {e}")
            self._audit_event_error(team_id, event_type or inner_type or "unknown", str(e))
            return json_response({"ok": True})  # Always 200 for events

    def _audit_event_error(self, workspace_id: str, event_type: str, error: str) -> None:
        """Helper to audit log event errors."""
        audit = _get_audit_logger()
        if audit:
            audit.log_event(
                workspace_id=workspace_id,
                event_type=event_type,
                payload_summary={"error_type": "processing_error"},
                success=False,
                error=error[:200],
            )

    def _handle_app_mention(self, event: dict[str, Any]) -> HandlerResult:
        """Handle @mentions of the app."""
        text = event.get("text", "")
        channel = event.get("channel", "")
        user = event.get("user", "")

        logger.info(f"App mention from {user} in {channel}: {text[:50]}...")

        # Parse the mention to extract command/question
        # Remove the bot mention from the text
        clean_text = re.sub(r"<@[A-Z0-9]+>", "", text).strip()

        if not clean_text:
            # Just mentioned with no text - show help
            response_text = 'Hi! You can ask me to:\n\u2022 Debate a topic: `@aragora debate "Should AI be regulated?"`\n\u2022 Show status: `@aragora status`\n\u2022 List agents: `@aragora agents`'
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
            response_text = f"I don't understand: `{clean_text[:50]}`. Try `/aragora help` for available commands."

        # Post reply using Web API if bot token is available
        if SLACK_BOT_TOKEN:
            create_tracked_task(
                self._post_message_async(channel, response_text, thread_ts=event.get("ts")),
                name=f"slack-reply-{channel}",
            )

        return json_response({"ok": True})

    def _handle_message_event(self, event: dict[str, Any]) -> HandlerResult:
        """Handle direct messages to the app."""
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
        if not text:
            response_text = 'Hi! Send me a command:\n\u2022 `help` - Show available commands\n\u2022 `status` - Check system status\n\u2022 `agents` - List available agents\n\u2022 `debate "topic"` - Start a debate'
        elif text.lower() == "help":
            response_text = (
                "*Aragora Direct Message Commands*\n\n"
                "\u2022 `help` - Show this message\n"
                "\u2022 `status` - Check system status\n"
                "\u2022 `agents` - List available agents\n"
                '\u2022 `debate "Your topic here"` - Start a debate on a topic\n'
                "\u2022 `recent` - Show recent debates\n\n"
                "_You can also use `/aragora` commands in any channel._"
            )
        elif text.lower() == "status":
            try:
                from aragora.ranking.elo import EloSystem

                store = EloSystem()
                agents = store.get_all_ratings()
                response_text = f"*Aragora Status*\n\u2022 Status: Online\n\u2022 Agents: {len(agents)} registered"
            except (ImportError, AttributeError, RuntimeError) as e:
                logger.debug(f"Failed to fetch status: {e}")
                response_text = "*Aragora Status*\n\u2022 Status: Online\n\u2022 Agents: Unknown"
        elif text.lower() == "agents":
            try:
                from aragora.ranking.elo import EloSystem

                store = EloSystem()
                agents = store.get_all_ratings()
                if agents:
                    agents = sorted(agents, key=lambda a: getattr(a, "elo", 1500), reverse=True)
                    lines = ["*Top Agents*"]
                    for i, agent in enumerate(agents[:5]):
                        name = getattr(agent, "name", "Unknown")
                        elo = getattr(agent, "elo", 1500)
                        lines.append(f"{i + 1}. {name} (ELO: {elo:.0f})")
                    response_text = "\n".join(lines)
                else:
                    response_text = "No agents registered yet."
            except (ImportError, AttributeError, RuntimeError) as e:
                logger.debug(f"Failed to fetch agent list: {e}")
                response_text = "Could not fetch agent list."
        elif text.lower() == "recent":
            try:
                from aragora.server.storage import get_debates_db

                db = get_debates_db()
                if db and hasattr(db, "list"):
                    debates = db.list(limit=5)
                    if debates:
                        lines = ["*Recent Debates*"]
                        for d in debates:
                            topic = d.get("task", "Unknown")[:40]
                            consensus = "\u2705" if d.get("consensus_reached") else "\u274c"
                            lines.append(f"\u2022 {consensus} {topic}...")
                        response_text = "\n".join(lines)
                    else:
                        response_text = "No recent debates found."
                else:
                    response_text = "Debate history not available."
            except (ImportError, AttributeError, RuntimeError) as e:
                logger.debug(f"Failed to fetch recent debates: {e}")
                response_text = "Could not fetch recent debates."
        elif text.lower().startswith("debate "):
            topic = text[7:].strip().strip("\"'")
            if len(topic) < 10:
                response_text = "Topic is too short. Please provide more detail."
            elif len(topic) > 500:
                response_text = "Topic is too long. Please limit to 500 characters."
            else:
                response_text = f"Starting debate on: _{topic}_\n\n_This may take a few minutes..._"
                # Queue the debate creation
                if SLACK_BOT_TOKEN:
                    create_tracked_task(
                        self._create_dm_debate_async(topic, channel, user),
                        name=f"slack-dm-debate-{topic[:30]}",
                    )
        else:
            response_text = (
                f"I don't understand: `{text[:30]}`. Send `help` for available commands."
            )

        # Send response
        if SLACK_BOT_TOKEN:
            create_tracked_task(
                self._post_message_async(channel, response_text),
                name=f"slack-dm-response-{channel}",
            )

        return json_response({"ok": True})

    async def _create_dm_debate_async(
        self,
        topic: str,
        channel: str,
        user_id: str,
    ) -> None:
        """Create debate from DM and send result back to user."""
        try:
            from aragora import Arena, DebateProtocol, Environment
            from aragora.agents import get_agents_by_names

            env = Environment(task=f"Debate: {topic}")
            agents = get_agents_by_names(["anthropic-api", "openai-api"])
            protocol = DebateProtocol(
                rounds=DEFAULT_ROUNDS,
                consensus="majority",
                convergence_detection=False,
                early_stopping=False,
            )

            if not agents:
                await self._post_message_async(channel, "Failed: No agents available")
                return

            arena = Arena.from_env(env, agents, protocol)
            result = await arena.run()

            consensus_emoji = "\u2705" if result.consensus_reached else "\u26a0\ufe0f"
            response = (
                f"*Debate Complete!* {consensus_emoji}\n\n"
                f"*Topic:* {topic[:100]}...\n"
                f"*Consensus:* {'Yes' if result.consensus_reached else 'No'}\n"
                f"*Confidence:* {result.confidence:.1%}\n"
                f"*Rounds:* {result.rounds_used}\n\n"
                f"*Conclusion:*\n{result.final_answer[:500] if result.final_answer else 'No conclusion'}..."
            )
            await self._post_message_async(channel, response)

        except ImportError as e:
            logger.warning(f"Debate modules not available: {e}")
            await self._post_message_async(channel, "Debate service temporarily unavailable")
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Invalid debate request data: {e}")
            await self._post_message_async(channel, f"Invalid request: {str(e)[:100]}")
        except Exception as e:
            logger.exception(f"Unexpected DM debate creation error: {e}")
            await self._post_message_async(channel, f"Debate failed: {str(e)[:100]}")
