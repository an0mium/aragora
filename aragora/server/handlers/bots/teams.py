"""
Microsoft Teams Bot endpoint handler.

Handles incoming messages and card actions from Microsoft Teams
via the Bot Framework messaging endpoint.

Endpoints:
- POST /api/v1/bots/teams/messages - Handle incoming Bot Framework activities
- GET /api/v1/bots/teams/status - Bot status

Environment Variables:
- TEAMS_APP_ID or MS_APP_ID - Required: Microsoft Bot Application ID for JWT validation
- TEAMS_APP_PASSWORD - Required for bot authentication
- TEAMS_TENANT_ID - Optional for single-tenant apps

JWT Validation (Phase 3.1):
  Incoming Bot Framework activities carry an Authorization: Bearer <JWT> header
  signed by Microsoft. The handler validates:
  - Signature via JWKS keys from Microsoft's OpenID configuration endpoint
  - Issuer (iss) against known Bot Framework issuers
  - Audience (aud) against the configured App ID
  - Token expiry (exp) and issued-at (iat) timestamps

Activity Types Handled:
- message: Regular messages and @mentions (commands)
- invoke: Adaptive Card actions (votes, button clicks)
- conversationUpdate: Bot added/removed from conversations
- messageReaction: Reactions added to messages
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.server.handlers.base import MaybeAsyncHandlerResult

from aragora.audit.unified import audit_data
from aragora.config import DEFAULT_AGENT_LIST, DEFAULT_ROUNDS
from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    json_response,
)
from aragora.server.handlers.bots.base import BotHandlerMixin
from aragora.server.handlers.secure import SecureHandler
from aragora.server.handlers.utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# Environment variables - None defaults make misconfiguration explicit
# MS_APP_ID serves as a fallback for TEAMS_APP_ID (common Microsoft convention)
TEAMS_APP_ID = os.environ.get("TEAMS_APP_ID") or os.environ.get("MS_APP_ID")
TEAMS_APP_PASSWORD = os.environ.get("TEAMS_APP_PASSWORD")
TEAMS_TENANT_ID = os.environ.get("TEAMS_TENANT_ID")

# Log warnings at module load time for missing secrets
if not TEAMS_APP_ID:
    logger.warning(
        "TEAMS_APP_ID (or MS_APP_ID) not configured - "
        "Bot Framework JWT validation will reject all requests"
    )
if not TEAMS_APP_PASSWORD:
    logger.warning("TEAMS_APP_PASSWORD not configured - Teams bot authentication disabled")

# Import extracted utility functions and shared state from teams_utils.
# These were split out to reduce teams.py from 2,088 LOC.
from aragora.server.handlers.bots.teams_utils import (
    _active_debates,
    _check_botframework_available,
    _check_connector_available,
    _conversation_references,
    _start_teams_debate,
    _store_conversation_reference,
    _user_votes,
    _verify_teams_token,
    build_consensus_card,
    build_debate_card,
    get_conversation_reference,
    get_debate_vote_counts,
)

# Agent display names for UI
AGENT_DISPLAY_NAMES: dict[str, str] = {
    "claude": "Claude",
    "gpt4": "GPT-4",
    "gemini": "Gemini",
    "mistral": "Mistral",
    "deepseek": "DeepSeek",
    "grok": "Grok",
    "qwen": "Qwen",
    "kimi": "Kimi",
    "anthropic-api": "Claude",
    "openai-api": "GPT-4",
}

# Command pattern for parsing @mentions
# Matches: @BotName command arguments
MENTION_PATTERN = re.compile(r"<at>.*?</at>\s*", re.IGNORECASE)

# Maximum length for card text fields to prevent API errors
MAX_CARD_TEXT_LENGTH = 500
MAX_TOPIC_DISPLAY_LENGTH = 200


class TeamsBot:
    """Microsoft Teams Bot for handling Bot Framework activities.

    Processes incoming activities from the Bot Framework Service and routes
    them to appropriate handlers. Supports:

    - Message activities: Regular messages and @mention commands
    - Invoke activities: Adaptive Card actions (votes, summaries, view details),
      compose extensions, and task module interactions
    - Conversation updates: Bot added/removed, member join/leave
    - Message reactions: Reaction added/removed tracking
    - Installation updates: App install/uninstall events

    The bot uses the TeamsConnector from ``aragora.connectors.chat.teams`` for
    sending replies and Adaptive Cards, and stores conversation references
    for proactive messaging support.
    """

    def __init__(self, app_id: str | None = None, app_password: str | None = None):
        """Initialize the Teams bot.

        Args:
            app_id: Bot application ID (defaults to TEAMS_APP_ID env var).
            app_password: Bot application password (defaults to TEAMS_APP_PASSWORD).
        """
        self.app_id = app_id or TEAMS_APP_ID or ""
        self.app_password = app_password or TEAMS_APP_PASSWORD or ""
        self._connector: Any | None = None

    async def _get_connector(self) -> Any:
        """Lazily get the Teams connector for sending messages."""
        if self._connector is None:
            try:
                from aragora.connectors.chat.teams import TeamsConnector

                self._connector = TeamsConnector(
                    app_id=self.app_id,
                    app_password=self.app_password,
                )
            except ImportError:
                logger.warning("Teams connector not available")
                return None
        return self._connector

    # =========================================================================
    # Activity entry point
    # =========================================================================

    async def process_activity(self, activity: dict[str, Any], auth_header: str) -> dict[str, Any]:
        """Process an incoming Bot Framework activity.

        This is the main entry point for all Teams bot interactions. It verifies
        the authentication token, stores the conversation reference for proactive
        messaging, and routes the activity to the appropriate handler.

        Args:
            activity: The Bot Framework activity payload.
            auth_header: Authorization header for token verification.

        Returns:
            Response dict (empty for most activities, invoke response for
            card actions).

        Raises:
            ValueError: If authentication token is invalid.
        """
        activity_type = activity.get("type", "")
        activity_id = activity.get("id", "")

        logger.debug(f"Processing Teams activity: type={activity_type}, id={activity_id}")

        # Verify token
        if self.app_id and not await _verify_teams_token(auth_header, self.app_id):
            logger.warning("Teams activity rejected - invalid token")
            raise ValueError("Invalid authentication token")

        # Store conversation reference for proactive messaging
        _store_conversation_reference(activity)

        # Route by activity type
        if activity_type == "message":
            return await self._handle_message(activity)
        elif activity_type == "invoke":
            return await self._handle_invoke(activity)
        elif activity_type == "conversationUpdate":
            return await self._handle_conversation_update(activity)
        elif activity_type == "messageReaction":
            return await self._handle_message_reaction(activity)
        elif activity_type == "installationUpdate":
            return await self._handle_installation_update(activity)
        else:
            logger.debug(f"Unhandled activity type: {activity_type}")
            return {}

    # =========================================================================
    # Message handling
    # =========================================================================

    async def _handle_message(self, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle incoming message activity.

        Distinguishes between @mention commands and regular messages.
        In 1:1 conversations (personal scope), all messages are treated as
        potential commands without requiring an @mention.
        """
        text = activity.get("text", "")
        conversation = activity.get("conversation", {})
        from_user = activity.get("from", {})
        service_url = activity.get("serviceUrl", "")

        conversation_id = conversation.get("id", "")
        conversation_type = conversation.get("conversationType", "")
        user_id = from_user.get("id", "")
        user_name = from_user.get("name", "unknown")

        logger.info(f"Teams message from {user_name}: {text[:100]}...")

        # Send typing indicator for better UX
        await self._send_typing(activity)

        # Check for @mention (command) in group chats
        entities = activity.get("entities", [])
        is_mention = any(e.get("type") == "mention" for e in entities)

        # In personal (1:1) conversations, treat all messages as commands
        is_personal = conversation_type == "personal"

        if is_mention or is_personal:
            # Extract command text - strip @mention markup if present
            if is_mention:
                clean_text = MENTION_PATTERN.sub("", text).strip()
            else:
                clean_text = text.strip()

            parts = clean_text.split(maxsplit=1)
            command = parts[0].lower() if parts else ""
            args = parts[1] if len(parts) > 1 else ""

            # In personal scope, if the first word is not a known command,
            # treat the entire message as a debate topic.
            known_commands = {
                "debate",
                "ask",
                "status",
                "help",
                "leaderboard",
                "agents",
                "vote",
            }
            if is_personal and command and command not in known_commands:
                args = clean_text
                command = "debate"

            return await self._handle_command(
                command=command,
                args=args,
                conversation_id=conversation_id,
                user_id=user_id,
                service_url=service_url,
                activity=activity,
            )
        else:
            # Regular message in group chat - prompt for @mention
            await self._send_reply(
                activity,
                "I received your message. Mention me with a command like "
                "'@Aragora debate <topic>' to start a debate.",
            )
            return {}

    # =========================================================================
    # Command routing and implementations
    # =========================================================================

    async def _handle_command(
        self,
        command: str,
        args: str,
        conversation_id: str,
        user_id: str,
        service_url: str,
        activity: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle a bot command from @mention or personal message."""
        logger.info(f"Teams command: {command} {args[:50]}...")
        thread_id = activity.get("replyToId")

        if command in ("debate", "ask"):
            return await self._cmd_debate(
                args, conversation_id, user_id, service_url, thread_id, activity
            )
        elif command == "status":
            return await self._cmd_status(activity)
        elif command == "help":
            return await self._cmd_help(activity)
        elif command == "leaderboard":
            return await self._cmd_leaderboard(activity)
        elif command == "agents":
            return await self._cmd_agents(activity)
        elif command == "vote":
            return await self._cmd_vote(args, activity)
        else:
            return await self._cmd_unknown(command, activity)

    async def _cmd_debate(
        self,
        topic: str,
        conversation_id: str,
        user_id: str,
        service_url: str,
        thread_id: str | None,
        activity: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle debate command - start a new multi-agent debate."""
        if not topic.strip():
            await self._send_reply(
                activity,
                "Please provide a topic. Example: @Aragora debate Should we use microservices?",
            )
            return {}

        # Start the debate
        debate_id = await _start_teams_debate(
            topic=topic,
            conversation_id=conversation_id,
            user_id=user_id,
            service_url=service_url,
            thread_id=thread_id,
        )

        # Build and send the debate card
        card = build_debate_card(
            debate_id=debate_id,
            topic=topic,
            agents=[AGENT_DISPLAY_NAMES.get(a, a) for a in DEFAULT_AGENT_LIST[:5]],
            current_round=1,
            total_rounds=DEFAULT_ROUNDS,
            include_vote_buttons=False,  # Voting enabled after round 1
        )

        await self._send_card(activity, card, f"Starting debate on: {topic[:100]}...")

        logger.info(f"Started debate {debate_id} from Teams user {user_id}")
        audit_data(
            user_id=f"teams:{user_id}",
            resource_type="debate",
            resource_id=debate_id,
            action="create",
            platform="teams",
            task_preview=topic[:100],
        )

        return {}

    async def _cmd_status(self, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle status command - show system status with Adaptive Card."""
        active_count = len(_active_debates)

        try:
            from aragora.ranking.elo import get_elo_store

            elo_store = get_elo_store()
            ratings = elo_store.get_all_ratings()
            agent_count = len(ratings)
        except (ImportError, AttributeError, RuntimeError):
            agent_count = 7

        card = {
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": [
                {
                    "type": "TextBlock",
                    "text": "Aragora Status: Online",
                    "weight": "Bolder",
                    "size": "Large",
                    "color": "Good",
                },
                {
                    "type": "FactSet",
                    "facts": [
                        {"title": "Active Debates", "value": str(active_count)},
                        {"title": "Registered Agents", "value": str(agent_count)},
                        {"title": "Integration", "value": "Connected"},
                        {"title": "Proactive Messaging", "value": "Enabled"},
                    ],
                },
                {
                    "type": "TextBlock",
                    "text": "All systems operational. Ready for debates.",
                    "isSubtle": True,
                    "size": "Small",
                },
            ],
            "actions": [
                {
                    "type": "Action.Submit",
                    "title": "Start Debate",
                    "data": {"action": "start_debate_prompt"},
                },
            ],
        }

        await self._send_card(
            activity, card, f"Aragora Status: Online - {active_count} active debates"
        )
        return {}

    async def _cmd_help(self, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle help command - show available commands with Adaptive Card."""
        try:
            from aragora.server.handlers.bots.teams_cards import create_help_card

            card = create_help_card()
            await self._send_card(
                activity, card, "Aragora Commands - use @Aragora help for details"
            )
            return {}
        except ImportError:
            pass

        help_text = """**Aragora Commands**

@Aragora **debate** <topic> - Start a multi-agent debate
@Aragora **ask** <topic> - Alias for debate
@Aragora **status** - Check system status
@Aragora **agents** - List available AI agents
@Aragora **leaderboard** - Show agent rankings
@Aragora **vote** - Cast votes on active debates
@Aragora **help** - Show this message

**Example:**
@Aragora debate Should we use microservices or a monolith?

Aragora orchestrates 15+ AI models to debate and deliver defensible decisions."""

        await self._send_reply(activity, help_text)
        return {}

    async def _cmd_leaderboard(self, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle leaderboard command - show agent ELO rankings."""
        standings: list[dict[str, Any]] = []

        try:
            from aragora.ranking.elo import get_elo_store

            elo_store = get_elo_store()
            ratings = elo_store.get_all_ratings()

            if ratings:
                sorted_ratings = sorted(ratings, key=lambda x: x.elo, reverse=True)
                standings = [
                    {
                        "name": r.agent_name,
                        "score": r.elo,
                        "wins": getattr(r, "wins", 0),
                        "debates": getattr(r, "total_debates", 0),
                    }
                    for r in sorted_ratings[:10]
                ]
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.debug(f"ELO store not available: {e}")

        if standings:
            try:
                from aragora.server.handlers.bots.teams_cards import create_leaderboard_card

                card = create_leaderboard_card(standings)
                await self._send_card(activity, card, "Agent Leaderboard")
                return {}
            except ImportError:
                pass

            lines = [f"{i + 1}. {s['name']}: {s['score']:.0f} ELO" for i, s in enumerate(standings)]
            leaderboard_text = "\n".join(lines)
        else:
            leaderboard_text = (
                "1. Claude: 1850 ELO\n"
                "2. GPT-4: 1820 ELO\n"
                "3. Gemini: 1780 ELO\n"
                "4. Grok: 1750 ELO\n"
                "5. Mistral: 1720 ELO"
            )

        await self._send_reply(activity, f"**Agent Leaderboard**\n\n{leaderboard_text}")
        return {}

    async def _cmd_agents(self, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle agents command - list available AI agents."""
        agents_text = """**Available AI Agents**

- **Claude** (Anthropic) - Advanced reasoning and analysis
- **GPT-4** (OpenAI) - Versatile language understanding
- **Gemini** (Google) - Multimodal capabilities
- **Grok** (xAI) - Real-time knowledge
- **Mistral** - Fast and efficient
- **DeepSeek** - Deep reasoning
- **Qwen** - Multilingual support
- **Kimi** - Extended context

Each debate uses a dynamic team selected based on task requirements and agent performance history."""

        await self._send_reply(activity, agents_text)
        return {}

    async def _cmd_vote(self, args: str, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle vote command - show voting cards or instructions."""
        if not args.strip():
            if _active_debates:
                for debate_id, info in list(_active_debates.items())[:3]:
                    topic = info.get("topic", "Unknown")
                    card = build_debate_card(
                        debate_id=debate_id,
                        topic=topic,
                        agents=[AGENT_DISPLAY_NAMES.get(a, a) for a in DEFAULT_AGENT_LIST[:5]],
                        current_round=info.get("current_round", 1),
                        total_rounds=info.get("total_rounds", DEFAULT_ROUNDS),
                        include_vote_buttons=True,
                    )
                    await self._send_card(activity, card, f"Vote on: {topic[:100]}")
            else:
                await self._send_reply(
                    activity,
                    "No active debates to vote on. "
                    "Use the vote buttons on debate cards to cast your vote.",
                )
        else:
            await self._send_reply(
                activity,
                "Use the vote buttons on active debate cards to cast your vote.",
            )
        return {}

    async def _cmd_unknown(self, command: str, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle unknown command."""
        await self._send_reply(
            activity,
            f"Unknown command: {command}\n\nUse @Aragora help to see available commands.",
        )
        return {}

    # =========================================================================
    # Invoke (Adaptive Card action) handling
    # =========================================================================

    async def _handle_invoke(self, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle invoke activity (Adaptive Card actions, compose extensions).

        Routes card action submits and compose extension requests to
        the appropriate handler.
        """
        invoke_name = activity.get("name", "")
        value = activity.get("value", {})
        from_user = activity.get("from", {})
        user_id = from_user.get("id", "")

        logger.info(f"Teams invoke: {invoke_name} from {user_id}")

        # Handle Adaptive Card action submit (most common)
        if invoke_name == "adaptiveCard/action" or not invoke_name:
            return await self._handle_card_action(value, user_id, activity)

        # Compose extension: submit action (messaging extension)
        if invoke_name == "composeExtension/submitAction":
            return await self._handle_compose_extension_submit(value, user_id, activity)

        # Compose extension: query (search in messaging extension)
        if invoke_name == "composeExtension/query":
            return await self._handle_compose_extension_query(value, user_id, activity)

        # Compose extension / task module: fetch task (open dialog)
        if invoke_name in ("composeExtension/fetchTask", "task/fetch"):
            return await self._handle_task_module_fetch(value, user_id, activity)

        # Task module: submit (dialog form submitted)
        if invoke_name == "task/submit":
            return await self._handle_task_module_submit(value, user_id, activity)

        # Messaging extension: link unfurling
        if invoke_name == "composeExtension/queryLink":
            return await self._handle_link_unfurling(value, activity)

        # Default invoke response (required for card actions)
        return {
            "status": 200,
            "body": {"statusCode": 200, "type": "message", "value": "Action processed"},
        }

    async def _handle_card_action(
        self, value: dict[str, Any], user_id: str, activity: dict[str, Any]
    ) -> dict[str, Any]:
        """Route Adaptive Card action submits to specific handlers."""
        action = value.get("action", "")

        if action == "vote":
            return await self._handle_vote(
                debate_id=value.get("debate_id", ""),
                agent=value.get("agent", value.get("value", "")),
                user_id=user_id,
                activity=activity,
            )
        elif action == "summary":
            return await self._handle_summary(
                debate_id=value.get("debate_id", ""),
                activity=activity,
            )
        elif action == "view_details":
            return await self._handle_view_details(
                debate_id=value.get("debate_id", ""),
                activity=activity,
            )
        elif action == "view_report":
            return await self._handle_view_report(
                debate_id=value.get("debate_id", ""),
                activity=activity,
            )
        elif action == "view_rankings":
            return await self._handle_view_rankings(
                period=value.get("period", "all_time"),
                activity=activity,
            )
        elif action == "watch":
            return await self._handle_watch_debate(
                debate_id=value.get("debate_id", ""),
                user_id=user_id,
                activity=activity,
            )
        elif action == "share":
            return await self._handle_share_result(
                debate_id=value.get("debate_id", ""),
                activity=activity,
            )
        elif action == "start_debate_prompt":
            return await self._handle_start_debate_prompt(activity)
        elif action == "help":
            await self._cmd_help(activity)
            return {
                "status": 200,
                "body": {"statusCode": 200, "type": "message", "value": "Help sent"},
            }

        logger.debug(f"Unhandled card action: {action}")
        return {
            "status": 200,
            "body": {"statusCode": 200, "type": "message", "value": "Action acknowledged"},
        }

    async def _handle_vote(
        self, debate_id: str, agent: str, user_id: str, activity: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle vote action from card."""
        if not debate_id or not agent:
            return {
                "status": 400,
                "body": {"statusCode": 400, "type": "error", "value": "Invalid vote data"},
            }

        if debate_id not in _user_votes:
            _user_votes[debate_id] = {}

        previous_vote = _user_votes[debate_id].get(user_id)
        _user_votes[debate_id][user_id] = agent

        logger.info(f"Vote recorded: {user_id} voted for {agent} in {debate_id}")

        audit_data(
            user_id=f"teams:{user_id}",
            resource_type="debate_vote",
            resource_id=debate_id,
            action="create",
            vote_option=agent,
            platform="teams",
        )

        if previous_vote and previous_vote != agent:
            message = f"Your vote changed from {previous_vote} to {agent}."
        else:
            message = f"Your vote for {agent} has been recorded!"

        vote_counts = get_debate_vote_counts(debate_id)
        total_votes = sum(vote_counts.values())

        return {
            "status": 200,
            "body": {
                "statusCode": 200,
                "type": "application/vnd.microsoft.card.adaptive",
                "value": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": [
                        {
                            "type": "TextBlock",
                            "text": message,
                            "weight": "Bolder",
                            "color": "Good",
                        },
                        {
                            "type": "TextBlock",
                            "text": f"Total votes cast: {total_votes}",
                            "isSubtle": True,
                            "size": "Small",
                        },
                    ],
                },
            },
        }

    async def _handle_summary(self, debate_id: str, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle summary request - show debate summary as an Adaptive Card."""
        debate_info = _active_debates.get(debate_id)

        if not debate_info:
            return {
                "status": 200,
                "body": {
                    "statusCode": 200,
                    "type": "message",
                    "value": f"Debate {debate_id[:8]}... not found or has completed.",
                },
            }

        topic = debate_info.get("topic", "Unknown")
        started = debate_info.get("started_at", 0)
        elapsed = time.time() - started if started else 0
        vote_counts = get_debate_vote_counts(debate_id)

        facts: list[dict[str, str]] = [
            {"title": "Topic", "value": topic[:100]},
            {"title": "Elapsed", "value": f"{elapsed / 60:.1f} minutes"},
            {"title": "Votes Cast", "value": str(sum(vote_counts.values()))},
        ]

        if vote_counts:
            top_agent = max(vote_counts, key=vote_counts.get)  # type: ignore[arg-type]
            facts.append(
                {
                    "title": "Leading Agent",
                    "value": f"{top_agent} ({vote_counts[top_agent]} votes)",
                }
            )

        return {
            "status": 200,
            "body": {
                "statusCode": 200,
                "type": "application/vnd.microsoft.card.adaptive",
                "value": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": [
                        {
                            "type": "TextBlock",
                            "text": "Debate Summary",
                            "weight": "Bolder",
                            "size": "Medium",
                        },
                        {"type": "FactSet", "facts": facts},
                    ],
                },
            },
        }

    async def _handle_view_details(
        self, debate_id: str, activity: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle view-details action - show detailed debate progress card."""
        debate_info = _active_debates.get(debate_id)

        if not debate_info:
            await self._send_reply(
                activity,
                f"Debate {debate_id[:8]}... not found. It may have completed.",
            )
            return {
                "status": 200,
                "body": {"statusCode": 200, "type": "message", "value": "Debate not found"},
            }

        topic = debate_info.get("topic", "Unknown")
        current_round = debate_info.get("current_round", 1)
        total_rounds = debate_info.get("total_rounds", DEFAULT_ROUNDS)

        try:
            from datetime import datetime

            from aragora.server.handlers.bots.teams_cards import create_debate_progress_card

            card = create_debate_progress_card(
                debate_id=debate_id,
                topic=topic,
                current_round=current_round,
                total_rounds=total_rounds,
                current_phase=debate_info.get("phase", "deliberation"),
                timestamp=datetime.now(),
            )
            await self._send_card(activity, card, f"Debate details: {topic[:80]}")
        except ImportError:
            started = debate_info.get("started_at", 0)
            elapsed = time.time() - started if started else 0
            await self._send_reply(
                activity,
                f"**Debate Details**\n\n"
                f"**Topic:** {topic[:200]}\n"
                f"**Round:** {current_round}/{total_rounds}\n"
                f"**Elapsed:** {elapsed / 60:.1f} minutes\n"
                f"**ID:** {debate_id[:8]}...",
            )

        return {
            "status": 200,
            "body": {"statusCode": 200, "type": "message", "value": "Details sent"},
        }

    async def _handle_view_report(self, debate_id: str, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle view-report action - provide link to full report."""
        await self._send_reply(
            activity,
            f"**Full Report**\n\n"
            f"View the complete debate report and audit trail:\n"
            f"[Open Report](https://aragora.ai/debate/{debate_id})",
        )
        return {
            "status": 200,
            "body": {"statusCode": 200, "type": "message", "value": "Report link sent"},
        }

    async def _handle_view_rankings(self, period: str, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle view-rankings action from leaderboard card."""
        await self._cmd_leaderboard(activity)
        return {
            "status": 200,
            "body": {"statusCode": 200, "type": "message", "value": "Rankings sent"},
        }

    async def _handle_watch_debate(
        self, debate_id: str, user_id: str, activity: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle watch action - subscribe user to live debate updates."""
        debate_info = _active_debates.get(debate_id)
        if not debate_info:
            return {
                "status": 200,
                "body": {
                    "statusCode": 200,
                    "type": "message",
                    "value": "Debate not found or already completed.",
                },
            }

        watchers: list[str] = debate_info.setdefault("watchers", [])
        if user_id not in watchers:
            watchers.append(user_id)

        logger.info(f"User {user_id} watching debate {debate_id}")

        return {
            "status": 200,
            "body": {
                "statusCode": 200,
                "type": "message",
                "value": "You will receive updates for this debate.",
            },
        }

    async def _handle_share_result(
        self, debate_id: str, activity: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle share action - re-post consensus card to channel."""
        debate_info = _active_debates.get(debate_id)
        if not debate_info:
            return {
                "status": 200,
                "body": {
                    "statusCode": 200,
                    "type": "message",
                    "value": "Debate result not available for sharing.",
                },
            }

        topic = debate_info.get("topic", "Unknown")
        vote_counts = get_debate_vote_counts(debate_id)

        card = build_consensus_card(
            debate_id=debate_id,
            topic=topic,
            consensus_reached=debate_info.get("consensus_reached", False),
            confidence=debate_info.get("confidence", 0.0),
            winner=debate_info.get("winner"),
            final_answer=debate_info.get("final_answer"),
            vote_counts=vote_counts,
        )
        await self._send_card(activity, card, f"Debate result: {topic[:100]}")

        return {
            "status": 200,
            "body": {"statusCode": 200, "type": "message", "value": "Result shared"},
        }

    async def _handle_start_debate_prompt(self, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle start-debate-prompt - return a task module for topic input."""
        return {
            "status": 200,
            "body": {
                "task": {
                    "type": "continue",
                    "value": {
                        "title": "Start a Debate",
                        "height": "medium",
                        "width": "medium",
                        "card": {
                            "contentType": "application/vnd.microsoft.card.adaptive",
                            "content": {
                                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                                "type": "AdaptiveCard",
                                "version": "1.4",
                                "body": [
                                    {
                                        "type": "TextBlock",
                                        "text": "What should the agents debate?",
                                        "weight": "Bolder",
                                    },
                                    {
                                        "type": "Input.Text",
                                        "id": "debate_topic",
                                        "placeholder": "Enter your debate topic...",
                                        "isMultiline": True,
                                        "maxLength": 1000,
                                    },
                                ],
                                "actions": [
                                    {
                                        "type": "Action.Submit",
                                        "title": "Start Debate",
                                        "data": {"action": "start_debate_from_task_module"},
                                    },
                                ],
                            },
                        },
                    },
                }
            },
        }

    # =========================================================================
    # Compose extension (messaging extension) handlers
    # =========================================================================

    async def _handle_compose_extension_submit(
        self, value: dict[str, Any], user_id: str, activity: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle compose extension submit action.

        Users can invoke this from the compose box to start a debate and
        insert the debate card into the conversation.
        """
        command_id = value.get("commandId", "")
        data = value.get("data", {})

        if command_id == "startDebate":
            topic = data.get("topic", data.get("debate_topic", ""))
            if not topic:
                return {
                    "status": 200,
                    "body": {
                        "composeExtension": {
                            "type": "message",
                            "text": "Please provide a debate topic.",
                        }
                    },
                }

            conversation = activity.get("conversation", {})
            service_url = activity.get("serviceUrl", "")

            debate_id = await _start_teams_debate(
                topic=topic,
                conversation_id=conversation.get("id", ""),
                user_id=user_id,
                service_url=service_url,
            )

            card = build_debate_card(
                debate_id=debate_id,
                topic=topic,
                agents=[AGENT_DISPLAY_NAMES.get(a, a) for a in DEFAULT_AGENT_LIST[:5]],
                current_round=1,
                total_rounds=DEFAULT_ROUNDS,
            )

            return {
                "status": 200,
                "body": {
                    "composeExtension": {
                        "type": "result",
                        "attachmentLayout": "list",
                        "attachments": [
                            {
                                "contentType": "application/vnd.microsoft.card.adaptive",
                                "content": card,
                                "preview": {
                                    "contentType": "application/vnd.microsoft.card.thumbnail",
                                    "content": {
                                        "title": f"Debate: {topic[:50]}",
                                        "text": f"ID: {debate_id[:8]}...",
                                    },
                                },
                            }
                        ],
                    }
                },
            }

        return {
            "status": 200,
            "body": {
                "composeExtension": {
                    "type": "message",
                    "text": "Command not recognized.",
                }
            },
        }

    async def _handle_compose_extension_query(
        self, value: dict[str, Any], user_id: str, activity: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle compose extension search query.

        Allows users to search for active debates from the compose box.
        """
        query_text = ""
        parameters = value.get("parameters", [])
        for param in parameters:
            if param.get("name") == "query":
                query_text = param.get("value", "")
                break

        results: list[dict[str, Any]] = []
        for debate_id, info in _active_debates.items():
            topic = info.get("topic", "")
            if query_text.lower() in topic.lower() or not query_text:
                card = build_debate_card(
                    debate_id=debate_id,
                    topic=topic,
                    agents=[AGENT_DISPLAY_NAMES.get(a, a) for a in DEFAULT_AGENT_LIST[:5]],
                    current_round=info.get("current_round", 1),
                    total_rounds=info.get("total_rounds", DEFAULT_ROUNDS),
                )

                results.append(
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": card,
                        "preview": {
                            "contentType": "application/vnd.microsoft.card.thumbnail",
                            "content": {
                                "title": f"Debate: {topic[:50]}",
                                "text": f"ID: {debate_id[:8]}...",
                            },
                        },
                    }
                )

                if len(results) >= 10:
                    break

        return {
            "status": 200,
            "body": {
                "composeExtension": {
                    "type": "result",
                    "attachmentLayout": "list",
                    "attachments": results,
                }
            },
        }

    async def _handle_task_module_fetch(
        self, value: dict[str, Any], user_id: str, activity: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle task module fetch - return a dialog for user input."""
        data = value.get("data", {})
        command_id = value.get("commandId", data.get("commandId", ""))

        if command_id == "startDebate" or data.get("action") == "start_debate_prompt":
            return await self._handle_start_debate_prompt(activity)

        return {
            "status": 200,
            "body": {
                "task": {
                    "type": "continue",
                    "value": {
                        "title": "Aragora",
                        "card": {
                            "contentType": "application/vnd.microsoft.card.adaptive",
                            "content": {
                                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                                "type": "AdaptiveCard",
                                "version": "1.4",
                                "body": [
                                    {
                                        "type": "TextBlock",
                                        "text": "Use @Aragora commands to interact.",
                                        "wrap": True,
                                    },
                                ],
                            },
                        },
                    },
                }
            },
        }

    async def _handle_task_module_submit(
        self, value: dict[str, Any], user_id: str, activity: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle task module form submission."""
        data = value.get("data", {})
        action = data.get("action", "")

        if action == "start_debate_from_task_module":
            topic = data.get("debate_topic", "")
            if topic:
                conversation = activity.get("conversation", {})
                service_url = activity.get("serviceUrl", "")

                debate_id = await _start_teams_debate(
                    topic=topic,
                    conversation_id=conversation.get("id", ""),
                    user_id=user_id,
                    service_url=service_url,
                )

                card = build_debate_card(
                    debate_id=debate_id,
                    topic=topic,
                    agents=[AGENT_DISPLAY_NAMES.get(a, a) for a in DEFAULT_AGENT_LIST[:5]],
                    current_round=1,
                    total_rounds=DEFAULT_ROUNDS,
                    include_vote_buttons=False,
                )
                await self._send_card(activity, card, f"Starting debate: {topic[:100]}")

                audit_data(
                    user_id=f"teams:{user_id}",
                    resource_type="debate",
                    resource_id=debate_id,
                    action="create",
                    platform="teams",
                    task_preview=topic[:100],
                )

        return {"status": 200, "body": {"task": {"type": "message", "value": "Done"}}}

    async def _handle_link_unfurling(
        self, value: dict[str, Any], activity: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle link unfurling for aragora.ai URLs."""
        url = value.get("url", "")

        debate_id_match = re.search(r"/debate/([a-f0-9-]+)", url)
        if debate_id_match:
            debate_id = debate_id_match.group(1)
            debate_info = _active_debates.get(debate_id)

            if debate_info:
                topic = debate_info.get("topic", "Unknown topic")
                card = build_debate_card(
                    debate_id=debate_id,
                    topic=topic,
                    agents=[AGENT_DISPLAY_NAMES.get(a, a) for a in DEFAULT_AGENT_LIST[:5]],
                    current_round=debate_info.get("current_round", 1),
                    total_rounds=debate_info.get("total_rounds", DEFAULT_ROUNDS),
                )

                return {
                    "status": 200,
                    "body": {
                        "composeExtension": {
                            "type": "result",
                            "attachmentLayout": "list",
                            "attachments": [
                                {
                                    "contentType": "application/vnd.microsoft.card.adaptive",
                                    "content": card,
                                    "preview": {
                                        "contentType": "application/vnd.microsoft.card.thumbnail",
                                        "content": {
                                            "title": f"Debate: {topic[:50]}",
                                            "text": f"ID: {debate_id[:8]}...",
                                        },
                                    },
                                }
                            ],
                        }
                    },
                }

        return {
            "status": 200,
            "body": {
                "composeExtension": {
                    "type": "result",
                    "attachmentLayout": "list",
                    "attachments": [],
                }
            },
        }

    # =========================================================================
    # Conversation update handling
    # =========================================================================

    async def _handle_conversation_update(self, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle conversation update (bot added/removed, members joined/left)."""
        members_added = activity.get("membersAdded", [])
        members_removed = activity.get("membersRemoved", [])

        for member in members_added:
            if member.get("id") == activity.get("recipient", {}).get("id"):
                logger.info("Bot added to Teams conversation")
                await self._send_welcome(activity)

        for member in members_removed:
            if member.get("id") == activity.get("recipient", {}).get("id"):
                logger.info("Bot removed from Teams conversation")
                conversation_id = activity.get("conversation", {}).get("id", "")
                _conversation_references.pop(conversation_id, None)

        return {}

    async def _handle_message_reaction(self, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle message reaction (reaction added/removed)."""
        reactions_added = activity.get("reactionsAdded", [])
        reactions_removed = activity.get("reactionsRemoved", [])

        from_user = activity.get("from", {})
        user_id = from_user.get("id", "")

        for reaction in reactions_added:
            reaction_type = reaction.get("type", "")
            logger.debug(f"Reaction added by {user_id}: {reaction_type}")

            if reaction_type in ("like", "heart"):
                reply_to = activity.get("replyToId", "")
                if reply_to:
                    logger.debug(f"Positive reaction on message {reply_to} from {user_id}")

        for reaction in reactions_removed:
            logger.debug(f"Reaction removed by {user_id}: {reaction.get('type')}")

        return {}

    async def _handle_installation_update(self, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle installation update (app installed/uninstalled)."""
        action = activity.get("action", "")
        logger.info(f"Teams installation update: {action}")

        conversation_id = activity.get("conversation", {}).get("id", "")

        if action == "add":
            audit_data(
                user_id="system",
                resource_type="teams_installation",
                resource_id=conversation_id,
                action="install",
                platform="teams",
            )
        elif action == "remove":
            audit_data(
                user_id="system",
                resource_type="teams_installation",
                resource_id=conversation_id,
                action="uninstall",
                platform="teams",
            )

        return {}

    # =========================================================================
    # Message sending utilities
    # =========================================================================

    async def _send_welcome(self, activity: dict[str, Any]) -> None:
        """Send welcome message when bot is added to a conversation."""
        try:
            from aragora.server.handlers.bots.teams_cards import create_help_card

            card = create_help_card()

            welcome_header: list[dict[str, Any]] = [
                {
                    "type": "TextBlock",
                    "text": "Welcome to Aragora!",
                    "weight": "Bolder",
                    "size": "ExtraLarge",
                },
                {
                    "type": "TextBlock",
                    "text": (
                        "I orchestrate 15+ AI models to debate and deliver "
                        "defensible decisions for your organization."
                    ),
                    "wrap": True,
                },
                {"type": "TextBlock", "text": "---", "separator": True},
            ]
            card["body"] = welcome_header + card.get("body", [])
            await self._send_card(activity, card, "Welcome to Aragora!")
            return
        except ImportError:
            pass

        welcome = """**Welcome to Aragora!**

I orchestrate 15+ AI models to debate and deliver defensible decisions for your organization.

**Quick Start:**
- @Aragora debate <topic> - Start a multi-agent debate
- @Aragora help - See all commands

Ready to make better decisions together!"""

        await self._send_reply(activity, welcome)

    async def _send_typing(self, activity: dict[str, Any]) -> None:
        """Send typing indicator to show the bot is processing.

        Best effort - failures are silently logged at debug level.
        """
        connector = await self._get_connector()
        if not connector:
            return

        try:
            conversation = activity.get("conversation", {})
            service_url = activity.get("serviceUrl", "")

            await connector.send_typing_indicator(
                channel_id=conversation.get("id", ""),
                service_url=service_url,
            )
        except (RuntimeError, OSError, ValueError, AttributeError) as e:
            logger.debug(f"Typing indicator failed (non-critical): {e}")

    async def _send_reply(self, activity: dict[str, Any], text: str) -> None:
        """Send a text reply to an activity."""
        connector = await self._get_connector()
        if not connector:
            logger.warning("Cannot send reply - connector not available")
            return

        try:
            conversation = activity.get("conversation", {})
            service_url = activity.get("serviceUrl", "")

            await connector.send_message(
                channel_id=conversation.get("id", ""),
                text=text,
                service_url=service_url,
                thread_id=activity.get("replyToId"),
            )
        except (RuntimeError, OSError, ValueError) as e:
            logger.error(f"Failed to send Teams reply: {e}")

    async def _send_card(
        self, activity: dict[str, Any], card: dict[str, Any], fallback_text: str
    ) -> None:
        """Send an Adaptive Card reply to an activity.

        Sends the full card structure (including actions) as an attachment.
        The fallback_text is used for notifications and accessibility.
        """
        connector = await self._get_connector()
        if not connector:
            logger.warning("Cannot send card - connector not available")
            return

        try:
            conversation = activity.get("conversation", {})
            service_url = activity.get("serviceUrl", "")
            conversation_id = conversation.get("id", "")

            # Build the activity payload with the full card as an attachment.
            # The connector.send_message treats blocks as card body elements,
            # but we need the complete card structure including actions.
            token = await connector._get_access_token()

            card_activity: dict[str, Any] = {
                "type": "message",
                "text": fallback_text,
                "attachments": [
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": card,
                    }
                ],
            }

            thread_id = activity.get("replyToId")
            if thread_id:
                card_activity["replyToId"] = thread_id

            await connector._http_request(
                method="POST",
                url=f"{service_url}/v3/conversations/{conversation_id}/activities",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=card_activity,
                operation="send_card",
            )

        except (RuntimeError, OSError, ValueError, AttributeError, KeyError) as e:
            logger.error(f"Failed to send Teams card: {e}")

    async def send_proactive_message(
        self,
        conversation_id: str,
        text: str | None = None,
        card: dict[str, Any] | None = None,
        fallback_text: str = "",
    ) -> bool:
        """Send a proactive message to a conversation.

        Uses stored conversation references to send messages outside of a
        direct reply context (e.g., debate results, notifications).

        Args:
            conversation_id: Target conversation ID.
            text: Plain text message (used if no card provided).
            card: Adaptive Card to send.
            fallback_text: Fallback text for card messages.

        Returns:
            True if the message was sent successfully.
        """
        ref = get_conversation_reference(conversation_id)
        if not ref:
            logger.warning(f"No conversation reference for {conversation_id}")
            return False

        connector = await self._get_connector()
        if not connector:
            logger.warning("Cannot send proactive message - connector not available")
            return False

        try:
            service_url = ref.get("service_url", "")

            if card:
                token = await connector._get_access_token()
                proactive_activity: dict[str, Any] = {
                    "type": "message",
                    "text": fallback_text or text or "",
                    "attachments": [
                        {
                            "contentType": "application/vnd.microsoft.card.adaptive",
                            "content": card,
                        }
                    ],
                }

                await connector._http_request(
                    method="POST",
                    url=f"{service_url}/v3/conversations/{conversation_id}/activities",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json=proactive_activity,
                    operation="proactive_card",
                )
                return True

            elif text:
                await connector.send_message(
                    channel_id=conversation_id,
                    text=text,
                    service_url=service_url,
                )
                return True

            else:
                logger.warning("No text or card provided for proactive message")
                return False

        except (RuntimeError, OSError, ValueError, AttributeError, KeyError) as e:
            logger.error(f"Failed to send proactive message: {e}")
            return False


class TeamsHandler(BotHandlerMixin, SecureHandler):
    """Handler for Microsoft Teams Bot endpoints.

    Uses BotHandlerMixin for shared auth/status patterns.

    RBAC Protected:
    - bots.read - required for status endpoint

    Note: Message webhook endpoints are authenticated via Bot Framework,
    not RBAC, since they are called by Microsoft servers directly.
    """

    # BotHandlerMixin configuration
    bot_platform = "teams"

    ROUTES = [
        "/api/v1/bots/teams/messages",
        "/api/v1/bots/teams/status",
    ]

    def __init__(self, ctx: dict | None = None):
        super().__init__(ctx or {})
        self._bot: TeamsBot | None = None
        self._bot_initialized = False

    def _is_bot_enabled(self) -> bool:
        """Check if Teams bot is configured."""
        return bool(TEAMS_APP_ID and TEAMS_APP_PASSWORD)

    def _get_platform_config_status(self) -> dict[str, Any]:
        """Return Teams-specific config fields for status response."""
        sdk_available, sdk_error = _check_botframework_available()
        connector_available, connector_error = _check_connector_available()

        return {
            "app_id_configured": bool(TEAMS_APP_ID),
            "password_configured": bool(TEAMS_APP_PASSWORD),
            "tenant_id_configured": bool(TEAMS_TENANT_ID),
            "sdk_available": sdk_available,
            "sdk_error": sdk_error,
            "connector_available": connector_available,
            "connector_error": connector_error,
            "active_debates": len(_active_debates),
            "conversation_references": len(_conversation_references),
            "features": {
                "adaptive_cards": True,
                "voting": True,
                "threading": True,
                "proactive_messaging": True,
                "compose_extensions": True,
                "task_modules": True,
                "link_unfurling": True,
            },
        }

    async def _ensure_bot(self) -> TeamsBot | None:
        """Lazily initialize the Teams bot."""
        if self._bot_initialized:
            return self._bot

        self._bot_initialized = True

        if not TEAMS_APP_ID or not TEAMS_APP_PASSWORD:
            logger.warning("Teams credentials not configured")
            return None

        self._bot = TeamsBot(app_id=TEAMS_APP_ID, app_password=TEAMS_APP_PASSWORD)
        logger.info("Teams bot initialized")
        return self._bot

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    @rate_limit(requests_per_minute=30, limiter_name="teams_status")
    async def handle(  # type: ignore[override]
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> "MaybeAsyncHandlerResult":
        """Route Teams requests with RBAC for status endpoint."""
        if path == "/api/v1/bots/teams/status":
            # Use BotHandlerMixin's RBAC-protected status handler
            return await self.handle_status_request(handler)

        return None

    @rate_limit(requests_per_minute=60, limiter_name="teams_messages")
    async def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle POST requests."""
        if path == "/api/v1/bots/teams/messages":
            return await self._handle_messages(handler)

        return None

    async def _handle_messages(self, handler: Any) -> HandlerResult:
        """Handle incoming Bot Framework messages.

        This endpoint receives activities from Microsoft Teams via the
        Bot Framework Service.
        """
        bot = await self._ensure_bot()
        if not bot:
            return json_response(
                {
                    "error": "Teams bot not configured",
                    "details": "Set TEAMS_APP_ID and TEAMS_APP_PASSWORD environment variables",
                },
                status=503,
            )

        try:
            # Read and parse body
            body = self._read_request_body(handler)
            activity, err = self._parse_json_body(body, "Teams message")
            if err:
                return err

            if not activity:
                return error_response("Empty activity", 400)

            # Get authorization header
            auth_header = handler.headers.get("Authorization", "")

            # Process the activity
            try:
                response = await bot.process_activity(activity, auth_header)

                # For invoke activities, return the response body
                if activity.get("type") == "invoke" and response:
                    status_code = response.get("status", 200)
                    return json_response(response.get("body", {}), status=status_code)

                return json_response({}, status=200)

            except ValueError as auth_error:
                logger.warning(f"Teams auth failed: {auth_error}")
                self._audit_webhook_auth_failure("auth_token", str(auth_error))
                return error_response("Unauthorized", 401)
            except (RuntimeError, OSError, AttributeError) as process_error:
                logger.exception(f"Teams activity processing error: {process_error}")
                return error_response("Internal processing error", 500)

        except (json.JSONDecodeError, ValueError, KeyError, TypeError, RuntimeError, OSError) as e:
            return self._handle_webhook_exception(e, "Teams message", return_200_on_error=False)


__all__ = [
    "TeamsHandler",
    "TeamsBot",
    "build_debate_card",
    "build_consensus_card",
    "get_debate_vote_counts",
    "get_conversation_reference",
]
