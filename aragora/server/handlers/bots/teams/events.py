"""
Teams Bot event processing.

Handles different Bot Framework activity types:
- message: Regular messages and @mentions (commands)
- invoke: Adaptive Card actions (delegated to cards.py)
- conversationUpdate: Bot added/removed from conversations
- messageReaction: Reactions added to messages
- installationUpdate: App install/uninstall events
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from aragora.audit.unified import audit_data
from aragora.config import DEFAULT_AGENT_LIST, DEFAULT_ROUNDS

from aragora.server.handlers.bots.teams_utils import (
    _active_debates,
    _conversation_references,
    _start_teams_debate,
    build_debate_card,
)

if TYPE_CHECKING:
    from aragora.server.handlers.bots.teams.handler import TeamsBot

logger = logging.getLogger(__name__)

# Import permission constants
PERM_TEAMS_MESSAGES_READ = "teams:messages:read"
PERM_TEAMS_DEBATES_CREATE = "teams:debates:create"

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
MENTION_PATTERN = re.compile(r"<at>.*?</at>\s*", re.IGNORECASE)


class TeamsEventProcessor:
    """Processes incoming Bot Framework activities.

    Routes activities to the appropriate handler based on activity type.
    """

    def __init__(self, bot: TeamsBot):
        """Initialize the event processor.

        Args:
            bot: The parent TeamsBot instance.
        """
        self.bot = bot

    async def process_activity(self, activity: dict[str, Any]) -> dict[str, Any]:
        """Process an incoming Bot Framework activity.

        Routes the activity to the appropriate handler based on type.

        Args:
            activity: The Bot Framework activity payload.

        Returns:
            Response dict (empty for most activities, invoke response for card actions).
        """
        activity_type = activity.get("type", "")

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
        """Handle incoming message activity."""
        # RBAC: Check permission to read/process messages
        perm_error = self.bot._check_permission(activity, PERM_TEAMS_MESSAGES_READ)
        if perm_error:
            await self.bot.send_reply(
                activity,
                "Sorry, you don't have permission to send commands to this bot.",
            )
            return {}

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
        await self.bot.send_typing(activity)

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
                "plan",
                "implement",
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
            await self.bot.send_reply(
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

        if command in ("debate", "ask", "plan", "implement"):
            decision_integrity = None
            if command in ("plan", "implement"):
                decision_integrity = {
                    "include_receipt": True,
                    "include_plan": True,
                    "include_context": command == "implement",
                    "plan_strategy": "single_task",
                    "notify_origin": True,
                }
                if command == "implement":
                    decision_integrity["execution_mode"] = "execute"
                    decision_integrity["execution_engine"] = "hybrid"
            return await self._cmd_debate(
                args,
                conversation_id,
                user_id,
                service_url,
                thread_id,
                activity,
                decision_integrity=decision_integrity,
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
        decision_integrity: dict[str, Any] | bool | None = None,
    ) -> dict[str, Any]:
        """Handle debate command - start a new multi-agent debate."""
        # RBAC: Check permission to create debates
        perm_error = self.bot._check_permission(activity, PERM_TEAMS_DEBATES_CREATE)
        if perm_error:
            await self.bot.send_reply(
                activity,
                "Sorry, you don't have permission to start debates. "
                "Please contact your administrator.",
            )
            return {}

        if not topic.strip():
            await self.bot.send_reply(
                activity,
                "Please provide a topic. Example: @Aragora debate Should we use microservices?",
            )
            return {}

        # Start the debate
        attachments = activity.get("attachments")
        if not isinstance(attachments, list):
            attachments = []

        debate_id = await _start_teams_debate(
            topic=topic,
            conversation_id=conversation_id,
            user_id=user_id,
            service_url=service_url,
            thread_id=thread_id,
            attachments=attachments,
            decision_integrity=decision_integrity,
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

        label = "debate"
        if decision_integrity:
            label = "implementation plan"
        await self.bot.send_card(activity, card, f"Starting {label} on: {topic[:100]}...")

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

        await self.bot.send_card(
            activity, card, f"Aragora Status: Online - {active_count} active debates"
        )
        return {}

    async def _cmd_help(self, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle help command - show available commands with Adaptive Card."""
        try:
            from aragora.server.handlers.bots.teams_cards import create_help_card

            card = create_help_card()
            await self.bot.send_card(
                activity, card, "Aragora Commands - use @Aragora help for details"
            )
            return {}
        except ImportError:
            pass

        help_text = """**Aragora Commands**

@Aragora **debate** <topic> - Start a multi-agent debate
@Aragora **ask** <topic> - Alias for debate
@Aragora **plan** <topic> - Debate + implementation plan
@Aragora **implement** <topic> - Debate + plan with context snapshot
@Aragora **status** - Check system status
@Aragora **agents** - List available AI agents
@Aragora **leaderboard** - Show agent rankings
@Aragora **vote** - Cast votes on active debates
@Aragora **help** - Show this message

**Example:**
@Aragora debate Should we use microservices or a monolith?

Aragora orchestrates 15+ AI models to debate and deliver defensible decisions."""

        await self.bot.send_reply(activity, help_text)
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
                await self.bot.send_card(activity, card, "Agent Leaderboard")
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

        await self.bot.send_reply(activity, f"**Agent Leaderboard**\n\n{leaderboard_text}")
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

        await self.bot.send_reply(activity, agents_text)
        return {}

    async def _cmd_vote(self, args: str, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle vote command - show voting cards or instructions."""
        # RBAC: Check permission to vote on debates
        from aragora.server.handlers.bots.teams.handler import PERM_TEAMS_DEBATES_VOTE

        perm_error = self.bot._check_permission(activity, PERM_TEAMS_DEBATES_VOTE)
        if perm_error:
            await self.bot.send_reply(
                activity,
                "Sorry, you don't have permission to vote on debates.",
            )
            return {}

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
                    await self.bot.send_card(activity, card, f"Vote on: {topic[:100]}")
            else:
                await self.bot.send_reply(
                    activity,
                    "No active debates to vote on. "
                    "Use the vote buttons on debate cards to cast your vote.",
                )
        else:
            await self.bot.send_reply(
                activity,
                "Use the vote buttons on active debate cards to cast your vote.",
            )
        return {}

    async def _cmd_unknown(self, command: str, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle unknown command."""
        await self.bot.send_reply(
            activity,
            f"Unknown command: {command}\n\nUse @Aragora help to see available commands.",
        )
        return {}

    # =========================================================================
    # Invoke (Adaptive Card action) handling
    # =========================================================================

    async def _handle_invoke(self, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle invoke activity (Adaptive Card actions, compose extensions).

        Delegates to the card actions handler for card-specific actions.
        """

        card_actions = self.bot._get_card_actions()
        return await card_actions.handle_invoke(activity)

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
    # Welcome message
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
            await self.bot.send_card(activity, card, "Welcome to Aragora!")
            return
        except ImportError:
            pass

        welcome = """**Welcome to Aragora!**

I orchestrate 15+ AI models to debate and deliver defensible decisions for your organization.

**Quick Start:**
- @Aragora debate <topic> - Start a multi-agent debate
- @Aragora help - See all commands

Ready to make better decisions together!"""

        await self.bot.send_reply(activity, welcome)


__all__ = [
    "TeamsEventProcessor",
    "AGENT_DISPLAY_NAMES",
    "MENTION_PATTERN",
]
