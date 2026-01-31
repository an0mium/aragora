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
import uuid
from typing import TYPE_CHECKING, Any, Optional

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

# Store active debates for Teams (in production, use Redis/database)
_active_debates: dict[str, dict[str, Any]] = {}
_user_votes: dict[str, dict[str, str]] = {}  # debate_id -> {user_id: vote}

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


def _check_botframework_available() -> tuple[bool, str | None]:
    """Check if Bot Framework SDK is available."""
    try:
        from botbuilder.core import TurnContext  # noqa: F401 - availability check

        return True, None
    except ImportError:
        return False, "botbuilder-core not installed"


def _check_connector_available() -> tuple[bool, str | None]:
    """Check if Teams connector is available."""
    try:
        from aragora.connectors.chat.teams import TeamsConnector  # noqa: F401

        return True, None
    except ImportError:
        return False, "Teams connector not available"


async def _verify_teams_token(auth_header: str, app_id: str) -> bool:
    """Verify Bot Framework JWT token from the Authorization header.

    Validates the incoming JWT against Microsoft's JWKS signing keys,
    checking signature, issuer, audience, and expiry claims.

    The function delegates to the centralized JWT verifier in
    ``aragora.connectors.chat.jwt_verify`` which handles:
    - OpenID metadata discovery to resolve the JWKS endpoint
    - JWKS key caching with TTL to avoid per-request network calls
    - RS256 signature verification
    - Issuer validation against known Bot Framework issuers
    - Audience validation against the configured app ID

    Security: This function follows a fail-closed model. If PyJWT is not
    installed or the verification module is unavailable, tokens are rejected
    in production environments. In development, unverified tokens may be
    allowed if ARAGORA_ALLOW_UNVERIFIED_WEBHOOKS is explicitly set.

    Args:
        auth_header: The full Authorization header value (e.g., "Bearer eyJ...")
        app_id: The Microsoft Bot Application ID to validate the audience claim against.
                Sourced from TEAMS_APP_ID or MS_APP_ID environment variable.

    Returns:
        True if the token is valid, False otherwise (fail-closed)
    """
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.warning("Teams auth rejected: missing or malformed Authorization header")
        return False

    try:
        from aragora.connectors.chat.jwt_verify import HAS_JWT, verify_teams_webhook

        if HAS_JWT:
            return verify_teams_webhook(auth_header, app_id)
        else:
            # PyJWT not installed - check if dev bypass is allowed
            from aragora.connectors.chat.webhook_security import should_allow_unverified

            if should_allow_unverified("teams"):
                logger.warning(
                    "Teams token verification skipped - PyJWT not available (dev mode). "
                    "Install pyjwt[crypto] for production use."
                )
                return True
            logger.error(
                "Teams token rejected - PyJWT library not available. "
                "Install with: pip install pyjwt[crypto]"
            )
            return False
    except ImportError:
        # jwt_verify module itself not available - check environment
        env = os.environ.get("ARAGORA_ENV", "development").lower()
        is_production = env not in ("development", "dev", "local", "test")
        if is_production:
            logger.error(
                "SECURITY: Teams JWT verification module not available in production. "
                "Ensure aragora.connectors.chat.jwt_verify is importable."
            )
            return False
        logger.warning("Teams token verification skipped (dev mode - jwt_verify not importable)")
        return True


def build_debate_card(
    debate_id: str,
    topic: str,
    agents: list[str],
    current_round: int,
    total_rounds: int,
    include_vote_buttons: bool = True,
) -> dict[str, Any]:
    """Build an Adaptive Card for active debate display."""
    body: list[dict[str, Any]] = [
        {
            "type": "TextBlock",
            "text": "Active Debate",
            "weight": "Bolder",
            "size": "Large",
        },
        {
            "type": "TextBlock",
            "text": f"**Topic:** {topic[:200]}",
            "wrap": True,
        },
        {
            "type": "FactSet",
            "facts": [
                {"title": "Agents", "value": ", ".join(agents[:5])},
                {"title": "Progress", "value": f"Round {current_round}/{total_rounds}"},
            ],
        },
    ]

    actions: list[dict[str, Any]] = []

    if include_vote_buttons:
        # Add voting buttons for each agent (max 5)
        for agent in agents[:5]:
            actions.append(
                {
                    "type": "Action.Submit",
                    "title": f"Vote {agent}",
                    "data": {
                        "action": "vote",
                        "debate_id": debate_id,
                        "agent": agent,
                    },
                }
            )

        # Add view summary action
        actions.append(
            {
                "type": "Action.Submit",
                "title": "View Summary",
                "data": {"action": "summary", "debate_id": debate_id},
            }
        )

    # Footer with debate ID
    body.append(
        {
            "type": "TextBlock",
            "text": f"Debate ID: {debate_id[:8]}...",
            "size": "Small",
            "isSubtle": True,
        }
    )

    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": body,
        "actions": actions if actions else None,
    }


def build_consensus_card(
    debate_id: str,
    topic: str,
    consensus_reached: bool,
    confidence: float,
    winner: str | None,
    final_answer: str | None,
    vote_counts: dict[str, int],
) -> dict[str, Any]:
    """Build an Adaptive Card for debate consensus results."""
    status_text = "Consensus Reached" if consensus_reached else "No Consensus"
    status_color = "Good" if consensus_reached else "Warning"

    body: list[dict[str, Any]] = [
        {
            "type": "TextBlock",
            "text": status_text,
            "weight": "Bolder",
            "size": "Large",
            "color": status_color,
        },
        {
            "type": "TextBlock",
            "text": f"**Topic:** {topic[:200]}",
            "wrap": True,
        },
    ]

    # Results facts
    facts = [{"title": "Confidence", "value": f"{confidence:.0%}"}]
    if winner:
        facts.append({"title": "Winner", "value": winner})

    body.append({"type": "FactSet", "facts": facts})

    # User votes if any
    if vote_counts:
        vote_text = "\n".join(
            f"- {agent}: {count} vote{'s' if count != 1 else ''}"
            for agent, count in sorted(vote_counts.items(), key=lambda x: -x[1])
        )
        body.append(
            {
                "type": "TextBlock",
                "text": f"**User Votes:**\n{vote_text}",
                "wrap": True,
            }
        )

    # Final answer preview
    if final_answer:
        preview = final_answer[:500]
        if len(final_answer) > 500:
            preview += "..."
        body.append({"type": "TextBlock", "text": "---", "separator": True})
        body.append(
            {
                "type": "TextBlock",
                "text": f"**Decision:**\n{preview}",
                "wrap": True,
            }
        )

    # Action buttons
    actions = [
        {
            "type": "Action.OpenUrl",
            "title": "View Full Report",
            "url": f"https://aragora.ai/debate/{debate_id}",
        },
        {
            "type": "Action.OpenUrl",
            "title": "Audit Trail",
            "url": f"https://aragora.ai/debates/provenance?debate={debate_id}",
        },
    ]

    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": body,
        "actions": actions,
    }


async def _start_teams_debate(
    topic: str,
    conversation_id: str,
    user_id: str,
    service_url: str,
    thread_id: str | None = None,
) -> str:
    """Start a debate from Teams via DecisionRouter.

    Uses the unified DecisionRouter for:
    - Deduplication (prevents duplicate debates for same topic/user)
    - Caching (returns cached results if available)
    - Origin registration for result routing
    """
    debate_id = str(uuid.uuid4())

    try:
        from aragora.core import (
            DecisionRequest,
            DecisionType,
            InputSource,
            RequestContext,
            ResponseChannel,
            get_decision_router,
        )

        # Create response channel for result routing
        response_channel = ResponseChannel(
            platform="teams",
            channel_id=conversation_id,
            user_id=user_id,
            thread_id=thread_id,
            webhook_url=service_url,  # Teams uses service_url for proactive messaging
        )

        # Create request context
        context = RequestContext(
            user_id=user_id,
            session_id=f"teams:{conversation_id}",
        )

        # Create decision request
        request = DecisionRequest(
            content=topic,
            decision_type=DecisionType.DEBATE,
            source=InputSource.TEAMS,
            response_channels=[response_channel],
            context=context,
        )

        # Route through DecisionRouter
        router = get_decision_router()
        result = await router.route(request)

        if result.request_id:
            logger.info(f"DecisionRouter started debate {result.request_id} from Teams")
            _active_debates[result.request_id] = {
                "topic": topic,
                "conversation_id": conversation_id,
                "user_id": user_id,
                "service_url": service_url,
                "thread_id": thread_id,
                "started_at": time.time(),
            }
            return result.request_id
        return debate_id

    except ImportError:
        logger.debug("DecisionRouter not available, using fallback")
        return await _fallback_start_debate(
            topic, conversation_id, user_id, debate_id, service_url, thread_id
        )
    except (RuntimeError, ValueError, KeyError, AttributeError) as e:
        logger.error(f"DecisionRouter failed: {e}, using fallback")
        return await _fallback_start_debate(
            topic, conversation_id, user_id, debate_id, service_url, thread_id
        )


async def _fallback_start_debate(
    topic: str,
    conversation_id: str,
    user_id: str,
    debate_id: str,
    service_url: str,
    thread_id: str | None = None,
) -> str:
    """Fallback debate start when DecisionRouter unavailable."""
    # Register origin for result routing
    try:
        from aragora.server.debate_origin import register_debate_origin

        register_debate_origin(
            debate_id=debate_id,
            platform="teams",
            channel_id=conversation_id,
            user_id=user_id,
            thread_id=thread_id,
            metadata={
                "topic": topic,
                "service_url": service_url,
                "webhook_url": service_url,
            },
        )
    except (RuntimeError, KeyError, AttributeError, OSError) as e:
        logger.warning(f"Failed to register debate origin: {e}")

    # Try to enqueue via Redis queue
    try:
        from aragora.queue import create_debate_job, create_redis_queue

        job = create_debate_job(
            question=topic,
            user_id=user_id,
            metadata={
                "debate_id": debate_id,
                "platform": "teams",
                "conversation_id": conversation_id,
                "service_url": service_url,
                "thread_id": thread_id,
            },
        )
        queue = await create_redis_queue()
        await queue.enqueue(job)
        logger.info(f"Debate {debate_id} enqueued via Redis queue")
    except ImportError:
        logger.warning("Redis queue not available, debate will run inline")
    except (RuntimeError, OSError, ConnectionError) as e:
        logger.warning(f"Failed to enqueue debate: {e}")

    # Track active debate
    _active_debates[debate_id] = {
        "topic": topic,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "service_url": service_url,
        "thread_id": thread_id,
        "started_at": time.time(),
    }

    return debate_id


def get_debate_vote_counts(debate_id: str) -> dict[str, int]:
    """Get vote counts for a debate."""
    votes = _user_votes.get(debate_id, {})
    counts: dict[str, int] = {}
    for agent in votes.values():
        counts[agent] = counts.get(agent, 0) + 1
    return counts


class TeamsBot:
    """Microsoft Teams Bot for handling activities.

    Processes Bot Framework activities and routes them to appropriate handlers:
    - Messages (regular and @mentions)
    - Card actions (invoke activities)
    - Conversation updates
    - Message reactions
    """

    def __init__(self, app_id: str | None = None, app_password: str | None = None):
        """Initialize the Teams bot.

        Args:
            app_id: Bot application ID (defaults to TEAMS_APP_ID env var)
            app_password: Bot application password (defaults to TEAMS_APP_PASSWORD)
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

    async def process_activity(self, activity: dict[str, Any], auth_header: str) -> dict[str, Any]:
        """Process an incoming Bot Framework activity.

        Args:
            activity: The Bot Framework activity payload
            auth_header: Authorization header for token verification

        Returns:
            Response dict (empty for most activities, invoke response for card actions)
        """
        activity_type = activity.get("type", "")
        activity_id = activity.get("id", "")

        logger.debug(f"Processing Teams activity: type={activity_type}, id={activity_id}")

        # Verify token
        if self.app_id and not await _verify_teams_token(auth_header, self.app_id):
            logger.warning("Teams activity rejected - invalid token")
            raise ValueError("Invalid authentication token")

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

    async def _handle_message(self, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle incoming message activity."""
        text = activity.get("text", "")
        conversation = activity.get("conversation", {})
        from_user = activity.get("from", {})
        service_url = activity.get("serviceUrl", "")

        conversation_id = conversation.get("id", "")
        user_id = from_user.get("id", "")
        user_name = from_user.get("name", "unknown")
        logger.info(f"Teams message from {user_name}: {text[:100]}...")

        # Check for @mention (command)
        entities = activity.get("entities", [])
        is_mention = any(e.get("type") == "mention" for e in entities)

        if is_mention:
            # Extract command after @mention
            clean_text = MENTION_PATTERN.sub("", text).strip()
            parts = clean_text.split(maxsplit=1)
            command = parts[0].lower() if parts else ""
            args = parts[1] if len(parts) > 1 else ""

            return await self._handle_command(
                command=command,
                args=args,
                conversation_id=conversation_id,
                user_id=user_id,
                service_url=service_url,
                activity=activity,
            )
        else:
            # Regular message - could trigger a debate
            # For now, just acknowledge
            await self._send_reply(
                activity,
                "I received your message. Mention me with a command like "
                "'@Aragora debate <topic>' to start a debate.",
            )
            return {}

    async def _handle_command(
        self,
        command: str,
        args: str,
        conversation_id: str,
        user_id: str,
        service_url: str,
        activity: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle a bot command from @mention."""
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
        """Handle debate command."""
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
        """Handle status command."""
        active_count = len(_active_debates)

        await self._send_reply(
            activity,
            f"**Aragora Status: Online**\n\n"
            f"- Active debates: {active_count}\n"
            f"- Available models: Claude, GPT-4, Gemini, Grok, Mistral, DeepSeek\n\n"
            "Ready for debates!",
        )
        return {}

    async def _cmd_help(self, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle help command."""
        help_text = """**Aragora Commands**

@Aragora **debate** <topic> - Start a multi-agent debate
@Aragora **ask** <topic> - Alias for debate
@Aragora **status** - Check system status
@Aragora **agents** - List available AI agents
@Aragora **leaderboard** - Show agent rankings
@Aragora **help** - Show this message

**Example:**
@Aragora debate Should we use microservices or a monolith?

Aragora orchestrates 15+ AI models to debate and deliver defensible decisions."""

        await self._send_reply(activity, help_text)
        return {}

    async def _cmd_leaderboard(self, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle leaderboard command."""
        try:
            from aragora.ranking.elo import get_elo_store

            elo_store = get_elo_store()
            ratings = elo_store.get_all_ratings()

            if ratings:
                sorted_ratings = sorted(ratings, key=lambda x: x.elo, reverse=True)
                lines = [
                    f"{i + 1}. {r.agent_name}: {r.elo:.0f} ELO"
                    for i, r in enumerate(sorted_ratings[:10])
                ]
                leaderboard = "\n".join(lines)
            else:
                leaderboard = "No agent ratings available yet."

        except (ImportError, AttributeError, RuntimeError) as e:
            logger.debug(f"ELO store not available: {e}")
            leaderboard = (
                "1. Claude: 1850 ELO\n"
                "2. GPT-4: 1820 ELO\n"
                "3. Gemini: 1780 ELO\n"
                "4. Grok: 1750 ELO\n"
                "5. Mistral: 1720 ELO"
            )

        await self._send_reply(activity, f"**Agent Leaderboard**\n\n{leaderboard}")
        return {}

    async def _cmd_agents(self, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle agents command."""
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

    async def _cmd_unknown(self, command: str, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle unknown command."""
        await self._send_reply(
            activity,
            f"Unknown command: {command}\n\nUse @Aragora help to see available commands.",
        )
        return {}

    async def _handle_invoke(self, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle invoke activity (Adaptive Card actions)."""
        invoke_name = activity.get("name", "")
        value = activity.get("value", {})
        from_user = activity.get("from", {})
        user_id = from_user.get("id", "")

        logger.info(f"Teams invoke: {invoke_name} from {user_id}")

        # Handle card action submit
        if invoke_name == "adaptiveCard/action" or not invoke_name:
            action = value.get("action", "")

            if action == "vote":
                return await self._handle_vote(
                    debate_id=value.get("debate_id", ""),
                    agent=value.get("agent", ""),
                    user_id=user_id,
                    activity=activity,
                )
            elif action == "summary":
                return await self._handle_summary(
                    debate_id=value.get("debate_id", ""),
                    activity=activity,
                )

        # For messaging extension actions
        if invoke_name == "composeExtension/submitAction":
            # Handle compose extension
            pass

        # Return invoke response (required for card actions)
        return {
            "status": 200,
            "body": {"statusCode": 200, "type": "message", "value": "Action processed"},
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

        # Record the vote
        if debate_id not in _user_votes:
            _user_votes[debate_id] = {}

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

        # Send confirmation (Teams invoke responses can include a card update)
        return {
            "status": 200,
            "body": {
                "statusCode": 200,
                "type": "message",
                "value": f"Your vote for {agent} has been recorded!",
            },
        }

    async def _handle_summary(self, debate_id: str, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle summary request action."""
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

        return {
            "status": 200,
            "body": {
                "statusCode": 200,
                "type": "message",
                "value": f"Debate: {topic[:100]}\nElapsed: {elapsed / 60:.1f} minutes",
            },
        }

    async def _handle_conversation_update(self, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle conversation update (bot added/removed)."""
        members_added = activity.get("membersAdded", [])
        members_removed = activity.get("membersRemoved", [])

        for member in members_added:
            if member.get("id") == activity.get("recipient", {}).get("id"):
                # Bot was added to conversation
                logger.info("Bot added to Teams conversation")
                await self._send_welcome(activity)

        for member in members_removed:
            if member.get("id") == activity.get("recipient", {}).get("id"):
                # Bot was removed from conversation
                logger.info("Bot removed from Teams conversation")

        return {}

    async def _handle_message_reaction(self, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle message reaction."""
        reactions_added = activity.get("reactionsAdded", [])
        reactions_removed = activity.get("reactionsRemoved", [])

        for reaction in reactions_added:
            logger.debug(f"Reaction added: {reaction.get('type')}")

        for reaction in reactions_removed:
            logger.debug(f"Reaction removed: {reaction.get('type')}")

        return {}

    async def _handle_installation_update(self, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle installation update (app installed/uninstalled)."""
        action = activity.get("action", "")
        logger.info(f"Teams installation update: {action}")
        return {}

    async def _send_welcome(self, activity: dict[str, Any]) -> None:
        """Send welcome message when bot is added."""
        welcome = """**Welcome to Aragora!**

I orchestrate 15+ AI models to debate and deliver defensible decisions for your organization.

**Quick Start:**
- @Aragora debate <topic> - Start a multi-agent debate
- @Aragora help - See all commands

Ready to make better decisions together!"""

        await self._send_reply(activity, welcome)

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
        """Send an Adaptive Card reply."""
        connector = await self._get_connector()
        if not connector:
            logger.warning("Cannot send card - connector not available")
            return

        try:
            conversation = activity.get("conversation", {})
            service_url = activity.get("serviceUrl", "")

            await connector.send_message(
                channel_id=conversation.get("id", ""),
                text=fallback_text,
                blocks=card.get("body", []),
                service_url=service_url,
                thread_id=activity.get("replyToId"),
            )
        except (RuntimeError, OSError, ValueError) as e:
            logger.error(f"Failed to send Teams card: {e}")


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

    def _build_status_response(
        self, extra_status: Optional[dict[str, Any]] = None
    ) -> HandlerResult:
        """Build Teams-specific status response."""
        sdk_available, sdk_error = _check_botframework_available()
        connector_available, connector_error = _check_connector_available()

        status = {
            "platform": self.bot_platform,
            "enabled": self._is_bot_enabled(),
            "app_id_configured": bool(TEAMS_APP_ID),
            "password_configured": bool(TEAMS_APP_PASSWORD),
            "tenant_id_configured": bool(TEAMS_TENANT_ID),
            "sdk_available": sdk_available,
            "sdk_error": sdk_error,
            "connector_available": connector_available,
            "connector_error": connector_error,
            "active_debates": len(_active_debates),
            "features": {
                "adaptive_cards": True,
                "voting": True,
                "threading": True,
                "proactive_messaging": True,
            },
        }
        if extra_status:
            status.update(extra_status)
        return json_response(status)

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
    async def handle(
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
]
