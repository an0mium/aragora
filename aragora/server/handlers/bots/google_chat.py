"""
Google Chat Bot webhook handler.

Handles Google Chat App webhook events for bidirectional chat.

Endpoints:
- POST /api/bots/google-chat/webhook - Handle Google Chat events
- GET  /api/bots/google-chat/status - Get bot status

Environment Variables:
- GOOGLE_CHAT_CREDENTIALS - Service account JSON or path to file
- GOOGLE_CHAT_PROJECT_ID - Google Cloud project ID
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Callable, Coroutine, Dict, Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)
from aragora.server.handlers.utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# Environment variables
GOOGLE_CHAT_CREDENTIALS = os.environ.get("GOOGLE_CHAT_CREDENTIALS", "")
GOOGLE_CHAT_PROJECT_ID = os.environ.get("GOOGLE_CHAT_PROJECT_ID", "")


def _handle_task_exception(task: asyncio.Task[Any], task_name: str) -> None:
    """Handle exceptions from fire-and-forget async tasks."""
    if task.cancelled():
        logger.debug(f"Task {task_name} was cancelled")
    elif task.exception():
        exc = task.exception()
        logger.error(f"Task {task_name} failed: {exc}", exc_info=exc)


def create_tracked_task(coro: Coroutine[Any, Any, Any], name: str) -> asyncio.Task[Any]:
    """Create an async task with exception logging."""
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(lambda t: _handle_task_exception(t, name))
    return task


# Cache for Google Chat connector
_google_chat_connector: Optional[Any] = None


def get_google_chat_connector() -> Optional[Any]:
    """Get or create the Google Chat connector singleton."""
    global _google_chat_connector
    if _google_chat_connector is None:
        if not GOOGLE_CHAT_CREDENTIALS:
            logger.debug("Google Chat connector disabled (no GOOGLE_CHAT_CREDENTIALS)")
            return None
        try:
            from aragora.connectors.chat.google_chat import GoogleChatConnector

            _google_chat_connector = GoogleChatConnector(
                credentials_json=GOOGLE_CHAT_CREDENTIALS,
                project_id=GOOGLE_CHAT_PROJECT_ID,
            )
            logger.info("Google Chat connector initialized")
        except ImportError as e:
            logger.warning(f"Google Chat connector module not available: {e}")
            return None
        except Exception as e:
            logger.exception(f"Error initializing Google Chat connector: {e}")
            return None
    return _google_chat_connector


class GoogleChatHandler(BaseHandler):
    """Handler for Google Chat App webhook endpoints."""

    ROUTES = [
        "/api/bots/google-chat/webhook",
        "/api/bots/google-chat/status",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    @rate_limit(rpm=60)
    def handle(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route Google Chat GET requests."""
        if path == "/api/bots/google-chat/status":
            return self._get_status()

        return None

    @rate_limit(rpm=120)
    def handle_post(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle POST requests (webhook events)."""
        if path == "/api/bots/google-chat/webhook":
            return self._handle_webhook(handler)

        return None

    def _get_status(self) -> HandlerResult:
        """Get Google Chat bot status."""
        connector = get_google_chat_connector()
        return json_response(
            {
                "platform": "google_chat",
                "enabled": connector is not None,
                "credentials_configured": bool(GOOGLE_CHAT_CREDENTIALS),
                "project_id_configured": bool(GOOGLE_CHAT_PROJECT_ID),
            }
        )

    def _handle_webhook(self, handler: Any) -> HandlerResult:
        """Handle Google Chat webhook events.

        Google Chat sends events to this endpoint for:
        - Messages (regular and slash commands)
        - Card button clicks
        - Bot added/removed from space
        """
        try:
            # Read body
            content_length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(content_length)

            # Parse event
            try:
                event = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in Google Chat event: {e}")
                return error_response("Invalid JSON", 400)

            event_type = event.get("type", "")
            logger.debug(f"Google Chat event: {event_type}")

            # Route by event type
            if event_type == "MESSAGE":
                return self._handle_message(event)
            elif event_type == "CARD_CLICKED":
                return self._handle_card_click(event)
            elif event_type == "ADDED_TO_SPACE":
                return self._handle_added_to_space(event)
            elif event_type == "REMOVED_FROM_SPACE":
                return self._handle_removed_from_space(event)
            else:
                logger.debug(f"Unhandled Google Chat event type: {event_type}")
                return json_response({})

        except Exception as e:
            logger.exception(f"Google Chat webhook error: {e}")
            return json_response({"text": f"Error: {str(e)[:100]}"})

    def _handle_message(self, event: Dict[str, Any]) -> HandlerResult:
        """Handle incoming Google Chat message."""
        message = event.get("message", {})
        space = event.get("space", {})
        user = event.get("user", {})

        space_name = space.get("name", "")
        space_type = space.get("type", "")
        user_name = user.get("displayName", "Unknown")
        text = message.get("text", "")

        logger.info(f"Google Chat message from {user_name}: {text[:50]}...")

        # Check for slash command
        slash_command = message.get("slashCommand")
        if slash_command:
            command_name = slash_command.get("commandName", "").lstrip("/")
            args = message.get("argumentText", "")
            return self._handle_slash_command(command_name, args, space_name, user, event)

        # Check if in DM - treat as implicit debate request
        if space_type == "DM" and text.strip():
            return self._start_debate_from_message(text, space_name, user, event)

        # Check for @mention
        mentions = message.get("annotations", [])
        for mention in mentions:
            if mention.get("type") == "USER_MENTION" and mention.get("userMention", {}).get("type") == "BOT":
                # Bot was mentioned - extract text after mention
                mention_text = text
                if mention_text.strip():
                    return self._start_debate_from_message(mention_text, space_name, user, event)

        # Regular message in group - acknowledge
        return json_response({})

    def _handle_slash_command(
        self,
        command: str,
        args: str,
        space_name: str,
        user: Dict[str, Any],
        event: Dict[str, Any],
    ) -> HandlerResult:
        """Handle Google Chat slash command."""
        command = command.lower()
        user_name = user.get("displayName", "Unknown")

        logger.info(f"Google Chat command from {user_name}: /{command} {args[:50]}...")

        if command == "debate":
            return self._cmd_debate(args, space_name, user, event)
        elif command == "gauntlet":
            return self._cmd_gauntlet(args, space_name, user, event)
        elif command == "status":
            return self._cmd_status(space_name)
        elif command == "help":
            return self._cmd_help()
        elif command == "agents":
            return self._cmd_agents()
        else:
            return self._card_response(
                title="Unknown Command",
                body=f"Unknown command: /{command}\n\nUse /help for available commands.",
            )

    def _handle_card_click(self, event: Dict[str, Any]) -> HandlerResult:
        """Handle Google Chat card button click."""
        action = event.get("action", {})
        function_name = action.get("actionMethodName", "")
        params = {p.get("key"): p.get("value") for p in action.get("parameters", [])}

        user = event.get("user", {})
        user_id = user.get("name", "").split("/")[-1]
        space = event.get("space", {})
        space_name = space.get("name", "")

        logger.info(f"Google Chat card click: {function_name} from {user_id}")

        # Route actions
        if function_name == "vote_agree":
            return self._handle_vote(params.get("debate_id", ""), "agree", user_id, space_name)
        elif function_name == "vote_disagree":
            return self._handle_vote(params.get("debate_id", ""), "disagree", user_id, space_name)
        elif function_name == "view_details":
            return self._handle_view_details(params.get("debate_id", ""))

        return json_response({})

    def _handle_vote(
        self, debate_id: str, vote_option: str, user_id: str, space_name: str
    ) -> HandlerResult:
        """Handle vote button click."""
        if not debate_id:
            return self._card_response(body="Error: No debate ID provided")

        logger.info(f"Vote from {user_id} on {debate_id}: {vote_option}")

        try:
            from aragora.memory.consensus import ConsensusStore

            store = ConsensusStore()
            store.record_vote(
                debate_id=debate_id,
                user_id=f"gchat:{user_id}",
                vote=vote_option,
                source="google_chat",
            )

            emoji = "✅" if vote_option == "agree" else "❌"
            return self._card_response(
                body=f"{emoji} Your vote for '{vote_option}' has been recorded!"
            )

        except ImportError:
            logger.warning("ConsensusStore not available")
            return self._card_response(body="Vote acknowledged")
        except Exception as e:
            logger.error(f"Vote recording failed: {e}")
            return self._card_response(body="Error recording vote")

    def _handle_view_details(self, debate_id: str) -> HandlerResult:
        """Handle view details button click."""
        if not debate_id:
            return self._card_response(body="Error: No debate ID provided")

        try:
            from aragora.server.storage import get_debates_db

            db = get_debates_db()
            debate = db.get(debate_id) if db else None

            if not debate:
                return self._card_response(body=f"Debate {debate_id} not found")

            task = debate.get("task", "Unknown topic")
            final_answer = debate.get("final_answer", "No conclusion")
            consensus = debate.get("consensus_reached", False)
            confidence = debate.get("confidence", 0)
            rounds_used = debate.get("rounds_used", 0)

            return self._card_response(
                title="Debate Details",
                body=f"*Topic:* {task[:200]}\n\n"
                     f"*Consensus:* {'Yes' if consensus else 'No'}\n"
                     f"*Confidence:* {confidence:.1%}\n"
                     f"*Rounds:* {rounds_used}\n\n"
                     f"*Conclusion:*\n{final_answer[:500]}...",
            )

        except Exception as e:
            logger.error(f"View details failed: {e}")
            return self._card_response(body="Error fetching debate details")

    def _handle_added_to_space(self, event: Dict[str, Any]) -> HandlerResult:
        """Handle bot added to space."""
        space = event.get("space", {})
        space_name = space.get("displayName", "this space")

        return self._card_response(
            title="Welcome to Aragora!",
            body="I'm an Omnivorous Multi-Agent Decision Making Engine.\n\n"
                 "I harness Claude, GPT, Gemini, Grok and more to help you "
                 "make better decisions through structured debate.\n\n"
                 "*Commands:*\n"
                 "/debate <topic> - Start a multi-agent debate\n"
                 "/gauntlet <statement> - Run adversarial validation\n"
                 "/status - Check system status\n"
                 "/help - Show available commands\n\n"
                 "Or just @mention me with a question!",
        )

    def _handle_removed_from_space(self, event: Dict[str, Any]) -> HandlerResult:
        """Handle bot removed from space."""
        space = event.get("space", {})
        logger.info(f"Bot removed from space: {space.get('name')}")
        return json_response({})

    # ==========================================================================
    # Slash Command Implementations
    # ==========================================================================

    def _cmd_help(self) -> HandlerResult:
        """Show help message."""
        return self._card_response(
            title="Aragora Commands",
            body="*/debate <topic>* - Start a multi-agent debate\n"
                 "*/gauntlet <statement>* - Run adversarial stress-test\n"
                 "*/status* - Check system status\n"
                 "*/agents* - List available agents\n"
                 "*/help* - Show this message\n\n"
                 "*Examples:*\n"
                 '/debate Should AI be regulated?\n'
                 '/gauntlet We should migrate to microservices',
        )

    def _cmd_status(self, space_name: str) -> HandlerResult:
        """Show system status."""
        try:
            from aragora.ranking.elo import EloSystem

            store = EloSystem()
            agents = store.get_all_ratings()

            return self._card_response(
                title="Aragora System Status",
                fields=[
                    ("Status", "Online"),
                    ("Agents", str(len(agents))),
                    ("Models", "Claude, GPT, Gemini, Grok, Mistral, DeepSeek"),
                ],
            )

        except ImportError:
            return self._card_response(
                title="Aragora System Status",
                fields=[("Status", "Online"), ("Agents", "Available")],
            )

    def _cmd_agents(self) -> HandlerResult:
        """List available agents."""
        try:
            from aragora.ranking.elo import EloSystem

            store = EloSystem()
            agents = store.get_all_ratings()

            if not agents:
                return self._card_response(body="No agents registered yet.")

            agents = sorted(agents, key=lambda a: getattr(a, "elo", 1500), reverse=True)

            lines = []
            for i, agent in enumerate(agents[:10]):
                name = getattr(agent, "name", "Unknown")
                elo = getattr(agent, "elo", 1500)
                medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i + 1}."
                lines.append(f"{medal} *{name}* - ELO: {elo:.0f}")

            return self._card_response(
                title="Top Agents by ELO",
                body="\n".join(lines),
            )

        except ImportError:
            return self._card_response(body="Agents service temporarily unavailable")

    def _cmd_debate(
        self,
        topic: str,
        space_name: str,
        user: Dict[str, Any],
        event: Dict[str, Any],
    ) -> HandlerResult:
        """Start a debate."""
        if not topic.strip():
            return self._card_response(
                body="Please provide a topic.\n\nExample: /debate Should AI be regulated?"
            )

        topic = topic.strip().strip("\"'")

        if len(topic) < 10:
            return self._card_response(body="Topic is too short. Please provide more detail.")

        if len(topic) > 500:
            return self._card_response(body="Topic is too long. Please limit to 500 characters.")

        user_name = user.get("displayName", "Unknown")
        user_id = user.get("name", "").split("/")[-1]

        # Queue debate asynchronously
        create_tracked_task(
            self._run_debate_async(topic, space_name, user_id),
            name=f"gchat-debate-{topic[:30]}",
        )

        return self._card_response(
            title="Starting Debate",
            body=f"*Topic:* {topic}\n\n_Processing... I'll post the result when complete._",
            context=f"Requested by {user_name}",
        )

    def _cmd_gauntlet(
        self,
        statement: str,
        space_name: str,
        user: Dict[str, Any],
        event: Dict[str, Any],
    ) -> HandlerResult:
        """Run gauntlet validation."""
        if not statement.strip():
            return self._card_response(
                body="Please provide a statement to stress-test.\n\n"
                     "Example: /gauntlet We should migrate to microservices"
            )

        statement = statement.strip().strip("\"'")

        if len(statement) < 10:
            return self._card_response(body="Statement is too short. Please provide more detail.")

        if len(statement) > 1000:
            return self._card_response(body="Statement is too long. Please limit to 1000 characters.")

        user_name = user.get("displayName", "Unknown")
        user_id = user.get("name", "").split("/")[-1]

        # Queue gauntlet asynchronously
        create_tracked_task(
            self._run_gauntlet_async(statement, space_name, user_id),
            name=f"gchat-gauntlet-{statement[:30]}",
        )

        return self._card_response(
            title="Running Gauntlet Stress-Test",
            body=f"*Statement:* {statement[:200]}{'...' if len(statement) > 200 else ''}\n\n"
                 "_Running adversarial validation..._",
            context=f"Requested by {user_name}",
        )

    def _start_debate_from_message(
        self,
        text: str,
        space_name: str,
        user: Dict[str, Any],
        event: Dict[str, Any],
    ) -> HandlerResult:
        """Start a debate from a regular message or @mention."""
        # Clean up the text (remove @mentions)
        import re
        clean_text = re.sub(r"<users/[^>]+>", "", text).strip()

        if not clean_text or len(clean_text) < 10:
            return self._card_response(
                body="I need more context. Try asking a specific question or use /debate <topic>."
            )

        return self._cmd_debate(clean_text, space_name, user, event)

    # ==========================================================================
    # Async Operations
    # ==========================================================================

    async def _run_debate_async(
        self,
        topic: str,
        space_name: str,
        user_id: str,
    ) -> None:
        """Run debate asynchronously and post result."""
        connector = get_google_chat_connector()
        if not connector:
            logger.warning("No Google Chat connector available")
            return

        try:
            from aragora import Arena, DebateProtocol, Environment
            from aragora.agents import get_agents_by_names  # type: ignore[attr-defined]

            env = Environment(task=f"Debate: {topic}")
            agents = get_agents_by_names(["anthropic-api", "openai-api"])
            protocol = DebateProtocol(
                rounds=3,
                consensus="majority",
                convergence_detection=False,
                early_stopping=False,
            )

            if not agents:
                await connector.send_message(
                    space_name,
                    "Failed to start debate: No agents available",
                )
                return

            arena = Arena.from_env(env, agents, protocol)
            result = await arena.run()

            # Build result card
            consensus_emoji = "✅" if result.consensus_reached else "❌"
            confidence_bar = "█" * int(result.confidence * 5) + "░" * (5 - int(result.confidence * 5))

            sections = connector.format_blocks(
                title=f"{consensus_emoji} Debate Complete",
                body=f"*Topic:* {topic[:100]}...\n\n"
                     f"*Conclusion:*\n{result.final_answer[:400] if result.final_answer else 'No conclusion'}...",
                fields=[
                    ("Consensus", "Yes" if result.consensus_reached else "No"),
                    ("Confidence", f"{confidence_bar} {result.confidence:.0%}"),
                    ("Rounds", str(result.rounds_used)),
                    ("Agents", str(len(agents))),
                ],
                actions=[
                    connector.connectors.chat.models.MessageButton(
                        text="👍 Agree",
                        action_id="vote_agree",
                        value=result.id,
                    ),
                    connector.connectors.chat.models.MessageButton(
                        text="👎 Disagree",
                        action_id="vote_disagree",
                        value=result.id,
                    ),
                ],
            ) if hasattr(connector, 'format_blocks') else []

            await connector.send_message(
                space_name,
                f"Debate complete: {topic[:50]}...",
                blocks=sections,
            )

        except Exception as e:
            logger.error(f"Async debate failed: {e}", exc_info=True)
            if connector:
                await connector.send_message(
                    space_name,
                    f"Debate failed: {str(e)[:100]}",
                )

    async def _run_gauntlet_async(
        self,
        statement: str,
        space_name: str,
        user_id: str,
    ) -> None:
        """Run gauntlet asynchronously and post result."""
        connector = get_google_chat_connector()
        if not connector:
            logger.warning("No Google Chat connector available")
            return

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:8080/api/v1/gauntlet/run",
                    json={
                        "statement": statement,
                        "intensity": "medium",
                        "metadata": {
                            "source": "google_chat",
                            "space_name": space_name,
                            "user_id": user_id,
                        },
                    },
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    data = await resp.json()

                    if resp.status != 200:
                        await connector.send_message(
                            space_name,
                            f"Gauntlet failed: {data.get('error', 'Unknown error')}",
                        )
                        return

                    score = data.get("score", 0)
                    passed = data.get("passed", False)
                    vulnerabilities = data.get("vulnerabilities", [])

                    status_emoji = "✅" if passed else "❌"
                    score_bar = "█" * int(score * 5) + "░" * (5 - int(score * 5))

                    body = f"*Statement:* {statement[:200]}{'...' if len(statement) > 200 else ''}\n\n"
                    body += f"*Score:* {score_bar} {score:.0%}\n"
                    body += f"*Status:* {'Passed' if passed else 'Failed'}\n"
                    body += f"*Issues Found:* {len(vulnerabilities)}\n"

                    if vulnerabilities:
                        body += "\n*Issues:*\n"
                        for v in vulnerabilities[:3]:
                            body += f"• {v.get('description', 'Unknown')[:80]}\n"
                        if len(vulnerabilities) > 3:
                            body += f"_...and {len(vulnerabilities) - 3} more_"

                    await connector.send_message(
                        space_name,
                        f"Gauntlet complete",
                        blocks=connector.format_blocks(
                            title=f"{status_emoji} Gauntlet Results",
                            body=body,
                        ) if hasattr(connector, 'format_blocks') else None,
                    )

        except Exception as e:
            logger.error(f"Async gauntlet failed: {e}", exc_info=True)
            if connector:
                await connector.send_message(
                    space_name,
                    f"Gauntlet failed: {str(e)[:100]}",
                )

    # ==========================================================================
    # Response Helpers
    # ==========================================================================

    def _card_response(
        self,
        title: Optional[str] = None,
        body: Optional[str] = None,
        fields: Optional[list[tuple[str, str]]] = None,
        context: Optional[str] = None,
        actions: Optional[list[dict]] = None,
    ) -> HandlerResult:
        """Create a Google Chat Card v2 response."""
        sections: list[dict] = []

        # Header
        if title:
            sections.append({"header": title})

        # Body text
        if body:
            sections.append({
                "widgets": [
                    {"textParagraph": {"text": body}}
                ]
            })

        # Fields as decorated text
        if fields:
            widgets = []
            for label, value in fields:
                widgets.append({
                    "decoratedText": {
                        "topLabel": label,
                        "text": value,
                    }
                })
            sections.append({"widgets": widgets})

        # Context footer
        if context:
            sections.append({
                "widgets": [
                    {"textParagraph": {"text": f"<i>{context}</i>"}}
                ]
            })

        # Action buttons
        if actions:
            sections.append({
                "widgets": [
                    {"buttonList": {"buttons": actions}}
                ]
            })

        # Build card response
        response: dict[str, Any] = {}

        if sections:
            response["cardsV2"] = [
                {
                    "cardId": "aragora_response",
                    "card": {"sections": sections},
                }
            ]
        elif body:
            response["text"] = body

        return json_response(response)


# Export handler factory
_google_chat_handler: Optional["GoogleChatHandler"] = None


def get_google_chat_handler(server_context: Optional[Dict] = None) -> "GoogleChatHandler":
    """Get or create the Google Chat handler instance."""
    global _google_chat_handler
    if _google_chat_handler is None:
        if server_context is None:
            server_context = {}
        _google_chat_handler = GoogleChatHandler(server_context)
    return _google_chat_handler


__all__ = ["GoogleChatHandler", "get_google_chat_handler", "get_google_chat_connector"]
