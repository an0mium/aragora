"""
Microsoft Teams integration endpoint handlers.

Provides high-level integration for running debates from Teams,
complementing the low-level Bot Framework handler in bots/teams.py.

Endpoints:
- POST /api/integrations/teams/commands  - Handle @aragora commands
- POST /api/integrations/teams/interactive - Handle Adaptive Card actions
- GET  /api/integrations/teams/status     - Integration status
- POST /api/integrations/teams/notify     - Send debate notifications

Environment Variables:
- TEAMS_APP_ID: Bot application ID (required)
- TEAMS_APP_PASSWORD: Bot application password (required)
- TEAMS_TENANT_ID: Tenant ID for Graph API (optional)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


def _handle_task_exception(task: asyncio.Task[Any], task_name: str) -> None:
    """Handle exceptions from fire-and-forget async tasks."""
    if task.cancelled():
        logger.debug(f"Task {task_name} was cancelled")
    elif task.exception():
        exc = task.exception()
        logger.error(f"Task {task_name} failed with exception: {exc}", exc_info=exc)


def create_tracked_task(coro: Coroutine[Any, Any, Any], name: str) -> asyncio.Task[Any]:
    """Create an async task with exception logging."""
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(lambda t: _handle_task_exception(t, name))
    return task


from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)
from ..utils.rate_limit import rate_limit

# Environment configuration
TEAMS_APP_ID = os.environ.get("TEAMS_APP_ID", "")
TEAMS_APP_PASSWORD = os.environ.get("TEAMS_APP_PASSWORD", "")
TEAMS_TENANT_ID = os.environ.get("TEAMS_TENANT_ID", "")

# Command patterns
COMMAND_PATTERN = re.compile(r"^(?:@\w+\s+)?(\w+)(?:\s+(.*))?$", re.IGNORECASE)
TOPIC_PATTERN = re.compile(r'^["\']?(.+?)["\']?$')

# Singleton connector
_teams_connector: Optional[Any] = None


def get_teams_connector() -> Optional[Any]:
    """Get or create the Teams connector singleton."""
    global _teams_connector
    if _teams_connector is None:
        if not TEAMS_APP_ID or not TEAMS_APP_PASSWORD:
            logger.debug("Teams integration disabled (missing credentials)")
            return None
        try:
            from aragora.connectors.chat.teams import TeamsConnector

            _teams_connector = TeamsConnector(
                app_id=TEAMS_APP_ID,
                app_password=TEAMS_APP_PASSWORD,
                tenant_id=TEAMS_TENANT_ID,
            )
            logger.info("Teams connector initialized")
        except ImportError as e:
            logger.warning(f"Teams connector module not available: {e}")
            return None
        except Exception as e:
            logger.exception(f"Error initializing Teams connector: {e}")
            return None
    return _teams_connector


class TeamsIntegrationHandler(BaseHandler):
    """Handler for Microsoft Teams integration endpoints."""

    ROUTES = [
        "/api/v1/integrations/teams/commands",
        "/api/v1/integrations/teams/interactive",
        "/api/v1/integrations/teams/status",
        "/api/v1/integrations/teams/notify",
    ]

    def __init__(self, server_context: Any):
        super().__init__(server_context)
        # Track active debates by conversation ID
        self._active_debates: Dict[str, Dict[str, Any]] = {}

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route Teams requests to appropriate methods."""
        logger.debug(f"Teams integration request: {path}")

        if path == "/api/v1/integrations/teams/status":
            return self._get_status()

        return None

    @rate_limit(rpm=30, limiter_name="teams_commands")
    def handle_post(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle POST requests."""
        if path == "/api/v1/integrations/teams/commands":
            return self._handle_command(handler)
        elif path == "/api/v1/integrations/teams/interactive":
            return self._handle_interactive(handler)
        elif path == "/api/v1/integrations/teams/notify":
            return self._handle_notify(handler)

        return error_response("Not found", 404)

    def _get_status(self) -> HandlerResult:
        """Get Teams integration status."""
        connector = get_teams_connector()
        return json_response(
            {
                "enabled": connector is not None,
                "app_id_configured": bool(TEAMS_APP_ID),
                "password_configured": bool(TEAMS_APP_PASSWORD),
                "tenant_id_configured": bool(TEAMS_TENANT_ID),
                "connector_ready": connector is not None,
            }
        )

    def _handle_command(self, handler: Any) -> HandlerResult:
        """Handle @aragora command from Teams.

        Commands:
        - debate <topic>  - Start a new debate
        - status          - Show active debate status
        - help            - Show available commands
        - cancel          - Cancel active debate
        """
        try:
            body = self._read_json_body(handler)
            if not body:
                return error_response("Invalid request body", 400)

            # Extract command from Bot Framework activity
            text = body.get("text", "")
            conversation = body.get("conversation", {})
            service_url = body.get("serviceUrl", "")
            from_user = body.get("from", {})

            # Parse command
            # Remove bot mention if present
            clean_text = re.sub(r"<at>.*?</at>\s*", "", text).strip()
            match = COMMAND_PATTERN.match(clean_text)

            if not match:
                return self._send_help_response(conversation, service_url)

            command = match.group(1).lower()
            args = match.group(2) or ""

            if command == "debate":
                return self._start_debate(
                    topic=args.strip(),
                    conversation=conversation,
                    service_url=service_url,
                    user=from_user,
                )
            elif command == "status":
                return self._get_debate_status(conversation)
            elif command == "cancel":
                return self._cancel_debate(conversation)
            elif command == "help":
                return self._send_help_response(conversation, service_url)
            else:
                return self._send_unknown_command(command, conversation, service_url)

        except json.JSONDecodeError:
            return error_response("Invalid JSON", 400)
        except Exception as e:
            logger.exception(f"Teams command error: {e}")
            return error_response(f"Error: {str(e)[:100]}", 500)

    def _handle_interactive(self, handler: Any) -> HandlerResult:
        """Handle Adaptive Card action submissions."""
        try:
            body = self._read_json_body(handler)
            if not body:
                return error_response("Invalid request body", 400)

            # Extract action data
            value = body.get("value", {})
            action = value.get("action", "")
            conversation = body.get("conversation", {})
            service_url = body.get("serviceUrl", "")

            if action == "vote":
                return self._handle_vote(value, conversation, service_url)
            elif action == "cancel_debate":
                return self._cancel_debate(conversation)
            elif action == "view_receipt":
                return self._handle_view_receipt(value, conversation, service_url)
            else:
                logger.warning(f"Unknown Teams action: {action}")
                return json_response({"status": "unknown_action"})

        except Exception as e:
            logger.exception(f"Teams interactive error: {e}")
            return error_response(f"Error: {str(e)[:100]}", 500)

    def _handle_notify(self, handler: Any) -> HandlerResult:
        """Send notification to a Teams channel/conversation."""
        try:
            body = self._read_json_body(handler)
            if not body:
                return error_response("Invalid request body", 400)

            conversation_id = body.get("conversation_id")
            service_url = body.get("service_url")
            message = body.get("message", "")
            blocks = body.get("blocks")

            if not conversation_id or not service_url:
                return error_response("Missing conversation_id or service_url", 400)

            connector = get_teams_connector()
            if not connector:
                return error_response("Teams integration not configured", 503)

            # Send message asynchronously
            async def send():
                return await connector.send_message(
                    channel_id=conversation_id,
                    text=message,
                    blocks=blocks,
                    service_url=service_url,
                )

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(send())
            finally:
                loop.close()

            return json_response(
                {
                    "success": result.success,
                    "message_id": result.message_id,
                    "error": result.error,
                }
            )

        except Exception as e:
            logger.exception(f"Teams notify error: {e}")
            return error_response(f"Error: {str(e)[:100]}", 500)

    def _start_debate(
        self,
        topic: str,
        conversation: Dict[str, Any],
        service_url: str,
        user: Dict[str, Any],
    ) -> HandlerResult:
        """Start a new debate in the conversation."""
        if not topic:
            return self._send_error(
                "Please provide a topic for the debate.", conversation, service_url
            )

        conv_id = conversation.get("id", "")

        # Check if debate already running
        if conv_id in self._active_debates:
            return self._send_error(
                "A debate is already running in this channel. Use `@aragora cancel` to cancel it.",
                conversation,
                service_url,
            )

        connector = get_teams_connector()
        if not connector:
            return error_response("Teams integration not configured", 503)

        # Send initial acknowledgment
        ack_blocks = self._build_starting_blocks(topic, user.get("name", "Unknown"))

        async def send_ack():
            return await connector.send_message(
                channel_id=conv_id,
                text=f"Starting debate on: {topic}",
                blocks=ack_blocks,
                service_url=service_url,
            )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            ack_result = loop.run_until_complete(send_ack())
        finally:
            loop.close()

        if not ack_result.success:
            return error_response(f"Failed to send message: {ack_result.error}", 500)

        # Store active debate
        self._active_debates[conv_id] = {
            "topic": topic,
            "thread_ts": ack_result.message_id,
            "user": user,
            "service_url": service_url,
            "status": "starting",
        }

        # Start debate asynchronously
        create_tracked_task(
            self._run_debate_async(conv_id, topic, service_url, ack_result.message_id),
            f"teams_debate_{conv_id}",
        )

        return json_response(
            {
                "success": True,
                "message": "Debate started",
                "conversation_id": conv_id,
                "topic": topic,
            }
        )

    async def _run_debate_async(
        self,
        conv_id: str,
        topic: str,
        service_url: str,
        thread_ts: Optional[str],
    ) -> None:
        """Run a debate asynchronously and post updates."""
        connector = get_teams_connector()
        if not connector:
            return

        try:
            # Import debate components
            from aragora.debate.orchestrator import Arena
            from aragora.core import Environment, DebateProtocol

            # Create environment and protocol
            env = Environment(task=topic)
            protocol = DebateProtocol(rounds=3, consensus="majority")

            # Get available agents
            from aragora.agents import get_agents_by_names

            agents = get_agents_by_names(["anthropic-api", "openai-api", "gemini"])[:3]

            if not agents:
                await connector.send_message(
                    channel_id=conv_id,
                    text="No AI agents available. Check API key configuration.",
                    service_url=service_url,
                    thread_id=thread_ts,
                )
                self._active_debates.pop(conv_id, None)
                return

            # Update status
            if conv_id in self._active_debates:
                self._active_debates[conv_id]["status"] = "running"

            # Run debate
            arena = Arena(env, agents, protocol)
            result = await arena.run()

            # Post result
            result_blocks = self._build_result_blocks(topic, result)
            consensus_text = (
                result.final_answer if result.consensus_reached else "No consensus reached"
            )
            await connector.send_message(
                channel_id=conv_id,
                text=f"Debate complete: {consensus_text}",
                blocks=result_blocks,
                service_url=service_url,
                thread_id=thread_ts,
            )

            # Generate receipt if available
            receipt_id = None
            try:
                from aragora.export.decision_receipt import DecisionReceipt

                receipt = DecisionReceipt.from_debate_result(result)
                receipt_id = receipt.receipt_id
                if conv_id in self._active_debates:
                    self._active_debates[conv_id]["receipt_id"] = receipt_id
            except Exception as e:
                logger.warning(f"Failed to generate receipt: {e}")

            # Update status
            if conv_id in self._active_debates:
                self._active_debates[conv_id]["status"] = "completed"
                self._active_debates[conv_id]["result"] = result

        except Exception as e:
            logger.exception(f"Teams debate error: {e}")
            await connector.send_message(
                channel_id=conv_id,
                text=f"Debate failed: {str(e)[:200]}",
                service_url=service_url,
                thread_id=thread_ts,
            )
            if conv_id in self._active_debates:
                self._active_debates[conv_id]["status"] = "failed"
                self._active_debates[conv_id]["error"] = str(e)
        finally:
            # Clean up after delay
            await asyncio.sleep(300)  # Keep for 5 minutes
            self._active_debates.pop(conv_id, None)

    def _get_debate_status(self, conversation: Dict[str, Any]) -> HandlerResult:
        """Get status of active debate in conversation."""
        conv_id = conversation.get("id", "")
        debate = self._active_debates.get(conv_id)

        if not debate:
            return json_response(
                {
                    "active": False,
                    "message": "No active debate in this channel.",
                }
            )

        return json_response(
            {
                "active": True,
                "topic": debate.get("topic"),
                "status": debate.get("status"),
                "receipt_id": debate.get("receipt_id"),
            }
        )

    def _cancel_debate(self, conversation: Dict[str, Any]) -> HandlerResult:
        """Cancel an active debate."""
        conv_id = conversation.get("id", "")
        debate = self._active_debates.pop(conv_id, None)

        if not debate:
            return json_response(
                {
                    "cancelled": False,
                    "message": "No active debate to cancel.",
                }
            )

        return json_response(
            {
                "cancelled": True,
                "topic": debate.get("topic"),
            }
        )

    def _handle_vote(
        self,
        value: Dict[str, Any],
        conversation: Dict[str, Any],
        service_url: str,
    ) -> HandlerResult:
        """Handle a vote action from Adaptive Card."""
        vote_value = value.get("vote")
        debate_id = value.get("debate_id")

        # TODO: Record vote in debate system
        logger.info(f"Vote received: {vote_value} for debate {debate_id}")

        return json_response(
            {
                "status": "vote_recorded",
                "vote": vote_value,
            }
        )

    def _handle_view_receipt(
        self,
        value: Dict[str, Any],
        conversation: Dict[str, Any],
        service_url: str,
    ) -> HandlerResult:
        """Handle view receipt action."""
        receipt_id = value.get("receipt_id")
        # Return acknowledgment - Teams will handle the URL navigation
        return json_response(
            {
                "status": "ok",
                "receipt_id": receipt_id,
            }
        )

    def _send_help_response(
        self,
        conversation: Dict[str, Any],
        service_url: str,
    ) -> HandlerResult:
        """Send help message with available commands."""
        help_blocks = [
            {
                "type": "TextBlock",
                "text": "Aragora Commands",
                "size": "Large",
                "weight": "Bolder",
            },
            {
                "type": "TextBlock",
                "text": "Available commands:",
                "wrap": True,
            },
            {
                "type": "FactSet",
                "facts": [
                    {
                        "title": "@aragora debate <topic>",
                        "value": "Start a new debate on the topic",
                    },
                    {"title": "@aragora status", "value": "Check status of active debate"},
                    {"title": "@aragora cancel", "value": "Cancel the active debate"},
                    {"title": "@aragora help", "value": "Show this help message"},
                ],
            },
        ]

        connector = get_teams_connector()
        if connector:

            async def send():
                return await connector.send_message(
                    channel_id=conversation.get("id", ""),
                    text="Aragora Help",
                    blocks=help_blocks,
                    service_url=service_url,
                )

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(send())
            finally:
                loop.close()

        return json_response({"status": "help_sent"})

    def _send_error(
        self,
        message: str,
        conversation: Dict[str, Any],
        service_url: str,
    ) -> HandlerResult:
        """Send error message to conversation."""
        connector = get_teams_connector()
        if connector:

            async def send():
                return await connector.send_message(
                    channel_id=conversation.get("id", ""),
                    text=message,
                    service_url=service_url,
                )

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(send())
            finally:
                loop.close()

        return json_response({"status": "error", "message": message})

    def _send_unknown_command(
        self,
        command: str,
        conversation: Dict[str, Any],
        service_url: str,
    ) -> HandlerResult:
        """Send unknown command message."""
        return self._send_error(
            f"Unknown command: {command}. Use `@aragora help` for available commands.",
            conversation,
            service_url,
        )

    def _build_starting_blocks(self, topic: str, user_name: str) -> List[Dict[str, Any]]:
        """Build Adaptive Card blocks for debate start."""
        return [
            {
                "type": "TextBlock",
                "text": "Debate Starting",
                "size": "Large",
                "weight": "Bolder",
                "color": "Accent",
            },
            {
                "type": "TextBlock",
                "text": f"**Topic:** {topic}",
                "wrap": True,
            },
            {
                "type": "TextBlock",
                "text": f"**Initiated by:** {user_name}",
                "isSubtle": True,
            },
            {
                "type": "TextBlock",
                "text": "AI agents are now deliberating...",
                "isSubtle": True,
            },
            {
                "type": "ActionSet",
                "actions": [
                    {
                        "type": "Action.Submit",
                        "title": "Cancel Debate",
                        "style": "destructive",
                        "data": {"action": "cancel_debate"},
                    }
                ],
            },
        ]

    def _build_result_blocks(self, topic: str, result: Any) -> List[Dict[str, Any]]:
        """Build Adaptive Card blocks for debate result."""
        consensus = getattr(result, "consensus", None) or "No consensus reached"
        confidence = getattr(result, "confidence", 0.0)
        rounds = getattr(result, "rounds_completed", 0)

        blocks: List[Dict[str, Any]] = [
            {
                "type": "TextBlock",
                "text": "Debate Complete",
                "size": "Large",
                "weight": "Bolder",
                "color": "Good",
            },
            {
                "type": "TextBlock",
                "text": f"**Topic:** {topic}",
                "wrap": True,
            },
            {
                "type": "TextBlock",
                "text": f"**Decision:** {consensus}",
                "wrap": True,
                "weight": "Bolder",
            },
            {
                "type": "FactSet",
                "facts": [
                    {"title": "Confidence", "value": f"{confidence:.0%}"},
                    {"title": "Rounds", "value": str(rounds)},
                ],
            },
        ]

        # Add receipt link if available
        receipt_id = getattr(result, "receipt_id", None)
        if receipt_id:
            blocks.append(
                {
                    "type": "ActionSet",
                    "actions": [
                        {
                            "type": "Action.OpenUrl",
                            "title": "View Receipt",
                            "url": f"/api/v1/gauntlet/receipts/{receipt_id}",
                        }
                    ],
                }
            )

        return blocks

    def _read_json_body(self, handler: Any) -> Optional[Dict[str, Any]]:
        """Read and parse JSON body from request."""
        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length == 0:
                return None
            body = handler.rfile.read(content_length)
            return json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, ValueError):
            return None


__all__ = ["TeamsIntegrationHandler", "get_teams_connector"]
