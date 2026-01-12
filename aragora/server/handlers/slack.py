"""
Slack integration endpoint handlers.

Endpoints:
- POST /api/integrations/slack/commands - Handle Slack slash commands
- POST /api/integrations/slack/interactive - Handle interactive components
- POST /api/integrations/slack/events - Handle Slack events

Environment Variables:
- SLACK_SIGNING_SECRET - Required for webhook verification
- SLACK_BOT_TOKEN - Optional for advanced API calls
- SLACK_WEBHOOK_URL - For sending notifications
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional
from urllib.parse import parse_qs

from aragora.server.http_utils import run_async
import asyncio

logger = logging.getLogger(__name__)


def _handle_task_exception(task: asyncio.Task, task_name: str) -> None:
    """Handle exceptions from fire-and-forget async tasks."""
    if task.cancelled():
        logger.debug(f"Task {task_name} was cancelled")
    elif task.exception():
        exc = task.exception()
        logger.error(f"Task {task_name} failed with exception: {exc}", exc_info=exc)


def create_tracked_task(coro, name: str) -> asyncio.Task:
    """Create an async task with exception logging.

    Use this instead of raw asyncio.create_task() for fire-and-forget tasks
    to ensure exceptions are logged rather than silently swallowed.
    """
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(lambda t: _handle_task_exception(t, name))
    return task


from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    safe_json_parse,
    auto_error_response,
)
from .utils.rate_limit import rate_limit

# Environment variables for Slack integration
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET", "")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")

# Patterns for command parsing
COMMAND_PATTERN = re.compile(r'^/aragora\s+(\w+)(?:\s+(.*))?$')
TOPIC_PATTERN = re.compile(r'^["\']?(.+?)["\']?$')


def get_slack_integration():
    """Get or create the Slack integration singleton."""
    global _slack_integration
    if "_slack_integration" not in globals():
        _slack_integration = None
    if _slack_integration is None:
        if not SLACK_WEBHOOK_URL:
            logger.debug("Slack integration disabled (no SLACK_WEBHOOK_URL)")
            return None
        try:
            from aragora.integrations.slack import SlackIntegration, SlackConfig
            config = SlackConfig(webhook_url=SLACK_WEBHOOK_URL)
            _slack_integration = SlackIntegration(config)
            logger.info("Slack integration initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Slack integration: {e}")
            return None
    return _slack_integration


class SlackHandler(BaseHandler):
    """Handler for Slack integration endpoints."""

    ROUTES = [
        "/api/integrations/slack/commands",
        "/api/integrations/slack/interactive",
        "/api/integrations/slack/events",
        "/api/integrations/slack/status",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route Slack requests to appropriate methods."""
        logger.debug(f"Slack request: {path}")

        if path == "/api/integrations/slack/status":
            return self._get_status()

        # All other endpoints require POST
        if handler.command != "POST":
            return error_response("Method not allowed", 405)

        # Verify Slack signature for security
        if SLACK_SIGNING_SECRET and not self._verify_signature(handler):
            logger.warning("Slack signature verification failed")
            return error_response("Invalid signature", 401)

        if path == "/api/integrations/slack/commands":
            return self._handle_slash_command(handler)
        elif path == "/api/integrations/slack/interactive":
            return self._handle_interactive(handler)
        elif path == "/api/integrations/slack/events":
            return self._handle_events(handler)

        return error_response("Not found", 404)

    def handle_post(
        self, path: str, body: dict, handler
    ) -> Optional[HandlerResult]:
        """Handle POST requests."""
        return self.handle(path, {}, handler)

    def _verify_signature(self, handler) -> bool:
        """Verify Slack request signature.

        Slack uses HMAC-SHA256 to sign requests.
        See: https://api.slack.com/authentication/verifying-requests-from-slack
        """
        if not SLACK_SIGNING_SECRET:
            return True  # Skip verification if no secret configured

        try:
            timestamp = handler.headers.get("X-Slack-Request-Timestamp", "")
            signature = handler.headers.get("X-Slack-Signature", "")

            if not timestamp or not signature:
                return False

            # Prevent replay attacks (allow 5 minute window)
            request_time = int(timestamp)
            if abs(time.time() - request_time) > 300:
                logger.warning("Slack request timestamp too old")
                return False

            # Read request body
            content_length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(content_length).decode("utf-8")

            # Compute signature
            sig_basestring = f"v0:{timestamp}:{body}"
            expected_sig = (
                "v0="
                + hmac.new(
                    SLACK_SIGNING_SECRET.encode(),
                    sig_basestring.encode(),
                    hashlib.sha256,
                ).hexdigest()
            )

            # Timing-safe comparison
            return hmac.compare_digest(expected_sig, signature)

        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            return False

    def _get_status(self) -> HandlerResult:
        """Get Slack integration status."""
        integration = get_slack_integration()
        return json_response({
            "enabled": integration is not None,
            "signing_secret_configured": bool(SLACK_SIGNING_SECRET),
            "bot_token_configured": bool(SLACK_BOT_TOKEN),
            "webhook_configured": bool(SLACK_WEBHOOK_URL),
        })

    @rate_limit(rpm=30, limiter_name="slack_commands")
    @auto_error_response
    def _handle_slash_command(self, handler) -> HandlerResult:
        """Handle Slack slash commands.

        Expected format: /aragora <command> [args]

        Commands:
        - /aragora debate "topic" - Start a debate on a topic
        - /aragora status - Get system status
        - /aragora help - Show available commands
        """
        try:
            # Parse form-encoded body
            content_length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(content_length).decode("utf-8")
            params = parse_qs(body)

            command = params.get("command", [""])[0]
            text = params.get("text", [""])[0].strip()
            user_id = params.get("user_id", [""])[0]
            channel_id = params.get("channel_id", [""])[0]
            response_url = params.get("response_url", [""])[0]

            logger.info(f"Slack command from {user_id}: {command} {text}")

            # Parse the subcommand
            if not text:
                return self._command_help()

            parts = text.split(maxsplit=1)
            subcommand = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if subcommand == "help":
                return self._command_help()
            elif subcommand == "status":
                return self._command_status()
            elif subcommand == "debate":
                return self._command_debate(args, user_id, channel_id, response_url)
            elif subcommand == "agents":
                return self._command_agents()
            else:
                return self._slack_response(
                    f"Unknown command: `{subcommand}`. Use `/aragora help` for available commands.",
                    response_type="ephemeral",
                )

        except Exception as e:
            logger.error(f"Slash command error: {e}", exc_info=True)
            return self._slack_response(
                f"Error processing command: {str(e)[:100]}",
                response_type="ephemeral",
            )

    def _command_help(self) -> HandlerResult:
        """Show help message."""
        help_text = """*Aragora Slash Commands*

`/aragora debate "topic"` - Start a debate on a topic
`/aragora status` - Get system status
`/aragora agents` - List available agents
`/aragora help` - Show this help message

*Examples:*
- `/aragora debate "Should AI be regulated?"`
- `/aragora debate AI ethics in healthcare`
- `/aragora status`
"""
        return self._slack_response(help_text, response_type="ephemeral")

    def _command_status(self) -> HandlerResult:
        """Get system status."""
        try:
            # Get basic stats
            from aragora.ranking.elo import get_elo_store

            store = get_elo_store()
            agents = store.get_all_ratings() if store else []

            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "Aragora System Status",
                        "emoji": True,
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Agents:* {len(agents)}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": "*Status:* Online",
                        },
                    ],
                },
            ]

            return self._slack_blocks_response(
                blocks,
                text="Aragora is online",
                response_type="ephemeral",
            )

        except Exception as e:
            logger.error(f"Status command error: {e}")
            return self._slack_response(
                f"Error getting status: {str(e)[:100]}",
                response_type="ephemeral",
            )

    def _command_agents(self) -> HandlerResult:
        """List available agents."""
        try:
            from aragora.ranking.elo import get_elo_store

            store = get_elo_store()
            agents = store.get_all_ratings() if store else []

            if not agents:
                return self._slack_response(
                    "No agents registered yet.",
                    response_type="ephemeral",
                )

            # Sort by ELO
            agents = sorted(agents, key=lambda a: getattr(a, "elo", 1500), reverse=True)

            text = "*Top Agents by ELO:*\n"
            for i, agent in enumerate(agents[:10]):
                name = getattr(agent, "name", "Unknown")
                elo = getattr(agent, "elo", 1500)
                wins = getattr(agent, "wins", 0)
                medal = ["", "", ""][i] if i < 3 else f"{i+1}."
                text += f"{medal} *{name}* - ELO: {elo:.0f} | Wins: {wins}\n"

            return self._slack_response(text, response_type="ephemeral")

        except Exception as e:
            logger.error(f"Agents command error: {e}")
            return self._slack_response(
                f"Error listing agents: {str(e)[:100]}",
                response_type="ephemeral",
            )

    def _command_debate(
        self,
        args: str,
        user_id: str,
        channel_id: str,
        response_url: str,
    ) -> HandlerResult:
        """Start a debate on a topic.

        Args:
            args: The topic text (may be quoted)
            user_id: Slack user ID
            channel_id: Slack channel ID
            response_url: URL for async responses
        """
        if not args:
            return self._slack_response(
                "Please provide a topic. Example: `/aragora debate \"Should AI be regulated?\"`",
                response_type="ephemeral",
            )

        # Strip quotes if present
        topic = args.strip().strip("\"'")

        if len(topic) < 10:
            return self._slack_response(
                "Topic is too short. Please provide a more detailed topic.",
                response_type="ephemeral",
            )

        if len(topic) > 500:
            return self._slack_response(
                "Topic is too long. Please limit to 500 characters.",
                response_type="ephemeral",
            )

        # Acknowledge immediately (Slack requires response within 3 seconds)
        # The actual debate will be processed asynchronously

        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Starting debate on:*\n_{topic}_",
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Requested by <@{user_id}> | Processing...",
                    },
                ],
            },
        ]

        # Queue the debate creation asynchronously
        if response_url:
            create_tracked_task(
                self._create_debate_async(topic, response_url, user_id, channel_id),
                name=f"slack-debate-{topic[:30]}"
            )

        return self._slack_blocks_response(
            blocks,
            text=f"Starting debate: {topic}",
            response_type="in_channel",
        )

    async def _create_debate_async(
        self,
        topic: str,
        response_url: str,
        user_id: str,
        channel_id: str,
    ) -> None:
        """Create debate asynchronously and POST result to Slack response_url."""
        import aiohttp

        try:
            from aragora import Arena, Environment, DebateProtocol
            from aragora.agents import get_agents_by_names

            # Create debate
            env = Environment(task=f"Debate: {topic}")
            agents = get_agents_by_names(["anthropic-api", "openai-api"])
            protocol = DebateProtocol(rounds=3, consensus="majority")

            if not agents:
                # Post error to Slack
                await self._post_to_response_url(
                    response_url,
                    {
                        "response_type": "in_channel",
                        "text": f"Failed to create debate: No agents available",
                        "replace_original": False,
                    }
                )
                return

            arena = Arena.from_env(env, agents, protocol)
            result = await arena.run()

            # Build result blocks
            consensus_emoji = "" if result.consensus_reached else ""
            confidence_bar = "" * int(result.confidence * 5)

            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"Debate Complete: {topic[:50]}...",
                        "emoji": True,
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Consensus:* {consensus_emoji} {'Yes' if result.consensus_reached else 'No'}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Confidence:* {confidence_bar} {result.confidence:.1%}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Rounds:* {result.rounds_used}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Agents:* {len(agents)}",
                        },
                    ],
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Conclusion:*\n{result.final_answer[:500] if result.final_answer else 'No conclusion reached'}...",
                    },
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": " Agree",
                                "emoji": True,
                            },
                            "action_id": f"vote_{result.id}_agree",
                            "value": result.id,
                        },
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": " Disagree",
                                "emoji": True,
                            },
                            "action_id": f"vote_{result.id}_disagree",
                            "value": result.id,
                        },
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": " Details",
                                "emoji": True,
                            },
                            "action_id": "view_details",
                            "value": result.id,
                        },
                    ],
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"Debate ID: `{result.id}` | Requested by <@{user_id}>",
                        },
                    ],
                },
            ]

            await self._post_to_response_url(
                response_url,
                {
                    "response_type": "in_channel",
                    "text": f"Debate complete: {topic}",
                    "blocks": blocks,
                    "replace_original": False,
                }
            )

        except Exception as e:
            logger.error(f"Async debate creation failed: {e}", exc_info=True)
            await self._post_to_response_url(
                response_url,
                {
                    "response_type": "in_channel",
                    "text": f"Debate failed: {str(e)[:100]}",
                    "replace_original": False,
                }
            )

    async def _post_to_response_url(self, url: str, payload: dict) -> None:
        """POST a message to Slack's response_url."""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.warning(f"Slack response_url POST failed: {response.status} - {text[:100]}")
        except Exception as e:
            logger.error(f"Failed to POST to response_url: {e}")

    @rate_limit(rpm=60, limiter_name="slack_interactive")
    @auto_error_response
    def _handle_interactive(self, handler) -> HandlerResult:
        """Handle interactive component callbacks.

        This handles button clicks, menu selections, etc. from Slack messages.
        """
        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(content_length).decode("utf-8")

            # Interactive payloads come as form-encoded with a 'payload' field
            params = parse_qs(body)
            payload_str = params.get("payload", ["{}"])[0]
            payload = json.loads(payload_str)

            action_type = payload.get("type")
            user = payload.get("user", {})
            user_id = user.get("id", "unknown")

            logger.info(f"Interactive action from {user_id}: {action_type}")

            if action_type == "block_actions":
                actions = payload.get("actions", [])
                if actions:
                    action = actions[0]
                    action_id = action.get("action_id", "")

                    if action_id.startswith("vote_"):
                        return self._handle_vote_action(payload, action)
                    elif action_id == "view_details":
                        return self._handle_view_details(payload, action)

            # Acknowledge the action
            return json_response({"text": "Action received"})

        except Exception as e:
            logger.error(f"Interactive handler error: {e}", exc_info=True)
            return json_response({"text": f"Error: {str(e)[:100]}"})

    def _handle_vote_action(self, payload: dict, action: dict) -> HandlerResult:
        """Handle vote button clicks."""
        action_id = action.get("action_id", "")
        value = action.get("value", "")
        user = payload.get("user", {})
        user_id = user.get("id", "unknown")

        # Extract debate_id and vote from action
        # Expected format: vote_<debate_id>_<option>
        parts = action_id.split("_")
        if len(parts) >= 3:
            debate_id = parts[1]
            vote_option = parts[2]  # 'agree' or 'disagree'

            logger.info(f"Vote received: {debate_id} -> {vote_option} from {user_id}")

            # Record vote in debate system
            try:
                from aragora.server.storage import get_debates_db
                db = get_debates_db()
                if db and hasattr(db, "record_vote"):
                    db.record_vote(
                        debate_id=debate_id,
                        voter_id=f"slack:{user_id}",
                        vote=vote_option,
                        source="slack",
                    )
                    logger.info(f"Vote recorded: {debate_id} -> {vote_option}")
            except Exception as e:
                logger.warning(f"Failed to record vote in storage: {e}")

            # Try to record in vote aggregator if available
            try:
                from aragora.debate.vote_aggregator import VoteAggregator
                aggregator = VoteAggregator.get_instance()
                if aggregator:
                    position = "for" if vote_option == "agree" else "against"
                    aggregator.record_vote(debate_id, f"slack:{user_id}", position)
            except (ImportError, AttributeError) as e:
                logger.debug(f"Vote aggregator not available: {e}")

            emoji = "" if vote_option == "agree" else ""
            return json_response({
                "text": f"{emoji} Your vote for '{vote_option}' has been recorded!",
                "replace_original": False,
            })

        return json_response({"text": "Vote recorded"})

    def _handle_view_details(self, payload: dict, action: dict) -> HandlerResult:
        """Handle view details button clicks."""
        debate_id = action.get("value", "")

        if not debate_id:
            return json_response({
                "text": "Error: No debate ID provided",
                "replace_original": False,
            })

        # Fetch debate details
        debate_data = None
        try:
            from aragora.server.storage import get_debates_db
            db = get_debates_db()
            if db:
                debate_data = db.get(debate_id)
        except Exception as e:
            logger.warning(f"Failed to fetch debate details: {e}")

        if not debate_data:
            return json_response({
                "text": f"Debate `{debate_id}` not found",
                "replace_original": False,
            })

        # Build detailed response
        task = debate_data.get("task", "Unknown topic")
        final_answer = debate_data.get("final_answer", "No conclusion")
        consensus = debate_data.get("consensus_reached", False)
        confidence = debate_data.get("confidence", 0)
        rounds_used = debate_data.get("rounds_used", 0)
        agents = debate_data.get("agents", [])
        created_at = debate_data.get("created_at", "Unknown")

        # Format agent names
        agent_list = ", ".join(agents[:5]) if agents else "Unknown"
        if len(agents) > 5:
            agent_list += f" (+{len(agents) - 5} more)"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Debate Details",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Topic:*\n{task[:200]}",
                },
            },
            {
                "type": "divider",
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Debate ID:*\n`{debate_id}`"},
                    {"type": "mrkdwn", "text": f"*Created:*\n{created_at}"},
                    {"type": "mrkdwn", "text": f"*Consensus:*\n{'Yes' if consensus else 'No'}"},
                    {"type": "mrkdwn", "text": f"*Confidence:*\n{confidence:.1%}"},
                    {"type": "mrkdwn", "text": f"*Rounds:*\n{rounds_used}"},
                    {"type": "mrkdwn", "text": f"*Agents:*\n{agent_list}"},
                ],
            },
            {
                "type": "divider",
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Conclusion:*\n{final_answer[:800] if final_answer else 'No conclusion available'}",
                },
            },
        ]

        return json_response({
            "response_type": "ephemeral",
            "text": f"Details for debate {debate_id}",
            "blocks": blocks,
            "replace_original": False,
        })

    @rate_limit(rpm=100, limiter_name="slack_events")
    @auto_error_response
    def _handle_events(self, handler) -> HandlerResult:
        """Handle Slack Events API callbacks.

        This handles events like app_mention, message, etc.
        """
        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(content_length).decode("utf-8")
            event = json.loads(body)

            event_type = event.get("type")

            # Handle URL verification challenge
            if event_type == "url_verification":
                challenge = event.get("challenge", "")
                return json_response({"challenge": challenge})

            # Handle event callbacks
            if event_type == "event_callback":
                inner_event = event.get("event", {})
                inner_type = inner_event.get("type")

                if inner_type == "app_mention":
                    return self._handle_app_mention(inner_event)
                elif inner_type == "message":
                    return self._handle_message_event(inner_event)

            # Acknowledge unknown events
            return json_response({"ok": True})

        except Exception as e:
            logger.error(f"Events handler error: {e}", exc_info=True)
            return json_response({"ok": True})  # Always 200 for events

    def _handle_app_mention(self, event: dict) -> HandlerResult:
        """Handle @mentions of the app."""
        text = event.get("text", "")
        channel = event.get("channel", "")
        user = event.get("user", "")

        logger.info(f"App mention from {user} in {channel}: {text[:50]}...")

        # Parse the mention to extract command/question
        # Remove the bot mention from the text
        clean_text = re.sub(r'<@[A-Z0-9]+>', '', text).strip()

        if not clean_text:
            # Just mentioned with no text - show help
            response_text = "Hi! You can ask me to:\n• Debate a topic: `@aragora debate \"Should AI be regulated?\"`\n• Show status: `@aragora status`\n• List agents: `@aragora agents`"
        elif clean_text.lower().startswith("debate "):
            topic = clean_text[7:].strip().strip("\"'")
            response_text = f"To start a debate, use the slash command: `/aragora debate \"{topic}\"`"
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
                name=f"slack-reply-{channel}"
            )

        return json_response({"ok": True})

    async def _post_message_async(
        self,
        channel: str,
        text: str,
        thread_ts: Optional[str] = None,
    ) -> None:
        """Post a message to Slack using the Web API."""
        import aiohttp

        if not SLACK_BOT_TOKEN:
            logger.warning("Cannot post message: SLACK_BOT_TOKEN not configured")
            return

        try:
            payload = {
                "channel": channel,
                "text": text,
            }
            if thread_ts:
                payload["thread_ts"] = thread_ts

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
        except Exception as e:
            logger.error(f"Failed to post Slack message: {e}")

    def _handle_message_event(self, event: dict) -> HandlerResult:
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
            response_text = "Hi! Send me a command:\n• `help` - Show available commands\n• `status` - Check system status\n• `agents` - List available agents\n• `debate \"topic\"` - Start a debate"
        elif text.lower() == "help":
            response_text = (
                "*Aragora Direct Message Commands*\n\n"
                "• `help` - Show this message\n"
                "• `status` - Check system status\n"
                "• `agents` - List available agents\n"
                "• `debate \"Your topic here\"` - Start a debate on a topic\n"
                "• `recent` - Show recent debates\n\n"
                "_You can also use `/aragora` commands in any channel._"
            )
        elif text.lower() == "status":
            try:
                from aragora.ranking.elo import get_elo_store
                store = get_elo_store()
                agents = store.get_all_ratings() if store else []
                response_text = f"*Aragora Status*\n• Status: Online\n• Agents: {len(agents)} registered"
            except (ImportError, AttributeError, RuntimeError) as e:
                logger.debug(f"Failed to fetch status: {e}")
                response_text = "*Aragora Status*\n• Status: Online\n• Agents: Unknown"
        elif text.lower() == "agents":
            try:
                from aragora.ranking.elo import get_elo_store
                store = get_elo_store()
                agents = store.get_all_ratings() if store else []
                if agents:
                    agents = sorted(agents, key=lambda a: getattr(a, "elo", 1500), reverse=True)
                    lines = [f"*Top Agents*"]
                    for i, agent in enumerate(agents[:5]):
                        name = getattr(agent, "name", "Unknown")
                        elo = getattr(agent, "elo", 1500)
                        lines.append(f"{i+1}. {name} (ELO: {elo:.0f})")
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
                            consensus = "" if d.get("consensus_reached") else ""
                            lines.append(f"• {consensus} {topic}...")
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
                        name=f"slack-dm-debate-{topic[:30]}"
                    )
        else:
            response_text = f"I don't understand: `{text[:30]}`. Send `help` for available commands."

        # Send response
        if SLACK_BOT_TOKEN:
            create_tracked_task(
                self._post_message_async(channel, response_text),
                name=f"slack-dm-response-{channel}"
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
            from aragora import Arena, Environment, DebateProtocol
            from aragora.agents import get_agents_by_names

            env = Environment(task=f"Debate: {topic}")
            agents = get_agents_by_names(["anthropic-api", "openai-api"])
            protocol = DebateProtocol(rounds=3, consensus="majority")

            if not agents:
                await self._post_message_async(channel, "Failed: No agents available")
                return

            arena = Arena.from_env(env, agents, protocol)
            result = await arena.run()

            consensus_emoji = "" if result.consensus_reached else ""
            response = (
                f"*Debate Complete!* {consensus_emoji}\n\n"
                f"*Topic:* {topic[:100]}...\n"
                f"*Consensus:* {'Yes' if result.consensus_reached else 'No'}\n"
                f"*Confidence:* {result.confidence:.1%}\n"
                f"*Rounds:* {result.rounds_used}\n\n"
                f"*Conclusion:*\n{result.final_answer[:500] if result.final_answer else 'No conclusion'}..."
            )
            await self._post_message_async(channel, response)

        except Exception as e:
            logger.error(f"DM debate creation failed: {e}", exc_info=True)
            await self._post_message_async(channel, f"Debate failed: {str(e)[:100]}")

    def _slack_response(
        self,
        text: str,
        response_type: str = "ephemeral",
    ) -> HandlerResult:
        """Create a simple Slack response."""
        return json_response({
            "response_type": response_type,
            "text": text,
        })

    def _slack_blocks_response(
        self,
        blocks: list,
        text: str,
        response_type: str = "ephemeral",
    ) -> HandlerResult:
        """Create a Slack response with blocks."""
        return json_response({
            "response_type": response_type,
            "text": text,
            "blocks": blocks,
        })


# Export handler factory (lazy instantiation - server_context required)
_slack_handler: Optional["SlackHandler"] = None


def get_slack_handler(server_context: Optional[Dict] = None) -> "SlackHandler":
    """Get or create the Slack handler instance.

    Args:
        server_context: Server context dict (required for first call)

    Returns:
        SlackHandler instance
    """
    global _slack_handler
    if _slack_handler is None:
        if server_context is None:
            server_context = {}  # Default empty context for standalone usage
        _slack_handler = SlackHandler(server_context)
    return _slack_handler


__all__ = ["SlackHandler", "get_slack_handler", "get_slack_integration"]
