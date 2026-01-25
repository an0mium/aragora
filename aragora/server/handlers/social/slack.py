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
- ARAGORA_API_BASE_URL - Base URL for internal API calls (default: http://localhost:8080)
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import time
from typing import Any, Coroutine, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

# Lazy import for audit logger (avoid circular imports)
_slack_audit: Any = None


def _get_audit_logger() -> Any:
    """Get or create Slack audit logger (lazy initialization)."""
    global _slack_audit
    if _slack_audit is None:
        try:
            from aragora.audit.slack_audit import get_slack_audit_logger

            _slack_audit = get_slack_audit_logger()
        except Exception as e:
            logger.debug(f"Slack audit logger not available: {e}")
            _slack_audit = None
    return _slack_audit


# Lazy import for user rate limiter
_slack_user_limiter: Any = None


def _get_user_rate_limiter() -> Any:
    """Get or create user rate limiter for per-user rate limiting."""
    global _slack_user_limiter
    if _slack_user_limiter is None:
        try:
            from aragora.server.middleware.rate_limit.user_limiter import (
                get_user_rate_limiter,
            )

            _slack_user_limiter = get_user_rate_limiter()
        except Exception as e:
            logger.debug(f"User rate limiter not available: {e}")
            _slack_user_limiter = None
    return _slack_user_limiter


# Allowed domains for Slack response URLs (SSRF protection)
SLACK_ALLOWED_DOMAINS = frozenset({"hooks.slack.com", "api.slack.com"})

# Base URL for internal API calls (configurable for production)
ARAGORA_API_BASE_URL = os.environ.get("ARAGORA_API_BASE_URL", "http://localhost:8080")


def _validate_slack_url(url: str) -> bool:
    """Validate that a URL is a legitimate Slack endpoint.

    This prevents SSRF attacks by ensuring we only POST to Slack's servers.

    Args:
        url: The URL to validate

    Returns:
        True if the URL is a valid Slack endpoint, False otherwise
    """
    try:
        parsed = urlparse(url)
        # Must be HTTPS
        if parsed.scheme != "https":
            return False
        # Must be a Slack domain
        if parsed.netloc not in SLACK_ALLOWED_DOMAINS:
            return False
        return True
    except Exception:
        return False


def _handle_task_exception(task: asyncio.Task[Any], task_name: str) -> None:
    """Handle exceptions from fire-and-forget async tasks."""
    if task.cancelled():
        logger.debug(f"Task {task_name} was cancelled")
    elif task.exception():
        exc = task.exception()
        logger.error(f"Task {task_name} failed with exception: {exc}", exc_info=exc)


def create_tracked_task(coro: Coroutine[Any, Any, Any], name: str) -> asyncio.Task[Any]:
    """Create an async task with exception logging.

    Use this instead of raw asyncio.create_task() for fire-and-forget tasks
    to ensure exceptions are logged rather than silently swallowed.
    """
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(lambda t: _handle_task_exception(t, name))
    return task


from ..base import (
    BaseHandler,
    HandlerResult,
    auto_error_response,
    error_response,
    json_response,
)
from ..utils.rate_limit import rate_limit

# Environment variables for Slack integration (fallback for single-workspace mode)
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET", "")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")

# Multi-workspace support
_workspace_store = None


def get_workspace_store():
    """Get the Slack workspace store for multi-workspace support."""
    global _workspace_store
    if _workspace_store is None:
        try:
            from aragora.storage.slack_workspace_store import get_slack_workspace_store

            _workspace_store = get_slack_workspace_store()
        except ImportError:
            logger.debug("Slack workspace store not available")
    return _workspace_store


def resolve_workspace(team_id: str):
    """Resolve a workspace by team_id.

    Returns workspace object if found, None otherwise.
    Falls back to environment variable configuration if no store configured.
    """
    if not team_id:
        return None

    store = get_workspace_store()
    if store:
        try:
            return store.get(team_id)
        except Exception as e:
            logger.debug(f"Failed to get workspace {team_id}: {e}")

    return None


# Patterns for command parsing
COMMAND_PATTERN = re.compile(r"^/aragora\s+(\w+)(?:\s+(.*))?$")
TOPIC_PATTERN = re.compile(r'^["\']?(.+?)["\']?$')


_slack_integration: Optional[Any] = None


def get_slack_integration() -> Optional[Any]:
    """Get or create the Slack integration singleton."""
    global _slack_integration
    if _slack_integration is None:
        if not SLACK_WEBHOOK_URL:
            logger.debug("Slack integration disabled (no SLACK_WEBHOOK_URL)")
            return None
        try:
            from aragora.integrations.slack import SlackConfig, SlackIntegration

            config = SlackConfig(webhook_url=SLACK_WEBHOOK_URL)
            _slack_integration = SlackIntegration(config)
            logger.info("Slack integration initialized")
        except ImportError as e:
            logger.warning(f"Slack integration module not available: {e}")
            return None
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Invalid Slack configuration: {e}")
            return None
        except Exception as e:
            logger.exception(f"Unexpected error initializing Slack integration: {e}")
            return None
    return _slack_integration


class SlackHandler(BaseHandler):
    """Handler for Slack integration endpoints."""

    ROUTES = [
        "/api/v1/integrations/slack/commands",
        "/api/v1/integrations/slack/interactive",
        "/api/v1/integrations/slack/events",
        "/api/v1/integrations/slack/status",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route Slack requests to appropriate methods."""
        logger.debug(f"Slack request: {path}")

        if path == "/api/v1/integrations/slack/status":
            return self._get_status()

        # All other endpoints require POST
        if handler.command != "POST":
            return error_response("Method not allowed", 405)

        # Read and store body for signature verification and parsing
        content_length = int(handler.headers.get("Content-Length", 0))
        body = handler.rfile.read(content_length).decode("utf-8")

        # Extract team_id for multi-workspace support
        team_id = self._extract_team_id(body, path)
        workspace = resolve_workspace(team_id) if team_id else None

        # Get signing secret (workspace-specific or fallback to env var)
        signing_secret = (
            workspace.signing_secret
            if workspace and workspace.signing_secret
            else SLACK_SIGNING_SECRET
        )

        # Verify Slack signature for security
        if signing_secret and not self._verify_signature(handler, body, signing_secret):
            logger.warning(f"Slack signature verification failed for team_id={team_id}")
            # Audit log signature failure (potential attack)
            audit = _get_audit_logger()
            if audit:
                ip_address = handler.client_address[0] if handler.client_address else ""
                user_agent = handler.headers.get("User-Agent", "")
                audit.log_signature_failure(
                    workspace_id=team_id or "",
                    ip_address=ip_address,
                    user_agent=user_agent,
                )
            return error_response("Invalid signature", 401)

        # Store workspace and body in handler for downstream methods
        handler._slack_workspace = workspace
        handler._slack_body = body
        handler._slack_team_id = team_id

        if path == "/api/v1/integrations/slack/commands":
            return self._handle_slash_command(handler)
        elif path == "/api/v1/integrations/slack/interactive":
            return self._handle_interactive(handler)
        elif path == "/api/v1/integrations/slack/events":
            return self._handle_events(handler)

        return error_response("Not found", 404)

    def _extract_team_id(self, body: str, path: str) -> Optional[str]:
        """Extract team_id from request body based on endpoint type.

        Args:
            body: Raw request body
            path: Request path to determine parsing strategy

        Returns:
            team_id string or None
        """
        try:
            if path.endswith("/commands"):
                # Slash commands are form-encoded
                params = parse_qs(body)
                return params.get("team_id", [None])[0]
            elif path.endswith("/interactive"):
                # Interactive payloads are JSON in 'payload' field
                params = parse_qs(body)
                payload_str = params.get("payload", ["{}"])[0]
                payload = json.loads(payload_str)
                # Team info can be in 'team' or root
                team = payload.get("team", {})
                return team.get("id") or payload.get("team_id")
            elif path.endswith("/events"):
                # Events API sends JSON
                data = json.loads(body)
                # Team ID in event or root
                return data.get("team_id") or data.get("event", {}).get("team")
        except Exception as e:
            logger.debug(f"Failed to extract team_id: {e}")
        return None

    def handle_post(self, path: str, body: Dict[str, Any], handler: Any) -> Optional[HandlerResult]:
        """Handle POST requests."""
        return self.handle(path, {}, handler)

    def _verify_signature(self, handler: Any, body: str, signing_secret: str) -> bool:
        """Verify Slack request signature.

        Slack uses HMAC-SHA256 to sign requests.
        See: https://api.slack.com/authentication/verifying-requests-from-slack

        Args:
            handler: HTTP request handler
            body: Pre-read request body
            signing_secret: Signing secret to use (workspace-specific or global)
        """
        if not signing_secret:
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

            # Compute signature
            sig_basestring = f"v0:{timestamp}:{body}"
            expected_sig = (
                "v0="
                + hmac.new(
                    signing_secret.encode(),
                    sig_basestring.encode(),
                    hashlib.sha256,
                ).hexdigest()
            )

            # Timing-safe comparison
            return hmac.compare_digest(expected_sig, signature)

        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid Slack signature format: {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected signature verification error: {e}")
            return False

    def _get_status(self) -> HandlerResult:
        """Get Slack integration status."""
        integration = get_slack_integration()
        return json_response(
            {
                "enabled": integration is not None,
                "signing_secret_configured": bool(SLACK_SIGNING_SECRET),
                "bot_token_configured": bool(SLACK_BOT_TOKEN),
                "webhook_configured": bool(SLACK_WEBHOOK_URL),
            }
        )

    @auto_error_response("handle slack slash command")
    @rate_limit(rpm=30, limiter_name="slack_commands")
    def _handle_slash_command(self, handler: Any) -> HandlerResult:
        """Handle Slack slash commands.

        Expected format: /aragora <command> [args]

        Commands:
        - /aragora debate "topic" - Start a debate on a topic
        - /aragora status - Get system status
        - /aragora help - Show available commands
        """
        start_time = time.time()
        command = ""
        subcommand = ""
        user_id = ""
        channel_id = ""
        team_id: Optional[str] = None

        try:
            # Parse form-encoded body (already read and stored in handle())
            body = getattr(handler, "_slack_body", "")
            params = parse_qs(body)
            workspace = getattr(handler, "_slack_workspace", None)
            team_id = getattr(handler, "_slack_team_id", None)

            command = params.get("command", [""])[0]
            text = params.get("text", [""])[0].strip()
            user_id = params.get("user_id", [""])[0]
            channel_id = params.get("channel_id", [""])[0]
            response_url = params.get("response_url", [""])[0]

            logger.info(f"Slack command from {user_id}: {command} {text}")

            # Parse the subcommand
            if not text:
                result = self._command_help()
                subcommand = "help"
            else:
                parts = text.split(maxsplit=1)
                subcommand = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                if subcommand == "help":
                    result = self._command_help()
                elif subcommand == "status":
                    result = self._command_status()
                elif subcommand == "debate":
                    result = self._command_debate(
                        args, user_id, channel_id, response_url, workspace, team_id
                    )
                elif subcommand == "gauntlet":
                    result = self._command_gauntlet(
                        args, user_id, channel_id, response_url, workspace, team_id
                    )
                elif subcommand == "agents":
                    result = self._command_agents()
                elif subcommand == "ask":
                    result = self._command_ask(
                        args, user_id, channel_id, response_url, workspace, team_id
                    )
                elif subcommand == "search":
                    result = self._command_search(args)
                elif subcommand == "leaderboard":
                    result = self._command_leaderboard()
                elif subcommand == "recent":
                    result = self._command_recent()
                else:
                    result = self._slack_response(
                        f"Unknown command: `{subcommand}`. Use `/aragora help` for available commands.",
                        response_type="ephemeral",
                    )

            # Audit log successful command
            audit = _get_audit_logger()
            if audit:
                response_time_ms = (time.time() - start_time) * 1000
                audit.log_command(
                    workspace_id=team_id or "",
                    user_id=user_id,
                    command=f"{command} {subcommand}".strip(),
                    args=args if "args" in dir() else "",
                    result="success",
                    channel_id=channel_id,
                    response_time_ms=response_time_ms,
                )

            return result

        except Exception as e:
            logger.error(f"Slash command error: {e}", exc_info=True)

            # Audit log error
            audit = _get_audit_logger()
            if audit:
                response_time_ms = (time.time() - start_time) * 1000
                audit.log_command(
                    workspace_id=team_id or "",
                    user_id=user_id,
                    command=f"{command} {subcommand}".strip(),
                    result="error",
                    channel_id=channel_id,
                    response_time_ms=response_time_ms,
                    error=str(e)[:200],
                )

            return self._slack_response(
                f"Error processing command: {str(e)[:100]}",
                response_type="ephemeral",
            )

    def _command_help(self) -> HandlerResult:
        """Show help message."""
        help_text = """*Aragora Slash Commands*

*Core Commands:*
`/aragora debate "topic"` - Start a multi-agent debate on a topic
`/aragora ask "question"` - Quick Q&A without full debate
`/aragora gauntlet "statement"` - Run adversarial stress-test validation

*Discovery:*
`/aragora search "query"` - Search debates and evidence
`/aragora recent` - Show recent debates
`/aragora leaderboard` - View agent rankings

*Info:*
`/aragora agents` - List available agents
`/aragora status` - Get system status
`/aragora help` - Show this help message

*Examples:*
- `/aragora debate "Should AI be regulated?"`
- `/aragora ask "What is the capital of France?"`
- `/aragora gauntlet "We should migrate to microservices"`
- `/aragora search "machine learning"`
"""
        return self._slack_response(help_text, response_type="ephemeral")

    def _command_status(self) -> HandlerResult:
        """Get system status."""
        try:
            # Get basic stats
            from aragora.ranking.elo import EloSystem

            store = EloSystem()
            agents = store.get_all_ratings()

            blocks: List[Dict[str, Any]] = [
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

        except ImportError as e:
            logger.warning(f"ELO system not available for status: {e}")
            return self._slack_response(
                "Status service temporarily unavailable",
                response_type="ephemeral",
            )
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Data error in status command: {e}")
            return self._slack_response(
                f"Error getting status: {str(e)[:100]}",
                response_type="ephemeral",
            )
        except Exception as e:
            logger.exception(f"Unexpected status command error: {e}")
            return self._slack_response(
                f"Error getting status: {str(e)[:100]}",
                response_type="ephemeral",
            )

    def _command_agents(self) -> HandlerResult:
        """List available agents."""
        try:
            from aragora.ranking.elo import EloSystem

            store = EloSystem()
            agents = store.get_all_ratings()

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
                medal = ["", "", ""][i] if i < 3 else f"{i + 1}."
                text += f"{medal} *{name}* - ELO: {elo:.0f} | Wins: {wins}\n"

            return self._slack_response(text, response_type="ephemeral")

        except ImportError as e:
            logger.warning(f"ELO system not available for agents listing: {e}")
            return self._slack_response(
                "Agents service temporarily unavailable",
                response_type="ephemeral",
            )
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Data error in agents command: {e}")
            return self._slack_response(
                f"Error listing agents: {str(e)[:100]}",
                response_type="ephemeral",
            )
        except Exception as e:
            logger.exception(f"Unexpected agents command error: {e}")
            return self._slack_response(
                f"Error listing agents: {str(e)[:100]}",
                response_type="ephemeral",
            )

    def _command_ask(
        self,
        args: str,
        user_id: str,
        channel_id: str,
        response_url: str,
        workspace: Optional[Any] = None,
        team_id: Optional[str] = None,
    ) -> HandlerResult:
        """Quick Q&A without full debate - uses single agent for fast answers.

        Args:
            args: The question to answer
            user_id: Slack user ID
            channel_id: Slack channel ID
            response_url: URL for async responses
            workspace: Resolved workspace object (for multi-workspace)
            team_id: Slack team/workspace ID
        """
        if not args:
            return self._slack_response(
                'Please provide a question. Example: `/aragora ask "What is the capital of France?"`',
                response_type="ephemeral",
            )

        # Strip quotes if present
        question = args.strip().strip("\"'")

        if len(question) < 5:
            return self._slack_response(
                "Question is too short. Please provide more detail.",
                response_type="ephemeral",
            )

        if len(question) > 500:
            return self._slack_response(
                "Question is too long. Please limit to 500 characters.",
                response_type="ephemeral",
            )

        # Acknowledge immediately
        blocks: List[Dict[str, Any]] = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Processing question:*\n_{question[:200]}{'...' if len(question) > 200 else ''}_",
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Asked by <@{user_id}> | Thinking...",
                    },
                ],
            },
        ]

        # Queue the question asynchronously
        if response_url:
            create_tracked_task(
                self._answer_question_async(question, response_url, user_id, channel_id),
                name=f"slack-ask-{question[:30]}",
            )

        return self._slack_blocks_response(
            blocks,
            text=f"Processing: {question[:50]}...",
            response_type="in_channel",
        )

    async def _answer_question_async(
        self,
        question: str,
        response_url: str,
        user_id: str,
        channel_id: str,
    ) -> None:
        """Answer a question asynchronously using a single agent."""
        import aiohttp

        try:
            # Call the quick answer API endpoint
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{ARAGORA_API_BASE_URL}/api/quick-answer",
                    json={
                        "question": question,
                        "metadata": {
                            "source": "slack",
                            "channel_id": channel_id,
                            "user_id": user_id,
                        },
                    },
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status != 200:
                        # Fallback to debate API if quick-answer not available
                        data = {"answer": None, "error": "Quick answer service unavailable"}
                    else:
                        data = await resp.json()

                    answer = data.get("answer")
                    if not answer:
                        # Fallback: use single-round debate
                        async with session.post(
                            f"{ARAGORA_API_BASE_URL}/api/debates",
                            json={
                                "task": question,
                                "rounds": 1,
                                "agents": ["anthropic-api"],
                            },
                            timeout=aiohttp.ClientTimeout(total=60),
                        ) as debate_resp:
                            if debate_resp.status == 200 or debate_resp.status == 201:
                                debate_data = await debate_resp.json()
                                answer = debate_data.get(
                                    "final_answer", "Unable to generate answer."
                                )
                            else:
                                answer = "Unable to generate answer at this time."

                    # Build response blocks
                    blocks = [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*Question:*\n_{question[:200]}{'...' if len(question) > 200 else ''}_",
                            },
                        },
                        {
                            "type": "divider",
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*Answer:*\n{answer[:2000] if answer else 'No answer available'}",
                            },
                        },
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"Asked by <@{user_id}>",
                                },
                            ],
                        },
                    ]

                    await self._post_to_response_url(
                        response_url,
                        {
                            "response_type": "in_channel",
                            "text": f"Answer: {answer[:100] if answer else 'No answer'}...",
                            "blocks": blocks,
                            "replace_original": False,
                        },
                    )

        except Exception as e:
            logger.error(f"Async question answering failed: {e}", exc_info=True)
            await self._post_to_response_url(
                response_url,
                {
                    "response_type": "in_channel",
                    "text": f"Failed to answer question: {str(e)[:100]}",
                    "replace_original": False,
                },
            )

    def _command_search(self, args: str) -> HandlerResult:
        """Search debates and evidence."""
        if not args:
            return self._slack_response(
                'Please provide a search query. Example: `/aragora search "machine learning"`',
                response_type="ephemeral",
            )

        query = args.strip().strip("\"'")

        if len(query) < 2:
            return self._slack_response(
                "Search query is too short.",
                response_type="ephemeral",
            )

        try:
            from aragora.server.storage import get_debates_db

            db = get_debates_db()
            results = []

            if db and hasattr(db, "search"):
                results = db.search(query, limit=5)
            elif db and hasattr(db, "list"):
                # Fallback: manual search through recent debates
                all_debates = db.list(limit=50)
                query_lower = query.lower()
                for d in all_debates:
                    task = d.get("task", "")
                    answer = d.get("final_answer", "")
                    if query_lower in task.lower() or query_lower in answer.lower():
                        results.append(d)
                        if len(results) >= 5:
                            break

            if not results:
                return self._slack_response(
                    f"No results found for: `{query}`",
                    response_type="ephemeral",
                )

            blocks: List[Dict[str, Any]] = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"Search Results: {query[:30]}",
                        "emoji": True,
                    },
                },
            ]

            for i, item in enumerate(results[:5]):
                # Handle both dict and object formats
                if isinstance(item, dict):
                    topic = item.get("task", "Unknown")[:60]
                    consensus = "" if item.get("consensus_reached") else ""
                    debate_id = str(item.get("id", "unknown"))
                    confidence = item.get("confidence", 0)
                else:
                    # Object format
                    topic = str(getattr(item, "task", "Unknown"))[:60]
                    consensus = "" if getattr(item, "consensus_reached", False) else ""
                    debate_id = str(getattr(item, "id", "unknown"))
                    confidence = getattr(item, "confidence", 0)

                blocks.append(
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*{i + 1}. {consensus} {topic}*\nConfidence: {confidence:.0%} | ID: `{debate_id[:8]}`",
                        },
                    }
                )

            blocks.append(
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"Found {len(results)} result(s)",
                        },
                    ],
                }
            )

            return self._slack_blocks_response(
                blocks,
                text=f"Found {len(results)} results for '{query}'",
                response_type="ephemeral",
            )

        except ImportError as e:
            logger.warning(f"Storage not available for search: {e}")
            return self._slack_response(
                "Search service temporarily unavailable",
                response_type="ephemeral",
            )
        except Exception as e:
            logger.exception(f"Unexpected search error: {e}")
            return self._slack_response(
                f"Search failed: {str(e)[:100]}",
                response_type="ephemeral",
            )

    def _command_leaderboard(self) -> HandlerResult:
        """Show agent rankings leaderboard."""
        try:
            from aragora.ranking.elo import EloSystem

            store = EloSystem()
            agents = store.get_all_ratings()

            if not agents:
                return self._slack_response(
                    "No agents ranked yet. Start some debates first!",
                    response_type="ephemeral",
                )

            # Sort by ELO
            agents = sorted(agents, key=lambda a: getattr(a, "elo", 1500), reverse=True)

            blocks: List[Dict[str, Any]] = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "Agent Leaderboard",
                        "emoji": True,
                    },
                },
            ]

            # Build leaderboard table
            leaderboard_text = "```\n"
            leaderboard_text += f"{'Rank':<5} {'Agent':<20} {'ELO':<8} {'W/L':<10}\n"
            leaderboard_text += "-" * 45 + "\n"

            for i, agent in enumerate(agents[:10]):
                name = getattr(agent, "name", "Unknown")[:18]
                elo = getattr(agent, "elo", 1500)
                wins = getattr(agent, "wins", 0)
                losses = getattr(agent, "losses", 0)
                medal = ["", "", ""][i] if i < 3 else f"{i + 1}."
                leaderboard_text += f"{medal:<5} {name:<20} {elo:<8.0f} {wins}/{losses}\n"

            leaderboard_text += "```"

            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": leaderboard_text,
                    },
                }
            )

            blocks.append(
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"Total agents: {len(agents)} | Rankings based on debate performance",
                        },
                    ],
                }
            )

            return self._slack_blocks_response(
                blocks,
                text="Agent Leaderboard",
                response_type="in_channel",
            )

        except ImportError as e:
            logger.warning(f"ELO system not available for leaderboard: {e}")
            return self._slack_response(
                "Leaderboard service temporarily unavailable",
                response_type="ephemeral",
            )
        except Exception as e:
            logger.exception(f"Unexpected leaderboard error: {e}")
            return self._slack_response(
                f"Leaderboard failed: {str(e)[:100]}",
                response_type="ephemeral",
            )

    def _command_recent(self) -> HandlerResult:
        """Show recent debates."""
        try:
            from aragora.server.storage import get_debates_db

            db = get_debates_db()
            if not db or not hasattr(db, "list"):
                return self._slack_response(
                    "Debate history not available",
                    response_type="ephemeral",
                )

            debates = db.list(limit=10)

            if not debates:
                return self._slack_response(
                    'No recent debates found. Start one with `/aragora debate "Your topic"`',
                    response_type="ephemeral",
                )

            blocks: List[Dict[str, Any]] = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "Recent Debates",
                        "emoji": True,
                    },
                },
            ]

            for i, debate in enumerate(debates[:10]):
                # Handle both dict and object formats
                if isinstance(debate, dict):
                    full_topic = debate.get("task", "Unknown topic")
                    consensus = "" if debate.get("consensus_reached") else ""
                    confidence = debate.get("confidence", 0)
                    debate_id = str(debate.get("id", "unknown"))
                    created = str(debate.get("created_at", ""))[:10]
                else:
                    full_topic = str(getattr(debate, "task", "Unknown topic"))
                    consensus = "" if getattr(debate, "consensus_reached", False) else ""
                    confidence = getattr(debate, "confidence", 0)
                    debate_id = str(getattr(debate, "id", "unknown"))
                    created = str(getattr(debate, "created_at", ""))[:10]

                topic = full_topic[:50]
                needs_ellipsis = len(full_topic) > 50

                blocks.append(
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*{i + 1}. {consensus} {topic}{'...' if needs_ellipsis else ''}*\n{confidence:.0%} confidence | {created} | `{debate_id[:8]}`",
                        },
                        "accessory": {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Details"},
                            "action_id": "view_details",
                            "value": debate_id,
                        },
                    }
                )

            blocks.append(
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"Showing {len(debates)} most recent debates",
                        },
                    ],
                }
            )

            return self._slack_blocks_response(
                blocks,
                text="Recent Debates",
                response_type="ephemeral",
            )

        except ImportError as e:
            logger.warning(f"Storage not available for recent debates: {e}")
            return self._slack_response(
                "Recent debates service temporarily unavailable",
                response_type="ephemeral",
            )
        except Exception as e:
            logger.exception(f"Unexpected recent debates error: {e}")
            return self._slack_response(
                f"Failed to get recent debates: {str(e)[:100]}",
                response_type="ephemeral",
            )

    def _command_gauntlet(
        self,
        args: str,
        user_id: str,
        channel_id: str,
        response_url: str,
        workspace: Optional[Any] = None,
        team_id: Optional[str] = None,
    ) -> HandlerResult:
        """Run gauntlet adversarial validation on a statement.

        Args:
            args: The statement to validate
            user_id: Slack user ID
            channel_id: Slack channel ID
            response_url: URL for async responses
            workspace: Resolved workspace object (for multi-workspace)
            team_id: Slack team/workspace ID
        """
        if not args:
            return self._slack_response(
                'Please provide a statement to stress-test. Example: `/aragora gauntlet "We should migrate to microservices"`',
                response_type="ephemeral",
            )

        # Strip quotes if present
        statement = args.strip().strip("\"'")

        if len(statement) < 10:
            return self._slack_response(
                "Statement is too short. Please provide more detail.",
                response_type="ephemeral",
            )

        if len(statement) > 1000:
            return self._slack_response(
                "Statement is too long. Please limit to 1000 characters.",
                response_type="ephemeral",
            )

        # Acknowledge immediately
        blocks: List[Dict[str, Any]] = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Running Gauntlet stress-test on:*\n_{statement[:200]}{'...' if len(statement) > 200 else ''}_",
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Requested by <@{user_id}> | Running adversarial validation...",
                    },
                ],
            },
        ]

        # Queue the gauntlet run asynchronously
        if response_url:
            create_tracked_task(
                self._run_gauntlet_async(statement, response_url, user_id, channel_id, team_id),
                name=f"slack-gauntlet-{statement[:30]}",
            )

        return self._slack_blocks_response(
            blocks,
            text=f"Running Gauntlet: {statement[:50]}...",
            response_type="in_channel",
        )

    async def _run_gauntlet_async(
        self,
        statement: str,
        response_url: str,
        user_id: str,
        channel_id: str,
        workspace_id: Optional[str] = None,
    ) -> None:
        """Run gauntlet asynchronously and POST result to Slack."""
        import aiohttp

        try:
            # Call the gauntlet API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{ARAGORA_API_BASE_URL}/api/gauntlet/run",
                    json={
                        "statement": statement,
                        "intensity": "medium",
                        "metadata": {
                            "source": "slack",
                            "channel_id": channel_id,
                            "user_id": user_id,
                        },
                    },
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    data = await resp.json()

                    if resp.status != 200:
                        await self._post_to_response_url(
                            response_url,
                            {
                                "response_type": "in_channel",
                                "text": f"Gauntlet failed: {data.get('error', 'Unknown error')}",
                                "replace_original": False,
                            },
                        )
                        return

                    # Build result blocks
                    run_id = data.get("run_id", "unknown")
                    score = data.get("score", 0)
                    passed = data.get("passed", False)
                    vulnerabilities = data.get("vulnerabilities", [])

                    score_bar = "" * int(score * 5) + "" * (5 - int(score * 5))
                    status_emoji = "" if passed else ""

                    blocks = [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": f"{status_emoji} Gauntlet Results",
                                "emoji": True,
                            },
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*Statement:*\n_{statement[:200]}{'...' if len(statement) > 200 else ''}_",
                            },
                        },
                        {
                            "type": "section",
                            "fields": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Score:* {score_bar} {score:.1%}",
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Status:* {'Passed' if passed else 'Failed'}",
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Vulnerabilities:* {len(vulnerabilities)}",
                                },
                            ],
                        },
                    ]

                    if vulnerabilities:
                        vuln_text = "*Issues Found:*\n"
                        for v in vulnerabilities[:5]:
                            vuln_text += f"• {v.get('description', 'Unknown issue')[:100]}\n"
                        if len(vulnerabilities) > 5:
                            vuln_text += f"_...and {len(vulnerabilities) - 5} more_"

                        blocks.append(
                            {
                                "type": "section",
                                "text": {"type": "mrkdwn", "text": vuln_text},
                            }
                        )

                    blocks.append(
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"Run ID: `{run_id}` | Requested by <@{user_id}>",
                                },
                            ],
                        }
                    )

                    await self._post_to_response_url(
                        response_url,
                        {
                            "response_type": "in_channel",
                            "text": f"Gauntlet complete: {statement[:50]}...",
                            "blocks": blocks,
                            "replace_original": False,
                        },
                    )

        except Exception as e:
            logger.error(f"Async gauntlet failed: {e}", exc_info=True)
            await self._post_to_response_url(
                response_url,
                {
                    "response_type": "in_channel",
                    "text": f"Gauntlet failed: {str(e)[:100]}",
                    "replace_original": False,
                },
            )

    def _command_debate(
        self,
        args: str,
        user_id: str,
        channel_id: str,
        response_url: str,
        workspace: Optional[Any] = None,
        team_id: Optional[str] = None,
    ) -> HandlerResult:
        """Start a debate on a topic.

        Args:
            args: The topic text (may be quoted)
            user_id: Slack user ID
            channel_id: Slack channel ID
            response_url: URL for async responses
            workspace: Resolved workspace object (for multi-workspace)
            team_id: Slack team/workspace ID
        """
        if not args:
            return self._slack_response(
                'Please provide a topic. Example: `/aragora debate "Should AI be regulated?"`',
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

        blocks: List[Dict[str, Any]] = [
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
                self._create_debate_async(topic, response_url, user_id, channel_id, team_id),
                name=f"slack-debate-{topic[:30]}",
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
        workspace_id: Optional[str] = None,
    ) -> None:
        """Create debate asynchronously with thread-based progress updates.

        Posts an initial message to start a thread, then posts progress
        updates and final result to that thread.
        """
        import uuid

        debate_id = f"debate-{uuid.uuid4().hex[:8]}"
        thread_ts: Optional[str] = None

        try:
            from aragora import Arena, DebateProtocol, Environment
            from aragora.agents import get_agents_by_names  # type: ignore[attr-defined]

            # Post initial "starting" message to create thread
            starting_blocks = self._build_starting_blocks(topic, user_id, debate_id)
            starting_text = f"Starting debate: {topic}"

            # Use Web API if bot token available (to capture thread_ts for tracking)
            if SLACK_BOT_TOKEN and channel_id:
                thread_ts = await self._post_message_async(
                    channel=channel_id,
                    text=starting_text,
                    blocks=starting_blocks,
                )
                if thread_ts:
                    logger.debug(f"Debate {debate_id} started thread: {thread_ts}")
                else:
                    # Fall back to response_url if Web API failed
                    logger.warning("Web API post failed, falling back to response_url")
                    await self._post_to_response_url(
                        response_url,
                        {
                            "response_type": "in_channel",
                            "text": starting_text,
                            "blocks": starting_blocks,
                            "replace_original": False,
                        },
                    )
            else:
                # No bot token - use response_url (can't track thread_ts)
                await self._post_to_response_url(
                    response_url,
                    {
                        "response_type": "in_channel",
                        "text": starting_text,
                        "blocks": starting_blocks,
                        "replace_original": False,
                    },
                )

            # Store active debate for tracking (if workspace_id available)
            if workspace_id:
                try:
                    from aragora.storage.slack_debate_store import (
                        SlackActiveDebate,
                        get_slack_debate_store,
                    )

                    active_debate = SlackActiveDebate(
                        debate_id=debate_id,
                        workspace_id=workspace_id,
                        channel_id=channel_id,
                        thread_ts=thread_ts,
                        topic=topic,
                        user_id=user_id,
                        status="running",
                    )
                    store = get_slack_debate_store()
                    store.save(active_debate)
                except ImportError:
                    logger.debug("Slack debate store not available")

            # Create debate
            env = Environment(task=f"Debate: {topic}")
            agents = get_agents_by_names(["anthropic-api", "openai-api"])
            protocol = DebateProtocol(
                rounds=3,
                consensus="majority",
                convergence_detection=False,
                early_stopping=False,
            )

            if not agents:
                await self._post_to_response_url(
                    response_url,
                    {
                        "response_type": "in_channel",
                        "text": "Failed to create debate: No agents available",
                        "replace_original": False,
                    },
                )
                self._update_debate_status(debate_id, "failed", error="No agents available")
                return

            # Track progress for thread updates
            last_round = 0
            # Capture thread_ts for closure (may be None if Web API not used)
            debate_thread_ts = thread_ts

            def on_round_complete(round_num: int, agent: str, response: str) -> None:
                nonlocal last_round
                if round_num > last_round:
                    last_round = round_num
                    # Post round update to thread (fire-and-forget)
                    create_tracked_task(
                        self._post_round_update(
                            response_url,
                            topic,
                            round_num,
                            protocol.rounds,
                            agent,
                            channel_id=channel_id,
                            thread_ts=debate_thread_ts,
                        ),
                        name=f"slack-round-{debate_id}-{round_num}",
                    )

            arena = Arena.from_env(env, agents, protocol)
            result = await arena.run()

            # Generate decision receipt if enabled
            receipt_id: Optional[str] = None
            receipt_url: Optional[str] = None
            try:
                from aragora.gauntlet.receipt import DecisionReceipt

                receipt = DecisionReceipt.from_debate_result(result)
                receipt_id = receipt.receipt_id

                # Build receipt URL
                base_url = os.environ.get("ARAGORA_PUBLIC_URL", "https://aragora.ai")
                receipt_url = f"{base_url}/receipts/{receipt_id}"

                # Persist receipt
                try:
                    from aragora.storage.receipt_store import get_receipt_store

                    receipt_store = get_receipt_store()
                    receipt_store.save(receipt.to_dict())
                except ImportError:
                    logger.debug("Receipt store not available")
            except ImportError:
                logger.debug("Receipt generation not available")

            # Build and post result blocks
            result_blocks = self._build_result_blocks(topic, result, user_id, receipt_url)
            result_text = f"Debate complete: {topic}"

            # Use Web API with thread_ts for proper threading when available
            if SLACK_BOT_TOKEN and channel_id and thread_ts:
                await self._post_message_async(
                    channel=channel_id,
                    text=result_text,
                    thread_ts=thread_ts,
                    blocks=result_blocks,
                )
            else:
                await self._post_to_response_url(
                    response_url,
                    {
                        "response_type": "in_channel",
                        "text": result_text,
                        "blocks": result_blocks,
                        "replace_original": False,
                    },
                )

            # Update debate status to completed
            self._update_debate_status(debate_id, "completed", receipt_id=receipt_id)

        except Exception as e:
            logger.error(f"Async debate creation failed: {e}", exc_info=True)
            error_text = f"Debate failed: {str(e)[:100]}"

            # Use Web API with thread_ts for error message when available
            if SLACK_BOT_TOKEN and channel_id and thread_ts:
                await self._post_message_async(
                    channel=channel_id,
                    text=error_text,
                    thread_ts=thread_ts,
                )
            else:
                await self._post_to_response_url(
                    response_url,
                    {
                        "response_type": "in_channel",
                        "text": error_text,
                        "replace_original": False,
                    },
                )
            self._update_debate_status(debate_id, "failed", error=str(e)[:200])

    def _build_starting_blocks(
        self, topic: str, user_id: str, debate_id: str
    ) -> List[Dict[str, Any]]:
        """Build Slack blocks for debate start message."""
        return [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Debate Starting...",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Topic:* {topic}",
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Requested by <@{user_id}> | ID: `{debate_id}`",
                    },
                ],
            },
        ]

    async def _post_round_update(
        self,
        response_url: str,
        topic: str,
        round_num: int,
        total_rounds: int,
        agent: str,
        channel_id: Optional[str] = None,
        thread_ts: Optional[str] = None,
    ) -> None:
        """Post a round progress update to the thread.

        Args:
            response_url: Slack response URL (webhook)
            topic: Debate topic
            round_num: Current round number
            total_rounds: Total rounds in debate
            agent: Name of agent that responded
            channel_id: Optional channel ID for Web API posting
            thread_ts: Optional thread timestamp for threaded replies
        """
        progress = round_num / total_rounds
        progress_bar = "" * int(progress * 10) + "" * (10 - int(progress * 10))

        text = f"Round {round_num} complete"
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Round {round_num}/{total_rounds}* {progress_bar}\n_{agent} responded_",
                },
            },
        ]

        # Use Web API with thread_ts when available for proper threading
        if SLACK_BOT_TOKEN and channel_id and thread_ts:
            await self._post_message_async(
                channel=channel_id,
                text=text,
                thread_ts=thread_ts,
                blocks=blocks,
            )
        else:
            # Fall back to response_url (not threaded)
            await self._post_to_response_url(
                response_url,
                {
                    "response_type": "in_channel",
                    "text": text,
                    "blocks": blocks,
                    "replace_original": False,
                },
            )

    def _build_result_blocks(
        self,
        topic: str,
        result: Any,
        user_id: str,
        receipt_url: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Build Slack blocks for debate result message."""
        consensus_emoji = "" if result.consensus_reached else ""
        confidence_bar = "" * int(result.confidence * 5)

        blocks: List[Dict[str, Any]] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"Debate Complete: {topic[:50]}",
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
                        "text": f"*Participants:* {len(result.participants)}",
                    },
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Conclusion:*\n{result.final_answer[:500] if result.final_answer else 'No conclusion reached'}",
                },
            },
        ]

        # Add action buttons
        action_elements: List[Dict[str, Any]] = [
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "Agree", "emoji": True},
                "action_id": f"vote_{result.id}_agree",
                "value": result.id,
                "style": "primary",
            },
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "Disagree", "emoji": True},
                "action_id": f"vote_{result.id}_disagree",
                "value": result.id,
            },
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "Details", "emoji": True},
                "action_id": "view_details",
                "value": result.id,
            },
        ]

        # Add receipt link button if available
        if receipt_url:
            action_elements.append(
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "View Receipt", "emoji": True},
                    "url": receipt_url,
                    "action_id": f"receipt_{result.id}",
                }
            )

        blocks.append({"type": "actions", "elements": action_elements})

        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Debate ID: `{result.id}` | Requested by <@{user_id}>",
                    },
                ],
            }
        )

        return blocks

    def _update_debate_status(
        self,
        debate_id: str,
        status: str,
        receipt_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update debate status in the store."""
        try:
            from aragora.storage.slack_debate_store import get_slack_debate_store

            store = get_slack_debate_store()
            store.update_status(debate_id, status, receipt_id=receipt_id, error_message=error)
        except ImportError:
            pass  # Store not available

    async def _post_to_response_url(self, url: str, payload: Dict[str, Any]) -> None:
        """POST a message to Slack's response_url.

        Includes SSRF protection by validating the URL is a Slack endpoint.
        """
        # Validate URL to prevent SSRF attacks
        if not _validate_slack_url(url):
            logger.warning(f"Invalid Slack response_url blocked (SSRF protection): {url[:50]}")
            return

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
                        logger.warning(
                            f"Slack response_url POST failed: {response.status} - {text[:100]}"
                        )
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Connection error posting to Slack response_url: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error posting to Slack response_url: {e}")

    @auto_error_response("handle slack interactive")
    @rate_limit(rpm=60, limiter_name="slack_interactive")
    def _handle_interactive(self, handler: Any) -> HandlerResult:
        """Handle interactive component callbacks.

        This handles button clicks, menu selections, etc. from Slack messages.
        """
        try:
            # Use pre-read body from handle()
            body = getattr(handler, "_slack_body", "")
            # Workspace context available for future use
            _workspace = getattr(handler, "_slack_workspace", None)  # noqa: F841
            _team_id = getattr(handler, "_slack_team_id", None)  # noqa: F841

            # Interactive payloads come as form-encoded with a 'payload' field
            params = parse_qs(body)
            payload_str = params.get("payload", ["{}"])[0]
            try:
                payload = json.loads(payload_str)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in Slack interactive payload: {e}")
                return error_response(f"Invalid JSON payload: {e}", 400)

            action_type = payload.get("type")
            user = payload.get("user", {})
            user_id = user.get("id", "unknown")
            team = payload.get("team", {})
            team_id = team.get("id", _team_id or "")

            logger.info(f"Interactive action from {user_id}: {action_type}")

            # Audit log the interactive action
            audit = _get_audit_logger()
            if audit:
                audit.log_event(
                    workspace_id=team_id,
                    event_type=f"interactive:{action_type}",
                    payload_summary={"action_type": action_type},
                    user_id=user_id,
                    success=True,
                )

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

        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Invalid interactive payload data: {e}")
            return json_response({"text": f"Error: {str(e)[:100]}"})
        except Exception as e:
            logger.exception(f"Unexpected interactive handler error: {e}")
            return json_response({"text": f"Error: {str(e)[:100]}"})

    def _handle_vote_action(self, payload: Dict[str, Any], action: Dict[str, Any]) -> HandlerResult:
        """Handle vote button clicks."""
        action_id = action.get("action_id", "")
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
            return json_response(
                {
                    "text": f"{emoji} Your vote for '{vote_option}' has been recorded!",
                    "replace_original": False,
                }
            )

        return json_response({"text": "Vote recorded"})

    def _handle_view_details(
        self, payload: Dict[str, Any], action: Dict[str, Any]
    ) -> HandlerResult:
        """Handle view details button clicks."""
        debate_id = action.get("value", "")

        if not debate_id:
            return json_response(
                {
                    "text": "Error: No debate ID provided",
                    "replace_original": False,
                }
            )

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
            return json_response(
                {
                    "text": f"Debate `{debate_id}` not found",
                    "replace_original": False,
                }
            )

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

        return json_response(
            {
                "response_type": "ephemeral",
                "text": f"Details for debate {debate_id}",
                "blocks": blocks,
                "replace_original": False,
            }
        )

    @auto_error_response("handle slack events")
    @rate_limit(rpm=100, limiter_name="slack_events")
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

    def _handle_app_mention(self, event: Dict[str, Any]) -> HandlerResult:
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
            response_text = 'Hi! You can ask me to:\n• Debate a topic: `@aragora debate "Should AI be regulated?"`\n• Show status: `@aragora status`\n• List agents: `@aragora agents`'
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

    def _handle_message_event(self, event: Dict[str, Any]) -> HandlerResult:
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
            response_text = 'Hi! Send me a command:\n• `help` - Show available commands\n• `status` - Check system status\n• `agents` - List available agents\n• `debate "topic"` - Start a debate'
        elif text.lower() == "help":
            response_text = (
                "*Aragora Direct Message Commands*\n\n"
                "• `help` - Show this message\n"
                "• `status` - Check system status\n"
                "• `agents` - List available agents\n"
                '• `debate "Your topic here"` - Start a debate on a topic\n'
                "• `recent` - Show recent debates\n\n"
                "_You can also use `/aragora` commands in any channel._"
            )
        elif text.lower() == "status":
            try:
                from aragora.ranking.elo import EloSystem

                store = EloSystem()
                agents = store.get_all_ratings()
                response_text = (
                    f"*Aragora Status*\n• Status: Online\n• Agents: {len(agents)} registered"
                )
            except (ImportError, AttributeError, RuntimeError) as e:
                logger.debug(f"Failed to fetch status: {e}")
                response_text = "*Aragora Status*\n• Status: Online\n• Agents: Unknown"
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

        except ImportError as e:
            logger.warning(f"Debate modules not available: {e}")
            await self._post_message_async(channel, "Debate service temporarily unavailable")
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Invalid debate request data: {e}")
            await self._post_message_async(channel, f"Invalid request: {str(e)[:100]}")
        except Exception as e:
            logger.exception(f"Unexpected DM debate creation error: {e}")
            await self._post_message_async(channel, f"Debate failed: {str(e)[:100]}")

    def _slack_response(
        self,
        text: str,
        response_type: str = "ephemeral",
    ) -> HandlerResult:
        """Create a simple Slack response."""
        return json_response(
            {
                "response_type": response_type,
                "text": text,
            }
        )

    def _slack_blocks_response(
        self,
        blocks: List[Dict[str, Any]],
        text: str,
        response_type: str = "ephemeral",
    ) -> HandlerResult:
        """Create a Slack response with blocks."""
        return json_response(
            {
                "response_type": response_type,
                "text": text,
                "blocks": blocks,
            }
        )


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
        _slack_handler = SlackHandler(server_context)  # type: ignore[arg-type]
    return _slack_handler


__all__ = ["SlackHandler", "get_slack_handler", "get_slack_integration"]
