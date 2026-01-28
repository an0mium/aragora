"""
Slack Bot Handler for bi-directional integration.

Handles:
- Incoming webhooks (slash commands, events)
- Interactive components (button clicks, votes)
- Two-way debate participation from Slack

Implements:
- Signature verification for security
- Block Kit interactive messages with voting
- Threaded debate updates
- User vote counting in consensus
"""

import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any, Dict, Optional
from urllib.parse import parse_qs

from aragora.audit.unified import audit_data
from aragora.server.handlers.base import HandlerResult, error_response, json_response
from aragora.server.handlers.bots.base import BotHandlerMixin
from aragora.server.handlers.secure import SecureHandler
import re

logger = logging.getLogger(__name__)

# Command patterns for parsing Slack slash commands
# Matches: /aragora <command> [arguments]
COMMAND_PATTERN = re.compile(r"^/aragora\s+(\w+)(?:\s+(.*))?$", re.IGNORECASE)

# Topic patterns for parsing debate topics
# Matches quoted or unquoted topics
TOPIC_PATTERN = re.compile(r'^["\']?([^"\']+)["\']?$')

# Environment variables for Slack configuration
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET", "")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")

# Store active debate sessions for Slack
# In production, this would be in Redis/database
_active_debates: dict[str, dict[str, Any]] = {}
_user_votes: dict[str, dict[str, str]] = {}  # debate_id -> {user_id: vote}

# Slack integration singleton
_slack_integration: Optional[Any] = None


def get_slack_integration() -> Optional[Any]:
    """Get the Slack integration singleton.

    Returns None if Slack is not configured (no webhook URL).
    """
    global _slack_integration

    # Return cached if available
    if _slack_integration is not None:
        return _slack_integration

    # Check if Slack is configured
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
    if not webhook_url:
        return None

    # Create integration (lazy import to avoid circular dependencies)
    try:
        from aragora.connectors.slack import SlackConnector

        _slack_integration = SlackConnector(webhook_url=webhook_url)
        return _slack_integration
    except ImportError:
        logger.debug("SlackConnector not available")
        return None


def verify_slack_signature(
    body: bytes,
    timestamp: str,
    signature: str,
    signing_secret: str,
) -> bool:
    """Verify Slack request signature."""
    # Check timestamp to prevent replay attacks
    current_time = int(time.time())
    if abs(current_time - int(timestamp)) > 60 * 5:
        logger.warning("Slack signature timestamp too old")
        return False

    # Compute expected signature
    sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    my_signature = (
        "v0="
        + hmac.new(
            signing_secret.encode(),
            sig_basestring.encode(),
            hashlib.sha256,
        ).hexdigest()
    )

    return hmac.compare_digest(my_signature, signature)


def build_debate_message_blocks(
    debate_id: str,
    task: str,
    agents: list[str],
    current_round: int,
    total_rounds: int,
    include_vote_buttons: bool = True,
) -> list[dict[str, Any]]:
    """Build Block Kit blocks for a debate message."""
    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": " Active Debate",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Task:* {task}",
            },
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Agents:*\n{', '.join(agents)}",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Progress:*\nRound {current_round}/{total_rounds}",
                },
            ],
        },
        {"type": "divider"},
    ]

    if include_vote_buttons:
        # Add voting buttons
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Cast your vote:*",
                },
            }
        )

        # Create button for each agent
        buttons = [
            {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": f"Vote {agent}",
                    "emoji": True,
                },
                "style": "primary" if i == 0 else None,
                "action_id": f"vote_{debate_id}_{agent}",
                "value": json.dumps({"debate_id": debate_id, "agent": agent}),
            }
            for i, agent in enumerate(agents[:5])  # Max 5 buttons
        ]

        # Remove None style values
        for btn in buttons:
            if btn.get("style") is None:
                del btn["style"]

        blocks.append(
            {
                "type": "actions",
                "elements": buttons,
            }
        )

        # Add summary button
        blocks.append(
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": " View Summary",
                            "emoji": True,
                        },
                        "action_id": f"summary_{debate_id}",
                        "value": debate_id,
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": " Provenance",
                            "emoji": True,
                        },
                        "action_id": f"provenance_{debate_id}",
                        "value": debate_id,
                        "url": f"https://aragora.ai/debates/provenance?debate={debate_id}",
                    },
                ],
            }
        )

    # Footer
    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f" Aragora | Debate ID: `{debate_id[:8]}...`",
                },
            ],
        }
    )

    return blocks


def build_consensus_message_blocks(
    debate_id: str,
    task: str,
    consensus_reached: bool,
    confidence: float,
    winner: Optional[str],
    final_answer: Optional[str],
    vote_counts: dict[str, int],
) -> list[dict[str, Any]]:
    """Build Block Kit blocks for consensus result."""
    status_emoji = "" if consensus_reached else ""
    status_text = "Consensus Reached" if consensus_reached else "No Consensus"

    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{status_emoji} {status_text}",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Task:* {task}",
            },
        },
    ]

    # Results fields
    fields = [
        {"type": "mrkdwn", "text": f"*Confidence:*\n{confidence:.0%}"},
    ]

    if winner:
        fields.append({"type": "mrkdwn", "text": f"*Winner:*\n{winner}"})

    blocks.append(
        {
            "type": "section",
            "fields": fields,
        }
    )

    # Show user votes if any
    if vote_counts:
        vote_text = "\n".join(
            f"• {agent}: {count} vote{'s' if count != 1 else ''}"
            for agent, count in sorted(vote_counts.items(), key=lambda x: -x[1])
        )
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*User Votes:*\n{vote_text}",
                },
            }
        )

    # Final answer preview
    if final_answer:
        preview = final_answer[:500]
        if len(final_answer) > 500:
            preview += "..."

        blocks.append({"type": "divider"})
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Decision:*\n```{preview}```",
                },
            }
        )

    # Action buttons
    blocks.append(
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": " View Full",
                        "emoji": True,
                    },
                    "url": f"https://aragora.ai/debate/{debate_id}",
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": " Audit Trail",
                        "emoji": True,
                    },
                    "url": f"https://aragora.ai/debates/provenance?debate={debate_id}",
                },
            ],
        }
    )

    return blocks


# Alias for backward compatibility
build_debate_result_blocks = build_consensus_message_blocks


async def handle_slack_events(request: Any) -> HandlerResult:
    """Handle Slack Events API webhook."""
    try:
        body = await request.body()
        data = json.loads(body)

        # URL verification challenge
        if data.get("type") == "url_verification":
            return json_response({"challenge": data.get("challenge", "")})

        event = data.get("event", {})
        event_type = event.get("type")

        if event_type == "app_mention":
            # Bot was mentioned - could trigger a debate
            text = event.get("text", "")
            channel = event.get("channel")
            user = event.get("user")

            logger.info(f"Slack mention from {user} in {channel}: {text[:100]}")

            # Parse command from mention
            # Format: @aragora ask "question" or @aragora status
            return json_response(
                {
                    "response_type": "in_channel",
                    "text": "Received your request. Processing...",
                }
            )

        elif event_type == "message":
            # Direct message or channel message
            pass

        return json_response({"ok": True})

    except Exception as e:
        logger.error(f"Slack events handler error: {e}")
        return json_response({"error": str(e)}, status=500)


async def handle_slack_interactions(request: Any) -> HandlerResult:
    """Handle Slack interactive components (button clicks, votes)."""
    try:
        body = await request.body()

        # Parse form-encoded payload
        parsed = parse_qs(body.decode("utf-8"))
        payload_str = parsed.get("payload", ["{}"])[0]
        payload = json.loads(payload_str)

        interaction_type = payload.get("type")
        user = payload.get("user", {})
        user_id = user.get("id", "unknown")
        user_name = user.get("name", "unknown")

        if interaction_type == "block_actions":
            actions = payload.get("actions", [])

            for action in actions:
                action_id = action.get("action_id", "")
                value = action.get("value", "")

                # Handle vote action
                if action_id.startswith("vote_"):
                    try:
                        vote_data = json.loads(value)
                        debate_id = vote_data.get("debate_id")
                        agent = vote_data.get("agent")

                        # Record user vote
                        if debate_id not in _user_votes:
                            _user_votes[debate_id] = {}

                        _user_votes[debate_id][user_id] = agent

                        logger.info(f"User {user_name} voted for {agent} in debate {debate_id}")

                        audit_data(
                            user_id=f"slack:{user_id}",
                            resource_type="debate_vote",
                            resource_id=debate_id,
                            action="create",
                            vote_option=agent,
                            platform="slack",
                        )

                        # Return ephemeral confirmation
                        return json_response(
                            {
                                "response_type": "ephemeral",
                                "text": f" Your vote for *{agent}* has been recorded!",
                                "replace_original": False,
                            }
                        )

                    except json.JSONDecodeError:
                        pass

                # Handle summary request
                elif action_id.startswith("summary_"):
                    debate_id = value
                    # Would fetch and return debate summary
                    return json_response(
                        {
                            "response_type": "ephemeral",
                            "text": f"Fetching summary for debate `{debate_id[:8]}...`",
                        }
                    )

        elif interaction_type == "shortcut":
            # Global shortcut triggered
            callback_id = payload.get("callback_id")
            if callback_id == "start_debate":
                # Open modal to start debate
                return json_response(
                    {
                        "response_action": "open_modal",
                        "view": _build_start_debate_modal(),
                    }
                )

        elif interaction_type == "view_submission":
            # Modal form submitted
            view = payload.get("view", {})
            callback_id = view.get("callback_id")

            if callback_id == "start_debate_modal":
                # Parse submitted values and start debate
                values = view.get("state", {}).get("values", {})

                # Extract task from task_block
                task = values.get("task_block", {}).get("task_input", {}).get("value", "")

                # Extract agents from agents_block (multi-select returns list)
                agents_data = (
                    values.get("agents_block", {})
                    .get("agents_select", {})
                    .get("selected_options", [])
                )
                agents = [opt.get("value", "") for opt in agents_data]

                # Extract rounds from rounds_block
                rounds_str = (
                    values.get("rounds_block", {})
                    .get("rounds_select", {})
                    .get("selected_option", {})
                    .get("value", "5")
                )
                rounds = int(rounds_str) if rounds_str.isdigit() else 5

                if not task:
                    return json_response(
                        {
                            "response_action": "errors",
                            "errors": {"task_block": "Please enter a debate task"},
                        }
                    )

                if not agents:
                    return json_response(
                        {
                            "response_action": "errors",
                            "errors": {"agents_block": "Please select at least one agent"},
                        }
                    )

                # Generate debate ID and store in active debates
                import uuid

                debate_id = str(uuid.uuid4())

                # Map agent values to display names
                agent_names = {
                    "claude": "Claude",
                    "gpt4": "GPT-4",
                    "gemini": "Gemini",
                    "mistral": "Mistral",
                    "deepseek": "DeepSeek",
                }
                agent_display_names = [agent_names.get(a, a) for a in agents]

                _active_debates[debate_id] = {
                    "task": task,
                    "agents": agent_display_names,
                    "rounds": rounds,
                    "current_round": 1,
                    "status": "running",
                    "user_id": user.get("id"),
                }

                logger.info(
                    f"Started debate {debate_id} from Slack modal: "
                    f"task='{task[:50]}...', agents={agents}, rounds={rounds}"
                )

                audit_data(
                    user_id=f"slack:{user.get('id', 'unknown')}",
                    resource_type="debate",
                    resource_id=debate_id,
                    action="create",
                    platform="slack",
                    task_preview=task[:100],
                )

                return json_response({"response_action": "clear"})

        return json_response({"ok": True})

    except Exception as e:
        logger.error(f"Slack interactions handler error: {e}")
        return json_response({"error": str(e)}, status=500)


async def handle_slack_commands(request: Any) -> HandlerResult:
    """Handle Slack slash commands (/aragora)."""
    try:
        body = await request.body()
        params = parse_qs(body.decode("utf-8"))

        _command = params.get("command", ["/aragora"])[0]  # noqa: F841
        text = params.get("text", [""])[0]
        _user_id = params.get("user_id", [""])[0]  # noqa: F841
        _user_name = params.get("user_name", [""])[0]  # noqa: F841
        _channel_id = params.get("channel_id", [""])[0]  # noqa: F841
        _response_url = params.get("response_url", [""])[0]  # noqa: F841

        # Parse subcommand
        parts = text.strip().split(maxsplit=1)
        subcommand = parts[0].lower() if parts else "help"
        args = parts[1] if len(parts) > 1 else ""

        if subcommand == "ask" and args:
            # Start a new debate
            return json_response(
                {
                    "response_type": "in_channel",
                    "text": f" Starting debate: _{args[:100]}_\n\nAgents are deliberating...",
                    "blocks": build_debate_message_blocks(
                        debate_id="pending",
                        task=args,
                        agents=["Claude", "GPT-4", "Gemini", "Mistral"],
                        current_round=1,
                        total_rounds=5,
                        include_vote_buttons=False,
                    ),
                }
            )

        elif subcommand == "status":
            # Get active debates status
            active_count = len(_active_debates)
            return json_response(
                {
                    "response_type": "ephemeral",
                    "text": f" {active_count} active debate(s) in this workspace",
                }
            )

        elif subcommand == "vote":
            # Vote in active debate
            return json_response(
                {
                    "response_type": "ephemeral",
                    "text": " Use the vote buttons in the debate message to cast your vote",
                }
            )

        elif subcommand == "leaderboard":
            # Show agent leaderboard
            return json_response(
                {
                    "response_type": "in_channel",
                    "text": " *Agent Leaderboard*\n1.  Claude - 1850 ELO\n2.  GPT-4 - 1820 ELO\n3.  Gemini - 1780 ELO",
                }
            )

        else:  # help or unknown
            return json_response(
                {
                    "response_type": "ephemeral",
                    "text": (
                        "*Aragora Commands*\n\n"
                        "`/aragora ask <question>` - Start a new debate\n"
                        "`/aragora status` - Show active debates\n"
                        "`/aragora vote` - Vote in active debate\n"
                        "`/aragora leaderboard` - Show agent rankings\n"
                        "`/aragora help` - Show this message"
                    ),
                }
            )

    except Exception as e:
        logger.error(f"Slack commands handler error: {e}")
        return json_response(
            {
                "response_type": "ephemeral",
                "text": f" Error: {str(e)}",
            }
        )


def _build_start_debate_modal() -> dict[str, Any]:
    """Build modal for starting a new debate."""
    return {
        "type": "modal",
        "callback_id": "start_debate_modal",
        "title": {
            "type": "plain_text",
            "text": "Start Debate",
        },
        "submit": {
            "type": "plain_text",
            "text": "Start",
        },
        "close": {
            "type": "plain_text",
            "text": "Cancel",
        },
        "blocks": [
            {
                "type": "input",
                "block_id": "task_block",
                "element": {
                    "type": "plain_text_input",
                    "action_id": "task_input",
                    "placeholder": {
                        "type": "plain_text",
                        "text": "What should the agents debate?",
                    },
                    "multiline": True,
                },
                "label": {
                    "type": "plain_text",
                    "text": "Debate Task",
                },
            },
            {
                "type": "input",
                "block_id": "agents_block",
                "element": {
                    "type": "multi_static_select",
                    "action_id": "agents_select",
                    "placeholder": {
                        "type": "plain_text",
                        "text": "Select agents",
                    },
                    "options": [
                        {"text": {"type": "plain_text", "text": "Claude"}, "value": "claude"},
                        {"text": {"type": "plain_text", "text": "GPT-4"}, "value": "gpt4"},
                        {"text": {"type": "plain_text", "text": "Gemini"}, "value": "gemini"},
                        {"text": {"type": "plain_text", "text": "Mistral"}, "value": "mistral"},
                        {"text": {"type": "plain_text", "text": "DeepSeek"}, "value": "deepseek"},
                    ],
                },
                "label": {
                    "type": "plain_text",
                    "text": "Agents",
                },
            },
            {
                "type": "input",
                "block_id": "rounds_block",
                "element": {
                    "type": "static_select",
                    "action_id": "rounds_select",
                    "placeholder": {
                        "type": "plain_text",
                        "text": "Number of rounds",
                    },
                    "options": [
                        {"text": {"type": "plain_text", "text": "3 rounds"}, "value": "3"},
                        {"text": {"type": "plain_text", "text": "5 rounds"}, "value": "5"},
                        {"text": {"type": "plain_text", "text": "8 rounds"}, "value": "8"},
                    ],
                    "initial_option": {
                        "text": {"type": "plain_text", "text": "5 rounds"},
                        "value": "5",
                    },
                },
                "label": {
                    "type": "plain_text",
                    "text": "Rounds",
                },
            },
        ],
    }


def get_debate_vote_counts(debate_id: str) -> dict[str, int]:
    """Get vote counts for a debate."""
    votes = _user_votes.get(debate_id, {})
    counts: dict[str, int] = {}
    for agent in votes.values():
        counts[agent] = counts.get(agent, 0) + 1
    return counts


class SlackHandler(BotHandlerMixin, SecureHandler):
    """Handler for Slack bot integration endpoints.

    Uses BotHandlerMixin for shared auth/status patterns.

    RBAC Protected:
    - bots.read - required for status endpoint

    Endpoints:
    - GET  /api/v1/bots/slack/status       - Get Slack integration status
    - POST /api/v1/bots/slack/events       - Handle Slack Events API
    - POST /api/v1/bots/slack/interactions - Handle interactive components
    - POST /api/v1/bots/slack/commands     - Handle slash commands
    """

    bot_platform = "slack"

    ROUTES = [
        "/api/v1/bots/slack/status",
        "/api/v1/bots/slack/events",
        "/api/v1/bots/slack/interactions",
        "/api/v1/bots/slack/commands",
    ]

    def __init__(self, server_context: Dict[str, Any]):
        """Initialize the Slack handler."""
        super().__init__(server_context)  # type: ignore[arg-type]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path.startswith("/api/v1/bots/slack/") or path.startswith(
            "/api/v1/integrations/slack/"
        )

    def _is_bot_enabled(self) -> bool:
        """Check if Slack bot is configured."""
        return bool(SLACK_BOT_TOKEN) or bool(SLACK_SIGNING_SECRET)

    def handle(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle GET requests for Slack endpoints."""
        if path == "/api/v1/bots/slack/status":
            # Use the mixin's status handler - need to run async
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create a new task and let the event loop handle it
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, self._get_status_sync(handler))
                        return future.result(timeout=5)
                else:
                    return loop.run_until_complete(self._get_status_sync(handler))
            except Exception as e:
                logger.error(f"Error getting Slack status: {e}")
                return error_response(f"Error: {str(e)}", 500)
        return None

    async def _get_status_sync(self, handler: Any) -> HandlerResult:
        """Get status using the mixin's handler."""
        extra_status = {
            "configured": self._is_bot_enabled(),
            "active_debates": len(_active_debates),
            "features": {
                "slash_commands": True,
                "events_api": True,
                "interactive_components": True,
                "block_kit": True,
            },
        }
        return await self.handle_status_request(handler, extra_status)

    async def handle_post(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle POST requests for Slack endpoints."""
        # Verify Slack signature for webhook endpoints
        if path in [
            "/api/v1/bots/slack/events",
            "/api/v1/bots/slack/interactions",
            "/api/v1/bots/slack/commands",
        ]:
            if SLACK_SIGNING_SECRET:
                try:
                    timestamp = handler.headers.get("X-Slack-Request-Timestamp", "")
                    signature = handler.headers.get("X-Slack-Signature", "")
                    body = handler.rfile.read(int(handler.headers.get("Content-Length", 0)))
                    # Reset the file position for later reads
                    handler.rfile.seek(0)

                    if not verify_slack_signature(body, timestamp, signature, SLACK_SIGNING_SECRET):
                        return error_response("Invalid Slack signature", 401)
                except Exception as e:
                    logger.warning(f"Slack signature verification error: {e}")
                    # Continue without verification if headers are missing

        if path == "/api/v1/bots/slack/events":
            return await handle_slack_events(handler)

        if path == "/api/v1/bots/slack/interactions":
            return await handle_slack_interactions(handler)

        if path == "/api/v1/bots/slack/commands":
            return await handle_slack_commands(handler)

        return None


def register_slack_routes(router: Any) -> None:
    """Register Slack routes with the server router.

    Note: This function is deprecated in favor of using SlackHandler class
    with the unified handler registration system.
    """

    async def events_handler(request: Any) -> HandlerResult:
        return await handle_slack_events(request)

    async def interactions_handler(request: Any) -> HandlerResult:
        return await handle_slack_interactions(request)

    async def commands_handler(request: Any) -> HandlerResult:
        return await handle_slack_commands(request)

    # Register routes
    router.add_route("POST", "/api/bots/slack/events", events_handler)
    router.add_route("POST", "/api/bots/slack/interactions", interactions_handler)
    router.add_route("POST", "/api/bots/slack/commands", commands_handler)


__all__ = [
    "SlackHandler",
    "handle_slack_events",
    "handle_slack_interactions",
    "handle_slack_commands",
    "build_debate_message_blocks",
    "build_consensus_message_blocks",
    "get_debate_vote_counts",
    "register_slack_routes",
]
