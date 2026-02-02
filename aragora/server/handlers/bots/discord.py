"""
Discord Interactions endpoint handler.

Handles Discord's HTTP-based Interactions API for slash commands
when not using the gateway (WebSocket) connection.

Endpoints:
- POST /api/bots/discord/interactions - Handle Discord interactions

Environment Variables:
- DISCORD_APPLICATION_ID - Required for interaction verification
- DISCORD_PUBLIC_KEY - Required for Ed25519 signature verification

Security (Phase 3.1):
- Ed25519 signature verification on all incoming interactions
- Replay attack protection via timestamp freshness checking (5-minute window)
- Fails closed in production: rejects requests when public key or PyNaCl missing
- Uses centralized webhook_security module for environment-aware behavior
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

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
DISCORD_APPLICATION_ID = os.environ.get("DISCORD_APPLICATION_ID")
DISCORD_PUBLIC_KEY = os.environ.get("DISCORD_PUBLIC_KEY")

# Maximum age of request timestamp before it is considered a replay (seconds)
_MAX_TIMESTAMP_AGE = 300  # 5 minutes, matching Discord's recommendation

# PyNaCl availability flag - checked once at import time for logging
_NACL_AVAILABLE = False
try:
    from nacl.signing import VerifyKey  # noqa: F401
    from nacl.exceptions import BadSignatureError  # noqa: F401

    _NACL_AVAILABLE = True
except ImportError:
    pass

# Log warnings at module load time for missing dependencies/secrets
if not DISCORD_PUBLIC_KEY:
    logger.warning("DISCORD_PUBLIC_KEY not configured - signature verification disabled")
if not _NACL_AVAILABLE:
    logger.warning(
        "PyNaCl not installed - Discord Ed25519 signature verification unavailable. "
        "Install with: pip install pynacl"
    )


def _should_allow_unverified() -> bool:
    """Check if unverified Discord webhooks should be allowed.

    Uses the centralized webhook_security module for environment-aware behavior.
    In production: always returns False (fail closed).
    In development: returns True only with explicit ARAGORA_ALLOW_UNVERIFIED_WEBHOOKS.
    """
    try:
        from aragora.connectors.chat.webhook_security import should_allow_unverified

        return should_allow_unverified("discord")
    except ImportError:
        # If webhook_security module is not available, fail closed
        logger.warning("webhook_security module not available, failing closed")
        return False


def _verify_discord_signature(
    signature: str,
    timestamp: str,
    body: bytes,
) -> bool:
    """Verify Discord request signature using Ed25519.

    Security properties:
    - Validates the request was signed by Discord using the application's public key
    - Rejects requests with timestamps older than 5 minutes (replay protection)
    - Fails closed in production when public key or PyNaCl is missing
    - Permits unverified requests only in dev mode with explicit opt-in

    See: https://discord.com/developers/docs/interactions/receiving-and-responding

    Args:
        signature: Value of X-Signature-Ed25519 header (hex-encoded).
        timestamp: Value of X-Signature-Timestamp header.
        body: Raw request body bytes.

    Returns:
        True if the signature is valid, False otherwise.
    """
    # --- Check: Public key configured ---
    if not DISCORD_PUBLIC_KEY:
        if _should_allow_unverified():
            logger.warning(
                "DISCORD_PUBLIC_KEY not configured, allowing unverified request (dev mode)"
            )
            return True
        logger.warning("DISCORD_PUBLIC_KEY not configured, rejecting request")
        return False

    # --- Check: Required headers present ---
    if not signature or not timestamp:
        logger.warning(
            "Missing required Discord signature headers: "
            f"signature={'present' if signature else 'missing'}, "
            f"timestamp={'present' if timestamp else 'missing'}"
        )
        return False

    # --- Check: Replay protection via timestamp freshness ---
    try:
        request_time = int(timestamp)
        current_time = int(time.time())
        if abs(current_time - request_time) > _MAX_TIMESTAMP_AGE:
            logger.warning(
                f"Discord request timestamp too old: "
                f"request_time={request_time}, current_time={current_time}, "
                f"delta={abs(current_time - request_time)}s > {_MAX_TIMESTAMP_AGE}s"
            )
            return False
    except (ValueError, OverflowError):
        logger.warning(f"Invalid Discord timestamp format: {timestamp!r}")
        return False

    # --- Check: PyNaCl available ---
    if not _NACL_AVAILABLE:
        if _should_allow_unverified():
            logger.warning(
                "PyNaCl not installed, allowing unverified request (dev mode). "
                "Install with: pip install pynacl"
            )
            return True
        logger.warning("PyNaCl not installed, rejecting request")
        return False

    # --- Verify Ed25519 signature ---
    try:
        from nacl.signing import VerifyKey
        from nacl.exceptions import BadSignatureError
    except ImportError:
        # Should not happen since _NACL_AVAILABLE was True, but handle gracefully
        logger.error("PyNaCl import failed despite _NACL_AVAILABLE=True")
        return False

    try:
        verify_key = VerifyKey(bytes.fromhex(DISCORD_PUBLIC_KEY))
        message = timestamp.encode("utf-8") + body
        verify_key.verify(message, bytes.fromhex(signature))
        return True
    except BadSignatureError:
        logger.warning("Discord Ed25519 signature verification failed: bad signature")
        return False
    except (ValueError, TypeError) as e:
        # ValueError: invalid hex in signature or public key
        # TypeError: unexpected argument types
        logger.warning(f"Discord signature verification error (invalid format): {e}")
        return False
    except (RuntimeError, OSError, AttributeError) as e:
        logger.exception(f"Unexpected Discord signature verification error: {e}")
        return False


class DiscordHandler(BotHandlerMixin, SecureHandler):
    """Handler for Discord Interactions API endpoints.

    Uses BotHandlerMixin for shared auth/status patterns.

    RBAC Protected:
    - bots.read - required for status endpoint

    Note: Webhook endpoints are authenticated via Discord's Ed25519 signature,
    not RBAC, since they are called by Discord servers directly.
    """

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    # BotHandlerMixin configuration
    bot_platform = "discord"

    ROUTES = [
        "/api/v1/bots/discord/interactions",
        "/api/v1/bots/discord/status",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def _is_bot_enabled(self) -> bool:
        """Check if Discord bot is configured."""
        return bool(DISCORD_APPLICATION_ID)

    def _get_platform_config_status(self) -> dict[str, Any]:
        """Return Discord-specific config fields for status response."""
        return {
            "application_id_configured": bool(DISCORD_APPLICATION_ID),
            "public_key_configured": bool(DISCORD_PUBLIC_KEY),
        }

    @rate_limit(requests_per_minute=30)
    async def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Route Discord requests with RBAC for status endpoint."""
        if path == "/api/v1/bots/discord/status":
            # Use BotHandlerMixin's RBAC-protected status handler
            return await self.handle_status_request(handler)

        return None

    @rate_limit(requests_per_minute=30)
    async def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle POST requests."""
        if path == "/api/v1/bots/discord/interactions":
            return await self._handle_interactions(handler)

        return None

    async def _handle_interactions(self, handler: Any) -> HandlerResult:
        """Handle Discord interaction webhooks.

        This endpoint receives interactions from Discord when using the
        HTTP-based Interactions API instead of the gateway.
        """
        try:
            # Get signature headers
            signature = handler.headers.get("X-Signature-Ed25519", "")
            timestamp = handler.headers.get("X-Signature-Timestamp", "")

            # Read body
            body = self._read_request_body(handler)

            # Verify signature
            if not _verify_discord_signature(signature, timestamp, body):
                logger.warning("Discord signature verification failed")
                self._audit_webhook_auth_failure("signature")
                return error_response("Invalid signature", 401)

            # Parse interaction
            interaction, err = self._parse_json_body(body, "Discord interaction")
            if err:
                return err

            interaction_type = interaction.get("type")

            # Handle PING (type 1) - required for URL verification
            if interaction_type == 1:
                logger.info("Discord PING received, responding with PONG")
                return json_response({"type": 1})

            # Handle APPLICATION_COMMAND (type 2)
            if interaction_type == 2:
                return await self._handle_application_command(interaction)

            # Handle MESSAGE_COMPONENT (type 3) - buttons, selects, etc.
            if interaction_type == 3:
                return self._handle_message_component(interaction)

            # Handle MODAL_SUBMIT (type 5)
            if interaction_type == 5:
                return self._handle_modal_submit(interaction)

            # Unknown interaction type
            logger.warning(f"Unknown Discord interaction type: {interaction_type}")
            return json_response(
                {
                    "type": 4,  # CHANNEL_MESSAGE_WITH_SOURCE
                    "data": {
                        "content": "Unknown interaction type",
                        "flags": 64,  # Ephemeral
                    },
                }
            )

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in Discord interaction: {e}")
            return error_response("Invalid JSON payload", 400)
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Data error in Discord interaction: {e}")
            return json_response(
                {
                    "type": 4,
                    "data": {
                        "content": f"Invalid request data: {str(e)[:100]}",
                        "flags": 64,
                    },
                }
            )
        except (RuntimeError, OSError, AttributeError) as e:
            logger.exception(f"Unexpected Discord interaction error: {e}")
            return json_response(
                {
                    "type": 4,
                    "data": {
                        "content": "An unexpected error occurred",
                        "flags": 64,
                    },
                }
            )

    async def _handle_application_command(self, interaction: dict[str, Any]) -> HandlerResult:
        """Handle slash command interactions."""
        data = interaction.get("data", {})
        command_name = data.get("name", "")
        options = data.get("options", [])

        user = interaction.get("user") or interaction.get("member", {}).get("user", {})
        user_id = user.get("id", "unknown")
        username = user.get("username", "unknown")

        logger.info(f"Discord command from {username}: {command_name}")

        # Parse options into args
        args = {}
        for opt in options:
            args[opt["name"]] = opt.get("value", "")

        # Route commands
        if command_name == "aragora":
            subcommand = args.get("command", "help")
            subargs = args.get("args", "")
            return await self._execute_command(subcommand, subargs, user_id, interaction)

        if command_name == "debate":
            topic = args.get("topic", "")
            return await self._execute_command("debate", topic, user_id, interaction)

        if command_name == "gauntlet":
            statement = args.get("statement", "")
            return await self._execute_command("gauntlet", statement, user_id, interaction)

        if command_name == "status":
            return await self._execute_command("status", "", user_id, interaction)

        # Unknown command
        return json_response(
            {
                "type": 4,
                "data": {
                    "content": f"Unknown command: {command_name}",
                    "flags": 64,
                },
            }
        )

    async def _execute_command(
        self,
        command: str,
        args: str,
        user_id: str,
        interaction: dict[str, Any],
    ) -> HandlerResult:
        """Execute a command and return Discord response."""
        from datetime import datetime, timezone
        from aragora.bots.base import (
            BotChannel,
            BotMessage,
            BotUser,
            CommandContext,
            Platform,
        )
        from aragora.bots.commands import get_default_registry

        registry = get_default_registry()

        # Build context
        user_data = interaction.get("user") or interaction.get("member", {}).get("user", {})
        user = BotUser(
            id=user_data.get("id", "unknown"),
            username=user_data.get("username", "unknown"),
            display_name=user_data.get("global_name"),
            platform=Platform.DISCORD,
        )

        channel = BotChannel(
            id=interaction.get("channel_id", "unknown"),
            platform=Platform.DISCORD,
        )

        message = BotMessage(
            id=interaction.get("id", "unknown"),
            text=f"/{command} {args}".strip(),
            user=user,
            channel=channel,
            timestamp=datetime.now(timezone.utc),
            platform=Platform.DISCORD,
        )

        ctx = CommandContext(
            message=message,
            user=user,
            channel=channel,
            platform=Platform.DISCORD,
            args=[command] + (args.split() if args else []),
            raw_args=args,
            metadata={
                "api_base": os.environ.get("ARAGORA_API_BASE", "http://localhost:8080"),
                "interaction_id": interaction.get("id"),
                "guild_id": interaction.get("guild_id"),
            },
        )

        # Execute command
        result = await registry.execute(ctx)

        # Build response
        if result.success:
            response_data: dict[str, Any] = {
                "content": result.message or "Command executed",
            }

            if result.discord_embed:
                response_data["embeds"] = [result.discord_embed]

            if result.ephemeral:
                response_data["flags"] = 64

            return json_response(
                {
                    "type": 4,  # CHANNEL_MESSAGE_WITH_SOURCE
                    "data": response_data,
                }
            )
        else:
            return json_response(
                {
                    "type": 4,
                    "data": {
                        "content": f"Error: {result.error}",
                        "flags": 64,
                    },
                }
            )

    def _handle_message_component(self, interaction: dict[str, Any]) -> HandlerResult:
        """Handle button/select interactions."""
        data = interaction.get("data", {})
        custom_id = data.get("custom_id", "")

        user = interaction.get("user") or interaction.get("member", {}).get("user", {})
        user_id = user.get("id", "unknown")

        logger.info(f"Discord component interaction from {user_id}: {custom_id}")

        # Parse custom_id (e.g., "vote_debateid_agree")
        if custom_id.startswith("vote_"):
            parts = custom_id.split("_")
            if len(parts) >= 3:
                debate_id = parts[1]
                vote = parts[2]

                # Record vote
                try:
                    from aragora.server.storage import get_debates_db

                    db = get_debates_db()
                    if db and hasattr(db, "record_vote"):
                        db.record_vote(
                            debate_id=debate_id,
                            voter_id=f"discord:{user_id}",
                            vote=vote,
                            source="discord",
                        )
                except (ValueError, KeyError, TypeError) as e:
                    logger.warning(f"Failed to record vote due to data error: {e}")
                except (RuntimeError, OSError, AttributeError) as e:
                    logger.exception(f"Unexpected error recording vote: {e}")

                emoji = "thumbsup" if vote == "agree" else "thumbsdown"
                return json_response(
                    {
                        "type": 4,
                        "data": {
                            "content": f":{emoji}: Your vote has been recorded!",
                            "flags": 64,
                        },
                    }
                )

        # Unknown component
        return json_response(
            {
                "type": 4,
                "data": {
                    "content": "Interaction received",
                    "flags": 64,
                },
            }
        )

    def _handle_modal_submit(self, interaction: dict[str, Any]) -> HandlerResult:
        """Handle modal submission interactions."""
        data = interaction.get("data", {})
        custom_id = data.get("custom_id", "")

        logger.info(f"Discord modal submit: {custom_id}")

        return json_response(
            {
                "type": 4,
                "data": {
                    "content": "Form submitted",
                    "flags": 64,
                },
            }
        )


__all__ = ["DiscordHandler"]
