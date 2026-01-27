"""
Discord Interactions endpoint handler.

Handles Discord's HTTP-based Interactions API for slash commands
when not using the gateway (WebSocket) connection.

Endpoints:
- POST /api/bots/discord/interactions - Handle Discord interactions

Environment Variables:
- DISCORD_APPLICATION_ID - Required for interaction verification
- DISCORD_PUBLIC_KEY - Required for signature verification
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from aragora.audit.unified import audit_security
from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    json_response,
)
from aragora.server.handlers.secure import SecureHandler, ForbiddenError, UnauthorizedError
from aragora.server.handlers.utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# RBAC permission for bot configuration endpoints
BOTS_READ_PERMISSION = "bots.read"

# Environment variables
DISCORD_APPLICATION_ID = os.environ.get("DISCORD_APPLICATION_ID", "")
DISCORD_PUBLIC_KEY = os.environ.get("DISCORD_PUBLIC_KEY", "")


def _verify_discord_signature(
    signature: str,
    timestamp: str,
    body: bytes,
) -> bool:
    """Verify Discord request signature using Ed25519.

    See: https://discord.com/developers/docs/interactions/receiving-and-responding#security-and-authorization
    """
    if not DISCORD_PUBLIC_KEY:
        logger.warning("DISCORD_PUBLIC_KEY not configured, skipping signature verification")
        return True

    try:
        from nacl.signing import VerifyKey
        from nacl.exceptions import BadSignature

        verify_key = VerifyKey(bytes.fromhex(DISCORD_PUBLIC_KEY))
        message = timestamp.encode() + body
        verify_key.verify(message, bytes.fromhex(signature))
        return True
    except ImportError:
        logger.warning("PyNaCl not installed, skipping signature verification")
        return True
    except BadSignature:
        return False
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid signature format: {e}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected signature verification error: {e}")
        return False


class DiscordHandler(SecureHandler):
    """Handler for Discord Interactions API endpoints.

    RBAC Protected:
    - bots.read - required for status endpoint

    Note: Webhook endpoints are authenticated via Discord's Ed25519 signature,
    not RBAC, since they are called by Discord servers directly.
    """

    ROUTES = [
        "/api/v1/bots/discord/interactions",
        "/api/v1/bots/discord/status",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    @rate_limit(rpm=30)
    async def handle(  # type: ignore[override]
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route Discord requests with RBAC for status endpoint."""
        if path == "/api/v1/bots/discord/status":
            # RBAC: Require authentication and bots.read permission
            try:
                auth_context = await self.get_auth_context(handler, require_auth=True)
                self.check_permission(auth_context, BOTS_READ_PERMISSION)
            except UnauthorizedError:
                return error_response("Authentication required", 401)
            except ForbiddenError as e:
                logger.warning(f"Discord status access denied: {e}")
                return error_response(str(e), 403)
            return self._get_status()

        return None

    @rate_limit(rpm=30)
    def handle_post(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle POST requests."""
        if path == "/api/v1/bots/discord/interactions":
            return self._handle_interactions(handler)

        return None

    def _get_status(self) -> HandlerResult:
        """Get Discord bot status."""
        return json_response(
            {
                "enabled": bool(DISCORD_APPLICATION_ID),
                "application_id_configured": bool(DISCORD_APPLICATION_ID),
                "public_key_configured": bool(DISCORD_PUBLIC_KEY),
            }
        )

    def _handle_interactions(self, handler: Any) -> HandlerResult:
        """Handle Discord interaction webhooks.

        This endpoint receives interactions from Discord when using the
        HTTP-based Interactions API instead of the gateway.
        """
        try:
            # Get signature headers
            signature = handler.headers.get("X-Signature-Ed25519", "")
            timestamp = handler.headers.get("X-Signature-Timestamp", "")

            # Read body
            content_length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(content_length)

            # Verify signature
            if not _verify_discord_signature(signature, timestamp, body):
                logger.warning("Discord signature verification failed")
                audit_security(
                    event_type="discord_webhook_auth_failed",
                    actor_id="unknown",
                    resource_type="discord_webhook",
                    resource_id="signature",
                )
                return error_response("Invalid signature", 401)

            # Parse interaction
            try:
                interaction = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in Discord interaction: {e}")
                return error_response("Invalid JSON", 400)

            interaction_type = interaction.get("type")

            # Handle PING (type 1) - required for URL verification
            if interaction_type == 1:
                logger.info("Discord PING received, responding with PONG")
                return json_response({"type": 1})

            # Handle APPLICATION_COMMAND (type 2)
            if interaction_type == 2:
                return self._handle_application_command(interaction)

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
        except Exception as e:
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

    def _handle_application_command(self, interaction: Dict[str, Any]) -> HandlerResult:
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
            return self._execute_command(subcommand, subargs, user_id, interaction)

        if command_name == "debate":
            topic = args.get("topic", "")
            return self._execute_command("debate", topic, user_id, interaction)

        if command_name == "gauntlet":
            statement = args.get("statement", "")
            return self._execute_command("gauntlet", statement, user_id, interaction)

        if command_name == "status":
            return self._execute_command("status", "", user_id, interaction)

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

    def _execute_command(
        self,
        command: str,
        args: str,
        user_id: str,
        interaction: Dict[str, Any],
    ) -> HandlerResult:
        """Execute a command and return Discord response."""
        import asyncio
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

        # Execute command (run async in sync context)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(registry.execute(ctx))
        finally:
            loop.close()

        # Build response
        if result.success:
            response_data: Dict[str, Any] = {
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

    def _handle_message_component(self, interaction: Dict[str, Any]) -> HandlerResult:
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
                except Exception as e:
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

    def _handle_modal_submit(self, interaction: Dict[str, Any]) -> HandlerResult:
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
