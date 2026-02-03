"""
WhatsApp Business API integration endpoint handlers.

Endpoints:
- GET  /api/integrations/whatsapp/webhook - Webhook verification (Meta verification)
- POST /api/integrations/whatsapp/webhook - Handle incoming messages
- GET  /api/integrations/whatsapp/status  - Get integration status

Environment Variables:
- WHATSAPP_ACCESS_TOKEN - Required for sending messages (Meta Business token)
- WHATSAPP_PHONE_NUMBER_ID - Required for sending messages
- WHATSAPP_VERIFY_TOKEN - Required for webhook verification
- WHATSAPP_APP_SECRET - Optional for signature verification

Supported Messages:
- Text messages -> Commands or debate topics
- Interactive replies -> Vote buttons

Bot Commands (send as text):
- help - Show available commands
- debate <topic> - Start a multi-agent debate
- gauntlet <statement> - Run adversarial validation
- status - Get system status
- agents - List available agents
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ...base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)
from .commands import (
    command_agents,
    command_debate,
    command_gauntlet,
    command_help,
    command_receipt,
    command_recent,
    command_search,
    command_status,
)
from . import config as _config
from .config import (
    PERM_WHATSAPP_DETAILS,
    PERM_WHATSAPP_MESSAGES,
    PERM_WHATSAPP_READ,
    PERM_WHATSAPP_VOTES,
    WHATSAPP_ACCESS_TOKEN,
    WHATSAPP_PHONE_NUMBER_ID,
    WHATSAPP_VERIFY_TOKEN,
    WHATSAPP_APP_SECRET,
    AuthorizationContext,
    RBAC_AVAILABLE,
    check_permission,
    extract_user_from_request,
)
from .messaging import send_text_message
from ..telemetry import (
    record_message,
    record_command,
    record_vote,
)
from ..chat_events import (
    emit_command_received,
    emit_message_received,
    emit_vote_received,
)
from .webhooks import (
    WebhookProcessor,
    verify_signature,
    verify_webhook,
)

logger = logging.getLogger(__name__)


class WhatsAppHandler(BaseHandler):
    """Handler for WhatsApp Business API integration endpoints."""

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}
        self._webhook_processor = WebhookProcessor(self)

    ROUTES = [
        "/api/v1/integrations/whatsapp/webhook",
        "/api/v1/integrations/whatsapp/status",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    # =========================================================================
    # RBAC Helper Methods
    # =========================================================================

    def _get_auth_context(self, handler: Any) -> Any | None:
        """Extract authorization context from the request."""
        if not RBAC_AVAILABLE or extract_user_from_request is None:
            return None

        try:
            user_info = extract_user_from_request(handler)
            if not user_info:
                return None

            return AuthorizationContext(
                user_id=user_info.user_id or "anonymous",
                roles={user_info.role} if user_info.role else set(),
                org_id=user_info.org_id,
            )
        except Exception as e:
            logger.debug(f"Could not extract auth context: {e}")
            return None

    def _check_permission(self, handler: Any, permission_key: str) -> HandlerResult | None:
        """Check if current user has permission. Returns error response if denied."""
        if not RBAC_AVAILABLE or check_permission is None:
            return None

        context = self._get_auth_context(handler)
        if context is None:
            return None

        try:
            decision = check_permission(context, permission_key)
            if not decision.allowed:
                logger.warning(f"Permission denied: {permission_key} for user {context.user_id}")
                return error_response(f"Permission denied: {decision.reason}", 403)
        except Exception as e:
            logger.warning(f"RBAC check failed: {e}")
            return None

        return None

    def _get_auth_context_from_message(
        self,
        from_number: str,
        profile_name: str | None = None,
    ) -> Any | None:
        """Build an authorization context from a WhatsApp message.

        Extracts user information from the WhatsApp message sender to create
        an RBAC context for permission checking.

        Args:
            from_number: The WhatsApp phone number (user identifier).
            profile_name: Optional display name from WhatsApp profile.

        Returns:
            AuthorizationContext if RBAC is available, None otherwise.
        """
        if not RBAC_AVAILABLE or AuthorizationContext is None:
            return None

        try:
            if not from_number:
                return None

            # Normalize phone number (remove any formatting)
            normalized_number = from_number.replace(" ", "").replace("-", "").replace("+", "")

            # Build context with whatsapp prefix for namespace isolation
            return AuthorizationContext(
                user_id=f"whatsapp:{normalized_number}",
                org_id=None,  # WhatsApp doesn't have org concept
                # Default roles - in production these would come from role assignment lookup
                roles={"whatsapp_user"},
            )
        except Exception as e:
            logger.debug(f"Could not build auth context from message: {e}")
            return None

    def _check_whatsapp_permission(
        self,
        from_number: str,
        permission_key: str,
        profile_name: str | None = None,
        resource_id: str | None = None,
    ) -> str | None:
        """Check if the WhatsApp user has a specific permission.

        Args:
            from_number: The WhatsApp phone number (user identifier).
            permission_key: Permission to check (e.g., "whatsapp:debates:create").
            profile_name: Optional display name for logging.
            resource_id: Optional resource ID for resource-scoped permissions.

        Returns:
            Error message string if permission denied, None if allowed or RBAC unavailable.
        """
        if not RBAC_AVAILABLE or check_permission is None:
            # RBAC not available - allow by default (fail open for backwards compat)
            return None

        context = self._get_auth_context_from_message(from_number, profile_name)
        if context is None:
            # Cannot determine user context - log and allow (fail open)
            logger.debug("Could not build auth context for WhatsApp permission check")
            return None

        try:
            decision = check_permission(context, permission_key, resource_id)
            if not decision.allowed:
                # Optional enforcement toggle (default: fail open for WhatsApp)
                enforce_rbac = False
                config = getattr(self, "ctx", {}).get("config", {})
                if isinstance(config, dict):
                    enforce_rbac = bool(config.get("whatsapp_enforce_rbac", False))

                if enforce_rbac:
                    user_display = profile_name or from_number
                    logger.warning(
                        f"WhatsApp permission denied: {permission_key} for user "
                        f"{user_display} ({context.user_id}), reason: {decision.reason}"
                    )
                    return f"Permission denied: {decision.reason}"
        except Exception as e:
            logger.warning(f"WhatsApp RBAC check failed: {e}")
            # On error, allow by default (fail open)
            return None

        return None

    def _validate_phone_number(self, phone_number: str) -> tuple[bool, str | None]:
        """Validate a WhatsApp phone number format.

        Performs basic validation on the phone number to ensure it meets
        expected format requirements before processing.

        Args:
            phone_number: The phone number to validate.

        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is None.
        """
        if not phone_number:
            return False, "Phone number is required"

        # Remove common formatting characters for validation
        normalized = phone_number.replace(" ", "").replace("-", "").replace("+", "")

        # Check minimum length (most phone numbers are 10+ digits)
        if len(normalized) < 10:
            return False, "Phone number is too short"

        # Check maximum length (E.164 max is 15 digits)
        if len(normalized) > 15:
            return False, "Phone number is too long"

        # Check that it contains only digits
        if not normalized.isdigit():
            return False, "Phone number contains invalid characters"

        return True, None

    # =========================================================================
    # Request routing
    # =========================================================================

    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        """Route WhatsApp requests to appropriate methods."""
        logger.debug(f"WhatsApp request: {path} {handler.command}")

        if path == "/api/v1/integrations/whatsapp/status":
            # RBAC: Check permission to read WhatsApp status
            perm_error = self._check_permission(handler, PERM_WHATSAPP_READ)
            if perm_error:
                return perm_error
            return self._get_status()

        if path == "/api/v1/integrations/whatsapp/webhook":
            if handler.command == "GET":
                # Webhook verification from Meta - no RBAC (Meta callback)
                return verify_webhook(query_params)
            elif handler.command == "POST":
                # Verify webhook signature
                # Note: No RBAC for webhook - uses signature verification instead
                if not verify_signature(handler):
                    logger.warning("WhatsApp signature verification failed")
                    return error_response("Unauthorized", 401)
                return self._webhook_processor.handle_webhook(handler)

        return error_response("Not found", 404)

    def handle_post(self, path: str, body: dict[str, Any], handler: Any) -> HandlerResult | None:
        """Handle POST requests."""
        return self.handle(path, {}, handler)

    def _get_status(self) -> HandlerResult:
        """Get WhatsApp integration status."""
        return json_response(
            {
                "enabled": bool(WHATSAPP_ACCESS_TOKEN and WHATSAPP_PHONE_NUMBER_ID),
                "access_token_configured": bool(WHATSAPP_ACCESS_TOKEN),
                "phone_number_id_configured": bool(WHATSAPP_PHONE_NUMBER_ID),
                "verify_token_configured": bool(WHATSAPP_VERIFY_TOKEN),
                "app_secret_configured": bool(WHATSAPP_APP_SECRET),
            }
        )

    # =========================================================================
    # Backward-compatible delegate methods
    # =========================================================================

    def _verify_signature(self, handler: Any) -> bool:
        """Verify WhatsApp webhook signature. Delegates to webhooks module."""
        return verify_signature(handler)

    def _verify_webhook(self, query_params: dict[str, Any]) -> HandlerResult:
        """Handle Meta webhook verification. Delegates to webhooks module."""
        return verify_webhook(query_params)

    def _handle_webhook(self, handler: Any) -> HandlerResult:
        """Handle incoming webhook events. Delegates to WebhookProcessor."""
        return self._webhook_processor.handle_webhook(handler)

    def _command_help(self) -> str:
        """Return help message. Delegates to commands module."""
        return command_help()

    def _command_status(self) -> str:
        """Return status message. Delegates to commands module."""
        return command_status()

    def _command_agents(self) -> str:
        """Return agents list. Delegates to commands module."""
        return command_agents()

    def _command_debate(
        self,
        from_number: str,
        profile_name: str,
        topic: str,
        decision_integrity: dict[str, Any] | bool | None = None,
    ) -> None:
        """Handle debate command. Delegates to commands module."""
        command_debate(self, from_number, profile_name, topic, decision_integrity)

    def _command_gauntlet(self, from_number: str, profile_name: str, statement: str) -> None:
        """Handle gauntlet command. Delegates to commands module."""
        command_gauntlet(self, from_number, profile_name, statement)

    async def _send_text_message_async(self, to_number: str, text: str) -> None:
        """Send text message. Delegates to messaging module."""
        await send_text_message(to_number, text)

    # =========================================================================
    # Message processing
    # =========================================================================

    def _process_messages(self, value: dict[str, Any]) -> None:
        """Process incoming messages from webhook."""
        messages = value.get("messages", [])
        contacts = value.get("contacts", [])

        # Build contact lookup
        contact_map = {c.get("wa_id"): c for c in contacts}

        for message in messages:
            msg_type = message.get("type")
            from_number = message.get("from")
            contact = contact_map.get(from_number, {})
            profile_name = contact.get("profile", {}).get("name", "User")

            if msg_type == "text":
                text = message.get("text", {}).get("body", "")
                record_message("whatsapp", "text")
                self._handle_text_message(from_number, profile_name, text)
            elif msg_type == "interactive":
                record_message("whatsapp", "interactive")
                self._handle_interactive_reply(from_number, profile_name, message)
            elif msg_type == "button":
                record_message("whatsapp", "button")
                # Quick reply button
                button_text = message.get("button", {}).get("text", "")
                self._handle_button_reply(from_number, profile_name, button_text, message)

    def _handle_text_message(
        self,
        from_number: str,
        profile_name: str,
        text: str,
    ) -> None:
        """Handle incoming text message.

        RBAC: Requires whatsapp:messages:send permission for sending responses.
        """
        text = text.strip()
        logger.info(f"WhatsApp message from {profile_name} ({from_number}): {text[:50]}...")

        # Validate phone number format
        is_valid, validation_error = self._validate_phone_number(from_number)
        if not is_valid:
            logger.warning(f"Invalid WhatsApp phone number: {validation_error}")
            return

        # RBAC: Check permission to interact with WhatsApp (send messages back)
        perm_error = self._check_whatsapp_permission(
            from_number, PERM_WHATSAPP_MESSAGES, profile_name
        )
        if perm_error:
            logger.warning(f"WhatsApp user {from_number} denied: {perm_error}")
            # Still allow receiving - just log and continue
            # (webhook must return 200 to WhatsApp regardless)

        # Emit webhook event for message received
        emit_message_received(
            platform="whatsapp",
            chat_id=from_number,
            user_id=from_number,
            username=profile_name,
            message_text=text,
            message_type="text",
        )

        # Parse commands (lowercase first word)
        lower_text = text.lower()

        if lower_text == "help":
            record_command("whatsapp", "help")
            emit_command_received("whatsapp", from_number, from_number, profile_name, "help")
            response = command_help()
        elif lower_text == "status":
            record_command("whatsapp", "status")
            emit_command_received("whatsapp", from_number, from_number, profile_name, "status")
            response = command_status()
        elif lower_text == "agents":
            record_command("whatsapp", "agents")
            emit_command_received("whatsapp", from_number, from_number, profile_name, "agents")
            response = command_agents()
        elif lower_text.startswith("debate "):
            record_command("whatsapp", "debate")
            topic = text[7:].strip()
            emit_command_received(
                "whatsapp", from_number, from_number, profile_name, "debate", topic
            )
            command_debate(self, from_number, profile_name, topic)
            return
        elif lower_text.startswith("plan "):
            record_command("whatsapp", "plan")
            topic = text[5:].strip()
            emit_command_received("whatsapp", from_number, from_number, profile_name, "plan", topic)
            decision_integrity = {
                "include_receipt": True,
                "include_plan": True,
                "include_context": False,
                "plan_strategy": "single_task",
                "notify_origin": True,
            }
            command_debate(self, from_number, profile_name, topic, decision_integrity)
            return
        elif lower_text.startswith("implement "):
            record_command("whatsapp", "implement")
            topic = text[10:].strip()
            emit_command_received(
                "whatsapp", from_number, from_number, profile_name, "implement", topic
            )
            decision_integrity = {
                "include_receipt": True,
                "include_plan": True,
                "include_context": True,
                "plan_strategy": "single_task",
                "notify_origin": True,
            }
            command_debate(self, from_number, profile_name, topic, decision_integrity)
            return
        elif lower_text.startswith("gauntlet "):
            record_command("whatsapp", "gauntlet")
            statement = text[9:].strip()
            emit_command_received(
                "whatsapp", from_number, from_number, profile_name, "gauntlet", statement
            )
            command_gauntlet(self, from_number, profile_name, statement)
            return
        elif lower_text.startswith("search "):
            record_command("whatsapp", "search")
            query = text[7:].strip()
            emit_command_received(
                "whatsapp", from_number, from_number, profile_name, "search", query
            )
            response = command_search(query)
        elif lower_text == "recent":
            record_command("whatsapp", "recent")
            emit_command_received("whatsapp", from_number, from_number, profile_name, "recent")
            response = command_recent()
        elif lower_text.startswith("receipt "):
            record_command("whatsapp", "receipt")
            debate_id = text[8:].strip()
            emit_command_received(
                "whatsapp", from_number, from_number, profile_name, "receipt", debate_id
            )
            response = command_receipt(debate_id)
        elif len(text) > 10:
            # Treat longer messages as potential topics
            response = (
                f'I received: "{text[:50]}..."\n\nTo start a debate, type:\ndebate {text[:50]}'
            )
        else:
            response = "Type *help* to see available commands."

        _config.create_tracked_task(
            send_text_message(from_number, response),
            name=f"whatsapp-reply-{from_number}",
        )

    def _handle_interactive_reply(
        self,
        from_number: str,
        profile_name: str,
        message: dict[str, Any],
    ) -> None:
        """Handle interactive message reply (button clicks).

        RBAC: Permission checks are delegated to specific action handlers
        (_record_vote, _send_debate_details) based on the button action.
        """
        # Validate phone number format first
        is_valid, validation_error = self._validate_phone_number(from_number)
        if not is_valid:
            logger.warning(
                f"Invalid WhatsApp phone number in interactive reply: {validation_error}"
            )
            return

        interactive = message.get("interactive", {})
        reply_type = interactive.get("type")

        if reply_type == "button_reply":
            button = interactive.get("button_reply", {})
            button_id = button.get("id", "")
            self._process_button_click(from_number, profile_name, button_id)
        elif reply_type == "list_reply":
            list_reply = interactive.get("list_reply", {})
            item_id = list_reply.get("id", "")
            self._process_button_click(from_number, profile_name, item_id)

    def _handle_button_reply(
        self,
        from_number: str,
        profile_name: str,
        button_text: str,
        message: dict[str, Any],
    ) -> None:
        """Handle quick reply button.

        RBAC: Validates phone number; specific permission checks are
        delegated to action handlers if needed.
        """
        # Validate phone number format first
        is_valid, validation_error = self._validate_phone_number(from_number)
        if not is_valid:
            logger.warning(f"Invalid WhatsApp phone number in button reply: {validation_error}")
            return

        # Map button text to action
        lower_text = button_text.lower()
        if "agree" in lower_text:
            # Extract debate_id from context if available
            message.get("context", {})
            # For quick replies, we might not have the ID directly
            logger.info(f"Quick reply 'agree' from {profile_name}")
        elif "disagree" in lower_text:
            logger.info(f"Quick reply 'disagree' from {profile_name}")

    def _process_button_click(
        self,
        from_number: str,
        profile_name: str,
        button_id: str,
    ) -> None:
        """Process button click by ID."""
        logger.info(f"Button click from {profile_name}: {button_id}")

        if button_id.startswith("vote_agree_"):
            debate_id = button_id[11:]
            self._record_vote(from_number, profile_name, debate_id, "agree")
        elif button_id.startswith("vote_disagree_"):
            debate_id = button_id[14:]
            self._record_vote(from_number, profile_name, debate_id, "disagree")
        elif button_id.startswith("details_"):
            debate_id = button_id[8:]
            self._send_debate_details(from_number, debate_id)

    def _record_vote(
        self,
        from_number: str,
        profile_name: str,
        debate_id: str,
        vote_option: str,
    ) -> None:
        """Record a vote.

        RBAC: Requires whatsapp:votes:record permission.
        """
        # RBAC: Check permission to record votes
        perm_error = self._check_whatsapp_permission(
            from_number, PERM_WHATSAPP_VOTES, profile_name, resource_id=debate_id
        )
        if perm_error:
            _config.create_tracked_task(
                send_text_message(
                    from_number,
                    "Sorry, you don't have permission to vote on debates. "
                    "Please contact your administrator.",
                ),
                name=f"whatsapp-vote-perm-denied-{from_number}",
            )
            return

        logger.info(f"Vote received: {debate_id} -> {vote_option} from {profile_name}")

        # Emit webhook event for vote received
        emit_vote_received(
            platform="whatsapp",
            chat_id=from_number,
            user_id=from_number,
            username=profile_name,
            debate_id=debate_id,
            vote=vote_option,
        )

        # Record vote metrics
        record_vote("whatsapp", vote_option)

        try:
            from aragora.server.storage import get_debates_db

            db = get_debates_db()
            if db and hasattr(db, "record_vote"):
                db.record_vote(
                    debate_id=debate_id,
                    voter_id=f"whatsapp:{from_number}",
                    vote=vote_option,
                    source="whatsapp",
                )
        except Exception as e:
            logger.warning(f"Failed to record vote: {e}")

        emoji = "+" if vote_option == "agree" else "-"
        _config.create_tracked_task(
            send_text_message(
                from_number,
                f"{emoji} Your vote for '{vote_option}' has been recorded!",
            ),
            name=f"whatsapp-vote-ack-{from_number}",
        )

    def _send_debate_details(
        self, from_number: str, debate_id: str, profile_name: str | None = None
    ) -> None:
        """Send debate details.

        RBAC: Requires whatsapp:debates:read permission.
        """
        # RBAC: Check permission to view debate details
        perm_error = self._check_whatsapp_permission(
            from_number, PERM_WHATSAPP_DETAILS, profile_name, resource_id=debate_id
        )
        if perm_error:
            _config.create_tracked_task(
                send_text_message(
                    from_number,
                    "Sorry, you don't have permission to view debate details. "
                    "Please contact your administrator.",
                ),
                name=f"whatsapp-details-perm-denied-{from_number}",
            )
            return

        debate_data = None
        try:
            from aragora.server.storage import get_debates_db

            db = get_debates_db()
            if db:
                debate_data = db.get(debate_id)
        except Exception as e:
            logger.warning(f"Failed to fetch debate: {e}")

        if not debate_data:
            _config.create_tracked_task(
                send_text_message(
                    from_number,
                    f"Debate {debate_id} not found",
                ),
                name=f"whatsapp-details-notfound-{from_number}",
            )
            return

        task = debate_data.get("task", "Unknown")
        final_answer = debate_data.get("final_answer", "No conclusion")
        consensus = debate_data.get("consensus_reached", False)
        confidence = debate_data.get("confidence", 0)
        rounds_used = debate_data.get("rounds_used", 0)
        agents = debate_data.get("agents", [])

        agent_list = ", ".join(agents[:5]) if agents else "Unknown"
        if len(agents) > 5:
            agent_list += f" (+{len(agents) - 5} more)"

        response = (
            f"*Debate Details*\n\n"
            f"*Topic:*\n{task[:200]}{'...' if len(task) > 200 else ''}\n\n"
            f"*ID:* {debate_id}\n"
            f"*Consensus:* {'Yes' if consensus else 'No'}\n"
            f"*Confidence:* {confidence:.1%}\n"
            f"*Rounds:* {rounds_used}\n"
            f"*Agents:* {agent_list}\n\n"
            f"*Conclusion:*\n{final_answer[:500] if final_answer else 'No conclusion'}{'...' if final_answer and len(final_answer) > 500 else ''}"
        )

        _config.create_tracked_task(
            send_text_message(from_number, response),
            name=f"whatsapp-details-{from_number}",
        )


# Export handler factory
_whatsapp_handler: Optional["WhatsAppHandler"] = None


def get_whatsapp_handler(server_context: dict[str, Any] | None = None) -> "WhatsAppHandler":
    """Get or create the WhatsApp handler instance."""
    global _whatsapp_handler
    if _whatsapp_handler is None:
        ctx: dict[str, Any] = server_context if server_context is not None else {}
        _whatsapp_handler = WhatsAppHandler(ctx)
    return _whatsapp_handler
