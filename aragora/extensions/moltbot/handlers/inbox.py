"""
Moltbot Inbox Handler - Message and Channel Management REST API.

Endpoints:
- GET  /api/v1/moltbot/channels         - List channels
- POST /api/v1/moltbot/channels         - Register channel
- GET  /api/v1/moltbot/channels/{id}    - Get channel
- DELETE /api/v1/moltbot/channels/{id}  - Unregister channel
- GET  /api/v1/moltbot/messages         - List messages
- POST /api/v1/moltbot/messages         - Send message
- GET  /api/v1/moltbot/messages/{id}    - Get message
- POST /api/v1/moltbot/messages/{id}/read - Mark as read
- GET  /api/v1/moltbot/threads/{id}     - Get thread
- GET  /api/v1/moltbot/inbox/stats      - Inbox statistics
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)
from aragora.server.handlers.utils.params import get_clamped_int_param


if TYPE_CHECKING:
    from aragora.extensions.moltbot import InboxManager

logger = logging.getLogger(__name__)

# Global inbox instance (lazily initialized)
_inbox: Optional["InboxManager"] = None


def get_inbox() -> "InboxManager":
    """Get or create the inbox manager instance."""
    global _inbox
    if _inbox is None:
        from aragora.extensions.moltbot import InboxManager

        _inbox = InboxManager()
    return _inbox


class MoltbotInboxHandler(BaseHandler):
    """HTTP handler for Moltbot inbox operations."""

    routes = [
        ("GET", "/api/v1/moltbot/channels"),
        ("POST", "/api/v1/moltbot/channels"),
        ("GET", "/api/v1/moltbot/channels/"),
        ("DELETE", "/api/v1/moltbot/channels/"),
        ("GET", "/api/v1/moltbot/messages"),
        ("POST", "/api/v1/moltbot/messages"),
        ("GET", "/api/v1/moltbot/messages/"),
        ("POST", "/api/v1/moltbot/messages/*/read"),
        ("GET", "/api/v1/moltbot/threads/"),
        ("GET", "/api/v1/moltbot/inbox/stats"),
    ]

    async def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle GET requests."""
        if path == "/api/v1/moltbot/channels":
            return await self._handle_list_channels(query_params, handler)
        elif path == "/api/v1/moltbot/messages":
            return await self._handle_list_messages(query_params, handler)
        elif path == "/api/v1/moltbot/inbox/stats":
            return await self._handle_inbox_stats(handler)
        elif path.startswith("/api/v1/moltbot/channels/"):
            parts = path.split("/")
            if len(parts) >= 5:
                channel_id = parts[4]
                return await self._handle_get_channel(channel_id, handler)
        elif path.startswith("/api/v1/moltbot/messages/"):
            parts = path.split("/")
            if len(parts) >= 5:
                message_id = parts[4]
                return await self._handle_get_message(message_id, handler)
        elif path.startswith("/api/v1/moltbot/threads/"):
            parts = path.split("/")
            if len(parts) >= 5:
                thread_id = parts[4]
                return await self._handle_get_thread(thread_id, handler)
        return None

    async def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle POST requests."""
        if path == "/api/v1/moltbot/channels":
            return await self._handle_register_channel(handler)
        elif path == "/api/v1/moltbot/messages":
            return await self._handle_send_message(handler)
        elif path.endswith("/read"):
            parts = path.split("/")
            if len(parts) >= 5:
                message_id = parts[4]
                return await self._handle_mark_read(message_id, handler)
        return None

    async def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle DELETE requests."""
        if path.startswith("/api/v1/moltbot/channels/"):
            parts = path.split("/")
            if len(parts) >= 5:
                channel_id = parts[4]
                return await self._handle_unregister_channel(channel_id, handler)
        return None

    # ========== Channel Handlers ==========

    async def _handle_list_channels(
        self, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult:
        """
        List channels with optional filters.

        GET /api/v1/moltbot/channels?user_id=...&channel_type=...
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        inbox = get_inbox()

        # Extract filters
        user_id = query_params.get("user_id")
        channel_type_str = query_params.get("channel_type")
        tenant_id = query_params.get("tenant_id")

        channel_type = None
        if channel_type_str:
            from aragora.extensions.moltbot import ChannelType

            try:
                channel_type = ChannelType(channel_type_str)
            except ValueError:
                pass

        channels = await inbox.list_channels(
            user_id=user_id,
            channel_type=channel_type,
            tenant_id=tenant_id,
        )

        return json_response(
            {
                "channels": [
                    {
                        "id": c.id,
                        "name": c.config.name,
                        "type": c.config.type.value,
                        "status": c.status,
                        "user_id": c.user_id,
                        "message_count": c.message_count,
                        "last_message_at": c.last_message_at.isoformat()
                        if c.last_message_at
                        else None,
                    }
                    for c in channels
                ],
                "total": len(channels),
            }
        )

    async def _handle_register_channel(self, handler: Any) -> HandlerResult:
        """
        Register a new channel.

        POST /api/v1/moltbot/channels
        Body: {name, type, provider_config?, metadata?}
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        if not body:
            return error_response("Request body required", 400)

        name = body.get("name")
        channel_type_str = body.get("type")

        if not name or not channel_type_str:
            return error_response("name and type are required", 400)

        from aragora.extensions.moltbot import ChannelConfig, ChannelType

        try:
            channel_type = ChannelType(channel_type_str)
        except ValueError:
            return error_response(f"Invalid channel type: {channel_type_str}", 400)

        config = ChannelConfig(  # type: ignore[call-arg]
            name=name,
            type=channel_type,
            provider_config=body.get("provider_config", {}),
            metadata=body.get("metadata", {}),
        )

        inbox = get_inbox()
        channel = await inbox.register_channel(
            config=config,
            user_id=user.user_id,
            tenant_id=body.get("tenant_id"),
        )

        return json_response(
            {
                "success": True,
                "channel": {
                    "id": channel.id,
                    "name": channel.config.name,
                    "type": channel.config.type.value,
                    "status": channel.status,
                },
            },
            status=201,
        )

    async def _handle_get_channel(self, channel_id: str, handler: Any) -> HandlerResult:
        """
        Get a channel by ID.

        GET /api/v1/moltbot/channels/{channel_id}
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        inbox = get_inbox()
        channel = await inbox.get_channel(channel_id)

        if not channel:
            return error_response("Channel not found", 404)

        return json_response(
            {
                "channel": {
                    "id": channel.id,
                    "name": channel.config.name,
                    "type": channel.config.type.value,
                    "status": channel.status,
                    "user_id": channel.user_id,
                    "tenant_id": channel.tenant_id,
                    "message_count": channel.message_count,
                    "last_message_at": channel.last_message_at.isoformat()
                    if channel.last_message_at
                    else None,
                    "created_at": channel.created_at.isoformat() if channel.created_at else None,
                    "updated_at": channel.updated_at.isoformat() if channel.updated_at else None,
                },
            }
        )

    async def _handle_unregister_channel(self, channel_id: str, handler: Any) -> HandlerResult:
        """
        Unregister a channel.

        DELETE /api/v1/moltbot/channels/{channel_id}
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        inbox = get_inbox()
        success = await inbox.unregister_channel(channel_id)

        if not success:
            return error_response("Channel not found", 404)

        return json_response(
            {
                "success": True,
                "message": f"Channel {channel_id} unregistered",
            }
        )

    # ========== Message Handlers ==========

    async def _handle_list_messages(
        self, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult:
        """
        List messages with optional filters.

        GET /api/v1/moltbot/messages?channel_id=...&thread_id=...&limit=...
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        inbox = get_inbox()

        # Extract filters
        channel_id = query_params.get("channel_id")
        user_id = query_params.get("user_id")
        thread_id = query_params.get("thread_id")
        direction = query_params.get("direction")
        limit = get_clamped_int_param(query_params, "limit", 100, 1, 1000)
        offset = get_clamped_int_param(query_params, "offset", 0, 0, 100000)

        messages = await inbox.list_messages(
            channel_id=channel_id,
            user_id=user_id,
            thread_id=thread_id,
            direction=direction,
            limit=limit,
            offset=offset,
        )

        return json_response(
            {
                "messages": [
                    {
                        "id": m.id,
                        "channel_id": m.channel_id,
                        "user_id": m.user_id,
                        "direction": m.direction,
                        "content": m.content,
                        "content_type": m.content_type,
                        "status": m.status.value,
                        "thread_id": m.thread_id,
                        "reply_to": m.reply_to,
                        "intent": m.intent,
                        "created_at": m.created_at.isoformat() if m.created_at else None,
                    }
                    for m in messages
                ],
                "total": len(messages),
                "limit": limit,
                "offset": offset,
            }
        )

    async def _handle_send_message(self, handler: Any) -> HandlerResult:
        """
        Send a message through a channel.

        POST /api/v1/moltbot/messages
        Body: {channel_id, recipient_user_id, content, content_type?, thread_id?, reply_to?}
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        if not body:
            return error_response("Request body required", 400)

        channel_id = body.get("channel_id")
        recipient = body.get("recipient_user_id")
        content = body.get("content")

        if not channel_id or not recipient or not content:
            return error_response("channel_id, recipient_user_id, and content are required", 400)

        inbox = get_inbox()

        try:
            message = await inbox.send_message(
                channel_id=channel_id,
                user_id=recipient,
                content=content,
                content_type=body.get("content_type", "text"),
                thread_id=body.get("thread_id"),
                reply_to=body.get("reply_to"),
                metadata=body.get("metadata"),
            )
        except ValueError as e:
            return error_response(str(e), 400)

        return json_response(
            {
                "success": True,
                "message": {
                    "id": message.id,
                    "channel_id": message.channel_id,
                    "status": message.status.value,
                    "thread_id": message.thread_id,
                },
            },
            status=201,
        )

    async def _handle_get_message(self, message_id: str, handler: Any) -> HandlerResult:
        """
        Get a message by ID.

        GET /api/v1/moltbot/messages/{message_id}
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        inbox = get_inbox()
        message = await inbox.get_message(message_id)

        if not message:
            return error_response("Message not found", 404)

        return json_response(
            {
                "message": {
                    "id": message.id,
                    "channel_id": message.channel_id,
                    "user_id": message.user_id,
                    "direction": message.direction,
                    "content": message.content,
                    "content_type": message.content_type,
                    "status": message.status.value,
                    "thread_id": message.thread_id,
                    "reply_to": message.reply_to,
                    "external_id": message.external_id,
                    "intent": message.intent,
                    "metadata": message.metadata,
                    "created_at": message.created_at.isoformat() if message.created_at else None,
                    "delivered_at": message.delivered_at.isoformat()
                    if message.delivered_at
                    else None,
                    "read_at": message.read_at.isoformat() if message.read_at else None,
                },
            }
        )

    async def _handle_mark_read(self, message_id: str, handler: Any) -> HandlerResult:
        """
        Mark a message as read.

        POST /api/v1/moltbot/messages/{message_id}/read
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        inbox = get_inbox()
        message = await inbox.mark_read(message_id)

        if not message:
            return error_response("Message not found", 404)

        return json_response(
            {
                "success": True,
                "message_id": message.id,
                "status": message.status.value,
                "read_at": message.read_at.isoformat() if message.read_at else None,
            }
        )

    async def _handle_get_thread(self, thread_id: str, handler: Any) -> HandlerResult:
        """
        Get all messages in a thread.

        GET /api/v1/moltbot/threads/{thread_id}
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        inbox = get_inbox()
        messages = await inbox.get_thread(thread_id)

        return json_response(
            {
                "thread_id": thread_id,
                "messages": [
                    {
                        "id": m.id,
                        "user_id": m.user_id,
                        "direction": m.direction,
                        "content": m.content,
                        "status": m.status.value,
                        "created_at": m.created_at.isoformat() if m.created_at else None,
                    }
                    for m in messages
                ],
                "total": len(messages),
            }
        )

    async def _handle_inbox_stats(self, handler: Any) -> HandlerResult:
        """
        Get inbox statistics.

        GET /api/v1/moltbot/inbox/stats
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        inbox = get_inbox()
        stats = await inbox.get_stats()

        return json_response(stats)
