"""
Chat Namespace API

Provides access to chat-based knowledge operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class ChatAPI:
    """Synchronous Chat API for knowledge operations."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def search_knowledge(
        self,
        query: str,
        channel_id: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Search knowledge from chat context.

        Args:
            query: Search query.
            channel_id: Optional channel to scope search.
            limit: Maximum results.

        Returns:
            Search results with matched knowledge.
        """
        body: dict[str, Any] = {"query": query, "limit": limit}
        if channel_id:
            body["channel_id"] = channel_id
        return self._client.request("POST", "/api/v1/chat/knowledge/search", json=body)

    def inject_knowledge(
        self,
        content: str,
        channel_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Inject knowledge into chat context.

        Args:
            content: Knowledge content to inject.
            channel_id: Target channel.
            metadata: Optional metadata.

        Returns:
            Injection result.
        """
        body: dict[str, Any] = {"content": content, "channel_id": channel_id}
        if metadata:
            body["metadata"] = metadata
        return self._client.request("POST", "/api/v1/chat/knowledge/inject", json=body)

    def store_knowledge(
        self,
        content: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store knowledge from chat.

        Args:
            content: Knowledge content.
            source: Source identifier.
            metadata: Optional metadata.

        Returns:
            Storage result with knowledge ID.
        """
        body: dict[str, Any] = {"content": content, "source": source}
        if metadata:
            body["metadata"] = metadata
        return self._client.request("POST", "/api/v1/chat/knowledge/store", json=body)

    def get_status(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Get chat integration status."""
        return self._client.request("GET", "/api/v1/chat/status", params=params)

    def receive_webhook(
        self, body: dict[str, Any] | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Submit an auto-detected chat webhook event."""
        return self._client.request("POST", "/api/v1/chat/webhook", json=body, params=params)

    def receive_slack_webhook(
        self, body: dict[str, Any] | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self._client.request("POST", "/api/v1/chat/slack/webhook", json=body, params=params)

    def receive_teams_webhook(
        self, body: dict[str, Any] | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self._client.request("POST", "/api/v1/chat/teams/webhook", json=body, params=params)

    def receive_discord_webhook(
        self, body: dict[str, Any] | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self._client.request(
            "POST", "/api/v1/chat/discord/webhook", json=body, params=params
        )

    def receive_google_chat_webhook(
        self, body: dict[str, Any] | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self._client.request(
            "POST", "/api/v1/chat/google_chat/webhook", json=body, params=params
        )

    def receive_telegram_webhook(
        self, body: dict[str, Any] | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self._client.request(
            "POST", "/api/v1/chat/telegram/webhook", json=body, params=params
        )

    def receive_whatsapp_webhook(
        self, body: dict[str, Any] | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self._client.request(
            "POST", "/api/v1/chat/whatsapp/webhook", json=body, params=params
        )


class AsyncChatAPI:
    """Asynchronous Chat API for knowledge operations."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def search_knowledge(
        self,
        query: str,
        channel_id: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Search knowledge from chat context.

        Args:
            query: Search query.
            channel_id: Optional channel to scope search.
            limit: Maximum results.

        Returns:
            Search results with matched knowledge.
        """
        body: dict[str, Any] = {"query": query, "limit": limit}
        if channel_id:
            body["channel_id"] = channel_id
        return await self._client.request("POST", "/api/v1/chat/knowledge/search", json=body)

    async def inject_knowledge(
        self,
        content: str,
        channel_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Inject knowledge into chat context.

        Args:
            content: Knowledge content to inject.
            channel_id: Target channel.
            metadata: Optional metadata.

        Returns:
            Injection result.
        """
        body: dict[str, Any] = {"content": content, "channel_id": channel_id}
        if metadata:
            body["metadata"] = metadata
        return await self._client.request("POST", "/api/v1/chat/knowledge/inject", json=body)

    async def store_knowledge(
        self,
        content: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store knowledge from chat.

        Args:
            content: Knowledge content.
            source: Source identifier.
            metadata: Optional metadata.

        Returns:
            Storage result with knowledge ID.
        """
        body: dict[str, Any] = {"content": content, "source": source}
        if metadata:
            body["metadata"] = metadata
        return await self._client.request("POST", "/api/v1/chat/knowledge/store", json=body)

    async def get_status(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Get chat integration status."""
        return await self._client.request("GET", "/api/v1/chat/status", params=params)

    async def receive_webhook(
        self, body: dict[str, Any] | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Submit an auto-detected chat webhook event."""
        return await self._client.request("POST", "/api/v1/chat/webhook", json=body, params=params)

    async def receive_slack_webhook(
        self, body: dict[str, Any] | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return await self._client.request(
            "POST", "/api/v1/chat/slack/webhook", json=body, params=params
        )

    async def receive_teams_webhook(
        self, body: dict[str, Any] | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return await self._client.request(
            "POST", "/api/v1/chat/teams/webhook", json=body, params=params
        )

    async def receive_discord_webhook(
        self, body: dict[str, Any] | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return await self._client.request(
            "POST", "/api/v1/chat/discord/webhook", json=body, params=params
        )

    async def receive_google_chat_webhook(
        self, body: dict[str, Any] | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return await self._client.request(
            "POST", "/api/v1/chat/google_chat/webhook", json=body, params=params
        )

    async def receive_telegram_webhook(
        self, body: dict[str, Any] | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return await self._client.request(
            "POST", "/api/v1/chat/telegram/webhook", json=body, params=params
        )

    async def receive_whatsapp_webhook(
        self, body: dict[str, Any] | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return await self._client.request(
            "POST", "/api/v1/chat/whatsapp/webhook", json=body, params=params
        )
