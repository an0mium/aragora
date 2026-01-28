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

    def get_channel_summary(self, channel_id: str) -> dict[str, Any]:
        """Get knowledge summary for a channel.

        Args:
            channel_id: Channel identifier.

        Returns:
            Channel knowledge summary.
        """
        return self._client.request("GET", f"/api/v1/chat/knowledge/channel/{channel_id}/summary")


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

    async def get_channel_summary(self, channel_id: str) -> dict[str, Any]:
        """Get knowledge summary for a channel.

        Args:
            channel_id: Channel identifier.

        Returns:
            Channel knowledge summary.
        """
        return await self._client.request(
            "GET", f"/api/v1/chat/knowledge/channel/{channel_id}/summary"
        )
