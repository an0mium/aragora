"""
Knowledge Chat Namespace API.

Provides methods for Knowledge + Chat bridge integration:
- Search knowledge from chat context
- Inject knowledge into conversations
- Store chat messages as knowledge
- Channel knowledge summaries

Endpoints:
    POST /api/v1/chat/knowledge/search               - Search knowledge
    POST /api/v1/chat/knowledge/inject               - Inject knowledge
    POST /api/v1/chat/knowledge/store                - Store chat as knowledge
    GET  /api/v1/chat/knowledge/channel/:id/summary  - Channel summary
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

SearchScope = Literal["workspace", "channel", "user", "global"]
SearchStrategy = Literal["hybrid", "semantic", "keyword", "exact"]

class KnowledgeChatAPI:
    """
    Synchronous Knowledge Chat API.

    Provides methods for integrating knowledge with chat platforms:
    - Search for relevant knowledge based on chat context
    - Inject knowledge into ongoing conversations
    - Store important chat exchanges as knowledge
    - Get summaries of channel-related knowledge

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> # Search knowledge
        >>> results = client.knowledge_chat.search(
        ...     query="What's our vacation policy?",
        ...     workspace_id="ws_123",
        ...     channel_id="C123456",
        ... )
        >>> # Store chat as knowledge
        >>> stored = client.knowledge_chat.store(
        ...     messages=[
        ...         {"author": "user1", "content": "We decided to use Python 3.11"},
        ...         {"author": "user2", "content": "Agreed, better performance"},
        ...     ],
        ...     workspace_id="ws_123",
        ...     channel_name="#engineering",
        ... )
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Search
    # =========================================================================

    def search(
        self,
        query: str,
        workspace_id: str = "default",
        channel_id: str | None = None,
        user_id: str | None = None,
        scope: SearchScope = "workspace",
        strategy: SearchStrategy = "hybrid",
        node_types: list[str] | None = None,
        min_confidence: float = 0.3,
        max_results: int = 10,
    ) -> dict[str, Any]:
        """
        Search knowledge from chat context.

        Args:
            query: Search query text.
            workspace_id: Workspace to search in.
            channel_id: Optional channel ID for context.
            user_id: Optional user ID for personalization.
            scope: Search scope (workspace, channel, user, global).
            strategy: Search strategy (hybrid, semantic, keyword, exact).
            node_types: Filter by knowledge node types.
            min_confidence: Minimum confidence threshold.
            max_results: Maximum results to return.

        Returns:
            Dict with search results and metadata.
        """
        data: dict[str, Any] = {
            "query": query,
            "workspace_id": workspace_id,
        }

        if channel_id:
            data["channel_id"] = channel_id
        if user_id:
            data["user_id"] = user_id
        if scope != "workspace":
            data["scope"] = scope
        if strategy != "hybrid":
            data["strategy"] = strategy
        if node_types:
            data["node_types"] = node_types
        if min_confidence != 0.3:
            data["min_confidence"] = min_confidence
        if max_results != 10:
            data["max_results"] = max_results

        return self._client.request("POST", "/api/v1/chat/knowledge/search", json=data)

    # =========================================================================
    # Inject
    # =========================================================================

    def inject(
        self,
        messages: list[dict[str, Any]],
        workspace_id: str = "default",
        channel_id: str | None = None,
        max_context_items: int = 5,
    ) -> dict[str, Any]:
        """
        Get relevant knowledge to inject into a conversation.

        Analyzes the conversation and returns relevant knowledge items
        that could enhance the discussion.

        Args:
            messages: List of chat messages with author and content.
            workspace_id: Workspace ID.
            channel_id: Optional channel ID.
            max_context_items: Maximum knowledge items to return.

        Returns:
            Dict with context items and count.
        """
        data: dict[str, Any] = {
            "messages": messages,
            "workspace_id": workspace_id,
        }

        if channel_id:
            data["channel_id"] = channel_id
        if max_context_items != 5:
            data["max_context_items"] = max_context_items

        return self._client.request("POST", "/api/v1/chat/knowledge/inject", json=data)

    # =========================================================================
    # Store
    # =========================================================================

    def store(
        self,
        messages: list[dict[str, Any]],
        workspace_id: str = "default",
        channel_id: str = "",
        channel_name: str = "",
        platform: str = "unknown",
        node_type: str = "chat_context",
    ) -> dict[str, Any]:
        """
        Store chat messages as persistent knowledge.

        Args:
            messages: List of chat messages (minimum 2 required).
            workspace_id: Workspace ID.
            channel_id: Channel ID.
            channel_name: Human-readable channel name.
            platform: Chat platform (slack, teams, discord, etc.).
            node_type: Knowledge node type.

        Returns:
            Dict with node_id and message count.

        Raises:
            ValueError: If fewer than 2 messages provided.
        """
        if len(messages) < 2:
            raise ValueError("At least 2 messages required")

        data: dict[str, Any] = {
            "messages": messages,
            "workspace_id": workspace_id,
        }

        if channel_id:
            data["channel_id"] = channel_id
        if channel_name:
            data["channel_name"] = channel_name
        if platform != "unknown":
            data["platform"] = platform
        if node_type != "chat_context":
            data["node_type"] = node_type

        return self._client.request("POST", "/api/v1/chat/knowledge/store", json=data)

    # =========================================================================
    # Channel Summary


class AsyncKnowledgeChatAPI:
    """Asynchronous Knowledge Chat API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    # =========================================================================
    # Search
    # =========================================================================

    async def search(
        self,
        query: str,
        workspace_id: str = "default",
        channel_id: str | None = None,
        user_id: str | None = None,
        scope: SearchScope = "workspace",
        strategy: SearchStrategy = "hybrid",
        node_types: list[str] | None = None,
        min_confidence: float = 0.3,
        max_results: int = 10,
    ) -> dict[str, Any]:
        """Search knowledge from chat context."""
        data: dict[str, Any] = {
            "query": query,
            "workspace_id": workspace_id,
        }

        if channel_id:
            data["channel_id"] = channel_id
        if user_id:
            data["user_id"] = user_id
        if scope != "workspace":
            data["scope"] = scope
        if strategy != "hybrid":
            data["strategy"] = strategy
        if node_types:
            data["node_types"] = node_types
        if min_confidence != 0.3:
            data["min_confidence"] = min_confidence
        if max_results != 10:
            data["max_results"] = max_results

        return await self._client.request("POST", "/api/v1/chat/knowledge/search", json=data)

    # =========================================================================
    # Inject
    # =========================================================================

    async def inject(
        self,
        messages: list[dict[str, Any]],
        workspace_id: str = "default",
        channel_id: str | None = None,
        max_context_items: int = 5,
    ) -> dict[str, Any]:
        """Get relevant knowledge to inject into a conversation."""
        data: dict[str, Any] = {
            "messages": messages,
            "workspace_id": workspace_id,
        }

        if channel_id:
            data["channel_id"] = channel_id
        if max_context_items != 5:
            data["max_context_items"] = max_context_items

        return await self._client.request("POST", "/api/v1/chat/knowledge/inject", json=data)

    # =========================================================================
    # Store
    # =========================================================================

    async def store(
        self,
        messages: list[dict[str, Any]],
        workspace_id: str = "default",
        channel_id: str = "",
        channel_name: str = "",
        platform: str = "unknown",
        node_type: str = "chat_context",
    ) -> dict[str, Any]:
        """Store chat messages as persistent knowledge."""
        if len(messages) < 2:
            raise ValueError("At least 2 messages required")

        data: dict[str, Any] = {
            "messages": messages,
            "workspace_id": workspace_id,
        }

        if channel_id:
            data["channel_id"] = channel_id
        if channel_name:
            data["channel_name"] = channel_name
        if platform != "unknown":
            data["platform"] = platform
        if node_type != "chat_context":
            data["node_type"] = node_type

        return await self._client.request("POST", "/api/v1/chat/knowledge/store", json=data)

    # =========================================================================
    # Channel Summary
