"""
Search Namespace API

Provides methods for semantic and full-text search:
- Universal search across all content types
- Search across debates and decisions
- Knowledge base search
- Agent search by capability
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

SearchScope = Literal["all", "debates", "knowledge", "agents", "decisions", "receipts"]


class SearchAPI:
    """
    Synchronous Search API.

    Provides universal search across debates, knowledge base, agents,
    decisions, and receipts with support for semantic and full-text modes.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> results = client.search.search("rate limiting best practices")
        >>> debates = client.search.search_debates("consensus algorithm")
        >>> agents = client.search.search_agents(capability="reasoning")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def search(
        self,
        query: str,
        scope: SearchScope = "all",
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Universal search across all content.

        Args:
            query: Search query string.
            scope: Search scope to restrict results (all, debates,
                knowledge, agents, decisions, receipts).
            limit: Maximum number of results.
            offset: Pagination offset.

        Returns:
            Dict with search results including:
            - results: List of matching items with relevance scores
            - total: Total number of matches
            - query: The search query used
        """
        return self._client.request(
            "POST",
            "/api/v1/search",
            json={
                "query": query,
                "scope": scope,
                "limit": limit,
                "offset": offset,
            },
        )

    def search_debates(
        self,
        query: str,
        status: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """
        Search across debates.

        Args:
            query: Search query string.
            status: Filter by debate status (active, completed, failed).
            limit: Maximum number of results.

        Returns:
            Dict with matching debates and relevance scores.
        """
        body: dict[str, Any] = {
            "query": query,
            "scope": "debates",
            "limit": limit,
        }
        if status:
            body["status"] = status
        return self._client.request("POST", "/api/v1/search", json=body)

    def search_knowledge(
        self,
        query: str,
        category: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """
        Search the knowledge base.

        Args:
            query: Search query string.
            category: Filter by knowledge category.
            limit: Maximum number of results.

        Returns:
            Dict with matching knowledge entries and relevance scores.
        """
        body: dict[str, Any] = {
            "query": query,
            "scope": "knowledge",
            "limit": limit,
        }
        if category:
            body["category"] = category
        return self._client.request("POST", "/api/v1/search", json=body)

    def search_agents(
        self,
        query: str | None = None,
        capability: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """
        Search for agents by name or capability.

        Args:
            query: Optional search query string.
            capability: Filter by agent capability.
            limit: Maximum number of results.

        Returns:
            Dict with matching agents and their capabilities.
        """
        body: dict[str, Any] = {
            "scope": "agents",
            "limit": limit,
        }
        if query:
            body["query"] = query
        if capability:
            body["capability"] = capability
        return self._client.request("POST", "/api/v1/search", json=body)


class AsyncSearchAPI:
    """
    Asynchronous Search API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     results = await client.search.search("rate limiting")
        ...     debates = await client.search.search_debates("consensus")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def search(
        self,
        query: str,
        scope: SearchScope = "all",
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Universal search across all content."""
        return await self._client.request(
            "POST",
            "/api/v1/search",
            json={
                "query": query,
                "scope": scope,
                "limit": limit,
                "offset": offset,
            },
        )

    async def search_debates(
        self,
        query: str,
        status: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Search across debates."""
        body: dict[str, Any] = {
            "query": query,
            "scope": "debates",
            "limit": limit,
        }
        if status:
            body["status"] = status
        return await self._client.request("POST", "/api/v1/search", json=body)

    async def search_knowledge(
        self,
        query: str,
        category: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Search the knowledge base."""
        body: dict[str, Any] = {
            "query": query,
            "scope": "knowledge",
            "limit": limit,
        }
        if category:
            body["category"] = category
        return await self._client.request("POST", "/api/v1/search", json=body)

    async def search_agents(
        self,
        query: str | None = None,
        capability: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Search for agents by name or capability."""
        body: dict[str, Any] = {
            "scope": "agents",
            "limit": limit,
        }
        if query:
            body["query"] = query
        if capability:
            body["capability"] = capability
        return await self._client.request("POST", "/api/v1/search", json=body)
