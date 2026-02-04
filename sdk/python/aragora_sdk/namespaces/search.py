"""
Search Namespace API

Provides methods for semantic and full-text search:
- Search across debates
- Knowledge base search
- Agent search
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class SearchAPI:
    """Synchronous Search API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def search(self, query: str, scope: str = "all", limit: int = 20) -> dict[str, Any]:
        """Universal search across all content."""
        return self._client.request(
            "POST",
            "/api/v1/search",
            json={
                "query": query,
                "scope": scope,
                "limit": limit,
            },
        )

    def search_debates(
        self, query: str, status: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        """Search debates."""
        params: dict[str, Any] = {"query": query, "limit": limit}
        if status:
            params["status"] = status
        return self._client.request("GET", "/api/v1/search/debates", params=params)

    def search_knowledge(
        self, query: str, mound_id: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        """Search knowledge base."""
        params: dict[str, Any] = {"query": query, "limit": limit}
        if mound_id:
            params["mound_id"] = mound_id
        return self._client.request("GET", "/api/v1/search/knowledge", params=params)

    def search_agents(self, query: str, limit: int = 20) -> dict[str, Any]:
        """Search agents."""
        return self._client.request(
            "GET", "/api/v1/search/agents", params={"query": query, "limit": limit}
        )

    def semantic_search(self, query: str, collection: str, limit: int = 20) -> dict[str, Any]:
        """Perform semantic search."""
        return self._client.request(
            "POST",
            "/api/v1/search/semantic",
            json={
                "query": query,
                "collection": collection,
                "limit": limit,
            },
        )

    def suggest(self, prefix: str, scope: str = "all", limit: int = 10) -> dict[str, Any]:
        """Get search suggestions."""
        return self._client.request(
            "GET",
            "/api/v1/search/suggest",
            params={
                "prefix": prefix,
                "scope": scope,
                "limit": limit,
            },
        )


class AsyncSearchAPI:
    """Asynchronous Search API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def search(self, query: str, scope: str = "all", limit: int = 20) -> dict[str, Any]:
        """Universal search across all content."""
        return await self._client.request(
            "POST",
            "/api/v1/search",
            json={
                "query": query,
                "scope": scope,
                "limit": limit,
            },
        )

    async def search_debates(
        self, query: str, status: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        """Search debates."""
        params: dict[str, Any] = {"query": query, "limit": limit}
        if status:
            params["status"] = status
        return await self._client.request("GET", "/api/v1/search/debates", params=params)

    async def search_knowledge(
        self, query: str, mound_id: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        """Search knowledge base."""
        params: dict[str, Any] = {"query": query, "limit": limit}
        if mound_id:
            params["mound_id"] = mound_id
        return await self._client.request("GET", "/api/v1/search/knowledge", params=params)

    async def search_agents(self, query: str, limit: int = 20) -> dict[str, Any]:
        """Search agents."""
        return await self._client.request(
            "GET", "/api/v1/search/agents", params={"query": query, "limit": limit}
        )

    async def semantic_search(self, query: str, collection: str, limit: int = 20) -> dict[str, Any]:
        """Perform semantic search."""
        return await self._client.request(
            "POST",
            "/api/v1/search/semantic",
            json={
                "query": query,
                "collection": collection,
                "limit": limit,
            },
        )

    async def suggest(self, prefix: str, scope: str = "all", limit: int = 10) -> dict[str, Any]:
        """Get search suggestions."""
        return await self._client.request(
            "GET",
            "/api/v1/search/suggest",
            params={
                "prefix": prefix,
                "scope": scope,
                "limit": limit,
            },
        )
