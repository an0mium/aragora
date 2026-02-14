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

