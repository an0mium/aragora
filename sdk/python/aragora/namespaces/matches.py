"""
Matches Namespace API

Provides access to agent match history and rankings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class MatchesAPI:
    """Synchronous Matches API for agent matches."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def list_recent(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List recent matches.

        Args:
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            Recent matches with outcomes.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return self._client.request("GET", "/api/v1/matches/recent", params=params)


class AsyncMatchesAPI:
    """Asynchronous Matches API for agent matches."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list_recent(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List recent matches.

        Args:
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            Recent matches with outcomes.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return await self._client.request("GET", "/api/v1/matches/recent", params=params)
