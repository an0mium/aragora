"""LeaderboardAPI resource for the Aragora client."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import LeaderboardEntry

if TYPE_CHECKING:
    from ..client import AragoraClient


class LeaderboardAPI:
    """API interface for leaderboard."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    def get(self, limit: int = 10) -> list[LeaderboardEntry]:
        """
        Get leaderboard rankings.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of LeaderboardEntry objects.
        """
        response = self._client._get("/api/leaderboard", params={"limit": limit})
        rankings = response.get("rankings", response) if isinstance(response, dict) else response
        return [LeaderboardEntry(**r) for r in rankings]

    async def get_async(self, limit: int = 10) -> list[LeaderboardEntry]:
        """Async version of get()."""
        response = await self._client._get_async("/api/leaderboard", params={"limit": limit})
        rankings = response.get("rankings", response) if isinstance(response, dict) else response
        return [LeaderboardEntry(**r) for r in rankings]
