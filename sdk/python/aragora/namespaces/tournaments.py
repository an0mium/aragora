"""
Tournaments Namespace API

Provides methods for managing agent tournaments including brackets,
matches, standings, and results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

TournamentFormat = Literal["single_elimination", "double_elimination", "round_robin", "swiss"]
TournamentStatus = Literal["pending", "active", "completed", "cancelled"]


class TournamentsAPI:
    """
    Synchronous Tournaments API.

    Provides methods for managing tournaments:
    - Creating and listing tournaments
    - Viewing standings and brackets
    - Submitting match results
    - Advancing tournament rounds

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> tournament = client.tournaments.create(
        ...     name="Weekly Championship",
        ...     format="single_elimination",
        ...     participants=["claude", "gpt-4", "gemini", "grok"]
        ... )
        >>> standings = client.tournaments.get_standings(tournament["id"])
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def list(
        self,
        status: TournamentStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List tournaments with optional filtering.

        Args:
            status: Filter by status
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of tournaments with pagination
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        return self._client.request("GET", "/api/v1/tournaments", params=params)

    def get(self, tournament_id: str) -> dict[str, Any]:
        """
        Get a tournament by ID.

        Args:
            tournament_id: Tournament identifier

        Returns:
            Tournament details
        """
        return self._client.request("GET", f"/api/v1/tournaments/{tournament_id}")

    def create(
        self,
        name: str,
        format: TournamentFormat = "single_elimination",
        participants: list[str] | None = None,
        description: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Create a new tournament.

        Args:
            name: Tournament name
            format: Tournament format
            participants: List of participant agent names
            description: Tournament description
            **kwargs: Additional tournament options

        Returns:
            Created tournament
        """
        data: dict[str, Any] = {"name": name, "format": format, **kwargs}
        if participants:
            data["participants"] = participants
        if description:
            data["description"] = description

        return self._client.request("POST", "/api/v1/tournaments", json=data)

    def get_standings(self, tournament_id: str) -> dict[str, Any]:
        """
        Get tournament standings.

        Args:
            tournament_id: Tournament identifier

        Returns:
            Tournament standings with rankings
        """
        return self._client.request("GET", f"/api/v1/tournaments/{tournament_id}/standings")

    def get_bracket(self, tournament_id: str) -> dict[str, Any]:
        """
        Get tournament bracket.

        Args:
            tournament_id: Tournament identifier

        Returns:
            Tournament bracket structure
        """
        return self._client.request("GET", f"/api/v1/tournaments/{tournament_id}/bracket")

    def list_matches(
        self,
        tournament_id: str,
        round: int | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        """
        List matches in a tournament.

        Args:
            tournament_id: Tournament identifier
            round: Filter by round number
            status: Filter by match status

        Returns:
            List of matches
        """
        params: dict[str, Any] = {}
        if round is not None:
            params["round"] = round
        if status:
            params["status"] = status

        return self._client.request(
            "GET", f"/api/v1/tournaments/{tournament_id}/matches", params=params
        )

    def get_match(self, tournament_id: str, match_id: str) -> dict[str, Any]:
        """
        Get a specific match.

        Args:
            tournament_id: Tournament identifier
            match_id: Match identifier

        Returns:
            Match details
        """
        return self._client.request(
            "GET", f"/api/v1/tournaments/{tournament_id}/matches/{match_id}"
        )

    def submit_result(
        self,
        tournament_id: str,
        match_id: str,
        winner: str,
        loser: str,
        score: dict[str, int] | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """
        Submit a match result.

        Args:
            tournament_id: Tournament identifier
            match_id: Match identifier
            winner: Winner agent name
            loser: Loser agent name
            score: Optional match score {"winner": N, "loser": M}
            notes: Optional match notes

        Returns:
            Updated match with result
        """
        data: dict[str, Any] = {"winner": winner, "loser": loser}
        if score:
            data["score"] = score
        if notes:
            data["notes"] = notes

        return self._client.request(
            "POST",
            f"/api/v1/tournaments/{tournament_id}/matches/{match_id}/result",
            json=data,
        )

    def advance(self, tournament_id: str) -> dict[str, Any]:
        """
        Advance the tournament to the next round.

        Args:
            tournament_id: Tournament identifier

        Returns:
            Advancement result with next round info
        """
        return self._client.request("POST", f"/api/v1/tournaments/{tournament_id}/advance")

    def start(self, tournament_id: str) -> dict[str, Any]:
        """
        Start a pending tournament.

        Args:
            tournament_id: Tournament identifier

        Returns:
            Updated tournament status
        """
        return self._client.request("POST", f"/api/v1/tournaments/{tournament_id}/start")

    def cancel(self, tournament_id: str, reason: str | None = None) -> dict[str, Any]:
        """
        Cancel a tournament.

        Args:
            tournament_id: Tournament identifier
            reason: Cancellation reason

        Returns:
            Cancellation confirmation
        """
        data: dict[str, Any] = {}
        if reason:
            data["reason"] = reason

        return self._client.request(
            "POST", f"/api/v1/tournaments/{tournament_id}/cancel", json=data
        )


class AsyncTournamentsAPI:
    """
    Asynchronous Tournaments API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     tournaments = await client.tournaments.list(status="active")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list(
        self,
        status: TournamentStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List tournaments with optional filtering."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        return await self._client.request("GET", "/api/v1/tournaments", params=params)

    async def get(self, tournament_id: str) -> dict[str, Any]:
        """Get a tournament by ID."""
        return await self._client.request("GET", f"/api/v1/tournaments/{tournament_id}")

    async def create(
        self,
        name: str,
        format: TournamentFormat = "single_elimination",
        participants: list[str] | None = None,
        description: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Create a new tournament."""
        data: dict[str, Any] = {"name": name, "format": format, **kwargs}
        if participants:
            data["participants"] = participants
        if description:
            data["description"] = description

        return await self._client.request("POST", "/api/v1/tournaments", json=data)

    async def get_standings(self, tournament_id: str) -> dict[str, Any]:
        """Get tournament standings."""
        return await self._client.request("GET", f"/api/v1/tournaments/{tournament_id}/standings")

    async def get_bracket(self, tournament_id: str) -> dict[str, Any]:
        """Get tournament bracket."""
        return await self._client.request("GET", f"/api/v1/tournaments/{tournament_id}/bracket")

    async def list_matches(
        self,
        tournament_id: str,
        round: int | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        """List matches in a tournament."""
        params: dict[str, Any] = {}
        if round is not None:
            params["round"] = round
        if status:
            params["status"] = status

        return await self._client.request(
            "GET", f"/api/v1/tournaments/{tournament_id}/matches", params=params
        )

    async def get_match(self, tournament_id: str, match_id: str) -> dict[str, Any]:
        """Get a specific match."""
        return await self._client.request(
            "GET", f"/api/v1/tournaments/{tournament_id}/matches/{match_id}"
        )

    async def submit_result(
        self,
        tournament_id: str,
        match_id: str,
        winner: str,
        loser: str,
        score: dict[str, int] | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Submit a match result."""
        data: dict[str, Any] = {"winner": winner, "loser": loser}
        if score:
            data["score"] = score
        if notes:
            data["notes"] = notes

        return await self._client.request(
            "POST",
            f"/api/v1/tournaments/{tournament_id}/matches/{match_id}/result",
            json=data,
        )

    async def advance(self, tournament_id: str) -> dict[str, Any]:
        """Advance the tournament to the next round."""
        return await self._client.request("POST", f"/api/v1/tournaments/{tournament_id}/advance")

    async def start(self, tournament_id: str) -> dict[str, Any]:
        """Start a pending tournament."""
        return await self._client.request("POST", f"/api/v1/tournaments/{tournament_id}/start")

    async def cancel(self, tournament_id: str, reason: str | None = None) -> dict[str, Any]:
        """Cancel a tournament."""
        data: dict[str, Any] = {}
        if reason:
            data["reason"] = reason

        return await self._client.request(
            "POST", f"/api/v1/tournaments/{tournament_id}/cancel", json=data
        )
