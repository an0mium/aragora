"""
Tournaments API for the Aragora Python SDK.

Provides access to agent tournament management.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.client.client import AragoraClient


@dataclass
class TournamentStanding:
    """An agent's standing in a tournament."""

    agent_id: str
    rank: int
    wins: int
    losses: int
    draws: int
    points: float
    elo_change: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TournamentStanding":
        return cls(
            agent_id=data.get("agent_id", ""),
            rank=data.get("rank", 0),
            wins=data.get("wins", 0),
            losses=data.get("losses", 0),
            draws=data.get("draws", 0),
            points=data.get("points", 0.0),
            elo_change=data.get("elo_change", 0.0),
        )


@dataclass
class TournamentSummary:
    """Summary of a tournament."""

    id: str
    name: str
    status: str
    participants: int
    rounds_completed: int
    total_rounds: int
    created_at: str
    completed_at: Optional[str] = None
    winner: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TournamentSummary":
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            status=data.get("status", "unknown"),
            participants=data.get("participants", 0),
            rounds_completed=data.get("rounds_completed", 0),
            total_rounds=data.get("total_rounds", 0),
            created_at=data.get("created_at", ""),
            completed_at=data.get("completed_at"),
            winner=data.get("winner"),
        )


@dataclass
class Tournament:
    """Full tournament details."""

    id: str
    name: str
    status: str
    format: str
    participants: List[str]
    standings: List[TournamentStanding]
    rounds_completed: int
    total_rounds: int
    created_at: str
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tournament":
        standings_data = data.get("standings", [])
        standings = [TournamentStanding.from_dict(s) for s in standings_data]

        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            status=data.get("status", "unknown"),
            format=data.get("format", "round_robin"),
            participants=data.get("participants", []),
            standings=standings,
            rounds_completed=data.get("rounds_completed", 0),
            total_rounds=data.get("total_rounds", 0),
            created_at=data.get("created_at", ""),
            completed_at=data.get("completed_at"),
            metadata=data.get("metadata", {}),
        )


class TournamentsAPI:
    """
    API interface for agent tournaments.

    Provides access to tournament creation, management, and results.

    Example:
        # List recent tournaments
        tournaments = client.tournaments.list()

        # Get tournament standings
        standings = client.tournaments.get_standings("tournament_123")
        for standing in standings:
            print(f"{standing.agent_id}: {standing.points} points")

        # Create a new tournament
        tournament = client.tournaments.create(
            name="Weekly Championship",
            agents=["claude", "gpt4", "gemini"],
            format="round_robin",
        )
    """

    def __init__(self, client: "AragoraClient"):
        self._client = client

    def list(
        self,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[TournamentSummary]:
        """
        List tournaments.

        Args:
            status: Filter by status (pending, running, completed, cancelled)
            limit: Maximum results to return
            offset: Offset for pagination

        Returns:
            List of TournamentSummary objects
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        response = self._client._get("/api/tournaments", params=params)
        tournaments = response.get("tournaments", [])
        return [TournamentSummary.from_dict(t) for t in tournaments]

    async def list_async(
        self,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[TournamentSummary]:
        """Async version of list."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        response = await self._client._get_async("/api/tournaments", params=params)
        tournaments = response.get("tournaments", [])
        return [TournamentSummary.from_dict(t) for t in tournaments]

    def get(self, tournament_id: str) -> Tournament:
        """
        Get tournament details.

        Args:
            tournament_id: The tournament ID

        Returns:
            Tournament object with full details
        """
        response = self._client._get(f"/api/tournaments/{tournament_id}")
        return Tournament.from_dict(response)

    async def get_async(self, tournament_id: str) -> Tournament:
        """Async version of get."""
        response = await self._client._get_async(f"/api/tournaments/{tournament_id}")
        return Tournament.from_dict(response)

    def get_standings(self, tournament_id: str) -> List[TournamentStanding]:
        """
        Get tournament standings.

        Args:
            tournament_id: The tournament ID

        Returns:
            List of TournamentStanding objects sorted by rank
        """
        response = self._client._get(f"/api/tournaments/{tournament_id}/standings")
        standings = response.get("standings", [])
        return [TournamentStanding.from_dict(s) for s in standings]

    async def get_standings_async(self, tournament_id: str) -> List[TournamentStanding]:
        """Async version of get_standings."""
        response = await self._client._get_async(f"/api/tournaments/{tournament_id}/standings")
        standings = response.get("standings", [])
        return [TournamentStanding.from_dict(s) for s in standings]

    def create(
        self,
        name: str,
        agents: List[str],
        format: str = "round_robin",
        topic: Optional[str] = None,
        rounds_per_match: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TournamentSummary:
        """
        Create a new tournament.

        Args:
            name: Tournament name
            agents: List of agent IDs to participate
            format: Tournament format (round_robin, elimination, swiss)
            topic: Optional topic/domain for debates
            rounds_per_match: Rounds per debate match
            metadata: Optional metadata

        Returns:
            TournamentSummary for the created tournament
        """
        data = {
            "name": name,
            "agents": agents,
            "format": format,
            "rounds_per_match": rounds_per_match,
        }
        if topic:
            data["topic"] = topic
        if metadata:
            data["metadata"] = metadata

        response = self._client._post("/api/tournaments", data)
        return TournamentSummary.from_dict(response)

    async def create_async(
        self,
        name: str,
        agents: List[str],
        format: str = "round_robin",
        topic: Optional[str] = None,
        rounds_per_match: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TournamentSummary:
        """Async version of create."""
        data = {
            "name": name,
            "agents": agents,
            "format": format,
            "rounds_per_match": rounds_per_match,
        }
        if topic:
            data["topic"] = topic
        if metadata:
            data["metadata"] = metadata

        response = await self._client._post_async("/api/tournaments", data)
        return TournamentSummary.from_dict(response)

    def cancel(self, tournament_id: str) -> bool:
        """
        Cancel a running tournament.

        Args:
            tournament_id: The tournament ID

        Returns:
            True if cancelled successfully
        """
        response = self._client._post(f"/api/tournaments/{tournament_id}/cancel", {})
        return response.get("cancelled", False)

    async def cancel_async(self, tournament_id: str) -> bool:
        """Async version of cancel."""
        response = await self._client._post_async(f"/api/tournaments/{tournament_id}/cancel", {})
        return response.get("cancelled", False)
