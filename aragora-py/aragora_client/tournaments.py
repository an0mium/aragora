"""Tournaments API for the Aragora SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from aragora_client.client import AragoraClient


class Tournament(BaseModel):
    """Tournament model."""

    id: str
    name: str
    description: str | None = None
    format: str = "single_elimination"
    status: str = "pending"
    current_round: int = 0
    total_rounds: int | None = None
    participants: list[str] = []
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


class CreateTournamentRequest(BaseModel):
    """Create tournament request."""

    name: str
    description: str | None = None
    format: str = "single_elimination"
    participants: list[str] = []
    task_template: str | None = None
    rounds_config: dict[str, Any] | None = None


class TournamentMatch(BaseModel):
    """Tournament match."""

    id: str
    tournament_id: str
    round: int
    participants: list[str]
    winner: str | None = None
    debate_id: str | None = None
    status: str = "pending"
    scheduled_at: str | None = None
    completed_at: str | None = None


class TournamentStanding(BaseModel):
    """Tournament standing entry."""

    participant: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    points: float = 0.0
    rank: int | None = None


class TournamentStandings(BaseModel):
    """Tournament standings."""

    tournament_id: str
    standings: list[TournamentStanding]
    updated_at: str | None = None


class TournamentBracket(BaseModel):
    """Tournament bracket."""

    tournament_id: str
    rounds: list[list[dict[str, Any]]]
    current_round: int
    champion: str | None = None


class MatchResult(BaseModel):
    """Match result submission."""

    winner: str
    debate_id: str
    notes: str | None = None


class TournamentsAPI:
    """API for tournament operations."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    async def list(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Tournament]:
        """List tournaments."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        data = await self._client._get("/api/tournaments", params=params)
        return [Tournament.model_validate(t) for t in data.get("tournaments", [])]

    async def get(self, tournament_id: str) -> Tournament:
        """Get a tournament by ID."""
        data = await self._client._get(f"/api/tournaments/{tournament_id}")
        return Tournament.model_validate(data)

    async def create(
        self,
        name: str,
        participants: list[str],
        *,
        description: str | None = None,
        format: str = "single_elimination",
        task_template: str | None = None,
        rounds_config: dict[str, Any] | None = None,
    ) -> Tournament:
        """Create a new tournament."""
        request = CreateTournamentRequest(
            name=name,
            description=description,
            format=format,
            participants=participants,
            task_template=task_template,
            rounds_config=rounds_config,
        )
        data = await self._client._post("/api/tournaments", request.model_dump())
        return Tournament.model_validate(data)

    async def start(self, tournament_id: str) -> Tournament:
        """Start a tournament."""
        data = await self._client._post(f"/api/tournaments/{tournament_id}/start", {})
        return Tournament.model_validate(data)

    async def cancel(self, tournament_id: str) -> Tournament:
        """Cancel a tournament."""
        data = await self._client._post(f"/api/tournaments/{tournament_id}/cancel", {})
        return Tournament.model_validate(data)

    async def get_standings(self, tournament_id: str) -> TournamentStandings:
        """Get tournament standings."""
        data = await self._client._get(f"/api/tournaments/{tournament_id}/standings")
        return TournamentStandings.model_validate(data)

    async def get_bracket(self, tournament_id: str) -> TournamentBracket:
        """Get tournament bracket."""
        data = await self._client._get(f"/api/tournaments/{tournament_id}/bracket")
        return TournamentBracket.model_validate(data)

    async def list_matches(
        self,
        tournament_id: str,
        *,
        round: int | None = None,
        status: str | None = None,
    ) -> list[TournamentMatch]:
        """List tournament matches."""
        params: dict[str, Any] = {}
        if round is not None:
            params["round"] = round
        if status:
            params["status"] = status
        data = await self._client._get(
            f"/api/tournaments/{tournament_id}/matches", params=params
        )
        return [TournamentMatch.model_validate(m) for m in data.get("matches", [])]

    async def get_match(self, tournament_id: str, match_id: str) -> TournamentMatch:
        """Get a specific match."""
        data = await self._client._get(
            f"/api/tournaments/{tournament_id}/matches/{match_id}"
        )
        return TournamentMatch.model_validate(data)

    async def submit_match_result(
        self,
        tournament_id: str,
        match_id: str,
        winner: str,
        debate_id: str,
        *,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Submit match result."""
        result = MatchResult(winner=winner, debate_id=debate_id, notes=notes)
        return await self._client._post(
            f"/api/tournaments/{tournament_id}/matches/{match_id}/result",
            result.model_dump(),
        )

    async def advance(self, tournament_id: str) -> dict[str, Any]:
        """Advance tournament to next round."""
        return await self._client._post(f"/api/tournaments/{tournament_id}/advance", {})

    async def delete(self, tournament_id: str) -> None:
        """Delete a tournament."""
        await self._client._delete(f"/api/tournaments/{tournament_id}")
