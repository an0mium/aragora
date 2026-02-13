"""Tests for TournamentsAPI client resource."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.client.client import AragoraClient
from aragora.client.resources.tournaments import (
    Tournament,
    TournamentStanding,
    TournamentSummary,
    TournamentsAPI,
)


@pytest.fixture
def mock_client() -> AragoraClient:
    client = MagicMock(spec=AragoraClient)
    return client


@pytest.fixture
def api(mock_client: AragoraClient) -> TournamentsAPI:
    return TournamentsAPI(mock_client)


SAMPLE_STANDING = {
    "agent_id": "claude",
    "rank": 1,
    "wins": 5,
    "losses": 1,
    "draws": 0,
    "points": 15.0,
    "elo_change": 32.5,
}

SAMPLE_STANDING_2 = {
    "agent_id": "gpt4",
    "rank": 2,
    "wins": 3,
    "losses": 2,
    "draws": 1,
    "points": 10.0,
    "elo_change": -12.0,
}

SAMPLE_SUMMARY = {
    "id": "tourn-001",
    "name": "Weekly Championship",
    "status": "completed",
    "participants": 4,
    "rounds_completed": 6,
    "total_rounds": 6,
    "created_at": "2026-02-01T10:00:00Z",
    "completed_at": "2026-02-01T12:00:00Z",
    "winner": "claude",
}

SAMPLE_SUMMARY_2 = {
    "id": "tourn-002",
    "name": "Daily Sprint",
    "status": "running",
    "participants": 3,
    "rounds_completed": 2,
    "total_rounds": 3,
    "created_at": "2026-02-10T08:00:00Z",
}

SAMPLE_TOURNAMENT = {
    "id": "tourn-001",
    "name": "Weekly Championship",
    "status": "completed",
    "format": "round_robin",
    "participants": ["claude", "gpt4", "gemini", "mistral"],
    "standings": [SAMPLE_STANDING, SAMPLE_STANDING_2],
    "rounds_completed": 6,
    "total_rounds": 6,
    "created_at": "2026-02-01T10:00:00Z",
    "completed_at": "2026-02-01T12:00:00Z",
    "metadata": {"topic": "code review", "season": 3},
}


# ── Dataclass construction and defaults ──────────────────────────────


class TestTournamentStandingDataclass:
    def test_construction(self) -> None:
        standing = TournamentStanding(
            agent_id="claude",
            rank=1,
            wins=5,
            losses=1,
            draws=0,
            points=15.0,
            elo_change=32.5,
        )
        assert standing.agent_id == "claude"
        assert standing.rank == 1
        assert standing.wins == 5
        assert standing.losses == 1
        assert standing.draws == 0
        assert standing.points == 15.0
        assert standing.elo_change == 32.5

    def test_default_elo_change(self) -> None:
        standing = TournamentStanding(
            agent_id="gpt4", rank=2, wins=3, losses=2, draws=1, points=10.0
        )
        assert standing.elo_change == 0.0

    def test_from_dict(self) -> None:
        standing = TournamentStanding.from_dict(SAMPLE_STANDING)
        assert standing.agent_id == "claude"
        assert standing.rank == 1
        assert standing.wins == 5
        assert standing.losses == 1
        assert standing.draws == 0
        assert standing.points == 15.0
        assert standing.elo_change == 32.5

    def test_from_dict_empty(self) -> None:
        standing = TournamentStanding.from_dict({})
        assert standing.agent_id == ""
        assert standing.rank == 0
        assert standing.wins == 0
        assert standing.losses == 0
        assert standing.draws == 0
        assert standing.points == 0.0
        assert standing.elo_change == 0.0

    def test_from_dict_partial(self) -> None:
        standing = TournamentStanding.from_dict({"agent_id": "gemini", "wins": 7})
        assert standing.agent_id == "gemini"
        assert standing.wins == 7
        assert standing.rank == 0
        assert standing.losses == 0
        assert standing.draws == 0
        assert standing.points == 0.0
        assert standing.elo_change == 0.0


class TestTournamentSummaryDataclass:
    def test_construction(self) -> None:
        summary = TournamentSummary(
            id="tourn-001",
            name="Weekly Championship",
            status="completed",
            participants=4,
            rounds_completed=6,
            total_rounds=6,
            created_at="2026-02-01T10:00:00Z",
            completed_at="2026-02-01T12:00:00Z",
            winner="claude",
        )
        assert summary.id == "tourn-001"
        assert summary.name == "Weekly Championship"
        assert summary.status == "completed"
        assert summary.participants == 4
        assert summary.rounds_completed == 6
        assert summary.total_rounds == 6
        assert summary.completed_at == "2026-02-01T12:00:00Z"
        assert summary.winner == "claude"

    def test_optional_defaults(self) -> None:
        summary = TournamentSummary(
            id="tourn-003",
            name="Pending",
            status="pending",
            participants=2,
            rounds_completed=0,
            total_rounds=1,
            created_at="2026-02-12T00:00:00Z",
        )
        assert summary.completed_at is None
        assert summary.winner is None

    def test_from_dict(self) -> None:
        summary = TournamentSummary.from_dict(SAMPLE_SUMMARY)
        assert summary.id == "tourn-001"
        assert summary.name == "Weekly Championship"
        assert summary.status == "completed"
        assert summary.participants == 4
        assert summary.rounds_completed == 6
        assert summary.total_rounds == 6
        assert summary.created_at == "2026-02-01T10:00:00Z"
        assert summary.completed_at == "2026-02-01T12:00:00Z"
        assert summary.winner == "claude"

    def test_from_dict_empty(self) -> None:
        summary = TournamentSummary.from_dict({})
        assert summary.id == ""
        assert summary.name == ""
        assert summary.status == "unknown"
        assert summary.participants == 0
        assert summary.rounds_completed == 0
        assert summary.total_rounds == 0
        assert summary.created_at == ""
        assert summary.completed_at is None
        assert summary.winner is None

    def test_from_dict_no_winner(self) -> None:
        summary = TournamentSummary.from_dict(SAMPLE_SUMMARY_2)
        assert summary.id == "tourn-002"
        assert summary.status == "running"
        assert summary.completed_at is None
        assert summary.winner is None


class TestTournamentDataclass:
    def test_construction(self) -> None:
        standings = [TournamentStanding.from_dict(SAMPLE_STANDING)]
        tournament = Tournament(
            id="tourn-001",
            name="Championship",
            status="completed",
            format="round_robin",
            participants=["claude", "gpt4"],
            standings=standings,
            rounds_completed=3,
            total_rounds=3,
            created_at="2026-02-01T10:00:00Z",
            completed_at="2026-02-01T11:00:00Z",
            metadata={"topic": "testing"},
        )
        assert tournament.id == "tourn-001"
        assert tournament.format == "round_robin"
        assert len(tournament.participants) == 2
        assert len(tournament.standings) == 1
        assert tournament.metadata == {"topic": "testing"}
        assert tournament.completed_at == "2026-02-01T11:00:00Z"

    def test_metadata_default_none_becomes_dict(self) -> None:
        tournament = Tournament(
            id="t",
            name="n",
            status="s",
            format="f",
            participants=[],
            standings=[],
            rounds_completed=0,
            total_rounds=0,
            created_at="",
        )
        assert tournament.metadata == {}

    def test_metadata_explicit_none_becomes_dict(self) -> None:
        tournament = Tournament(
            id="t",
            name="n",
            status="s",
            format="f",
            participants=[],
            standings=[],
            rounds_completed=0,
            total_rounds=0,
            created_at="",
            metadata=None,
        )
        assert tournament.metadata == {}

    def test_metadata_provided_value_preserved(self) -> None:
        tournament = Tournament(
            id="t",
            name="n",
            status="s",
            format="f",
            participants=[],
            standings=[],
            rounds_completed=0,
            total_rounds=0,
            created_at="",
            metadata={"key": "val"},
        )
        assert tournament.metadata == {"key": "val"}

    def test_from_dict(self) -> None:
        tournament = Tournament.from_dict(SAMPLE_TOURNAMENT)
        assert tournament.id == "tourn-001"
        assert tournament.name == "Weekly Championship"
        assert tournament.status == "completed"
        assert tournament.format == "round_robin"
        assert tournament.participants == ["claude", "gpt4", "gemini", "mistral"]
        assert len(tournament.standings) == 2
        assert tournament.standings[0].agent_id == "claude"
        assert tournament.standings[1].agent_id == "gpt4"
        assert tournament.rounds_completed == 6
        assert tournament.total_rounds == 6
        assert tournament.completed_at == "2026-02-01T12:00:00Z"
        assert tournament.metadata == {"topic": "code review", "season": 3}

    def test_from_dict_empty(self) -> None:
        tournament = Tournament.from_dict({})
        assert tournament.id == ""
        assert tournament.name == ""
        assert tournament.status == "unknown"
        assert tournament.format == "round_robin"
        assert tournament.participants == []
        assert tournament.standings == []
        assert tournament.rounds_completed == 0
        assert tournament.total_rounds == 0
        assert tournament.created_at == ""
        assert tournament.completed_at is None
        assert tournament.metadata == {}

    def test_from_dict_no_standings(self) -> None:
        data = {**SAMPLE_TOURNAMENT, "standings": []}
        tournament = Tournament.from_dict(data)
        assert tournament.standings == []

    def test_from_dict_missing_standings_key(self) -> None:
        data = {k: v for k, v in SAMPLE_TOURNAMENT.items() if k != "standings"}
        tournament = Tournament.from_dict(data)
        assert tournament.standings == []


# ── TournamentsAPI.list ──────────────────────────────────────────────


class TestTournamentsList:
    def test_list_default(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"tournaments": [SAMPLE_SUMMARY]}
        results = api.list()
        assert len(results) == 1
        assert isinstance(results[0], TournamentSummary)
        assert results[0].id == "tourn-001"
        mock_client._get.assert_called_once_with(
            "/api/tournaments", params={"limit": 20, "offset": 0}
        )

    def test_list_with_status_filter(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"tournaments": [SAMPLE_SUMMARY]}
        api.list(status="completed")
        params = mock_client._get.call_args[1]["params"]
        assert params["status"] == "completed"

    def test_list_with_pagination(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"tournaments": []}
        api.list(limit=5, offset=10)
        params = mock_client._get.call_args[1]["params"]
        assert params["limit"] == 5
        assert params["offset"] == 10

    def test_list_no_status_param_omitted(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"tournaments": []}
        api.list()
        params = mock_client._get.call_args[1]["params"]
        assert "status" not in params

    def test_list_multiple_results(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"tournaments": [SAMPLE_SUMMARY, SAMPLE_SUMMARY_2]}
        results = api.list()
        assert len(results) == 2
        assert results[0].id == "tourn-001"
        assert results[1].id == "tourn-002"

    def test_list_empty_response(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"tournaments": []}
        results = api.list()
        assert results == []

    def test_list_missing_tournaments_key(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {}
        results = api.list()
        assert results == []

    @pytest.mark.asyncio
    async def test_list_async(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(
            return_value={"tournaments": [SAMPLE_SUMMARY, SAMPLE_SUMMARY_2]}
        )
        results = await api.list_async()
        assert len(results) == 2
        assert results[0].name == "Weekly Championship"
        assert results[1].name == "Daily Sprint"

    @pytest.mark.asyncio
    async def test_list_async_with_status(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={"tournaments": []})
        await api.list_async(status="running")
        params = mock_client._get_async.call_args[1]["params"]
        assert params["status"] == "running"

    @pytest.mark.asyncio
    async def test_list_async_empty(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={"tournaments": []})
        results = await api.list_async()
        assert results == []

    @pytest.mark.asyncio
    async def test_list_async_missing_key(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={})
        results = await api.list_async()
        assert results == []


# ── TournamentsAPI.get ───────────────────────────────────────────────


class TestTournamentsGet:
    def test_get(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_TOURNAMENT
        result = api.get("tourn-001")
        assert isinstance(result, Tournament)
        assert result.id == "tourn-001"
        assert result.name == "Weekly Championship"
        assert result.format == "round_robin"
        assert len(result.standings) == 2
        mock_client._get.assert_called_once_with("/api/tournaments/tourn-001")

    def test_get_standings_parsed(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_TOURNAMENT
        result = api.get("tourn-001")
        assert result.standings[0].agent_id == "claude"
        assert result.standings[0].rank == 1
        assert result.standings[0].elo_change == 32.5
        assert result.standings[1].agent_id == "gpt4"
        assert result.standings[1].elo_change == -12.0

    def test_get_metadata(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_TOURNAMENT
        result = api.get("tourn-001")
        assert result.metadata["topic"] == "code review"
        assert result.metadata["season"] == 3

    def test_get_empty_response(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {}
        result = api.get("tourn-missing")
        assert result.id == ""
        assert result.status == "unknown"
        assert result.standings == []

    @pytest.mark.asyncio
    async def test_get_async(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_TOURNAMENT)
        result = await api.get_async("tourn-001")
        assert isinstance(result, Tournament)
        assert result.id == "tourn-001"
        assert result.name == "Weekly Championship"
        mock_client._get_async.assert_called_once_with("/api/tournaments/tourn-001")

    @pytest.mark.asyncio
    async def test_get_async_empty(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={})
        result = await api.get_async("tourn-missing")
        assert result.id == ""
        assert result.standings == []


# ── TournamentsAPI.get_standings ─────────────────────────────────────


class TestTournamentsGetStandings:
    def test_get_standings(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"standings": [SAMPLE_STANDING, SAMPLE_STANDING_2]}
        standings = api.get_standings("tourn-001")
        assert len(standings) == 2
        assert all(isinstance(s, TournamentStanding) for s in standings)
        assert standings[0].agent_id == "claude"
        assert standings[1].agent_id == "gpt4"
        mock_client._get.assert_called_once_with("/api/tournaments/tourn-001/standings")

    def test_get_standings_single(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"standings": [SAMPLE_STANDING]}
        standings = api.get_standings("tourn-001")
        assert len(standings) == 1
        assert standings[0].points == 15.0

    def test_get_standings_empty(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"standings": []}
        standings = api.get_standings("tourn-001")
        assert standings == []

    def test_get_standings_missing_key(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {}
        standings = api.get_standings("tourn-001")
        assert standings == []

    @pytest.mark.asyncio
    async def test_get_standings_async(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(
            return_value={"standings": [SAMPLE_STANDING, SAMPLE_STANDING_2]}
        )
        standings = await api.get_standings_async("tourn-001")
        assert len(standings) == 2
        assert standings[0].agent_id == "claude"
        mock_client._get_async.assert_called_once_with("/api/tournaments/tourn-001/standings")

    @pytest.mark.asyncio
    async def test_get_standings_async_empty(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={"standings": []})
        standings = await api.get_standings_async("tourn-001")
        assert standings == []

    @pytest.mark.asyncio
    async def test_get_standings_async_missing_key(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={})
        standings = await api.get_standings_async("tourn-001")
        assert standings == []


# ── TournamentsAPI.create ────────────────────────────────────────────


class TestTournamentsCreate:
    def test_create_minimal(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_SUMMARY
        result = api.create("Weekly Championship", agents=["claude", "gpt4"])
        assert isinstance(result, TournamentSummary)
        assert result.id == "tourn-001"
        mock_client._post.assert_called_once()
        body = mock_client._post.call_args[0][1]
        assert body["name"] == "Weekly Championship"
        assert body["agents"] == ["claude", "gpt4"]
        assert body["format"] == "round_robin"
        assert body["rounds_per_match"] == 3

    def test_create_with_format(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_SUMMARY
        api.create("Elimination", agents=["a", "b"], format="elimination")
        body = mock_client._post.call_args[0][1]
        assert body["format"] == "elimination"

    def test_create_with_topic(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_SUMMARY
        api.create("Topical", agents=["a", "b"], topic="machine learning")
        body = mock_client._post.call_args[0][1]
        assert body["topic"] == "machine learning"

    def test_create_topic_omitted_when_none(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_SUMMARY
        api.create("No Topic", agents=["a"])
        body = mock_client._post.call_args[0][1]
        assert "topic" not in body

    def test_create_with_rounds_per_match(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_SUMMARY
        api.create("Custom Rounds", agents=["a", "b"], rounds_per_match=5)
        body = mock_client._post.call_args[0][1]
        assert body["rounds_per_match"] == 5

    def test_create_with_metadata(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_SUMMARY
        api.create("Meta", agents=["a"], metadata={"season": 3, "league": "pro"})
        body = mock_client._post.call_args[0][1]
        assert body["metadata"] == {"season": 3, "league": "pro"}

    def test_create_metadata_omitted_when_none(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_SUMMARY
        api.create("No Meta", agents=["a"])
        body = mock_client._post.call_args[0][1]
        assert "metadata" not in body

    def test_create_all_options(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_SUMMARY
        api.create(
            name="Full Options",
            agents=["claude", "gpt4", "gemini"],
            format="swiss",
            topic="code review",
            rounds_per_match=7,
            metadata={"prize": "badge"},
        )
        body = mock_client._post.call_args[0][1]
        assert body["name"] == "Full Options"
        assert body["agents"] == ["claude", "gpt4", "gemini"]
        assert body["format"] == "swiss"
        assert body["topic"] == "code review"
        assert body["rounds_per_match"] == 7
        assert body["metadata"] == {"prize": "badge"}

    def test_create_endpoint(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_SUMMARY
        api.create("T", agents=["a"])
        assert mock_client._post.call_args[0][0] == "/api/tournaments"

    @pytest.mark.asyncio
    async def test_create_async(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._post_async = AsyncMock(return_value=SAMPLE_SUMMARY)
        result = await api.create_async("Async Tourn", agents=["claude", "gpt4"])
        assert isinstance(result, TournamentSummary)
        assert result.id == "tourn-001"
        body = mock_client._post_async.call_args[0][1]
        assert body["name"] == "Async Tourn"
        assert body["agents"] == ["claude", "gpt4"]

    @pytest.mark.asyncio
    async def test_create_async_with_all_options(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._post_async = AsyncMock(return_value=SAMPLE_SUMMARY)
        await api.create_async(
            name="Full Async",
            agents=["a", "b"],
            format="elimination",
            topic="security",
            rounds_per_match=10,
            metadata={"env": "prod"},
        )
        body = mock_client._post_async.call_args[0][1]
        assert body["format"] == "elimination"
        assert body["topic"] == "security"
        assert body["rounds_per_match"] == 10
        assert body["metadata"] == {"env": "prod"}

    @pytest.mark.asyncio
    async def test_create_async_topic_omitted(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._post_async = AsyncMock(return_value=SAMPLE_SUMMARY)
        await api.create_async("No Topic", agents=["a"])
        body = mock_client._post_async.call_args[0][1]
        assert "topic" not in body

    @pytest.mark.asyncio
    async def test_create_async_metadata_omitted(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._post_async = AsyncMock(return_value=SAMPLE_SUMMARY)
        await api.create_async("No Meta", agents=["a"])
        body = mock_client._post_async.call_args[0][1]
        assert "metadata" not in body


# ── TournamentsAPI.cancel ────────────────────────────────────────────


class TestTournamentsCancel:
    def test_cancel_success(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"cancelled": True}
        result = api.cancel("tourn-001")
        assert result is True
        mock_client._post.assert_called_once_with("/api/tournaments/tourn-001/cancel", {})

    def test_cancel_failure(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"cancelled": False}
        result = api.cancel("tourn-001")
        assert result is False

    def test_cancel_missing_key(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {}
        result = api.cancel("tourn-001")
        assert result is False

    def test_cancel_uses_correct_path(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"cancelled": True}
        api.cancel("my-special-tourn")
        mock_client._post.assert_called_once_with("/api/tournaments/my-special-tourn/cancel", {})

    @pytest.mark.asyncio
    async def test_cancel_async_success(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._post_async = AsyncMock(return_value={"cancelled": True})
        result = await api.cancel_async("tourn-001")
        assert result is True
        mock_client._post_async.assert_called_once_with("/api/tournaments/tourn-001/cancel", {})

    @pytest.mark.asyncio
    async def test_cancel_async_failure(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._post_async = AsyncMock(return_value={"cancelled": False})
        result = await api.cancel_async("tourn-001")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_async_missing_key(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._post_async = AsyncMock(return_value={})
        result = await api.cancel_async("tourn-001")
        assert result is False


# ── Integration-like workflow tests ──────────────────────────────────


class TestTournamentWorkflows:
    def test_create_then_get(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        """Simulate creating a tournament then fetching its full details."""
        mock_client._post.return_value = SAMPLE_SUMMARY
        summary = api.create("Workflow Test", agents=["claude", "gpt4"])
        assert summary.id == "tourn-001"

        mock_client._get.return_value = SAMPLE_TOURNAMENT
        full = api.get(summary.id)
        assert full.id == summary.id
        assert full.name == summary.name
        assert len(full.standings) == 2

    def test_create_get_standings_cancel(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        """Simulate full lifecycle: create, check standings, then cancel."""
        mock_client._post.return_value = {
            "id": "tourn-live",
            "name": "Live Tournament",
            "status": "running",
            "participants": 3,
            "rounds_completed": 1,
            "total_rounds": 3,
            "created_at": "2026-02-12T09:00:00Z",
        }
        summary = api.create("Live Tournament", agents=["a", "b", "c"])
        assert summary.status == "running"

        mock_client._get.return_value = {
            "standings": [
                {"agent_id": "a", "rank": 1, "wins": 1, "losses": 0, "draws": 0, "points": 3.0},
                {"agent_id": "b", "rank": 2, "wins": 0, "losses": 1, "draws": 0, "points": 0.0},
            ]
        }
        standings = api.get_standings(summary.id)
        assert len(standings) == 2
        assert standings[0].agent_id == "a"

        mock_client._post.return_value = {"cancelled": True}
        cancelled = api.cancel(summary.id)
        assert cancelled is True

    def test_list_then_get_details(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        """List tournaments and then fetch details for each."""
        mock_client._get.side_effect = [
            {"tournaments": [SAMPLE_SUMMARY, SAMPLE_SUMMARY_2]},
            SAMPLE_TOURNAMENT,
            {**SAMPLE_TOURNAMENT, "id": "tourn-002", "name": "Daily Sprint"},
        ]
        summaries = api.list()
        assert len(summaries) == 2

        details = []
        for s in summaries:
            details.append(api.get(s.id))
        assert details[0].id == "tourn-001"
        assert details[1].id == "tourn-002"

    @pytest.mark.asyncio
    async def test_async_workflow(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        """Full async workflow: create, get, standings, cancel."""
        mock_client._post_async = AsyncMock(return_value=SAMPLE_SUMMARY)
        summary = await api.create_async("Async Workflow", agents=["claude", "gpt4"])
        assert summary.id == "tourn-001"

        mock_client._get_async = AsyncMock(return_value=SAMPLE_TOURNAMENT)
        tournament = await api.get_async(summary.id)
        assert tournament.format == "round_robin"
        assert len(tournament.standings) == 2

        mock_client._get_async = AsyncMock(
            return_value={"standings": [SAMPLE_STANDING]}
        )
        standings = await api.get_standings_async(summary.id)
        assert len(standings) == 1

        mock_client._post_async = AsyncMock(return_value={"cancelled": True})
        cancelled = await api.cancel_async(summary.id)
        assert cancelled is True


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_standing_negative_elo_change(self) -> None:
        standing = TournamentStanding.from_dict(
            {"agent_id": "loser", "rank": 5, "wins": 0, "losses": 5, "draws": 0, "points": 0.0, "elo_change": -50.0}
        )
        assert standing.elo_change == -50.0

    def test_standing_zero_values(self) -> None:
        standing = TournamentStanding.from_dict(
            {"agent_id": "newcomer", "rank": 0, "wins": 0, "losses": 0, "draws": 0, "points": 0.0}
        )
        assert standing.agent_id == "newcomer"
        assert standing.elo_change == 0.0

    def test_tournament_empty_participants(self) -> None:
        data = {**SAMPLE_TOURNAMENT, "participants": []}
        tournament = Tournament.from_dict(data)
        assert tournament.participants == []

    def test_tournament_many_standings(self) -> None:
        standings_data = [
            {"agent_id": f"agent_{i}", "rank": i, "wins": 10 - i, "losses": i, "draws": 0, "points": float(30 - 3 * i)}
            for i in range(10)
        ]
        data = {**SAMPLE_TOURNAMENT, "standings": standings_data}
        tournament = Tournament.from_dict(data)
        assert len(tournament.standings) == 10
        assert tournament.standings[0].agent_id == "agent_0"
        assert tournament.standings[9].agent_id == "agent_9"

    def test_summary_from_dict_extra_keys_ignored(self) -> None:
        data = {**SAMPLE_SUMMARY, "extra_field": "should_be_ignored", "another": 42}
        summary = TournamentSummary.from_dict(data)
        assert summary.id == "tourn-001"
        assert not hasattr(summary, "extra_field")

    def test_tournament_from_dict_extra_keys_ignored(self) -> None:
        data = {**SAMPLE_TOURNAMENT, "unknown_key": True}
        tournament = Tournament.from_dict(data)
        assert tournament.id == "tourn-001"

    def test_standing_from_dict_extra_keys_ignored(self) -> None:
        data = {**SAMPLE_STANDING, "extra": "nope"}
        standing = TournamentStanding.from_dict(data)
        assert standing.agent_id == "claude"

    def test_tournament_with_special_characters_in_id(self, api: TournamentsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {**SAMPLE_TOURNAMENT, "id": "tourn-with-dashes-and_underscores"}
        result = api.get("tourn-with-dashes-and_underscores")
        assert result.id == "tourn-with-dashes-and_underscores"
        mock_client._get.assert_called_once_with("/api/tournaments/tourn-with-dashes-and_underscores")

    def test_api_init_stores_client(self, mock_client: AragoraClient) -> None:
        api = TournamentsAPI(mock_client)
        assert api._client is mock_client

    def test_float_points_precision(self) -> None:
        standing = TournamentStanding.from_dict(
            {"agent_id": "precise", "rank": 1, "wins": 3, "losses": 1, "draws": 2, "points": 11.333333, "elo_change": 0.001}
        )
        assert standing.points == 11.333333
        assert standing.elo_change == 0.001
