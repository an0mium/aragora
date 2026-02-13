"""Tests for the Tournaments API."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aragora_client.tournaments import Tournament


class TestTournamentsAPI:
    """Tests for TournamentsAPI methods."""

    @pytest.mark.asyncio
    async def test_list_tournaments(
        self, mock_client, mock_response, tournament_response
    ):
        """Test listing tournaments."""
        response_data = {"tournaments": [tournament_response]}
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.tournaments.list()

        assert len(result) == 1
        assert result[0].id == "tournament-123"
        assert result[0].name == "Test Tournament"

    @pytest.mark.asyncio
    async def test_get_tournament(
        self, mock_client, mock_response, tournament_response
    ):
        """Test getting a tournament by ID."""
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, tournament_response)
        )

        result = await mock_client.tournaments.get("tournament-123")

        assert isinstance(result, Tournament)
        assert result.id == "tournament-123"
        assert result.format == "round_robin"

    @pytest.mark.asyncio
    async def test_create_tournament(
        self, mock_client, mock_response, tournament_response
    ):
        """Test creating a tournament."""
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, tournament_response)
        )

        result = await mock_client.tournaments.create(
            name="Test Tournament",
            format="round_robin",
            participants=["claude", "gpt4", "gemini"],
        )

        assert result.id == "tournament-123"
        assert "claude" in result.participants

    @pytest.mark.asyncio
    async def test_start_tournament(self, mock_client, mock_response):
        """Test starting a tournament."""
        response_data = {
            "id": "tournament-123",
            "name": "Test Tournament",
            "status": "running",
            "format": "round_robin",
            "participants": ["claude", "gpt4"],
            "created_at": "2026-01-01T00:00:00Z",
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.tournaments.start("tournament-123")

        assert result.status == "running"

    @pytest.mark.asyncio
    async def test_cancel_tournament(self, mock_client, mock_response):
        """Test cancelling a tournament."""
        response_data = {
            "id": "tournament-123",
            "name": "Test Tournament",
            "status": "cancelled",
            "format": "round_robin",
            "participants": [],
            "created_at": "2026-01-01T00:00:00Z",
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.tournaments.cancel("tournament-123")

        assert result.status == "cancelled"

    @pytest.mark.asyncio
    async def test_get_standings(self, mock_client, mock_response):
        """Test getting tournament standings."""
        response_data = {
            "tournament_id": "tournament-123",
            "standings": [
                {
                    "rank": 1,
                    "participant": "claude",
                    "wins": 5,
                    "losses": 1,
                    "points": 15,
                    "elo": 1250,
                },
                {
                    "rank": 2,
                    "participant": "gpt4",
                    "wins": 4,
                    "losses": 2,
                    "points": 12,
                    "elo": 1200,
                },
            ],
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.tournaments.get_standings("tournament-123")

        assert result.tournament_id == "tournament-123"
        assert len(result.standings) == 2
        assert result.standings[0].participant == "claude"
        assert result.standings[0].wins == 5
