"""Tests for LeaderboardAPI resource."""

import pytest

from aragora.client import AragoraClient
from aragora.client.resources.leaderboard import LeaderboardAPI


class TestLeaderboardAPI:
    """Tests for LeaderboardAPI resource."""

    def test_leaderboard_api_exists(self):
        """Test that LeaderboardAPI is accessible on client."""
        client = AragoraClient()
        assert isinstance(client.leaderboard, LeaderboardAPI)

    def test_leaderboard_api_has_get_methods(self):
        """Test that LeaderboardAPI has get methods."""
        client = AragoraClient()
        assert hasattr(client.leaderboard, "get")
        assert hasattr(client.leaderboard, "get_async")
        assert callable(client.leaderboard.get)


class TestLeaderboardModels:
    """Tests for Leaderboard model classes."""

    def test_leaderboard_entry_import(self):
        """Test LeaderboardEntry model can be imported."""
        from aragora.client.models import LeaderboardEntry

        entry = LeaderboardEntry(
            rank=1,
            agent_id="claude",
            elo_rating=1650,
            matches_played=100,
            win_rate=0.75,
        )
        assert entry.rank == 1
        assert entry.agent_id == "claude"
        assert entry.elo_rating == 1650
