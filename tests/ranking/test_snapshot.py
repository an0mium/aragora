"""Tests for ELO snapshot engine."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from aragora.ranking.snapshot import (
    write_snapshot,
    read_snapshot_leaderboard,
    read_snapshot_matches,
)


class MockRating:
    """Mock AgentRating for testing."""

    def __init__(self, name: str, elo: float, wins: int, losses: int, draws: int):
        self.agent_name = name
        self.elo = elo
        self.wins = wins
        self.losses = losses
        self.draws = draws

    @property
    def games_played(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        total = self.games_played
        return self.wins / total if total > 0 else 0.0


class TestWriteSnapshot:
    """Test write_snapshot function."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_write_snapshot_success(self, temp_dir):
        """Test successful snapshot write."""
        snapshot_path = temp_dir / "snapshot.json"

        ratings = [
            MockRating("claude", 1200.0, 10, 5, 2),
            MockRating("gpt", 1150.0, 8, 7, 2),
        ]
        matches = [
            {"debate_id": "d1", "winner": "claude"},
            {"debate_id": "d2", "winner": "gpt"},
        ]

        leaderboard_getter = MagicMock(return_value=ratings)
        matches_getter = MagicMock(return_value=matches)

        write_snapshot(
            snapshot_path,
            leaderboard_getter,
            matches_getter,
            leaderboard_limit=10,
            matches_limit=5,
        )

        # File should exist
        assert snapshot_path.exists()

        # Verify content
        with open(snapshot_path) as f:
            data = json.load(f)

        assert len(data["leaderboard"]) == 2
        assert data["leaderboard"][0]["agent_name"] == "claude"
        assert data["leaderboard"][0]["elo"] == 1200.0
        assert data["leaderboard"][0]["wins"] == 10
        assert data["leaderboard"][0]["games_played"] == 17
        assert len(data["recent_matches"]) == 2
        assert "updated_at" in data

        # Verify getters were called with correct limits
        leaderboard_getter.assert_called_once_with(10)
        matches_getter.assert_called_once_with(5)

    def test_write_snapshot_atomic(self, temp_dir):
        """Test snapshot is written atomically."""
        snapshot_path = temp_dir / "snapshot.json"
        temp_path = snapshot_path.with_suffix(".tmp")

        ratings = [MockRating("claude", 1200.0, 10, 5, 2)]
        matches = []

        write_snapshot(
            snapshot_path,
            MagicMock(return_value=ratings),
            MagicMock(return_value=matches),
        )

        # Temp file should not exist after successful write
        assert not temp_path.exists()
        # Main file should exist
        assert snapshot_path.exists()

    def test_write_snapshot_handles_error(self, temp_dir):
        """Test snapshot write handles errors gracefully."""
        # Use read-only directory to cause write error
        snapshot_path = Path("/nonexistent_dir/snapshot.json")

        ratings = [MockRating("claude", 1200.0, 10, 5, 2)]
        matches = []

        # Should not raise exception
        write_snapshot(
            snapshot_path,
            MagicMock(return_value=ratings),
            MagicMock(return_value=matches),
        )

    def test_write_snapshot_empty_data(self, temp_dir):
        """Test writing snapshot with empty data."""
        snapshot_path = temp_dir / "snapshot.json"

        write_snapshot(
            snapshot_path,
            MagicMock(return_value=[]),
            MagicMock(return_value=[]),
        )

        assert snapshot_path.exists()

        with open(snapshot_path) as f:
            data = json.load(f)

        assert data["leaderboard"] == []
        assert data["recent_matches"] == []

    def test_write_snapshot_preserves_rating_properties(self, temp_dir):
        """Test all rating properties are preserved in snapshot."""
        snapshot_path = temp_dir / "snapshot.json"

        rating = MockRating("claude", 1234.5, 15, 10, 5)

        write_snapshot(
            snapshot_path,
            MagicMock(return_value=[rating]),
            MagicMock(return_value=[]),
        )

        with open(snapshot_path) as f:
            data = json.load(f)

        entry = data["leaderboard"][0]
        assert entry["agent_name"] == "claude"
        assert entry["elo"] == 1234.5
        assert entry["wins"] == 15
        assert entry["losses"] == 10
        assert entry["draws"] == 5
        assert entry["games_played"] == 30
        assert abs(entry["win_rate"] - 0.5) < 0.01


class TestReadSnapshotLeaderboard:
    """Test read_snapshot_leaderboard function."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_read_nonexistent_file(self, temp_dir):
        """Test reading nonexistent file returns None."""
        result = read_snapshot_leaderboard(temp_dir / "nonexistent.json")
        assert result is None

    def test_read_valid_snapshot(self, temp_dir):
        """Test reading valid snapshot."""
        snapshot_path = temp_dir / "snapshot.json"
        data = {
            "leaderboard": [
                {"agent_name": "claude", "elo": 1200},
                {"agent_name": "gpt", "elo": 1150},
                {"agent_name": "gemini", "elo": 1100},
            ],
            "recent_matches": [],
            "updated_at": "2025-01-01T00:00:00",
        }

        with open(snapshot_path, "w") as f:
            json.dump(data, f)

        result = read_snapshot_leaderboard(snapshot_path)

        assert result is not None
        assert len(result) == 3
        assert result[0]["agent_name"] == "claude"

    def test_read_with_limit(self, temp_dir):
        """Test reading with limit."""
        snapshot_path = temp_dir / "snapshot.json"
        data = {
            "leaderboard": [{"agent_name": f"agent{i}", "elo": 1200 - i * 10} for i in range(10)],
            "recent_matches": [],
        }

        with open(snapshot_path, "w") as f:
            json.dump(data, f)

        result = read_snapshot_leaderboard(snapshot_path, limit=3)

        assert result is not None
        assert len(result) == 3

    def test_read_corrupted_json(self, temp_dir):
        """Test reading corrupted JSON returns None."""
        snapshot_path = temp_dir / "snapshot.json"

        with open(snapshot_path, "w") as f:
            f.write("not valid json {{{")

        result = read_snapshot_leaderboard(snapshot_path)
        assert result is None

    def test_read_missing_leaderboard_key(self, temp_dir):
        """Test reading file without leaderboard key."""
        snapshot_path = temp_dir / "snapshot.json"
        data = {"recent_matches": [], "updated_at": "2025-01-01"}

        with open(snapshot_path, "w") as f:
            json.dump(data, f)

        result = read_snapshot_leaderboard(snapshot_path)

        # Should return empty list (default for missing key)
        assert result == []

    def test_read_permission_error(self, temp_dir):
        """Test handling permission error."""
        # Mock open to raise PermissionError
        snapshot_path = temp_dir / "snapshot.json"

        # Create the file first
        with open(snapshot_path, "w") as f:
            json.dump({"leaderboard": []}, f)

        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            result = read_snapshot_leaderboard(snapshot_path)

        assert result is None


class TestReadSnapshotMatches:
    """Test read_snapshot_matches function."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_read_nonexistent_file(self, temp_dir):
        """Test reading nonexistent file returns None."""
        result = read_snapshot_matches(temp_dir / "nonexistent.json")
        assert result is None

    def test_read_valid_matches(self, temp_dir):
        """Test reading valid matches."""
        snapshot_path = temp_dir / "snapshot.json"
        data = {
            "leaderboard": [],
            "recent_matches": [
                {"debate_id": "d1", "winner": "claude"},
                {"debate_id": "d2", "winner": "gpt"},
            ],
        }

        with open(snapshot_path, "w") as f:
            json.dump(data, f)

        result = read_snapshot_matches(snapshot_path)

        assert result is not None
        assert len(result) == 2
        assert result[0]["debate_id"] == "d1"

    def test_read_matches_with_limit(self, temp_dir):
        """Test reading matches with limit."""
        snapshot_path = temp_dir / "snapshot.json"
        data = {
            "leaderboard": [],
            "recent_matches": [{"debate_id": f"d{i}"} for i in range(20)],
        }

        with open(snapshot_path, "w") as f:
            json.dump(data, f)

        result = read_snapshot_matches(snapshot_path, limit=5)

        assert result is not None
        assert len(result) == 5

    def test_read_corrupted_json(self, temp_dir):
        """Test reading corrupted JSON returns None."""
        snapshot_path = temp_dir / "snapshot.json"

        with open(snapshot_path, "w") as f:
            f.write("invalid json")

        result = read_snapshot_matches(snapshot_path)
        assert result is None

    def test_read_missing_matches_key(self, temp_dir):
        """Test reading file without recent_matches key."""
        snapshot_path = temp_dir / "snapshot.json"
        data = {"leaderboard": []}

        with open(snapshot_path, "w") as f:
            json.dump(data, f)

        result = read_snapshot_matches(snapshot_path)

        # Should return empty list
        assert result == []

    def test_read_permission_error(self, temp_dir):
        """Test handling permission error."""
        snapshot_path = temp_dir / "snapshot.json"

        with open(snapshot_path, "w") as f:
            json.dump({"recent_matches": []}, f)

        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            result = read_snapshot_matches(snapshot_path)

        assert result is None

    def test_read_os_error(self, temp_dir):
        """Test handling OS error."""
        snapshot_path = temp_dir / "snapshot.json"

        with open(snapshot_path, "w") as f:
            json.dump({"recent_matches": []}, f)

        with patch("builtins.open", side_effect=OSError("Disk error")):
            result = read_snapshot_matches(snapshot_path)

        assert result is None


class TestSnapshotIntegration:
    """Integration tests for snapshot workflow."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_write_then_read_leaderboard(self, temp_dir):
        """Test writing and reading leaderboard."""
        snapshot_path = temp_dir / "snapshot.json"

        ratings = [
            MockRating("claude", 1250.0, 15, 5, 3),
            MockRating("gpt", 1180.0, 12, 8, 2),
        ]
        matches = [{"debate_id": "d1", "winner": "claude"}]

        write_snapshot(
            snapshot_path,
            MagicMock(return_value=ratings),
            MagicMock(return_value=matches),
        )

        leaderboard = read_snapshot_leaderboard(snapshot_path)

        assert leaderboard is not None
        assert len(leaderboard) == 2
        assert leaderboard[0]["agent_name"] == "claude"
        assert leaderboard[0]["elo"] == 1250.0
        assert leaderboard[0]["wins"] == 15

    def test_write_then_read_matches(self, temp_dir):
        """Test writing and reading matches."""
        snapshot_path = temp_dir / "snapshot.json"

        ratings = [MockRating("claude", 1200.0, 10, 5, 2)]
        matches = [
            {"debate_id": "d1", "winner": "claude", "domain": "security"},
            {"debate_id": "d2", "winner": "gpt", "domain": "performance"},
        ]

        write_snapshot(
            snapshot_path,
            MagicMock(return_value=ratings),
            MagicMock(return_value=matches),
        )

        result = read_snapshot_matches(snapshot_path)

        assert result is not None
        assert len(result) == 2
        assert result[0]["debate_id"] == "d1"
        assert result[1]["domain"] == "performance"

    def test_overwrite_existing_snapshot(self, temp_dir):
        """Test overwriting existing snapshot."""
        snapshot_path = temp_dir / "snapshot.json"

        # Write initial snapshot
        write_snapshot(
            snapshot_path,
            MagicMock(return_value=[MockRating("old", 1000.0, 1, 1, 0)]),
            MagicMock(return_value=[]),
        )

        # Overwrite with new data
        write_snapshot(
            snapshot_path,
            MagicMock(return_value=[MockRating("new", 1500.0, 20, 5, 0)]),
            MagicMock(return_value=[]),
        )

        leaderboard = read_snapshot_leaderboard(snapshot_path)

        assert leaderboard is not None
        assert len(leaderboard) == 1
        assert leaderboard[0]["agent_name"] == "new"
        assert leaderboard[0]["elo"] == 1500.0

    def test_snapshot_has_timestamp(self, temp_dir):
        """Test snapshot includes timestamp."""
        snapshot_path = temp_dir / "snapshot.json"

        write_snapshot(
            snapshot_path,
            MagicMock(return_value=[]),
            MagicMock(return_value=[]),
        )

        with open(snapshot_path) as f:
            data = json.load(f)

        assert "updated_at" in data
        # Should be ISO format
        assert "T" in data["updated_at"]
