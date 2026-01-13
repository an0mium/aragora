"""Tests for TournamentHandler - tournament endpoints."""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile

from aragora.server.handlers.tournaments import TournamentHandler, TOURNAMENT_AVAILABLE


class MockStanding:
    """Mock tournament standing for testing."""

    def __init__(self, agent="claude", wins=5, losses=2, draws=1):
        self.agent = agent
        self.wins = wins
        self.losses = losses
        self.draws = draws
        self.points = wins * 3 + draws
        self.total_score = wins - losses
        self.win_rate = wins / (wins + losses + draws) if (wins + losses + draws) > 0 else 0


@pytest.fixture
def mock_ctx():
    """Create mock context for handler."""
    return {
        "nomic_dir": Path("/tmp/test_nomic"),
        "storage": None,
    }


@pytest.fixture
def handler(mock_ctx):
    """Create TournamentHandler with mock context."""
    return TournamentHandler(mock_ctx)


class TestTournamentHandlerRouting:
    """Test route matching for TournamentHandler."""

    def test_can_handle_list_tournaments(self, handler):
        """Handler should match /api/tournaments."""
        assert handler.can_handle("/api/tournaments")

    def test_can_handle_standings(self, handler):
        """Handler should match /api/tournaments/{id}/standings."""
        assert handler.can_handle("/api/tournaments/main/standings")
        assert handler.can_handle("/api/tournaments/weekly-2024/standings")

    def test_cannot_handle_invalid_path(self, handler):
        """Handler should not match invalid paths."""
        assert not handler.can_handle("/api/tournament")
        assert not handler.can_handle("/api/tournaments/main")
        assert not handler.can_handle("/api/tournaments/main/other")

    def test_cannot_handle_partial_paths(self, handler):
        """Handler should not match partial paths."""
        assert not handler.can_handle("/api/tournaments/main/standings/extra")


class TestListTournamentsEndpoint:
    """Test /api/tournaments endpoint."""

    def test_list_no_nomic_dir(self):
        """Returns 503 when nomic_dir not configured."""
        handler = TournamentHandler({"nomic_dir": None})
        result = handler.handle("/api/tournaments", {}, None)
        assert result is not None

    def test_list_empty_tournaments(self, mock_ctx):
        """Returns empty list when no tournaments exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nomic_dir = Path(tmpdir)
            mock_ctx["nomic_dir"] = nomic_dir
            handler = TournamentHandler(mock_ctx)

            result = handler.handle("/api/tournaments", {}, None)
            assert result is not None

    @pytest.mark.skipif(not TOURNAMENT_AVAILABLE, reason="TournamentManager not available")
    def test_list_with_tournaments(self, mock_ctx):
        """Returns list of tournaments when they exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nomic_dir = Path(tmpdir)
            tournaments_dir = nomic_dir / "tournaments"
            tournaments_dir.mkdir()

            mock_ctx["nomic_dir"] = nomic_dir
            handler = TournamentHandler(mock_ctx)

            with patch("aragora.server.handlers.tournaments.TournamentManager") as MockManager:
                mock_manager = Mock()
                mock_manager.get_current_standings.return_value = [
                    MockStanding("claude", 5, 2, 1),
                    MockStanding("gpt4", 3, 4, 1),
                ]
                MockManager.return_value = mock_manager

                # Create a tournament file
                (tournaments_dir / "main.db").touch()

                result = handler.handle("/api/tournaments", {}, None)
                assert result is not None


class TestStandingsEndpoint:
    """Test /api/tournaments/{id}/standings endpoint."""

    def test_standings_no_nomic_dir(self):
        """Returns 503 when nomic_dir not configured."""
        handler = TournamentHandler({"nomic_dir": None})
        result = handler.handle("/api/tournaments/main/standings", {}, None)
        assert result is not None

    def test_standings_tournament_not_found(self, mock_ctx):
        """Returns 404 when tournament doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nomic_dir = Path(tmpdir)
            tournaments_dir = nomic_dir / "tournaments"
            tournaments_dir.mkdir()

            mock_ctx["nomic_dir"] = nomic_dir
            handler = TournamentHandler(mock_ctx)

            result = handler.handle("/api/tournaments/nonexistent/standings", {}, None)
            assert result is not None

    def test_standings_path_traversal_blocked(self, mock_ctx):
        """Path traversal attempts are blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nomic_dir = Path(tmpdir)
            mock_ctx["nomic_dir"] = nomic_dir
            handler = TournamentHandler(mock_ctx)

            result = handler.handle("/api/tournaments/../etc/passwd/standings", {}, None)
            assert result is not None

    def test_standings_invalid_id(self, mock_ctx):
        """Invalid tournament IDs are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nomic_dir = Path(tmpdir)
            mock_ctx["nomic_dir"] = nomic_dir
            handler = TournamentHandler(mock_ctx)

            result = handler.handle("/api/tournaments/<script>/standings", {}, None)
            assert result is not None

    @pytest.mark.skipif(not TOURNAMENT_AVAILABLE, reason="TournamentManager not available")
    def test_standings_success(self, mock_ctx):
        """Returns standings when tournament exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nomic_dir = Path(tmpdir)
            tournaments_dir = nomic_dir / "tournaments"
            tournaments_dir.mkdir()
            (tournaments_dir / "main.db").touch()

            mock_ctx["nomic_dir"] = nomic_dir
            handler = TournamentHandler(mock_ctx)

            with patch("aragora.server.handlers.tournaments.TournamentManager") as MockManager:
                mock_manager = Mock()
                mock_manager.get_current_standings.return_value = [
                    MockStanding("claude", 5, 2, 1),
                    MockStanding("gpt4", 3, 4, 1),
                ]
                MockManager.return_value = mock_manager

                result = handler.handle("/api/tournaments/main/standings", {}, None)
                assert result is not None


class TestTournamentNotConfigured:
    """Test error handling when tournament system not configured."""

    @pytest.mark.skipif(TOURNAMENT_AVAILABLE, reason="Test requires TournamentManager unavailable")
    def test_list_unavailable(self, mock_ctx):
        """Returns 503 when tournament system unavailable."""
        handler = TournamentHandler(mock_ctx)
        result = handler.handle("/api/tournaments", {}, None)
        assert result is not None

    @pytest.mark.skipif(TOURNAMENT_AVAILABLE, reason="Test requires TournamentManager unavailable")
    def test_standings_unavailable(self, mock_ctx):
        """Returns 503 when tournament system unavailable."""
        handler = TournamentHandler(mock_ctx)
        result = handler.handle("/api/tournaments/main/standings", {}, None)
        assert result is not None


class TestTournamentHandlerImport:
    """Test TournamentHandler import and export."""

    def test_handler_importable(self):
        """TournamentHandler can be imported from handlers package."""
        from aragora.server.handlers import TournamentHandler

        assert TournamentHandler is not None

    def test_handler_in_all_exports(self):
        """TournamentHandler is in __all__ exports."""
        from aragora.server.handlers import __all__

        assert "TournamentHandler" in __all__

    def test_tournament_available_flag(self):
        """TOURNAMENT_AVAILABLE flag is defined."""
        from aragora.server.handlers.tournaments import TOURNAMENT_AVAILABLE

        assert isinstance(TOURNAMENT_AVAILABLE, bool)


class TestErrorHandling:
    """Test error handling in TournamentHandler."""

    def test_handle_returns_none_for_unmatched(self, handler):
        """Handle returns None for unmatched paths."""
        result = handler.handle("/api/unmatched", {}, None)
        assert result is None

    def test_list_handles_exception(self, mock_ctx):
        """List tournaments handles exceptions gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nomic_dir = Path(tmpdir)
            tournaments_dir = nomic_dir / "tournaments"
            tournaments_dir.mkdir()
            (tournaments_dir / "corrupt.db").touch()

            mock_ctx["nomic_dir"] = nomic_dir
            handler = TournamentHandler(mock_ctx)

            # Should handle gracefully even if manager fails
            result = handler.handle("/api/tournaments", {}, None)
            assert result is not None


class TestTournamentIdValidation:
    """Test tournament ID validation."""

    def test_valid_ids(self, mock_ctx):
        """Valid tournament IDs are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nomic_dir = Path(tmpdir)
            mock_ctx["nomic_dir"] = nomic_dir
            handler = TournamentHandler(mock_ctx)

            valid_ids = ["main", "weekly-2024", "tournament_1", "test.tournament"]
            for tid in valid_ids:
                result = handler.handle(f"/api/tournaments/{tid}/standings", {}, None)
                # Should not return 400 for valid IDs
                assert result is not None

    def test_invalid_ids(self, mock_ctx):
        """Invalid tournament IDs are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nomic_dir = Path(tmpdir)
            mock_ctx["nomic_dir"] = nomic_dir
            handler = TournamentHandler(mock_ctx)

            invalid_ids = ["../etc", "foo;bar", "<script>"]
            for tid in invalid_ids:
                result = handler.handle(f"/api/tournaments/{tid}/standings", {}, None)
                assert result is not None
