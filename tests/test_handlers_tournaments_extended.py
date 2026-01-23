"""Extended tests for TournamentHandler - rate limiting, response format, edge cases.

These tests supplement test_handlers_tournaments.py with coverage for:
- Rate limiting enforcement (30 req/min)
- Response JSON structure validation
- Large tournament handling
- Empty/malformed data handling
- Concurrent request scenarios
"""

import json
import pytest
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from aragora.server.handlers.tournaments import (
    TournamentHandler,
    TOURNAMENT_AVAILABLE,
    MAX_TOURNAMENTS_TO_LIST,
    _tournament_limiter,
)
from aragora.server.handlers.utils.rate_limit import RateLimiter


# =============================================================================
# Test Data
# =============================================================================


@dataclass
class MockStanding:
    """Mock tournament standing matching actual TournamentManager output."""

    agent: str
    wins: int
    losses: int
    draws: int
    points: int
    total_score: float
    win_rate: float

    @classmethod
    def create(cls, agent: str, wins: int = 5, losses: int = 2, draws: int = 1):
        """Create a mock standing with calculated fields."""
        total = wins + losses + draws
        return cls(
            agent=agent,
            wins=wins,
            losses=losses,
            draws=draws,
            points=wins * 3 + draws,
            total_score=float(wins - losses),
            win_rate=wins / total if total > 0 else 0.0,
        )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fresh_limiter():
    """Create a fresh rate limiter for testing."""
    limiter = RateLimiter(requests_per_minute=30)
    yield limiter
    limiter.clear()


@pytest.fixture
def mock_handler():
    """Create a mock HTTP handler with client address."""
    handler = Mock()
    handler.client_address = ("192.168.1.100", 12345)
    handler.headers = {}
    return handler


@pytest.fixture
def mock_handler_factory():
    """Factory for creating mock handlers with different IPs."""

    def create(ip: str = "192.168.1.100"):
        handler = Mock()
        handler.client_address = (ip, 12345)
        handler.headers = {}
        return handler

    return create


@pytest.fixture
def tournament_ctx(tmp_path):
    """Create context with temporary nomic directory."""
    tournaments_dir = tmp_path / "tournaments"
    tournaments_dir.mkdir()
    return {
        "nomic_dir": tmp_path,
        "storage": None,
    }


@pytest.fixture
def handler_with_ctx(tournament_ctx):
    """Create TournamentHandler with context."""
    return TournamentHandler(tournament_ctx)


# =============================================================================
# Rate Limiting Tests
# =============================================================================


class TestTournamentRateLimiting:
    """Test rate limiting enforcement for tournament endpoints."""

    def test_rate_limiter_allows_under_limit(self, fresh_limiter):
        """Rate limiter allows requests under the limit."""
        for i in range(29):
            assert fresh_limiter.is_allowed("test-ip")
        # Should still have one request left
        assert fresh_limiter.get_remaining("test-ip") == 1

    def test_rate_limiter_blocks_at_limit(self, fresh_limiter):
        """Rate limiter blocks requests at the limit."""
        for i in range(30):
            assert fresh_limiter.is_allowed("test-ip")
        # 31st request should be blocked
        assert not fresh_limiter.is_allowed("test-ip")

    def test_rate_limiter_per_ip(self, fresh_limiter):
        """Rate limiter tracks limits per IP."""
        # Exhaust limit for IP1
        for i in range(30):
            fresh_limiter.is_allowed("ip1")

        # IP2 should still be allowed
        assert fresh_limiter.is_allowed("ip2")

        # IP1 should be blocked
        assert not fresh_limiter.is_allowed("ip1")

    def test_rate_limiter_reset(self, fresh_limiter):
        """Rate limiter can be reset for a specific key."""
        for i in range(30):
            fresh_limiter.is_allowed("test-ip")
        assert not fresh_limiter.is_allowed("test-ip")

        fresh_limiter.reset("test-ip")
        assert fresh_limiter.is_allowed("test-ip")

    def test_rate_limiter_clear(self, fresh_limiter):
        """Rate limiter clear removes all buckets."""
        for i in range(30):
            fresh_limiter.is_allowed("ip1")
            fresh_limiter.is_allowed("ip2")

        fresh_limiter.clear()

        assert fresh_limiter.is_allowed("ip1")
        assert fresh_limiter.is_allowed("ip2")

    def test_handler_returns_429_when_rate_limited(self, tournament_ctx, mock_handler_factory):
        """Handler returns 429 when rate limit exceeded."""
        handler = TournamentHandler(tournament_ctx)
        mock_h = mock_handler_factory("10.0.0.1")

        # Clear the global limiter first
        _tournament_limiter.clear()

        # Make 30 requests (all should succeed or return valid response)
        for _ in range(30):
            handler.handle("/api/tournaments", {}, mock_h)

        # 31st request should hit rate limit
        result = handler.handle("/api/tournaments", {}, mock_h)
        assert result is not None
        assert result.status_code == 429
        body = json.loads(result.body)
        assert "rate limit" in body.get("error", "").lower()

    def test_rate_limit_different_endpoints_same_limiter(
        self, tournament_ctx, mock_handler_factory
    ):
        """Both tournament endpoints share the same rate limiter."""
        tournaments_dir = Path(tournament_ctx["nomic_dir"]) / "tournaments"
        (tournaments_dir / "test.db").touch()

        handler = TournamentHandler(tournament_ctx)
        mock_h = mock_handler_factory("10.0.0.2")

        _tournament_limiter.clear()

        # Mix requests to both endpoints
        for i in range(15):
            handler.handle("/api/tournaments", {}, mock_h)
            handler.handle("/api/tournaments/test/standings", {}, mock_h)

        # Should be rate limited now (30 total)
        result = handler.handle("/api/tournaments", {}, mock_h)
        assert result is not None
        assert result.status_code == 429

    def test_rate_limiter_thread_safe(self, fresh_limiter):
        """Rate limiter is thread-safe."""
        success_count = [0]
        block_count = [0]
        lock = threading.Lock()

        def make_requests():
            for _ in range(10):
                if fresh_limiter.is_allowed("shared-ip"):
                    with lock:
                        success_count[0] += 1
                else:
                    with lock:
                        block_count[0] += 1

        threads = [threading.Thread(target=make_requests) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 30 should succeed, 20 should be blocked
        assert success_count[0] == 30
        assert block_count[0] == 20


# =============================================================================
# Response Format Tests
# =============================================================================


class TestTournamentResponseFormat:
    """Test response JSON structure and field types."""

    @pytest.mark.skipif(not TOURNAMENT_AVAILABLE, reason="TournamentManager not available")
    def test_list_response_structure(self, tournament_ctx, mock_handler_factory):
        """List tournaments response has correct structure."""
        tournaments_dir = Path(tournament_ctx["nomic_dir"]) / "tournaments"
        (tournaments_dir / "test.db").touch()

        handler = TournamentHandler(tournament_ctx)
        mock_h = mock_handler_factory("10.0.0.3")

        _tournament_limiter.clear()

        with patch("aragora.server.handlers.tournaments.TournamentManager") as MockManager:
            mock_manager = Mock()
            mock_manager.get_current_standings.return_value = [
                MockStanding.create("claude", 5, 2, 1),
            ]
            MockManager.return_value = mock_manager

            result = handler.handle("/api/tournaments", {}, mock_h)
            assert result is not None
            assert result.status_code == 200

            body = json.loads(result.body)
            assert "tournaments" in body
            assert "count" in body
            assert isinstance(body["tournaments"], list)
            assert isinstance(body["count"], int)

    @pytest.mark.skipif(not TOURNAMENT_AVAILABLE, reason="TournamentManager not available")
    def test_list_tournament_item_fields(self, tournament_ctx, mock_handler_factory):
        """Tournament list items have expected fields."""
        tournaments_dir = Path(tournament_ctx["nomic_dir"]) / "tournaments"
        (tournaments_dir / "test.db").touch()

        handler = TournamentHandler(tournament_ctx)
        mock_h = mock_handler_factory("10.0.0.4")

        _tournament_limiter.clear()

        with patch("aragora.server.handlers.tournaments.TournamentManager") as MockManager:
            mock_manager = Mock()
            mock_manager.get_current_standings.return_value = [
                MockStanding.create("claude", 5, 2, 1),
                MockStanding.create("gpt4", 3, 4, 1),
            ]
            MockManager.return_value = mock_manager

            result = handler.handle("/api/tournaments", {}, mock_h)
            body = json.loads(result.body)

            assert len(body["tournaments"]) == 1
            t = body["tournaments"][0]

            assert "tournament_id" in t
            assert "participants" in t
            assert "total_matches" in t
            assert "top_agent" in t

            assert t["tournament_id"] == "test"
            assert t["participants"] == 2
            assert t["top_agent"] == "claude"

    @pytest.mark.skipif(not TOURNAMENT_AVAILABLE, reason="TournamentManager not available")
    def test_standings_response_structure(self, tournament_ctx, mock_handler_factory):
        """Standings response has correct structure."""
        tournaments_dir = Path(tournament_ctx["nomic_dir"]) / "tournaments"
        (tournaments_dir / "main.db").touch()

        handler = TournamentHandler(tournament_ctx)
        mock_h = mock_handler_factory("10.0.0.5")

        _tournament_limiter.clear()

        with patch("aragora.server.handlers.tournaments.TournamentManager") as MockManager:
            mock_manager = Mock()
            mock_manager.get_current_standings.return_value = [
                MockStanding.create("claude", 5, 2, 1),
            ]
            MockManager.return_value = mock_manager

            result = handler.handle("/api/tournaments/main/standings", {}, mock_h)
            assert result.status_code == 200

            body = json.loads(result.body)
            assert "tournament_id" in body
            assert "standings" in body
            assert "count" in body

            assert body["tournament_id"] == "main"
            assert isinstance(body["standings"], list)
            assert body["count"] == 1

    @pytest.mark.skipif(not TOURNAMENT_AVAILABLE, reason="TournamentManager not available")
    def test_standing_item_fields(self, tournament_ctx, mock_handler_factory):
        """Standing items have expected fields with correct types."""
        tournaments_dir = Path(tournament_ctx["nomic_dir"]) / "tournaments"
        (tournaments_dir / "main.db").touch()

        handler = TournamentHandler(tournament_ctx)
        mock_h = mock_handler_factory("10.0.0.6")

        _tournament_limiter.clear()

        with patch("aragora.server.handlers.tournaments.TournamentManager") as MockManager:
            mock_manager = Mock()
            mock_manager.get_current_standings.return_value = [
                MockStanding.create("claude", 5, 2, 1),
            ]
            MockManager.return_value = mock_manager

            result = handler.handle("/api/tournaments/main/standings", {}, mock_h)
            body = json.loads(result.body)

            standing = body["standings"][0]

            # Required fields
            assert "agent" in standing
            assert "wins" in standing
            assert "losses" in standing
            assert "draws" in standing
            assert "points" in standing
            assert "total_score" in standing
            assert "win_rate" in standing

            # Type checks
            assert isinstance(standing["agent"], str)
            assert isinstance(standing["wins"], int)
            assert isinstance(standing["losses"], int)
            assert isinstance(standing["draws"], int)
            assert isinstance(standing["points"], int)
            assert isinstance(standing["total_score"], (int, float))
            assert isinstance(standing["win_rate"], (int, float))

    @pytest.mark.skipif(not TOURNAMENT_AVAILABLE, reason="TournamentManager not available")
    def test_error_response_format(self, tournament_ctx, mock_handler_factory):
        """Error responses have consistent format."""
        handler = TournamentHandler(tournament_ctx)
        mock_h = mock_handler_factory("10.0.0.7")

        _tournament_limiter.clear()

        # Request non-existent tournament
        result = handler.handle("/api/tournaments/nonexistent/standings", {}, mock_h)
        assert result.status_code == 404

        body = json.loads(result.body)
        assert "error" in body
        assert isinstance(body["error"], str)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestTournamentEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.skipif(not TOURNAMENT_AVAILABLE, reason="TournamentManager not available")
    def test_empty_standings(self, tournament_ctx, mock_handler_factory):
        """Handle tournament with no participants."""
        tournaments_dir = Path(tournament_ctx["nomic_dir"]) / "tournaments"
        (tournaments_dir / "empty.db").touch()

        handler = TournamentHandler(tournament_ctx)
        mock_h = mock_handler_factory("10.0.0.8")

        _tournament_limiter.clear()

        with patch("aragora.server.handlers.tournaments.TournamentManager") as MockManager:
            mock_manager = Mock()
            mock_manager.get_current_standings.return_value = []
            MockManager.return_value = mock_manager

            result = handler.handle("/api/tournaments/empty/standings", {}, mock_h)
            assert result.status_code == 200

            body = json.loads(result.body)
            assert body["standings"] == []
            assert body["count"] == 0

    @pytest.mark.skipif(not TOURNAMENT_AVAILABLE, reason="TournamentManager not available")
    def test_list_with_empty_tournament(self, tournament_ctx, mock_handler_factory):
        """List includes empty tournaments correctly."""
        tournaments_dir = Path(tournament_ctx["nomic_dir"]) / "tournaments"
        (tournaments_dir / "empty.db").touch()

        handler = TournamentHandler(tournament_ctx)
        mock_h = mock_handler_factory("10.0.0.9")

        _tournament_limiter.clear()

        with patch("aragora.server.handlers.tournaments.TournamentManager") as MockManager:
            mock_manager = Mock()
            mock_manager.get_current_standings.return_value = []
            MockManager.return_value = mock_manager

            result = handler.handle("/api/tournaments", {}, mock_h)
            body = json.loads(result.body)

            assert len(body["tournaments"]) == 1
            t = body["tournaments"][0]
            assert t["participants"] == 0
            assert t["total_matches"] == 0
            assert t["top_agent"] is None

    @pytest.mark.skipif(not TOURNAMENT_AVAILABLE, reason="TournamentManager not available")
    def test_max_tournaments_limit(self, tournament_ctx, mock_handler_factory):
        """List tournaments respects MAX_TOURNAMENTS_TO_LIST."""
        tournaments_dir = Path(tournament_ctx["nomic_dir"]) / "tournaments"

        # Create more than MAX tournaments
        for i in range(MAX_TOURNAMENTS_TO_LIST + 10):
            (tournaments_dir / f"tournament_{i:03d}.db").touch()

        handler = TournamentHandler(tournament_ctx)
        mock_h = mock_handler_factory("10.0.0.10")

        _tournament_limiter.clear()

        with patch("aragora.server.handlers.tournaments.TournamentManager") as MockManager:
            mock_manager = Mock()
            mock_manager.get_current_standings.return_value = [
                MockStanding.create("agent", 1, 0, 0)
            ]
            MockManager.return_value = mock_manager

            result = handler.handle("/api/tournaments", {}, mock_h)
            body = json.loads(result.body)

            # Should be capped at MAX_TOURNAMENTS_TO_LIST
            assert body["count"] <= MAX_TOURNAMENTS_TO_LIST

    @pytest.mark.skipif(not TOURNAMENT_AVAILABLE, reason="TournamentManager not available")
    def test_large_standings_list(self, tournament_ctx, mock_handler_factory):
        """Handle tournament with many participants."""
        tournaments_dir = Path(tournament_ctx["nomic_dir"]) / "tournaments"
        (tournaments_dir / "large.db").touch()

        handler = TournamentHandler(tournament_ctx)
        mock_h = mock_handler_factory("10.0.0.11")

        _tournament_limiter.clear()

        with patch("aragora.server.handlers.tournaments.TournamentManager") as MockManager:
            mock_manager = Mock()
            # Create 500 participants
            mock_manager.get_current_standings.return_value = [
                MockStanding.create(f"agent_{i}", i % 10, i % 5, i % 3) for i in range(500)
            ]
            MockManager.return_value = mock_manager

            result = handler.handle("/api/tournaments/large/standings", {}, mock_h)
            assert result.status_code == 200

            body = json.loads(result.body)
            assert body["count"] == 500
            assert len(body["standings"]) == 500

    @pytest.mark.skipif(not TOURNAMENT_AVAILABLE, reason="TournamentManager not available")
    def test_tournaments_dir_not_exists(self, tmp_path, mock_handler_factory):
        """Handle missing tournaments directory."""
        ctx = {"nomic_dir": tmp_path, "storage": None}
        # Don't create tournaments dir

        handler = TournamentHandler(ctx)
        mock_h = mock_handler_factory("10.0.0.12")

        _tournament_limiter.clear()

        result = handler.handle("/api/tournaments", {}, mock_h)
        assert result.status_code == 200

        body = json.loads(result.body)
        assert body["tournaments"] == []
        assert body["count"] == 0

    @pytest.mark.skipif(not TOURNAMENT_AVAILABLE, reason="TournamentManager not available")
    def test_corrupted_tournament_skipped(self, tournament_ctx, mock_handler_factory):
        """Corrupted tournament files are skipped in list."""
        tournaments_dir = Path(tournament_ctx["nomic_dir"]) / "tournaments"
        (tournaments_dir / "good.db").touch()
        (tournaments_dir / "corrupt.db").touch()

        handler = TournamentHandler(tournament_ctx)
        mock_h = mock_handler_factory("10.0.0.13")

        _tournament_limiter.clear()

        call_count = [0]

        def mock_init(db_path):
            call_count[0] += 1
            mock = Mock()
            if "corrupt" in db_path:
                mock.get_current_standings.side_effect = Exception("DB corrupt")
            else:
                mock.get_current_standings.return_value = [MockStanding.create("agent", 1, 0, 0)]
            return mock

        with patch("aragora.server.handlers.tournaments.TournamentManager", side_effect=mock_init):
            result = handler.handle("/api/tournaments", {}, mock_h)
            body = json.loads(result.body)

            # Only good tournament should be listed
            assert body["count"] == 1
            assert body["tournaments"][0]["tournament_id"] == "good"

    @pytest.mark.skipif(not TOURNAMENT_AVAILABLE, reason="TournamentManager not available")
    def test_no_nomic_dir_list(self, mock_handler_factory):
        """Returns 503 when nomic_dir not configured for list."""
        ctx = {"nomic_dir": None, "storage": None}
        handler = TournamentHandler(ctx)
        mock_h = mock_handler_factory("10.0.0.14")

        _tournament_limiter.clear()

        result = handler.handle("/api/tournaments", {}, mock_h)
        assert result.status_code == 503

        body = json.loads(result.body)
        assert "not configured" in body["error"].lower()

    @pytest.mark.skipif(not TOURNAMENT_AVAILABLE, reason="TournamentManager not available")
    def test_no_nomic_dir_standings(self, mock_handler_factory):
        """Returns 503 when nomic_dir not configured for standings."""
        ctx = {"nomic_dir": None, "storage": None}
        handler = TournamentHandler(ctx)
        mock_h = mock_handler_factory("10.0.0.15")

        _tournament_limiter.clear()

        result = handler.handle("/api/tournaments/test/standings", {}, mock_h)
        assert result.status_code == 503


# =============================================================================
# Data Parsing Tests
# =============================================================================


class TestTournamentDataParsing:
    """Test parsing of tournament data and calculations."""

    @pytest.mark.skipif(not TOURNAMENT_AVAILABLE, reason="TournamentManager not available")
    def test_total_matches_calculation(self, tournament_ctx, mock_handler_factory):
        """Total matches calculated correctly from standings."""
        tournaments_dir = Path(tournament_ctx["nomic_dir"]) / "tournaments"
        (tournaments_dir / "calc.db").touch()

        handler = TournamentHandler(tournament_ctx)
        mock_h = mock_handler_factory("10.0.0.16")

        _tournament_limiter.clear()

        with patch("aragora.server.handlers.tournaments.TournamentManager") as MockManager:
            mock_manager = Mock()
            # 2 agents: A has 3 wins/2 losses/1 draw, B has 2 wins/3 losses/1 draw
            # Total matches = (3+2+1 + 2+3+1) / 2 = 6
            mock_manager.get_current_standings.return_value = [
                MockStanding.create("agent_a", 3, 2, 1),
                MockStanding.create("agent_b", 2, 3, 1),
            ]
            MockManager.return_value = mock_manager

            result = handler.handle("/api/tournaments", {}, mock_h)
            body = json.loads(result.body)

            t = body["tournaments"][0]
            # (3+2+1) + (2+3+1) = 12, divided by 2 = 6
            assert t["total_matches"] == 6

    @pytest.mark.skipif(not TOURNAMENT_AVAILABLE, reason="TournamentManager not available")
    def test_top_agent_selection(self, tournament_ctx, mock_handler_factory):
        """Top agent is first in standings list."""
        tournaments_dir = Path(tournament_ctx["nomic_dir"]) / "tournaments"
        (tournaments_dir / "ranked.db").touch()

        handler = TournamentHandler(tournament_ctx)
        mock_h = mock_handler_factory("10.0.0.17")

        _tournament_limiter.clear()

        with patch("aragora.server.handlers.tournaments.TournamentManager") as MockManager:
            mock_manager = Mock()
            # gpt4 is first (top agent)
            mock_manager.get_current_standings.return_value = [
                MockStanding.create("gpt4", 10, 0, 0),
                MockStanding.create("claude", 5, 5, 0),
                MockStanding.create("gemini", 0, 10, 0),
            ]
            MockManager.return_value = mock_manager

            result = handler.handle("/api/tournaments", {}, mock_h)
            body = json.loads(result.body)

            assert body["tournaments"][0]["top_agent"] == "gpt4"

    @pytest.mark.skipif(not TOURNAMENT_AVAILABLE, reason="TournamentManager not available")
    def test_win_rate_in_standings(self, tournament_ctx, mock_handler_factory):
        """Win rate is correctly included in standings."""
        tournaments_dir = Path(tournament_ctx["nomic_dir"]) / "tournaments"
        (tournaments_dir / "rates.db").touch()

        handler = TournamentHandler(tournament_ctx)
        mock_h = mock_handler_factory("10.0.0.18")

        _tournament_limiter.clear()

        with patch("aragora.server.handlers.tournaments.TournamentManager") as MockManager:
            mock_manager = Mock()
            mock_manager.get_current_standings.return_value = [
                MockStanding.create("agent", 5, 3, 2),  # 5/10 = 0.5 win rate
            ]
            MockManager.return_value = mock_manager

            result = handler.handle("/api/tournaments/rates/standings", {}, mock_h)
            body = json.loads(result.body)

            standing = body["standings"][0]
            assert standing["win_rate"] == 0.5

    @pytest.mark.skipif(not TOURNAMENT_AVAILABLE, reason="TournamentManager not available")
    def test_points_calculation_in_standings(self, tournament_ctx, mock_handler_factory):
        """Points are correctly passed through."""
        tournaments_dir = Path(tournament_ctx["nomic_dir"]) / "tournaments"
        (tournaments_dir / "points.db").touch()

        handler = TournamentHandler(tournament_ctx)
        mock_h = mock_handler_factory("10.0.0.19")

        _tournament_limiter.clear()

        with patch("aragora.server.handlers.tournaments.TournamentManager") as MockManager:
            mock_manager = Mock()
            # 5 wins * 3 + 2 draws = 17 points
            mock_manager.get_current_standings.return_value = [
                MockStanding.create("agent", 5, 3, 2),
            ]
            MockManager.return_value = mock_manager

            result = handler.handle("/api/tournaments/points/standings", {}, mock_h)
            body = json.loads(result.body)

            standing = body["standings"][0]
            assert standing["points"] == 17

    @pytest.mark.skipif(not TOURNAMENT_AVAILABLE, reason="TournamentManager not available")
    def test_multiple_tournaments_different_data(self, tournament_ctx, mock_handler_factory):
        """Multiple tournaments have independent data."""
        tournaments_dir = Path(tournament_ctx["nomic_dir"]) / "tournaments"
        (tournaments_dir / "weekly.db").touch()
        (tournaments_dir / "monthly.db").touch()

        handler = TournamentHandler(tournament_ctx)
        mock_h = mock_handler_factory("10.0.0.20")

        _tournament_limiter.clear()

        standings_data = {
            "weekly": [MockStanding.create("a", 1, 0, 0)],
            "monthly": [MockStanding.create("b", 2, 0, 0), MockStanding.create("c", 1, 1, 0)],
        }

        def mock_init(db_path):
            mock = Mock()
            name = Path(db_path).stem
            mock.get_current_standings.return_value = standings_data.get(name, [])
            return mock

        with patch("aragora.server.handlers.tournaments.TournamentManager", side_effect=mock_init):
            result = handler.handle("/api/tournaments", {}, mock_h)
            body = json.loads(result.body)

            tournaments_by_id = {t["tournament_id"]: t for t in body["tournaments"]}

            assert tournaments_by_id["weekly"]["participants"] == 1
            assert tournaments_by_id["monthly"]["participants"] == 2


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestTournamentInputValidation:
    """Test input validation and security."""

    def test_sql_injection_in_id(self, tournament_ctx, mock_handler_factory):
        """SQL injection attempts in tournament ID are safe."""
        handler = TournamentHandler(tournament_ctx)
        mock_h = mock_handler_factory("10.0.0.21")

        _tournament_limiter.clear()

        malicious_ids = [
            "test'; DROP TABLE tournaments;--",
            'test" OR "1"="1',
            "test; DELETE FROM *;",
        ]

        for tid in malicious_ids:
            result = handler.handle(f"/api/tournaments/{tid}/standings", {}, mock_h)
            assert result is not None
            # Should return 400 or 404, not succeed or crash
            assert result.status_code in (400, 404)

    def test_special_characters_in_id(self, tournament_ctx, mock_handler_factory):
        """Special characters in tournament ID handled safely."""
        handler = TournamentHandler(tournament_ctx)
        mock_h = mock_handler_factory("10.0.0.22")

        _tournament_limiter.clear()

        special_ids = [
            "test<script>alert(1)</script>",
            "test%00null",
            "test\x00null",
        ]

        for tid in special_ids:
            result = handler.handle(f"/api/tournaments/{tid}/standings", {}, mock_h)
            assert result is not None
            # Should not crash

    def test_unicode_in_id(self, tournament_ctx, mock_handler_factory):
        """Unicode in tournament ID handled safely."""
        handler = TournamentHandler(tournament_ctx)
        mock_h = mock_handler_factory("10.0.0.23")

        _tournament_limiter.clear()

        result = handler.handle("/api/tournaments/日本語/standings", {}, mock_h)
        assert result is not None

    def test_very_long_tournament_id(self, tournament_ctx, mock_handler_factory):
        """Very long tournament IDs handled safely."""
        handler = TournamentHandler(tournament_ctx)
        mock_h = mock_handler_factory("10.0.0.24")

        _tournament_limiter.clear()

        long_id = "a" * 10000
        result = handler.handle(f"/api/tournaments/{long_id}/standings", {}, mock_h)
        assert result is not None
        # Should not crash or hang


# =============================================================================
# Concurrent Access Tests
# =============================================================================


class TestTournamentConcurrency:
    """Test concurrent request handling."""

    @pytest.mark.skipif(not TOURNAMENT_AVAILABLE, reason="TournamentManager not available")
    def test_concurrent_list_requests(self, tournament_ctx, mock_handler_factory):
        """Multiple concurrent list requests handled correctly."""
        tournaments_dir = Path(tournament_ctx["nomic_dir"]) / "tournaments"
        (tournaments_dir / "test.db").touch()

        handler = TournamentHandler(tournament_ctx)
        _tournament_limiter.clear()

        results = []
        errors = []

        def make_request(ip_suffix):
            try:
                mock_h = mock_handler_factory(f"10.1.0.{ip_suffix}")
                with patch("aragora.server.handlers.tournaments.TournamentManager") as MockManager:
                    mock_manager = Mock()
                    mock_manager.get_current_standings.return_value = [
                        MockStanding.create("agent", 1, 0, 0)
                    ]
                    MockManager.return_value = mock_manager

                    result = handler.handle("/api/tournaments", {}, mock_h)
                    results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=make_request, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        # All should succeed (different IPs, no rate limiting)
        for r in results:
            assert r is not None
            assert r.status_code == 200

    @pytest.mark.skipif(not TOURNAMENT_AVAILABLE, reason="TournamentManager not available")
    def test_concurrent_standings_requests(self, tournament_ctx, mock_handler_factory):
        """Multiple concurrent standings requests handled correctly."""
        tournaments_dir = Path(tournament_ctx["nomic_dir"]) / "tournaments"
        (tournaments_dir / "test.db").touch()

        handler = TournamentHandler(tournament_ctx)
        _tournament_limiter.clear()

        results = []
        errors = []

        def make_request(ip_suffix):
            try:
                mock_h = mock_handler_factory(f"10.2.0.{ip_suffix}")
                with patch("aragora.server.handlers.tournaments.TournamentManager") as MockManager:
                    mock_manager = Mock()
                    mock_manager.get_current_standings.return_value = [
                        MockStanding.create("agent", ip_suffix, 0, 0)
                    ]
                    MockManager.return_value = mock_manager

                    result = handler.handle("/api/tournaments/test/standings", {}, mock_h)
                    results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=make_request, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10


# =============================================================================
# Handler Registration Tests
# =============================================================================


class TestTournamentHandlerRegistration:
    """Test handler route registration."""

    def test_routes_constant(self):
        """ROUTES constant defined correctly."""
        assert hasattr(TournamentHandler, "ROUTES")
        assert "/api/tournaments" in TournamentHandler.ROUTES
        assert "/api/tournaments/*/standings" in TournamentHandler.ROUTES

    def test_can_handle_all_routes(self, tournament_ctx):
        """Handler can handle all declared routes."""
        handler = TournamentHandler(tournament_ctx)

        assert handler.can_handle("/api/v1/tournaments")
        assert handler.can_handle("/api/v1/tournaments/any-id/standings")

    def test_max_tournaments_constant(self):
        """MAX_TOURNAMENTS_TO_LIST is reasonable."""
        assert MAX_TOURNAMENTS_TO_LIST > 0
        assert MAX_TOURNAMENTS_TO_LIST <= 1000  # Reasonable upper bound
