"""Tests for leaderboard engine - read-only analytics operations."""

import pytest
import json
from unittest.mock import MagicMock, patch

from aragora.ranking.leaderboard_engine import (
    LeaderboardEngine,
    _validate_agent_name,
    MAX_AGENT_NAME_LENGTH,
)
from aragora.exceptions import ConfigurationError


class TestValidateAgentName:
    """Test agent name validation."""

    def test_valid_short_name(self):
        """Test valid short agent name."""
        _validate_agent_name("claude")  # Should not raise

    def test_valid_exact_limit(self):
        """Test name exactly at limit."""
        name = "a" * MAX_AGENT_NAME_LENGTH
        _validate_agent_name(name)  # Should not raise

    def test_invalid_too_long(self):
        """Test name exceeding limit."""
        name = "a" * (MAX_AGENT_NAME_LENGTH + 1)
        with pytest.raises(ValueError) as exc:
            _validate_agent_name(name)
        assert "exceeds" in str(exc.value)
        assert str(MAX_AGENT_NAME_LENGTH) in str(exc.value)

    def test_empty_name(self):
        """Test empty name is valid (no length issue)."""
        _validate_agent_name("")  # Should not raise


class TestLeaderboardEngineInit:
    """Test LeaderboardEngine initialization."""

    def test_init_minimal(self):
        """Test initialization with minimal arguments."""
        mock_db = MagicMock()
        engine = LeaderboardEngine(mock_db)

        assert engine._db is mock_db
        assert engine._leaderboard_cache is None
        assert engine._stats_cache is None
        assert engine._rating_cache is None
        assert engine._rating_factory is None

    def test_init_with_caches(self):
        """Test initialization with caches."""
        mock_db = MagicMock()
        mock_lb_cache = MagicMock()
        mock_stats_cache = MagicMock()
        mock_rating_cache = MagicMock()

        engine = LeaderboardEngine(
            mock_db,
            leaderboard_cache=mock_lb_cache,
            stats_cache=mock_stats_cache,
            rating_cache=mock_rating_cache,
        )

        assert engine._leaderboard_cache is mock_lb_cache
        assert engine._stats_cache is mock_stats_cache
        assert engine._rating_cache is mock_rating_cache

    def test_init_with_rating_factory(self):
        """Test initialization with rating factory."""
        mock_db = MagicMock()

        def factory(row):
            return {"name": row[0], "elo": row[1]}

        engine = LeaderboardEngine(mock_db, rating_factory=factory)
        assert engine._rating_factory is factory


class TestGetLeaderboard:
    """Test get_leaderboard method."""

    def test_get_leaderboard_no_factory_raises(self):
        """Test that missing rating_factory raises error."""
        mock_db = MagicMock()
        engine = LeaderboardEngine(mock_db)

        with pytest.raises(ConfigurationError) as exc:
            engine.get_leaderboard()
        assert "rating_factory" in str(exc.value)

    def test_get_leaderboard_global(self):
        """Test getting global leaderboard."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        mock_db.connection.return_value = mock_conn

        mock_cursor.fetchall.return_value = [
            ("claude", 1200.0, "{}", 10, 5, 2, 17, 8, 10, "2025-01-01"),
            ("gpt", 1150.0, "{}", 8, 6, 1, 15, 7, 9, "2025-01-01"),
        ]

        def factory(row):
            return {"name": row[0], "elo": row[1]}

        engine = LeaderboardEngine(mock_db, rating_factory=factory)

        result = engine.get_leaderboard(limit=10)

        assert len(result) == 2
        assert result[0]["name"] == "claude"
        assert result[0]["elo"] == 1200.0
        # Check SQL didn't use domain
        sql = mock_cursor.execute.call_args[0][0]
        assert "ORDER BY elo DESC" in sql

    def test_get_leaderboard_with_domain(self):
        """Test getting domain-specific leaderboard."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        mock_db.connection.return_value = mock_conn

        mock_cursor.fetchall.return_value = [
            ("claude", 1200.0, '{"security": 1300}', 10, 5, 2, 17, 8, 10, "2025-01-01"),
        ]

        def factory(row):
            return {"name": row[0], "elo": row[1]}

        engine = LeaderboardEngine(mock_db, rating_factory=factory)

        result = engine.get_leaderboard(limit=10, domain="security")

        assert len(result) == 1
        # Check SQL used domain
        sql = mock_cursor.execute.call_args[0][0]
        assert "json_extract" in sql
        assert "domain_elos" in sql

    def test_get_leaderboard_empty(self):
        """Test getting empty leaderboard."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        mock_db.connection.return_value = mock_conn

        mock_cursor.fetchall.return_value = []

        def factory(row):
            return {"name": row[0], "elo": row[1]}

        engine = LeaderboardEngine(mock_db, rating_factory=factory)

        result = engine.get_leaderboard()
        assert result == []


class TestGetCachedLeaderboard:
    """Test get_cached_leaderboard method."""

    def test_cached_leaderboard_no_cache(self):
        """Test cached leaderboard without cache falls back to direct query."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        mock_db.connection.return_value = mock_conn
        mock_cursor.fetchall.return_value = []

        def factory(row):
            return {"name": row[0]}

        engine = LeaderboardEngine(mock_db, rating_factory=factory)

        result = engine.get_cached_leaderboard()
        assert result == []
        # Should have queried database
        mock_cursor.execute.assert_called()

    def test_cached_leaderboard_cache_hit(self):
        """Test cached leaderboard returns cached data."""
        mock_db = MagicMock()
        mock_cache = MagicMock()
        cached_data = [{"name": "cached_agent", "elo": 1500}]
        mock_cache.get.return_value = cached_data

        def factory(row):
            return {"name": row[0]}

        engine = LeaderboardEngine(
            mock_db,
            rating_factory=factory,
            leaderboard_cache=mock_cache,
        )

        result = engine.get_cached_leaderboard(limit=10)

        assert result == cached_data
        mock_cache.get.assert_called_once_with("leaderboard:10:global")
        # Should NOT have queried database
        mock_db.connection.assert_not_called()

    def test_cached_leaderboard_cache_miss(self):
        """Test cached leaderboard populates cache on miss."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        mock_db.connection.return_value = mock_conn
        mock_cursor.fetchall.return_value = [
            ("claude", 1200.0, "{}", 10, 5, 2, 17, 8, 10, "2025-01-01"),
        ]

        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # Cache miss

        def factory(row):
            return {"name": row[0], "elo": row[1]}

        engine = LeaderboardEngine(
            mock_db,
            rating_factory=factory,
            leaderboard_cache=mock_cache,
        )

        result = engine.get_cached_leaderboard(limit=10)

        assert len(result) == 1
        mock_cache.get.assert_called_once()
        mock_cache.set.assert_called_once()

    def test_cached_leaderboard_domain_key(self):
        """Test cache key includes domain."""
        mock_db = MagicMock()
        mock_cache = MagicMock()
        mock_cache.get.return_value = []

        def factory(row):
            return {"name": row[0]}

        engine = LeaderboardEngine(
            mock_db,
            rating_factory=factory,
            leaderboard_cache=mock_cache,
        )

        engine.get_cached_leaderboard(limit=5, domain="security")
        mock_cache.get.assert_called_with("leaderboard:5:security")


class TestInvalidateCache:
    """Test cache invalidation methods."""

    def test_invalidate_leaderboard_cache_no_caches(self):
        """Test invalidation with no caches."""
        mock_db = MagicMock()
        engine = LeaderboardEngine(mock_db)

        count = engine.invalidate_leaderboard_cache()
        assert count == 0

    def test_invalidate_leaderboard_cache_with_caches(self):
        """Test invalidation clears all caches."""
        mock_db = MagicMock()
        mock_lb_cache = MagicMock()
        mock_lb_cache.clear.return_value = 5
        mock_stats_cache = MagicMock()

        engine = LeaderboardEngine(
            mock_db,
            leaderboard_cache=mock_lb_cache,
            stats_cache=mock_stats_cache,
        )

        count = engine.invalidate_leaderboard_cache()

        assert count == 5
        mock_lb_cache.clear.assert_called_once()
        mock_stats_cache.clear.assert_called_once()

    def test_invalidate_rating_cache_no_cache(self):
        """Test rating cache invalidation with no cache."""
        mock_db = MagicMock()
        engine = LeaderboardEngine(mock_db)

        count = engine.invalidate_rating_cache()
        assert count == 0

    def test_invalidate_rating_cache_specific_agent(self):
        """Test invalidating specific agent's rating cache."""
        mock_db = MagicMock()
        mock_rating_cache = MagicMock()
        mock_rating_cache.invalidate.return_value = True

        engine = LeaderboardEngine(mock_db, rating_cache=mock_rating_cache)

        count = engine.invalidate_rating_cache("claude")

        assert count == 1
        mock_rating_cache.invalidate.assert_called_once_with("rating:claude")

    def test_invalidate_rating_cache_all(self):
        """Test invalidating all rating cache."""
        mock_db = MagicMock()
        mock_rating_cache = MagicMock()
        mock_rating_cache.clear.return_value = 10

        engine = LeaderboardEngine(mock_db, rating_cache=mock_rating_cache)

        count = engine.invalidate_rating_cache()

        assert count == 10
        mock_rating_cache.clear.assert_called_once()


class TestGetTopAgentsForDomain:
    """Test get_top_agents_for_domain method."""

    def test_delegates_to_cached_leaderboard(self):
        """Test it delegates to get_cached_leaderboard with domain."""
        mock_db = MagicMock()
        mock_cache = MagicMock()
        mock_cache.get.return_value = [{"name": "top_agent"}]

        def factory(row):
            return {"name": row[0]}

        engine = LeaderboardEngine(
            mock_db,
            rating_factory=factory,
            leaderboard_cache=mock_cache,
        )

        result = engine.get_top_agents_for_domain("security", limit=3)

        assert result == [{"name": "top_agent"}]
        mock_cache.get.assert_called_with("leaderboard:3:security")


class TestGetEloHistory:
    """Test get_elo_history method."""

    def test_get_history(self):
        """Test getting ELO history."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        mock_db.connection.return_value = mock_conn

        mock_cursor.fetchall.return_value = [
            ("2025-01-03", 1250.0),
            ("2025-01-02", 1220.0),
            ("2025-01-01", 1200.0),
        ]

        engine = LeaderboardEngine(mock_db)
        history = engine.get_elo_history("claude", limit=10)

        assert len(history) == 3
        assert history[0] == ("2025-01-03", 1250.0)
        assert history[2] == ("2025-01-01", 1200.0)

    def test_get_history_empty(self):
        """Test getting history for agent with no history."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        mock_db.connection.return_value = mock_conn

        mock_cursor.fetchall.return_value = []

        engine = LeaderboardEngine(mock_db)
        history = engine.get_elo_history("unknown")

        assert history == []


class TestGetRecentMatches:
    """Test get_recent_matches method."""

    def test_get_recent_matches(self):
        """Test getting recent matches."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        mock_db.connection.return_value = mock_conn

        mock_cursor.fetchall.return_value = [
            (
                "d1",
                "claude",
                '["claude", "gpt"]',
                "general",
                '{"claude": 15, "gpt": -15}',
                "2025-01-01",
            ),
        ]

        engine = LeaderboardEngine(mock_db)
        matches = engine.get_recent_matches(limit=5)

        assert len(matches) == 1
        assert matches[0]["debate_id"] == "d1"
        assert matches[0]["winner"] == "claude"
        assert matches[0]["participants"] == ["claude", "gpt"]
        assert matches[0]["domain"] == "general"
        assert matches[0]["elo_changes"] == {"claude": 15, "gpt": -15}

    def test_get_recent_matches_invalid_json(self):
        """Test handling of invalid JSON in matches."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        mock_db.connection.return_value = mock_conn

        mock_cursor.fetchall.return_value = [
            ("d1", "claude", "invalid json", "general", "invalid json", "2025-01-01"),
        ]

        engine = LeaderboardEngine(mock_db)
        matches = engine.get_recent_matches()

        assert len(matches) == 1
        # Should use defaults for invalid JSON
        assert matches[0]["participants"] == []
        assert matches[0]["elo_changes"] == {}

    def test_get_recent_matches_empty(self):
        """Test getting matches when none exist."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        mock_db.connection.return_value = mock_conn

        mock_cursor.fetchall.return_value = []

        engine = LeaderboardEngine(mock_db)
        matches = engine.get_recent_matches()

        assert matches == []


class TestGetHeadToHead:
    """Test get_head_to_head method."""

    def test_get_head_to_head(self):
        """Test getting head-to-head stats."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        mock_db.connection.return_value = mock_conn

        mock_cursor.fetchall.return_value = [
            ("claude", "{}"),
            ("claude", "{}"),
            ("gpt", "{}"),
            (None, "{}"),  # Draw
        ]

        engine = LeaderboardEngine(mock_db)
        result = engine.get_head_to_head("claude", "gpt")

        assert result["matches"] == 4
        assert result["claude_wins"] == 2
        assert result["gpt_wins"] == 1
        assert result["draws"] == 1

    def test_get_head_to_head_no_matches(self):
        """Test head-to-head with no matches."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        mock_db.connection.return_value = mock_conn

        mock_cursor.fetchall.return_value = []

        engine = LeaderboardEngine(mock_db)
        result = engine.get_head_to_head("claude", "gpt")

        assert result["matches"] == 0
        assert result["claude_wins"] == 0
        assert result["gpt_wins"] == 0
        assert result["draws"] == 0

    def test_get_head_to_head_validates_names(self):
        """Test that agent names are validated."""
        mock_db = MagicMock()
        engine = LeaderboardEngine(mock_db)

        long_name = "a" * (MAX_AGENT_NAME_LENGTH + 1)
        with pytest.raises(ValueError):
            engine.get_head_to_head(long_name, "gpt")


class TestGetStats:
    """Test get_stats method."""

    def test_get_stats(self):
        """Test getting system statistics."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        mock_db.connection.return_value = mock_conn

        mock_cursor.fetchone.side_effect = [
            (10, 1150.0, 1300.0, 1000.0),  # ratings stats
            (50,),  # match count
        ]

        engine = LeaderboardEngine(mock_db)
        stats = engine.get_stats(use_cache=False)

        assert stats["total_agents"] == 10
        assert stats["avg_elo"] == 1150.0
        assert stats["max_elo"] == 1300.0
        assert stats["min_elo"] == 1000.0
        assert stats["total_matches"] == 50

    def test_get_stats_empty_database(self):
        """Test stats for empty database."""
        from aragora.config import ELO_INITIAL_RATING

        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        mock_db.connection.return_value = mock_conn

        mock_cursor.fetchone.side_effect = [
            (0, None, None, None),  # Empty ratings
            (0,),  # No matches
        ]

        engine = LeaderboardEngine(mock_db)
        stats = engine.get_stats(use_cache=False)

        assert stats["total_agents"] == 0
        assert stats["total_matches"] == 0
        # Should use defaults for ELO values
        assert stats["avg_elo"] == ELO_INITIAL_RATING  # DEFAULT_ELO from config

    def test_get_stats_cached(self):
        """Test stats are cached."""
        mock_db = MagicMock()
        mock_cache = MagicMock()
        cached_stats = {"total_agents": 5, "avg_elo": 1200}
        mock_cache.get.return_value = cached_stats

        engine = LeaderboardEngine(mock_db, stats_cache=mock_cache)
        stats = engine.get_stats(use_cache=True)

        assert stats == cached_stats
        mock_cache.get.assert_called_once_with("elo_stats")
        mock_db.connection.assert_not_called()

    def test_get_stats_cache_miss(self):
        """Test stats cache miss populates cache."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        mock_db.connection.return_value = mock_conn

        mock_cursor.fetchone.side_effect = [
            (10, 1150.0, 1300.0, 1000.0),
            (50,),
        ]

        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        engine = LeaderboardEngine(mock_db, stats_cache=mock_cache)
        stats = engine.get_stats(use_cache=True)

        mock_cache.get.assert_called_once()
        mock_cache.set.assert_called_once()
        assert stats["total_agents"] == 10

    def test_get_stats_bypass_cache(self):
        """Test bypassing cache with use_cache=False."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        mock_db.connection.return_value = mock_conn

        mock_cursor.fetchone.side_effect = [
            (10, 1150.0, 1300.0, 1000.0),
            (50,),
        ]

        mock_cache = MagicMock()

        engine = LeaderboardEngine(mock_db, stats_cache=mock_cache)
        stats = engine.get_stats(use_cache=False)

        # Should NOT check cache
        mock_cache.get.assert_not_called()
        # But should still store result
        mock_cache.set.assert_called_once()


class TestLeaderboardEngineIntegration:
    """Integration tests that require multiple components."""

    def test_full_workflow(self):
        """Test typical workflow: get leaderboard, stats, history."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        mock_db.connection.return_value = mock_conn

        # Setup responses for different queries
        def execute_handler(query, params=None):
            if "FROM ratings" in query and "AVG" in query:
                return None  # For stats query
            return None

        mock_cursor.execute.side_effect = execute_handler

        # Leaderboard query
        mock_cursor.fetchall.return_value = [
            ("claude", 1200.0, "{}", 10, 5, 2, 17, 8, 10, "2025-01-01"),
        ]
        mock_cursor.fetchone.side_effect = [
            (5, 1100.0, 1200.0, 1000.0),  # ratings stats
            (20,),  # match count
        ]

        mock_lb_cache = MagicMock()
        mock_lb_cache.get.return_value = None
        mock_stats_cache = MagicMock()
        mock_stats_cache.get.return_value = None

        def factory(row):
            return {"name": row[0], "elo": row[1]}

        engine = LeaderboardEngine(
            mock_db,
            rating_factory=factory,
            leaderboard_cache=mock_lb_cache,
            stats_cache=mock_stats_cache,
        )

        # Should be able to call multiple methods
        leaderboard = engine.get_cached_leaderboard(limit=10)
        assert len(leaderboard) == 1

        stats = engine.get_stats()
        assert "total_agents" in stats
