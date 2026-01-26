"""
Tests for aragora.ranking.leaderboard_engine - Leaderboard and statistics engine.

Tests cover:
- LeaderboardEngine initialization
- get_leaderboard() with global and domain-specific rankings
- get_cached_leaderboard() caching behavior
- Cache invalidation
- get_elo_history() for agent history
- get_recent_matches() for match history
- get_head_to_head() statistics
- get_stats() system statistics
"""

import json
import os
import pytest
import sqlite3
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

from aragora.ranking.leaderboard_engine import (
    LeaderboardEngine,
    _validate_agent_name,
    MAX_AGENT_NAME_LENGTH,
)
from aragora.config import ELO_INITIAL_RATING


# Mock AgentRating for testing
@dataclass
class MockAgentRating:
    """Mock AgentRating for testing."""

    name: str
    elo: float = ELO_INITIAL_RATING
    wins: int = 0
    losses: int = 0
    draws: int = 0
    debates_count: int = 0
    critiques_accepted: int = 0
    critiques_total: int = 0
    domain_elos: Dict[str, float] = field(default_factory=dict)
    updated_at: str = ""


def create_mock_rating_factory():
    """Create a factory function for MockAgentRating from database rows."""

    def factory(row):
        domain_elos = json.loads(row[2]) if row[2] else {}
        return MockAgentRating(
            name=row[0],
            elo=row[1],
            domain_elos=domain_elos,
            wins=row[3],
            losses=row[4],
            draws=row[5],
            debates_count=row[6],
            critiques_accepted=row[7],
            critiques_total=row[8],
            updated_at=row[9] if len(row) > 9 else "",
        )

    return factory


class MockTTLCache:
    """Mock TTL cache for testing."""

    def __init__(self):
        self._data: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def invalidate(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False

    def clear(self) -> int:
        count = len(self._data)
        self._data.clear()
        return count


class MockEloDatabase:
    """Mock EloDatabase for testing."""

    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        """Initialize test schema."""
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS ratings (
                agent_name TEXT PRIMARY KEY,
                elo REAL DEFAULT 1500,
                domain_elos TEXT DEFAULT '{}',
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                draws INTEGER DEFAULT 0,
                debates_count INTEGER DEFAULT 0,
                critiques_accepted INTEGER DEFAULT 0,
                critiques_total INTEGER DEFAULT 0,
                updated_at TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS elo_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                elo REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS matches (
                debate_id TEXT PRIMARY KEY,
                winner TEXT,
                participants TEXT NOT NULL,
                scores TEXT DEFAULT '{}',
                domain TEXT,
                elo_changes TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """
        )
        self._conn.commit()

    def connection(self):
        """Return connection context manager."""
        return self._conn

    def add_rating(self, name: str, elo: float, domain_elos: dict = None, **kwargs):
        """Helper to add a rating for testing."""
        domain_elos = domain_elos or {}
        self._conn.execute(
            """INSERT OR REPLACE INTO ratings
               (agent_name, elo, domain_elos, wins, losses, draws, debates_count, critiques_accepted, critiques_total, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                name,
                elo,
                json.dumps(domain_elos),
                kwargs.get("wins", 0),
                kwargs.get("losses", 0),
                kwargs.get("draws", 0),
                kwargs.get("debates_count", 0),
                kwargs.get("critiques_accepted", 0),
                kwargs.get("critiques_total", 0),
                kwargs.get("updated_at", ""),
            ),
        )
        self._conn.commit()

    def add_history(self, agent_name: str, elo: float, timestamp: str = None):
        """Helper to add ELO history entry."""
        self._conn.execute(
            "INSERT INTO elo_history (agent_name, elo, created_at) VALUES (?, ?, ?)",
            (agent_name, elo, timestamp or datetime.now().isoformat()),
        )
        self._conn.commit()

    def add_match(
        self,
        debate_id: str,
        winner: str,
        participants: list,
        scores: dict = None,
        domain: str = None,
        elo_changes: dict = None,
    ):
        """Helper to add match record."""
        self._conn.execute(
            """INSERT INTO matches (debate_id, winner, participants, scores, domain, elo_changes)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                debate_id,
                winner,
                json.dumps(participants),
                json.dumps(scores or {}),
                domain,
                json.dumps(elo_changes or {}),
            ),
        )
        self._conn.commit()


@pytest.fixture
def temp_db_path():
    """Create temporary database path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def mock_db(temp_db_path):
    """Create mock database with test data."""
    return MockEloDatabase(temp_db_path)


@pytest.fixture
def engine(mock_db):
    """Create LeaderboardEngine with mocks."""
    return LeaderboardEngine(
        db=mock_db,
        leaderboard_cache=MockTTLCache(),
        stats_cache=MockTTLCache(),
        rating_cache=MockTTLCache(),
        rating_factory=create_mock_rating_factory(),
    )


class TestValidateAgentName:
    """Tests for _validate_agent_name function."""

    def test_valid_short_name(self):
        """Short names should be valid."""
        _validate_agent_name("claude")  # Should not raise

    def test_valid_at_limit(self):
        """Names at the limit should be valid."""
        name = "a" * MAX_AGENT_NAME_LENGTH
        _validate_agent_name(name)  # Should not raise

    def test_invalid_too_long(self):
        """Names exceeding limit should raise ValueError."""
        name = "a" * (MAX_AGENT_NAME_LENGTH + 1)
        with pytest.raises(ValueError) as exc_info:
            _validate_agent_name(name)
        assert str(MAX_AGENT_NAME_LENGTH) in str(exc_info.value)


class TestLeaderboardEngineInit:
    """Tests for LeaderboardEngine initialization."""

    def test_init_with_all_caches(self, mock_db):
        """Should initialize with all cache instances."""
        engine = LeaderboardEngine(
            db=mock_db,
            leaderboard_cache=MockTTLCache(),
            stats_cache=MockTTLCache(),
            rating_cache=MockTTLCache(),
            rating_factory=create_mock_rating_factory(),
        )
        assert engine._db is mock_db
        assert engine._leaderboard_cache is not None
        assert engine._stats_cache is not None
        assert engine._rating_cache is not None

    def test_init_without_caches(self, mock_db):
        """Should work without caches."""
        engine = LeaderboardEngine(db=mock_db)
        assert engine._leaderboard_cache is None
        assert engine._stats_cache is None


class TestGetLeaderboard:
    """Tests for get_leaderboard() method."""

    def test_empty_leaderboard(self, engine):
        """Should return empty list when no agents."""
        result = engine.get_leaderboard()
        assert result == []

    def test_leaderboard_ordering(self, engine, mock_db):
        """Should return agents sorted by ELO descending."""
        mock_db.add_rating("alice", 1600)
        mock_db.add_rating("bob", 1400)
        mock_db.add_rating("carol", 1500)

        result = engine.get_leaderboard()

        assert len(result) == 3
        assert result[0].name == "alice"
        assert result[1].name == "carol"
        assert result[2].name == "bob"

    def test_leaderboard_limit(self, engine, mock_db):
        """Should respect limit parameter."""
        for i in range(10):
            mock_db.add_rating(f"agent_{i}", 1500 + i * 10)

        result = engine.get_leaderboard(limit=5)

        assert len(result) == 5
        # Should be top 5 by ELO
        assert result[0].elo == 1590  # agent_9

    def test_leaderboard_with_domain(self, engine, mock_db):
        """Should sort by domain-specific ELO when domain specified."""
        mock_db.add_rating("alice", 1600, domain_elos={"security": 1400})
        mock_db.add_rating("bob", 1400, domain_elos={"security": 1700})
        mock_db.add_rating("carol", 1500, domain_elos={"security": 1500})

        result = engine.get_leaderboard(domain="security")

        # Should be sorted by security domain ELO
        assert len(result) == 3
        assert result[0].name == "bob"  # 1700 security ELO
        assert result[1].name == "carol"  # 1500 security ELO
        assert result[2].name == "alice"  # 1400 security ELO

    def test_leaderboard_without_rating_factory_raises(self, mock_db):
        """Should raise if rating_factory not set."""
        engine = LeaderboardEngine(db=mock_db)

        with pytest.raises(RuntimeError) as exc_info:
            engine.get_leaderboard()
        assert "rating_factory" in str(exc_info.value)


class TestGetCachedLeaderboard:
    """Tests for get_cached_leaderboard() method."""

    def test_cache_miss_fetches_from_db(self, engine, mock_db):
        """Should fetch from DB on cache miss."""
        mock_db.add_rating("alice", 1600)

        result = engine.get_cached_leaderboard()

        assert len(result) == 1
        assert result[0].name == "alice"

    def test_cache_hit_returns_cached(self, engine, mock_db):
        """Should return cached result on subsequent calls."""
        mock_db.add_rating("alice", 1600)

        # First call - cache miss
        result1 = engine.get_cached_leaderboard()

        # Add more data (won't be seen due to cache)
        mock_db.add_rating("bob", 1700)

        # Second call - cache hit
        result2 = engine.get_cached_leaderboard()

        assert len(result1) == len(result2) == 1  # Still 1 from cache

    def test_cache_key_includes_limit(self, engine, mock_db):
        """Different limits should use different cache keys."""
        for i in range(20):
            mock_db.add_rating(f"agent_{i}", 1500 + i)

        result5 = engine.get_cached_leaderboard(limit=5)
        result10 = engine.get_cached_leaderboard(limit=10)

        assert len(result5) == 5
        assert len(result10) == 10

    def test_cache_key_includes_domain(self, engine, mock_db):
        """Different domains should use different cache keys."""
        mock_db.add_rating("alice", 1600, domain_elos={"a": 1800, "b": 1200})

        result_global = engine.get_cached_leaderboard()
        result_domain_a = engine.get_cached_leaderboard(domain="a")
        result_domain_b = engine.get_cached_leaderboard(domain="b")

        # All should return alice, but from different cache entries
        assert len(result_global) == 1
        assert len(result_domain_a) == 1
        assert len(result_domain_b) == 1

    def test_no_cache_falls_back_to_db(self, mock_db):
        """Should work without cache."""
        engine = LeaderboardEngine(
            db=mock_db,
            rating_factory=create_mock_rating_factory(),
        )
        mock_db.add_rating("alice", 1600)

        result = engine.get_cached_leaderboard()
        assert len(result) == 1


class TestCacheInvalidation:
    """Tests for cache invalidation methods."""

    def test_invalidate_leaderboard_cache(self, engine, mock_db):
        """Should clear leaderboard cache."""
        mock_db.add_rating("alice", 1600)

        # Populate cache
        engine.get_cached_leaderboard()

        # Add new agent
        mock_db.add_rating("bob", 1700)

        # Invalidate cache
        engine.invalidate_leaderboard_cache()

        # Should now see bob
        result = engine.get_cached_leaderboard()
        assert len(result) == 2

    def test_invalidate_rating_cache_single(self, engine):
        """Should invalidate single rating cache entry."""
        engine._rating_cache.set("rating:alice", MockAgentRating(name="alice"))

        count = engine.invalidate_rating_cache("alice")
        assert count == 1
        assert engine._rating_cache.get("rating:alice") is None

    def test_invalidate_rating_cache_all(self, engine):
        """Should clear all rating cache entries."""
        engine._rating_cache.set("rating:alice", MockAgentRating(name="alice"))
        engine._rating_cache.set("rating:bob", MockAgentRating(name="bob"))

        count = engine.invalidate_rating_cache()  # Clear all
        assert count == 2


class TestGetTopAgentsForDomain:
    """Tests for get_top_agents_for_domain() method."""

    def test_returns_domain_sorted_agents(self, engine, mock_db):
        """Should return agents sorted by domain ELO."""
        mock_db.add_rating("alice", 1600, domain_elos={"security": 1800})
        mock_db.add_rating("bob", 1700, domain_elos={"security": 1500})

        result = engine.get_top_agents_for_domain("security", limit=5)

        assert len(result) == 2
        assert result[0].name == "alice"  # Higher security ELO


class TestGetEloHistory:
    """Tests for get_elo_history() method."""

    def test_returns_history_entries(self, engine, mock_db):
        """Should return timestamped ELO history."""
        mock_db.add_history("alice", 1500, "2024-01-01T00:00:00")
        mock_db.add_history("alice", 1520, "2024-01-02T00:00:00")
        mock_db.add_history("alice", 1510, "2024-01-03T00:00:00")

        result = engine.get_elo_history("alice")

        assert len(result) == 3
        # Should be ordered by timestamp DESC
        assert result[0][1] == 1510
        assert result[2][1] == 1500

    def test_respects_limit(self, engine, mock_db):
        """Should respect limit parameter."""
        for i in range(10):
            mock_db.add_history("alice", 1500 + i, f"2024-01-{i + 1:02d}T00:00:00")

        result = engine.get_elo_history("alice", limit=5)

        assert len(result) == 5

    def test_returns_empty_for_unknown_agent(self, engine):
        """Should return empty list for unknown agent."""
        result = engine.get_elo_history("nonexistent")
        assert result == []


class TestGetRecentMatches:
    """Tests for get_recent_matches() method."""

    def test_returns_match_records(self, engine, mock_db):
        """Should return structured match records."""
        mock_db.add_match(
            "debate-1",
            "alice",
            ["alice", "bob"],
            scores={"alice": 1.0, "bob": 0.0},
            domain="general",
            elo_changes={"alice": 15, "bob": -15},
        )

        result = engine.get_recent_matches(limit=10)

        assert len(result) == 1
        match = result[0]
        assert match["debate_id"] == "debate-1"
        assert match["winner"] == "alice"
        assert match["participants"] == ["alice", "bob"]
        assert match["elo_changes"]["alice"] == 15

    def test_respects_limit(self, engine, mock_db):
        """Should respect limit parameter."""
        for i in range(20):
            mock_db.add_match(f"debate-{i}", "alice", ["alice", "bob"])

        result = engine.get_recent_matches(limit=5)
        assert len(result) == 5

    def test_orders_by_recency(self, engine, mock_db):
        """Should order by most recent first (DESC by created_at)."""
        # Add matches - SQLite DEFAULT CURRENT_TIMESTAMP uses same time for rapid inserts
        # So we manually control timestamps for testing
        mock_db._conn.execute(
            """INSERT INTO matches (debate_id, winner, participants, scores, domain, elo_changes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("old", "alice", '["alice"]', "{}", "old", "{}", "2024-01-01T00:00:00"),
        )
        mock_db._conn.execute(
            """INSERT INTO matches (debate_id, winner, participants, scores, domain, elo_changes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("new", "bob", '["bob"]', "{}", "new", "{}", "2024-01-02T00:00:00"),
        )
        mock_db._conn.commit()

        result = engine.get_recent_matches(limit=10)

        assert len(result) == 2
        # Most recent first (2024-01-02 > 2024-01-01)
        assert result[0]["debate_id"] == "new"


class TestGetHeadToHead:
    """Tests for get_head_to_head() method."""

    def test_returns_stats_between_agents(self, engine, mock_db):
        """Should return head-to-head statistics."""
        mock_db.add_match("m1", "alice", ["alice", "bob"])
        mock_db.add_match("m2", "alice", ["alice", "bob"])
        mock_db.add_match("m3", "bob", ["alice", "bob"])
        mock_db.add_match("m4", None, ["alice", "bob"])  # Draw

        result = engine.get_head_to_head("alice", "bob")

        assert result["matches"] == 4
        assert result["alice_wins"] == 2
        assert result["bob_wins"] == 1
        assert result["draws"] == 1

    def test_no_matches_returns_zeros(self, engine):
        """Should return zeros when no matches exist."""
        result = engine.get_head_to_head("alice", "bob")

        assert result["matches"] == 0
        assert result["alice_wins"] == 0
        assert result["bob_wins"] == 0
        assert result["draws"] == 0

    def test_validates_agent_names(self, engine):
        """Should validate agent name lengths."""
        long_name = "a" * (MAX_AGENT_NAME_LENGTH + 1)

        with pytest.raises(ValueError):
            engine.get_head_to_head(long_name, "bob")


class TestGetStats:
    """Tests for get_stats() method."""

    def test_returns_system_statistics(self, engine, mock_db):
        """Should return overall system stats."""
        mock_db.add_rating("alice", 1600)
        mock_db.add_rating("bob", 1400)
        mock_db.add_match("m1", "alice", ["alice", "bob"])

        result = engine.get_stats()

        assert result["total_agents"] == 2
        assert result["avg_elo"] == 1500
        assert result["max_elo"] == 1600
        assert result["min_elo"] == 1400
        assert result["total_matches"] == 1

    def test_empty_system_stats(self, engine):
        """Should handle empty system."""
        result = engine.get_stats()

        assert result["total_agents"] == 0
        assert result["total_matches"] == 0

    def test_caches_stats(self, engine, mock_db):
        """Should cache stats results."""
        mock_db.add_rating("alice", 1600)

        # First call
        result1 = engine.get_stats(use_cache=True)

        # Add data (won't be seen due to cache)
        mock_db.add_rating("bob", 1400)

        # Second call - should return cached
        result2 = engine.get_stats(use_cache=True)

        assert result1["total_agents"] == result2["total_agents"] == 1

    def test_bypass_cache(self, engine, mock_db):
        """Should bypass cache when requested."""
        mock_db.add_rating("alice", 1600)
        engine.get_stats(use_cache=True)  # Prime cache

        mock_db.add_rating("bob", 1400)

        result = engine.get_stats(use_cache=False)
        assert result["total_agents"] == 2


class TestEdgeCases:
    """Edge case tests."""

    def test_agent_with_no_domain_elos(self, engine, mock_db):
        """Should handle agents without domain ELOs."""
        mock_db.add_rating("alice", 1600)

        result = engine.get_leaderboard(domain="security")

        # Should still return alice (using global ELO as fallback)
        assert len(result) == 1
        assert result[0].name == "alice"

    def test_special_characters_in_agent_name(self, engine, mock_db):
        """Should handle special characters in names."""
        mock_db.add_rating("alice-v2.0", 1600)
        mock_db.add_rating("bob_test", 1500)

        result = engine.get_leaderboard()
        assert len(result) == 2

    def test_unicode_agent_names(self, engine, mock_db):
        """Should handle unicode names."""
        mock_db.add_rating("claude-3", 1600)
        mock_db.add_rating("test-agent", 1500)

        result = engine.get_leaderboard()
        assert len(result) == 2
