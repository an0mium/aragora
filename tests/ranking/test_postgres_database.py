"""
Tests for PostgreSQL ELO database implementation.

These tests verify the PostgresEloDatabase class provides the same
functionality as the SQLite-based EloDatabase.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import asynccontextmanager
import json


# Check if asyncpg is available
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False


# Tests that don't require asyncpg
class TestPostgresEloSchema:
    """Tests for schema definitions (no asyncpg required)."""

    def test_schema_module_imports(self):
        """Module should import without errors."""
        from aragora.ranking.postgres_database import (
            POSTGRES_ELO_SCHEMA,
            POSTGRES_ELO_SCHEMA_VERSION,
        )
        assert POSTGRES_ELO_SCHEMA_VERSION >= 1
        assert len(POSTGRES_ELO_SCHEMA) > 100

    def test_schema_has_required_tables(self):
        """Schema should define all required tables."""
        from aragora.ranking.postgres_database import POSTGRES_ELO_SCHEMA

        required_tables = [
            "elo_ratings",
            "elo_matches",
            "elo_history",
            "elo_calibration_predictions",
            "elo_domain_calibration",
            "elo_calibration_buckets",
            "elo_agent_relationships",
        ]

        for table in required_tables:
            assert table in POSTGRES_ELO_SCHEMA, f"Missing table: {table}"

    def test_schema_has_required_indexes(self):
        """Schema should define performance indexes."""
        from aragora.ranking.postgres_database import POSTGRES_ELO_SCHEMA

        required_indexes = [
            "idx_elo_matches_created_at",
            "idx_elo_matches_winner",
            "idx_elo_history_agent",
            "idx_elo_ratings_elo",
        ]

        for index in required_indexes:
            assert index in POSTGRES_ELO_SCHEMA, f"Missing index: {index}"

    def test_schema_uses_jsonb_for_complex_types(self):
        """Schema should use JSONB for JSON data."""
        from aragora.ranking.postgres_database import POSTGRES_ELO_SCHEMA

        assert "JSONB" in POSTGRES_ELO_SCHEMA

    def test_schema_uses_timestamptz(self):
        """Schema should use TIMESTAMPTZ for timestamps."""
        from aragora.ranking.postgres_database import POSTGRES_ELO_SCHEMA

        assert "TIMESTAMPTZ" in POSTGRES_ELO_SCHEMA


# Skip remaining tests if asyncpg not available
@pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg required")


class TestPostgresEloDatabase:
    """Tests for PostgresEloDatabase class."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock connection pool."""
        pool = MagicMock()
        return pool

    @pytest.fixture
    def mock_connection(self):
        """Create a mock database connection."""
        conn = AsyncMock()
        conn.execute = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        conn.fetchrow = AsyncMock(return_value=None)
        return conn

    @pytest.fixture
    def mock_db(self, mock_pool, mock_connection):
        """Create a mocked PostgresEloDatabase instance."""
        from aragora.ranking.postgres_database import PostgresEloDatabase

        db = PostgresEloDatabase(mock_pool)

        # Create async context manager for connection
        @asynccontextmanager
        async def mock_connection_ctx():
            yield mock_connection

        # Patch the connection method
        db.connection = mock_connection_ctx
        return db, mock_connection

    @pytest.mark.asyncio
    async def test_get_rating_returns_none_for_missing(self, mock_db):
        """get_rating should return None for non-existent agent."""
        db, mock_conn = mock_db
        mock_conn.fetchrow.return_value = None

        result = await db.get_rating("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_rating_returns_dict_for_existing(self, mock_db):
        """get_rating should return dict for existing agent."""
        db, mock_conn = mock_db
        mock_row = {
            "agent_name": "claude",
            "elo": 1550.0,
            "wins": 10,
            "losses": 5,
        }
        mock_conn.fetchrow.return_value = mock_row

        result = await db.get_rating("claude")
        assert result == mock_row
        assert result["elo"] == 1550.0

    @pytest.mark.asyncio
    async def test_set_rating_inserts_new_agent(self, mock_db):
        """set_rating should insert a new agent rating."""
        db, mock_conn = mock_db

        await db.set_rating(
            agent_name="new_agent",
            elo=1500.0,
            wins=0,
            losses=0,
        )

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        assert "INSERT INTO elo_ratings" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_update_elo_basic(self, mock_db):
        """update_elo should update agent's ELO rating."""
        db, mock_conn = mock_db

        await db.update_elo("claude", 1600.0)

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        assert "UPDATE elo_ratings" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_update_elo_with_domain(self, mock_db):
        """update_elo should update domain-specific ELO."""
        db, mock_conn = mock_db

        await db.update_elo("claude", 1600.0, domain="coding", domain_elo=1650.0)

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        assert "jsonb_set" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_leaderboard(self, mock_db):
        """get_leaderboard should return top agents."""
        db, mock_conn = mock_db
        mock_rows = [
            {"agent_name": "claude", "elo": 1600.0, "wins": 15},
            {"agent_name": "gpt4", "elo": 1550.0, "wins": 12},
        ]
        mock_conn.fetch.return_value = mock_rows

        result = await db.get_leaderboard(limit=10)

        assert len(result) == 2
        assert result[0]["agent_name"] == "claude"
        assert result[0]["elo"] == 1600.0

    @pytest.mark.asyncio
    async def test_save_match(self, mock_db):
        """save_match should persist match result."""
        db, mock_conn = mock_db
        mock_conn.fetchrow.return_value = {"id": 42}

        result = await db.save_match(
            debate_id="debate-123",
            winner="claude",
            participants=["claude", "gpt4"],
            domain="general",
            scores={"claude": 1.0, "gpt4": 0.0},
            elo_changes={"claude": 16.0, "gpt4": -16.0},
        )

        assert result == 42
        mock_conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_match_parses_json(self, mock_db):
        """get_match should parse JSON fields."""
        db, mock_conn = mock_db
        mock_row = {
            "id": 1,
            "debate_id": "debate-123",
            "winner": "claude",
            "participants": '["claude", "gpt4"]',
            "scores": '{"claude": 1.0, "gpt4": 0.0}',
            "elo_changes": '{"claude": 16.0, "gpt4": -16.0}',
        }
        mock_conn.fetchrow.return_value = mock_row

        result = await db.get_match("debate-123")

        assert result["participants"] == ["claude", "gpt4"]
        assert result["scores"] == {"claude": 1.0, "gpt4": 0.0}
        assert result["elo_changes"] == {"claude": 16.0, "gpt4": -16.0}

    @pytest.mark.asyncio
    async def test_save_elo_history(self, mock_db):
        """save_elo_history should persist ELO progression."""
        db, mock_conn = mock_db

        await db.save_elo_history("claude", 1550.0, "debate-123")

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        assert "INSERT INTO elo_history" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_update_relationship_normalizes_order(self, mock_db):
        """update_relationship should normalize agent order alphabetically."""
        db, mock_conn = mock_db

        # Pass agents in reverse alphabetical order
        await db.update_relationship(
            agent_a="zoe",
            agent_b="alice",
            debate_increment=1,
        )

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        # Should be normalized to alphabetical order
        assert call_args[0][1] == "alice"  # agent_a
        assert call_args[0][2] == "zoe"    # agent_b

    @pytest.mark.asyncio
    async def test_count_ratings(self, mock_db):
        """count_ratings should return total count."""
        db, mock_conn = mock_db
        mock_conn.fetchrow.return_value = {"count": 42}

        result = await db.count_ratings()
        assert result == 42

    @pytest.mark.asyncio
    async def test_delete_rating(self, mock_db):
        """delete_rating should remove agent rating."""
        db, mock_conn = mock_db
        mock_conn.execute.return_value = "DELETE 1"

        result = await db.delete_rating("claude")

        assert result is True
        call_args = mock_conn.execute.call_args
        assert "DELETE FROM elo_ratings" in call_args[0][0]


class TestPostgresEloDatabaseFactory:
    """Tests for factory function."""

    @pytest.mark.asyncio
    async def test_factory_raises_without_asyncpg(self):
        """get_postgres_elo_database should raise if asyncpg unavailable."""
        from aragora.ranking import postgres_database

        # Temporarily disable asyncpg
        original = postgres_database.ASYNCPG_AVAILABLE
        postgres_database.ASYNCPG_AVAILABLE = False

        try:
            with pytest.raises(RuntimeError, match="asyncpg"):
                await postgres_database.get_postgres_elo_database()
        finally:
            postgres_database.ASYNCPG_AVAILABLE = original
