"""
Tests for PostgreSQL Continuum Memory implementation.

These tests verify the PostgresContinuumMemory class provides the same
functionality as the SQLite-based ContinuumMemory.
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
class TestPostgresContinuumSchema:
    """Tests for schema definitions (no asyncpg required)."""

    def test_schema_module_imports(self):
        """Module should import without errors."""
        from aragora.memory.postgres_continuum import (
            POSTGRES_CONTINUUM_SCHEMA,
            POSTGRES_CONTINUUM_SCHEMA_VERSION,
        )

        assert POSTGRES_CONTINUUM_SCHEMA_VERSION >= 1
        assert len(POSTGRES_CONTINUUM_SCHEMA) > 100

    def test_schema_has_required_tables(self):
        """Schema should define all required tables."""
        from aragora.memory.postgres_continuum import POSTGRES_CONTINUUM_SCHEMA

        required_tables = [
            "continuum_memory",
            "meta_learning_state",
            "tier_transitions",
            "continuum_memory_archive",
        ]

        for table in required_tables:
            assert table in POSTGRES_CONTINUUM_SCHEMA, f"Missing table: {table}"

    def test_schema_has_required_indexes(self):
        """Schema should define performance indexes."""
        from aragora.memory.postgres_continuum import POSTGRES_CONTINUUM_SCHEMA

        required_indexes = [
            "idx_continuum_tier",
            "idx_continuum_surprise",
            "idx_continuum_importance",
            "idx_continuum_tier_updated",
            "idx_continuum_expires",
            "idx_continuum_red_line",
        ]

        for index in required_indexes:
            assert index in POSTGRES_CONTINUUM_SCHEMA, f"Missing index: {index}"

    def test_schema_uses_jsonb_for_metadata(self):
        """Schema should use JSONB for metadata."""
        from aragora.memory.postgres_continuum import POSTGRES_CONTINUUM_SCHEMA

        assert "JSONB" in POSTGRES_CONTINUUM_SCHEMA

    def test_schema_uses_timestamptz(self):
        """Schema should use TIMESTAMPTZ for timestamps."""
        from aragora.memory.postgres_continuum import POSTGRES_CONTINUUM_SCHEMA

        assert "TIMESTAMPTZ" in POSTGRES_CONTINUUM_SCHEMA

    def test_schema_has_foreign_key_for_transitions(self):
        """Schema should have foreign key from tier_transitions to continuum_memory."""
        from aragora.memory.postgres_continuum import POSTGRES_CONTINUUM_SCHEMA

        assert "REFERENCES continuum_memory" in POSTGRES_CONTINUUM_SCHEMA


# Skip remaining tests if asyncpg not available
@pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg required")
class TestPostgresContinuumMemory:
    """Tests for PostgresContinuumMemory class."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock connection pool."""
        pool = MagicMock()
        return pool

    @pytest.fixture
    def mock_connection(self):
        """Create a mock database connection."""
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")
        conn.fetch = AsyncMock(return_value=[])
        conn.fetchrow = AsyncMock(return_value=None)
        return conn

    @pytest.fixture
    def mock_db(self, mock_pool, mock_connection):
        """Create a mocked PostgresContinuumMemory instance."""
        from aragora.memory.postgres_continuum import PostgresContinuumMemory

        db = PostgresContinuumMemory(mock_pool)

        # Create async context manager for connection
        @asynccontextmanager
        async def mock_connection_ctx():
            yield mock_connection

        # Create async context manager for transaction
        @asynccontextmanager
        async def mock_transaction_ctx():
            yield mock_connection

        # Patch the connection and transaction methods
        db.connection = mock_connection_ctx
        db.transaction = mock_transaction_ctx
        return db, mock_connection

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing(self, mock_db):
        """get should return None for non-existent memory."""
        db, mock_conn = mock_db
        mock_conn.fetchrow.return_value = None

        result = await db.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_entry_for_existing(self, mock_db):
        """get should return entry dict for existing memory."""
        db, mock_conn = mock_db
        from datetime import datetime

        now = datetime.now()
        mock_row = [
            "memory-123",  # id
            "slow",  # tier
            "Test content",  # content
            0.8,  # importance
            0.2,  # surprise_score
            0.5,  # consolidation_score
            10,  # update_count
            7,  # success_count
            3,  # failure_count
            now,  # created_at
            now,  # updated_at
            {},  # metadata
            False,  # red_line
            "",  # red_line_reason
        ]
        mock_conn.fetchrow.return_value = mock_row

        result = await db.get("memory-123")
        assert result is not None
        assert result["id"] == "memory-123"
        assert result["tier"] == "slow"
        assert result["importance"] == 0.8

    @pytest.mark.asyncio
    async def test_add_inserts_new_memory(self, mock_db):
        """add should insert a new memory entry."""
        db, mock_conn = mock_db

        result = await db.add(
            memory_id="new_memory",
            content="Test pattern",
            tier="fast",
            importance=0.9,
        )

        mock_conn.execute.assert_called_once()
        assert result["id"] == "new_memory"
        assert result["content"] == "Test pattern"
        assert result["tier"] == "fast"
        assert result["importance"] == 0.9

    @pytest.mark.asyncio
    async def test_update_modifies_fields(self, mock_db):
        """update should modify specified fields."""
        db, mock_conn = mock_db

        result = await db.update(
            memory_id="memory-123",
            importance=0.95,
            surprise_score=0.3,
        )

        assert result is True
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        assert "UPDATE continuum_memory" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_retrieve_with_tier_filter(self, mock_db):
        """retrieve should filter by tier."""
        db, mock_conn = mock_db
        from datetime import datetime

        now = datetime.now()
        mock_rows = [
            [
                "memory-1",
                "fast",
                "Content 1",
                0.9,
                0.1,
                0.5,
                10,
                7,
                3,
                now,
                now,
                {},
                False,
                "",
                0.8,
            ],
        ]
        mock_conn.fetch.return_value = mock_rows

        result = await db.retrieve(tier="fast", limit=10)

        assert len(result) == 1
        mock_conn.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_with_keyword_query(self, mock_db):
        """retrieve should filter by keywords."""
        db, mock_conn = mock_db
        mock_conn.fetch.return_value = []

        await db.retrieve(query="error pattern", limit=10)

        mock_conn.fetch.assert_called_once()
        call_args = mock_conn.fetch.call_args
        assert "LIKE" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_update_outcome_success(self, mock_db):
        """update_outcome should update success count and surprise."""
        db, mock_conn = mock_db
        mock_conn.fetchrow.return_value = {
            "success_count": 5,
            "failure_count": 3,
            "surprise_score": 0.2,
            "tier": "slow",
        }

        result = await db.update_outcome("memory-123", success=True)

        assert result > 0
        mock_conn.execute.assert_called()
        call_args = mock_conn.execute.call_args
        assert "success_count = success_count + 1" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_update_outcome_failure(self, mock_db):
        """update_outcome should update failure count on failure."""
        db, mock_conn = mock_db
        mock_conn.fetchrow.return_value = {
            "success_count": 5,
            "failure_count": 3,
            "surprise_score": 0.2,
            "tier": "slow",
        }

        result = await db.update_outcome("memory-123", success=False)

        assert result > 0
        mock_conn.execute.assert_called()
        call_args = mock_conn.execute.call_args
        assert "failure_count = failure_count + 1" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_mark_red_line(self, mock_db):
        """mark_red_line should protect memory from deletion."""
        db, mock_conn = mock_db
        mock_conn.fetchrow.return_value = {"tier": "medium"}

        result = await db.mark_red_line(
            memory_id="critical-memory",
            reason="Safety-critical decision",
        )

        assert result is True
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        assert "red_line = TRUE" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_delete_archives_by_default(self, mock_db):
        """delete should archive before deletion by default."""
        db, mock_conn = mock_db
        mock_conn.fetchrow.return_value = {"red_line": False}

        result = await db.delete("memory-123")

        assert result["deleted"] is True
        assert result["archived"] is True
        # Should have called execute for archive and delete
        assert mock_conn.execute.call_count >= 2

    @pytest.mark.asyncio
    async def test_delete_blocked_for_red_line(self, mock_db):
        """delete should block deletion of red-lined memory."""
        db, mock_conn = mock_db
        mock_conn.fetchrow.return_value = {"red_line": True}

        result = await db.delete("critical-memory")

        assert result["deleted"] is False
        assert result["blocked"] is True

    @pytest.mark.asyncio
    async def test_get_stats(self, mock_db):
        """get_stats should return tier statistics."""
        db, mock_conn = mock_db
        mock_conn.fetch.return_value = [
            {"tier": "fast", "count": 100, "avg_importance": 0.7, "avg_surprise": 0.3},
            {"tier": "slow", "count": 500, "avg_importance": 0.5, "avg_surprise": 0.2},
        ]
        mock_conn.fetchrow.return_value = {"count": 600}

        result = await db.get_stats()

        assert result["total_entries"] == 600
        assert "by_tier" in result
        assert "fast" in result["by_tier"]
        assert "slow" in result["by_tier"]

    @pytest.mark.asyncio
    async def test_count_all(self, mock_db):
        """count should return total entry count."""
        db, mock_conn = mock_db
        mock_conn.fetchrow.return_value = {"count": 42}

        result = await db.count()
        assert result == 42

    @pytest.mark.asyncio
    async def test_count_by_tier(self, mock_db):
        """count should filter by tier when specified."""
        db, mock_conn = mock_db
        mock_conn.fetchrow.return_value = {"count": 15}

        from aragora.memory.tier_manager import MemoryTier

        result = await db.count(tier=MemoryTier.FAST)

        assert result == 15
        call_args = mock_conn.fetchrow.call_args
        assert "tier = $1" in call_args[0][0]


@pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg required")
class TestPostgresContinuumTierOperations:
    """Tests for tier promotion/demotion operations."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock connection pool."""
        return MagicMock()

    @pytest.fixture
    def mock_connection(self):
        """Create a mock database connection."""
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")
        conn.fetch = AsyncMock(return_value=[])
        conn.fetchrow = AsyncMock(return_value=None)
        return conn

    @pytest.fixture
    def mock_tier_manager(self):
        """Create a mock tier manager."""
        from aragora.memory.tier_manager import MemoryTier

        tm = MagicMock()
        tm.should_promote.return_value = True
        tm.should_demote.return_value = True
        tm.get_next_tier.return_value = MemoryTier.MEDIUM
        tm.record_promotion = MagicMock()
        tm.record_demotion = MagicMock()
        return tm

    @pytest.fixture
    def mock_db(self, mock_pool, mock_connection, mock_tier_manager):
        """Create a mocked PostgresContinuumMemory with tier manager."""
        from aragora.memory.postgres_continuum import PostgresContinuumMemory

        db = PostgresContinuumMemory(mock_pool, mock_tier_manager)

        @asynccontextmanager
        async def mock_transaction_ctx():
            yield mock_connection

        db.transaction = mock_transaction_ctx
        return db, mock_connection, mock_tier_manager

    @pytest.mark.asyncio
    async def test_promote_updates_tier(self, mock_db):
        """promote should update memory tier to faster tier."""
        db, mock_conn, mock_tm = mock_db
        mock_conn.fetchrow.return_value = {
            "tier": "slow",
            "surprise_score": 0.8,
            "last_promotion_at": None,
        }

        result = await db.promote("memory-123")

        assert result is not None
        mock_conn.execute.assert_called()
        mock_tm.record_promotion.assert_called_once()

    @pytest.mark.asyncio
    async def test_promote_records_transition(self, mock_db):
        """promote should record tier transition."""
        db, mock_conn, mock_tm = mock_db
        mock_conn.fetchrow.return_value = {
            "tier": "slow",
            "surprise_score": 0.8,
            "last_promotion_at": None,
        }

        await db.promote("memory-123")

        # Should insert into tier_transitions
        calls = mock_conn.execute.call_args_list
        transition_call = [c for c in calls if "tier_transitions" in str(c)]
        assert len(transition_call) > 0

    @pytest.mark.asyncio
    async def test_demote_updates_tier(self, mock_db):
        """demote should update memory tier to slower tier."""
        db, mock_conn, mock_tm = mock_db
        from aragora.memory.tier_manager import MemoryTier

        mock_conn.fetchrow.return_value = {
            "tier": "fast",
            "surprise_score": 0.1,
            "update_count": 20,
        }
        mock_tm.get_next_tier.return_value = MemoryTier.MEDIUM

        result = await db.demote("memory-123")

        assert result is not None
        mock_conn.execute.assert_called()
        mock_tm.record_demotion.assert_called_once()


class TestPostgresContinuumFactory:
    """Tests for factory function."""

    @pytest.mark.asyncio
    async def test_factory_raises_without_asyncpg(self):
        """get_postgres_continuum_memory should raise if asyncpg unavailable."""
        from aragora.memory import postgres_continuum

        # Temporarily disable asyncpg
        original = postgres_continuum.ASYNCPG_AVAILABLE
        postgres_continuum.ASYNCPG_AVAILABLE = False

        try:
            with pytest.raises(RuntimeError, match="asyncpg"):
                await postgres_continuum.get_postgres_continuum_memory()
        finally:
            postgres_continuum.ASYNCPG_AVAILABLE = original


class TestContinuumMemoryEntryDict:
    """Tests for ContinuumMemoryEntryDict helper class."""

    def test_success_rate_calculation(self):
        """success_rate should calculate correctly."""
        from aragora.memory.postgres_continuum import ContinuumMemoryEntryDict

        entry = ContinuumMemoryEntryDict(
            {
                "id": "test",
                "success_count": 7,
                "failure_count": 3,
            }
        )

        assert entry.success_rate == 0.7

    def test_success_rate_zero_total(self):
        """success_rate should return 0.5 for zero total."""
        from aragora.memory.postgres_continuum import ContinuumMemoryEntryDict

        entry = ContinuumMemoryEntryDict(
            {
                "id": "test",
                "success_count": 0,
                "failure_count": 0,
            }
        )

        assert entry.success_rate == 0.5

    def test_stability_score(self):
        """stability_score should be inverse of surprise."""
        from aragora.memory.postgres_continuum import ContinuumMemoryEntryDict

        entry = ContinuumMemoryEntryDict(
            {
                "id": "test",
                "surprise_score": 0.3,
            }
        )

        assert entry.stability_score == 0.7

    def test_knowledge_mound_id(self):
        """knowledge_mound_id should have cm_ prefix."""
        from aragora.memory.postgres_continuum import ContinuumMemoryEntryDict

        entry = ContinuumMemoryEntryDict({"id": "memory-123"})

        assert entry.knowledge_mound_id == "cm_memory-123"

    def test_cross_references_from_metadata(self):
        """cross_references should read from metadata."""
        from aragora.memory.postgres_continuum import ContinuumMemoryEntryDict

        entry = ContinuumMemoryEntryDict(
            {
                "id": "test",
                "metadata": {"cross_references": ["ref-1", "ref-2"]},
            }
        )

        assert entry.cross_references == ["ref-1", "ref-2"]

    def test_cross_references_empty_metadata(self):
        """cross_references should return empty list for no refs."""
        from aragora.memory.postgres_continuum import ContinuumMemoryEntryDict

        entry = ContinuumMemoryEntryDict({"id": "test", "metadata": {}})

        assert entry.cross_references == []
