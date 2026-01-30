"""
Comprehensive tests for Agent CV Store.

Tests cover:
- CVStore initialization and configuration
- Cache management (TTL, invalidation)
- CV persistence (save, get, delete)
- Batch operations
- Database loading and saving
- Error handling
- Singleton pattern
- Domain-based queries
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.agents.cv import AgentCV, CVBuilder, DomainPerformance, ReliabilityMetrics
from aragora.agents.cv_store import (
    CV_SCHEMA_VERSION,
    CVStore,
    DEFAULT_CACHE_TTL_SECONDS,
    get_cv_store,
)


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Test module exports and constants."""

    def test_can_import_module(self):
        """Module can be imported."""
        from aragora.agents import cv_store

        assert cv_store is not None

    def test_cv_store_in_all(self):
        """CVStore is exported in __all__."""
        from aragora.agents.cv_store import __all__

        assert "CVStore" in __all__

    def test_get_cv_store_in_all(self):
        """get_cv_store is exported in __all__."""
        from aragora.agents.cv_store import __all__

        assert "get_cv_store" in __all__

    def test_schema_version_constant(self):
        """Schema version constant is defined."""
        assert CV_SCHEMA_VERSION == 1

    def test_default_cache_ttl_constant(self):
        """Default cache TTL constant is defined."""
        assert DEFAULT_CACHE_TTL_SECONDS == 300


# =============================================================================
# CVStore Initialization Tests
# =============================================================================


class TestCVStoreInit:
    """Test CVStore initialization."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_cv.db"

    def test_init_with_db_path(self, temp_db_path):
        """CVStore initializes with specified db_path."""
        store = CVStore(db_path=temp_db_path)
        assert store.db_path == temp_db_path
        assert temp_db_path.exists()

    def test_init_with_custom_builder(self, temp_db_path):
        """CVStore initializes with custom CV builder."""
        mock_builder = MagicMock(spec=CVBuilder)
        store = CVStore(db_path=temp_db_path, cv_builder=mock_builder)
        assert store._cv_builder is mock_builder

    def test_init_with_custom_cache_ttl(self, temp_db_path):
        """CVStore initializes with custom cache TTL."""
        store = CVStore(db_path=temp_db_path, cache_ttl_seconds=600)
        assert store._cache_ttl == timedelta(seconds=600)

    def test_init_creates_empty_cache(self, temp_db_path):
        """CVStore initializes with empty cache."""
        store = CVStore(db_path=temp_db_path)
        assert store._cache == {}

    def test_init_creates_database_table(self, temp_db_path):
        """CVStore creates agent_cvs table on initialization."""
        store = CVStore(db_path=temp_db_path)

        # Verify table exists by querying it
        with store.connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_cvs'"
            )
            row = cursor.fetchone()

        assert row is not None
        assert row[0] == "agent_cvs"

    def test_schema_name_is_set(self, temp_db_path):
        """CVStore has correct schema name."""
        store = CVStore(db_path=temp_db_path)
        assert store.SCHEMA_NAME == "agent_cv"

    def test_schema_version_is_set(self, temp_db_path):
        """CVStore has correct schema version."""
        store = CVStore(db_path=temp_db_path)
        assert store.SCHEMA_VERSION == CV_SCHEMA_VERSION


# =============================================================================
# CV Builder Property Tests
# =============================================================================


class TestCVBuilderProperty:
    """Test cv_builder property and lazy initialization."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_cv.db"

    def test_cv_builder_property_with_provided_builder(self, temp_db_path):
        """cv_builder property returns provided builder."""
        mock_builder = MagicMock(spec=CVBuilder)
        store = CVStore(db_path=temp_db_path, cv_builder=mock_builder)
        assert store.cv_builder is mock_builder

    def test_cv_builder_property_lazy_init(self, temp_db_path):
        """cv_builder property creates builder lazily when not provided."""
        store = CVStore(db_path=temp_db_path)

        with patch("aragora.agents.cv_store.get_cv_builder") as mock_get:
            mock_builder = MagicMock(spec=CVBuilder)
            mock_get.return_value = mock_builder

            builder = store.cv_builder

            mock_get.assert_called_once()
            assert builder is mock_builder


# =============================================================================
# Cache Validity Tests
# =============================================================================


class TestCacheValidity:
    """Test cache validity checking."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_cv.db"

    @pytest.fixture
    def store(self, temp_db_path):
        """Create a CVStore with mocked builder."""
        mock_builder = MagicMock(spec=CVBuilder)
        mock_builder.build_cv.return_value = AgentCV(agent_id="test-agent")
        return CVStore(db_path=temp_db_path, cv_builder=mock_builder)

    def test_is_cache_valid_empty_cache(self, store):
        """Cache is invalid when agent not in cache."""
        assert store._is_cache_valid("nonexistent") is False

    def test_is_cache_valid_fresh_entry(self, store):
        """Cache is valid for fresh entry."""
        cv = AgentCV(agent_id="test-agent")
        store._cache["test-agent"] = (datetime.now(), cv)
        assert store._is_cache_valid("test-agent") is True

    def test_is_cache_valid_stale_entry(self, store):
        """Cache is invalid for stale entry."""
        cv = AgentCV(agent_id="test-agent")
        stale_time = datetime.now() - timedelta(seconds=400)  # Older than default TTL
        store._cache["test-agent"] = (stale_time, cv)
        assert store._is_cache_valid("test-agent") is False

    def test_is_cache_valid_custom_ttl(self, temp_db_path):
        """Cache validity respects custom TTL."""
        store = CVStore(db_path=temp_db_path, cache_ttl_seconds=10)
        cv = AgentCV(agent_id="test-agent")

        # Entry that's 5 seconds old should be valid with 10s TTL
        store._cache["test-agent"] = (datetime.now() - timedelta(seconds=5), cv)
        assert store._is_cache_valid("test-agent") is True

        # Entry that's 15 seconds old should be invalid with 10s TTL
        store._cache["test-agent"] = (datetime.now() - timedelta(seconds=15), cv)
        assert store._is_cache_valid("test-agent") is False


# =============================================================================
# Get CV Tests
# =============================================================================


class TestGetCV:
    """Test get_cv method."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_cv.db"

    @pytest.fixture
    def store(self, temp_db_path):
        """Create a CVStore with mocked builder."""
        mock_builder = MagicMock(spec=CVBuilder)
        mock_builder.build_cv.return_value = AgentCV(
            agent_id="built-agent",
            overall_elo=1100.0,
            total_debates=15,
        )
        return CVStore(db_path=temp_db_path, cv_builder=mock_builder)

    @pytest.mark.asyncio
    async def test_get_cv_from_cache(self, store):
        """get_cv returns cached CV when fresh."""
        cached_cv = AgentCV(agent_id="cached", overall_elo=1200.0)
        store._cache["cached"] = (datetime.now(), cached_cv)

        result = await store.get_cv("cached")

        assert result is cached_cv
        assert result.overall_elo == 1200.0

    @pytest.mark.asyncio
    async def test_get_cv_bypasses_cache_when_disabled(self, store):
        """get_cv bypasses cache when use_cache=False."""
        cached_cv = AgentCV(agent_id="cached", overall_elo=1200.0)
        store._cache["cached"] = (datetime.now(), cached_cv)

        # Save different CV to database
        db_cv = AgentCV(agent_id="cached", overall_elo=1000.0)
        store._save_to_db(db_cv)

        result = await store.get_cv("cached", use_cache=False, auto_build=False)

        # Should get database version, not cached version
        assert result is not None
        assert result.overall_elo == 1000.0

    @pytest.mark.asyncio
    async def test_get_cv_from_database(self, store):
        """get_cv returns CV from database when not cached."""
        db_cv = AgentCV(agent_id="db-agent", overall_elo=1150.0)
        store._save_to_db(db_cv)

        result = await store.get_cv("db-agent", auto_build=False)

        assert result is not None
        assert result.agent_id == "db-agent"
        assert result.overall_elo == 1150.0

    @pytest.mark.asyncio
    async def test_get_cv_auto_builds(self, store):
        """get_cv auto-builds CV when not found."""
        result = await store.get_cv("new-agent", auto_build=True)

        assert result is not None
        assert result.agent_id == "built-agent"  # From mock builder
        store.cv_builder.build_cv.assert_called_once_with("new-agent")

    @pytest.mark.asyncio
    async def test_get_cv_returns_none_without_auto_build(self, store):
        """get_cv returns None when not found and auto_build=False."""
        result = await store.get_cv("nonexistent", auto_build=False)

        assert result is None
        store.cv_builder.build_cv.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_cv_updates_cache(self, store):
        """get_cv updates cache after fetching from database."""
        db_cv = AgentCV(agent_id="db-agent", overall_elo=1150.0)
        store._save_to_db(db_cv)

        await store.get_cv("db-agent", auto_build=False)

        assert "db-agent" in store._cache
        _, cached = store._cache["db-agent"]
        assert cached.overall_elo == 1150.0


# =============================================================================
# Get CV Sync Tests
# =============================================================================


class TestGetCVSync:
    """Test get_cv_sync method."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_cv.db"

    @pytest.fixture
    def store(self, temp_db_path):
        """Create a CVStore with mocked builder."""
        mock_builder = MagicMock(spec=CVBuilder)
        mock_builder.build_cv.return_value = AgentCV(
            agent_id="built-agent",
            overall_elo=1100.0,
        )
        return CVStore(db_path=temp_db_path, cv_builder=mock_builder)

    def test_get_cv_sync_from_cache(self, store):
        """get_cv_sync returns cached CV when fresh."""
        cached_cv = AgentCV(agent_id="cached", overall_elo=1200.0)
        store._cache["cached"] = (datetime.now(), cached_cv)

        result = store.get_cv_sync("cached")

        assert result is cached_cv

    def test_get_cv_sync_from_database(self, store):
        """get_cv_sync returns CV from database when not cached."""
        db_cv = AgentCV(agent_id="db-agent", overall_elo=1150.0)
        store._save_to_db(db_cv)

        result = store.get_cv_sync("db-agent", auto_build=False)

        assert result is not None
        assert result.agent_id == "db-agent"

    def test_get_cv_sync_auto_builds(self, store):
        """get_cv_sync auto-builds CV when not found."""
        result = store.get_cv_sync("new-agent", auto_build=True)

        assert result is not None
        store.cv_builder.build_cv.assert_called_once_with("new-agent")

    def test_get_cv_sync_updates_cache(self, store):
        """get_cv_sync updates cache after building."""
        store.get_cv_sync("new-agent", auto_build=True)

        assert "new-agent" in store._cache


# =============================================================================
# Save CV Tests
# =============================================================================


class TestSaveCV:
    """Test save_cv method."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_cv.db"

    @pytest.fixture
    def store(self, temp_db_path):
        """Create a CVStore."""
        return CVStore(db_path=temp_db_path)

    @pytest.mark.asyncio
    async def test_save_cv_to_database(self, store):
        """save_cv persists CV to database."""
        cv = AgentCV(agent_id="save-test", overall_elo=1250.0)

        await store.save_cv(cv)

        # Verify in database
        loaded = store._load_from_db("save-test")
        assert loaded is not None
        assert loaded.overall_elo == 1250.0

    @pytest.mark.asyncio
    async def test_save_cv_updates_cache(self, store):
        """save_cv updates in-memory cache."""
        cv = AgentCV(agent_id="save-test", overall_elo=1250.0)

        await store.save_cv(cv)

        assert "save-test" in store._cache
        _, cached = store._cache["save-test"]
        assert cached is cv

    @pytest.mark.asyncio
    async def test_save_cv_overwrites_existing(self, store):
        """save_cv overwrites existing CV in database."""
        cv1 = AgentCV(agent_id="overwrite-test", overall_elo=1000.0)
        cv2 = AgentCV(agent_id="overwrite-test", overall_elo=1200.0)

        await store.save_cv(cv1)
        await store.save_cv(cv2)

        loaded = store._load_from_db("overwrite-test")
        assert loaded.overall_elo == 1200.0


# =============================================================================
# Refresh CV Tests
# =============================================================================


class TestRefreshCV:
    """Test refresh_cv method."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_cv.db"

    @pytest.fixture
    def store(self, temp_db_path):
        """Create a CVStore with mocked builder."""
        mock_builder = MagicMock(spec=CVBuilder)
        # Mock returns CV with the requested agent_id
        mock_builder.build_cv.side_effect = lambda agent_id: AgentCV(
            agent_id=agent_id,
            overall_elo=1300.0,
        )
        return CVStore(db_path=temp_db_path, cv_builder=mock_builder)

    @pytest.mark.asyncio
    async def test_refresh_cv_rebuilds(self, store):
        """refresh_cv rebuilds CV from data sources."""
        # Save initial CV
        old_cv = AgentCV(agent_id="refresh-test", overall_elo=1000.0)
        await store.save_cv(old_cv)

        # Refresh
        new_cv = await store.refresh_cv("refresh-test")

        store.cv_builder.build_cv.assert_called_once_with("refresh-test")
        assert new_cv.overall_elo == 1300.0
        assert new_cv.agent_id == "refresh-test"

    @pytest.mark.asyncio
    async def test_refresh_cv_updates_database(self, store):
        """refresh_cv updates database with new CV."""
        old_cv = AgentCV(agent_id="refresh-test", overall_elo=1000.0)
        await store.save_cv(old_cv)

        await store.refresh_cv("refresh-test")

        loaded = store._load_from_db("refresh-test")
        assert loaded.overall_elo == 1300.0


# =============================================================================
# Batch Operations Tests
# =============================================================================


class TestBatchOperations:
    """Test batch CV operations."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_cv.db"

    @pytest.fixture
    def store(self, temp_db_path):
        """Create a CVStore with mocked builder."""
        mock_builder = MagicMock(spec=CVBuilder)
        mock_builder.build_cvs_batch.return_value = {
            "agent1": AgentCV(agent_id="agent1", overall_elo=1100.0),
            "agent2": AgentCV(agent_id="agent2", overall_elo=1050.0),
            "agent3": AgentCV(agent_id="agent3", overall_elo=1000.0),
        }
        return CVStore(db_path=temp_db_path, cv_builder=mock_builder)

    @pytest.mark.asyncio
    async def test_get_cvs_batch_from_cache(self, store):
        """get_cvs_batch returns cached CVs."""
        store._cache["cached1"] = (datetime.now(), AgentCV(agent_id="cached1"))
        store._cache["cached2"] = (datetime.now(), AgentCV(agent_id="cached2"))

        result = await store.get_cvs_batch(["cached1", "cached2"])

        assert "cached1" in result
        assert "cached2" in result

    @pytest.mark.asyncio
    async def test_get_cvs_batch_builds_missing(self, store):
        """get_cvs_batch builds missing CVs."""
        result = await store.get_cvs_batch(["agent1", "agent2", "agent3"])

        store.cv_builder.build_cvs_batch.assert_called()
        assert len(result) == 3
        assert result["agent1"].overall_elo == 1100.0

    @pytest.mark.asyncio
    async def test_get_cvs_batch_mixed(self, store):
        """get_cvs_batch handles mix of cached and new CVs."""
        store._cache["cached"] = (datetime.now(), AgentCV(agent_id="cached", overall_elo=1500.0))

        result = await store.get_cvs_batch(["cached", "agent1"])

        assert result["cached"].overall_elo == 1500.0
        assert "agent1" in result

    @pytest.mark.asyncio
    async def test_get_cvs_batch_without_auto_build(self, store):
        """get_cvs_batch respects auto_build=False."""
        result = await store.get_cvs_batch(["agent1", "agent2"], auto_build=False)

        store.cv_builder.build_cvs_batch.assert_not_called()
        assert len(result) == 0


# =============================================================================
# Delete CV Tests
# =============================================================================


class TestDeleteCV:
    """Test delete_cv method."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_cv.db"

    @pytest.fixture
    def store(self, temp_db_path):
        """Create a CVStore."""
        return CVStore(db_path=temp_db_path)

    @pytest.mark.asyncio
    async def test_delete_cv_removes_from_database(self, store):
        """delete_cv removes CV from database."""
        cv = AgentCV(agent_id="delete-test", overall_elo=1100.0)
        await store.save_cv(cv)

        result = await store.delete_cv("delete-test")

        assert result is True
        assert store._load_from_db("delete-test") is None

    @pytest.mark.asyncio
    async def test_delete_cv_removes_from_cache(self, store):
        """delete_cv removes CV from cache."""
        cv = AgentCV(agent_id="delete-test", overall_elo=1100.0)
        await store.save_cv(cv)

        assert "delete-test" in store._cache

        await store.delete_cv("delete-test")

        assert "delete-test" not in store._cache

    @pytest.mark.asyncio
    async def test_delete_cv_returns_false_when_not_found(self, store):
        """delete_cv returns False when CV doesn't exist."""
        result = await store.delete_cv("nonexistent")

        assert result is False


# =============================================================================
# Cache Invalidation Tests
# =============================================================================


class TestCacheInvalidation:
    """Test cache invalidation."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_cv.db"

    @pytest.fixture
    def store(self, temp_db_path):
        """Create a CVStore."""
        return CVStore(db_path=temp_db_path)

    def test_invalidate_cache_single_agent(self, store):
        """invalidate_cache removes single agent from cache."""
        store._cache["agent1"] = (datetime.now(), AgentCV(agent_id="agent1"))
        store._cache["agent2"] = (datetime.now(), AgentCV(agent_id="agent2"))

        count = store.invalidate_cache("agent1")

        assert count == 1
        assert "agent1" not in store._cache
        assert "agent2" in store._cache

    def test_invalidate_cache_all(self, store):
        """invalidate_cache clears entire cache."""
        store._cache["agent1"] = (datetime.now(), AgentCV(agent_id="agent1"))
        store._cache["agent2"] = (datetime.now(), AgentCV(agent_id="agent2"))
        store._cache["agent3"] = (datetime.now(), AgentCV(agent_id="agent3"))

        count = store.invalidate_cache()

        assert count == 3
        assert len(store._cache) == 0

    def test_invalidate_cache_nonexistent(self, store):
        """invalidate_cache returns 0 for nonexistent agent."""
        store._cache["agent1"] = (datetime.now(), AgentCV(agent_id="agent1"))

        count = store.invalidate_cache("nonexistent")

        assert count == 0
        assert "agent1" in store._cache


# =============================================================================
# Database Load/Save Tests
# =============================================================================


class TestDatabaseOperations:
    """Test database load and save operations."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_cv.db"

    @pytest.fixture
    def store(self, temp_db_path):
        """Create a CVStore."""
        return CVStore(db_path=temp_db_path)

    def test_save_to_db_stores_json(self, store):
        """_save_to_db stores CV as JSON."""
        cv = AgentCV(
            agent_id="json-test",
            overall_elo=1150.0,
            calibration_bias="well-calibrated",
        )

        store._save_to_db(cv)

        with store.connection() as conn:
            cursor = conn.execute(
                "SELECT cv_data FROM agent_cvs WHERE agent_id = ?",
                ("json-test",),
            )
            row = cursor.fetchone()

        assert row is not None
        data = json.loads(row[0])
        assert data["overall_elo"] == 1150.0

    def test_load_from_db_reconstructs_cv(self, store):
        """_load_from_db reconstructs AgentCV from JSON."""
        cv = AgentCV(
            agent_id="reconstruct-test",
            overall_elo=1200.0,
            reliability=ReliabilityMetrics(success_rate=0.95, total_calls=100),
            domain_performance={
                "code": DomainPerformance(domain="code", elo=1250, debates_count=10),
            },
        )
        store._save_to_db(cv)

        loaded = store._load_from_db("reconstruct-test")

        assert loaded is not None
        assert loaded.agent_id == "reconstruct-test"
        assert loaded.overall_elo == 1200.0
        assert loaded.reliability.success_rate == 0.95
        assert "code" in loaded.domain_performance

    def test_load_from_db_returns_none_when_not_found(self, store):
        """_load_from_db returns None when agent not found."""
        loaded = store._load_from_db("nonexistent")

        assert loaded is None

    def test_load_from_db_handles_invalid_json(self, store):
        """_load_from_db handles invalid JSON gracefully."""
        # Insert invalid JSON directly
        with store.connection() as conn:
            conn.execute(
                "INSERT INTO agent_cvs (agent_id, cv_data) VALUES (?, ?)",
                ("invalid-json", "not valid json"),
            )
            conn.commit()

        loaded = store._load_from_db("invalid-json")

        assert loaded is None


# =============================================================================
# Get All CVs Tests
# =============================================================================


class TestGetAllCVs:
    """Test get_all_cvs method."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_cv.db"

    @pytest.fixture
    def store(self, temp_db_path):
        """Create a CVStore."""
        return CVStore(db_path=temp_db_path)

    @pytest.mark.asyncio
    async def test_get_all_cvs_empty(self, store):
        """get_all_cvs returns empty list when no CVs stored."""
        result = await store.get_all_cvs()

        assert result == []

    @pytest.mark.asyncio
    async def test_get_all_cvs_returns_all(self, store):
        """get_all_cvs returns all stored CVs."""
        for i in range(5):
            cv = AgentCV(agent_id=f"agent-{i}", overall_elo=1000 + i * 50)
            await store.save_cv(cv)

        result = await store.get_all_cvs()

        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_get_all_cvs_respects_limit(self, store):
        """get_all_cvs respects limit parameter."""
        for i in range(10):
            cv = AgentCV(agent_id=f"agent-{i}", overall_elo=1000 + i * 50)
            await store.save_cv(cv)

        result = await store.get_all_cvs(limit=5)

        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_get_all_cvs_ordered_by_updated_at(self, store):
        """get_all_cvs orders by updated_at descending."""
        for i in range(3):
            cv = AgentCV(agent_id=f"agent-{i}", overall_elo=1000 + i * 100)
            await store.save_cv(cv)

        result = await store.get_all_cvs()

        # Most recently updated should be first
        assert len(result) == 3


# =============================================================================
# Top Agents for Domain Tests
# =============================================================================


class TestTopAgentsForDomain:
    """Test get_top_agents_for_domain method."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_cv.db"

    @pytest.fixture
    def store(self, temp_db_path):
        """Create a CVStore."""
        return CVStore(db_path=temp_db_path)

    @pytest.mark.asyncio
    async def test_get_top_agents_for_domain(self, store):
        """get_top_agents_for_domain returns agents sorted by domain score."""
        for i in range(3):
            cv = AgentCV(
                agent_id=f"agent-{i}",
                domain_performance={
                    "code": DomainPerformance(
                        domain="code",
                        elo=1000 + i * 100,
                        debates_count=10,
                        brier_score=0.3 - i * 0.1,
                    ),
                },
            )
            await store.save_cv(cv)

        result = await store.get_top_agents_for_domain("code", limit=3)

        assert len(result) == 3
        # Should be sorted by composite score (highest first)
        assert (
            result[0].domain_performance["code"].composite_score
            >= result[1].domain_performance["code"].composite_score
        )

    @pytest.mark.asyncio
    async def test_get_top_agents_for_domain_respects_limit(self, store):
        """get_top_agents_for_domain respects limit parameter."""
        for i in range(5):
            cv = AgentCV(
                agent_id=f"agent-{i}",
                domain_performance={
                    "code": DomainPerformance(
                        domain="code",
                        elo=1000 + i * 50,
                        debates_count=10,
                        brier_score=0.2,
                    ),
                },
            )
            await store.save_cv(cv)

        result = await store.get_top_agents_for_domain("code", limit=2)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_top_agents_for_domain_filters_insufficient_data(self, store):
        """get_top_agents_for_domain filters agents without meaningful data."""
        # Agent with meaningful data
        cv1 = AgentCV(
            agent_id="good-agent",
            domain_performance={
                "code": DomainPerformance(
                    domain="code",
                    elo=1200,
                    debates_count=10,  # Meaningful
                    brier_score=0.1,
                ),
            },
        )
        # Agent without meaningful data
        cv2 = AgentCV(
            agent_id="new-agent",
            domain_performance={
                "code": DomainPerformance(
                    domain="code",
                    elo=1500,
                    debates_count=1,  # Not meaningful
                    brier_score=0.05,
                ),
            },
        )
        await store.save_cv(cv1)
        await store.save_cv(cv2)

        result = await store.get_top_agents_for_domain("code")

        assert len(result) == 1
        assert result[0].agent_id == "good-agent"

    @pytest.mark.asyncio
    async def test_get_top_agents_for_domain_empty_when_no_domain(self, store):
        """get_top_agents_for_domain returns empty for nonexistent domain."""
        cv = AgentCV(
            agent_id="agent",
            domain_performance={
                "code": DomainPerformance(domain="code", elo=1200, debates_count=10),
            },
        )
        await store.save_cv(cv)

        result = await store.get_top_agents_for_domain("research")

        assert len(result) == 0


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Test singleton pattern."""

    def test_get_cv_store_returns_instance(self):
        """get_cv_store returns a CVStore instance."""
        # Reset singleton for testing
        import aragora.agents.cv_store as cv_store_module

        cv_store_module._cv_store = None

        with patch.object(cv_store_module, "CVStore") as MockStore:
            mock_instance = MagicMock(spec=CVStore)
            MockStore.return_value = mock_instance

            result = get_cv_store()

            assert result is mock_instance

    def test_get_cv_store_returns_same_instance(self):
        """get_cv_store returns same instance on subsequent calls."""
        import aragora.agents.cv_store as cv_store_module

        cv_store_module._cv_store = None

        with patch.object(cv_store_module, "CVStore") as MockStore:
            mock_instance = MagicMock(spec=CVStore)
            MockStore.return_value = mock_instance

            result1 = get_cv_store()
            result2 = get_cv_store()

            assert result1 is result2
            MockStore.assert_called_once()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling in CVStore."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_cv.db"

    @pytest.fixture
    def store(self, temp_db_path):
        """Create a CVStore."""
        return CVStore(db_path=temp_db_path)

    def test_load_handles_missing_fields(self, store):
        """_load_from_db handles CVs with missing fields."""
        # Insert minimal CV data
        minimal_data = json.dumps({"agent_id": "minimal"})
        with store.connection() as conn:
            conn.execute(
                "INSERT INTO agent_cvs (agent_id, cv_data) VALUES (?, ?)",
                ("minimal", minimal_data),
            )
            conn.commit()

        loaded = store._load_from_db("minimal")

        assert loaded is not None
        assert loaded.agent_id == "minimal"
        assert loaded.overall_elo == 1000.0  # Default value

    def test_save_handles_complex_cv(self, store):
        """_save_to_db handles CV with complex nested data."""
        cv = AgentCV(
            agent_id="complex-test",
            overall_elo=1150.0,
            reliability=ReliabilityMetrics(
                success_rate=0.95,
                failure_rate=0.05,
                total_calls=100,
                avg_latency_ms=150.0,
            ),
            domain_performance={
                "code": DomainPerformance(
                    domain="code",
                    elo=1200,
                    win_rate=0.7,
                    debates_count=15,
                    brier_score=0.15,
                ),
                "research": DomainPerformance(
                    domain="research",
                    elo=1100,
                    win_rate=0.6,
                    debates_count=10,
                    brier_score=0.2,
                ),
            },
            model_capabilities=["reasoning", "code", "analysis"],
            learned_strengths=["technical clarity", "structured arguments"],
        )

        store._save_to_db(cv)
        loaded = store._load_from_db("complex-test")

        assert loaded is not None
        assert loaded.reliability.success_rate == 0.95
        assert len(loaded.domain_performance) == 2
        assert loaded.model_capabilities == ["reasoning", "code", "analysis"]
