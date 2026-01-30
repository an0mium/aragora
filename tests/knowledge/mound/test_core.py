"""
Comprehensive unit tests for aragora/knowledge/mound/core.py.

Tests cover:
- Helper functions (_to_iso_string, _to_enum_value)
- KnowledgeMoundCore initialization and properties
- Storage backend initialization (SQLite, PostgreSQL, Redis, Weaviate)
- Private storage adapter methods (_save_node, _get_node, _update_node, _delete_node)
- Relationship operations (_save_relationship, _get_relationships, _get_relationships_batch)
- Query helper methods (_query_local, _query_continuum, _query_consensus, etc.)
- Lifecycle management (close, session context manager)
- Statistics methods (get_stats, _get_stats)
- Ops mixin adapter methods (dedup/pruning operations)
- Archive and restore operations
- Converter wrappers
- Error handling and edge cases
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.knowledge.mound.core import (
    KnowledgeMoundCore,
    _to_enum_value,
    _to_iso_string,
)
from aragora.knowledge.mound.types import (
    ConfidenceLevel,
    KnowledgeItem,
    KnowledgeLink,
    KnowledgeSource,
    MoundBackend,
    MoundConfig,
    MoundStats,
    QueryFilters,
    RelationshipType,
)


# ============================================================
# Helper Functions Tests
# ============================================================


class TestToIsoString:
    """Tests for _to_iso_string helper function."""

    def test_returns_none_for_none(self):
        """Should return None when given None."""
        assert _to_iso_string(None) is None

    def test_returns_string_unchanged(self):
        """Should return string unchanged."""
        iso_str = "2025-01-15T12:00:00"
        assert _to_iso_string(iso_str) == iso_str

    def test_converts_datetime_to_iso_format(self):
        """Should convert datetime to ISO format string."""
        dt = datetime(2025, 1, 15, 12, 0, 0)
        assert _to_iso_string(dt) == "2025-01-15T12:00:00"

    def test_converts_datetime_with_microseconds(self):
        """Should handle datetime with microseconds."""
        dt = datetime(2025, 1, 15, 12, 30, 45, 123456)
        result = _to_iso_string(dt)
        assert result.startswith("2025-01-15T12:30:45")

    def test_falls_back_to_str_for_unknown_types(self):
        """Should fall back to str() for objects without isoformat."""
        assert _to_iso_string(42) == "42"
        assert _to_iso_string(3.14) == "3.14"
        assert _to_iso_string(["list"]) == "['list']"


class TestToEnumValue:
    """Tests for _to_enum_value helper function."""

    class SampleEnum(str, Enum):
        """Sample enum for testing."""

        OPTION_A = "option_a"
        OPTION_B = "option_b"

    def test_returns_none_for_none(self):
        """Should return None when given None."""
        assert _to_enum_value(None) is None

    def test_returns_string_unchanged(self):
        """Should return string unchanged."""
        assert _to_enum_value("test_value") == "test_value"

    def test_extracts_value_from_enum(self):
        """Should extract .value from enum instances."""
        assert _to_enum_value(self.SampleEnum.OPTION_A) == "option_a"
        assert _to_enum_value(self.SampleEnum.OPTION_B) == "option_b"

    def test_extracts_value_from_mound_types_enum(self):
        """Should extract value from actual mound type enums."""
        assert _to_enum_value(MoundBackend.SQLITE) == "sqlite"
        assert _to_enum_value(MoundBackend.POSTGRES) == "postgres"
        assert _to_enum_value(KnowledgeSource.FACT) == "fact"

    def test_falls_back_to_str_for_objects_without_value(self):
        """Should fall back to str() for objects without .value."""
        assert _to_enum_value(123) == "123"
        assert _to_enum_value({"key": "val"}) == "{'key': 'val'}"


# ============================================================
# KnowledgeMoundCore Initialization Tests
# ============================================================


class TestKnowledgeMoundCoreInit:
    """Tests for KnowledgeMoundCore initialization."""

    def test_default_initialization(self):
        """Should initialize with defaults when no config provided."""
        core = KnowledgeMoundCore()

        assert core.config is not None
        assert core.config.backend == MoundBackend.SQLITE
        assert core.workspace_id == "default"
        assert core.event_emitter is None
        assert core._initialized is False

    def test_initialization_with_config(self):
        """Should use provided config."""
        config = MoundConfig(
            backend=MoundBackend.POSTGRES,
            postgres_url="postgresql://localhost/test",
            default_workspace_id="custom_ws",
        )
        core = KnowledgeMoundCore(config=config)

        assert core.config.backend == MoundBackend.POSTGRES
        assert core.workspace_id == "custom_ws"

    def test_workspace_id_overrides_config(self):
        """Should use workspace_id parameter over config default."""
        config = MoundConfig(default_workspace_id="config_ws")
        core = KnowledgeMoundCore(config=config, workspace_id="override_ws")

        assert core.workspace_id == "override_ws"

    def test_event_emitter_assignment(self):
        """Should assign event emitter."""
        mock_emitter = MagicMock()
        core = KnowledgeMoundCore(event_emitter=mock_emitter)

        assert core.event_emitter is mock_emitter

    def test_storage_backends_not_initialized(self):
        """Should not initialize storage backends in constructor."""
        core = KnowledgeMoundCore()

        assert core._meta_store is None
        assert core._store is None
        assert core._cache is None
        assert core._vector_store is None
        assert core._semantic_store is None

    def test_connected_memory_systems_not_initialized(self):
        """Should not initialize connected memory systems in constructor."""
        core = KnowledgeMoundCore()

        assert core._continuum is None
        assert core._consensus is None
        assert core._facts is None
        assert core._evidence is None
        assert core._critique is None


class TestKnowledgeMoundCoreIsInitialized:
    """Tests for is_initialized property."""

    def test_not_initialized_by_default(self):
        """Should return False before initialize() is called."""
        core = KnowledgeMoundCore()
        assert core.is_initialized is False

    @pytest.mark.asyncio
    async def test_initialized_after_initialize(self):
        """Should return True after initialize() is called."""
        core = KnowledgeMoundCore()

        # Mock the initialization methods
        with patch.object(core, "_init_sqlite", new_callable=AsyncMock) as mock_sqlite:
            with patch.object(
                core, "_init_semantic_store", new_callable=AsyncMock
            ) as mock_semantic:
                await core.initialize()

        assert core.is_initialized is True


# ============================================================
# KnowledgeMoundCore Initialize Tests
# ============================================================


class TestKnowledgeMoundCoreInitialize:
    """Tests for KnowledgeMoundCore.initialize()."""

    @pytest.mark.asyncio
    async def test_initialize_sqlite_backend(self):
        """Should initialize SQLite backend."""
        config = MoundConfig(backend=MoundBackend.SQLITE)
        core = KnowledgeMoundCore(config=config)

        with patch.object(core, "_init_sqlite", new_callable=AsyncMock) as mock_sqlite:
            with patch.object(
                core, "_init_semantic_store", new_callable=AsyncMock
            ) as mock_semantic:
                await core.initialize()

        mock_sqlite.assert_called_once()
        mock_semantic.assert_called_once()
        assert core._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_postgres_backend(self):
        """Should initialize PostgreSQL backend."""
        config = MoundConfig(
            backend=MoundBackend.POSTGRES,
            postgres_url="postgresql://localhost/test",
        )
        core = KnowledgeMoundCore(config=config)

        with patch.object(core, "_init_postgres", new_callable=AsyncMock) as mock_postgres:
            with patch.object(
                core, "_init_semantic_store", new_callable=AsyncMock
            ) as mock_semantic:
                await core.initialize()

        mock_postgres.assert_called_once()
        assert core._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_hybrid_backend(self):
        """Should initialize both Postgres and Redis for hybrid backend."""
        config = MoundConfig(
            backend=MoundBackend.HYBRID,
            postgres_url="postgresql://localhost/test",
            redis_url="redis://localhost",
        )
        core = KnowledgeMoundCore(config=config)

        with patch.object(core, "_init_postgres", new_callable=AsyncMock) as mock_postgres:
            with patch.object(core, "_init_redis", new_callable=AsyncMock) as mock_redis:
                with patch.object(
                    core, "_init_semantic_store", new_callable=AsyncMock
                ) as mock_semantic:
                    await core.initialize()

        mock_postgres.assert_called_once()
        mock_redis.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_with_redis_cache(self):
        """Should initialize Redis cache when configured."""
        config = MoundConfig(
            backend=MoundBackend.SQLITE,
            redis_url="redis://localhost",
        )
        core = KnowledgeMoundCore(config=config)

        with patch.object(core, "_init_sqlite", new_callable=AsyncMock):
            with patch.object(core, "_init_redis", new_callable=AsyncMock) as mock_redis:
                with patch.object(core, "_init_semantic_store", new_callable=AsyncMock):
                    await core.initialize()

        mock_redis.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_with_weaviate(self):
        """Should initialize Weaviate vector store when configured."""
        config = MoundConfig(
            backend=MoundBackend.SQLITE,
            weaviate_url="http://localhost:8080",
        )
        core = KnowledgeMoundCore(config=config)

        with patch.object(core, "_init_sqlite", new_callable=AsyncMock):
            with patch.object(core, "_init_weaviate", new_callable=AsyncMock) as mock_weaviate:
                with patch.object(core, "_init_semantic_store", new_callable=AsyncMock):
                    await core.initialize()

        mock_weaviate.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_with_staleness_detection(self):
        """Should initialize staleness detector when enabled."""
        config = MoundConfig(
            backend=MoundBackend.SQLITE,
            enable_staleness_detection=True,
        )
        core = KnowledgeMoundCore(config=config)

        with patch.object(core, "_init_sqlite", new_callable=AsyncMock):
            with patch.object(core, "_init_semantic_store", new_callable=AsyncMock):
                with patch("aragora.knowledge.mound.staleness.StalenessDetector") as mock_detector:
                    await core.initialize()

        mock_detector.assert_called_once()
        assert core._staleness_detector is not None

    @pytest.mark.asyncio
    async def test_initialize_with_culture_accumulator(self):
        """Should initialize culture accumulator when enabled."""
        config = MoundConfig(
            backend=MoundBackend.SQLITE,
            enable_culture_accumulator=True,
        )
        core = KnowledgeMoundCore(config=config)

        with patch.object(core, "_init_sqlite", new_callable=AsyncMock):
            with patch.object(core, "_init_semantic_store", new_callable=AsyncMock):
                with patch(
                    "aragora.knowledge.mound.culture.CultureAccumulator"
                ) as mock_accumulator:
                    await core.initialize()

        mock_accumulator.assert_called_once()
        assert core._culture_accumulator is not None

    @pytest.mark.asyncio
    async def test_double_initialize_is_noop(self):
        """Should skip initialization if already initialized."""
        core = KnowledgeMoundCore()
        core._initialized = True

        with patch.object(core, "_init_sqlite", new_callable=AsyncMock) as mock_sqlite:
            await core.initialize()

        mock_sqlite.assert_not_called()


class TestKnowledgeMoundCoreInitBackends:
    """Tests for individual backend initialization methods."""

    @pytest.mark.asyncio
    async def test_init_sqlite(self):
        """Should initialize SQLite meta store."""
        core = KnowledgeMoundCore()

        with patch("aragora.knowledge.mound.KnowledgeMoundMetaStore") as mock_store_class:
            await core._init_sqlite()

        mock_store_class.assert_called_once()
        assert core._meta_store is not None

    @pytest.mark.asyncio
    async def test_init_postgres_without_url(self):
        """Should fall back to SQLite when no Postgres URL configured."""
        config = MoundConfig(backend=MoundBackend.POSTGRES, postgres_url=None)
        core = KnowledgeMoundCore(config=config)

        with patch.object(core, "_init_sqlite", new_callable=AsyncMock) as mock_sqlite:
            await core._init_postgres()

        mock_sqlite.assert_called_once()

    @pytest.mark.asyncio
    async def test_init_postgres_connection_error(self):
        """Should fall back to SQLite on Postgres connection error."""
        config = MoundConfig(
            backend=MoundBackend.POSTGRES,
            postgres_url="postgresql://invalid/db",
        )
        core = KnowledgeMoundCore(config=config)

        with patch(
            "aragora.knowledge.mound.postgres_store.PostgresStore",
            side_effect=ConnectionError("Connection failed"),
        ):
            with patch.object(core, "_init_sqlite", new_callable=AsyncMock) as mock_sqlite:
                await core._init_postgres()

        mock_sqlite.assert_called_once()

    @pytest.mark.asyncio
    async def test_init_redis_without_url(self):
        """Should skip Redis initialization when no URL configured."""
        config = MoundConfig(redis_url=None)
        core = KnowledgeMoundCore(config=config)

        await core._init_redis()

        assert core._cache is None

    @pytest.mark.asyncio
    async def test_init_redis_import_error(self):
        """Should handle missing redis package gracefully."""
        config = MoundConfig(redis_url="redis://localhost")
        core = KnowledgeMoundCore(config=config)

        with patch(
            "aragora.knowledge.mound.redis_cache.RedisCache",
            side_effect=ImportError("redis not installed"),
        ):
            await core._init_redis()

        assert core._cache is None

    @pytest.mark.asyncio
    async def test_init_weaviate_import_error(self):
        """Should handle missing weaviate package gracefully."""
        config = MoundConfig(weaviate_url="http://localhost:8080")
        core = KnowledgeMoundCore(config=config)

        with patch(
            "aragora.documents.indexing.weaviate_store.WeaviateStore",
            side_effect=ImportError("weaviate not installed"),
        ):
            await core._init_weaviate()

        assert core._vector_store is None

    @pytest.mark.asyncio
    async def test_init_semantic_store_import_error(self):
        """Should handle missing semantic store dependencies gracefully."""
        core = KnowledgeMoundCore()

        with patch(
            "aragora.knowledge.mound.semantic_store.SemanticStore",
            side_effect=ImportError("dependencies not installed"),
        ):
            await core._init_semantic_store()

        assert core._semantic_store is None


# ============================================================
# KnowledgeMoundCore Ensure Initialized Tests
# ============================================================


class TestEnsureInitialized:
    """Tests for _ensure_initialized method."""

    def test_raises_when_not_initialized(self):
        """Should raise RuntimeError when not initialized."""
        core = KnowledgeMoundCore()

        with pytest.raises(RuntimeError, match="not initialized"):
            core._ensure_initialized()

    def test_no_error_when_initialized(self):
        """Should not raise when initialized."""
        core = KnowledgeMoundCore()
        core._initialized = True

        # Should not raise
        core._ensure_initialized()


# ============================================================
# KnowledgeMoundCore Lifecycle Tests
# ============================================================


class TestKnowledgeMoundCoreClose:
    """Tests for KnowledgeMoundCore.close()."""

    @pytest.mark.asyncio
    async def test_close_cache(self):
        """Should close cache connection."""
        core = KnowledgeMoundCore()
        mock_cache = AsyncMock()
        core._cache = mock_cache
        core._initialized = True

        await core.close()

        mock_cache.close.assert_called_once()
        assert core._initialized is False

    @pytest.mark.asyncio
    async def test_close_vector_store(self):
        """Should close vector store connection."""
        core = KnowledgeMoundCore()
        mock_vector = AsyncMock()
        core._vector_store = mock_vector
        core._initialized = True

        await core.close()

        mock_vector.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_vector_store_error_handled(self):
        """Should handle errors when closing vector store."""
        core = KnowledgeMoundCore()
        mock_vector = AsyncMock()
        mock_vector.close.side_effect = RuntimeError("Close error")
        core._vector_store = mock_vector
        core._initialized = True

        # Should not raise
        await core.close()

    @pytest.mark.asyncio
    async def test_close_meta_store(self):
        """Should close meta store if it has close method."""
        core = KnowledgeMoundCore()
        mock_meta = AsyncMock()
        core._meta_store = mock_meta
        core._initialized = True

        await core.close()

        mock_meta.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_resets_initialized(self):
        """Should reset _initialized flag."""
        core = KnowledgeMoundCore()
        core._initialized = True

        await core.close()

        assert core._initialized is False


class TestKnowledgeMoundCoreSession:
    """Tests for KnowledgeMoundCore.session() context manager."""

    @pytest.mark.asyncio
    async def test_session_initializes_and_closes(self):
        """Should initialize on enter and close on exit."""
        core = KnowledgeMoundCore()

        with patch.object(core, "initialize", new_callable=AsyncMock) as mock_init:
            with patch.object(core, "close", new_callable=AsyncMock) as mock_close:
                async with core.session() as session:
                    assert session is core
                    mock_init.assert_called_once()

                mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_closes_on_exception(self):
        """Should close even when exception occurs."""
        core = KnowledgeMoundCore()

        with patch.object(core, "initialize", new_callable=AsyncMock):
            with patch.object(core, "close", new_callable=AsyncMock) as mock_close:
                with pytest.raises(ValueError):
                    async with core.session():
                        raise ValueError("Test error")

                mock_close.assert_called_once()


# ============================================================
# KnowledgeMoundCore Statistics Tests
# ============================================================


class TestKnowledgeMoundCoreStats:
    """Tests for get_stats and _get_stats methods."""

    @pytest.mark.asyncio
    async def test_get_stats_requires_initialization(self):
        """Should raise if not initialized."""
        core = KnowledgeMoundCore()

        with pytest.raises(RuntimeError, match="not initialized"):
            await core.get_stats()

    @pytest.mark.asyncio
    async def test_get_stats_with_workspace(self):
        """Should use provided workspace ID."""
        core = KnowledgeMoundCore(workspace_id="default_ws")
        core._initialized = True

        mock_stats = MoundStats(
            total_nodes=10,
            nodes_by_type={},
            nodes_by_tier={},
            nodes_by_validation={},
            total_relationships=5,
            relationships_by_type={},
            average_confidence=0.8,
            stale_nodes_count=2,
            workspace_id="custom_ws",
        )

        with patch.object(core, "_get_stats", new_callable=AsyncMock) as mock_get_stats:
            mock_get_stats.return_value = mock_stats
            result = await core.get_stats(workspace_id="custom_ws")

        mock_get_stats.assert_called_once_with("custom_ws")
        assert result.workspace_id == "custom_ws"

    @pytest.mark.asyncio
    async def test_get_stats_uses_default_workspace(self):
        """Should use default workspace if none provided."""
        core = KnowledgeMoundCore(workspace_id="default_ws")
        core._initialized = True

        with patch.object(core, "_get_stats", new_callable=AsyncMock) as mock_get_stats:
            mock_get_stats.return_value = MoundStats(
                total_nodes=0,
                nodes_by_type={},
                nodes_by_tier={},
                nodes_by_validation={},
                total_relationships=0,
                relationships_by_type={},
                average_confidence=0.0,
                stale_nodes_count=0,
            )
            await core.get_stats()

        mock_get_stats.assert_called_once_with("default_ws")

    @pytest.mark.asyncio
    async def test_get_stats_async_method(self):
        """Should use async method if available."""
        core = KnowledgeMoundCore()
        core._initialized = True

        mock_stats = MoundStats(
            total_nodes=100,
            nodes_by_type={"fact": 50, "claim": 50},
            nodes_by_tier={"slow": 100},
            nodes_by_validation={},
            total_relationships=25,
            relationships_by_type={},
            average_confidence=0.75,
            stale_nodes_count=5,
        )

        mock_store = MagicMock()
        mock_store.get_stats_async = AsyncMock(return_value=mock_stats)
        core._meta_store = mock_store

        result = await core._get_stats("ws_1")

        mock_store.get_stats_async.assert_called_once_with("ws_1")
        assert result == mock_stats

    @pytest.mark.asyncio
    async def test_get_stats_sync_fallback(self):
        """Should use sync method as fallback."""
        core = KnowledgeMoundCore()
        core._initialized = True

        # Use a MagicMock that doesn't have get_stats_async attribute
        mock_store = MagicMock(spec=["get_stats"])
        mock_store.get_stats.return_value = {
            "total_nodes": 50,
            "by_type": {"fact": 30},
            "by_tier": {"slow": 50},
            "by_validation_status": {},
            "total_relationships": 10,
            "average_confidence": 0.6,
        }
        core._meta_store = mock_store

        result = await core._get_stats("ws_1")

        assert result.total_nodes == 50
        assert result.nodes_by_type == {"fact": 30}
        assert result.average_confidence == 0.6


# ============================================================
# KnowledgeMoundCore Storage Adapter Tests
# ============================================================


class TestKnowledgeMoundCoreSaveNode:
    """Tests for _save_node method."""

    @pytest.mark.asyncio
    async def test_save_node_async(self):
        """Should use async method if available."""
        core = KnowledgeMoundCore()
        mock_store = MagicMock()
        mock_store.save_node_async = AsyncMock()
        core._meta_store = mock_store

        node_data = {
            "id": "node_1",
            "node_type": "fact",
            "content": "Test content",
            "confidence": 0.8,
            "workspace_id": "ws_1",
        }

        await core._save_node(node_data)

        mock_store.save_node_async.assert_called_once_with(node_data)

    @pytest.mark.asyncio
    async def test_save_node_sync_fallback(self):
        """Should use sync method as fallback."""
        core = KnowledgeMoundCore()
        # Use spec to ensure only specific methods exist
        mock_store = MagicMock(spec=["save_node"])
        mock_store.save_node = MagicMock()
        core._meta_store = mock_store

        node_data = {
            "id": "node_1",
            "node_type": "fact",
            "content": "Test content",
            "confidence": 0.8,
            "workspace_id": "ws_1",
            "source_type": "debate",
            "debate_id": "debate_1",
        }

        with patch("aragora.knowledge.mound.KnowledgeNode") as mock_node_class:
            with patch("aragora.knowledge.mound.ProvenanceChain"):
                with patch("aragora.knowledge.mound.ProvenanceType"):
                    await core._save_node(node_data)

        mock_store.save_node.assert_called_once()


class TestKnowledgeMoundCoreGetNode:
    """Tests for _get_node method."""

    @pytest.mark.asyncio
    async def test_get_node_async(self):
        """Should use async method if available."""
        core = KnowledgeMoundCore()
        mock_item = MagicMock(spec=KnowledgeItem)
        mock_store = MagicMock()
        mock_store.get_node_async = AsyncMock(return_value=mock_item)
        core._meta_store = mock_store

        result = await core._get_node("node_1")

        mock_store.get_node_async.assert_called_once_with("node_1")
        assert result is mock_item

    @pytest.mark.asyncio
    async def test_get_node_sync_fallback(self):
        """Should use sync method and convert to item."""
        core = KnowledgeMoundCore()
        mock_node = MagicMock()
        # Use spec to ensure get_node_async doesn't exist
        mock_store = MagicMock(spec=["get_node"])
        mock_store.get_node = MagicMock(return_value=mock_node)
        core._meta_store = mock_store

        mock_item = MagicMock(spec=KnowledgeItem)
        with patch.object(core, "_node_to_item", return_value=mock_item) as mock_convert:
            result = await core._get_node("node_1")

        mock_store.get_node.assert_called_once_with("node_1")
        mock_convert.assert_called_once_with(mock_node)
        assert result is mock_item

    @pytest.mark.asyncio
    async def test_get_node_returns_none_when_not_found(self):
        """Should return None when node not found."""
        core = KnowledgeMoundCore()
        # Use spec to ensure get_node_async doesn't exist
        mock_store = MagicMock(spec=["get_node"])
        mock_store.get_node = MagicMock(return_value=None)
        core._meta_store = mock_store

        result = await core._get_node("nonexistent")

        assert result is None


class TestKnowledgeMoundCoreUpdateNode:
    """Tests for _update_node method."""

    @pytest.mark.asyncio
    async def test_update_node_async(self):
        """Should use async method if available."""
        core = KnowledgeMoundCore()
        mock_store = MagicMock()
        mock_store.update_node_async = AsyncMock()
        core._meta_store = mock_store

        updates = {"confidence": 0.9, "content": "Updated content"}
        await core._update_node("node_1", updates)

        mock_store.update_node_async.assert_called_once_with("node_1", updates)

    @pytest.mark.asyncio
    async def test_update_node_sync_fallback(self):
        """Should get, update, and save node using sync methods."""
        core = KnowledgeMoundCore()
        mock_node = MagicMock()
        mock_node.confidence = 0.5
        # Use spec to ensure update_node_async doesn't exist
        mock_store = MagicMock(spec=["get_node", "save_node"])
        mock_store.get_node = MagicMock(return_value=mock_node)
        mock_store.save_node = MagicMock()
        core._meta_store = mock_store

        updates = {"confidence": 0.9}
        await core._update_node("node_1", updates)

        mock_store.get_node.assert_called_once_with("node_1")
        assert mock_node.confidence == 0.9
        mock_store.save_node.assert_called_once_with(mock_node)


class TestKnowledgeMoundCoreDeleteNode:
    """Tests for _delete_node method."""

    @pytest.mark.asyncio
    async def test_delete_node_async(self):
        """Should use async method if available."""
        core = KnowledgeMoundCore()
        mock_store = MagicMock()
        mock_store.delete_node_async = AsyncMock(return_value=True)
        core._meta_store = mock_store

        result = await core._delete_node("node_1")

        mock_store.delete_node_async.assert_called_once_with("node_1")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_node_sync_fallback(self):
        """Should use raw SQL for SQLite backend."""
        core = KnowledgeMoundCore()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        # Use spec to ensure delete_node_async doesn't exist
        mock_store = MagicMock(spec=["connection"])
        mock_store.connection.return_value = mock_conn
        core._meta_store = mock_store

        result = await core._delete_node("node_1")

        mock_conn.execute.assert_called_once()
        assert result is True


# ============================================================
# KnowledgeMoundCore Relationship Tests
# ============================================================


class TestKnowledgeMoundCoreSaveRelationship:
    """Tests for _save_relationship method."""

    @pytest.mark.asyncio
    async def test_save_relationship_async(self):
        """Should use async method if available."""
        core = KnowledgeMoundCore()
        mock_store = MagicMock()
        mock_store.save_relationship_async = AsyncMock()
        core._meta_store = mock_store

        await core._save_relationship("from_id", "to_id", "supports")

        mock_store.save_relationship_async.assert_called_once_with("from_id", "to_id", "supports")

    @pytest.mark.asyncio
    async def test_save_relationship_sync_fallback(self):
        """Should use sync method as fallback."""
        core = KnowledgeMoundCore()
        # Use spec to ensure save_relationship_async doesn't exist
        mock_store = MagicMock(spec=["save_relationship"])
        mock_store.save_relationship = MagicMock()
        core._meta_store = mock_store

        with patch("aragora.knowledge.mound.KnowledgeRelationship"):
            await core._save_relationship("from_id", "to_id", "supports")

        mock_store.save_relationship.assert_called_once()


class TestKnowledgeMoundCoreGetRelationships:
    """Tests for _get_relationships method."""

    @pytest.mark.asyncio
    async def test_get_relationships_async(self):
        """Should use async method if available."""
        core = KnowledgeMoundCore()
        mock_links = [MagicMock(spec=KnowledgeLink)]
        mock_store = MagicMock()
        mock_store.get_relationships_async = AsyncMock(return_value=mock_links)
        core._meta_store = mock_store

        result = await core._get_relationships("node_1")

        mock_store.get_relationships_async.assert_called_once_with("node_1", None)
        assert result == mock_links

    @pytest.mark.asyncio
    async def test_get_relationships_with_types_filter(self):
        """Should pass relationship types filter."""
        core = KnowledgeMoundCore()
        mock_store = MagicMock()
        mock_store.get_relationships_async = AsyncMock(return_value=[])
        core._meta_store = mock_store

        types = [RelationshipType.SUPPORTS, RelationshipType.CONTRADICTS]
        await core._get_relationships("node_1", types=types)

        mock_store.get_relationships_async.assert_called_once_with("node_1", types)

    @pytest.mark.asyncio
    async def test_get_relationships_sync_fallback(self):
        """Should use sync method and convert relationships."""
        core = KnowledgeMoundCore()
        mock_rel = MagicMock()
        # Use spec to ensure get_relationships_async doesn't exist
        mock_store = MagicMock(spec=["get_relationships"])
        mock_store.get_relationships = MagicMock(return_value=[mock_rel])
        core._meta_store = mock_store

        mock_link = MagicMock(spec=KnowledgeLink)
        with patch.object(core, "_rel_to_link", return_value=mock_link) as mock_convert:
            result = await core._get_relationships("node_1")

        mock_store.get_relationships.assert_called_once_with("node_1")
        mock_convert.assert_called_once_with(mock_rel)
        assert result == [mock_link]


class TestKnowledgeMoundCoreGetRelationshipsBatch:
    """Tests for _get_relationships_batch method."""

    @pytest.mark.asyncio
    async def test_get_relationships_batch_empty_ids(self):
        """Should return empty dict for empty node_ids."""
        core = KnowledgeMoundCore()

        result = await core._get_relationships_batch([])

        assert result == {}

    @pytest.mark.asyncio
    async def test_get_relationships_batch_async(self):
        """Should use batch async method if available."""
        core = KnowledgeMoundCore()
        mock_result = {
            "node_1": [MagicMock(spec=KnowledgeLink)],
            "node_2": [],
        }
        mock_store = MagicMock()
        mock_store.get_relationships_batch_async = AsyncMock(return_value=mock_result)
        core._meta_store = mock_store

        result = await core._get_relationships_batch(["node_1", "node_2"])

        mock_store.get_relationships_batch_async.assert_called_once_with(["node_1", "node_2"], None)
        assert result == mock_result

    @pytest.mark.asyncio
    async def test_get_relationships_batch_parallel_fallback(self):
        """Should fall back to parallel fetching."""
        core = KnowledgeMoundCore()
        # Use spec to ensure get_relationships_batch_async doesn't exist
        mock_store = MagicMock(spec=[])
        core._meta_store = mock_store

        mock_links = {"node_1": [MagicMock(spec=KnowledgeLink)], "node_2": []}

        async def mock_get_rels(node_id, types=None):
            return mock_links.get(node_id, [])

        with patch.object(core, "_get_relationships", side_effect=mock_get_rels) as mock_get:
            result = await core._get_relationships_batch(["node_1", "node_2"])

        assert mock_get.call_count == 2
        assert "node_1" in result
        assert "node_2" in result

    @pytest.mark.asyncio
    async def test_get_relationships_batch_handles_errors(self):
        """Should handle individual errors gracefully."""
        core = KnowledgeMoundCore()
        # Use spec to ensure get_relationships_batch_async doesn't exist
        mock_store = MagicMock(spec=[])
        core._meta_store = mock_store

        async def mock_get_rels(node_id, types=None):
            if node_id == "node_2":
                raise RuntimeError("DB error")
            return [MagicMock(spec=KnowledgeLink)]

        with patch.object(core, "_get_relationships", side_effect=mock_get_rels):
            result = await core._get_relationships_batch(["node_1", "node_2"])

        assert len(result["node_1"]) == 1
        assert result["node_2"] == []  # Error case returns empty list


# ============================================================
# KnowledgeMoundCore Query Helper Tests
# ============================================================


class TestKnowledgeMoundCoreQueryLocal:
    """Tests for _query_local method."""

    @pytest.mark.asyncio
    async def test_query_local_async(self):
        """Should use async method if available."""
        core = KnowledgeMoundCore()
        mock_items = [MagicMock(spec=KnowledgeItem)]
        mock_store = MagicMock()
        mock_store.query_async = AsyncMock(return_value=mock_items)
        core._meta_store = mock_store

        result = await core._query_local("test query", None, 10, "ws_1")

        mock_store.query_async.assert_called_once_with("test query", None, 10, "ws_1")
        assert result == mock_items

    @pytest.mark.asyncio
    async def test_query_local_keyword_matching(self):
        """Should use simple keyword matching for sync fallback."""
        core = KnowledgeMoundCore()
        mock_node1 = MagicMock()
        mock_node1.content = "Python programming language"
        mock_node2 = MagicMock()
        mock_node2.content = "Java programming language"
        mock_node3 = MagicMock()
        mock_node3.content = "Cooking recipes"

        # Use spec to ensure query_async doesn't exist
        mock_store = MagicMock(spec=["query_nodes"])
        mock_store.query_nodes = MagicMock(return_value=[mock_node1, mock_node2, mock_node3])
        core._meta_store = mock_store

        mock_item = MagicMock(spec=KnowledgeItem)
        with patch.object(core, "_node_to_item", return_value=mock_item):
            result = await core._query_local("python programming", None, 10, "ws_1")

        # Should be sorted by relevance (Python should rank highest)
        assert len(result) > 0


class TestKnowledgeMoundCoreQueryContinuum:
    """Tests for _query_continuum method."""

    @pytest.mark.asyncio
    async def test_query_continuum_no_continuum(self):
        """Should return empty list when continuum not connected."""
        core = KnowledgeMoundCore()
        core._continuum = None

        result = await core._query_continuum("test", None, 10)

        assert result == []

    @pytest.mark.asyncio
    async def test_query_continuum_success(self):
        """Should query continuum and convert results."""
        core = KnowledgeMoundCore()
        mock_entry = MagicMock()
        mock_continuum = MagicMock()
        mock_continuum.search_by_keyword = MagicMock(return_value=[mock_entry])
        core._continuum = mock_continuum

        mock_item = MagicMock(spec=KnowledgeItem)
        with patch.object(core, "_continuum_to_item", return_value=mock_item):
            result = await core._query_continuum("test query", None, 10)

        mock_continuum.search_by_keyword.assert_called_once_with("test query", limit=10)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_query_continuum_error_handling(self):
        """Should handle errors gracefully."""
        core = KnowledgeMoundCore()
        mock_continuum = MagicMock()
        mock_continuum.search_by_keyword = MagicMock(side_effect=KeyError("Error"))
        core._continuum = mock_continuum

        result = await core._query_continuum("test", None, 10)

        assert result == []


class TestKnowledgeMoundCoreQueryConsensus:
    """Tests for _query_consensus method."""

    @pytest.mark.asyncio
    async def test_query_consensus_no_consensus(self):
        """Should return empty list when consensus not connected."""
        core = KnowledgeMoundCore()
        core._consensus = None

        result = await core._query_consensus("test", None, 10)

        assert result == []

    @pytest.mark.asyncio
    async def test_query_consensus_success(self):
        """Should query consensus and convert results."""
        core = KnowledgeMoundCore()
        mock_entry = MagicMock()
        mock_consensus = MagicMock()
        mock_consensus.search_by_topic = AsyncMock(return_value=[mock_entry])
        core._consensus = mock_consensus

        mock_item = MagicMock(spec=KnowledgeItem)
        with patch.object(core, "_consensus_to_item", return_value=mock_item):
            result = await core._query_consensus("test query", None, 10)

        mock_consensus.search_by_topic.assert_called_once_with("test query", limit=10)
        assert len(result) == 1


class TestKnowledgeMoundCoreQueryFacts:
    """Tests for _query_facts method."""

    @pytest.mark.asyncio
    async def test_query_facts_no_facts_store(self):
        """Should return empty list when facts store not connected."""
        core = KnowledgeMoundCore()
        core._facts = None

        result = await core._query_facts("test", None, 10, "ws_1")

        assert result == []

    @pytest.mark.asyncio
    async def test_query_facts_success(self):
        """Should query facts and convert results."""
        core = KnowledgeMoundCore()
        mock_fact = MagicMock()
        mock_facts_store = MagicMock()
        mock_facts_store.query_facts = MagicMock(return_value=[mock_fact])
        core._facts = mock_facts_store

        mock_item = MagicMock(spec=KnowledgeItem)
        with patch.object(core, "_fact_to_item", return_value=mock_item):
            result = await core._query_facts("test query", None, 10, "ws_1")

        mock_facts_store.query_facts.assert_called_once()
        assert len(result) == 1


class TestKnowledgeMoundCoreQueryEvidence:
    """Tests for _query_evidence method."""

    @pytest.mark.asyncio
    async def test_query_evidence_no_evidence_store(self):
        """Should return empty list when evidence store not connected."""
        core = KnowledgeMoundCore()
        core._evidence = None

        result = await core._query_evidence("test", None, 10, "ws_1")

        assert result == []

    @pytest.mark.asyncio
    async def test_query_evidence_success(self):
        """Should query evidence and convert results."""
        core = KnowledgeMoundCore()
        mock_evidence = MagicMock()
        mock_evidence_store = MagicMock()
        mock_evidence_store.search = MagicMock(return_value=[mock_evidence])
        core._evidence = mock_evidence_store

        mock_item = MagicMock(spec=KnowledgeItem)
        with patch.object(core, "_evidence_to_item", return_value=mock_item):
            result = await core._query_evidence("test query", None, 10, "ws_1")

        mock_evidence_store.search.assert_called_once_with("test query", limit=10)
        assert len(result) == 1


class TestKnowledgeMoundCoreQueryCritique:
    """Tests for _query_critique method."""

    @pytest.mark.asyncio
    async def test_query_critique_no_critique_store(self):
        """Should return empty list when critique store not connected."""
        core = KnowledgeMoundCore()
        core._critique = None

        result = await core._query_critique("test", None, 10)

        assert result == []

    @pytest.mark.asyncio
    async def test_query_critique_success(self):
        """Should query critique patterns and convert results."""
        core = KnowledgeMoundCore()
        mock_pattern = MagicMock()
        mock_critique_store = MagicMock()
        mock_critique_store.search_patterns = MagicMock(return_value=[mock_pattern])
        core._critique = mock_critique_store

        mock_item = MagicMock(spec=KnowledgeItem)
        with patch.object(core, "_critique_to_item", return_value=mock_item):
            result = await core._query_critique("test query", None, 10)

        mock_critique_store.search_patterns.assert_called_once_with("test query", limit=10)
        assert len(result) == 1


# ============================================================
# KnowledgeMoundCore Archive Tests
# ============================================================


class TestKnowledgeMoundCoreArchiveNode:
    """Tests for _archive_node method."""

    @pytest.mark.asyncio
    async def test_archive_node_no_get_method(self):
        """Should skip archive when get method not available."""
        core = KnowledgeMoundCore()

        # Should not raise
        await core._archive_node("node_1")

    @pytest.mark.asyncio
    async def test_archive_node_not_found(self):
        """Should skip archive when node not found."""
        core = KnowledgeMoundCore()

        async def mock_get(node_id):
            return None

        core.get = mock_get

        # Should not raise
        await core._archive_node("nonexistent")

    @pytest.mark.asyncio
    async def test_archive_node_async_method(self):
        """Should use async archive method if available."""
        core = KnowledgeMoundCore()

        mock_node = MagicMock()
        mock_node.content = "Test content"
        mock_node.source = MagicMock(value="fact")
        mock_node.source_id = "src_1"
        mock_node.confidence = MagicMock(value="high")
        mock_node.importance = 0.8
        mock_node.metadata = {}
        mock_node.created_at = datetime.now()
        mock_node.updated_at = datetime.now()

        async def mock_get(node_id):
            return mock_node

        core.get = mock_get
        core.workspace_id = "ws_1"

        mock_store = MagicMock()
        mock_store.archive_node_async = AsyncMock()
        core._meta_store = mock_store

        await core._archive_node("node_1")

        mock_store.archive_node_async.assert_called_once()


class TestKnowledgeMoundCoreArchiveNodeWithReason:
    """Tests for _archive_node_with_reason method."""

    @pytest.mark.asyncio
    async def test_archive_node_with_reason_not_found(self):
        """Should skip when node not found."""
        core = KnowledgeMoundCore()

        with patch.object(core, "_get_node", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None
            await core._archive_node_with_reason("node_1", "ws_1", "deduplication")

        mock_get.assert_called_once_with("node_1")


class TestKnowledgeMoundCoreRestoreArchivedNode:
    """Tests for _restore_archived_node method."""

    @pytest.mark.asyncio
    async def test_restore_archived_node_not_found(self):
        """Should return False when archive not found."""
        core = KnowledgeMoundCore()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = None
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_store = MagicMock()
        mock_store.connection.return_value = mock_conn
        core._meta_store = mock_store

        result = await core._restore_archived_node("node_1", "ws_1")

        assert result is False


# ============================================================
# KnowledgeMoundCore Ops Mixin Adapter Tests
# ============================================================


class TestKnowledgeMoundCoreOpsMixinAdapters:
    """Tests for ops mixin adapter methods."""

    @pytest.mark.asyncio
    async def test_get_nodes_for_workspace_async(self):
        """Should use async method if available."""
        core = KnowledgeMoundCore()
        mock_items = [MagicMock(spec=KnowledgeItem)]
        mock_store = MagicMock()
        mock_store.get_nodes_for_workspace_async = AsyncMock(return_value=mock_items)
        core._meta_store = mock_store

        result = await core._get_nodes_for_workspace("ws_1", limit=100)

        mock_store.get_nodes_for_workspace_async.assert_called_once_with("ws_1", 100)
        assert result == mock_items

    @pytest.mark.asyncio
    async def test_count_nodes_async(self):
        """Should use async method if available."""
        core = KnowledgeMoundCore()
        mock_store = MagicMock()
        mock_store.count_nodes_async = AsyncMock(return_value=50)
        core._meta_store = mock_store

        result = await core._count_nodes("ws_1")

        mock_store.count_nodes_async.assert_called_once_with("ws_1")
        assert result == 50

    @pytest.mark.asyncio
    async def test_count_nodes_stats_fallback(self):
        """Should use get_stats as fallback."""
        core = KnowledgeMoundCore()
        # Use spec to ensure count_nodes_async doesn't exist
        mock_store = MagicMock(spec=["get_stats"])
        mock_store.get_stats.return_value = {"total_nodes": 25}
        core._meta_store = mock_store

        result = await core._count_nodes("ws_1")

        assert result == 25

    @pytest.mark.asyncio
    async def test_create_relationship(self):
        """Should delegate to _save_relationship."""
        core = KnowledgeMoundCore()

        with patch.object(core, "_save_relationship", new_callable=AsyncMock) as mock_save:
            await core._create_relationship("src", "tgt", "supports", "ws_1")

        mock_save.assert_called_once_with("src", "tgt", "supports")

    @pytest.mark.asyncio
    async def test_get_node_relationships_for_ops(self):
        """Should delegate to _get_relationships."""
        core = KnowledgeMoundCore()
        mock_links = [MagicMock(spec=KnowledgeLink)]

        with patch.object(core, "_get_relationships", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_links
            result = await core._get_node_relationships_for_ops("node_1", "ws_1")

        mock_get.assert_called_once_with("node_1")
        assert result == mock_links


class TestKnowledgeMoundCorePruneHistory:
    """Tests for prune history methods."""

    @pytest.mark.asyncio
    async def test_get_prune_history_empty(self):
        """Should return empty list when no history exists."""
        core = KnowledgeMoundCore()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_store = MagicMock()
        mock_store.connection.return_value = mock_conn
        core._meta_store = mock_store

        result = await core._get_prune_history("ws_1")

        assert result == []

    @pytest.mark.asyncio
    async def test_save_prune_history(self):
        """Should save prune history to database."""
        core = KnowledgeMoundCore()
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_store = MagicMock()
        mock_store.connection.return_value = mock_conn
        core._meta_store = mock_store

        mock_history = MagicMock()
        mock_history.history_id = "hist_1"
        mock_history.workspace_id = "ws_1"
        mock_history.executed_at = datetime.now()
        mock_history.policy_id = "policy_1"
        mock_history.action = MagicMock(value="archive")
        mock_history.items_pruned = 5
        mock_history.pruned_item_ids = ["id1", "id2"]
        mock_history.reason = "Low confidence"
        mock_history.executed_by = "admin"

        await core._save_prune_history(mock_history)

        # Verify execute was called twice (create table + insert)
        assert mock_conn.execute.call_count == 2


# ============================================================
# KnowledgeMoundCore Converter Tests
# ============================================================


class TestKnowledgeMoundCoreConverters:
    """Tests for converter wrapper methods."""

    def test_node_to_item(self):
        """Should delegate to node_to_item converter."""
        core = KnowledgeMoundCore()
        mock_node = MagicMock()
        mock_node.id = "node_1"
        mock_node.content = "Test"
        mock_node.created_at = datetime.now()
        mock_node.updated_at = datetime.now()
        mock_node.metadata = {}
        mock_node.confidence = 0.8

        with patch("aragora.knowledge.mound.core.node_to_item") as mock_converter:
            mock_item = MagicMock(spec=KnowledgeItem)
            mock_converter.return_value = mock_item

            result = core._node_to_item(mock_node)

        mock_converter.assert_called_once_with(mock_node)
        assert result is mock_item

    def test_rel_to_link(self):
        """Should delegate to relationship_to_link converter."""
        core = KnowledgeMoundCore()
        mock_rel = MagicMock()

        with patch("aragora.knowledge.mound.core.relationship_to_link") as mock_converter:
            mock_link = MagicMock(spec=KnowledgeLink)
            mock_converter.return_value = mock_link

            result = core._rel_to_link(mock_rel)

        mock_converter.assert_called_once_with(mock_rel)
        assert result is mock_link

    def test_continuum_to_item(self):
        """Should delegate to continuum_to_item converter."""
        core = KnowledgeMoundCore()
        mock_entry = MagicMock()

        with patch("aragora.knowledge.mound.core.continuum_to_item") as mock_converter:
            result = core._continuum_to_item(mock_entry)

        mock_converter.assert_called_once_with(mock_entry)

    def test_consensus_to_item(self):
        """Should delegate to consensus_to_item converter."""
        core = KnowledgeMoundCore()
        mock_entry = MagicMock()

        with patch("aragora.knowledge.mound.core.consensus_to_item") as mock_converter:
            result = core._consensus_to_item(mock_entry)

        mock_converter.assert_called_once_with(mock_entry)

    def test_fact_to_item(self):
        """Should delegate to fact_to_item converter."""
        core = KnowledgeMoundCore()
        mock_fact = MagicMock()

        with patch("aragora.knowledge.mound.core.fact_to_item") as mock_converter:
            result = core._fact_to_item(mock_fact)

        mock_converter.assert_called_once_with(mock_fact)

    def test_vector_result_to_item(self):
        """Should delegate to vector_result_to_item converter."""
        core = KnowledgeMoundCore()
        mock_result = MagicMock()

        with patch("aragora.knowledge.mound.core.vector_result_to_item") as mock_converter:
            result = core._vector_result_to_item(mock_result)

        mock_converter.assert_called_once_with(mock_result)

    def test_evidence_to_item(self):
        """Should delegate to evidence_to_item converter."""
        core = KnowledgeMoundCore()
        mock_evidence = MagicMock()

        with patch("aragora.knowledge.mound.core.evidence_to_item") as mock_converter:
            result = core._evidence_to_item(mock_evidence)

        mock_converter.assert_called_once_with(mock_evidence)

    def test_critique_to_item(self):
        """Should delegate to critique_to_item converter."""
        core = KnowledgeMoundCore()
        mock_pattern = MagicMock()

        with patch("aragora.knowledge.mound.core.critique_to_item") as mock_converter:
            result = core._critique_to_item(mock_pattern)

        mock_converter.assert_called_once_with(mock_pattern)


# ============================================================
# KnowledgeMoundCore Search Similar Tests
# ============================================================


class TestKnowledgeMoundCoreSearchSimilar:
    """Tests for _search_similar method."""

    @pytest.mark.asyncio
    async def test_search_similar_async_method(self):
        """Should use async method if available."""
        core = KnowledgeMoundCore()
        mock_results = [MagicMock()]
        mock_store = MagicMock()
        mock_store.search_similar_async = AsyncMock(return_value=mock_results)
        core._meta_store = mock_store

        result = await core._search_similar("ws_1", embedding=[0.1, 0.2], top_k=10, min_score=0.7)

        mock_store.search_similar_async.assert_called_once()
        assert result == mock_results

    @pytest.mark.asyncio
    async def test_search_similar_semantic_store_fallback(self):
        """Should use semantic store when available."""
        core = KnowledgeMoundCore()
        mock_results = [MagicMock(score=0.9)]
        mock_semantic = MagicMock()
        mock_semantic.search = AsyncMock(return_value=mock_results)
        core._semantic_store = mock_semantic
        # Use spec to ensure search_similar_async doesn't exist
        mock_store = MagicMock(spec=[])
        core._meta_store = mock_store

        result = await core._search_similar("ws_1", query="test query", top_k=10, min_score=0.8)

        mock_semantic.search.assert_called_once()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_search_similar_query_local_fallback(self):
        """Should fall back to _query_local."""
        core = KnowledgeMoundCore()
        mock_items = [MagicMock()]
        # Use spec to ensure search_similar_async doesn't exist
        mock_store = MagicMock(spec=[])
        core._meta_store = mock_store
        core._semantic_store = None

        with patch.object(core, "_query_local", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = mock_items
            result = await core._search_similar("ws_1", query="test query", top_k=10, min_score=0.8)

        mock_query.assert_called_once()
        assert len(result) == 1


# ============================================================
# KnowledgeMoundCore Content Hash Tests
# ============================================================


class TestKnowledgeMoundCoreFindByContentHash:
    """Tests for _find_by_content_hash method."""

    @pytest.mark.asyncio
    async def test_find_by_content_hash_async(self):
        """Should use async method if available."""
        core = KnowledgeMoundCore()
        mock_store = MagicMock()
        mock_store.find_by_content_hash_async = AsyncMock(return_value="node_1")
        core._meta_store = mock_store

        result = await core._find_by_content_hash("abc123", "ws_1")

        mock_store.find_by_content_hash_async.assert_called_once_with("abc123", "ws_1")
        assert result == "node_1"

    @pytest.mark.asyncio
    async def test_find_by_content_hash_sync_fallback(self):
        """Should use sync method as fallback."""
        core = KnowledgeMoundCore()
        mock_node = MagicMock()
        mock_node.id = "node_1"
        # Use spec to ensure find_by_content_hash_async doesn't exist
        mock_store = MagicMock(spec=["find_by_content_hash"])
        mock_store.find_by_content_hash = MagicMock(return_value=mock_node)
        core._meta_store = mock_store

        result = await core._find_by_content_hash("abc123", "ws_1")

        mock_store.find_by_content_hash.assert_called_once_with("abc123", "ws_1")
        assert result == "node_1"

    @pytest.mark.asyncio
    async def test_find_by_content_hash_not_found(self):
        """Should return None when not found."""
        core = KnowledgeMoundCore()
        # Use spec to ensure find_by_content_hash_async doesn't exist
        mock_store = MagicMock(spec=["find_by_content_hash"])
        mock_store.find_by_content_hash = MagicMock(return_value=None)
        core._meta_store = mock_store

        result = await core._find_by_content_hash("abc123", "ws_1")

        assert result is None


# ============================================================
# KnowledgeMoundCore Get Nodes By Content Hash Tests
# ============================================================


class TestKnowledgeMoundCoreGetNodesByContentHash:
    """Tests for _get_nodes_by_content_hash method."""

    @pytest.mark.asyncio
    async def test_get_nodes_by_content_hash_async(self):
        """Should use async method if available."""
        core = KnowledgeMoundCore()
        mock_result = {"hash1": ["id1", "id2"], "hash2": ["id3"]}
        mock_store = MagicMock()
        mock_store.get_nodes_by_content_hash_async = AsyncMock(return_value=mock_result)
        core._meta_store = mock_store

        result = await core._get_nodes_by_content_hash("ws_1")

        mock_store.get_nodes_by_content_hash_async.assert_called_once_with("ws_1")
        assert result == mock_result

    @pytest.mark.asyncio
    async def test_get_nodes_by_content_hash_sync_fallback(self):
        """Should build hash map from query_nodes."""
        core = KnowledgeMoundCore()
        mock_node1 = MagicMock()
        mock_node1.id = "id1"
        mock_node1.content_hash = "hash1"
        mock_node2 = MagicMock()
        mock_node2.id = "id2"
        mock_node2.content_hash = "hash1"
        # Use spec to ensure get_nodes_by_content_hash_async doesn't exist
        mock_store = MagicMock(spec=["query_nodes"])
        mock_store.query_nodes = MagicMock(return_value=[mock_node1, mock_node2])
        core._meta_store = mock_store

        result = await core._get_nodes_by_content_hash("ws_1")

        assert "hash1" in result
        assert len(result["hash1"]) == 2


# ============================================================
# KnowledgeMoundCore Increment Update Count Tests
# ============================================================


class TestKnowledgeMoundCoreIncrementUpdateCount:
    """Tests for _increment_update_count method."""

    @pytest.mark.asyncio
    async def test_increment_update_count(self):
        """Should call _update_node with increment expression."""
        core = KnowledgeMoundCore()

        with patch.object(core, "_update_node", new_callable=AsyncMock) as mock_update:
            await core._increment_update_count("node_1")

        mock_update.assert_called_once_with("node_1", {"update_count": "update_count + 1"})
