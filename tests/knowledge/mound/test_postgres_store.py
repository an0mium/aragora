"""
Comprehensive tests for the PostgresStore.

Tests cover:
1. Initialization and configuration
2. Connection management and pooling
3. Node CRUD operations
4. Relationship operations
5. Query and filter operations
6. Culture pattern operations
7. Access grant operations (visibility)
8. Statistics and metrics
9. Error handling
10. Concurrent access

Run with: pytest tests/knowledge/mound/test_postgres_store.py -v
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import types directly from unified types to avoid package import issues
from aragora.knowledge.unified.types import (
    ConfidenceLevel,
    KnowledgeItem,
    KnowledgeLink,
    KnowledgeSource,
    QueryFilters,
    RelationshipType,
)
from aragora.knowledge.mound.types import (
    AccessGrant,
    AccessGrantType,
    MoundStats,
    VisibilityLevel,
)


# ============================================================================
# Mock Classes and Fixtures
# ============================================================================


@dataclass
class MockRecord:
    """Mock asyncpg Record that supports item access."""

    _data: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self._data.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)


class MockConnection:
    """Mock asyncpg connection with full async interface."""

    def __init__(self):
        self.executed_queries: list[tuple[str, tuple]] = []
        self._mock_data: dict[str, Any] = {}
        self._fetchrow_result: MockRecord | None = None
        self._fetch_result: list[MockRecord] = []
        self._fetchval_result: Any = None
        self._execute_result: str = "INSERT 1"

    async def execute(self, query: str, *args) -> str:
        """Mock execute that tracks calls."""
        self.executed_queries.append((query, args))
        return self._execute_result

    async def executemany(self, query: str, args_list: list) -> None:
        """Mock executemany for batch operations."""
        for args in args_list:
            self.executed_queries.append((query, tuple(args)))

    async def fetch(self, query: str, *args) -> list[MockRecord]:
        """Mock fetch for multi-row queries."""
        self.executed_queries.append((query, args))
        return self._fetch_result

    async def fetchrow(self, query: str, *args) -> MockRecord | None:
        """Mock fetchrow for single-row queries."""
        self.executed_queries.append((query, args))
        return self._fetchrow_result

    async def fetchval(self, query: str, *args) -> Any:
        """Mock fetchval for single-value queries."""
        self.executed_queries.append((query, args))
        return self._fetchval_result

    def set_fetchrow_result(self, data: dict[str, Any] | None) -> None:
        """Set the result for fetchrow calls."""
        if data is None:
            self._fetchrow_result = None
        else:
            self._fetchrow_result = MockRecord(_data=data)

    def set_fetch_result(self, data_list: list[dict[str, Any]]) -> None:
        """Set the result for fetch calls."""
        self._fetch_result = [MockRecord(_data=d) for d in data_list]

    def set_fetchval_result(self, value: Any) -> None:
        """Set the result for fetchval calls."""
        self._fetchval_result = value

    def set_execute_result(self, result: str) -> None:
        """Set the result for execute calls."""
        self._execute_result = result


class MockPool:
    """Mock asyncpg connection pool."""

    def __init__(self):
        self._connection = MockConnection()
        self._closed = False
        self._acquire_count = 0
        self._release_count = 0

    def acquire(self):
        """Return context manager for connection."""
        self._acquire_count += 1
        return MockPoolAcquireContext(self)

    async def close(self) -> None:
        """Close the pool."""
        self._closed = True


class MockPoolAcquireContext:
    """Mock context manager for pool.acquire()."""

    def __init__(self, pool: MockPool):
        self._pool = pool

    async def __aenter__(self) -> MockConnection:
        return self._pool._connection

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._pool._release_count += 1


@pytest.fixture
def mock_pool():
    """Create a mock connection pool."""
    return MockPool()


@pytest.fixture
def mock_connection(mock_pool):
    """Get the mock connection from the pool."""
    return mock_pool._connection


def _get_postgres_store_class():
    """Helper to import PostgresStore class directly from module."""
    # Import directly from the module to avoid package __init__ issues
    import importlib.util
    import os

    module_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "aragora",
        "knowledge",
        "mound",
        "postgres_store.py",
    )
    spec = importlib.util.spec_from_file_location("postgres_store", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PostgresStore, module.POSTGRES_SCHEMA


@pytest.fixture
def postgres_store(mock_pool):
    """Create a PostgresStore with mocked asyncpg."""
    PostgresStore, _ = _get_postgres_store_class()
    store = PostgresStore(
        url="postgresql://test:test@localhost:5432/testdb",
        pool_size=10,
        max_overflow=5,
    )
    # Inject mock pool
    store._pool = mock_pool
    store._initialized = True
    return store


@pytest.fixture
def uninitialized_store():
    """Create an uninitialized PostgresStore."""
    PostgresStore, _ = _get_postgres_store_class()
    return PostgresStore(
        url="postgresql://test:test@localhost:5432/testdb",
    )


@pytest.fixture
def sample_node_data():
    """Sample node data for testing."""
    return {
        "id": "km_test_node_1",
        "workspace_id": "workspace_1",
        "node_type": "fact",
        "content": "Test content for the node",
        "content_hash": "abc123hash",
        "confidence": 0.85,
        "tier": "slow",
        "surprise_score": 0.1,
        "update_count": 1,
        "consolidation_score": 0.0,
        "validation_status": "verified",
        "consensus_proof_id": None,
        "staleness_score": 0.0,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "metadata": {"source": "test"},
    }


@pytest.fixture
def sample_db_row():
    """Sample database row for testing node retrieval."""
    return {
        "id": "km_test_node_1",
        "workspace_id": "workspace_1",
        "node_type": "fact",
        "content": "Test content for the node",
        "content_hash": "abc123hash",
        "confidence": 0.85,
        "tier": "slow",
        "surprise_score": 0.1,
        "update_count": 1,
        "consolidation_score": 0.0,
        "validation_status": "verified",
        "consensus_proof_id": None,
        "last_validated_at": None,
        "staleness_score": 0.0,
        "revalidation_requested": False,
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "metadata": json.dumps({"source": "test"}),
        "visibility": "workspace",
        "visibility_set_by": None,
        "is_discoverable": True,
    }


# ============================================================================
# Initialization Tests
# ============================================================================


class TestInitialization:
    """Tests for PostgresStore initialization."""

    def test_init_with_default_parameters(self):
        """Should initialize with default pool parameters."""
        PostgresStore, _ = _get_postgres_store_class()
        store = PostgresStore(url="postgresql://localhost/test")

        assert store._url == "postgresql://localhost/test"
        assert store._pool_size == 10
        assert store._max_overflow == 5
        assert store._pool is None
        assert store._initialized is False

    def test_init_with_custom_pool_size(self):
        """Should initialize with custom pool size."""
        PostgresStore, _ = _get_postgres_store_class()
        store = PostgresStore(
            url="postgresql://localhost/test",
            pool_size=20,
            max_overflow=10,
        )

        assert store._pool_size == 20
        assert store._max_overflow == 10

    @pytest.mark.asyncio
    async def test_initialize_creates_pool(self):
        """Should create connection pool on initialize."""
        mock_pool = MockPool()
        PostgresStore, _ = _get_postgres_store_class()

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_pool

            store = PostgresStore(url="postgresql://localhost/test")
            await store.initialize()

            assert store._initialized is True
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_double_initialization_is_safe(self, postgres_store):
        """Should be safe to call initialize multiple times."""
        # Store is already initialized via fixture
        assert postgres_store._initialized is True

        # Call initialize again - should not raise
        await postgres_store.initialize()

        assert postgres_store._initialized is True

    @pytest.mark.asyncio
    async def test_close_cleans_up_pool(self, postgres_store, mock_pool):
        """Should close pool and reset state on close."""
        assert postgres_store._initialized is True
        assert postgres_store._pool is not None

        await postgres_store.close()

        assert postgres_store._initialized is False
        assert postgres_store._pool is None
        assert mock_pool._closed is True


# ============================================================================
# Connection Management Tests
# ============================================================================


class TestConnectionManagement:
    """Tests for connection management."""

    @pytest.mark.asyncio
    async def test_connection_context_manager(self, postgres_store, mock_pool):
        """Should provide connection via context manager."""
        async with postgres_store.connection() as conn:
            assert conn is mock_pool._connection

        assert mock_pool._acquire_count == 1
        assert mock_pool._release_count == 1

    @pytest.mark.asyncio
    async def test_connection_raises_if_not_initialized(self, uninitialized_store):
        """Should raise RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            async with uninitialized_store.connection():
                pass

    @pytest.mark.asyncio
    async def test_multiple_connections_sequential(self, postgres_store, mock_pool):
        """Should handle multiple sequential connection requests."""
        for i in range(5):
            async with postgres_store.connection() as conn:
                assert conn is not None

        assert mock_pool._acquire_count == 5
        assert mock_pool._release_count == 5


# ============================================================================
# Node CRUD Operations Tests
# ============================================================================


class TestNodeOperations:
    """Tests for node CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_node_async_creates_node(
        self, postgres_store, mock_connection, sample_node_data
    ):
        """Should save a new node to the database."""
        node_id = await postgres_store.save_node_async(sample_node_data)

        assert node_id == sample_node_data["id"]
        assert len(mock_connection.executed_queries) >= 1

        # Check that INSERT query was executed
        query, _ = mock_connection.executed_queries[0]
        assert "INSERT INTO knowledge_nodes" in query

    @pytest.mark.asyncio
    async def test_save_node_async_with_metadata(
        self, postgres_store, mock_connection, sample_node_data
    ):
        """Should save node with complex metadata."""
        sample_node_data["metadata"] = {
            "tags": ["important", "verified"],
            "source_url": "https://example.com",
            "nested": {"key": "value"},
        }

        node_id = await postgres_store.save_node_async(sample_node_data)

        assert node_id == sample_node_data["id"]
        # Verify metadata was serialized
        query, args = mock_connection.executed_queries[0]
        # Last argument should be the JSON-serialized metadata
        metadata_arg = args[-1]
        assert "tags" in metadata_arg

    @pytest.mark.asyncio
    async def test_save_node_async_with_provenance(
        self, postgres_store, mock_connection, sample_node_data
    ):
        """Should save provenance when source_type is provided."""
        sample_node_data["source_type"] = "debate"
        sample_node_data["debate_id"] = "debate_123"
        sample_node_data["agent_id"] = "claude"

        await postgres_store.save_node_async(sample_node_data)

        # Should have both node insert and provenance insert
        queries = [q[0] for q in mock_connection.executed_queries]
        assert any("knowledge_nodes" in q for q in queries)
        assert any("provenance_chains" in q for q in queries)

    @pytest.mark.asyncio
    async def test_save_node_async_with_topics(
        self, postgres_store, mock_connection, sample_node_data
    ):
        """Should save topics as separate records."""
        sample_node_data["topics"] = ["legal", "contracts", "termination"]

        await postgres_store.save_node_async(sample_node_data)

        # Should have node insert, delete old topics, and insert new topics
        queries = [q[0] for q in mock_connection.executed_queries]
        assert any("DELETE FROM node_topics" in q for q in queries)
        assert any("INSERT INTO node_topics" in q for q in queries)

    @pytest.mark.asyncio
    async def test_get_node_async_returns_node(
        self, postgres_store, mock_connection, sample_db_row
    ):
        """Should return KnowledgeItem when node exists."""
        mock_connection.set_fetchrow_result(sample_db_row)
        mock_connection.set_fetch_result([{"topic": "test_topic"}])

        result = await postgres_store.get_node_async("km_test_node_1")

        assert result is not None
        assert isinstance(result, KnowledgeItem)
        assert result.id == "km_test_node_1"
        assert result.content == "Test content for the node"

    @pytest.mark.asyncio
    async def test_get_node_async_returns_none_for_missing(self, postgres_store, mock_connection):
        """Should return None when node doesn't exist."""
        mock_connection.set_fetchrow_result(None)

        result = await postgres_store.get_node_async("nonexistent_node")

        assert result is None

    @pytest.mark.asyncio
    async def test_update_node_async_modifies_node(self, postgres_store, mock_connection):
        """Should update node with provided fields."""
        updates = {
            "content": "Updated content",
            "confidence": 0.95,
            "validation_status": "verified",
        }

        await postgres_store.update_node_async("km_test_node_1", updates)

        assert len(mock_connection.executed_queries) == 1
        query, args = mock_connection.executed_queries[0]
        assert "UPDATE knowledge_nodes" in query
        assert args[0] == "km_test_node_1"

    @pytest.mark.asyncio
    async def test_update_node_async_with_increment(self, postgres_store, mock_connection):
        """Should handle special update_count increment."""
        updates = {"update_count": "update_count + 1"}

        await postgres_store.update_node_async("km_test_node_1", updates)

        query, _ = mock_connection.executed_queries[0]
        assert "update_count = update_count + 1" in query

    @pytest.mark.asyncio
    async def test_update_node_async_empty_updates(self, postgres_store, mock_connection):
        """Should not execute query for empty updates."""
        await postgres_store.update_node_async("km_test_node_1", {})

        assert len(mock_connection.executed_queries) == 0

    @pytest.mark.asyncio
    async def test_delete_node_async_removes_node(self, postgres_store, mock_connection):
        """Should delete node and return True on success."""
        mock_connection.set_execute_result("DELETE 1")

        result = await postgres_store.delete_node_async("km_test_node_1")

        assert result is True
        query, args = mock_connection.executed_queries[0]
        assert "DELETE FROM knowledge_nodes" in query
        assert args[0] == "km_test_node_1"

    @pytest.mark.asyncio
    async def test_delete_node_async_returns_false_for_missing(
        self, postgres_store, mock_connection
    ):
        """Should return False when node doesn't exist."""
        mock_connection.set_execute_result("DELETE 0")

        result = await postgres_store.delete_node_async("nonexistent_node")

        assert result is False

    @pytest.mark.asyncio
    async def test_find_by_content_hash_async_found(self, postgres_store, mock_connection):
        """Should return node ID when content hash matches."""
        mock_connection.set_fetchrow_result({"id": "km_found_node"})

        result = await postgres_store.find_by_content_hash_async("abc123hash", "workspace_1")

        assert result == "km_found_node"

    @pytest.mark.asyncio
    async def test_find_by_content_hash_async_not_found(self, postgres_store, mock_connection):
        """Should return None when content hash not found."""
        mock_connection.set_fetchrow_result(None)

        result = await postgres_store.find_by_content_hash_async("nonexistent_hash", "workspace_1")

        assert result is None


# ============================================================================
# Relationship Operations Tests
# ============================================================================


class TestRelationshipOperations:
    """Tests for relationship operations."""

    @pytest.mark.asyncio
    async def test_save_relationship_async_creates_relationship(
        self, postgres_store, mock_connection
    ):
        """Should create a relationship between nodes."""
        rel_id = await postgres_store.save_relationship_async(
            from_id="km_node_a",
            to_id="km_node_b",
            rel_type="supports",
        )

        assert rel_id.startswith("kr_")
        query, args = mock_connection.executed_queries[0]
        assert "INSERT INTO knowledge_relationships" in query
        assert args[1] == "km_node_a"
        assert args[2] == "km_node_b"
        assert args[3] == "supports"

    @pytest.mark.asyncio
    async def test_get_relationships_async_returns_relationships(
        self, postgres_store, mock_connection
    ):
        """Should return relationships for a node."""
        mock_connection.set_fetch_result(
            [
                {
                    "id": "kr_1",
                    "from_node_id": "km_a",
                    "to_node_id": "km_b",
                    "relationship_type": "supports",
                    "strength": 0.8,
                    "created_by": "claude",
                    "created_at": datetime.now(),
                    "metadata": json.dumps({}),
                }
            ]
        )

        result = await postgres_store.get_relationships_async("km_a")

        assert len(result) == 1
        assert isinstance(result[0], KnowledgeLink)
        assert result[0].source_id == "km_a"
        assert result[0].target_id == "km_b"
        assert result[0].relationship == RelationshipType.SUPPORTS

    @pytest.mark.asyncio
    async def test_get_relationships_async_with_type_filter(self, postgres_store, mock_connection):
        """Should filter relationships by type."""
        mock_connection.set_fetch_result([])

        await postgres_store.get_relationships_async(
            "km_a",
            types=[RelationshipType.SUPPORTS, RelationshipType.CONTRADICTS],
        )

        query, args = mock_connection.executed_queries[0]
        assert "relationship_type = ANY" in query

    @pytest.mark.asyncio
    async def test_get_relationships_async_empty_result(self, postgres_store, mock_connection):
        """Should return empty list when no relationships exist."""
        mock_connection.set_fetch_result([])

        result = await postgres_store.get_relationships_async("km_isolated")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_relationships_batch_async(self, postgres_store, mock_connection):
        """Should fetch relationships for multiple nodes in one query."""
        mock_connection.set_fetch_result(
            [
                {
                    "id": "kr_1",
                    "from_node_id": "km_a",
                    "to_node_id": "km_b",
                    "relationship_type": "supports",
                    "strength": 0.8,
                    "created_by": None,
                    "created_at": datetime.now(),
                    "metadata": json.dumps({}),
                }
            ]
        )

        result = await postgres_store.get_relationships_batch_async(["km_a", "km_b", "km_c"])

        assert "km_a" in result
        assert "km_b" in result
        assert "km_c" in result
        assert len(result["km_a"]) == 1

    @pytest.mark.asyncio
    async def test_get_relationships_batch_async_empty_input(self, postgres_store, mock_connection):
        """Should return empty dict for empty node list."""
        result = await postgres_store.get_relationships_batch_async([])

        assert result == {}
        assert len(mock_connection.executed_queries) == 0


# ============================================================================
# Query Operations Tests
# ============================================================================


class TestQueryOperations:
    """Tests for query and filter operations."""

    @pytest.mark.asyncio
    async def test_query_async_with_full_text_search(
        self, postgres_store, mock_connection, sample_db_row
    ):
        """Should query with PostgreSQL full-text search."""
        mock_connection.set_fetch_result([sample_db_row])

        results = await postgres_store.query_async(
            query="test content",
            filters=None,
            limit=10,
            workspace_id="workspace_1",
        )

        assert len(results) == 1
        query, _ = mock_connection.executed_queries[0]
        assert "to_tsvector" in query
        assert "plainto_tsquery" in query

    @pytest.mark.asyncio
    async def test_query_async_respects_limit(self, postgres_store, mock_connection):
        """Should respect the limit parameter."""
        mock_connection.set_fetch_result([])

        await postgres_store.query_async(
            query="test",
            filters=None,
            limit=5,
            workspace_id="workspace_1",
        )

        query, args = mock_connection.executed_queries[0]
        assert "LIMIT" in query
        assert args[2] == 5  # limit is the third argument

    @pytest.mark.asyncio
    async def test_query_async_filters_by_workspace(self, postgres_store, mock_connection):
        """Should filter results by workspace_id."""
        mock_connection.set_fetch_result([])

        await postgres_store.query_async(
            query="test",
            filters=None,
            limit=10,
            workspace_id="specific_workspace",
        )

        query, args = mock_connection.executed_queries[0]
        assert "workspace_id" in query
        assert args[1] == "specific_workspace"


# ============================================================================
# Statistics Tests
# ============================================================================


class TestStatistics:
    """Tests for statistics operations."""

    @pytest.mark.asyncio
    async def test_get_stats_async_returns_mound_stats(self, postgres_store, mock_connection):
        """Should return comprehensive statistics."""
        # Set up mock responses for all stat queries
        mock_connection._fetchval_result = 100  # total count
        mock_connection.set_fetch_result(
            [
                {"node_type": "fact", "count": 50},
                {"node_type": "claim", "count": 30},
            ]
        )

        stats = await postgres_store.get_stats_async("workspace_1")

        assert isinstance(stats, MoundStats)
        assert stats.workspace_id == "workspace_1"

    @pytest.mark.asyncio
    async def test_get_stats_async_handles_empty_workspace(self, postgres_store, mock_connection):
        """Should handle workspace with no nodes."""
        mock_connection._fetchval_result = 0
        mock_connection.set_fetch_result([])

        stats = await postgres_store.get_stats_async("empty_workspace")

        assert stats.total_nodes == 0
        assert stats.nodes_by_type == {}


# ============================================================================
# Culture Pattern Tests
# ============================================================================


class TestCulturePatterns:
    """Tests for culture pattern operations."""

    @pytest.mark.asyncio
    async def test_save_culture_pattern_async(self, postgres_store, mock_connection):
        """Should save a culture pattern."""
        pattern = {
            "id": "cp_1",
            "workspace_id": "workspace_1",
            "pattern_type": "decision_style",
            "pattern_key": "consensus_preference",
            "pattern_value": {"preference": "unanimous"},
            "observation_count": 5,
            "confidence": 0.85,
            "first_observed_at": datetime.now(),
            "last_observed_at": datetime.now(),
            "contributing_debates": ["debate_1", "debate_2"],
            "metadata": {},
        }

        result = await postgres_store.save_culture_pattern_async(pattern)

        assert result == "cp_1"
        query, _ = mock_connection.executed_queries[0]
        assert "INSERT INTO culture_patterns" in query

    @pytest.mark.asyncio
    async def test_get_culture_patterns_async_all_types(self, postgres_store, mock_connection):
        """Should get all culture patterns for workspace."""
        mock_connection.set_fetch_result(
            [
                {
                    "id": "cp_1",
                    "workspace_id": "workspace_1",
                    "pattern_type": "decision_style",
                    "pattern_key": "key_1",
                    "pattern_value": json.dumps({"val": 1}),
                    "observation_count": 5,
                    "confidence": 0.8,
                    "first_observed_at": datetime.now(),
                    "last_observed_at": datetime.now(),
                    "contributing_debates": ["d1"],
                    "metadata": json.dumps({}),
                }
            ]
        )

        result = await postgres_store.get_culture_patterns_async("workspace_1")

        assert len(result) == 1
        assert result[0]["pattern_type"] == "decision_style"

    @pytest.mark.asyncio
    async def test_get_culture_patterns_async_with_type_filter(
        self, postgres_store, mock_connection
    ):
        """Should filter patterns by type."""
        mock_connection.set_fetch_result([])

        await postgres_store.get_culture_patterns_async(
            "workspace_1",
            pattern_type="decision_style",
        )

        query, args = mock_connection.executed_queries[0]
        assert "pattern_type = $2" in query
        assert args[1] == "decision_style"


# ============================================================================
# Access Grant Tests (Visibility)
# ============================================================================


class TestAccessGrants:
    """Tests for access grant operations."""

    @pytest.mark.asyncio
    async def test_save_access_grant_async(self, postgres_store, mock_connection):
        """Should save an access grant."""
        grant = AccessGrant(
            id="grant_1",
            item_id="km_node_1",
            grantee_type=AccessGrantType.USER,
            grantee_id="user_123",
            permissions=["read", "write"],
            granted_by="admin",
            granted_at=datetime.now(),
            expires_at=None,
        )

        result = await postgres_store.save_access_grant_async(grant)

        assert result == "grant_1"
        query, _ = mock_connection.executed_queries[0]
        assert "INSERT INTO access_grants" in query

    @pytest.mark.asyncio
    async def test_get_access_grants_async(self, postgres_store, mock_connection):
        """Should get access grants for an item."""
        mock_connection.set_fetch_result(
            [
                {
                    "id": "grant_1",
                    "item_id": "km_node_1",
                    "grantee_type": "user",
                    "grantee_id": "user_123",
                    "permissions": ["read"],
                    "granted_by": "admin",
                    "granted_at": datetime.now(),
                    "expires_at": None,
                }
            ]
        )

        result = await postgres_store.get_access_grants_async("km_node_1")

        assert len(result) == 1
        assert isinstance(result[0], AccessGrant)
        assert result[0].grantee_id == "user_123"

    @pytest.mark.asyncio
    async def test_get_grants_for_grantee_async(self, postgres_store, mock_connection):
        """Should get grants for a specific grantee."""
        mock_connection.set_fetch_result([])

        await postgres_store.get_grants_for_grantee_async("user_123")

        query, args = mock_connection.executed_queries[0]
        assert "grantee_id = $1" in query
        assert args[0] == "user_123"

    @pytest.mark.asyncio
    async def test_get_grants_for_grantee_async_with_type(self, postgres_store, mock_connection):
        """Should filter grants by grantee type."""
        mock_connection.set_fetch_result([])

        await postgres_store.get_grants_for_grantee_async(
            "user_123",
            grantee_type=AccessGrantType.USER,
        )

        query, args = mock_connection.executed_queries[0]
        assert "grantee_type = $2" in query
        assert args[1] == "user"

    @pytest.mark.asyncio
    async def test_delete_access_grant_async(self, postgres_store, mock_connection):
        """Should delete an access grant."""
        mock_connection.set_execute_result("DELETE 1")

        result = await postgres_store.delete_access_grant_async("km_node_1", "user_123")

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_access_grant_async_not_found(self, postgres_store, mock_connection):
        """Should return False when grant doesn't exist."""
        mock_connection.set_execute_result("DELETE 0")

        result = await postgres_store.delete_access_grant_async("km_node_1", "nonexistent_user")

        assert result is False


# ============================================================================
# Visibility Query Tests
# ============================================================================


class TestVisibilityQueries:
    """Tests for visibility-aware queries."""

    @pytest.mark.asyncio
    async def test_query_with_visibility_async(
        self, postgres_store, mock_connection, sample_db_row
    ):
        """Should query with visibility filtering."""
        sample_db_row["visibility"] = "public"
        mock_connection.set_fetch_result([sample_db_row])

        results = await postgres_store.query_with_visibility_async(
            query="test",
            workspace_id="workspace_1",
            actor_id="user_123",
            actor_workspace_id="workspace_1",
            actor_org_id="org_1",
            limit=10,
        )

        assert len(results) == 1
        query, _ = mock_connection.executed_queries[0]
        assert "visibility" in query.lower()

    @pytest.mark.asyncio
    async def test_update_visibility_async(self, postgres_store, mock_connection):
        """Should update node visibility."""
        await postgres_store.update_visibility_async(
            node_id="km_node_1",
            visibility=VisibilityLevel.PUBLIC,
            set_by="admin",
        )

        query, args = mock_connection.executed_queries[0]
        assert "UPDATE knowledge_nodes" in query
        assert "visibility" in query
        assert args[1] == "public"


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_handles_asyncpg_import_error(self):
        """Should raise ImportError with helpful message when asyncpg missing."""
        PostgresStore, _ = _get_postgres_store_class()
        store = PostgresStore(url="postgresql://localhost/test")

        # Patch asyncpg import to raise ImportError during initialize
        with patch.dict(sys.modules, {"asyncpg": None}):
            with pytest.raises(ImportError, match="asyncpg required"):
                await store.initialize()

    @pytest.mark.asyncio
    async def test_handles_connection_error(self):
        """Should handle connection errors gracefully."""
        PostgresStore, _ = _get_postgres_store_class()

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = ConnectionError("Connection refused")

            store = PostgresStore(url="postgresql://localhost/test")

            with pytest.raises(ConnectionError):
                await store.initialize()

    @pytest.mark.asyncio
    async def test_handles_timeout_error(self):
        """Should handle timeout errors gracefully."""
        PostgresStore, _ = _get_postgres_store_class()

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = TimeoutError("Connection timed out")

            store = PostgresStore(url="postgresql://localhost/test")

            with pytest.raises(TimeoutError):
                await store.initialize()


# ============================================================================
# Validation Helper Tests
# ============================================================================


class TestValidationHelpers:
    """Tests for validation helper methods."""

    def test_validation_to_confidence_verified(self, postgres_store):
        """Should map 'verified' to VERIFIED confidence."""
        result = postgres_store._validation_to_confidence("verified")
        assert result == ConfidenceLevel.VERIFIED

    def test_validation_to_confidence_majority_agreed(self, postgres_store):
        """Should map 'majority_agreed' to HIGH confidence."""
        result = postgres_store._validation_to_confidence("majority_agreed")
        assert result == ConfidenceLevel.HIGH

    def test_validation_to_confidence_unverified(self, postgres_store):
        """Should map 'unverified' to MEDIUM confidence."""
        result = postgres_store._validation_to_confidence("unverified")
        assert result == ConfidenceLevel.MEDIUM

    def test_validation_to_confidence_contradicted(self, postgres_store):
        """Should map 'contradicted' to LOW confidence."""
        result = postgres_store._validation_to_confidence("contradicted")
        assert result == ConfidenceLevel.LOW

    def test_validation_to_confidence_unknown_status(self, postgres_store):
        """Should default to MEDIUM for unknown status."""
        result = postgres_store._validation_to_confidence("unknown_status")
        assert result == ConfidenceLevel.MEDIUM

    def test_validation_to_confidence_case_insensitive(self, postgres_store):
        """Should handle case-insensitive status."""
        result = postgres_store._validation_to_confidence("VERIFIED")
        assert result == ConfidenceLevel.VERIFIED


# ============================================================================
# Concurrent Access Tests
# ============================================================================


class TestConcurrentAccess:
    """Tests for concurrent access patterns."""

    @pytest.mark.asyncio
    async def test_concurrent_node_saves(self, postgres_store, mock_connection, sample_node_data):
        """Should handle concurrent node saves."""

        async def save_node(i: int):
            data = sample_node_data.copy()
            data["id"] = f"km_node_{i}"
            data["content"] = f"Content for node {i}"
            return await postgres_store.save_node_async(data)

        # Run concurrent saves
        tasks = [save_node(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert all(r.startswith("km_node_") for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_reads_and_writes(
        self, postgres_store, mock_connection, sample_node_data, sample_db_row
    ):
        """Should handle concurrent reads and writes."""
        mock_connection.set_fetchrow_result(sample_db_row)
        mock_connection.set_fetch_result([{"topic": "test"}])

        async def write_node(i: int):
            data = sample_node_data.copy()
            data["id"] = f"km_write_{i}"
            return await postgres_store.save_node_async(data)

        async def read_node():
            return await postgres_store.get_node_async("km_test_node_1")

        # Mix reads and writes
        tasks = []
        for i in range(5):
            tasks.append(write_node(i))
            tasks.append(read_node())

        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        # Check that reads returned nodes
        read_results = [r for r in results if isinstance(r, KnowledgeItem)]
        assert len(read_results) == 5


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_save_node_with_empty_metadata(
        self, postgres_store, mock_connection, sample_node_data
    ):
        """Should handle empty metadata dict."""
        sample_node_data["metadata"] = {}

        node_id = await postgres_store.save_node_async(sample_node_data)

        assert node_id is not None

    @pytest.mark.asyncio
    async def test_save_node_with_none_optional_fields(self, postgres_store, mock_connection):
        """Should handle None values for optional fields."""
        minimal_data = {
            "id": "km_minimal",
            "workspace_id": "workspace_1",
            "content": "Minimal content",
        }

        node_id = await postgres_store.save_node_async(minimal_data)

        assert node_id == "km_minimal"

    @pytest.mark.asyncio
    async def test_query_with_special_characters(self, postgres_store, mock_connection):
        """Should handle special characters in queries."""
        mock_connection.set_fetch_result([])

        # Query with special characters
        await postgres_store.query_async(
            query='test\'s "content" & (special)',
            filters=None,
            limit=10,
            workspace_id="workspace_1",
        )

        # Should execute without error
        assert len(mock_connection.executed_queries) == 1

    @pytest.mark.asyncio
    async def test_unicode_content_handling(
        self, postgres_store, mock_connection, sample_node_data
    ):
        """Should handle unicode content correctly."""
        sample_node_data["content"] = "Unicode content: \u2764 \u2728 \u4e2d\u6587"

        node_id = await postgres_store.save_node_async(sample_node_data)

        assert node_id is not None

    @pytest.mark.asyncio
    async def test_very_long_content(self, postgres_store, mock_connection, sample_node_data):
        """Should handle very long content."""
        sample_node_data["content"] = "Long content. " * 10000

        node_id = await postgres_store.save_node_async(sample_node_data)

        assert node_id is not None

    @pytest.mark.asyncio
    async def test_access_grant_with_expiry(self, postgres_store, mock_connection):
        """Should handle access grants with expiry dates."""
        grant = AccessGrant(
            id="grant_expiring",
            item_id="km_node_1",
            grantee_type=AccessGrantType.USER,
            grantee_id="user_123",
            permissions=["read"],
            granted_by="admin",
            granted_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=30),
        )

        result = await postgres_store.save_access_grant_async(grant)

        assert result == "grant_expiring"
        query, args = mock_connection.executed_queries[0]
        assert args[7] is not None  # expires_at should be set


# ============================================================================
# Schema Constant Tests
# ============================================================================


class TestSchemaConstants:
    """Tests for schema constants."""

    def test_postgres_schema_defined(self):
        """Should have POSTGRES_SCHEMA constant defined."""
        _, POSTGRES_SCHEMA = _get_postgres_store_class()

        assert POSTGRES_SCHEMA is not None
        assert "CREATE TABLE" in POSTGRES_SCHEMA
        assert "knowledge_nodes" in POSTGRES_SCHEMA
        assert "knowledge_relationships" in POSTGRES_SCHEMA
        assert "provenance_chains" in POSTGRES_SCHEMA

    def test_schema_includes_indexes(self):
        """Should include index definitions."""
        _, POSTGRES_SCHEMA = _get_postgres_store_class()

        assert "CREATE INDEX" in POSTGRES_SCHEMA
        assert "idx_nodes_workspace" in POSTGRES_SCHEMA

    def test_schema_includes_federation_tables(self):
        """Should include federation tables."""
        _, POSTGRES_SCHEMA = _get_postgres_store_class()

        assert "federation_nodes" in POSTGRES_SCHEMA
        assert "federation_sync_state" in POSTGRES_SCHEMA
        assert "distributed_locks" in POSTGRES_SCHEMA


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
