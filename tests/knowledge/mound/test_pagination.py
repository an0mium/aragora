"""
Tests for database pagination in PostgresStore.

Tests cover:
1. PaginatedResult dataclass
2. Offset-based pagination
3. Cursor-based pagination
4. Limit enforcement and defaults
5. Edge cases (empty results, last page, etc.)
6. Query pagination (full-text search)
7. Relationship pagination
8. Culture pattern pagination

Run with: pytest tests/knowledge/mound/test_pagination.py -v
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.knowledge.mound.types import (
    KnowledgeItem,
    KnowledgeLink,
    KnowledgeSource,
    PaginatedResult,
    RelationshipType,
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
        self._fetchrow_result: MockRecord | None = None
        self._fetch_result: list[MockRecord] = []
        self._fetchval_result: Any = None
        self._execute_result: str = "INSERT 1"

    async def execute(self, query: str, *args) -> str:
        self.executed_queries.append((query, args))
        return self._execute_result

    async def executemany(self, query: str, args_list: list) -> None:
        for args in args_list:
            self.executed_queries.append((query, tuple(args)))

    async def fetch(self, query: str, *args) -> list[MockRecord]:
        self.executed_queries.append((query, args))
        return self._fetch_result

    async def fetchrow(self, query: str, *args) -> MockRecord | None:
        self.executed_queries.append((query, args))
        return self._fetchrow_result

    async def fetchval(self, query: str, *args) -> Any:
        self.executed_queries.append((query, args))
        return self._fetchval_result

    def set_fetchrow_result(self, data: dict[str, Any] | None) -> None:
        if data is None:
            self._fetchrow_result = None
        else:
            self._fetchrow_result = MockRecord(_data=data)

    def set_fetch_result(self, data_list: list[dict[str, Any]]) -> None:
        self._fetch_result = [MockRecord(_data=d) for d in data_list]

    def set_fetchval_result(self, value: Any) -> None:
        self._fetchval_result = value


class MockPool:
    """Mock asyncpg connection pool."""

    def __init__(self):
        self._connection = MockConnection()
        self._closed = False

    def acquire(self):
        return MockPoolAcquireContext(self)

    async def close(self) -> None:
        self._closed = True


class MockPoolAcquireContext:
    """Mock context manager for pool.acquire()."""

    def __init__(self, pool: MockPool):
        self._pool = pool

    async def __aenter__(self) -> MockConnection:
        return self._pool._connection

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


def _get_postgres_store_class():
    """Helper to import PostgresStore class directly from module."""
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
    return module.PostgresStore


@pytest.fixture
def mock_pool():
    """Create a mock connection pool."""
    return MockPool()


@pytest.fixture
def mock_connection(mock_pool):
    """Get the mock connection from the pool."""
    return mock_pool._connection


@pytest.fixture
def postgres_store(mock_pool):
    """Create a PostgresStore with mocked asyncpg."""
    PostgresStore = _get_postgres_store_class()
    store = PostgresStore(
        url="postgresql://test:test@localhost:5432/testdb",
        pool_size=10,
        max_overflow=5,
    )
    store._pool = mock_pool
    store._initialized = True
    return store


def make_node_row(i: int, workspace_id: str = "workspace_1") -> dict[str, Any]:
    """Create a sample node row for testing."""
    now = datetime.now()
    return {
        "id": f"km_node_{i}",
        "workspace_id": workspace_id,
        "node_type": "fact",
        "content": f"Test content for node {i}",
        "content_hash": f"hash_{i}",
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
        "created_at": now - timedelta(hours=i),
        "updated_at": now - timedelta(minutes=i),
        "metadata": json.dumps({"index": i}),
        "visibility": "workspace",
        "visibility_set_by": None,
        "is_discoverable": True,
    }


def make_relationship_row(i: int) -> dict[str, Any]:
    """Create a sample relationship row for testing."""
    return {
        "id": f"kr_{i}",
        "from_node_id": f"km_node_{i}",
        "to_node_id": f"km_node_{i + 1}",
        "relationship_type": "supports",
        "strength": 0.8,
        "created_by": "test",
        "created_at": datetime.now() - timedelta(hours=i),
        "metadata": json.dumps({}),
    }


def make_culture_pattern_row(i: int, workspace_id: str = "workspace_1") -> dict[str, Any]:
    """Create a sample culture pattern row for testing."""
    return {
        "id": f"cp_{i}",
        "workspace_id": workspace_id,
        "pattern_type": "decision_style",
        "pattern_key": f"pattern_{i}",
        "pattern_value": json.dumps({"value": i}),
        "observation_count": i + 1,
        "confidence": 0.9 - (i * 0.1),
        "first_observed_at": datetime.now() - timedelta(days=i),
        "last_observed_at": datetime.now() - timedelta(hours=i),
        "contributing_debates": [f"debate_{i}"],
        "metadata": json.dumps({}),
    }


# ============================================================================
# PaginatedResult Tests
# ============================================================================


class TestPaginatedResult:
    """Tests for PaginatedResult dataclass."""

    def test_basic_creation(self):
        """Should create PaginatedResult with required fields."""
        result = PaginatedResult(
            items=[1, 2, 3],
            total=10,
            limit=3,
            offset=0,
            has_more=True,
        )

        assert result.items == [1, 2, 3]
        assert result.total == 10
        assert result.limit == 3
        assert result.offset == 0
        assert result.has_more is True
        assert result.next_cursor is None

    def test_with_cursor(self):
        """Should create PaginatedResult with cursor."""
        result = PaginatedResult(
            items=["a", "b"],
            total=100,
            limit=2,
            offset=0,
            has_more=True,
            next_cursor="abc123",
        )

        assert result.next_cursor == "abc123"

    def test_to_dict_basic(self):
        """Should convert to dictionary."""
        result = PaginatedResult(
            items=[{"id": 1}, {"id": 2}],
            total=5,
            limit=2,
            offset=0,
            has_more=True,
            next_cursor="cursor_xyz",
        )

        d = result.to_dict()

        assert d["items"] == [{"id": 1}, {"id": 2}]
        assert d["total"] == 5
        assert d["limit"] == 2
        assert d["offset"] == 0
        assert d["has_more"] is True
        assert d["next_cursor"] == "cursor_xyz"

    def test_to_dict_with_knowledge_items(self):
        """Should convert KnowledgeItems via to_dict."""
        from aragora.knowledge.mound.types import ConfidenceLevel

        items = [
            KnowledgeItem(
                id="km_1",
                content="Test 1",
                source=KnowledgeSource.FACT,
                source_id="km_1",
                confidence=ConfidenceLevel.HIGH,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
            KnowledgeItem(
                id="km_2",
                content="Test 2",
                source=KnowledgeSource.FACT,
                source_id="km_2",
                confidence=ConfidenceLevel.HIGH,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
        ]

        result = PaginatedResult(
            items=items,
            total=2,
            limit=10,
            offset=0,
            has_more=False,
        )

        d = result.to_dict()

        assert len(d["items"]) == 2
        assert d["items"][0]["id"] == "km_1"
        assert d["items"][1]["id"] == "km_2"

    def test_empty_result(self):
        """Should handle empty results."""
        result = PaginatedResult(
            items=[],
            total=0,
            limit=100,
            offset=0,
            has_more=False,
        )

        assert result.items == []
        assert result.total == 0
        assert result.has_more is False

    def test_last_page(self):
        """Should correctly represent last page."""
        result = PaginatedResult(
            items=["item_8", "item_9", "item_10"],
            total=10,
            limit=5,
            offset=7,
            has_more=False,
        )

        assert len(result.items) == 3
        assert result.has_more is False


# ============================================================================
# List Nodes Pagination Tests
# ============================================================================


class TestListNodesPagination:
    """Tests for list_nodes_async pagination."""

    @pytest.mark.asyncio
    async def test_basic_pagination(self, postgres_store, mock_connection):
        """Should return paginated results with defaults."""
        # Set up mock data - 3 items (not exceeding limit)
        node_rows = [make_node_row(i) for i in range(3)]
        mock_connection.set_fetchval_result(3)  # Total count
        mock_connection.set_fetch_result(node_rows)

        result = await postgres_store.list_nodes_async("workspace_1")

        assert isinstance(result, PaginatedResult)
        assert result.total == 3
        assert result.limit == 100  # Default limit
        assert result.offset == 0
        assert result.has_more is False

    @pytest.mark.asyncio
    async def test_pagination_with_limit(self, postgres_store, mock_connection):
        """Should respect custom limit."""
        node_rows = [make_node_row(i) for i in range(6)]  # 5 + 1 extra for has_more
        mock_connection.set_fetchval_result(20)  # Total count
        mock_connection.set_fetch_result(node_rows)

        result = await postgres_store.list_nodes_async("workspace_1", limit=5)

        assert result.limit == 5
        assert len(result.items) == 5
        assert result.has_more is True
        assert result.total == 20

    @pytest.mark.asyncio
    async def test_pagination_with_offset(self, postgres_store, mock_connection):
        """Should respect offset parameter."""
        node_rows = [make_node_row(i) for i in range(5, 8)]
        mock_connection.set_fetchval_result(10)
        mock_connection.set_fetch_result(node_rows)

        result = await postgres_store.list_nodes_async("workspace_1", limit=5, offset=5)

        assert result.offset == 5
        assert result.has_more is False
        # Verify OFFSET was passed in query (check main query, not topic subqueries)
        main_queries = [
            q
            for q in mock_connection.executed_queries
            if "knowledge_nodes" in q[0] and "SELECT *" in q[0]
        ]
        assert len(main_queries) > 0
        query, args = main_queries[0]
        assert "OFFSET" in query

    @pytest.mark.asyncio
    async def test_cursor_based_pagination(self, postgres_store, mock_connection):
        """Should support cursor-based pagination."""
        # Create a cursor
        cursor_time = datetime.now() - timedelta(hours=5)
        cursor_value = f"km_node_5:{cursor_time.isoformat()}"
        cursor = base64.urlsafe_b64encode(cursor_value.encode()).decode()

        node_rows = [make_node_row(i) for i in range(6, 9)]
        mock_connection.set_fetchval_result(10)
        mock_connection.set_fetch_result(node_rows)

        result = await postgres_store.list_nodes_async(
            "workspace_1",
            limit=3,
            cursor=cursor,
        )

        # With cursor, offset should be 0 (cursor overrides)
        assert result.offset == 0
        # Verify cursor filter was applied
        queries = [q[0] for q in mock_connection.executed_queries]
        assert any("updated_at" in q and "<" in q for q in queries)

    @pytest.mark.asyncio
    async def test_next_cursor_generation(self, postgres_store, mock_connection):
        """Should generate next_cursor when has_more."""
        # 6 items means has_more=True when limit=5
        node_rows = [make_node_row(i) for i in range(6)]
        mock_connection.set_fetchval_result(100)
        mock_connection.set_fetch_result(node_rows)

        result = await postgres_store.list_nodes_async("workspace_1", limit=5)

        assert result.has_more is True
        assert result.next_cursor is not None
        # Verify cursor is decodable
        decoded = base64.urlsafe_b64decode(result.next_cursor.encode()).decode()
        assert "km_node_4" in decoded  # Last item in result

    @pytest.mark.asyncio
    async def test_node_type_filter(self, postgres_store, mock_connection):
        """Should filter by node_type."""
        mock_connection.set_fetchval_result(5)
        mock_connection.set_fetch_result([])

        await postgres_store.list_nodes_async("workspace_1", node_type="claim")

        query, args = mock_connection.executed_queries[-1]
        assert "node_type" in query
        assert "claim" in args

    @pytest.mark.asyncio
    async def test_tier_filter(self, postgres_store, mock_connection):
        """Should filter by tier."""
        mock_connection.set_fetchval_result(3)
        mock_connection.set_fetch_result([])

        await postgres_store.list_nodes_async("workspace_1", tier="fast")

        query, args = mock_connection.executed_queries[-1]
        assert "tier" in query
        assert "fast" in args

    @pytest.mark.asyncio
    async def test_order_by_validation(self, postgres_store, mock_connection):
        """Should validate order_by column."""
        mock_connection.set_fetchval_result(0)
        mock_connection.set_fetch_result([])

        # Invalid column should default to updated_at
        await postgres_store.list_nodes_async(
            "workspace_1",
            order_by="invalid_column",
        )

        query, _ = mock_connection.executed_queries[-1]
        assert "updated_at" in query
        assert "invalid_column" not in query

    @pytest.mark.asyncio
    async def test_order_direction(self, postgres_store, mock_connection):
        """Should support ASC and DESC ordering."""
        mock_connection.set_fetchval_result(0)
        mock_connection.set_fetch_result([])

        await postgres_store.list_nodes_async("workspace_1", order_dir="ASC")

        query, _ = mock_connection.executed_queries[-1]
        assert "ASC" in query

    @pytest.mark.asyncio
    async def test_max_limit_enforcement(self, postgres_store, mock_connection):
        """Should enforce MAX_LIMIT."""
        mock_connection.set_fetchval_result(0)
        mock_connection.set_fetch_result([])

        result = await postgres_store.list_nodes_async("workspace_1", limit=10000)

        assert result.limit == postgres_store.MAX_LIMIT

    @pytest.mark.asyncio
    async def test_invalid_limit_uses_default(self, postgres_store, mock_connection):
        """Should use default limit for invalid values."""
        mock_connection.set_fetchval_result(0)
        mock_connection.set_fetch_result([])

        result = await postgres_store.list_nodes_async("workspace_1", limit=-5)

        assert result.limit == postgres_store.DEFAULT_LIMIT

    @pytest.mark.asyncio
    async def test_invalid_cursor_ignored(self, postgres_store, mock_connection):
        """Should ignore invalid cursor and use offset."""
        mock_connection.set_fetchval_result(0)
        mock_connection.set_fetch_result([])

        result = await postgres_store.list_nodes_async(
            "workspace_1",
            cursor="invalid_cursor_not_base64!",
            offset=10,
        )

        assert result.offset == 10

    @pytest.mark.asyncio
    async def test_empty_workspace(self, postgres_store, mock_connection):
        """Should handle empty workspace."""
        mock_connection.set_fetchval_result(0)
        mock_connection.set_fetch_result([])

        result = await postgres_store.list_nodes_async("empty_workspace")

        assert result.items == []
        assert result.total == 0
        assert result.has_more is False


# ============================================================================
# List Relationships Pagination Tests
# ============================================================================


class TestListRelationshipsPagination:
    """Tests for list_relationships_async pagination."""

    @pytest.mark.asyncio
    async def test_basic_relationship_pagination(self, postgres_store, mock_connection):
        """Should return paginated relationships."""
        rel_rows = [make_relationship_row(i) for i in range(3)]
        mock_connection.set_fetchval_result(3)
        mock_connection.set_fetch_result(rel_rows)

        result = await postgres_store.list_relationships_async()

        assert isinstance(result, PaginatedResult)
        assert len(result.items) == 3
        assert all(isinstance(item, KnowledgeLink) for item in result.items)

    @pytest.mark.asyncio
    async def test_relationship_type_filter(self, postgres_store, mock_connection):
        """Should filter by relationship type."""
        mock_connection.set_fetchval_result(0)
        mock_connection.set_fetch_result([])

        await postgres_store.list_relationships_async(relationship_type=RelationshipType.SUPPORTS)

        query, args = mock_connection.executed_queries[-1]
        assert "relationship_type" in query
        assert "supports" in args

    @pytest.mark.asyncio
    async def test_relationship_workspace_filter(self, postgres_store, mock_connection):
        """Should filter relationships by workspace via joined nodes."""
        mock_connection.set_fetchval_result(0)
        mock_connection.set_fetch_result([])

        await postgres_store.list_relationships_async(workspace_id="workspace_1")

        query, _ = mock_connection.executed_queries[-1]
        assert "JOIN knowledge_nodes" in query
        assert "workspace_id" in query

    @pytest.mark.asyncio
    async def test_relationship_cursor_pagination(self, postgres_store, mock_connection):
        """Should support cursor-based pagination for relationships."""
        cursor_time = datetime.now() - timedelta(hours=5)
        cursor_value = f"kr_5:{cursor_time.isoformat()}"
        cursor = base64.urlsafe_b64encode(cursor_value.encode()).decode()

        rel_rows = [make_relationship_row(i) for i in range(6, 9)]
        mock_connection.set_fetchval_result(10)
        mock_connection.set_fetch_result(rel_rows)

        result = await postgres_store.list_relationships_async(limit=3, cursor=cursor)

        assert result.offset == 0  # Cursor overrides offset


# ============================================================================
# List Culture Patterns Pagination Tests
# ============================================================================


class TestListCulturePatternsPagination:
    """Tests for list_culture_patterns_async pagination."""

    @pytest.mark.asyncio
    async def test_basic_culture_pattern_pagination(self, postgres_store, mock_connection):
        """Should return paginated culture patterns."""
        pattern_rows = [make_culture_pattern_row(i) for i in range(3)]
        mock_connection.set_fetchval_result(3)
        mock_connection.set_fetch_result(pattern_rows)

        result = await postgres_store.list_culture_patterns_async("workspace_1")

        assert isinstance(result, PaginatedResult)
        assert len(result.items) == 3
        assert all(isinstance(item, dict) for item in result.items)

    @pytest.mark.asyncio
    async def test_culture_pattern_type_filter(self, postgres_store, mock_connection):
        """Should filter by pattern type."""
        mock_connection.set_fetchval_result(0)
        mock_connection.set_fetch_result([])

        await postgres_store.list_culture_patterns_async(
            "workspace_1",
            pattern_type="decision_style",
        )

        query, args = mock_connection.executed_queries[-1]
        assert "pattern_type" in query
        assert "decision_style" in args

    @pytest.mark.asyncio
    async def test_culture_pattern_no_cursor(self, postgres_store, mock_connection):
        """Culture patterns should use offset pagination only (no cursor)."""
        pattern_rows = [make_culture_pattern_row(i) for i in range(6)]
        mock_connection.set_fetchval_result(10)
        mock_connection.set_fetch_result(pattern_rows)

        result = await postgres_store.list_culture_patterns_async(
            "workspace_1",
            limit=5,
        )

        assert result.has_more is True
        assert result.next_cursor is None  # No cursor support


# ============================================================================
# Query Pagination Tests
# ============================================================================


class TestQueryPagination:
    """Tests for query_async and query_paginated_async pagination."""

    @pytest.mark.asyncio
    async def test_query_with_offset(self, postgres_store, mock_connection):
        """Should support offset in query_async."""
        mock_connection.set_fetch_result([])

        await postgres_store.query_async(
            query="test",
            filters=None,
            limit=10,
            workspace_id="workspace_1",
            offset=20,
        )

        query, args = mock_connection.executed_queries[0]
        assert "OFFSET" in query
        assert 20 in args

    @pytest.mark.asyncio
    async def test_query_paginated_returns_paginated_result(self, postgres_store, mock_connection):
        """Should return PaginatedResult from query_paginated_async."""
        node_rows = [make_node_row(i) for i in range(3)]
        mock_connection.set_fetchval_result(3)
        mock_connection.set_fetch_result(node_rows)

        result = await postgres_store.query_paginated_async(
            query="test content",
            workspace_id="workspace_1",
        )

        assert isinstance(result, PaginatedResult)
        assert result.total == 3

    @pytest.mark.asyncio
    async def test_query_paginated_has_more(self, postgres_store, mock_connection):
        """Should correctly detect has_more in paginated query."""
        # 6 items returned (5 + 1 extra)
        node_rows = [make_node_row(i) for i in range(6)]
        mock_connection.set_fetchval_result(20)
        mock_connection.set_fetch_result(node_rows)

        result = await postgres_store.query_paginated_async(
            query="test",
            workspace_id="workspace_1",
            limit=5,
        )

        assert result.has_more is True
        assert len(result.items) == 5

    @pytest.mark.asyncio
    async def test_query_enforces_max_limit(self, postgres_store, mock_connection):
        """Should enforce MAX_LIMIT in queries."""
        mock_connection.set_fetchval_result(0)
        mock_connection.set_fetch_result([])

        result = await postgres_store.query_paginated_async(
            query="test",
            workspace_id="workspace_1",
            limit=5000,
        )

        assert result.limit == postgres_store.MAX_LIMIT


# ============================================================================
# Query with Visibility Pagination Tests
# ============================================================================


class TestQueryWithVisibilityPagination:
    """Tests for query_with_visibility_async pagination."""

    @pytest.mark.asyncio
    async def test_visibility_query_with_offset(self, postgres_store, mock_connection):
        """Should support offset in visibility query."""
        mock_connection.set_fetch_result([])

        await postgres_store.query_with_visibility_async(
            query="test",
            workspace_id="workspace_1",
            actor_id="user_1",
            actor_workspace_id="workspace_1",
            limit=10,
            offset=5,
        )

        query, args = mock_connection.executed_queries[0]
        assert "OFFSET" in query
        assert 5 in args

    @pytest.mark.asyncio
    async def test_visibility_query_enforces_limit(self, postgres_store, mock_connection):
        """Should enforce MAX_LIMIT in visibility queries."""
        mock_connection.set_fetch_result([])

        await postgres_store.query_with_visibility_async(
            query="test",
            workspace_id="workspace_1",
            actor_id="user_1",
            actor_workspace_id="workspace_1",
            limit=5000,
        )

        query, args = mock_connection.executed_queries[0]
        # The limit param should be capped
        assert postgres_store.MAX_LIMIT in args


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestPaginationEdgeCases:
    """Tests for edge cases in pagination."""

    @pytest.mark.asyncio
    async def test_exactly_at_limit(self, postgres_store, mock_connection):
        """Should handle result count exactly at limit."""
        # Exactly 5 items, requesting 5
        node_rows = [make_node_row(i) for i in range(5)]
        mock_connection.set_fetchval_result(5)
        mock_connection.set_fetch_result(node_rows)

        result = await postgres_store.list_nodes_async("workspace_1", limit=5)

        assert len(result.items) == 5
        assert result.has_more is False

    @pytest.mark.asyncio
    async def test_large_offset_returns_empty(self, postgres_store, mock_connection):
        """Should handle offset beyond total items."""
        mock_connection.set_fetchval_result(10)
        mock_connection.set_fetch_result([])  # No items at this offset

        result = await postgres_store.list_nodes_async(
            "workspace_1",
            offset=1000,
        )

        assert result.items == []
        assert result.total == 10
        assert result.has_more is False

    @pytest.mark.asyncio
    async def test_zero_limit_uses_default(self, postgres_store, mock_connection):
        """Should use default limit when limit is 0."""
        mock_connection.set_fetchval_result(0)
        mock_connection.set_fetch_result([])

        result = await postgres_store.list_nodes_async("workspace_1", limit=0)

        assert result.limit == postgres_store.DEFAULT_LIMIT

    @pytest.mark.asyncio
    async def test_concurrent_pagination(self, postgres_store, mock_connection):
        """Should handle concurrent paginated requests."""
        import asyncio

        mock_connection.set_fetchval_result(100)
        mock_connection.set_fetch_result([make_node_row(i) for i in range(10)])

        async def paginated_request(offset: int):
            return await postgres_store.list_nodes_async(
                "workspace_1",
                limit=10,
                offset=offset,
            )

        # Run multiple concurrent requests
        tasks = [paginated_request(i * 10) for i in range(5)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(isinstance(r, PaginatedResult) for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
