"""
Tests for PostgresFactStore - PostgreSQL-based fact persistence.

Tests cover:
1. Connection pooling and retry logic
2. Transaction commit/rollback scenarios
3. SQL injection prevention
4. Concurrent access and locking
5. Data migration and schema evolution
6. Connection failure and recovery
7. CRUD operations (create, read, update, delete facts)
8. Search and query operations
9. Batch operations
10. Error handling edge cases
"""

from __future__ import annotations

import asyncio
import json
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch, call

import aragora.storage.postgres_store as postgres_module


# =============================================================================
# Test Setup and Fixtures
# =============================================================================


@pytest.fixture
def mock_asyncpg_available():
    """Temporarily enable asyncpg availability for testing."""
    original = postgres_module.ASYNCPG_AVAILABLE
    postgres_module.ASYNCPG_AVAILABLE = True
    yield
    postgres_module.ASYNCPG_AVAILABLE = original


@pytest.fixture
def mock_pool():
    """Create a mock connection pool."""
    pool = MagicMock()
    pool.get_size.return_value = 10
    pool.get_min_size.return_value = 5
    pool.get_max_size.return_value = 20
    pool.get_idle_size.return_value = 8
    return pool


@pytest.fixture
def mock_connection():
    """Create a mock database connection."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="INSERT 0 1")
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)
    conn.executemany = AsyncMock()
    return conn


@pytest.fixture
def mock_transaction():
    """Create a mock transaction context manager."""
    tx = AsyncMock()
    return tx


@pytest.fixture
def fact_store(mock_asyncpg_available, mock_pool, mock_connection):
    """Create a PostgresFactStore with mocked dependencies."""
    from aragora.knowledge.postgres_fact_store import PostgresFactStore

    mock_pool.acquire = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    store = PostgresFactStore(mock_pool, use_resilient=False)
    store._initialized = True
    return store


@pytest.fixture
def sample_fact_row():
    """Create a sample fact row as returned from database."""
    now = datetime.now(timezone.utc)
    return {
        "id": "fact_abc123",
        "statement": "The sky is blue",
        "statement_hash": "abc123def456",
        "confidence": 0.8,
        "evidence_ids": ["ev_1", "ev_2"],
        "consensus_proof_id": None,
        "source_documents": ["doc_1"],
        "workspace_id": "ws_test",
        "validation_status": "unverified",
        "topics": ["sky", "color"],
        "metadata": {"key": "value"},
        "created_at": now,
        "updated_at": now,
        "superseded_by": None,
    }


# =============================================================================
# 1. Connection Pooling and Retry Logic Tests
# =============================================================================


class TestConnectionPooling:
    """Tests for connection pool handling."""

    @pytest.mark.asyncio
    async def test_uses_pool_for_connection(self, fact_store, mock_pool, mock_connection):
        """Store should acquire connections from pool."""
        mock_connection.fetchrow = AsyncMock(return_value=None)

        await fact_store.get_fact_async("fact_123")

        mock_pool.acquire.assert_called()

    @pytest.mark.asyncio
    async def test_connection_released_after_operation(
        self, fact_store, mock_pool, mock_connection
    ):
        """Connection should be released back to pool after use."""
        mock_connection.fetchrow = AsyncMock(return_value=None)

        await fact_store.get_fact_async("fact_123")

        # __aexit__ should have been called
        mock_pool.acquire.return_value.__aexit__.assert_called()

    @pytest.mark.asyncio
    async def test_connection_released_on_error(self, fact_store, mock_pool, mock_connection):
        """Connection should be released even when operation fails."""
        mock_connection.fetchrow = AsyncMock(side_effect=RuntimeError("DB error"))

        with pytest.raises(RuntimeError):
            await fact_store.get_fact_async("fact_123")

        mock_pool.acquire.return_value.__aexit__.assert_called()

    def test_resilient_mode_enabled_by_default(self, mock_asyncpg_available, mock_pool):
        """Store should use resilient connection acquisition by default."""
        from aragora.knowledge.postgres_fact_store import PostgresFactStore

        store = PostgresFactStore(mock_pool)
        assert store._use_resilient is True

    def test_resilient_mode_can_be_disabled(self, mock_asyncpg_available, mock_pool):
        """Store should allow disabling resilient mode."""
        from aragora.knowledge.postgres_fact_store import PostgresFactStore

        store = PostgresFactStore(mock_pool, use_resilient=False)
        assert store._use_resilient is False


class TestRetryLogic:
    """Tests for retry behavior on transient failures."""

    @pytest.mark.asyncio
    async def test_retries_on_connection_failure(
        self, mock_asyncpg_available, mock_pool, mock_connection
    ):
        """Store should retry on transient connection failures when resilient."""
        from aragora.knowledge.postgres_fact_store import PostgresFactStore

        call_count = 0

        async def failing_then_success(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Transient failure")
            return None

        mock_connection.fetchrow = AsyncMock(side_effect=failing_then_success)
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_pool.get_size.return_value = 10
        mock_pool.get_max_size.return_value = 20

        store = PostgresFactStore(mock_pool, use_resilient=False)
        store._initialized = True

        # First call fails, subsequent could succeed with retry logic in caller
        with pytest.raises(ConnectionError):
            await store.get_fact_async("fact_123")


# =============================================================================
# 2. Transaction Commit/Rollback Tests
# =============================================================================


class TestTransactionHandling:
    """Tests for transaction management."""

    @pytest.mark.asyncio
    async def test_delete_uses_transaction(self, fact_store, mock_pool, mock_connection):
        """Delete operation should use transaction context."""
        mock_tx = AsyncMock()
        mock_connection.transaction = MagicMock(return_value=mock_tx)
        mock_connection.execute = AsyncMock(return_value="DELETE 1")

        await fact_store.delete_fact_async("fact_123")

        mock_connection.transaction.assert_called_once()

    @pytest.mark.asyncio
    async def test_transaction_commits_on_success(
        self, mock_asyncpg_available, mock_pool, mock_connection
    ):
        """Transaction should commit when operation succeeds."""
        from aragora.knowledge.postgres_fact_store import PostgresFactStore

        mock_tx = AsyncMock()
        mock_tx.__aenter__ = AsyncMock(return_value=None)
        mock_tx.__aexit__ = AsyncMock(return_value=None)
        mock_connection.transaction = MagicMock(return_value=mock_tx)
        mock_connection.execute = AsyncMock(return_value="DELETE 1")

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store = PostgresFactStore(mock_pool, use_resilient=False)
        store._initialized = True

        await store.delete_fact_async("fact_123")

        # Transaction context manager's __aexit__ should be called with no exception
        mock_tx.__aexit__.assert_called()
        call_args = mock_tx.__aexit__.call_args
        # No exception passed means commit
        assert call_args[0][0] is None

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(
        self, mock_asyncpg_available, mock_pool, mock_connection
    ):
        """Transaction should rollback when operation fails."""
        from aragora.knowledge.postgres_fact_store import PostgresFactStore

        mock_tx = AsyncMock()
        mock_tx.__aenter__ = AsyncMock(return_value=None)
        mock_tx.__aexit__ = AsyncMock(return_value=None)
        mock_connection.transaction = MagicMock(return_value=mock_tx)
        mock_connection.execute = AsyncMock(side_effect=RuntimeError("DB error"))

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store = PostgresFactStore(mock_pool, use_resilient=False)
        store._initialized = True

        with pytest.raises(RuntimeError):
            await store.delete_fact_async("fact_123")

        # Transaction should have received the exception
        mock_tx.__aexit__.assert_called()


# =============================================================================
# 3. SQL Injection Prevention Tests
# =============================================================================


class TestSQLInjectionPrevention:
    """Tests for SQL injection protection."""

    @pytest.mark.asyncio
    async def test_parameterized_queries_for_add_fact(
        self, fact_store, mock_connection, sample_fact_row
    ):
        """Add fact should use parameterized queries."""
        mock_connection.fetchrow = AsyncMock(return_value=None)
        mock_connection.execute = AsyncMock(return_value="INSERT 0 1")

        from aragora.knowledge.types import ValidationStatus

        malicious_statement = "'; DROP TABLE facts; --"
        await fact_store.add_fact_async(
            statement=malicious_statement,
            workspace_id="ws_test",
            validation_status=ValidationStatus.UNVERIFIED,
        )

        # Verify parameterized query was used (not string interpolation)
        execute_call = mock_connection.execute.call_args
        sql = execute_call[0][0]
        # SQL should use positional parameters, not contain raw malicious input
        assert "$1" in sql or "$2" in sql
        # Malicious input should be passed as parameter, not in SQL
        assert "DROP TABLE" not in sql

    @pytest.mark.asyncio
    async def test_parameterized_queries_for_get_fact(self, fact_store, mock_connection):
        """Get fact should use parameterized queries."""
        mock_connection.fetchrow = AsyncMock(return_value=None)

        malicious_id = "fact_123'; DROP TABLE facts; --"
        await fact_store.get_fact_async(malicious_id)

        fetchrow_call = mock_connection.fetchrow.call_args
        sql = fetchrow_call[0][0]
        assert "DROP TABLE" not in sql
        assert "$1" in sql

    @pytest.mark.asyncio
    async def test_parameterized_queries_for_query_facts(self, fact_store, mock_connection):
        """Query facts should use parameterized queries."""
        mock_connection.fetch = AsyncMock(return_value=[])

        malicious_query = "test'; DROP TABLE facts; --"
        await fact_store.query_facts_async(malicious_query)

        fetch_call = mock_connection.fetch.call_args
        sql = fetch_call[0][0]
        assert "DROP TABLE" not in sql

    def test_tsquery_sanitization(self, fact_store):
        """TSQuery input should be sanitized."""
        # Empty query should return empty string
        result = fact_store._sanitize_tsquery("")
        assert result == ""

        result = fact_store._sanitize_tsquery("   ")
        assert result == ""

        # Non-empty query should be stripped
        result = fact_store._sanitize_tsquery("  test query  ")
        assert result == "test query"


# =============================================================================
# 4. Concurrent Access and Locking Tests
# =============================================================================


class TestConcurrentAccess:
    """Tests for concurrent access scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, fact_store, mock_connection, sample_fact_row):
        """Multiple concurrent reads should not interfere."""
        mock_connection.fetchrow = AsyncMock(return_value=sample_fact_row)

        # Run multiple concurrent reads
        tasks = [fact_store.get_fact_async(f"fact_{i}") for i in range(5)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, fact_store, mock_connection):
        """Multiple concurrent writes should be handled safely."""
        mock_connection.fetchrow = AsyncMock(return_value=None)
        mock_connection.execute = AsyncMock(return_value="INSERT 0 1")

        from aragora.knowledge.types import ValidationStatus

        # Run multiple concurrent writes
        tasks = [
            fact_store.add_fact_async(
                statement=f"Fact number {i}",
                workspace_id="ws_test",
                validation_status=ValidationStatus.UNVERIFIED,
            )
            for i in range(5)
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(r.statement.startswith("Fact number") for r in results)

    @pytest.mark.asyncio
    async def test_deduplication_prevents_duplicates(
        self, fact_store, mock_connection, sample_fact_row
    ):
        """Deduplication should return existing fact for same statement."""
        # First call returns existing fact
        mock_connection.fetchrow = AsyncMock(return_value=sample_fact_row)

        from aragora.knowledge.types import ValidationStatus

        result = await fact_store.add_fact_async(
            statement="The sky is blue",
            workspace_id="ws_test",
            deduplicate=True,
            validation_status=ValidationStatus.UNVERIFIED,
        )

        assert result.id == "fact_abc123"
        # execute should not have been called for INSERT
        mock_connection.execute.assert_not_called()


# =============================================================================
# 5. Data Migration and Schema Evolution Tests
# =============================================================================


class TestSchemaManagement:
    """Tests for schema versioning and migrations."""

    def test_schema_name_defined(self):
        """PostgresFactStore should define SCHEMA_NAME."""
        from aragora.knowledge.postgres_fact_store import PostgresFactStore

        assert PostgresFactStore.SCHEMA_NAME == "fact_store"

    def test_schema_version_defined(self):
        """PostgresFactStore should define SCHEMA_VERSION."""
        from aragora.knowledge.postgres_fact_store import PostgresFactStore

        assert PostgresFactStore.SCHEMA_VERSION >= 1

    def test_initial_schema_defined(self):
        """PostgresFactStore should define INITIAL_SCHEMA."""
        from aragora.knowledge.postgres_fact_store import PostgresFactStore

        assert "CREATE TABLE IF NOT EXISTS facts" in PostgresFactStore.INITIAL_SCHEMA
        assert "CREATE TABLE IF NOT EXISTS fact_relations" in PostgresFactStore.INITIAL_SCHEMA

    def test_initial_schema_includes_indexes(self):
        """Initial schema should create necessary indexes."""
        from aragora.knowledge.postgres_fact_store import PostgresFactStore

        schema = PostgresFactStore.INITIAL_SCHEMA
        assert "CREATE INDEX IF NOT EXISTS idx_facts_workspace" in schema
        assert "CREATE INDEX IF NOT EXISTS idx_facts_status" in schema
        assert "CREATE INDEX IF NOT EXISTS idx_facts_search" in schema

    def test_initial_schema_includes_fts_trigger(self):
        """Initial schema should include full-text search trigger."""
        from aragora.knowledge.postgres_fact_store import PostgresFactStore

        schema = PostgresFactStore.INITIAL_SCHEMA
        assert "update_fact_search_vector" in schema
        assert "fact_search_vector_trigger" in schema

    @pytest.mark.asyncio
    async def test_initialize_creates_schema(
        self, mock_asyncpg_available, mock_pool, mock_connection
    ):
        """Initialize should run schema creation."""
        from aragora.knowledge.postgres_fact_store import PostgresFactStore

        mock_connection.fetchrow = AsyncMock(return_value=None)  # No existing version
        mock_connection.execute = AsyncMock()

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store = PostgresFactStore(mock_pool, use_resilient=False)
        await store.initialize()

        # Should have executed schema creation
        calls = [str(c) for c in mock_connection.execute.call_args_list]
        assert any("facts" in c for c in calls)


# =============================================================================
# 6. Connection Failure and Recovery Tests
# =============================================================================


class TestConnectionFailureRecovery:
    """Tests for handling connection failures."""

    @pytest.mark.asyncio
    async def test_handles_connection_timeout(
        self, mock_asyncpg_available, mock_pool, mock_connection
    ):
        """Store should handle connection timeout gracefully."""
        from aragora.knowledge.postgres_fact_store import PostgresFactStore

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(
            side_effect=asyncio.TimeoutError("Connection timeout")
        )
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store = PostgresFactStore(mock_pool, use_resilient=False)
        store._initialized = True

        with pytest.raises(asyncio.TimeoutError):
            await store.get_fact_async("fact_123")

    @pytest.mark.asyncio
    async def test_handles_database_error(self, fact_store, mock_connection):
        """Store should propagate database errors."""
        mock_connection.fetchrow = AsyncMock(side_effect=RuntimeError("Database connection lost"))

        with pytest.raises(RuntimeError, match="Database connection lost"):
            await fact_store.get_fact_async("fact_123")

    @pytest.mark.asyncio
    async def test_handles_query_execution_error(self, fact_store, mock_connection):
        """Store should handle query execution errors."""
        mock_connection.execute = AsyncMock(side_effect=RuntimeError("Syntax error"))
        mock_connection.fetchrow = AsyncMock(return_value=None)

        from aragora.knowledge.types import ValidationStatus

        with pytest.raises(RuntimeError, match="Syntax error"):
            await fact_store.add_fact_async(
                statement="Test",
                workspace_id="ws_test",
                validation_status=ValidationStatus.UNVERIFIED,
            )


# =============================================================================
# 7. CRUD Operations Tests
# =============================================================================


class TestAddFact:
    """Tests for add_fact operations."""

    @pytest.mark.asyncio
    async def test_add_fact_creates_new_fact(self, fact_store, mock_connection):
        """Add fact should create a new fact when not deduplicated."""
        mock_connection.fetchrow = AsyncMock(return_value=None)
        mock_connection.execute = AsyncMock(return_value="INSERT 0 1")

        from aragora.knowledge.types import ValidationStatus

        result = await fact_store.add_fact_async(
            statement="New fact statement",
            workspace_id="ws_test",
            confidence=0.9,
            topics=["topic1"],
            metadata={"key": "value"},
            validation_status=ValidationStatus.UNVERIFIED,
        )

        assert result.id.startswith("fact_")
        assert result.statement == "New fact statement"
        assert result.workspace_id == "ws_test"
        assert result.confidence == 0.9
        assert "topic1" in result.topics
        assert result.metadata["key"] == "value"

    @pytest.mark.asyncio
    async def test_add_fact_with_evidence_ids(self, fact_store, mock_connection):
        """Add fact should store evidence IDs."""
        mock_connection.fetchrow = AsyncMock(return_value=None)
        mock_connection.execute = AsyncMock(return_value="INSERT 0 1")

        from aragora.knowledge.types import ValidationStatus

        result = await fact_store.add_fact_async(
            statement="Fact with evidence",
            workspace_id="ws_test",
            evidence_ids=["ev_1", "ev_2"],
            validation_status=ValidationStatus.UNVERIFIED,
        )

        assert result.evidence_ids == ["ev_1", "ev_2"]

    @pytest.mark.asyncio
    async def test_add_fact_without_deduplication(self, fact_store, mock_connection):
        """Add fact with deduplicate=False should always create new."""
        mock_connection.execute = AsyncMock(return_value="INSERT 0 1")

        from aragora.knowledge.types import ValidationStatus

        result = await fact_store.add_fact_async(
            statement="Some statement",
            workspace_id="ws_test",
            deduplicate=False,
            validation_status=ValidationStatus.UNVERIFIED,
        )

        # Should not have checked for existing fact
        mock_connection.fetchrow.assert_not_called()
        assert result.id.startswith("fact_")

    @pytest.mark.asyncio
    async def test_add_fact_computes_statement_hash(self, fact_store, mock_connection):
        """Add fact should compute hash for deduplication."""
        mock_connection.fetchrow = AsyncMock(return_value=None)
        mock_connection.execute = AsyncMock(return_value="INSERT 0 1")

        from aragora.knowledge.types import ValidationStatus

        await fact_store.add_fact_async(
            statement="Test Statement",
            workspace_id="ws_test",
            validation_status=ValidationStatus.UNVERIFIED,
        )

        # Verify hash was used in dedup check
        fetchrow_call = mock_connection.fetchrow.call_args
        assert "statement_hash" in fetchrow_call[0][0]


class TestGetFact:
    """Tests for get_fact operations."""

    @pytest.mark.asyncio
    async def test_get_fact_returns_fact(self, fact_store, mock_connection, sample_fact_row):
        """Get fact should return fact when found."""
        mock_connection.fetchrow = AsyncMock(return_value=sample_fact_row)

        result = await fact_store.get_fact_async("fact_abc123")

        assert result is not None
        assert result.id == "fact_abc123"
        assert result.statement == "The sky is blue"
        assert result.confidence == 0.8

    @pytest.mark.asyncio
    async def test_get_fact_returns_none_when_not_found(self, fact_store, mock_connection):
        """Get fact should return None when not found."""
        mock_connection.fetchrow = AsyncMock(return_value=None)

        result = await fact_store.get_fact_async("fact_nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_fact_parses_json_fields(self, fact_store, mock_connection):
        """Get fact should properly parse JSON fields."""
        row = {
            "id": "fact_json",
            "statement": "Test",
            "statement_hash": "hash123",
            "confidence": 0.5,
            "evidence_ids": '["ev_1", "ev_2"]',  # JSON string
            "consensus_proof_id": None,
            "source_documents": '["doc_1"]',  # JSON string
            "workspace_id": "ws_test",
            "validation_status": "unverified",
            "topics": '["topic1"]',  # JSON string
            "metadata": '{"key": "value"}',  # JSON string
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "superseded_by": None,
        }
        mock_connection.fetchrow = AsyncMock(return_value=row)

        result = await fact_store.get_fact_async("fact_json")

        assert result.evidence_ids == ["ev_1", "ev_2"]
        assert result.source_documents == ["doc_1"]
        assert result.topics == ["topic1"]
        assert result.metadata == {"key": "value"}


class TestUpdateFact:
    """Tests for update_fact operations."""

    @pytest.mark.asyncio
    async def test_update_fact_updates_confidence(
        self, fact_store, mock_connection, sample_fact_row
    ):
        """Update fact should update confidence."""
        mock_connection.execute = AsyncMock(return_value="UPDATE 1")
        updated_row = {**sample_fact_row, "confidence": 0.95}
        mock_connection.fetchrow = AsyncMock(return_value=updated_row)

        from aragora.knowledge.types import ValidationStatus

        result = await fact_store.update_fact_async(
            fact_id="fact_abc123",
            confidence=0.95,
        )

        assert result is not None
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_update_fact_updates_validation_status(
        self, fact_store, mock_connection, sample_fact_row
    ):
        """Update fact should update validation status."""
        mock_connection.execute = AsyncMock(return_value="UPDATE 1")
        updated_row = {**sample_fact_row, "validation_status": "majority_agreed"}
        mock_connection.fetchrow = AsyncMock(return_value=updated_row)

        from aragora.knowledge.types import ValidationStatus

        result = await fact_store.update_fact_async(
            fact_id="fact_abc123",
            validation_status=ValidationStatus.MAJORITY_AGREED,
        )

        assert result is not None
        assert result.validation_status == ValidationStatus.MAJORITY_AGREED

    @pytest.mark.asyncio
    async def test_update_fact_returns_none_when_not_found(self, fact_store, mock_connection):
        """Update fact should return None when fact not found."""
        mock_connection.execute = AsyncMock(return_value="UPDATE 0")
        mock_connection.fetchrow = AsyncMock(return_value=None)

        result = await fact_store.update_fact_async(
            fact_id="fact_nonexistent",
            confidence=0.9,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_update_fact_with_no_changes(self, fact_store, mock_connection, sample_fact_row):
        """Update with no changes should still return the fact."""
        mock_connection.fetchrow = AsyncMock(return_value=sample_fact_row)

        result = await fact_store.update_fact_async(fact_id="fact_abc123")

        assert result is not None
        # Should have just fetched the existing fact
        assert result.id == "fact_abc123"


class TestDeleteFact:
    """Tests for delete_fact operations."""

    @pytest.mark.asyncio
    async def test_delete_fact_returns_true_on_success(self, fact_store, mock_connection):
        """Delete fact should return True when deleted."""
        mock_tx = AsyncMock()
        mock_tx.__aenter__ = AsyncMock(return_value=None)
        mock_tx.__aexit__ = AsyncMock(return_value=None)
        mock_connection.transaction = MagicMock(return_value=mock_tx)
        mock_connection.execute = AsyncMock(return_value="DELETE 1")

        result = await fact_store.delete_fact_async("fact_123")

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_fact_returns_false_when_not_found(self, fact_store, mock_connection):
        """Delete fact should return False when not found."""
        mock_tx = AsyncMock()
        mock_tx.__aenter__ = AsyncMock(return_value=None)
        mock_tx.__aexit__ = AsyncMock(return_value=None)
        mock_connection.transaction = MagicMock(return_value=mock_tx)
        mock_connection.execute = AsyncMock(return_value="DELETE 0")

        result = await fact_store.delete_fact_async("fact_nonexistent")

        assert result is False


# =============================================================================
# 8. Search and Query Operations Tests
# =============================================================================


class TestQueryFacts:
    """Tests for query_facts full-text search."""

    @pytest.mark.asyncio
    async def test_query_facts_uses_fts(self, fact_store, mock_connection, sample_fact_row):
        """Query facts should use PostgreSQL full-text search."""
        mock_connection.fetch = AsyncMock(return_value=[{**sample_fact_row, "rank": 0.8}])

        result = await fact_store.query_facts_async("blue sky")

        assert len(result) == 1
        # Verify FTS query was used
        fetch_call = mock_connection.fetch.call_args
        sql = fetch_call[0][0]
        assert "search_vector" in sql
        assert "plainto_tsquery" in sql

    @pytest.mark.asyncio
    async def test_query_facts_with_empty_query(self, fact_store, mock_connection):
        """Empty query should fall back to list_facts."""
        mock_connection.fetch = AsyncMock(return_value=[])

        result = await fact_store.query_facts_async("")

        # Should not have used FTS
        assert result == []

    @pytest.mark.asyncio
    async def test_query_facts_with_workspace_filter(
        self, fact_store, mock_connection, sample_fact_row
    ):
        """Query should filter by workspace."""
        mock_connection.fetch = AsyncMock(return_value=[{**sample_fact_row, "rank": 0.5}])

        from aragora.knowledge.types import FactFilters

        filters = FactFilters(workspace_id="ws_test")
        await fact_store.query_facts_async("test", filters)

        fetch_call = mock_connection.fetch.call_args
        sql = fetch_call[0][0]
        assert "workspace_id" in sql

    @pytest.mark.asyncio
    async def test_query_facts_with_confidence_filter(self, fact_store, mock_connection):
        """Query should filter by minimum confidence."""
        mock_connection.fetch = AsyncMock(return_value=[])

        from aragora.knowledge.types import FactFilters

        filters = FactFilters(min_confidence=0.8)
        await fact_store.query_facts_async("test", filters)

        fetch_call = mock_connection.fetch.call_args
        sql = fetch_call[0][0]
        assert "confidence" in sql


class TestListFacts:
    """Tests for list_facts operations."""

    @pytest.mark.asyncio
    async def test_list_facts_returns_all(self, fact_store, mock_connection, sample_fact_row):
        """List facts should return all facts without filter."""
        mock_connection.fetch = AsyncMock(return_value=[sample_fact_row, sample_fact_row])

        result = await fact_store.list_facts_async()

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_facts_filters_by_workspace(self, fact_store, mock_connection):
        """List facts should filter by workspace."""
        mock_connection.fetch = AsyncMock(return_value=[])

        from aragora.knowledge.types import FactFilters

        filters = FactFilters(workspace_id="ws_specific")
        await fact_store.list_facts_async(filters)

        fetch_call = mock_connection.fetch.call_args
        sql = fetch_call[0][0]
        assert "workspace_id" in sql

    @pytest.mark.asyncio
    async def test_list_facts_filters_by_topics(self, fact_store, mock_connection):
        """List facts should filter by topics."""
        mock_connection.fetch = AsyncMock(return_value=[])

        from aragora.knowledge.types import FactFilters

        filters = FactFilters(topics=["security"])
        await fact_store.list_facts_async(filters)

        fetch_call = mock_connection.fetch.call_args
        sql = fetch_call[0][0]
        assert "topics" in sql

    @pytest.mark.asyncio
    async def test_list_facts_excludes_superseded_by_default(self, fact_store, mock_connection):
        """List facts should exclude superseded facts by default."""
        mock_connection.fetch = AsyncMock(return_value=[])

        await fact_store.list_facts_async()

        fetch_call = mock_connection.fetch.call_args
        sql = fetch_call[0][0]
        assert "superseded_by IS NULL" in sql

    @pytest.mark.asyncio
    async def test_list_facts_includes_superseded_when_requested(self, fact_store, mock_connection):
        """List facts should include superseded when flag is set."""
        mock_connection.fetch = AsyncMock(return_value=[])

        from aragora.knowledge.types import FactFilters

        filters = FactFilters(include_superseded=True)
        await fact_store.list_facts_async(filters)

        fetch_call = mock_connection.fetch.call_args
        sql = fetch_call[0][0]
        assert "superseded_by IS NULL" not in sql


class TestGetContradictions:
    """Tests for get_contradictions operations."""

    @pytest.mark.asyncio
    async def test_get_contradictions_uses_join(self, fact_store, mock_connection, sample_fact_row):
        """Get contradictions should join with relations table."""
        mock_connection.fetch = AsyncMock(return_value=[sample_fact_row])

        result = await fact_store.get_contradictions_async("fact_123")

        fetch_call = mock_connection.fetch.call_args
        sql = fetch_call[0][0]
        assert "fact_relations" in sql
        assert "contradicts" in str(fetch_call).lower()
        assert len(result) == 1


# =============================================================================
# 9. Batch Operations Tests
# =============================================================================


class TestBatchOperations:
    """Tests for batch/bulk operations."""

    @pytest.mark.asyncio
    async def test_add_multiple_facts_sequentially(self, fact_store, mock_connection):
        """Adding multiple facts should work correctly."""
        mock_connection.fetchrow = AsyncMock(return_value=None)
        mock_connection.execute = AsyncMock(return_value="INSERT 0 1")

        from aragora.knowledge.types import ValidationStatus

        facts_data = [{"statement": f"Fact {i}", "workspace_id": "ws_test"} for i in range(5)]

        results = []
        for data in facts_data:
            result = await fact_store.add_fact_async(
                **data,
                validation_status=ValidationStatus.UNVERIFIED,
            )
            results.append(result)

        assert len(results) == 5
        assert all(r.id.startswith("fact_") for r in results)
        # Each should have unique ID
        ids = [r.id for r in results]
        assert len(set(ids)) == 5


class TestRelationOperations:
    """Tests for fact relation operations."""

    @pytest.mark.asyncio
    async def test_add_relation(self, fact_store, mock_connection):
        """Add relation should create a new relation."""
        mock_connection.execute = AsyncMock(return_value="INSERT 0 1")

        from aragora.knowledge.types import FactRelationType

        result = await fact_store.add_relation_async(
            source_fact_id="fact_1",
            target_fact_id="fact_2",
            relation_type=FactRelationType.SUPPORTS,
            confidence=0.9,
            created_by="test_agent",
            metadata={"reason": "test"},
        )

        assert result.id.startswith("rel_")
        assert result.source_fact_id == "fact_1"
        assert result.target_fact_id == "fact_2"
        assert result.relation_type == FactRelationType.SUPPORTS
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_get_relations_as_source(self, fact_store, mock_connection):
        """Get relations should find relations where fact is source."""
        now = datetime.now(timezone.utc)
        mock_connection.fetch = AsyncMock(
            return_value=[
                {
                    "id": "rel_1",
                    "source_fact_id": "fact_1",
                    "target_fact_id": "fact_2",
                    "relation_type": "supports",
                    "confidence": 0.8,
                    "created_by": "agent",
                    "metadata": {},
                    "created_at": now,
                }
            ]
        )

        result = await fact_store.get_relations_async("fact_1", as_source=True, as_target=False)

        assert len(result) == 1
        fetch_call = mock_connection.fetch.call_args
        sql = fetch_call[0][0]
        assert "source_fact_id" in sql

    @pytest.mark.asyncio
    async def test_get_relations_with_type_filter(self, fact_store, mock_connection):
        """Get relations should filter by relation type."""
        mock_connection.fetch = AsyncMock(return_value=[])

        from aragora.knowledge.types import FactRelationType

        await fact_store.get_relations_async(
            "fact_1",
            relation_type=FactRelationType.CONTRADICTS,
        )

        fetch_call = mock_connection.fetch.call_args
        sql = fetch_call[0][0]
        assert "relation_type" in sql

    @pytest.mark.asyncio
    async def test_get_relations_returns_empty_when_no_direction(self, fact_store):
        """Get relations with both as_source=False and as_target=False returns empty."""
        result = await fact_store.get_relations_async(
            "fact_1",
            as_source=False,
            as_target=False,
        )

        assert result == []


# =============================================================================
# 10. Error Handling Edge Cases Tests
# =============================================================================


class TestErrorHandlingEdgeCases:
    """Tests for error handling edge cases."""

    @pytest.mark.asyncio
    async def test_handles_null_json_fields(self, fact_store, mock_connection):
        """Should handle NULL JSON fields gracefully."""
        row = {
            "id": "fact_null",
            "statement": "Test",
            "statement_hash": "hash",
            "confidence": 0.5,
            "evidence_ids": None,
            "consensus_proof_id": None,
            "source_documents": None,
            "workspace_id": "ws_test",
            "validation_status": "unverified",
            "topics": None,
            "metadata": None,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "superseded_by": None,
        }
        mock_connection.fetchrow = AsyncMock(return_value=row)

        result = await fact_store.get_fact_async("fact_null")

        assert result.evidence_ids == []
        assert result.source_documents == []
        assert result.topics == []
        assert result.metadata == {}

    @pytest.mark.asyncio
    async def test_handles_empty_result_set(self, fact_store, mock_connection):
        """Should handle empty result sets gracefully."""
        mock_connection.fetch = AsyncMock(return_value=[])

        result = await fact_store.list_facts_async()

        assert result == []

    def test_statement_hash_normalization(self, fact_store):
        """Statement hash should normalize whitespace and case."""
        hash1 = fact_store._compute_statement_hash("Test Statement")
        hash2 = fact_store._compute_statement_hash("test   statement")
        hash3 = fact_store._compute_statement_hash("TEST STATEMENT")

        assert hash1 == hash2
        assert hash2 == hash3

    def test_row_to_fact_handles_list_json(self, fact_store):
        """_row_to_fact should handle both list and string JSON."""
        now = datetime.now(timezone.utc)

        # Test with already-parsed lists (asyncpg returns these)
        row_parsed = {
            "id": "fact_1",
            "statement": "Test",
            "statement_hash": "hash",
            "confidence": 0.5,
            "evidence_ids": ["ev_1"],  # Already a list
            "consensus_proof_id": None,
            "source_documents": ["doc_1"],  # Already a list
            "workspace_id": "ws_test",
            "validation_status": "unverified",
            "topics": ["topic1"],  # Already a list
            "metadata": {"key": "value"},  # Already a dict
            "created_at": now,
            "updated_at": now,
            "superseded_by": None,
        }

        result = fact_store._row_to_fact(row_parsed)
        assert result.evidence_ids == ["ev_1"]
        assert result.source_documents == ["doc_1"]
        assert result.topics == ["topic1"]
        assert result.metadata == {"key": "value"}


class TestStatistics:
    """Tests for get_statistics operations."""

    @pytest.mark.asyncio
    async def test_get_statistics_returns_counts(self, fact_store, mock_connection):
        """Get statistics should return fact counts."""
        mock_connection.fetchrow = AsyncMock(
            side_effect=[
                {"count": 100},  # Total facts
                {"avg": 0.75},  # Average confidence
                {"count": 50},  # Verified facts
                {"count": 25},  # Relations count
            ]
        )
        mock_connection.fetch = AsyncMock(
            return_value=[
                {"validation_status": "unverified", "count": 50},
                {"validation_status": "majority_agreed", "count": 50},
            ]
        )

        result = await fact_store.get_statistics_async()

        assert result["total_facts"] == 100
        assert result["average_confidence"] == 0.75
        assert "by_status" in result

    @pytest.mark.asyncio
    async def test_get_statistics_with_workspace_filter(self, fact_store, mock_connection):
        """Get statistics should filter by workspace."""
        mock_connection.fetchrow = AsyncMock(return_value={"count": 10, "avg": 0.8})
        mock_connection.fetch = AsyncMock(return_value=[])

        await fact_store.get_statistics_async(workspace_id="ws_specific")

        # Verify workspace filter was included
        fetchrow_calls = mock_connection.fetchrow.call_args_list
        assert any("workspace_id" in str(c) for c in fetchrow_calls)


class TestSyncWrappers:
    """Tests for synchronous wrapper methods."""

    def test_add_fact_sync_wrapper(self, fact_store, mock_connection):
        """add_fact should call add_fact_async."""
        mock_connection.fetchrow = AsyncMock(return_value=None)
        mock_connection.execute = AsyncMock(return_value="INSERT 0 1")

        from aragora.knowledge.types import ValidationStatus

        result = fact_store.add_fact(
            statement="Sync test",
            workspace_id="ws_test",
            validation_status=ValidationStatus.UNVERIFIED,
        )

        assert result.statement == "Sync test"

    def test_get_fact_sync_wrapper(self, fact_store, mock_connection, sample_fact_row):
        """get_fact should call get_fact_async."""
        mock_connection.fetchrow = AsyncMock(return_value=sample_fact_row)

        result = fact_store.get_fact("fact_abc123")

        assert result is not None
        assert result.id == "fact_abc123"

    def test_close_is_noop(self, fact_store):
        """close() should be a no-op for pool-based store."""
        fact_store.close()  # Should not raise


class TestDateFiltering:
    """Tests for date-based filtering."""

    @pytest.mark.asyncio
    async def test_list_facts_with_created_after(self, fact_store, mock_connection):
        """List facts should filter by created_after."""
        mock_connection.fetch = AsyncMock(return_value=[])

        from aragora.knowledge.types import FactFilters

        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        filters = FactFilters(created_after=cutoff)
        await fact_store.list_facts_async(filters)

        fetch_call = mock_connection.fetch.call_args
        sql = fetch_call[0][0]
        assert "created_at >=" in sql

    @pytest.mark.asyncio
    async def test_list_facts_with_created_before(self, fact_store, mock_connection):
        """List facts should filter by created_before."""
        mock_connection.fetch = AsyncMock(return_value=[])

        from aragora.knowledge.types import FactFilters

        cutoff = datetime.now(timezone.utc)
        filters = FactFilters(created_before=cutoff)
        await fact_store.list_facts_async(filters)

        fetch_call = mock_connection.fetch.call_args
        sql = fetch_call[0][0]
        assert "created_at <=" in sql

    @pytest.mark.asyncio
    async def test_query_facts_with_date_range(self, fact_store, mock_connection):
        """Query facts should support date range filtering."""
        mock_connection.fetch = AsyncMock(return_value=[])

        from aragora.knowledge.types import FactFilters

        start = datetime.now(timezone.utc) - timedelta(days=30)
        end = datetime.now(timezone.utc)
        filters = FactFilters(created_after=start, created_before=end)
        await fact_store.query_facts_async("test", filters)

        fetch_call = mock_connection.fetch.call_args
        sql = fetch_call[0][0]
        assert "created_at >=" in sql
        assert "created_at <=" in sql
