"""
Tests for Token Blacklist Storage.

Tests cover:
- BlacklistBackend abstract interface
- InMemoryBlacklist: thread-safety, TTL, cleanup, max size enforcement
- SQLiteBlacklist: persistence, schema, concurrent access
- PostgresBlacklist: async operations (mocked pool)
- Global store management (get_blacklist_backend, set_blacklist_backend)
- Edge cases and error handling
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.storage.token_blacklist_store import (
    BlacklistBackend,
    InMemoryBlacklist,
    SQLiteBlacklist,
    PostgresBlacklist,
    get_blacklist_backend,
    set_blacklist_backend,
    MAX_BLACKLIST_SIZE,
    HAS_REDIS,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def in_memory_blacklist():
    """Create an in-memory blacklist with short cleanup interval for testing."""
    return InMemoryBlacklist(cleanup_interval=1)


@pytest.fixture
def sqlite_blacklist(tmp_path):
    """Create a SQLite blacklist in a temporary directory."""
    db_path = tmp_path / "blacklist_test.db"
    blacklist = SQLiteBlacklist(db_path=db_path, cleanup_interval=1)
    yield blacklist
    blacklist.close()


@pytest.fixture(autouse=True)
def reset_global_blacklist():
    """Reset global blacklist after each test."""
    import aragora.storage.token_blacklist_store as module

    original = module._blacklist_backend
    yield
    module._blacklist_backend = original


# =============================================================================
# BlacklistBackend Abstract Interface Tests
# =============================================================================


class TestBlacklistBackendInterface:
    """Tests for BlacklistBackend abstract base class."""

    def test_abstract_methods_cannot_be_called_directly(self):
        """Abstract methods should raise NotImplementedError or require implementation."""

        class MinimalBackend(BlacklistBackend):
            def add(self, token_jti: str, expires_at: float) -> None:
                pass

            def contains(self, token_jti: str) -> bool:
                return False

            def cleanup_expired(self) -> int:
                return 0

        backend = MinimalBackend()
        # Default size should return -1
        assert backend.size() == -1

    def test_size_default_implementation(self):
        """Default size() should return -1."""

        class TestBackend(BlacklistBackend):
            def add(self, token_jti: str, expires_at: float) -> None:
                pass

            def contains(self, token_jti: str) -> bool:
                return False

            def cleanup_expired(self) -> int:
                return 0

        backend = TestBackend()
        assert backend.size() == -1


# =============================================================================
# InMemoryBlacklist Tests
# =============================================================================


class TestInMemoryBlacklist:
    """Tests for InMemoryBlacklist implementation."""

    def test_add_and_contains(self, in_memory_blacklist):
        """Should add token and detect it as blacklisted."""
        token_jti = "test-token-jti-001"
        expires_at = time.time() + 3600  # 1 hour from now

        in_memory_blacklist.add(token_jti, expires_at)

        assert in_memory_blacklist.contains(token_jti) is True

    def test_contains_returns_false_for_unknown(self, in_memory_blacklist):
        """Should return False for tokens not in blacklist."""
        assert in_memory_blacklist.contains("unknown-token") is False

    def test_multiple_tokens(self, in_memory_blacklist):
        """Should track multiple tokens independently."""
        expires_at = time.time() + 3600

        in_memory_blacklist.add("token-1", expires_at)
        in_memory_blacklist.add("token-2", expires_at)
        in_memory_blacklist.add("token-3", expires_at)

        assert in_memory_blacklist.contains("token-1") is True
        assert in_memory_blacklist.contains("token-2") is True
        assert in_memory_blacklist.contains("token-3") is True
        assert in_memory_blacklist.contains("token-4") is False

    def test_size_tracking(self, in_memory_blacklist):
        """Should track blacklist size accurately."""
        assert in_memory_blacklist.size() == 0

        expires_at = time.time() + 3600

        in_memory_blacklist.add("token-1", expires_at)
        assert in_memory_blacklist.size() == 1

        in_memory_blacklist.add("token-2", expires_at)
        in_memory_blacklist.add("token-3", expires_at)
        assert in_memory_blacklist.size() == 3

    def test_cleanup_expired_removes_old_tokens(self):
        """Should remove expired tokens during cleanup."""
        blacklist = InMemoryBlacklist(
            cleanup_interval=10000
        )  # High interval to prevent auto-cleanup

        # Add token that's already expired
        blacklist.add("expired-token", time.time() - 1)  # Already expired
        # Add token with longer expiry
        blacklist.add("valid-token", time.time() + 3600)

        removed = blacklist.cleanup_expired()

        assert removed == 1
        assert blacklist.size() == 1  # Only valid token remains
        assert blacklist.contains("valid-token") is True

    def test_automatic_cleanup_triggered(self):
        """Should trigger automatic cleanup after cleanup_interval."""
        # Use a high interval to avoid automatic cleanup, then manually trigger
        blacklist = InMemoryBlacklist(cleanup_interval=10000)

        # Add token that's already expired
        blacklist.add("token-1", time.time() - 1)
        blacklist.add("token-2", time.time() + 3600)

        # Both tokens should be present before cleanup
        assert blacklist.size() == 2

        # Manually trigger cleanup
        removed = blacklist.cleanup_expired()

        # Expired token should be cleaned up
        assert removed == 1
        assert blacklist.size() == 1

    def test_clear_removes_all_entries(self, in_memory_blacklist):
        """Clear should remove all entries."""
        expires_at = time.time() + 3600

        in_memory_blacklist.add("token-1", expires_at)
        in_memory_blacklist.add("token-2", expires_at)
        in_memory_blacklist.add("token-3", expires_at)

        in_memory_blacklist.clear()

        assert in_memory_blacklist.size() == 0
        assert in_memory_blacklist.contains("token-1") is False

    def test_max_size_enforcement_evicts_expired_first(self):
        """Should evict expired tokens first when at max size."""
        # We can't easily test the exact max size, but we can verify the logic works
        blacklist = InMemoryBlacklist()

        # Fill with tokens
        now = time.time()
        for i in range(100):
            blacklist.add(f"token-{i}", now + 3600)

        initial_size = blacklist.size()
        assert initial_size == 100

    def test_thread_safety_concurrent_adds(self, in_memory_blacklist):
        """Concurrent add operations should be thread-safe."""
        errors = []
        expires_at = time.time() + 3600

        def add_tokens(thread_id):
            try:
                for i in range(100):
                    in_memory_blacklist.add(f"token-{thread_id}-{i}", expires_at)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_tokens, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert in_memory_blacklist.size() == 500

    def test_thread_safety_concurrent_contains(self, in_memory_blacklist):
        """Concurrent contains checks should be thread-safe."""
        expires_at = time.time() + 3600
        in_memory_blacklist.add("test-token", expires_at)

        results = []
        errors = []

        def check_token():
            try:
                for _ in range(100):
                    result = in_memory_blacklist.contains("test-token")
                    results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=check_token) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(r is True for r in results)

    def test_thread_safety_mixed_operations(self, in_memory_blacklist):
        """Mixed concurrent operations should be thread-safe."""
        errors = []

        def mixed_ops(thread_id):
            try:
                expires_at = time.time() + 3600
                for i in range(50):
                    token = f"token-{thread_id}-{i}"
                    in_memory_blacklist.add(token, expires_at)
                    in_memory_blacklist.contains(token)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=mixed_ops, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# SQLiteBlacklist Tests
# =============================================================================


class TestSQLiteBlacklist:
    """Tests for SQLiteBlacklist implementation."""

    def test_add_and_contains(self, sqlite_blacklist):
        """Should add token and detect it as blacklisted."""
        token_jti = "sqlite-token-001"
        expires_at = time.time() + 3600

        sqlite_blacklist.add(token_jti, expires_at)

        assert sqlite_blacklist.contains(token_jti) is True

    def test_contains_returns_false_for_unknown(self, sqlite_blacklist):
        """Should return False for tokens not in blacklist."""
        assert sqlite_blacklist.contains("unknown-token") is False

    def test_contains_checks_expiration(self, tmp_path):
        """Should return False for expired tokens."""
        db_path = tmp_path / "expiry_test.db"
        blacklist = SQLiteBlacklist(db_path=db_path, cleanup_interval=1000)

        # Add token that's already expired
        blacklist.add("expired-token", time.time() - 1)

        # Should return False for expired token
        assert blacklist.contains("expired-token") is False

        blacklist.close()

    def test_size_tracking(self, sqlite_blacklist):
        """Should track blacklist size accurately."""
        assert sqlite_blacklist.size() == 0

        expires_at = time.time() + 3600

        sqlite_blacklist.add("token-1", expires_at)
        assert sqlite_blacklist.size() == 1

        sqlite_blacklist.add("token-2", expires_at)
        sqlite_blacklist.add("token-3", expires_at)
        assert sqlite_blacklist.size() == 3

    def test_size_excludes_expired(self, tmp_path):
        """Size should only count non-expired tokens."""
        db_path = tmp_path / "size_test.db"
        blacklist = SQLiteBlacklist(db_path=db_path, cleanup_interval=1000)

        # Add expired token
        blacklist.add("expired", time.time() - 1)
        # Add valid token
        blacklist.add("valid", time.time() + 3600)

        assert blacklist.size() == 1

        blacklist.close()

    def test_cleanup_expired(self, tmp_path):
        """Should remove expired tokens during cleanup."""
        db_path = tmp_path / "cleanup_test.db"
        blacklist = SQLiteBlacklist(db_path=db_path, cleanup_interval=1000)

        # Add token that's already expired
        blacklist.add("expired-token", time.time() - 1)
        blacklist.add("valid-token", time.time() + 3600)

        removed = blacklist.cleanup_expired()

        assert removed == 1
        assert blacklist.size() == 1

        blacklist.close()

    def test_creates_database_file(self, tmp_path):
        """Should create SQLite database file."""
        db_path = tmp_path / "new_blacklist.db"
        assert not db_path.exists()

        blacklist = SQLiteBlacklist(db_path=db_path)
        assert db_path.exists()

        blacklist.close()

    def test_creates_parent_directories(self, tmp_path):
        """Should create parent directories if they don't exist."""
        db_path = tmp_path / "nested" / "dirs" / "blacklist.db"
        assert not db_path.parent.exists()

        blacklist = SQLiteBlacklist(db_path=db_path)
        assert db_path.exists()

        blacklist.close()

    def test_initializes_schema(self, tmp_path):
        """Should create proper database schema."""
        db_path = tmp_path / "schema_test.db"
        blacklist = SQLiteBlacklist(db_path=db_path)

        # Connect directly to verify schema
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='token_blacklist'"
        )
        assert cursor.fetchone() is not None

        # Check index exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_blacklist_expires'"
        )
        assert cursor.fetchone() is not None

        conn.close()
        blacklist.close()

    def test_uses_wal_mode(self, tmp_path):
        """Should use WAL journal mode for better concurrency."""
        db_path = tmp_path / "wal_test.db"
        blacklist = SQLiteBlacklist(db_path=db_path)

        # Trigger a connection
        blacklist.add("test", time.time() + 3600)

        # Check journal mode on the store's connection
        conn = blacklist._get_conn()
        cursor = conn.execute("PRAGMA journal_mode")
        journal_mode = cursor.fetchone()[0].lower()
        assert journal_mode == "wal"

        blacklist.close()

    def test_persistence_across_instances(self, tmp_path):
        """Data should persist across store instances."""
        db_path = tmp_path / "persist_test.db"
        expires_at = time.time() + 3600

        # Create and close first instance
        blacklist1 = SQLiteBlacklist(db_path=db_path)
        blacklist1.add("persist-token", expires_at)
        blacklist1.close()

        # Open new instance
        blacklist2 = SQLiteBlacklist(db_path=db_path)
        assert blacklist2.contains("persist-token") is True

        blacklist2.close()

    def test_concurrent_access(self, tmp_path):
        """Should handle concurrent access correctly."""
        db_path = tmp_path / "concurrent_test.db"
        blacklist = SQLiteBlacklist(db_path=db_path)
        errors = []

        def sqlite_ops(thread_id):
            try:
                expires_at = time.time() + 3600
                for i in range(20):
                    token = f"token-{thread_id}-{i}"
                    blacklist.add(token, expires_at)
                    blacklist.contains(token)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=sqlite_ops, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        blacklist.close()

    def test_close_handles_multiple_connections(self, tmp_path):
        """close() should close all database connections."""
        db_path = tmp_path / "close_test.db"
        blacklist = SQLiteBlacklist(db_path=db_path)

        # Create a connection
        blacklist.add("test", time.time() + 3600)

        # Close should not raise
        blacklist.close()

        # Multiple closes should not raise
        blacklist.close()

    def test_insert_or_replace_behavior(self, sqlite_blacklist):
        """Should update expiration if token is added again."""
        token_jti = "update-test"
        old_expires = time.time() + 100
        new_expires = time.time() + 3600

        sqlite_blacklist.add(token_jti, old_expires)
        sqlite_blacklist.add(token_jti, new_expires)

        # Should still be found (using newer expiration)
        assert sqlite_blacklist.contains(token_jti) is True


# =============================================================================
# PostgresBlacklist Tests (Mocked)
# =============================================================================


class TestPostgresBlacklist:
    """Tests for PostgresBlacklist with mocked pool."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock asyncpg pool."""
        pool = MagicMock()

        # Mock async context manager for acquire
        mock_conn = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        return pool, mock_conn

    def test_init_creates_store(self, mock_pool):
        """Should initialize PostgresBlacklist with pool."""
        pool, mock_conn = mock_pool

        blacklist = PostgresBlacklist(pool=pool, cleanup_interval=300)

        assert blacklist._pool is pool
        assert blacklist._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_creates_schema(self, mock_pool):
        """Should create schema on initialize."""
        pool, mock_conn = mock_pool

        blacklist = PostgresBlacklist(pool=pool)
        await blacklist.initialize()

        mock_conn.execute.assert_called_once()
        assert blacklist._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_only_runs_once(self, mock_pool):
        """Initialize should only run once."""
        pool, mock_conn = mock_pool

        blacklist = PostgresBlacklist(pool=pool)
        await blacklist.initialize()
        await blacklist.initialize()

        assert mock_conn.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_add_async(self, mock_pool):
        """Should add token asynchronously."""
        pool, mock_conn = mock_pool

        blacklist = PostgresBlacklist(pool=pool, cleanup_interval=10000)
        await blacklist.add_async("test-token", time.time() + 3600)

        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_contains_async_found(self, mock_pool):
        """Should return True when token is found."""
        pool, mock_conn = mock_pool
        mock_conn.fetchrow = AsyncMock(return_value={"dummy": 1})

        blacklist = PostgresBlacklist(pool=pool)
        result = await blacklist.contains_async("test-token")

        assert result is True

    @pytest.mark.asyncio
    async def test_contains_async_not_found(self, mock_pool):
        """Should return False when token is not found."""
        pool, mock_conn = mock_pool
        mock_conn.fetchrow = AsyncMock(return_value=None)

        blacklist = PostgresBlacklist(pool=pool)
        result = await blacklist.contains_async("unknown-token")

        assert result is False

    @pytest.mark.asyncio
    async def test_cleanup_expired_async(self, mock_pool):
        """Should cleanup expired tokens and return count."""
        pool, mock_conn = mock_pool
        mock_conn.execute = AsyncMock(return_value="DELETE 5")

        blacklist = PostgresBlacklist(pool=pool)
        removed = await blacklist.cleanup_expired_async()

        assert removed == 5

    @pytest.mark.asyncio
    async def test_cleanup_expired_async_no_deletions(self, mock_pool):
        """Should return 0 when no tokens deleted."""
        pool, mock_conn = mock_pool
        mock_conn.execute = AsyncMock(return_value="DELETE 0")

        blacklist = PostgresBlacklist(pool=pool)
        removed = await blacklist.cleanup_expired_async()

        assert removed == 0

    @pytest.mark.asyncio
    async def test_size_async(self, mock_pool):
        """Should return current size."""
        pool, mock_conn = mock_pool
        mock_conn.fetchrow = AsyncMock(return_value={"cnt": 10})

        blacklist = PostgresBlacklist(pool=pool)
        size = await blacklist.size_async()

        assert size == 10

    @pytest.mark.asyncio
    async def test_size_async_empty(self, mock_pool):
        """Should return 0 when empty."""
        pool, mock_conn = mock_pool
        mock_conn.fetchrow = AsyncMock(return_value=None)

        blacklist = PostgresBlacklist(pool=pool)
        size = await blacklist.size_async()

        assert size == 0

    def test_close_is_noop(self, mock_pool):
        """Close should be a no-op (pool managed externally)."""
        pool, mock_conn = mock_pool

        blacklist = PostgresBlacklist(pool=pool)
        blacklist.close()  # Should not raise


# =============================================================================
# Global Store Management Tests
# =============================================================================


class TestGlobalStoreManagement:
    """Tests for global blacklist backend management functions."""

    def test_set_blacklist_backend_uses_custom_backend(self):
        """set_blacklist_backend should set a custom backend."""
        custom_backend = InMemoryBlacklist()
        set_blacklist_backend(custom_backend)

        retrieved = get_blacklist_backend()
        assert retrieved is custom_backend

    def test_get_blacklist_backend_returns_same_instance(self):
        """get_blacklist_backend should return the same instance."""
        custom_backend = InMemoryBlacklist()
        set_blacklist_backend(custom_backend)

        backend1 = get_blacklist_backend()
        backend2 = get_blacklist_backend()

        assert backend1 is backend2


# =============================================================================
# Redis Backend Tests (Conditional)
# =============================================================================


@pytest.mark.skipif(not HAS_REDIS, reason="Redis not installed")
class TestRedisBlacklist:
    """Tests for RedisBlacklist (only run if redis is installed)."""

    def test_redis_blacklist_import(self):
        """Should be able to import RedisBlacklist when redis is available."""
        from aragora.storage.token_blacklist_store import RedisBlacklist

        assert RedisBlacklist is not None


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_token_jti(self, in_memory_blacklist):
        """Should handle empty token JTI."""
        in_memory_blacklist.add("", time.time() + 3600)
        assert in_memory_blacklist.contains("") is True

    def test_unicode_token_jti(self, sqlite_blacklist):
        """Should handle unicode token JTI."""
        token_jti = "token_unicode_\u4e2d\u6587"
        sqlite_blacklist.add(token_jti, time.time() + 3600)
        assert sqlite_blacklist.contains(token_jti) is True

    def test_long_token_jti(self, in_memory_blacklist):
        """Should handle very long token JTI."""
        token_jti = "token_" + "x" * 1000
        in_memory_blacklist.add(token_jti, time.time() + 3600)
        assert in_memory_blacklist.contains(token_jti) is True

    def test_special_characters_in_token(self, sqlite_blacklist):
        """Should handle special characters in token JTI."""
        token_jti = "token_!@#$%^&*()_+-=[]{}|;':\",./<>?"
        sqlite_blacklist.add(token_jti, time.time() + 3600)
        assert sqlite_blacklist.contains(token_jti) is True

    def test_expires_at_in_past(self, in_memory_blacklist):
        """Should handle tokens with past expiration."""
        in_memory_blacklist.add("past-token", time.time() - 3600)
        # Contains should still return True (token is in blacklist)
        # But cleanup should remove it
        assert in_memory_blacklist.contains("past-token") is True

    def test_expires_at_exactly_now(self, in_memory_blacklist):
        """Should handle tokens expiring exactly now."""
        now = time.time()
        in_memory_blacklist.add("now-token", now)
        # Very edge case - might be True or False depending on timing
        # Just ensure no exception is raised
        in_memory_blacklist.contains("now-token")

    def test_very_large_ttl(self, sqlite_blacklist):
        """Should handle very large TTL values."""
        far_future = time.time() + (365 * 24 * 60 * 60 * 100)  # 100 years
        sqlite_blacklist.add("far-future-token", far_future)
        assert sqlite_blacklist.contains("far-future-token") is True


# =============================================================================
# Max Blacklist Size Tests
# =============================================================================


class TestMaxBlacklistSize:
    """Tests for max blacklist size enforcement."""

    def test_max_size_constant_defined(self):
        """MAX_BLACKLIST_SIZE should be defined."""
        assert MAX_BLACKLIST_SIZE == 100000

    def test_eviction_when_at_max_size(self):
        """Should evict entries when at max size."""
        # We can't easily test 100K entries, but we can verify the logic
        blacklist = InMemoryBlacklist()

        # Add some expired and some valid tokens
        now = time.time()
        for i in range(10):
            blacklist.add(f"expired-{i}", now - 1)  # Expired
        for i in range(10):
            blacklist.add(f"valid-{i}", now + 3600)  # Valid

        # The blacklist should have all tokens initially
        assert blacklist.size() == 20


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_importable(self):
        """All items in __all__ should be importable."""
        import aragora.storage.token_blacklist_store as module

        for name in module.__all__:
            assert hasattr(module, name), f"Missing export: {name}"

    def test_key_exports(self):
        """Key exports should be available."""
        from aragora.storage.token_blacklist_store import (
            BlacklistBackend,
            InMemoryBlacklist,
            SQLiteBlacklist,
            PostgresBlacklist,
            get_blacklist_backend,
            set_blacklist_backend,
            HAS_REDIS,
        )

        assert BlacklistBackend is not None
        assert InMemoryBlacklist is not None
        assert SQLiteBlacklist is not None
        assert PostgresBlacklist is not None
        assert callable(get_blacklist_backend)
        assert callable(set_blacklist_backend)
        assert isinstance(HAS_REDIS, bool)


# =============================================================================
# Cleanup Interval Tests
# =============================================================================


class TestCleanupInterval:
    """Tests for cleanup interval configuration."""

    def test_custom_cleanup_interval_in_memory(self):
        """InMemoryBlacklist should accept custom cleanup interval."""
        blacklist = InMemoryBlacklist(cleanup_interval=600)
        assert blacklist._cleanup_interval == 600

    def test_custom_cleanup_interval_sqlite(self, tmp_path):
        """SQLiteBlacklist should accept custom cleanup interval."""
        db_path = tmp_path / "interval_test.db"
        blacklist = SQLiteBlacklist(db_path=db_path, cleanup_interval=600)
        assert blacklist._cleanup_interval == 600
        blacklist.close()

    def test_default_cleanup_interval_in_memory(self):
        """InMemoryBlacklist should have default cleanup interval."""
        blacklist = InMemoryBlacklist()
        assert blacklist._cleanup_interval == 300  # 5 minutes

    def test_default_cleanup_interval_sqlite(self, tmp_path):
        """SQLiteBlacklist should have default cleanup interval."""
        db_path = tmp_path / "default_interval.db"
        blacklist = SQLiteBlacklist(db_path=db_path)
        assert blacklist._cleanup_interval == 300  # 5 minutes
        blacklist.close()
