"""
E2E tests for connection pool under load.

Tests the system's ability to:
- Handle concurrent connections exceeding pool size
- Timeout on pool exhaustion
- Recover from database lock scenarios
- Track pool metrics correctly
- Handle WAL mode concurrent writes
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import patch

import pytest

from aragora.storage.schema import ConnectionPool, get_wal_connection


class TestConnectionPoolBasics:
    """Test basic connection pool functionality."""

    @pytest.fixture
    def temp_db(self, tmp_path: Path) -> Path:
        """Create a temporary database."""
        db_path = tmp_path / "test.db"
        # Initialize with WAL mode
        conn = get_wal_connection(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        conn.commit()
        conn.close()
        return db_path

    def test_pool_creates_connections_up_to_max(self, temp_db: Path):
        """Verify pool creates connections up to max_connections."""
        pool = ConnectionPool(temp_db, max_connections=5)

        connections = []
        try:
            # Acquire all 5 connections
            for _ in range(5):
                conn = pool.acquire(timeout=1.0)
                connections.append(conn)

            stats = pool.stats()
            assert stats["active"] == 5
            assert stats["idle"] == 0
        finally:
            for conn in connections:
                pool.release(conn)
            pool.close()

    def test_pool_reuses_idle_connections(self, temp_db: Path):
        """Verify pool reuses released connections."""
        pool = ConnectionPool(temp_db, max_connections=3)

        try:
            # Acquire and release
            conn1 = pool.acquire()
            pool.release(conn1)

            # Should reuse the same connection
            conn2 = pool.acquire()

            stats = pool.stats()
            assert stats["active"] == 1
            assert stats["idle"] == 0

            pool.release(conn2)
        finally:
            pool.close()

    def test_pool_timeout_when_exhausted(self, temp_db: Path):
        """Verify pool raises TimeoutError when exhausted."""
        pool = ConnectionPool(temp_db, max_connections=2, timeout=0.5)

        connections = []
        try:
            # Exhaust the pool
            connections.append(pool.acquire())
            connections.append(pool.acquire())

            # Should timeout waiting for third connection
            with pytest.raises(TimeoutError) as exc_info:
                pool.acquire(timeout=0.2)

            assert "Timeout waiting for connection" in str(exc_info.value)
        finally:
            for conn in connections:
                pool.release(conn)
            pool.close()


class TestConnectionPoolConcurrency:
    """Test pool behavior under concurrent load."""

    @pytest.fixture
    def temp_db(self, tmp_path: Path) -> Path:
        """Create a temporary database."""
        db_path = tmp_path / "concurrent.db"
        conn = get_wal_connection(str(db_path))
        conn.execute("CREATE TABLE counter (id INTEGER PRIMARY KEY, value INTEGER)")
        conn.execute("INSERT INTO counter (id, value) VALUES (1, 0)")
        conn.commit()
        conn.close()
        return db_path

    def test_concurrent_connections_within_limit(self, temp_db: Path):
        """Verify pool handles concurrent access within limits."""
        pool = ConnectionPool(temp_db, max_connections=10)
        results: List[int] = []
        errors: List[Exception] = []

        def db_operation(operation_id: int) -> int:
            try:
                with pool.connection() as conn:
                    time.sleep(0.05)  # Hold connection briefly
                    conn.execute("SELECT value FROM counter WHERE id = 1")
                    return operation_id
            except Exception as e:
                errors.append(e)
                return -1

        try:
            # Run 10 concurrent operations (should fit in pool)
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(db_operation, i) for i in range(10)]
                results = [f.result() for f in as_completed(futures)]

            assert len(errors) == 0
            assert len(results) == 10
        finally:
            pool.close()

    def test_concurrent_connections_exceed_limit_with_wait(self, temp_db: Path):
        """Verify operations queue when pool exhausted."""
        pool = ConnectionPool(temp_db, max_connections=3, timeout=5.0)
        results: List[int] = []
        errors: List[Exception] = []
        operation_times: List[float] = []

        def db_operation(operation_id: int) -> int:
            try:
                start = time.time()
                with pool.connection() as conn:
                    time.sleep(0.1)  # Hold connection longer
                    conn.execute("SELECT value FROM counter WHERE id = 1")
                operation_times.append(time.time() - start)
                return operation_id
            except Exception as e:
                errors.append(e)
                return -1

        try:
            # Run 9 operations with only 3 connections
            # Should complete in ~3 batches
            with ThreadPoolExecutor(max_workers=9) as executor:
                futures = [executor.submit(db_operation, i) for i in range(9)]
                results = [f.result() for f in as_completed(futures)]

            assert len(errors) == 0
            assert len(results) == 9
            # Some operations should have waited
            assert max(operation_times) > 0.1  # At least some wait time
        finally:
            pool.close()

    def test_pool_handles_rapid_acquire_release(self, temp_db: Path):
        """Verify pool handles rapid acquire/release cycles."""
        pool = ConnectionPool(temp_db, max_connections=10, timeout=5.0)

        def rapid_cycle(cycle_id: int) -> bool:
            for _ in range(5):
                conn = pool.acquire(timeout=5.0)
                conn.execute("SELECT 1")
                pool.release(conn)
            return True

        try:
            # Use fewer workers to reduce contention
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(rapid_cycle, i) for i in range(10)]
                results = [f.result() for f in as_completed(futures)]

            assert all(results)
        finally:
            pool.close()


class TestPoolExhaustionMetrics:
    """Test metrics collection during pool exhaustion."""

    @pytest.fixture
    def temp_db(self, tmp_path: Path) -> Path:
        """Create a temporary database."""
        db_path = tmp_path / "metrics.db"
        conn = get_wal_connection(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()
        return db_path

    def test_stats_track_active_and_idle(self, temp_db: Path):
        """Verify pool stats track active/idle correctly."""
        pool = ConnectionPool(temp_db, max_connections=5)

        try:
            # Initially empty
            stats = pool.stats()
            assert stats["active"] == 0
            assert stats["idle"] == 0

            # Acquire some connections
            conn1 = pool.acquire()
            conn2 = pool.acquire()

            stats = pool.stats()
            assert stats["active"] == 2
            assert stats["idle"] == 0

            # Release one
            pool.release(conn1)

            stats = pool.stats()
            assert stats["active"] == 1
            assert stats["idle"] == 1

            pool.release(conn2)
        finally:
            pool.close()

    def test_stats_max_capacity(self, temp_db: Path):
        """Verify stats show max capacity."""
        pool = ConnectionPool(temp_db, max_connections=10)

        try:
            stats = pool.stats()
            assert stats["max"] == 10
        finally:
            pool.close()


class TestWALModeConcurrentWrites:
    """Test WAL mode handles concurrent write operations."""

    @pytest.fixture
    def temp_db(self, tmp_path: Path) -> Path:
        """Create a WAL-mode database."""
        db_path = tmp_path / "wal.db"
        conn = get_wal_connection(str(db_path))
        conn.execute("CREATE TABLE writes (id INTEGER PRIMARY KEY, writer_id INTEGER, timestamp REAL)")
        conn.commit()
        conn.close()
        return db_path

    def test_concurrent_writes_succeed(self, temp_db: Path):
        """Verify concurrent writes succeed with WAL mode."""
        pool = ConnectionPool(temp_db, max_connections=10)
        write_count = 50
        results: List[bool] = []
        errors: List[Exception] = []

        def write_operation(writer_id: int) -> bool:
            try:
                with pool.connection() as conn:
                    conn.execute(
                        "INSERT INTO writes (writer_id, timestamp) VALUES (?, ?)",
                        (writer_id, time.time())
                    )
                    conn.commit()
                return True
            except Exception as e:
                errors.append(e)
                return False

        try:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(write_operation, i) for i in range(write_count)]
                results = [f.result() for f in as_completed(futures)]

            # All writes should succeed
            assert all(results)
            assert len(errors) == 0

            # Verify all rows written
            with pool.connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM writes")
                count = cursor.fetchone()[0]
            assert count == write_count
        finally:
            pool.close()

    def test_concurrent_read_write_mix(self, temp_db: Path):
        """Verify concurrent reads and writes work together."""
        pool = ConnectionPool(temp_db, max_connections=10)
        results: List[Any] = []
        errors: List[Exception] = []

        def read_operation() -> int:
            try:
                with pool.connection() as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM writes")
                    return cursor.fetchone()[0]
            except Exception as e:
                errors.append(e)
                return -1

        def write_operation(writer_id: int) -> bool:
            try:
                with pool.connection() as conn:
                    conn.execute(
                        "INSERT INTO writes (writer_id, timestamp) VALUES (?, ?)",
                        (writer_id, time.time())
                    )
                    conn.commit()
                return True
            except Exception as e:
                errors.append(e)
                return False

        try:
            with ThreadPoolExecutor(max_workers=20) as executor:
                # Mix of read and write operations
                write_futures = [executor.submit(write_operation, i) for i in range(30)]
                read_futures = [executor.submit(read_operation) for _ in range(20)]

                write_results = [f.result() for f in as_completed(write_futures)]
                read_results = [f.result() for f in as_completed(read_futures)]

            # All operations should succeed
            assert all(write_results)
            assert all(r >= 0 for r in read_results)
            assert len(errors) == 0
        finally:
            pool.close()


class TestPoolRecoveryFromLock:
    """Test pool recovery from database lock scenarios."""

    @pytest.fixture
    def temp_db(self, tmp_path: Path) -> Path:
        """Create a temporary database."""
        db_path = tmp_path / "lock_test.db"
        conn = get_wal_connection(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        conn.commit()
        conn.close()
        return db_path

    def test_connection_validates_on_acquire(self, temp_db: Path):
        """Verify pool validates connections on acquire."""
        pool = ConnectionPool(temp_db, max_connections=3)

        try:
            # Acquire and release a connection
            conn1 = pool.acquire()
            pool.release(conn1)

            # Should successfully reuse or create new
            conn2 = pool.acquire()
            # Connection should be valid
            conn2.execute("SELECT 1")
            pool.release(conn2)
        finally:
            pool.close()

    def test_broken_connection_discarded(self, temp_db: Path):
        """Verify broken connections are discarded from pool."""
        pool = ConnectionPool(temp_db, max_connections=3)

        try:
            # Acquire a connection
            conn = pool.acquire()

            # Force close it (simulating broken connection)
            conn.close()

            # Release it back (pool should handle gracefully)
            pool.release(conn)

            # Next acquire should work (either revalidate fails and creates new, or creates new)
            conn2 = pool.acquire()
            result = conn2.execute("SELECT 1").fetchone()
            assert result is not None
            pool.release(conn2)
        finally:
            pool.close()


class TestPoolClosure:
    """Test pool cleanup and closure."""

    @pytest.fixture
    def temp_db(self, tmp_path: Path) -> Path:
        """Create a temporary database."""
        db_path = tmp_path / "closure.db"
        conn = get_wal_connection(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()
        return db_path

    def test_close_releases_all_connections(self, temp_db: Path):
        """Verify close() releases all connections."""
        pool = ConnectionPool(temp_db, max_connections=5)

        # Create some connections
        connections = [pool.acquire() for _ in range(3)]
        for conn in connections:
            pool.release(conn)

        # Close pool
        pool.close()

        # Pool should be empty
        stats = pool.stats()
        assert stats["active"] == 0
        assert stats["idle"] == 0

    def test_acquire_after_close_raises(self, temp_db: Path):
        """Verify acquire raises after pool is closed."""
        pool = ConnectionPool(temp_db, max_connections=3)
        pool.close()

        with pytest.raises(Exception):  # Should raise DatabaseError
            pool.acquire()

    def test_context_manager_releases_on_exception(self, temp_db: Path):
        """Verify context manager releases connection on exception."""
        pool = ConnectionPool(temp_db, max_connections=3)

        try:
            try:
                with pool.connection() as conn:
                    conn.execute("SELECT 1")
                    raise ValueError("Test exception")
            except ValueError:
                pass

            # Connection should be released back to pool
            stats = pool.stats()
            assert stats["active"] == 0
            assert stats["idle"] >= 0  # Could be 0 if connection was broken
        finally:
            pool.close()


class TestAsyncPoolBehavior:
    """Test pool behavior in async contexts."""

    @pytest.fixture
    def temp_db(self, tmp_path: Path) -> Path:
        """Create a temporary database."""
        db_path = tmp_path / "async.db"
        conn = get_wal_connection(str(db_path))
        conn.execute("CREATE TABLE async_test (id INTEGER PRIMARY KEY, value INTEGER)")
        conn.commit()
        conn.close()
        return db_path

    @pytest.mark.asyncio
    async def test_pool_works_from_async_context(self, temp_db: Path):
        """Verify pool can be used from async context via run_in_executor."""
        pool = ConnectionPool(temp_db, max_connections=5)
        loop = asyncio.get_event_loop()

        def sync_operation(op_id: int) -> int:
            with pool.connection() as conn:
                conn.execute("INSERT INTO async_test (value) VALUES (?)", (op_id,))
                conn.commit()
            return op_id

        try:
            # Run multiple operations from async context
            tasks = [
                loop.run_in_executor(None, sync_operation, i)
                for i in range(10)
            ]
            results = await asyncio.gather(*tasks)

            assert len(results) == 10
            assert set(results) == set(range(10))

            # Verify data was written
            with pool.connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM async_test")
                count = cursor.fetchone()[0]
            assert count == 10
        finally:
            pool.close()

    @pytest.mark.asyncio
    async def test_concurrent_async_operations(self, temp_db: Path):
        """Verify many concurrent async operations don't deadlock."""
        pool = ConnectionPool(temp_db, max_connections=5)
        loop = asyncio.get_event_loop()
        operation_count = 50

        def sync_read_write(op_id: int) -> bool:
            with pool.connection() as conn:
                # Write
                conn.execute("INSERT INTO async_test (value) VALUES (?)", (op_id,))
                conn.commit()
                # Read
                cursor = conn.execute("SELECT value FROM async_test WHERE value = ?", (op_id,))
                result = cursor.fetchone()
                return result is not None

        try:
            tasks = [
                loop.run_in_executor(None, sync_read_write, i)
                for i in range(operation_count)
            ]
            results = await asyncio.gather(*tasks)

            assert all(results)
            assert len(results) == operation_count
        finally:
            pool.close()


class TestPoolConfigVariations:
    """Test pool with different configurations."""

    @pytest.fixture
    def temp_db(self, tmp_path: Path) -> Path:
        """Create a temporary database."""
        db_path = tmp_path / "config.db"
        conn = get_wal_connection(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()
        return db_path

    def test_single_connection_pool(self, temp_db: Path):
        """Verify pool works with single connection."""
        pool = ConnectionPool(temp_db, max_connections=1)

        try:
            # Should work for sequential access
            with pool.connection() as conn:
                conn.execute("SELECT 1")

            with pool.connection() as conn:
                conn.execute("SELECT 1")
        finally:
            pool.close()

    def test_large_pool_size(self, temp_db: Path):
        """Verify large pool size works correctly."""
        pool = ConnectionPool(temp_db, max_connections=100)

        try:
            # Acquire many connections
            connections = []
            for _ in range(50):
                conn = pool.acquire()
                connections.append(conn)

            stats = pool.stats()
            assert stats["active"] == 50
            assert stats["max"] == 100

            # Release all
            for conn in connections:
                pool.release(conn)
        finally:
            pool.close()

    def test_custom_timeout(self, temp_db: Path):
        """Verify custom timeout is respected."""
        pool = ConnectionPool(temp_db, max_connections=1, timeout=0.1)

        try:
            # Hold the only connection
            conn = pool.acquire()

            start = time.time()
            with pytest.raises(TimeoutError):
                pool.acquire(timeout=0.1)
            elapsed = time.time() - start

            # Should timeout quickly
            assert elapsed < 0.5

            pool.release(conn)
        finally:
            pool.close()
