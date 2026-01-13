"""Tests for connection pool functionality."""

import pytest
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from aragora.storage.schema import ConnectionPool


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        # Create the database and a test table
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        conn.commit()
        conn.close()
        yield db_path


@pytest.fixture
def pool(temp_db):
    """Create a connection pool."""
    p = ConnectionPool(temp_db, max_connections=3)
    yield p
    p.close()


class TestConnectionPool:
    """Tests for ConnectionPool."""

    def test_acquire_and_release(self, pool):
        """Can acquire and release connections."""
        conn = pool.acquire()
        assert conn is not None

        # Connection should be usable
        cursor = conn.execute("SELECT 1")
        assert cursor.fetchone() == (1,)

        pool.release(conn)

        # Stats should show idle connection
        stats = pool.stats()
        assert stats["idle"] == 1
        assert stats["active"] == 0

    def test_context_manager(self, pool):
        """Context manager acquires and releases."""
        with pool.connection() as conn:
            conn.execute("INSERT INTO test (value) VALUES (?)", ("test",))

        stats = pool.stats()
        assert stats["active"] == 0
        assert stats["idle"] == 1

    def test_context_manager_commits(self, pool):
        """Context manager commits on success."""
        with pool.connection() as conn:
            conn.execute("INSERT INTO test (value) VALUES (?)", ("committed",))

        # Verify data was committed
        with pool.connection() as conn:
            cursor = conn.execute("SELECT value FROM test WHERE value = ?", ("committed",))
            assert cursor.fetchone() is not None

    def test_context_manager_rollbacks_on_error(self, pool):
        """Context manager rolls back on exception."""
        try:
            with pool.connection() as conn:
                conn.execute("INSERT INTO test (value) VALUES (?)", ("rollback_test",))
                raise ValueError("Intentional error")
        except ValueError:
            pass

        # Verify data was rolled back
        with pool.connection() as conn:
            cursor = conn.execute("SELECT value FROM test WHERE value = ?", ("rollback_test",))
            assert cursor.fetchone() is None

    def test_reuses_idle_connections(self, pool):
        """Released connections are reused."""
        conn1 = pool.acquire()
        conn1_id = id(conn1)
        pool.release(conn1)

        conn2 = pool.acquire()
        assert id(conn2) == conn1_id  # Same connection object

    def test_respects_max_connections(self, pool):
        """Pool doesn't exceed max connections."""
        connections = [pool.acquire() for _ in range(3)]

        stats = pool.stats()
        assert stats["active"] == 3
        assert stats["total"] == 3

        # Release one
        pool.release(connections[0])
        stats = pool.stats()
        assert stats["active"] == 2
        assert stats["idle"] == 1

    def test_waits_when_at_max(self, temp_db):
        """Blocks when pool is at max capacity."""
        pool = ConnectionPool(temp_db, max_connections=1, timeout=1.0)

        conn1 = pool.acquire()

        # Should timeout waiting for connection
        with pytest.raises(TimeoutError):
            pool.acquire(timeout=0.1)

        pool.release(conn1)
        pool.close()

    def test_thread_safety(self, temp_db):
        """Pool is thread-safe under concurrent access."""
        pool = ConnectionPool(temp_db, max_connections=5)
        results = []
        errors = []

        def worker(worker_id):
            try:
                for _ in range(10):
                    with pool.connection() as conn:
                        conn.execute(
                            "INSERT INTO test (value) VALUES (?)", (f"worker_{worker_id}",)
                        )
                        time.sleep(0.001)  # Small delay to increase contention
                results.append(worker_id)
            except Exception as e:
                errors.append((worker_id, e))

        # Run 10 concurrent workers
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            for future in as_completed(futures):
                future.result()  # Propagate any exceptions

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 10

        # Verify all inserts succeeded
        with pool.connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM test")
            count = cursor.fetchone()[0]
            assert count == 100  # 10 workers * 10 inserts each

        pool.close()

    def test_stats(self, pool):
        """Stats reflect pool state."""
        stats = pool.stats()
        assert stats["active"] == 0
        assert stats["idle"] == 0
        assert stats["total"] == 0
        assert stats["max"] == 3

        conn = pool.acquire()
        stats = pool.stats()
        assert stats["active"] == 1

        pool.release(conn)
        stats = pool.stats()
        assert stats["idle"] == 1
        assert stats["active"] == 0

    def test_close_pool(self, temp_db):
        """Closing pool closes all connections."""
        pool = ConnectionPool(temp_db, max_connections=3)

        # Create some connections
        conn1 = pool.acquire()
        conn2 = pool.acquire()
        pool.release(conn1)
        pool.release(conn2)

        assert pool.stats()["idle"] == 2

        pool.close()

        # Pool should be closed
        with pytest.raises(RuntimeError):
            pool.acquire()

    def test_release_connection_not_from_pool(self, pool, temp_db):
        """Releasing foreign connection is handled gracefully."""
        foreign_conn = sqlite3.connect(temp_db)

        # Should not raise, just log warning
        pool.release(foreign_conn)

        foreign_conn.close()

    def test_broken_connection_discarded(self, temp_db):
        """Broken connections are discarded."""
        pool = ConnectionPool(temp_db, max_connections=2)

        # Acquire and release a connection
        conn = pool.acquire()
        pool.release(conn)

        # Manually close the connection to simulate it being broken
        conn.close()

        # Next acquire should get a new connection (broken one discarded)
        conn2 = pool.acquire()
        assert conn2 is not None

        # Should still work
        cursor = conn2.execute("SELECT 1")
        assert cursor.fetchone() == (1,)

        pool.release(conn2)
        pool.close()

    def test_repr(self, pool):
        """Repr shows useful information."""
        repr_str = repr(pool)
        assert "ConnectionPool" in repr_str
        assert "active=" in repr_str
        assert "idle=" in repr_str

    def test_concurrent_acquire_release(self, temp_db):
        """Multiple threads can acquire/release concurrently."""
        # Use more connections than threads to avoid timeouts
        pool = ConnectionPool(temp_db, max_connections=10, timeout=10.0)
        acquired_count = [0]
        lock = threading.Lock()

        def worker():
            for _ in range(20):
                conn = pool.acquire(timeout=5.0)
                with lock:
                    acquired_count[0] += 1
                time.sleep(0.001)
                pool.release(conn)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert acquired_count[0] == 100  # 5 threads * 20 acquires

        pool.close()
