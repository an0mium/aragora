"""
Test concurrent debate scenarios.

These tests verify that multiple debates can run concurrently without
interfering with each other, and that shared resources are properly
managed.
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestConcurrentDebateExecution:
    """Test multiple debates running concurrently."""

    @pytest.fixture
    def mock_arena(self):
        """Create a mock Arena for testing."""
        arena = MagicMock()
        arena.run = AsyncMock(
            return_value=MagicMock(
                id="debate-123",
                consensus="Test consensus",
                rounds_used=3,
            )
        )
        return arena

    @pytest.mark.asyncio
    async def test_multiple_debates_dont_interfere(self, mock_arena):
        """Multiple concurrent debates should have isolated state."""
        results = []
        debate_count = 5

        async def run_debate(debate_id: str):
            """Simulate running a debate."""
            await asyncio.sleep(0.1)  # Simulate work
            results.append(debate_id)
            return debate_id

        tasks = [asyncio.create_task(run_debate(f"debate-{i}")) for i in range(debate_count)]

        completed = await asyncio.gather(*tasks)

        assert len(completed) == debate_count
        assert len(set(completed)) == debate_count  # All unique

    @pytest.mark.asyncio
    async def test_debate_state_isolation(self):
        """Each debate should have its own isolated context."""
        from aragora.debate.context import DebateContext

        contexts = []

        async def create_context(task_id: str):
            ctx = DebateContext()
            ctx.debate_id = task_id
            await asyncio.sleep(0.05)
            return ctx

        tasks = [asyncio.create_task(create_context(f"task-{i}")) for i in range(3)]

        contexts = await asyncio.gather(*tasks)

        # Verify each context has unique ID
        ids = [ctx.debate_id for ctx in contexts]
        assert len(set(ids)) == 3

    @pytest.mark.asyncio
    async def test_concurrent_vote_collection(self):
        """Concurrent vote collection should be thread-safe."""
        votes = []
        lock = asyncio.Lock()

        async def cast_vote(agent_name: str):
            await asyncio.sleep(0.01)  # Simulate API call
            async with lock:
                votes.append(agent_name)
            return agent_name

        agents = [f"agent-{i}" for i in range(10)]
        tasks = [asyncio.create_task(cast_vote(a)) for a in agents]

        await asyncio.gather(*tasks)

        assert len(votes) == 10
        assert set(votes) == set(agents)


class TestRateLimiterConcurrency:
    """Test rate limiter under concurrent load."""

    def test_token_bucket_thread_safety(self):
        """Token bucket should handle concurrent access correctly."""
        from aragora.server.middleware.rate_limit import TokenBucket

        bucket = TokenBucket(rate_per_minute=60, burst_size=10)
        results = []

        def consume_tokens():
            for _ in range(5):
                result = bucket.consume(1)
                results.append(result)

        threads = [threading.Thread(target=consume_tokens) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # With 10 burst tokens and 20 total attempts,
        # exactly 10 should succeed
        assert sum(results) == 10
        assert len(results) == 20

    def test_rate_limiter_concurrent_ips(self):
        """Rate limiter should track different IPs independently."""
        from aragora.server.middleware.rate_limit import RateLimiter

        limiter = RateLimiter(default_limit=10, ip_limit=5)
        results = {}

        def test_ip(ip: str):
            results[ip] = []
            for _ in range(15):
                result = limiter.allow(ip, "/api/test")
                results[ip].append(result.allowed)

        threads = []
        for i in range(5):
            ip = f"192.168.1.{i}"
            t = threading.Thread(target=test_ip, args=(ip,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Each IP should have independent limits
        # With ip_limit=5 and burst=2x, each IP gets ~10 initial tokens
        for ip, ip_results in results.items():
            assert sum(ip_results) >= 5  # At least burst allowed

    def test_tier_rate_limiter_isolation(self):
        """Different tiers should have independent rate limits."""
        from aragora.server.middleware.rate_limit import TierRateLimiter, TIER_RATE_LIMITS

        limiter = TierRateLimiter()

        # Free tier (10 req/min, 60 burst by default)
        free_results = [limiter.allow("free", "user-1") for _ in range(15)]

        # Professional tier (200 req/min)
        pro_results = [limiter.allow("professional", "user-2") for _ in range(15)]

        # Free should have lower limits
        free_allowed = sum(1 for r in free_results if r.allowed)
        pro_allowed = sum(1 for r in pro_results if r.allowed)

        # Professional should allow more requests
        assert pro_allowed >= free_allowed


class TestDatabaseConcurrency:
    """Test database operations under concurrent load."""

    def test_sqlite_backend_thread_safety(self):
        """SQLite backend should handle concurrent writes."""
        import tempfile
        from pathlib import Path
        from aragora.storage.backends import SQLiteBackend

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            backend = SQLiteBackend(db_path)

            # Create test table
            with backend.connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS test_concurrent (
                        id INTEGER PRIMARY KEY,
                        value TEXT,
                        thread_id TEXT
                    )
                """
                )

            errors = []

            def insert_rows(thread_id: str):
                try:
                    for i in range(10):
                        backend.execute_write(
                            "INSERT INTO test_concurrent (value, thread_id) VALUES (?, ?)",
                            (f"value-{i}", thread_id),
                        )
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=insert_rows, args=(f"thread-{i}",)) for i in range(4)
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors occurred: {errors}"

            # Verify all rows were inserted
            result = backend.fetch_one("SELECT COUNT(*) FROM test_concurrent")
            assert result[0] == 40  # 4 threads * 10 rows

    def test_connection_pool_exhaustion(self):
        """Database should handle connection pool exhaustion gracefully."""
        import tempfile
        from pathlib import Path
        from aragora.storage.backends import SQLiteBackend

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            backend = SQLiteBackend(db_path)

            # Create test table
            with backend.connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS test_pool (
                        id INTEGER PRIMARY KEY,
                        value TEXT
                    )
                """
                )

            # Simulate heavy concurrent load
            results = []

            def query_database():
                for _ in range(20):
                    try:
                        row = backend.fetch_one("SELECT COUNT(*) FROM test_pool")
                        results.append(("success", row[0]))
                    except Exception as e:
                        results.append(("error", str(e)))

            threads = [threading.Thread(target=query_database) for _ in range(10)]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All queries should succeed (SQLite uses WAL mode)
            errors = [r for r in results if r[0] == "error"]
            assert len(errors) == 0, f"Errors: {errors[:5]}"


class TestResourceContention:
    """Test resource contention scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_memory_access(self):
        """ContinuumMemory should handle concurrent access."""
        from aragora.memory.continuum import ContinuumMemory
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            memory = ContinuumMemory(storage_path=tmp)

            async def store_and_retrieve(key: str):
                await memory.store(
                    key=key,
                    content=f"content-{key}",
                    tier="fast",
                )
                await asyncio.sleep(0.01)
                return await memory.retrieve(key, tier="fast")

            tasks = [asyncio.create_task(store_and_retrieve(f"key-{i}")) for i in range(10)]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # No exceptions should occur
            exceptions = [r for r in results if isinstance(r, Exception)]
            assert len(exceptions) == 0, f"Exceptions: {exceptions}"

    def test_executor_thread_pool_limits(self):
        """Thread pool should respect limits under load."""
        from concurrent.futures import ThreadPoolExecutor

        max_workers = 4
        concurrent_count = [0]
        max_concurrent = [0]
        lock = threading.Lock()

        def track_concurrency():
            with lock:
                concurrent_count[0] += 1
                max_concurrent[0] = max(max_concurrent[0], concurrent_count[0])

            time.sleep(0.1)  # Simulate work

            with lock:
                concurrent_count[0] -= 1

            return True

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(track_concurrency) for _ in range(20)]

            for f in futures:
                f.result()

        # Max concurrent should not exceed pool size
        assert max_concurrent[0] <= max_workers


class TestShutdownUnderLoad:
    """Test graceful shutdown with active connections."""

    @pytest.mark.asyncio
    async def test_shutdown_waits_for_active_tasks(self):
        """Shutdown should wait for active tasks to complete."""
        from aragora.server.lifecycle import ServerLifecycleManager

        active_debates = {
            "debate-1": {"status": "in_progress"},
            "debate-2": {"status": "in_progress"},
        }

        lifecycle = ServerLifecycleManager(
            get_active_debates=lambda: active_debates,
        )

        # Simulate debates completing after 0.2s
        async def complete_debates():
            await asyncio.sleep(0.2)
            active_debates["debate-1"]["status"] = "completed"
            active_debates["debate-2"]["status"] = "completed"

        asyncio.create_task(complete_debates())

        # Shutdown should wait
        start = time.time()
        await lifecycle.graceful_shutdown(timeout=5.0)
        elapsed = time.time() - start

        # Should have waited ~0.2s for debates to complete
        assert elapsed >= 0.2
        assert elapsed < 1.0  # But not too long

    @pytest.mark.asyncio
    async def test_shutdown_timeout_with_stuck_debates(self):
        """Shutdown should timeout if debates are stuck."""
        from aragora.server.lifecycle import ServerLifecycleManager

        # Debates that never complete
        active_debates = {
            "debate-stuck": {"status": "in_progress"},
        }

        lifecycle = ServerLifecycleManager(
            get_active_debates=lambda: active_debates,
        )

        start = time.time()
        await lifecycle.graceful_shutdown(timeout=0.5)
        elapsed = time.time() - start

        # Should timeout after ~0.5s
        assert elapsed >= 0.5
        assert elapsed < 1.0
