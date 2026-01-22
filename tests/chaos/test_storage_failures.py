"""
Chaos tests for storage and memory failure scenarios.

Tests system resilience when:
- Database connections fail
- Cache becomes unavailable
- Disk I/O errors occur
- Memory stores corrupt or lose data
- Concurrent writes cause conflicts
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class MockFailingDatabase:
    """A database mock that can simulate various failures."""

    def __init__(
        self,
        failure_rate: float = 0.0,
        latency: float = 0.0,
        connection_failures: int = 0,
    ):
        self.failure_rate = failure_rate
        self.latency = latency
        self.connection_failures = connection_failures
        self._connection_attempts = 0
        self._data: dict[str, Any] = {}

    async def connect(self):
        """Simulate connection with possible failures."""
        self._connection_attempts += 1
        if self._connection_attempts <= self.connection_failures:
            raise ConnectionError("Database connection failed")
        await asyncio.sleep(self.latency)
        return self

    async def execute(self, query: str, params: tuple = ()) -> Any:
        """Execute query with possible failures."""
        import random

        await asyncio.sleep(self.latency)
        if random.random() < self.failure_rate:
            raise sqlite3.OperationalError("Database is locked")
        return MagicMock(rowcount=1)

    async def get(self, key: str) -> Any:
        """Get value with possible failures."""
        import random

        await asyncio.sleep(self.latency)
        if random.random() < self.failure_rate:
            raise sqlite3.OperationalError("Read error")
        return self._data.get(key)

    async def set(self, key: str, value: Any) -> bool:
        """Set value with possible failures."""
        import random

        await asyncio.sleep(self.latency)
        if random.random() < self.failure_rate:
            raise sqlite3.OperationalError("Write error")
        self._data[key] = value
        return True


class TestDatabaseConnectionFailures:
    """Tests for database connection failure handling."""

    @pytest.mark.asyncio
    async def test_connection_retry_succeeds(self):
        """Should retry connection and eventually succeed."""
        db = MockFailingDatabase(connection_failures=2)

        connection = None
        for attempt in range(5):
            try:
                connection = await db.connect()
                break
            except ConnectionError:
                await asyncio.sleep(0.01)

        assert connection is not None
        assert db._connection_attempts == 3

    @pytest.mark.asyncio
    async def test_connection_exhausts_retries(self):
        """Should fail gracefully after exhausting retries."""
        db = MockFailingDatabase(connection_failures=10)

        connection = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                connection = await db.connect()
                break
            except ConnectionError:
                await asyncio.sleep(0.01)

        assert connection is None
        assert db._connection_attempts == max_retries

    @pytest.mark.asyncio
    async def test_connection_pool_handles_failures(self):
        """Connection pool should handle individual connection failures."""
        # Simulate a pool of connections with some failing
        connections = []
        for i in range(5):
            db = MockFailingDatabase(connection_failures=1 if i % 2 == 0 else 0)
            try:
                conn = await db.connect()
                connections.append(conn)
            except ConnectionError:
                pass

        # Should have some successful connections
        assert len(connections) >= 2


class TestDatabaseWriteFailures:
    """Tests for database write failure handling."""

    @pytest.mark.asyncio
    async def test_write_failure_doesnt_corrupt_data(self):
        """Write failure should not corrupt existing data."""
        db = MockFailingDatabase()
        await db.connect()

        # Write initial data
        await db.set("key1", "value1")

        # Simulate write failure
        db.failure_rate = 1.0
        try:
            await db.set("key2", "value2")
        except sqlite3.OperationalError:
            pass

        # Original data should be intact
        db.failure_rate = 0.0
        assert await db.get("key1") == "value1"

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_failure(self):
        """Failed transaction should rollback cleanly."""
        # Simulate transaction with rollback
        committed = False
        rolled_back = False

        class Transaction:
            def __init__(self):
                self.operations = []

            async def add_operation(self, op: str):
                self.operations.append(op)
                if op == "fail":
                    raise sqlite3.OperationalError("Transaction failed")

            async def commit(self):
                nonlocal committed
                committed = True

            async def rollback(self):
                nonlocal rolled_back
                self.operations.clear()
                rolled_back = True

        tx = Transaction()
        try:
            await tx.add_operation("insert1")
            await tx.add_operation("fail")
            await tx.commit()
        except sqlite3.OperationalError:
            await tx.rollback()

        assert not committed
        assert rolled_back
        assert len(tx.operations) == 0


class TestCacheFailures:
    """Tests for cache failure handling."""

    @pytest.mark.asyncio
    async def test_cache_miss_fallback_to_database(self):
        """Cache miss should fallback to database."""
        cache_data: dict[str, Any] = {}
        db_data = {"key1": "db_value1"}

        async def get_with_fallback(key: str) -> Any:
            # Try cache first
            if key in cache_data:
                return cache_data[key]
            # Fallback to database
            if key in db_data:
                value = db_data[key]
                cache_data[key] = value  # Populate cache
                return value
            return None

        result = await get_with_fallback("key1")
        assert result == "db_value1"
        assert "key1" in cache_data

    @pytest.mark.asyncio
    async def test_cache_unavailable_continues_operation(self):
        """System should continue when cache is unavailable."""
        cache_available = False
        db_data = {"key1": "value1"}

        async def get_with_cache_check(key: str) -> Any:
            if cache_available:
                pass  # Would check cache
            return db_data.get(key)

        result = await get_with_cache_check("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_cache_corruption_detected(self):
        """Cache corruption should be detected and handled."""
        # Use invalid UTF-8 bytes (0xff is never valid in UTF-8)
        cache_data = {"key1": b"\xff\xfe\x80\x81"}  # Invalid UTF-8 sequence

        def validate_cache_entry(value: Any) -> bool:
            if isinstance(value, bytes):
                try:
                    value.decode("utf-8")
                    return True
                except UnicodeDecodeError:
                    return False
            return isinstance(value, str)

        assert not validate_cache_entry(cache_data["key1"])


class TestMemoryStoreFailures:
    """Tests for in-memory store failure handling."""

    @pytest.mark.asyncio
    async def test_memory_limit_exceeded(self):
        """Should handle memory limit gracefully."""
        max_size = 100
        store: dict[str, str] = {}

        def add_with_limit(key: str, value: str) -> bool:
            if len(store) >= max_size:
                # Evict oldest entry (simple LRU simulation)
                oldest = next(iter(store))
                del store[oldest]
            store[key] = value
            return True

        # Fill the store
        for i in range(150):
            add_with_limit(f"key{i}", f"value{i}")

        assert len(store) == max_size

    @pytest.mark.asyncio
    async def test_concurrent_memory_access(self):
        """Concurrent access should not corrupt memory store."""
        store: dict[str, int] = {"counter": 0}
        lock = asyncio.Lock()

        async def increment():
            async with lock:
                current = store["counter"]
                await asyncio.sleep(0.001)  # Simulate work
                store["counter"] = current + 1

        # Run concurrent increments
        tasks = [increment() for _ in range(100)]
        await asyncio.gather(*tasks)

        assert store["counter"] == 100


class TestDiskIOFailures:
    """Tests for disk I/O failure handling."""

    @pytest.mark.asyncio
    async def test_disk_full_handling(self):
        """Should handle disk full errors gracefully."""

        async def write_with_disk_check(path: Path, data: str) -> bool:
            try:
                # Simulate disk full error
                if len(data) > 1000:
                    raise OSError(28, "No space left on device")
                path.write_text(data)
                return True
            except OSError as e:
                if e.errno == 28:  # ENOSPC
                    return False
                raise

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"

            # Small write should succeed
            result = await write_with_disk_check(path, "small data")
            assert result is True

            # Large write should fail gracefully
            result = await write_with_disk_check(path, "x" * 2000)
            assert result is False

    @pytest.mark.asyncio
    async def test_file_locked_retry(self):
        """Should retry when file is locked."""
        lock_count = [0]
        max_locks = 2

        async def read_with_lock_retry(path: Path) -> str:
            for attempt in range(5):
                try:
                    if lock_count[0] < max_locks:
                        lock_count[0] += 1
                        raise PermissionError("File is locked")
                    return "file contents"
                except PermissionError:
                    await asyncio.sleep(0.01)
            raise PermissionError("Could not acquire lock")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            result = await read_with_lock_retry(path)
            assert result == "file contents"
            assert lock_count[0] == max_locks


class TestDataCorruption:
    """Tests for data corruption detection and recovery."""

    @pytest.mark.asyncio
    async def test_checksum_validation(self):
        """Should detect data corruption via checksum."""
        import hashlib

        def compute_checksum(data: str) -> str:
            return hashlib.sha256(data.encode()).hexdigest()

        def validate_data(data: str, expected_checksum: str) -> bool:
            return compute_checksum(data) == expected_checksum

        original = "important data"
        checksum = compute_checksum(original)

        # Valid data
        assert validate_data(original, checksum) is True

        # Corrupted data
        corrupted = "important datb"  # Single bit flip
        assert validate_data(corrupted, checksum) is False

    @pytest.mark.asyncio
    async def test_recovery_from_backup(self):
        """Should recover data from backup on corruption."""
        primary_data = {"corrupted": True, "data": None}
        backup_data = {"corrupted": False, "data": "valid data"}

        async def get_data_with_recovery() -> str:
            if primary_data.get("corrupted"):
                # Recover from backup
                if not backup_data.get("corrupted"):
                    return backup_data["data"]
                raise RuntimeError("No valid data available")
            return primary_data["data"]

        result = await get_data_with_recovery()
        assert result == "valid data"


class TestConcurrentWriteConflicts:
    """Tests for concurrent write conflict handling."""

    @pytest.mark.asyncio
    async def test_optimistic_locking_conflict(self):
        """Should detect optimistic locking conflicts."""
        version = [1]
        data = ["initial"]

        async def update_with_version(new_data: str, expected_version: int) -> bool:
            await asyncio.sleep(0.01)  # Simulate latency
            if version[0] != expected_version:
                return False  # Conflict detected
            data[0] = new_data
            version[0] += 1
            return True

        # Simulate concurrent updates
        v = version[0]

        async def updater1():
            return await update_with_version("update1", v)

        async def updater2():
            await asyncio.sleep(0.005)  # Slight delay
            return await update_with_version("update2", v)

        results = await asyncio.gather(updater1(), updater2())

        # One should succeed, one should fail
        assert results.count(True) == 1
        assert results.count(False) == 1

    @pytest.mark.asyncio
    async def test_write_ahead_log_recovery(self):
        """Should recover from WAL on crash."""
        wal: list[dict[str, Any]] = []
        data: dict[str, str] = {}

        async def write_with_wal(key: str, value: str):
            # Write to WAL first
            wal.append({"type": "write", "key": key, "value": value})
            # Then to data store
            data[key] = value

        async def recover_from_wal():
            for entry in wal:
                if entry["type"] == "write":
                    data[entry["key"]] = entry["value"]

        # Write some data
        await write_with_wal("key1", "value1")
        await write_with_wal("key2", "value2")

        # Simulate crash by clearing data
        data.clear()

        # Recover from WAL
        await recover_from_wal()

        assert data["key1"] == "value1"
        assert data["key2"] == "value2"


class TestStorageResilience:
    """Integration tests for storage resilience."""

    @pytest.mark.asyncio
    async def test_multi_tier_storage_failover(self):
        """Should failover between storage tiers."""
        tier1_available = False
        tier2_available = True
        tier3_available = True

        tier1_data: dict[str, str] = {}
        tier2_data = {"key1": "tier2_value"}
        tier3_data = {"key1": "tier3_value"}

        async def get_from_tiers(key: str) -> str | None:
            if tier1_available and key in tier1_data:
                return tier1_data[key]
            if tier2_available and key in tier2_data:
                return tier2_data[key]
            if tier3_available and key in tier3_data:
                return tier3_data[key]
            return None

        result = await get_from_tiers("key1")
        assert result == "tier2_value"  # Tier1 unavailable, falls through to tier2

    @pytest.mark.asyncio
    async def test_storage_health_check(self):
        """Should detect unhealthy storage."""

        async def check_storage_health(storage: MockFailingDatabase) -> dict[str, Any]:
            health = {"healthy": True, "latency_ms": 0, "errors": []}

            try:
                import time

                start = time.time()
                await storage.execute("SELECT 1")
                health["latency_ms"] = (time.time() - start) * 1000
            except Exception as e:
                health["healthy"] = False
                health["errors"].append(str(e))

            return health

        # Healthy storage
        healthy_db = MockFailingDatabase(latency=0.01)
        await healthy_db.connect()
        health = await check_storage_health(healthy_db)
        assert health["healthy"] is True

        # Unhealthy storage
        unhealthy_db = MockFailingDatabase(failure_rate=1.0)
        await unhealthy_db.connect()
        health = await check_storage_health(unhealthy_db)
        assert health["healthy"] is False
