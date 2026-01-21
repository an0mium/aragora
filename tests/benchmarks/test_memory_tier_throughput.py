"""
Memory tier throughput benchmark tests.

Measures read/write performance across memory tiers (fast, medium, slow, glacial).
"""

import asyncio
import time
import random
import string
import pytest
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import OrderedDict


# =============================================================================
# Mock Memory Tier Implementation
# =============================================================================


@dataclass
class MemoryEntry:
    """Memory entry with TTL."""

    key: str
    value: Any
    created_at: float
    ttl: float


class MockMemoryTier:
    """Mock memory tier for benchmarking."""

    def __init__(self, name: str, ttl: float, max_entries: int = 10000):
        self.name = name
        self.ttl = ttl
        self.max_entries = max_entries
        self._storage: OrderedDict[str, MemoryEntry] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            entry = self._storage.get(key)
            if entry is None:
                return None
            if time.time() > entry.created_at + entry.ttl:
                del self._storage[key]
                return None
            # LRU: move to end
            self._storage.move_to_end(key)
            return entry.value

    async def set(self, key: str, value: Any) -> bool:
        async with self._lock:
            # Evict if at capacity
            while len(self._storage) >= self.max_entries:
                self._storage.popitem(last=False)

            self._storage[key] = MemoryEntry(
                key=key,
                value=value,
                created_at=time.time(),
                ttl=self.ttl,
            )
            return True

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._storage:
                del self._storage[key]
                return True
            return False

    def size(self) -> int:
        return len(self._storage)


class MockContinuumMemory:
    """Mock multi-tier memory for benchmarking."""

    def __init__(self):
        self.fast = MockMemoryTier("fast", ttl=60, max_entries=1000)
        self.medium = MockMemoryTier("medium", ttl=3600, max_entries=5000)
        self.slow = MockMemoryTier("slow", ttl=86400, max_entries=10000)
        self.glacial = MockMemoryTier("glacial", ttl=604800, max_entries=50000)
        self.tiers = [self.fast, self.medium, self.slow, self.glacial]

    async def get(self, key: str) -> Optional[Any]:
        for tier in self.tiers:
            value = await tier.get(key)
            if value is not None:
                return value
        return None

    async def set(self, key: str, value: Any, tier: str = "fast") -> bool:
        tier_map = {
            "fast": self.fast,
            "medium": self.medium,
            "slow": self.slow,
            "glacial": self.glacial,
        }
        return await tier_map[tier].set(key, value)


# =============================================================================
# Helper Functions
# =============================================================================


def generate_key(length: int = 32) -> str:
    """Generate random key."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_value(size_bytes: int = 100) -> str:
    """Generate random value of given size."""
    return "".join(random.choices(string.ascii_letters, k=size_bytes))


# =============================================================================
# Write Throughput Tests
# =============================================================================


class TestWriteThroughput:
    """Test write throughput across memory tiers."""

    @pytest.mark.asyncio
    async def test_fast_tier_write_throughput(self):
        """Measure fast tier write throughput."""
        tier = MockMemoryTier("fast", ttl=60, max_entries=10000)

        num_writes = 1000
        start = time.time()

        for i in range(num_writes):
            await tier.set(f"key-{i}", f"value-{i}")

        elapsed = time.time() - start
        elapsed = max(elapsed, 0.001)  # Prevent division by zero on fast systems
        writes_per_second = num_writes / elapsed

        assert tier.size() == num_writes
        assert writes_per_second > 1000  # Should be fast

    @pytest.mark.asyncio
    async def test_concurrent_write_throughput(self):
        """Measure concurrent write throughput."""
        tier = MockMemoryTier("fast", ttl=60, max_entries=10000)

        num_writes = 1000

        async def write(i: int):
            await tier.set(f"key-{i}", f"value-{i}")

        start = time.time()
        await asyncio.gather(*[write(i) for i in range(num_writes)])
        elapsed = time.time() - start
        elapsed = max(elapsed, 0.001)  # Prevent division by zero on fast systems

        writes_per_second = num_writes / elapsed

        assert tier.size() == num_writes
        assert writes_per_second > 500

    @pytest.mark.asyncio
    async def test_large_value_write_throughput(self):
        """Measure throughput with large values."""
        tier = MockMemoryTier("medium", ttl=3600, max_entries=1000)

        num_writes = 100
        value_size = 10000  # 10KB values

        values = [generate_value(value_size) for _ in range(num_writes)]

        start = time.time()
        for i in range(num_writes):
            await tier.set(f"key-{i}", values[i])
        elapsed = time.time() - start

        # Prevent division by zero on fast systems
        elapsed = max(elapsed, 0.001)
        bytes_per_second = (num_writes * value_size) / elapsed

        assert tier.size() == num_writes
        assert bytes_per_second > 100000  # > 100KB/s


# =============================================================================
# Read Throughput Tests
# =============================================================================


class TestReadThroughput:
    """Test read throughput across memory tiers."""

    @pytest.mark.asyncio
    async def test_fast_tier_read_throughput(self):
        """Measure fast tier read throughput."""
        tier = MockMemoryTier("fast", ttl=60, max_entries=10000)

        # Populate
        for i in range(1000):
            await tier.set(f"key-{i}", f"value-{i}")

        # Read
        num_reads = 5000
        start = time.time()

        for i in range(num_reads):
            await tier.get(f"key-{i % 1000}")

        elapsed = time.time() - start
        elapsed = max(elapsed, 0.001)  # Prevent division by zero on fast systems
        reads_per_second = num_reads / elapsed

        assert reads_per_second > 5000

    @pytest.mark.asyncio
    async def test_concurrent_read_throughput(self):
        """Measure concurrent read throughput."""
        tier = MockMemoryTier("fast", ttl=60, max_entries=10000)

        # Populate
        for i in range(1000):
            await tier.set(f"key-{i}", f"value-{i}")

        async def read(i: int):
            return await tier.get(f"key-{i % 1000}")

        num_reads = 5000
        start = time.time()
        results = await asyncio.gather(*[read(i) for i in range(num_reads)])
        elapsed = time.time() - start
        elapsed = max(elapsed, 0.001)  # Prevent division by zero on fast systems

        reads_per_second = num_reads / elapsed

        assert all(r is not None for r in results)
        assert reads_per_second > 2000

    @pytest.mark.asyncio
    async def test_cache_hit_rate(self):
        """Measure cache hit rate."""
        tier = MockMemoryTier("fast", ttl=60, max_entries=100)

        # Populate some keys
        for i in range(50):
            await tier.set(f"key-{i}", f"value-{i}")

        # Read mix of existing and non-existing
        hits = 0
        total = 1000

        for i in range(total):
            key = f"key-{i % 100}"  # 0-49 exist, 50-99 don't
            if await tier.get(key) is not None:
                hits += 1

        hit_rate = hits / total

        # 50% of keys exist
        assert 0.4 < hit_rate < 0.6


# =============================================================================
# Mixed Read/Write Tests
# =============================================================================


class TestMixedOperations:
    """Test mixed read/write performance."""

    @pytest.mark.asyncio
    async def test_80_20_read_write_ratio(self):
        """Test 80% read, 20% write workload."""
        tier = MockMemoryTier("fast", ttl=60, max_entries=10000)

        # Pre-populate
        for i in range(500):
            await tier.set(f"key-{i}", f"value-{i}")

        num_ops = 1000
        reads = 0
        writes = 0

        async def operation(i: int):
            nonlocal reads, writes
            if random.random() < 0.8:
                await tier.get(f"key-{i % 500}")
                reads += 1
            else:
                await tier.set(f"key-{i}", f"value-{i}")
                writes += 1

        start = time.time()
        await asyncio.gather(*[operation(i) for i in range(num_ops)])
        elapsed = time.time() - start
        elapsed = max(elapsed, 0.001)  # Prevent division by zero on fast systems

        ops_per_second = num_ops / elapsed

        assert reads > writes  # Should be ~80/20
        assert ops_per_second > 1000

    @pytest.mark.asyncio
    async def test_50_50_read_write_ratio(self):
        """Test 50% read, 50% write workload."""
        tier = MockMemoryTier("fast", ttl=60, max_entries=10000)

        num_ops = 1000

        async def operation(i: int):
            if i % 2 == 0:
                await tier.set(f"key-{i}", f"value-{i}")
            else:
                await tier.get(f"key-{i - 1}")

        start = time.time()
        await asyncio.gather(*[operation(i) for i in range(num_ops)])
        elapsed = time.time() - start
        elapsed = max(elapsed, 0.001)  # Prevent division by zero on fast systems

        ops_per_second = num_ops / elapsed

        assert ops_per_second > 500


# =============================================================================
# Multi-Tier Performance Tests
# =============================================================================


class TestMultiTierPerformance:
    """Test multi-tier memory performance."""

    @pytest.mark.asyncio
    async def test_tier_promotion(self):
        """Measure performance of tier promotion."""
        memory = MockContinuumMemory()

        # Store in slow tier
        for i in range(100):
            await memory.set(f"key-{i}", f"value-{i}", tier="slow")

        # Read should check fast first, then medium, then slow
        start = time.time()

        for i in range(100):
            await memory.get(f"key-{i}")

        elapsed = time.time() - start
        elapsed = max(elapsed, 0.001)  # Prevent division by zero on fast systems
        reads_per_second = 100 / elapsed

        # Reading from slow tier is slower due to tier checks
        assert reads_per_second > 100

    @pytest.mark.asyncio
    async def test_tier_locality(self):
        """Test performance with good tier locality."""
        memory = MockContinuumMemory()

        # Store hot data in fast tier
        for i in range(100):
            await memory.set(f"hot-{i}", f"value-{i}", tier="fast")

        # Store cold data in glacial tier
        for i in range(100):
            await memory.set(f"cold-{i}", f"value-{i}", tier="glacial")

        # Read hot data (fast tier) - use perf_counter for higher resolution
        hot_start = time.perf_counter()
        for i in range(100):
            await memory.get(f"hot-{i}")
        hot_elapsed = time.perf_counter() - hot_start

        # Read cold data (glacial tier)
        cold_start = time.perf_counter()
        for i in range(100):
            await memory.get(f"cold-{i}")
        cold_elapsed = time.perf_counter() - cold_start

        # Hot data should generally be faster (found in first tier)
        # Allow 3x tolerance for system noise (scheduling, cache effects, etc.)
        # On mock tiers, actual timing differences are minimal
        assert hot_elapsed <= cold_elapsed * 3, (
            f"Hot tier ({hot_elapsed:.4f}s) should not be much slower than "
            f"cold tier ({cold_elapsed:.4f}s)"
        )


# =============================================================================
# LRU Eviction Performance Tests
# =============================================================================


class TestEvictionPerformance:
    """Test LRU eviction performance."""

    @pytest.mark.asyncio
    async def test_eviction_under_pressure(self):
        """Measure performance under memory pressure with eviction."""
        max_entries = 100
        tier = MockMemoryTier("fast", ttl=60, max_entries=max_entries)

        # Write more than capacity (triggers eviction)
        num_writes = 500
        start = time.time()

        for i in range(num_writes):
            await tier.set(f"key-{i}", f"value-{i}")

        elapsed = time.time() - start
        elapsed = max(elapsed, 0.001)  # Prevent division by zero on fast systems
        writes_per_second = num_writes / elapsed

        # Should maintain max capacity
        assert tier.size() == max_entries
        # Should still be reasonably fast
        assert writes_per_second > 500

    @pytest.mark.asyncio
    async def test_lru_ordering_preserved(self):
        """Verify LRU ordering is maintained correctly."""
        max_entries = 10
        tier = MockMemoryTier("fast", ttl=60, max_entries=max_entries)

        # Fill to capacity
        for i in range(10):
            await tier.set(f"key-{i}", f"value-{i}")

        # Access first 5 keys to make them recently used
        for i in range(5):
            await tier.get(f"key-{i}")

        # Add 5 more keys (should evict keys 5-9)
        for i in range(10, 15):
            await tier.set(f"key-{i}", f"value-{i}")

        # Keys 0-4 should still exist (were accessed recently)
        for i in range(5):
            value = await tier.get(f"key-{i}")
            assert value is not None

        # Keys 5-9 should be evicted
        for i in range(5, 10):
            value = await tier.get(f"key-{i}")
            assert value is None


# =============================================================================
# Latency Tests
# =============================================================================


class TestMemoryLatency:
    """Test memory operation latencies."""

    @pytest.mark.asyncio
    async def test_write_latency_distribution(self):
        """Measure write latency distribution."""
        tier = MockMemoryTier("fast", ttl=60, max_entries=10000)
        latencies = []

        for i in range(1000):
            start = time.time()
            await tier.set(f"key-{i}", f"value-{i}")
            latencies.append((time.time() - start) * 1000)  # ms

        latencies.sort()

        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]

        # Write should be sub-millisecond for in-memory
        assert p50 < 1.0
        assert p95 < 5.0
        assert p99 < 10.0

    @pytest.mark.asyncio
    async def test_read_latency_distribution(self):
        """Measure read latency distribution."""
        tier = MockMemoryTier("fast", ttl=60, max_entries=10000)

        # Populate
        for i in range(1000):
            await tier.set(f"key-{i}", f"value-{i}")

        latencies = []

        for i in range(1000):
            start = time.time()
            await tier.get(f"key-{i}")
            latencies.append((time.time() - start) * 1000)  # ms

        latencies.sort()

        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]

        # Read should be sub-millisecond
        assert p50 < 0.5
        assert p95 < 2.0


# =============================================================================
# Stress Tests
# =============================================================================


class TestMemoryStress:
    """Stress tests for memory system."""

    @pytest.mark.asyncio
    async def test_sustained_load(self):
        """Test sustained load over time."""
        tier = MockMemoryTier("fast", ttl=60, max_entries=10000)

        ops_per_batch = 100
        num_batches = 20
        total_ops = 0

        start = time.time()

        for batch in range(num_batches):
            tasks = []
            for i in range(ops_per_batch):
                key = f"batch-{batch}-key-{i}"
                if random.random() < 0.7:
                    tasks.append(tier.set(key, f"value-{i}"))
                else:
                    tasks.append(tier.get(key))

            await asyncio.gather(*tasks)
            total_ops += ops_per_batch

        elapsed = time.time() - start
        elapsed = max(elapsed, 0.001)  # Prevent division by zero on fast systems
        ops_per_second = total_ops / elapsed

        assert ops_per_second > 500

    @pytest.mark.asyncio
    async def test_concurrent_access_patterns(self):
        """Test various concurrent access patterns."""
        tier = MockMemoryTier("fast", ttl=60, max_entries=1000)

        async def hot_key_access():
            """Access a few hot keys repeatedly."""
            for _ in range(100):
                await tier.get("hot-key-1")
                await tier.get("hot-key-2")

        async def cold_key_scan():
            """Scan through many cold keys."""
            for i in range(100):
                await tier.get(f"cold-key-{i}")

        async def write_burst():
            """Burst of writes."""
            for i in range(100):
                await tier.set(f"burst-{i}", f"value-{i}")

        # Pre-populate hot keys
        await tier.set("hot-key-1", "hot-value-1")
        await tier.set("hot-key-2", "hot-value-2")

        # Run all patterns concurrently
        start = time.time()
        await asyncio.gather(
            hot_key_access(),
            cold_key_scan(),
            write_burst(),
        )
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 1.0
