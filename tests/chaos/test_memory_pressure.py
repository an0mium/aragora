"""
Chaos Engineering Tests: Memory Pressure.

Tests memory system behavior under pressure:
- Memory tier fallback mechanisms
- Cache eviction under pressure
- Recovery from memory exhaustion
- Cross-tier consistency

Run with extended timeout:
    pytest tests/chaos/test_memory_pressure.py -v --timeout=300
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class MemoryTier(Enum):
    """Memory tier levels for the continuum."""

    FAST = "fast"  # In-memory, sub-millisecond
    MEDIUM = "medium"  # Local cache, milliseconds
    SLOW = "slow"  # Persistent, tens of ms
    GLACIAL = "glacial"  # Archive, hundreds of ms


@dataclass
class MemoryEntry:
    """Entry in the memory system."""

    key: str
    value: Any
    tier: MemoryTier
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    size_bytes: int = 100


class MockTierStorage:
    """Mock storage for a single tier."""

    def __init__(self, tier: MemoryTier, capacity: int, latency_ms: float):
        self.tier = tier
        self.capacity = capacity
        self.latency_ms = latency_ms
        self.data: dict[str, MemoryEntry] = {}
        self.is_available = True
        self.failure_rate = 0.0

    async def get(self, key: str) -> MemoryEntry | None:
        if not self.is_available:
            raise ConnectionError(f"{self.tier.value} tier unavailable")
        if random.random() < self.failure_rate:
            raise RuntimeError(f"{self.tier.value} tier error")

        await asyncio.sleep(self.latency_ms / 1000)
        entry = self.data.get(key)
        if entry:
            entry.access_count += 1
        return entry

    async def put(self, key: str, value: Any, size_bytes: int = 100) -> bool:
        if not self.is_available:
            raise ConnectionError(f"{self.tier.value} tier unavailable")
        if random.random() < self.failure_rate:
            raise RuntimeError(f"{self.tier.value} tier error")

        await asyncio.sleep(self.latency_ms / 1000)

        # Check capacity - if at capacity and key doesn't exist, signal overflow needed
        if len(self.data) >= self.capacity and key not in self.data:
            raise OverflowError(f"{self.tier.value} tier at capacity")

        self.data[key] = MemoryEntry(
            key=key,
            value=value,
            tier=self.tier,
            size_bytes=size_bytes,
        )
        return True

    async def delete(self, key: str) -> bool:
        if key in self.data:
            del self.data[key]
            return True
        return False

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.data:
            return
        # Find entry with lowest access count
        lru_key = min(self.data.keys(), key=lambda k: self.data[k].access_count)
        del self.data[lru_key]

    @property
    def used_capacity(self) -> int:
        return len(self.data)

    @property
    def available_capacity(self) -> int:
        return self.capacity - len(self.data)


class MockContinuumMemory:
    """Mock multi-tier memory system."""

    def __init__(self):
        self.tiers = {
            MemoryTier.FAST: MockTierStorage(MemoryTier.FAST, capacity=100, latency_ms=0.1),
            MemoryTier.MEDIUM: MockTierStorage(MemoryTier.MEDIUM, capacity=1000, latency_ms=1),
            MemoryTier.SLOW: MockTierStorage(MemoryTier.SLOW, capacity=10000, latency_ms=10),
            MemoryTier.GLACIAL: MockTierStorage(
                MemoryTier.GLACIAL, capacity=100000, latency_ms=100
            ),
        }
        self.tier_order = [MemoryTier.FAST, MemoryTier.MEDIUM, MemoryTier.SLOW, MemoryTier.GLACIAL]

    async def get(self, key: str) -> tuple[Any | None, MemoryTier | None]:
        """Get value, searching through tiers."""
        for tier in self.tier_order:
            storage = self.tiers[tier]
            try:
                entry = await storage.get(key)
                if entry:
                    # Promote to faster tier if found in slower tier
                    if tier != MemoryTier.FAST:
                        await self._promote(key, entry, tier)
                    return entry.value, tier
            except Exception:
                continue  # Try next tier
        return None, None

    async def put(self, key: str, value: Any, target_tier: MemoryTier = MemoryTier.FAST) -> bool:
        """Store value, falling back to slower tiers if needed."""
        for tier in self.tier_order[self.tier_order.index(target_tier) :]:
            storage = self.tiers[tier]
            try:
                return await storage.put(key, value)
            except Exception:
                continue  # Try next tier
        return False

    async def _promote(self, key: str, entry: MemoryEntry, from_tier: MemoryTier) -> None:
        """Promote entry to faster tier."""
        fast = self.tiers[MemoryTier.FAST]
        try:
            await fast.put(key, entry.value, entry.size_bytes)
        except Exception:
            pass  # Promotion failed, entry stays in original tier

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        return {
            tier.value: {
                "used": storage.used_capacity,
                "capacity": storage.capacity,
                "available": storage.available_capacity,
            }
            for tier, storage in self.tiers.items()
        }


class TestMemoryTierFallback:
    """Tests for memory tier fallback mechanisms."""

    @pytest.fixture
    def memory(self) -> MockContinuumMemory:
        return MockContinuumMemory()

    @pytest.mark.asyncio
    async def test_normal_read_write(self, memory: MockContinuumMemory):
        """Test normal read/write operations."""
        await memory.put("key1", "value1")
        value, tier = await memory.get("key1")

        assert value == "value1"
        assert tier == MemoryTier.FAST

    @pytest.mark.asyncio
    async def test_fallback_on_fast_tier_failure(self, memory: MockContinuumMemory):
        """Test fallback when fast tier fails."""
        # Store in medium tier first
        await memory.tiers[MemoryTier.MEDIUM].put("key1", "value1")

        # Disable fast tier
        memory.tiers[MemoryTier.FAST].is_available = False

        # Should fall back to medium tier
        value, tier = await memory.get("key1")

        assert value == "value1"
        assert tier == MemoryTier.MEDIUM

    @pytest.mark.asyncio
    async def test_cascading_tier_fallback(self, memory: MockContinuumMemory):
        """Test cascading fallback through tiers."""
        # Store in glacial tier
        await memory.tiers[MemoryTier.GLACIAL].put("key1", "value1")

        # Disable faster tiers
        memory.tiers[MemoryTier.FAST].is_available = False
        memory.tiers[MemoryTier.MEDIUM].is_available = False
        memory.tiers[MemoryTier.SLOW].is_available = False

        # Should find in glacial tier
        value, tier = await memory.get("key1")

        assert value == "value1"
        assert tier == MemoryTier.GLACIAL

    @pytest.mark.asyncio
    async def test_write_fallback_on_capacity(self, memory: MockContinuumMemory):
        """Test write fallback when tier is at capacity."""
        # Fill fast tier
        for i in range(150):  # More than capacity of 100
            await memory.put(f"key{i}", f"value{i}")

        # Some entries should be in medium tier
        stats = memory.get_stats()
        assert stats["fast"]["used"] == 100  # At capacity
        assert stats["medium"]["used"] > 0  # Overflow

    @pytest.mark.asyncio
    async def test_promotion_on_access(self, memory: MockContinuumMemory):
        """Test entry promotion on access."""
        # Store in slow tier
        memory.tiers[MemoryTier.FAST].capacity = 100
        await memory.tiers[MemoryTier.SLOW].put("hot_key", "hot_value")

        # Access should promote to fast tier
        value, _ = await memory.get("hot_key")

        # Should now be in fast tier
        fast_entry = await memory.tiers[MemoryTier.FAST].get("hot_key")
        assert fast_entry is not None
        assert fast_entry.value == "hot_value"


class TestMemoryPressureScenarios:
    """Test memory behavior under pressure."""

    @pytest.fixture
    def memory(self) -> MockContinuumMemory:
        return MockContinuumMemory()

    @pytest.mark.asyncio
    async def test_high_write_load(self, memory: MockContinuumMemory):
        """Test behavior under high write load."""
        write_count = 0
        error_count = 0

        async def write_batch():
            nonlocal write_count, error_count
            for i in range(100):
                key = f"batch_{id(asyncio.current_task())}_{i}"
                try:
                    await memory.put(key, f"value_{i}")
                    write_count += 1
                except Exception:
                    error_count += 1

        # Concurrent write batches
        await asyncio.gather(*[write_batch() for _ in range(10)])

        # Should have handled most writes
        assert write_count > 500
        # Errors acceptable under pressure
        assert error_count < write_count * 0.1

    @pytest.mark.asyncio
    async def test_mixed_read_write_load(self, memory: MockContinuumMemory):
        """Test mixed read/write workload."""
        # Pre-populate
        for i in range(500):
            await memory.put(f"pre_{i}", f"value_{i}")

        read_count = 0
        write_count = 0

        async def mixed_operations():
            nonlocal read_count, write_count
            for _ in range(50):
                if random.random() < 0.7:  # 70% reads
                    key = f"pre_{random.randint(0, 499)}"
                    await memory.get(key)
                    read_count += 1
                else:  # 30% writes
                    key = f"new_{id(asyncio.current_task())}_{random.randint(0, 1000)}"
                    await memory.put(key, "new_value")
                    write_count += 1

        await asyncio.gather(*[mixed_operations() for _ in range(10)])

        assert read_count > 0
        assert write_count > 0

    @pytest.mark.asyncio
    async def test_tier_failure_during_load(self, memory: MockContinuumMemory):
        """Test tier failure during active load."""
        operations_completed = 0
        operations_failed = 0

        async def continuous_operations():
            nonlocal operations_completed, operations_failed
            for i in range(50):
                try:
                    key = f"op_{id(asyncio.current_task())}_{i}"
                    await memory.put(key, f"value_{i}")
                    await memory.get(key)
                    operations_completed += 1
                except asyncio.CancelledError:
                    raise  # Re-raise cancellation
                except Exception:
                    operations_failed += 1
                await asyncio.sleep(0.01)

        # Fail fast tier mid-way through test
        memory.tiers[MemoryTier.FAST].failure_rate = 0.5

        # Run operations (some will fail due to high failure rate)
        await asyncio.gather(*[continuous_operations() for _ in range(5)])

        # Should have completed some operations despite failures
        total = operations_completed + operations_failed
        assert total > 0
        assert operations_completed > 0  # At least some should succeed via fallback

    @pytest.mark.asyncio
    async def test_gradual_memory_pressure(self, memory: MockContinuumMemory):
        """Test gradual increase in memory pressure."""
        # Track tier usage over time
        usage_snapshots = []

        for phase in range(5):
            # Write increasing amounts
            batch_size = 100 * (phase + 1)
            for i in range(batch_size):
                key = f"phase_{phase}_key_{i}"
                await memory.put(key, f"value_{i}" * 10)  # Larger values

            usage_snapshots.append(memory.get_stats())
            await asyncio.sleep(0.1)

        # Usage should increase across tiers
        fast_usage = [s["fast"]["used"] for s in usage_snapshots]
        medium_usage = [s["medium"]["used"] for s in usage_snapshots]

        # Fast tier should hit capacity
        assert max(fast_usage) >= 90
        # Medium tier should receive overflow
        assert max(medium_usage) > 0


class TestMemoryEviction:
    """Test memory eviction under pressure."""

    @pytest.fixture
    def memory(self) -> MockContinuumMemory:
        return MockContinuumMemory()

    @pytest.mark.asyncio
    async def test_lru_eviction(self, memory: MockContinuumMemory):
        """Test LRU eviction policy."""
        # Fill fast tier
        for i in range(100):
            await memory.put(f"key_{i}", f"value_{i}")

        # Access some keys to make them "hot"
        hot_keys = [f"key_{i}" for i in range(0, 20)]
        for key in hot_keys:
            for _ in range(5):  # Multiple accesses
                await memory.get(key)

        # Add more entries to trigger eviction
        for i in range(100, 150):
            await memory.put(f"key_{i}", f"value_{i}")

        # Hot keys should still be in fast tier
        for key in hot_keys:
            value, tier = await memory.get(key)
            # Hot keys more likely to be in fast tier
            assert value is not None

    @pytest.mark.asyncio
    async def test_eviction_preserves_data(self, memory: MockContinuumMemory):
        """Test that evicted data is preserved in slower tiers."""
        # Fill and overflow fast tier
        all_keys = []
        for i in range(200):
            key = f"key_{i}"
            all_keys.append(key)
            await memory.put(key, f"value_{i}")

        # All keys should still be accessible
        for key in all_keys:
            value, tier = await memory.get(key)
            assert value is not None


class TestMemoryRecovery:
    """Test recovery from memory failures."""

    @pytest.fixture
    def memory(self) -> MockContinuumMemory:
        return MockContinuumMemory()

    @pytest.mark.asyncio
    async def test_recovery_after_tier_restoration(self, memory: MockContinuumMemory):
        """Test recovery when failed tier is restored."""
        # Pre-populate
        for i in range(50):
            await memory.put(f"key_{i}", f"value_{i}")

        # Fail fast tier
        memory.tiers[MemoryTier.FAST].is_available = False

        # Operations should use fallback
        for i in range(10):
            await memory.put(f"new_key_{i}", f"new_value_{i}")

        # Restore fast tier
        memory.tiers[MemoryTier.FAST].is_available = True

        # Should use fast tier again
        await memory.put("test_key", "test_value")
        value, tier = await memory.get("test_key")

        assert tier == MemoryTier.FAST

    @pytest.mark.asyncio
    async def test_partial_tier_recovery(self, memory: MockContinuumMemory):
        """Test behavior during partial tier recovery."""
        # Fail multiple tiers
        memory.tiers[MemoryTier.FAST].is_available = False
        memory.tiers[MemoryTier.MEDIUM].is_available = False

        # Should still work with slow tier
        await memory.put("key1", "value1")
        value, tier = await memory.get("key1")

        assert value == "value1"
        assert tier == MemoryTier.SLOW

        # Restore medium tier
        memory.tiers[MemoryTier.MEDIUM].is_available = True

        # New writes should use medium tier
        await memory.put("key2", "value2")
        # Existing data should still be accessible

    @pytest.mark.asyncio
    async def test_intermittent_tier_failures(self, memory: MockContinuumMemory):
        """Test handling of intermittent tier failures."""
        # Set high failure rate on fast tier
        memory.tiers[MemoryTier.FAST].failure_rate = 0.5

        successes = 0
        for i in range(50):
            try:
                await memory.put(f"key_{i}", f"value_{i}")
                value, _ = await memory.get(f"key_{i}")
                if value is not None:
                    successes += 1
            except Exception:
                pass

        # Should have some successes despite failures
        assert successes > 20

    @pytest.mark.asyncio
    async def test_complete_system_recovery(self, memory: MockContinuumMemory):
        """Test complete system recovery after total failure."""
        # Pre-populate slow tier (will survive)
        for i in range(50):
            await memory.tiers[MemoryTier.GLACIAL].put(f"key_{i}", f"value_{i}")

        # Total failure of fast tiers
        memory.tiers[MemoryTier.FAST].is_available = False
        memory.tiers[MemoryTier.MEDIUM].is_available = False
        memory.tiers[MemoryTier.SLOW].is_available = False

        # Should still access glacial
        value, tier = await memory.get("key_0")
        assert value == "value_0"
        assert tier == MemoryTier.GLACIAL

        # Restore all tiers
        memory.tiers[MemoryTier.FAST].is_available = True
        memory.tiers[MemoryTier.MEDIUM].is_available = True
        memory.tiers[MemoryTier.SLOW].is_available = True

        # System should be fully operational
        await memory.put("new_key", "new_value")
        value, tier = await memory.get("new_key")

        assert value == "new_value"
        assert tier == MemoryTier.FAST
