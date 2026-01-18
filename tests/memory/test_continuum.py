"""
Tests for Continuum Memory System.
Tests the multi-tier memory system including:
- Fast/Medium/Slow/Glacial tier management
- TTL-based expiration
- Memory retrieval and storage
- Tier promotion and demotion
"""
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


# =============================================================================
# Mock Memory Classes (mirrors actual implementation)
# =============================================================================

class MemoryTier(str, Enum):
    """Memory tier levels."""
    FAST = "fast"      # 1 minute TTL
    MEDIUM = "medium"  # 1 hour TTL
    SLOW = "slow"      # 1 day TTL
    GLACIAL = "glacial"  # 1 week TTL


@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str
    content: str
    tier: MemoryTier
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def is_expired(self, ttls: Dict[MemoryTier, timedelta]) -> bool:
        """Check if entry is expired based on tier TTL."""
        ttl = ttls.get(self.tier, timedelta(hours=1))
        return datetime.now(timezone.utc) - self.created_at > ttl


class MockContinuumMemory:
    """Mock Continuum memory system for testing."""

    DEFAULT_TTLS = {
        MemoryTier.FAST: timedelta(minutes=1),
        MemoryTier.MEDIUM: timedelta(hours=1),
        MemoryTier.SLOW: timedelta(days=1),
        MemoryTier.GLACIAL: timedelta(weeks=1),
    }

    PROMOTION_THRESHOLD = 5  # accesses to promote
    DEMOTION_THRESHOLD = timedelta(hours=12)  # time without access to demote

    def __init__(
        self,
        ttls: Dict[MemoryTier, timedelta] = None,
        max_entries_per_tier: int = 1000,
    ):
        self.ttls = ttls or self.DEFAULT_TTLS.copy()
        self.max_entries_per_tier = max_entries_per_tier

        self._entries: Dict[str, MemoryEntry] = {}
        self._tier_indices: Dict[MemoryTier, set] = {
            tier: set() for tier in MemoryTier
        }

    @property
    def total_entries(self) -> int:
        return len(self._entries)

    def get_tier_count(self, tier: MemoryTier) -> int:
        return len(self._tier_indices[tier])

    async def store(
        self,
        id: str,
        content: str,
        tier: MemoryTier = MemoryTier.FAST,
        metadata: Dict[str, Any] = None,
        embedding: List[float] = None,
    ) -> MemoryEntry:
        """Store a memory entry."""
        now = datetime.now(timezone.utc)

        entry = MemoryEntry(
            id=id,
            content=content,
            tier=tier,
            created_at=now,
            accessed_at=now,
            metadata=metadata or {},
            embedding=embedding,
        )

        # Remove from old tier if exists
        if id in self._entries:
            old_entry = self._entries[id]
            self._tier_indices[old_entry.tier].discard(id)

        self._entries[id] = entry
        self._tier_indices[tier].add(id)

        # Evict if over capacity
        await self._evict_if_needed(tier)

        return entry

    async def retrieve(self, id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID."""
        entry = self._entries.get(id)

        if entry is None:
            return None

        # Check expiration
        if entry.is_expired(self.ttls):
            await self.delete(id)
            return None

        # Update access tracking
        entry.accessed_at = datetime.now(timezone.utc)
        entry.access_count += 1

        # Check for promotion
        await self._check_promotion(entry)

        return entry

    async def search(
        self,
        query: str,
        limit: int = 10,
        tiers: List[MemoryTier] = None,
    ) -> List[MemoryEntry]:
        """Search memory entries by content."""
        tiers = tiers or list(MemoryTier)
        results = []

        for id, entry in self._entries.items():
            if entry.tier not in tiers:
                continue

            if entry.is_expired(self.ttls):
                continue

            # Simple substring search
            if query.lower() in entry.content.lower():
                results.append(entry)

            if len(results) >= limit:
                break

        return results

    async def delete(self, id: str) -> bool:
        """Delete a memory entry."""
        entry = self._entries.pop(id, None)

        if entry:
            self._tier_indices[entry.tier].discard(id)
            return True

        return False

    async def promote(self, id: str) -> Optional[MemoryEntry]:
        """Promote entry to next tier."""
        entry = self._entries.get(id)

        if entry is None:
            return None

        tier_order = [MemoryTier.FAST, MemoryTier.MEDIUM, MemoryTier.SLOW, MemoryTier.GLACIAL]
        current_idx = tier_order.index(entry.tier)

        if current_idx >= len(tier_order) - 1:
            return entry  # Already at highest tier

        # Move to next tier
        self._tier_indices[entry.tier].discard(id)
        entry.tier = tier_order[current_idx + 1]
        self._tier_indices[entry.tier].add(id)

        return entry

    async def demote(self, id: str) -> Optional[MemoryEntry]:
        """Demote entry to previous tier."""
        entry = self._entries.get(id)

        if entry is None:
            return None

        tier_order = [MemoryTier.FAST, MemoryTier.MEDIUM, MemoryTier.SLOW, MemoryTier.GLACIAL]
        current_idx = tier_order.index(entry.tier)

        if current_idx <= 0:
            return entry  # Already at lowest tier

        # Move to previous tier
        self._tier_indices[entry.tier].discard(id)
        entry.tier = tier_order[current_idx - 1]
        self._tier_indices[entry.tier].add(id)

        return entry

    async def _check_promotion(self, entry: MemoryEntry) -> None:
        """Check if entry should be promoted based on access count."""
        if entry.access_count >= self.PROMOTION_THRESHOLD:
            await self.promote(entry.id)
            entry.access_count = 0  # Reset counter after promotion

    async def _evict_if_needed(self, tier: MemoryTier) -> None:
        """Evict oldest entries if tier is over capacity."""
        tier_ids = self._tier_indices[tier]

        while len(tier_ids) > self.max_entries_per_tier:
            # Find oldest entry
            oldest_id = None
            oldest_time = None

            for id in tier_ids:
                entry = self._entries[id]
                if oldest_time is None or entry.accessed_at < oldest_time:
                    oldest_id = id
                    oldest_time = entry.accessed_at

            if oldest_id:
                await self.delete(oldest_id)

    async def cleanup_expired(self) -> int:
        """Remove all expired entries."""
        expired_ids = []

        for id, entry in self._entries.items():
            if entry.is_expired(self.ttls):
                expired_ids.append(id)

        for id in expired_ids:
            await self.delete(id)

        return len(expired_ids)

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_entries": self.total_entries,
            "tiers": {
                tier.value: self.get_tier_count(tier)
                for tier in MemoryTier
            },
            "ttls": {
                tier.value: ttl.total_seconds()
                for tier, ttl in self.ttls.items()
            },
        }


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def memory():
    """Create memory system for testing."""
    return MockContinuumMemory(max_entries_per_tier=100)


@pytest.fixture
def populated_memory(memory):
    """Memory with some entries."""
    import asyncio

    async def populate():
        await memory.store("id1", "First entry about Python", MemoryTier.FAST)
        await memory.store("id2", "Second entry about JavaScript", MemoryTier.MEDIUM)
        await memory.store("id3", "Third entry about databases", MemoryTier.SLOW)
        return memory

    return asyncio.get_event_loop().run_until_complete(populate())


# =============================================================================
# Test Classes
# =============================================================================

class TestMemoryInit:
    """Test memory system initialization."""

    def test_default_ttls(self, memory):
        """Test default TTL values."""
        assert memory.ttls[MemoryTier.FAST] == timedelta(minutes=1)
        assert memory.ttls[MemoryTier.MEDIUM] == timedelta(hours=1)
        assert memory.ttls[MemoryTier.SLOW] == timedelta(days=1)
        assert memory.ttls[MemoryTier.GLACIAL] == timedelta(weeks=1)

    def test_custom_ttls(self):
        """Test custom TTL values."""
        custom_ttls = {
            MemoryTier.FAST: timedelta(seconds=30),
            MemoryTier.MEDIUM: timedelta(minutes=30),
            MemoryTier.SLOW: timedelta(hours=12),
            MemoryTier.GLACIAL: timedelta(days=3),
        }
        memory = MockContinuumMemory(ttls=custom_ttls)

        assert memory.ttls[MemoryTier.FAST] == timedelta(seconds=30)

    def test_initial_state(self, memory):
        """Test initial empty state."""
        assert memory.total_entries == 0
        for tier in MemoryTier:
            assert memory.get_tier_count(tier) == 0


class TestStore:
    """Test memory storage."""

    @pytest.mark.asyncio
    async def test_store_basic(self, memory):
        """Test basic storage."""
        entry = await memory.store("test-id", "Test content")

        assert entry.id == "test-id"
        assert entry.content == "Test content"
        assert entry.tier == MemoryTier.FAST
        assert memory.total_entries == 1

    @pytest.mark.asyncio
    async def test_store_with_tier(self, memory):
        """Test storage with specific tier."""
        entry = await memory.store("id", "content", tier=MemoryTier.SLOW)

        assert entry.tier == MemoryTier.SLOW
        assert memory.get_tier_count(MemoryTier.SLOW) == 1

    @pytest.mark.asyncio
    async def test_store_with_metadata(self, memory):
        """Test storage with metadata."""
        metadata = {"source": "debate", "round": 3}
        entry = await memory.store("id", "content", metadata=metadata)

        assert entry.metadata["source"] == "debate"
        assert entry.metadata["round"] == 3

    @pytest.mark.asyncio
    async def test_store_with_embedding(self, memory):
        """Test storage with embedding."""
        embedding = [0.1, 0.2, 0.3, 0.4]
        entry = await memory.store("id", "content", embedding=embedding)

        assert entry.embedding == embedding

    @pytest.mark.asyncio
    async def test_store_overwrites_existing(self, memory):
        """Test that storing with same ID overwrites."""
        await memory.store("id", "original content", tier=MemoryTier.FAST)
        await memory.store("id", "new content", tier=MemoryTier.MEDIUM)

        assert memory.total_entries == 1
        entry = await memory.retrieve("id")
        assert entry.content == "new content"
        assert entry.tier == MemoryTier.MEDIUM


class TestRetrieve:
    """Test memory retrieval."""

    @pytest.mark.asyncio
    async def test_retrieve_existing(self, populated_memory):
        """Test retrieving existing entry."""
        entry = await populated_memory.retrieve("id1")

        assert entry is not None
        assert entry.content == "First entry about Python"

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent(self, memory):
        """Test retrieving non-existent entry."""
        entry = await memory.retrieve("nonexistent")

        assert entry is None

    @pytest.mark.asyncio
    async def test_retrieve_updates_access_time(self, memory):
        """Test that retrieval updates access time."""
        await memory.store("id", "content")
        entry1 = await memory.retrieve("id")
        first_access = entry1.accessed_at

        # Small delay
        import asyncio
        await asyncio.sleep(0.01)

        entry2 = await memory.retrieve("id")

        assert entry2.accessed_at >= first_access

    @pytest.mark.asyncio
    async def test_retrieve_increments_access_count(self, memory):
        """Test that retrieval increments access count."""
        await memory.store("id", "content")

        for i in range(3):
            entry = await memory.retrieve("id")
            assert entry.access_count == i + 1


class TestSearch:
    """Test memory search."""

    @pytest.mark.asyncio
    async def test_search_finds_matching(self, populated_memory):
        """Test search finds matching entries."""
        results = await populated_memory.search("Python")

        assert len(results) == 1
        assert results[0].id == "id1"

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self, populated_memory):
        """Test case-insensitive search."""
        results = await populated_memory.search("python")

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_with_limit(self, memory):
        """Test search respects limit."""
        for i in range(10):
            await memory.store(f"id{i}", f"content {i}")

        results = await memory.search("content", limit=3)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_by_tier(self, populated_memory):
        """Test search filtered by tier."""
        results = await populated_memory.search("entry", tiers=[MemoryTier.FAST])

        assert len(results) == 1
        assert results[0].tier == MemoryTier.FAST

    @pytest.mark.asyncio
    async def test_search_no_matches(self, populated_memory):
        """Test search with no matches."""
        results = await populated_memory.search("nonexistent")

        assert len(results) == 0


class TestDelete:
    """Test memory deletion."""

    @pytest.mark.asyncio
    async def test_delete_existing(self, populated_memory):
        """Test deleting existing entry."""
        result = await populated_memory.delete("id1")

        assert result is True
        assert populated_memory.total_entries == 2

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, memory):
        """Test deleting non-existent entry."""
        result = await memory.delete("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_updates_tier_index(self, memory):
        """Test that delete updates tier index."""
        await memory.store("id", "content", tier=MemoryTier.SLOW)
        assert memory.get_tier_count(MemoryTier.SLOW) == 1

        await memory.delete("id")

        assert memory.get_tier_count(MemoryTier.SLOW) == 0


class TestTierManagement:
    """Test tier promotion and demotion."""

    @pytest.mark.asyncio
    async def test_promote_tier(self, memory):
        """Test promoting entry to next tier."""
        await memory.store("id", "content", tier=MemoryTier.FAST)

        entry = await memory.promote("id")

        assert entry.tier == MemoryTier.MEDIUM
        assert memory.get_tier_count(MemoryTier.FAST) == 0
        assert memory.get_tier_count(MemoryTier.MEDIUM) == 1

    @pytest.mark.asyncio
    async def test_demote_tier(self, memory):
        """Test demoting entry to previous tier."""
        await memory.store("id", "content", tier=MemoryTier.MEDIUM)

        entry = await memory.demote("id")

        assert entry.tier == MemoryTier.FAST

    @pytest.mark.asyncio
    async def test_promote_at_max_tier(self, memory):
        """Test promoting at maximum tier stays same."""
        await memory.store("id", "content", tier=MemoryTier.GLACIAL)

        entry = await memory.promote("id")

        assert entry.tier == MemoryTier.GLACIAL

    @pytest.mark.asyncio
    async def test_demote_at_min_tier(self, memory):
        """Test demoting at minimum tier stays same."""
        await memory.store("id", "content", tier=MemoryTier.FAST)

        entry = await memory.demote("id")

        assert entry.tier == MemoryTier.FAST

    @pytest.mark.asyncio
    async def test_auto_promotion_on_access(self, memory):
        """Test automatic promotion after threshold accesses."""
        memory.PROMOTION_THRESHOLD = 3
        await memory.store("id", "content", tier=MemoryTier.FAST)

        # Access multiple times
        for _ in range(3):
            await memory.retrieve("id")

        entry = await memory.retrieve("id")

        assert entry.tier == MemoryTier.MEDIUM


class TestExpiration:
    """Test TTL-based expiration."""

    @pytest.mark.asyncio
    async def test_entry_not_expired(self, memory):
        """Test entry is not expired immediately."""
        entry = await memory.store("id", "content")

        assert not entry.is_expired(memory.ttls)

    @pytest.mark.asyncio
    async def test_retrieve_expired_returns_none(self, memory):
        """Test retrieving expired entry returns None."""
        # Store with very short TTL
        memory.ttls[MemoryTier.FAST] = timedelta(milliseconds=1)
        await memory.store("id", "content", tier=MemoryTier.FAST)

        # Wait for expiration
        import asyncio
        await asyncio.sleep(0.01)

        entry = await memory.retrieve("id")

        assert entry is None

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, memory):
        """Test cleanup removes expired entries."""
        memory.ttls[MemoryTier.FAST] = timedelta(milliseconds=1)

        await memory.store("id1", "content1", tier=MemoryTier.FAST)
        await memory.store("id2", "content2", tier=MemoryTier.GLACIAL)

        import asyncio
        await asyncio.sleep(0.01)

        removed = await memory.cleanup_expired()

        assert removed == 1
        assert memory.total_entries == 1


class TestEviction:
    """Test capacity-based eviction."""

    @pytest.mark.asyncio
    async def test_evict_when_over_capacity(self):
        """Test eviction when tier is over capacity."""
        memory = MockContinuumMemory(max_entries_per_tier=3)

        # Store more than capacity
        for i in range(5):
            await memory.store(f"id{i}", f"content{i}", tier=MemoryTier.FAST)

        assert memory.get_tier_count(MemoryTier.FAST) == 3

    @pytest.mark.asyncio
    async def test_eviction_removes_oldest(self):
        """Test that eviction removes oldest accessed entries."""
        memory = MockContinuumMemory(max_entries_per_tier=2)

        await memory.store("old", "old content")
        import asyncio
        await asyncio.sleep(0.01)
        await memory.store("new1", "new content 1")
        await memory.store("new2", "new content 2")  # Should evict "old"

        assert await memory.retrieve("old") is None
        assert await memory.retrieve("new1") is not None


class TestStats:
    """Test statistics functionality."""

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, memory):
        """Test stats for empty memory."""
        stats = await memory.get_stats()

        assert stats["total_entries"] == 0
        assert all(count == 0 for count in stats["tiers"].values())

    @pytest.mark.asyncio
    async def test_get_stats_populated(self, populated_memory):
        """Test stats for populated memory."""
        stats = await populated_memory.get_stats()

        assert stats["total_entries"] == 3
        assert stats["tiers"]["fast"] == 1
        assert stats["tiers"]["medium"] == 1
        assert stats["tiers"]["slow"] == 1

    @pytest.mark.asyncio
    async def test_stats_include_ttls(self, memory):
        """Test that stats include TTL information."""
        stats = await memory.get_stats()

        assert "ttls" in stats
        assert stats["ttls"]["fast"] == 60  # 1 minute in seconds


class TestContinuumIntegration:
    """Integration tests for memory workflows."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, memory):
        """Test complete entry lifecycle."""
        # Store
        entry = await memory.store("id", "content", metadata={"test": True})
        assert entry is not None

        # Retrieve
        retrieved = await memory.retrieve("id")
        assert retrieved.content == "content"

        # Update
        await memory.store("id", "updated content")
        retrieved = await memory.retrieve("id")
        assert retrieved.content == "updated content"

        # Delete
        result = await memory.delete("id")
        assert result is True

        # Verify deleted
        retrieved = await memory.retrieve("id")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_tier_progression(self, memory):
        """Test entry progressing through tiers."""
        memory.PROMOTION_THRESHOLD = 3

        await memory.store("id", "content", tier=MemoryTier.FAST)

        # Access to promote from FAST to MEDIUM (need threshold accesses)
        for _ in range(3):
            await memory.retrieve("id")

        entry = await memory.retrieve("id")
        # After threshold accesses, should be in MEDIUM
        assert entry.tier in [MemoryTier.MEDIUM, MemoryTier.SLOW, MemoryTier.GLACIAL]

    @pytest.mark.asyncio
    async def test_concurrent_access(self, memory):
        """Test concurrent memory access."""
        import asyncio

        await memory.store("id", "content")

        async def access():
            for _ in range(10):
                await memory.retrieve("id")

        # Run concurrent accesses
        await asyncio.gather(access(), access(), access())

        entry = await memory.retrieve("id")
        assert entry.access_count > 0
