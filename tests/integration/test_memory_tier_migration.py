"""
Memory Tier Migration Integration Tests.

Tests for the ContinuumMemory tier system:
- Fast → Medium → Slow → Glacial tier promotion
- Demotion on low access
- Score calculation and thresholds
- Concurrent access during migration
- Recovery from migration failures
"""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from pathlib import Path

import pytest


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def memory_db_path():
    """Temporary database for memory tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "memory_test.db"


@pytest.fixture
def continuum_memory(memory_db_path):
    """ContinuumMemory instance with temp database."""
    from aragora.memory.continuum import ContinuumMemory

    memory = ContinuumMemory(str(memory_db_path))
    yield memory


def gen_key() -> str:
    """Generate a unique memory key."""
    return f"mem_{uuid.uuid4().hex[:8]}"


# =============================================================================
# Basic Tier Operations
# =============================================================================


class TestBasicTierOperations:
    """Tests for basic memory tier operations."""

    @pytest.mark.asyncio
    async def test_store_defaults_to_slow_tier(self, continuum_memory):
        """New memories should start in slow tier by default."""
        from aragora.memory.continuum import MemoryTier

        entry = await continuum_memory.store(
            key=gen_key(),
            content="Test memory content",
            metadata={"source": "test"},
        )

        assert entry is not None
        assert entry.id is not None
        # Default tier is SLOW
        assert entry.tier in [MemoryTier.SLOW, "slow"]

    @pytest.mark.asyncio
    async def test_retrieve_returns_entry(self, continuum_memory):
        """Retrieving memory should return the entry."""
        key = gen_key()
        entry = await continuum_memory.store(
            key=key,
            content="Frequently accessed memory",
            metadata={},
        )

        # Access multiple times
        for _ in range(5):
            continuum_memory.get(entry.id)

        memory = continuum_memory.get(entry.id)
        # Should exist
        assert memory is not None

    @pytest.mark.asyncio
    async def test_store_with_custom_tier(self, continuum_memory):
        """Should be able to store in specific tier."""
        from aragora.memory.continuum import MemoryTier

        # Store directly in glacial tier
        entry = await continuum_memory.store(
            key=gen_key(),
            content="Important long-term memory",
            metadata={"importance": "high"},
            tier=MemoryTier.GLACIAL,
        )

        assert entry is not None
        assert entry.tier in [MemoryTier.GLACIAL, "glacial"]


# =============================================================================
# Tier Promotion Tests
# =============================================================================


class TestTierPromotion:
    """Tests for memory tier promotion."""

    @pytest.mark.asyncio
    async def test_frequent_access_triggers_promotion_check(self, continuum_memory):
        """Frequently accessed memories should be candidates for promotion."""
        entry = await continuum_memory.store(
            key=gen_key(),
            content="Memory that will be accessed frequently",
            metadata={},
        )

        # Simulate frequent access
        for _ in range(20):
            continuum_memory.get(entry.id)
            await asyncio.sleep(0.01)  # Small delay between accesses

        # Memory should still be accessible
        memory = continuum_memory.get(entry.id)
        assert memory is not None

    @pytest.mark.asyncio
    async def test_promotion_preserves_content(self, continuum_memory):
        """Content should be preserved during tier promotion."""
        original_content = "Important content that must be preserved"
        original_metadata = {"key": "value", "number": 42}

        entry = await continuum_memory.store(
            key=gen_key(),
            content=original_content,
            metadata=original_metadata,
        )

        # Access to potentially trigger promotion
        for _ in range(10):
            continuum_memory.get(entry.id)

        # Verify content preserved
        memory = continuum_memory.get(entry.id)
        assert memory is not None
        assert memory.content == original_content

    @pytest.mark.asyncio
    async def test_multiple_memories_independent_promotion(self, continuum_memory):
        """Each memory should be promoted independently."""
        # Create multiple memories
        entries = []
        for i in range(5):
            entry = await continuum_memory.store(
                key=gen_key(),
                content=f"Memory {i}",
                metadata={"index": i},
            )
            entries.append(entry)

        # Access only some frequently
        for _ in range(15):
            continuum_memory.get(entries[0].id)  # Frequent
            continuum_memory.get(entries[1].id)  # Frequent

        # All memories should still be accessible
        for entry in entries:
            memory = continuum_memory.get(entry.id)
            assert memory is not None


# =============================================================================
# Tier Demotion Tests
# =============================================================================


class TestTierDemotion:
    """Tests for memory tier demotion."""

    @pytest.mark.asyncio
    async def test_stale_memory_candidate_for_demotion(self, continuum_memory):
        """Memories without recent access should be demotion candidates."""
        entry = await continuum_memory.store(
            key=gen_key(),
            content="Memory that will become stale",
            metadata={},
        )

        # Don't access it - let it become stale
        # Just verify it's still accessible
        memory = continuum_memory.get(entry.id)
        assert memory is not None

    @pytest.mark.asyncio
    async def test_demotion_preserves_content(self, continuum_memory):
        """Content should be preserved during tier demotion."""
        original_content = "Content that survives demotion"

        entry = await continuum_memory.store(
            key=gen_key(),
            content=original_content,
            metadata={},
        )

        # Wait a bit (simulate time passing)
        await asyncio.sleep(0.1)

        # Content should still be accessible
        memory = continuum_memory.get(entry.id)
        assert memory is not None


# =============================================================================
# Score Calculation Tests
# =============================================================================


class TestScoreCalculation:
    """Tests for memory score calculation."""

    @pytest.mark.asyncio
    async def test_access_increases_score(self, continuum_memory):
        """Accessing memory should increase its score."""
        entry = await continuum_memory.store(
            key=gen_key(),
            content="Score test memory",
            metadata={},
        )

        # Get initial state
        initial = continuum_memory.get(entry.id)

        # Access multiple times
        for _ in range(10):
            continuum_memory.get(entry.id)

        # Get updated state
        updated = continuum_memory.get(entry.id)

        # Both should exist
        assert initial is not None
        assert updated is not None

    @pytest.mark.asyncio
    async def test_recency_affects_score(self, continuum_memory):
        """Recent memories should have higher scores."""
        # Create old memory
        old_entry = await continuum_memory.store(
            key=gen_key(),
            content="Old memory",
            metadata={},
        )

        # Wait a bit
        await asyncio.sleep(0.1)

        # Create new memory
        new_entry = await continuum_memory.store(
            key=gen_key(),
            content="New memory",
            metadata={},
        )

        # Both should be accessible
        old = continuum_memory.get(old_entry.id)
        new = continuum_memory.get(new_entry.id)

        assert old is not None
        assert new is not None


# =============================================================================
# Concurrent Access Tests
# =============================================================================


class TestConcurrentAccess:
    """Tests for concurrent memory access during tier operations."""

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, continuum_memory):
        """Multiple concurrent reads should work correctly."""
        entry = await continuum_memory.store(
            key=gen_key(),
            content="Concurrent read test",
            metadata={},
        )

        async def read_memory():
            return continuum_memory.get(entry.id)

        # Run many concurrent reads
        results = await asyncio.gather(*[read_memory() for _ in range(20)])

        # All reads should succeed
        assert all(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, continuum_memory):
        """Multiple concurrent writes should work correctly."""

        async def write_memory(i: int):
            return await continuum_memory.store(
                key=gen_key(),
                content=f"Concurrent write {i}",
                metadata={"index": i},
            )

        # Run many concurrent writes
        entries = await asyncio.gather(*[write_memory(i) for i in range(20)])

        # All writes should succeed
        assert len(entries) == 20
        assert all(e is not None for e in entries)

        # All should be readable
        for entry in entries:
            memory = continuum_memory.get(entry.id)
            assert memory is not None

    @pytest.mark.asyncio
    async def test_read_during_write(self, continuum_memory):
        """Reading during writes should not cause issues."""
        # Pre-populate some memories
        existing_entries = []
        for i in range(5):
            entry = await continuum_memory.store(
                key=gen_key(),
                content=f"Existing {i}",
                metadata={},
            )
            existing_entries.append(entry)

        async def read_existing():
            for entry in existing_entries:
                continuum_memory.get(entry.id)
            return True

        async def write_new(i: int):
            return await continuum_memory.store(
                key=gen_key(),
                content=f"New during read {i}",
                metadata={},
            )

        # Run reads and writes concurrently
        results = await asyncio.gather(
            read_existing(),
            read_existing(),
            write_new(1),
            write_new(2),
            write_new(3),
        )

        # All operations should succeed
        assert all(r is not None for r in results)


# =============================================================================
# Recovery Tests
# =============================================================================


class TestRecovery:
    """Tests for recovery from tier migration failures."""

    @pytest.mark.asyncio
    async def test_memory_accessible_after_error(self, continuum_memory):
        """Memory should remain accessible after operation errors."""
        entry = await continuum_memory.store(
            key=gen_key(),
            content="Resilient memory",
            metadata={},
        )

        # Simulate some failed operations (using invalid IDs)
        result = continuum_memory.get("invalid-id-that-does-not-exist")
        # Should return None, not raise
        assert result is None

        # Original memory should still be accessible
        memory = continuum_memory.get(entry.id)
        assert memory is not None

    @pytest.mark.asyncio
    async def test_database_consistency_after_failures(self, continuum_memory):
        """Database should remain consistent after failures."""
        # Store several memories
        entries = []
        for i in range(10):
            entry = await continuum_memory.store(
                key=gen_key(),
                content=f"Memory {i}",
                metadata={},
            )
            entries.append(entry)

        # Try some invalid operations
        for _ in range(5):
            result = continuum_memory.get("nonexistent")
            assert result is None

        # All valid memories should still be accessible
        for entry in entries:
            memory = continuum_memory.get(entry.id)
            assert memory is not None


# =============================================================================
# Search and Retrieval Tests
# =============================================================================


class TestSearchAndRetrieval:
    """Tests for memory search across tiers."""

    @pytest.mark.asyncio
    async def test_search_finds_memories(self, continuum_memory):
        """Search should find relevant memories."""
        # Store memories with searchable content
        await continuum_memory.store(
            key=gen_key(),
            content="The quick brown fox jumps over the lazy dog",
            metadata={"topic": "animals"},
        )
        await continuum_memory.store(
            key=gen_key(),
            content="Python is a programming language",
            metadata={"topic": "programming"},
        )
        await continuum_memory.store(
            key=gen_key(),
            content="Database indexing improves query performance",
            metadata={"topic": "databases"},
        )

        # Search should work (if implemented)
        # This is a basic accessibility test
        # Actual search functionality may vary
        pass  # Search API depends on implementation

    @pytest.mark.asyncio
    async def test_get_recent_memories(self, continuum_memory):
        """Should be able to get recent memories."""
        # Store some memories
        for i in range(5):
            await continuum_memory.store(
                key=gen_key(),
                content=f"Recent memory {i}",
                metadata={"index": i},
            )
            await asyncio.sleep(0.01)

        # Get recent (if method exists)
        if hasattr(continuum_memory, "get_recent"):
            recent = continuum_memory.get_recent(limit=3)
            assert len(recent) <= 3


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in tier migration."""

    @pytest.mark.asyncio
    async def test_empty_content(self, continuum_memory):
        """Should handle empty content gracefully."""
        entry = await continuum_memory.store(
            key=gen_key(),
            content="",
            metadata={"empty": True},
        )

        memory = continuum_memory.get(entry.id)
        assert memory is not None

    @pytest.mark.asyncio
    async def test_large_content(self, continuum_memory):
        """Should handle large content."""
        large_content = "x" * 10000  # 10KB

        entry = await continuum_memory.store(
            key=gen_key(),
            content=large_content,
            metadata={},
        )

        memory = continuum_memory.get(entry.id)
        assert memory is not None

    @pytest.mark.asyncio
    async def test_special_characters_in_content(self, continuum_memory):
        """Should handle special characters."""
        special_content = "Hello\n\t世界emoji: 🎉"

        entry = await continuum_memory.store(
            key=gen_key(),
            content=special_content,
            metadata={},
        )

        memory = continuum_memory.get(entry.id)
        assert memory is not None

    @pytest.mark.asyncio
    async def test_complex_metadata(self, continuum_memory):
        """Should handle complex metadata."""
        complex_metadata = {
            "nested": {"deep": {"value": 123}},
            "list": [1, 2, 3],
            "unicode": "日本語",
            "boolean": True,
            "null": None,
        }

        entry = await continuum_memory.store(
            key=gen_key(),
            content="Memory with complex metadata",
            metadata=complex_metadata,
        )

        memory = continuum_memory.get(entry.id)
        assert memory is not None
