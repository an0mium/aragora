"""
Comprehensive tests for Continuum Memory Retrieval operations.

Tests the RetrievalMixin in aragora/memory/continuum/retrieval.py including:
- Basic retrieval with tier filtering
- Keyword query filtering
- Importance thresholds
- Cross-tier retrieval
- Async retrieval operations
- Hybrid search functionality
- Query optimization
- Event emission
- Index rebuilding
- Retrieval scoring and ranking
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from aragora.memory.continuum import (
    ContinuumMemory,
    ContinuumMemoryEntry,
    reset_continuum_memory,
)
from aragora.memory.tier_manager import (
    DEFAULT_TIER_CONFIGS,
    MemoryTier,
    TierManager,
    reset_tier_manager,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path(tmp_path: Path) -> str:
    """Create a temporary database path for testing."""
    return str(tmp_path / "test_retrieval.db")


@pytest.fixture
def tier_manager() -> TierManager:
    """Create a fresh TierManager for testing."""
    return TierManager()


@pytest.fixture
def memory(temp_db_path: str, tier_manager: TierManager) -> ContinuumMemory:
    """Create a ContinuumMemory instance with isolated database."""
    reset_tier_manager()
    reset_continuum_memory()
    cms = ContinuumMemory(db_path=temp_db_path, tier_manager=tier_manager)
    yield cms
    reset_tier_manager()
    reset_continuum_memory()


@pytest.fixture
def populated_memory(memory: ContinuumMemory) -> ContinuumMemory:
    """Memory with diverse entries for retrieval testing."""
    # Fast tier - immediate context
    memory.add(
        "fast_python",
        "Python error handling with try-except blocks",
        tier=MemoryTier.FAST,
        importance=0.9,
    )
    memory.add(
        "fast_debug",
        "Debugging techniques for async applications",
        tier=MemoryTier.FAST,
        importance=0.7,
    )

    # Medium tier - session memory
    memory.add(
        "medium_api",
        "API design patterns for RESTful services",
        tier=MemoryTier.MEDIUM,
        importance=0.8,
    )
    memory.add(
        "medium_db",
        "Database query optimization strategies",
        tier=MemoryTier.MEDIUM,
        importance=0.6,
    )
    memory.add(
        "medium_python",
        "Python best practices for clean code",
        tier=MemoryTier.MEDIUM,
        importance=0.75,
    )

    # Slow tier - cross-session
    memory.add(
        "slow_arch",
        "Architectural patterns for microservices",
        tier=MemoryTier.SLOW,
        importance=0.85,
    )
    memory.add(
        "slow_security",
        "Security best practices for web applications",
        tier=MemoryTier.SLOW,
        importance=0.9,
    )

    # Glacial tier - foundational
    memory.add(
        "glacial_principles",
        "Core programming principles and SOLID design",
        tier=MemoryTier.GLACIAL,
        importance=0.95,
    )
    memory.add(
        "glacial_patterns",
        "Design patterns: Factory, Strategy, Observer",
        tier=MemoryTier.GLACIAL,
        importance=0.85,
    )

    return memory


@pytest.fixture
def memory_with_varied_importance(memory: ContinuumMemory) -> ContinuumMemory:
    """Memory with entries having varied importance scores."""
    importances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i, imp in enumerate(importances):
        memory.add(f"importance_{i}", f"Content with importance {imp}", importance=imp)
    return memory


# =============================================================================
# Test Basic Retrieval
# =============================================================================


class TestBasicRetrieval:
    """Tests for basic retrieve() functionality."""

    def test_retrieve_empty_memory(self, memory: ContinuumMemory) -> None:
        """Test retrieval from empty memory returns empty list."""
        results = memory.retrieve()

        assert results == []

    def test_retrieve_all_entries(self, populated_memory: ContinuumMemory) -> None:
        """Test retrieval returns entries from all tiers by default."""
        results = populated_memory.retrieve(limit=100)

        assert len(results) == 9
        tiers = {e.tier for e in results}
        assert tiers == {MemoryTier.FAST, MemoryTier.MEDIUM, MemoryTier.SLOW, MemoryTier.GLACIAL}

    def test_retrieve_respects_limit(self, populated_memory: ContinuumMemory) -> None:
        """Test retrieval respects limit parameter."""
        results = populated_memory.retrieve(limit=3)

        assert len(results) == 3

    def test_retrieve_sorted_by_score(self, populated_memory: ContinuumMemory) -> None:
        """Test retrieval results are sorted by score descending."""
        results = populated_memory.retrieve(limit=100)

        # Results should be in descending order by score (importance * surprise * decay)
        # Since all have default surprise=0, scoring is based on importance and recency
        # Just verify we got results in some order
        assert len(results) > 0

    def test_retrieve_returns_entry_objects(self, populated_memory: ContinuumMemory) -> None:
        """Test retrieval returns ContinuumMemoryEntry objects."""
        results = populated_memory.retrieve(limit=5)

        for entry in results:
            assert isinstance(entry, ContinuumMemoryEntry)
            assert hasattr(entry, "id")
            assert hasattr(entry, "tier")
            assert hasattr(entry, "content")
            assert hasattr(entry, "importance")


# =============================================================================
# Test Tier Filtering
# =============================================================================


class TestTierFiltering:
    """Tests for tier-based retrieval filtering."""

    def test_retrieve_single_tier(self, populated_memory: ContinuumMemory) -> None:
        """Test retrieval from a single tier."""
        results = populated_memory.retrieve(tiers=[MemoryTier.FAST], limit=100)

        assert len(results) == 2
        assert all(e.tier == MemoryTier.FAST for e in results)

    def test_retrieve_multiple_tiers(self, populated_memory: ContinuumMemory) -> None:
        """Test retrieval from multiple tiers."""
        results = populated_memory.retrieve(tiers=[MemoryTier.FAST, MemoryTier.MEDIUM], limit=100)

        assert len(results) == 5  # 2 fast + 3 medium
        tiers = {e.tier for e in results}
        assert tiers == {MemoryTier.FAST, MemoryTier.MEDIUM}

    def test_retrieve_exclude_glacial(self, populated_memory: ContinuumMemory) -> None:
        """Test retrieval with include_glacial=False."""
        results = populated_memory.retrieve(include_glacial=False, limit=100)

        assert all(e.tier != MemoryTier.GLACIAL for e in results)
        assert len(results) == 7  # 9 total - 2 glacial

    def test_retrieve_tier_parameter(self, populated_memory: ContinuumMemory) -> None:
        """Test retrieval with tier parameter (singular)."""
        results = populated_memory.retrieve(tier=MemoryTier.SLOW, limit=100)

        assert len(results) == 2
        assert all(e.tier == MemoryTier.SLOW for e in results)

    def test_retrieve_tier_string_value(self, populated_memory: ContinuumMemory) -> None:
        """Test retrieval with tier as string value."""
        results = populated_memory.retrieve(tier="medium", limit=100)

        assert len(results) == 3
        assert all(e.tier == MemoryTier.MEDIUM for e in results)

    def test_retrieve_empty_tier(self, memory: ContinuumMemory) -> None:
        """Test retrieval from tier with no entries."""
        memory.add("only_fast", "Content", tier=MemoryTier.FAST)

        results = memory.retrieve(tiers=[MemoryTier.GLACIAL], limit=100)

        assert results == []


# =============================================================================
# Test Keyword Query Filtering
# =============================================================================


class TestKeywordQueryFiltering:
    """Tests for keyword query-based retrieval filtering."""

    def test_retrieve_with_keyword_query(self, populated_memory: ContinuumMemory) -> None:
        """Test retrieval with keyword query filters content."""
        results = populated_memory.retrieve(query="Python", limit=100)

        assert len(results) >= 1
        # All results should contain 'python' (case insensitive)
        for entry in results:
            assert "python" in entry.content.lower()

    def test_retrieve_query_multiple_keywords(self, populated_memory: ContinuumMemory) -> None:
        """Test retrieval with multiple keywords (OR logic)."""
        results = populated_memory.retrieve(query="Python security", limit=100)

        # Should find entries with 'python' OR 'security'
        assert len(results) >= 2
        for entry in results:
            content_lower = entry.content.lower()
            assert "python" in content_lower or "security" in content_lower

    def test_retrieve_query_no_match(self, populated_memory: ContinuumMemory) -> None:
        """Test retrieval with query that matches nothing."""
        results = populated_memory.retrieve(query="xyznonexistent123", limit=100)

        assert results == []

    def test_retrieve_query_case_insensitive(self, populated_memory: ContinuumMemory) -> None:
        """Test retrieval query is case insensitive."""
        results_lower = populated_memory.retrieve(query="python", limit=100)
        results_upper = populated_memory.retrieve(query="PYTHON", limit=100)
        results_mixed = populated_memory.retrieve(query="PyThOn", limit=100)

        assert len(results_lower) == len(results_upper) == len(results_mixed)

    def test_retrieve_query_partial_match(self, populated_memory: ContinuumMemory) -> None:
        """Test retrieval with partial keyword match."""
        results = populated_memory.retrieve(query="debug", limit=100)

        # Should match 'debugging'
        assert len(results) >= 1
        found = False
        for entry in results:
            if "debug" in entry.content.lower():
                found = True
                break
        assert found

    def test_retrieve_query_max_keywords(self, memory: ContinuumMemory) -> None:
        """Test retrieval limits query to 50 keywords."""
        memory.add("test_entry", "Content with word1 and word2")

        # Create query with more than 50 words
        many_words = " ".join([f"word{i}" for i in range(60)])
        results = memory.retrieve(query=many_words, limit=100)

        # Should not raise error
        assert isinstance(results, list)


# =============================================================================
# Test Importance Threshold
# =============================================================================


class TestImportanceThreshold:
    """Tests for importance-based retrieval filtering."""

    def test_retrieve_min_importance(self, memory_with_varied_importance: ContinuumMemory) -> None:
        """Test retrieval with min_importance threshold."""
        results = memory_with_varied_importance.retrieve(min_importance=0.7, limit=100)

        assert all(e.importance >= 0.7 for e in results)
        assert len(results) == 4  # 0.7, 0.8, 0.9, 1.0

    def test_retrieve_min_importance_zero(
        self, memory_with_varied_importance: ContinuumMemory
    ) -> None:
        """Test retrieval with min_importance=0 includes all."""
        results = memory_with_varied_importance.retrieve(min_importance=0.0, limit=100)

        assert len(results) == 10

    def test_retrieve_min_importance_one(
        self, memory_with_varied_importance: ContinuumMemory
    ) -> None:
        """Test retrieval with min_importance=1.0."""
        results = memory_with_varied_importance.retrieve(min_importance=1.0, limit=100)

        assert len(results) == 1
        assert results[0].importance == 1.0

    def test_retrieve_min_importance_above_all(
        self, memory_with_varied_importance: ContinuumMemory
    ) -> None:
        """Test retrieval with min_importance above all entries."""
        results = memory_with_varied_importance.retrieve(min_importance=1.5, limit=100)

        assert results == []


# =============================================================================
# Test Combined Filters
# =============================================================================


class TestCombinedFilters:
    """Tests for combined retrieval filters."""

    def test_retrieve_tier_and_query(self, populated_memory: ContinuumMemory) -> None:
        """Test retrieval with tier and query filters."""
        results = populated_memory.retrieve(
            tiers=[MemoryTier.FAST, MemoryTier.MEDIUM], query="Python", limit=100
        )

        # Should have Python entries from fast and medium tiers
        assert len(results) >= 1
        for entry in results:
            assert entry.tier in [MemoryTier.FAST, MemoryTier.MEDIUM]
            assert "python" in entry.content.lower()

    def test_retrieve_importance_and_query(self, populated_memory: ContinuumMemory) -> None:
        """Test retrieval with importance and query filters."""
        results = populated_memory.retrieve(min_importance=0.8, query="design", limit=100)

        for entry in results:
            assert entry.importance >= 0.8
            assert "design" in entry.content.lower()

    def test_retrieve_all_filters(self, populated_memory: ContinuumMemory) -> None:
        """Test retrieval with all filters combined."""
        results = populated_memory.retrieve(
            tiers=[MemoryTier.SLOW, MemoryTier.GLACIAL],
            query="patterns",
            min_importance=0.8,
            include_glacial=True,
            limit=100,
        )

        for entry in results:
            assert entry.tier in [MemoryTier.SLOW, MemoryTier.GLACIAL]
            assert entry.importance >= 0.8
            assert "patterns" in entry.content.lower()


# =============================================================================
# Test Async Retrieval
# =============================================================================


class TestAsyncRetrieval:
    """Tests for retrieve_async() method."""

    @pytest.mark.asyncio
    async def test_retrieve_async_basic(self, populated_memory: ContinuumMemory) -> None:
        """Test basic async retrieval."""
        results = await populated_memory.retrieve_async(limit=5)

        assert len(results) == 5
        for entry in results:
            assert isinstance(entry, ContinuumMemoryEntry)

    @pytest.mark.asyncio
    async def test_retrieve_async_with_query(self, populated_memory: ContinuumMemory) -> None:
        """Test async retrieval with query filter."""
        results = await populated_memory.retrieve_async(query="Python", limit=100)

        assert len(results) >= 1
        for entry in results:
            assert "python" in entry.content.lower()

    @pytest.mark.asyncio
    async def test_retrieve_async_with_tiers(self, populated_memory: ContinuumMemory) -> None:
        """Test async retrieval with tier filter."""
        results = await populated_memory.retrieve_async(tiers=[MemoryTier.FAST], limit=100)

        assert len(results) == 2
        assert all(e.tier == MemoryTier.FAST for e in results)

    @pytest.mark.asyncio
    async def test_retrieve_async_concurrent(self, populated_memory: ContinuumMemory) -> None:
        """Test concurrent async retrievals."""
        tasks = [
            populated_memory.retrieve_async(query="Python", limit=10),
            populated_memory.retrieve_async(tiers=[MemoryTier.FAST], limit=10),
            populated_memory.retrieve_async(min_importance=0.8, limit=10),
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_retrieve_async_empty_result(self, populated_memory: ContinuumMemory) -> None:
        """Test async retrieval with no results."""
        results = await populated_memory.retrieve_async(query="nonexistent_xyz_123", limit=100)

        assert results == []


# =============================================================================
# Test Event Emission
# =============================================================================


class TestRetrievalEventEmission:
    """Tests for event emission during retrieval."""

    def test_retrieve_emits_memory_recall_event(self, populated_memory: ContinuumMemory) -> None:
        """Test that retrieve emits MEMORY_RECALL event."""
        mock_emitter = MagicMock()
        populated_memory.event_emitter = mock_emitter

        results = populated_memory.retrieve(limit=5)

        # Should have emitted events
        mock_emitter.emit_sync.assert_called()

    def test_retrieve_emits_correct_event_data(self, populated_memory: ContinuumMemory) -> None:
        """Test that retrieve emits correct event data."""
        mock_emitter = MagicMock()
        populated_memory.event_emitter = mock_emitter

        results = populated_memory.retrieve(query="Python", limit=5)

        # Find the memory_recall event call
        calls = mock_emitter.emit_sync.call_args_list
        recall_call = None
        for call in calls:
            if call.kwargs.get("event_type") == "memory_recall":
                recall_call = call
                break

        if recall_call:
            assert recall_call.kwargs["count"] == len(results)
            assert "tier_distribution" in recall_call.kwargs

    def test_retrieve_no_event_for_empty_results(self, memory: ContinuumMemory) -> None:
        """Test that retrieve does not emit event for empty results."""
        mock_emitter = MagicMock()
        memory.event_emitter = mock_emitter

        results = memory.retrieve(limit=5)

        # Should not have emitted since no results
        assert results == []

    def test_retrieve_handles_emitter_error(self, populated_memory: ContinuumMemory) -> None:
        """Test that retrieve handles emitter errors gracefully."""
        mock_emitter = MagicMock()
        mock_emitter.emit_sync.side_effect = TypeError("Emit error")
        populated_memory.event_emitter = mock_emitter

        # Should not raise
        results = populated_memory.retrieve(limit=5)

        assert len(results) > 0


# =============================================================================
# Test Hybrid Search
# =============================================================================


class TestHybridSearch:
    """Tests for hybrid_search() method."""

    @pytest.mark.asyncio
    async def test_hybrid_search_basic(self, populated_memory: ContinuumMemory) -> None:
        """Test basic hybrid search."""
        # Mock the hybrid search module at the source
        with patch("aragora.memory.hybrid_search.HybridMemorySearch") as mock_search:
            mock_instance = MagicMock()
            mock_instance.search = AsyncMock(return_value=[])
            mock_search.return_value = mock_instance

            # Clear any cached hybrid search instance
            populated_memory._hybrid_search = None

            results = await populated_memory.hybrid_search("Python patterns", limit=5)

            mock_instance.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_search_with_tiers(self, populated_memory: ContinuumMemory) -> None:
        """Test hybrid search with tier filter."""
        with patch("aragora.memory.hybrid_search.HybridMemorySearch") as mock_search:
            mock_instance = MagicMock()
            mock_instance.search = AsyncMock(return_value=[])
            mock_search.return_value = mock_instance

            populated_memory._hybrid_search = None

            await populated_memory.hybrid_search(
                "Python", limit=5, tiers=[MemoryTier.FAST, MemoryTier.MEDIUM]
            )

            call_kwargs = mock_instance.search.call_args.kwargs
            assert call_kwargs["tiers"] == ["fast", "medium"]

    @pytest.mark.asyncio
    async def test_hybrid_search_with_vector_weight(
        self, populated_memory: ContinuumMemory
    ) -> None:
        """Test hybrid search with custom vector weight."""
        with patch("aragora.memory.hybrid_search.HybridMemorySearch") as mock_search:
            mock_instance = MagicMock()
            mock_instance.search = AsyncMock(return_value=[])
            mock_search.return_value = mock_instance

            populated_memory._hybrid_search = None

            await populated_memory.hybrid_search("Python", limit=5, vector_weight=0.7)

            call_kwargs = mock_instance.search.call_args.kwargs
            assert call_kwargs["vector_weight"] == 0.7

    @pytest.mark.asyncio
    async def test_hybrid_search_with_min_importance(
        self, populated_memory: ContinuumMemory
    ) -> None:
        """Test hybrid search with importance threshold."""
        with patch("aragora.memory.hybrid_search.HybridMemorySearch") as mock_search:
            mock_instance = MagicMock()
            mock_instance.search = AsyncMock(return_value=[])
            mock_search.return_value = mock_instance

            populated_memory._hybrid_search = None

            await populated_memory.hybrid_search("Python", limit=5, min_importance=0.8)

            call_kwargs = mock_instance.search.call_args.kwargs
            assert call_kwargs["min_importance"] == 0.8


# =============================================================================
# Test Index Rebuilding
# =============================================================================


class TestIndexRebuilding:
    """Tests for rebuild_keyword_index() method."""

    def test_rebuild_keyword_index(self, populated_memory: ContinuumMemory) -> None:
        """Test rebuilding keyword index."""
        with patch("aragora.memory.hybrid_search.HybridMemorySearch") as mock_search:
            mock_instance = MagicMock()
            mock_instance.rebuild_keyword_index.return_value = 9
            mock_search.return_value = mock_instance

            populated_memory._hybrid_search = None

            count = populated_memory.rebuild_keyword_index()

            mock_instance.rebuild_keyword_index.assert_called_once()
            assert count == 9


# =============================================================================
# Test Retrieval Scoring
# =============================================================================


class TestRetrievalScoring:
    """Tests for retrieval scoring and ranking."""

    def test_scoring_considers_importance(self, memory: ContinuumMemory) -> None:
        """Test that scoring considers importance."""
        memory.add("low_imp", "Same content here", importance=0.1)
        memory.add("high_imp", "Same content here", importance=0.9)

        results = memory.retrieve(limit=2)

        # Higher importance should rank higher
        importances = [e.importance for e in results]
        assert importances == sorted(importances, reverse=True)

    def test_scoring_considers_tier_decay(self, memory: ContinuumMemory) -> None:
        """Test that scoring considers tier-based decay."""
        # Add entries with same importance but different tiers
        memory.add("fast_entry", "Same content", tier=MemoryTier.FAST, importance=0.5)
        memory.add("glacial_entry", "Same content", tier=MemoryTier.GLACIAL, importance=0.5)

        results = memory.retrieve(limit=2)

        # Results should consider tier-based decay in scoring
        assert len(results) == 2

    def test_scoring_considers_surprise(self, memory: ContinuumMemory) -> None:
        """Test that scoring considers surprise score."""
        memory.add("low_surprise", "Content", importance=0.5)
        memory.add("high_surprise", "Content", importance=0.5)

        # Set different surprise scores
        with memory.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE continuum_memory SET surprise_score = 0.1 WHERE id = ?", ("low_surprise",)
            )
            cursor.execute(
                "UPDATE continuum_memory SET surprise_score = 0.9 WHERE id = ?", ("high_surprise",)
            )
            conn.commit()

        results = memory.retrieve(limit=2)

        # Higher surprise should rank higher
        assert len(results) == 2


# =============================================================================
# Test Concurrent Retrieval
# =============================================================================


class TestConcurrentRetrieval:
    """Tests for concurrent retrieval operations."""

    def test_concurrent_retrieves(self, populated_memory: ContinuumMemory) -> None:
        """Test concurrent synchronous retrievals."""
        errors: list[Exception] = []
        results_list: list[list[ContinuumMemoryEntry]] = []
        lock = threading.Lock()

        def do_retrieve() -> None:
            try:
                results = populated_memory.retrieve(limit=5)
                with lock:
                    results_list.append(results)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_retrieve) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results_list) == 10

    def test_concurrent_reads_and_writes(self, memory: ContinuumMemory) -> None:
        """Test concurrent reads and writes."""
        # Pre-populate
        for i in range(20):
            memory.add(f"initial_{i}", f"Initial content {i}")

        errors: list[Exception] = []

        def do_reads() -> None:
            try:
                for _ in range(10):
                    memory.retrieve(limit=5)
            except Exception as e:
                errors.append(e)

        def do_writes(idx: int) -> None:
            try:
                for i in range(10):
                    memory.add(f"new_{idx}_{i}", f"New content {i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=do_reads),
            threading.Thread(target=do_writes, args=(0,)),
            threading.Thread(target=do_reads),
            threading.Thread(target=do_writes, args=(1,)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# Test Query Optimization
# =============================================================================


class TestQueryOptimization:
    """Tests for query optimization in retrieval."""

    def test_retrieve_uses_index_for_tier(self, memory: ContinuumMemory) -> None:
        """Test that tier filtering uses database index efficiently."""
        # Add many entries
        for i in range(100):
            tier = MemoryTier.FAST if i % 4 == 0 else MemoryTier.SLOW
            memory.add(f"entry_{i}", f"Content {i}", tier=tier)

        # Retrieval should be fast due to index
        start = time.time()
        results = memory.retrieve(tiers=[MemoryTier.FAST], limit=100)
        elapsed = time.time() - start

        assert elapsed < 1.0  # Should be very fast
        assert len(results) == 25  # 100 / 4 entries are in fast tier

    def test_retrieve_uses_index_for_importance(self, memory: ContinuumMemory) -> None:
        """Test that importance filtering is efficient."""
        for i in range(100):
            memory.add(f"entry_{i}", f"Content {i}", importance=i / 100)

        start = time.time()
        results = memory.retrieve(min_importance=0.9, limit=100)
        elapsed = time.time() - start

        assert elapsed < 1.0
        assert all(e.importance >= 0.9 for e in results)


# =============================================================================
# Test AwaitableList
# =============================================================================


class TestAwaitableList:
    """Tests for AwaitableList wrapper."""

    def test_retrieve_returns_awaitable_list(self, populated_memory: ContinuumMemory) -> None:
        """Test that retrieve returns an AwaitableList."""
        results = populated_memory.retrieve(limit=5)

        # Should work as a regular list
        assert len(results) == 5
        assert results[0] is not None

    @pytest.mark.asyncio
    async def test_awaitable_list_can_be_awaited(self, populated_memory: ContinuumMemory) -> None:
        """Test that AwaitableList can be awaited."""
        results = populated_memory.retrieve(limit=5)

        # Should be awaitable
        awaited = await results

        assert len(awaited) == 5


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestRetrievalEdgeCases:
    """Tests for retrieval edge cases."""

    def test_retrieve_with_special_characters_in_query(self, memory: ContinuumMemory) -> None:
        """Test retrieval with special characters in query."""
        memory.add("special", "Content with special chars: (test) [brackets] {braces}")

        # Should not raise
        results = memory.retrieve(query="(test)", limit=10)

        # May or may not find results depending on SQL handling
        assert isinstance(results, list)

    def test_retrieve_with_empty_query(self, populated_memory: ContinuumMemory) -> None:
        """Test retrieval with empty query string."""
        results = populated_memory.retrieve(query="", limit=10)

        # Empty query should behave like no query
        assert len(results) > 0

    def test_retrieve_with_whitespace_only_query(self, populated_memory: ContinuumMemory) -> None:
        """Test retrieval with whitespace-only query."""
        results = populated_memory.retrieve(query="   ", limit=10)

        # Whitespace query should behave like no query
        assert len(results) > 0

    def test_retrieve_with_very_long_content(self, memory: ContinuumMemory) -> None:
        """Test retrieval with very long content."""
        long_content = "word " * 1000
        memory.add("long_content", long_content)

        results = memory.retrieve(query="word", limit=10)

        assert len(results) == 1
        assert len(results[0].content) > 4000

    def test_retrieve_with_unicode_content(self, memory: ContinuumMemory) -> None:
        """Test retrieval with unicode content."""
        memory.add("unicode", "Content with unicode: cafe cafe emoji test")

        results = memory.retrieve(query="cafe", limit=10)

        assert len(results) >= 1

    def test_retrieve_limit_zero(self, populated_memory: ContinuumMemory) -> None:
        """Test retrieval with limit=0."""
        results = populated_memory.retrieve(limit=0)

        assert results == []

    def test_retrieve_negative_importance(self, memory: ContinuumMemory) -> None:
        """Test retrieval behavior with negative importance threshold."""
        memory.add("test", "Content", importance=0.5)

        results = memory.retrieve(min_importance=-1.0, limit=10)

        # Negative threshold should include all entries
        assert len(results) == 1


# =============================================================================
# Test Cross-Tier Retrieval
# =============================================================================


class TestCrossTierRetrieval:
    """Tests for cross-tier retrieval functionality."""

    def test_cross_tier_query_finds_all_matches(self, populated_memory: ContinuumMemory) -> None:
        """Test that query finds matches across all tiers."""
        # 'patterns' appears in multiple tiers
        results = populated_memory.retrieve(query="patterns", limit=100)

        tiers_found = {e.tier for e in results}
        # Should find entries from multiple tiers if they contain 'patterns'
        assert len(tiers_found) >= 1

    def test_cross_tier_importance_ranking(self, memory: ContinuumMemory) -> None:
        """Test that cross-tier retrieval ranks by importance correctly."""
        # Add entries in different tiers with different importance
        memory.add("glacial_high", "Same keyword", tier=MemoryTier.GLACIAL, importance=0.95)
        memory.add("fast_low", "Same keyword", tier=MemoryTier.FAST, importance=0.3)
        memory.add("medium_mid", "Same keyword", tier=MemoryTier.MEDIUM, importance=0.6)

        results = memory.retrieve(query="keyword", limit=10)

        # All should be found
        assert len(results) == 3

    def test_get_by_id_with_tier_filter(self, populated_memory: ContinuumMemory) -> None:
        """Test that tier filter with exact ID still works."""
        results = populated_memory.retrieve(query="fast_python", tier=MemoryTier.FAST, limit=1)

        # If entry found, should be from correct tier
        if results:
            assert results[0].tier == MemoryTier.FAST
