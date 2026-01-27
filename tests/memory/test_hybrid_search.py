"""
Tests for Hybrid Memory Search system.

Tests the combination of vector similarity and keyword (FTS5) search
using Reciprocal Rank Fusion for improved retrieval.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import sqlite3

from aragora.memory.hybrid_search import (
    HybridMemorySearch,
    HybridMemoryConfig,
    MemorySearchResult,
    KeywordIndex,
    get_hybrid_memory_search,
)


# =============================================================================
# MemorySearchResult Tests
# =============================================================================


class TestMemorySearchResult:
    """Tests for MemorySearchResult dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        result = MemorySearchResult(
            memory_id="mem-1",
            content="Test content",
            tier="slow",
            importance=0.7,
            combined_score=0.5,
            vector_score=0.6,
            keyword_score=0.4,
        )
        assert result.memory_id == "mem-1"
        assert result.tier == "slow"
        assert result.vector_rank == 0
        assert result.keyword_rank == 0

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = MemorySearchResult(
            memory_id="mem-1",
            content="Test content",
            tier="slow",
            importance=0.7,
            combined_score=0.5,
            vector_score=0.6,
            keyword_score=0.4,
            vector_rank=1,
            keyword_rank=2,
        )
        data = result.to_dict()
        assert data["memory_id"] == "mem-1"
        assert data["combined_score"] == 0.5
        assert data["vector_rank"] == 1


# =============================================================================
# HybridMemoryConfig Tests
# =============================================================================


class TestHybridMemoryConfig:
    """Tests for HybridMemoryConfig dataclass."""

    def test_default_values(self):
        """Test default configuration."""
        config = HybridMemoryConfig()
        assert config.rrf_k == 60
        assert config.vector_weight == 0.6
        assert config.keyword_weight == 0.4
        assert config.vector_limit == 50
        assert config.keyword_limit == 50

    def test_custom_values(self):
        """Test custom configuration."""
        config = HybridMemoryConfig(
            rrf_k=30,
            vector_weight=0.8,
            keyword_weight=0.2,
            tiers=["slow", "glacial"],
        )
        assert config.rrf_k == 30
        assert config.vector_weight == 0.8
        assert config.tiers == ["slow", "glacial"]


# =============================================================================
# KeywordIndex Tests
# =============================================================================


class TestKeywordIndex:
    """Tests for KeywordIndex FTS5 functionality."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database with continuum_memory table."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        # Create the main table
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE continuum_memory (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL,
                tier TEXT NOT NULL,
                importance REAL DEFAULT 0.5
            )
        """)
        conn.commit()
        conn.close()

        yield db_path

        # Cleanup
        db_path.unlink(missing_ok=True)

    def test_create_index(self, temp_db):
        """Test FTS index creation."""
        index = KeywordIndex(temp_db)

        # Verify FTS table exists
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_fts'"
        )
        assert cursor.fetchone() is not None
        conn.close()

        index.close()

    def test_search_empty_index(self, temp_db):
        """Test search on empty index."""
        index = KeywordIndex(temp_db)
        results = index.search("test query")
        assert results == []
        index.close()

    def test_search_with_data(self, temp_db):
        """Test search with data in index."""
        # Add test data
        conn = sqlite3.connect(str(temp_db))
        conn.execute("""
            INSERT INTO continuum_memory (id, content, tier, importance)
            VALUES
                ('mem-1', 'The circuit breaker pattern prevents cascade failures', 'slow', 0.8),
                ('mem-2', 'Rate limiting protects against abuse', 'slow', 0.7),
                ('mem-3', 'Circuit boards are electronic components', 'fast', 0.3)
        """)
        conn.commit()
        conn.close()

        # Create index and rebuild
        index = KeywordIndex(temp_db)
        index.rebuild_index()

        # Search for "circuit"
        results = index.search("circuit")

        assert len(results) >= 1
        # Should find circuit breaker and circuit boards
        result_ids = [r[0] for r in results]
        assert "mem-1" in result_ids or "mem-3" in result_ids

        index.close()

    def test_search_with_tier_filter(self, temp_db):
        """Test search with tier filtering."""
        # Add test data
        conn = sqlite3.connect(str(temp_db))
        conn.execute("""
            INSERT INTO continuum_memory (id, content, tier, importance)
            VALUES
                ('mem-1', 'Important slow tier memory', 'slow', 0.8),
                ('mem-2', 'Fast tier memory content', 'fast', 0.7),
                ('mem-3', 'Glacial tier memory store', 'glacial', 0.9)
        """)
        conn.commit()
        conn.close()

        index = KeywordIndex(temp_db)
        index.rebuild_index()

        # Search only in slow tier
        results = index.search("memory", tiers=["slow"])

        result_ids = [r[0] for r in results]
        assert "mem-1" in result_ids
        assert "mem-2" not in result_ids

        index.close()

    def test_search_with_importance_filter(self, temp_db):
        """Test search with importance threshold."""
        # Add test data
        conn = sqlite3.connect(str(temp_db))
        conn.execute("""
            INSERT INTO continuum_memory (id, content, tier, importance)
            VALUES
                ('mem-1', 'High importance memory', 'slow', 0.9),
                ('mem-2', 'Low importance memory', 'slow', 0.3),
                ('mem-3', 'Medium importance memory', 'slow', 0.6)
        """)
        conn.commit()
        conn.close()

        index = KeywordIndex(temp_db)
        index.rebuild_index()

        # Search with high importance threshold
        results = index.search("memory", min_importance=0.7)

        result_ids = [r[0] for r in results]
        assert "mem-1" in result_ids
        assert "mem-2" not in result_ids

        index.close()

    def test_rebuild_index(self, temp_db):
        """Test rebuilding the FTS index."""
        # Add test data
        conn = sqlite3.connect(str(temp_db))
        conn.execute("""
            INSERT INTO continuum_memory (id, content, tier, importance)
            VALUES
                ('mem-1', 'First memory entry', 'slow', 0.8),
                ('mem-2', 'Second memory entry', 'slow', 0.7)
        """)
        conn.commit()
        conn.close()

        index = KeywordIndex(temp_db)
        count = index.rebuild_index()

        assert count == 2

        index.close()


# =============================================================================
# HybridMemorySearch Tests
# =============================================================================


class TestHybridMemorySearch:
    """Tests for HybridMemorySearch class."""

    @pytest.fixture
    def mock_memory(self):
        """Create a mock ContinuumMemory."""
        memory = MagicMock()
        memory._km_adapter = None

        # Create a temporary database for the keyword index
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        # Create the main table
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE continuum_memory (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL,
                tier TEXT NOT NULL,
                importance REAL DEFAULT 0.5
            )
        """)
        # Add test data
        conn.execute("""
            INSERT INTO continuum_memory (id, content, tier, importance)
            VALUES
                ('mem-1', 'The circuit breaker pattern prevents cascade failures', 'slow', 0.8),
                ('mem-2', 'Rate limiting protects against abuse', 'slow', 0.7),
                ('mem-3', 'Retry logic with exponential backoff', 'medium', 0.6),
                ('mem-4', 'Circuit boards are electronic components', 'fast', 0.3)
        """)
        conn.commit()
        conn.close()

        memory.db_path = db_path

        yield memory

        # Cleanup
        db_path.unlink(missing_ok=True)

    def test_init(self, mock_memory):
        """Test initialization."""
        search = HybridMemorySearch(mock_memory)
        assert search.memory == mock_memory
        assert search.config is not None

        # Rebuild index for test data
        search.rebuild_keyword_index()

        search.close()

    @pytest.mark.asyncio
    async def test_search_keyword_only(self, mock_memory):
        """Test keyword-only search."""
        search = HybridMemorySearch(mock_memory)
        search.rebuild_keyword_index()

        results = await search.search_keyword_only("circuit", limit=10)

        assert len(results) >= 1
        # Results should have keyword scores
        for result in results:
            assert result.keyword_score > 0
            assert result.vector_score == 0

        search.close()

    @pytest.mark.asyncio
    async def test_search_with_tier_filter(self, mock_memory):
        """Test search with tier filtering."""
        search = HybridMemorySearch(mock_memory)
        search.rebuild_keyword_index()

        # Search only in slow tier
        results = await search.search("pattern", tiers=["slow"], limit=10)

        for result in results:
            assert result.tier == "slow"

        search.close()

    @pytest.mark.asyncio
    async def test_hybrid_search_combines_results(self, mock_memory):
        """Test that hybrid search combines vector and keyword results."""
        # Add a mock KM adapter for vector search
        mock_km_adapter = MagicMock()
        mock_memory._km_adapter = mock_km_adapter
        mock_memory.query_km_for_similar = MagicMock(
            return_value=[
                {
                    "id": "mem-1",
                    "content": "Circuit breaker",
                    "similarity": 0.9,
                    "tier": "slow",
                    "importance": 0.8,
                },
                {
                    "id": "mem-5",
                    "content": "Vector only result",
                    "similarity": 0.7,
                    "tier": "slow",
                    "importance": 0.6,
                },
            ]
        )

        search = HybridMemorySearch(mock_memory)
        search.rebuild_keyword_index()

        results = await search.search("circuit", limit=10)

        # Should have results from both sources
        result_ids = [r.memory_id for r in results]
        assert "mem-1" in result_ids  # In both sources

        # mem-1 should have both scores
        mem_1 = next(r for r in results if r.memory_id == "mem-1")
        assert mem_1.vector_score > 0  # From vector search
        assert mem_1.keyword_score > 0  # From keyword search

        search.close()

    @pytest.mark.asyncio
    async def test_rrf_ranking(self, mock_memory):
        """Test that RRF ranking works correctly."""
        # Add mock vector results with known ranking
        mock_memory._km_adapter = MagicMock()
        mock_memory.query_km_for_similar = MagicMock(
            return_value=[
                {
                    "id": "vec-1",
                    "content": "Top vector result",
                    "similarity": 0.95,
                    "tier": "slow",
                    "importance": 0.8,
                },
                {
                    "id": "vec-2",
                    "content": "Second vector result",
                    "similarity": 0.85,
                    "tier": "slow",
                    "importance": 0.7,
                },
            ]
        )

        search = HybridMemorySearch(mock_memory)
        search.rebuild_keyword_index()

        # Search for something that matches keyword results
        results = await search.search("circuit", limit=20)

        # Results should be sorted by combined score
        for i in range(len(results) - 1):
            assert results[i].combined_score >= results[i + 1].combined_score

        search.close()

    def test_custom_config(self, mock_memory):
        """Test using custom configuration."""
        config = HybridMemoryConfig(
            rrf_k=30,
            vector_weight=0.3,
            keyword_weight=0.7,
            min_combined_score=0.1,
        )

        search = HybridMemorySearch(mock_memory, config=config)

        assert search.config.rrf_k == 30
        assert search.config.vector_weight == 0.3

        search.close()


# =============================================================================
# Integration Tests
# =============================================================================


class TestHybridSearchIntegration:
    """Integration tests for hybrid search workflow."""

    @pytest.fixture
    def populated_db(self):
        """Create a populated test database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE continuum_memory (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL,
                tier TEXT NOT NULL,
                importance REAL DEFAULT 0.5
            )
        """)

        # Add realistic test data
        test_memories = [
            (
                "debate-pattern-1",
                "Agents use deliberative alignment for ethical decisions",
                "glacial",
                0.9,
            ),
            ("debate-pattern-2", "Multi-agent debates improve decision quality", "glacial", 0.85),
            (
                "resilience-1",
                "Circuit breakers prevent cascade failures in distributed systems",
                "slow",
                0.8,
            ),
            (
                "resilience-2",
                "Retry with exponential backoff handles transient failures",
                "slow",
                0.75,
            ),
            (
                "resilience-3",
                "Bulkheads isolate failures to prevent system-wide outages",
                "slow",
                0.7,
            ),
            ("perf-1", "Caching reduces latency for repeated queries", "medium", 0.6),
            ("perf-2", "Connection pooling improves database performance", "medium", 0.55),
            ("context-1", "User asked about deployment strategies", "fast", 0.4),
            ("context-2", "Discussion about API rate limiting", "fast", 0.35),
        ]

        conn.executemany(
            "INSERT INTO continuum_memory (id, content, tier, importance) VALUES (?, ?, ?, ?)",
            test_memories,
        )
        conn.commit()
        conn.close()

        yield db_path

        db_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_search_resilience_patterns(self, populated_db):
        """Test searching for resilience-related patterns."""
        mock_memory = MagicMock()
        mock_memory.db_path = populated_db
        mock_memory._km_adapter = None

        search = HybridMemorySearch(mock_memory)
        search.rebuild_keyword_index()

        results = await search.search("circuit breaker failures", limit=5)

        # Should find resilience-related memories
        assert len(results) > 0

        result_ids = [r.memory_id for r in results]
        # Circuit breaker should be highly ranked
        assert "resilience-1" in result_ids

        search.close()

    @pytest.mark.asyncio
    async def test_search_by_tier(self, populated_db):
        """Test searching within specific tiers."""
        mock_memory = MagicMock()
        mock_memory.db_path = populated_db
        mock_memory._km_adapter = None

        search = HybridMemorySearch(mock_memory)
        search.rebuild_keyword_index()

        # Search only glacial tier (foundational knowledge)
        results = await search.search("decisions", tiers=["glacial"], limit=5)

        for result in results:
            assert result.tier == "glacial"

        search.close()

    @pytest.mark.asyncio
    async def test_importance_filtering(self, populated_db):
        """Test filtering by importance threshold."""
        mock_memory = MagicMock()
        mock_memory.db_path = populated_db
        mock_memory._km_adapter = None

        search = HybridMemorySearch(mock_memory)
        search.rebuild_keyword_index()

        # Only high importance memories
        results = await search.search("failures", min_importance=0.7, limit=10)

        for result in results:
            assert result.importance >= 0.7

        search.close()


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Tests for singleton accessor."""

    def test_get_hybrid_memory_search(self):
        """Test getting the singleton instance."""
        # Reset singleton for testing
        import aragora.memory.hybrid_search as hybrid_module

        hybrid_module._hybrid_search = None

        # Create mock memory
        mock_memory = MagicMock()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE continuum_memory (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL,
                tier TEXT NOT NULL,
                importance REAL DEFAULT 0.5
            )
        """)
        conn.commit()
        conn.close()

        mock_memory.db_path = db_path
        mock_memory._km_adapter = None

        search = get_hybrid_memory_search(continuum_memory=mock_memory)
        assert search is not None

        # Second call returns same instance
        search2 = get_hybrid_memory_search()
        assert search is search2

        search.close()
        db_path.unlink(missing_ok=True)

        # Reset for other tests
        hybrid_module._hybrid_search = None
