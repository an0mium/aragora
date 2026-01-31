"""
Tests for RlmAdapter - Bridges RLM Compression to Knowledge Mound.

Tests cover:
- CompressionPattern dataclass
- ContentPriority dataclass
- Adapter initialization
- Compression pattern storage
- Pattern lookup by content markers
- Access pattern tracking
- Priority content retrieval
- Compression hints
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from aragora.knowledge.mound.adapters.rlm_adapter import (
    RlmAdapter,
    CompressionPattern,
    ContentPriority,
)


# =============================================================================
# CompressionPattern Dataclass Tests
# =============================================================================


class TestCompressionPattern:
    """Tests for CompressionPattern dataclass."""

    def test_create_pattern(self):
        """Should create a compression pattern."""
        pattern = CompressionPattern(
            id="pattern-001",
            compression_ratio=0.3,
            value_score=0.85,
            content_markers=["security", "api", "authentication"],
        )

        assert pattern.id == "pattern-001"
        assert pattern.compression_ratio == 0.3
        assert pattern.value_score == 0.85
        assert len(pattern.content_markers) == 3

    def test_pattern_with_defaults(self):
        """Should use default values."""
        pattern = CompressionPattern(
            id="pattern-002",
            compression_ratio=0.5,
            value_score=0.7,
            content_markers=["code"],
        )

        assert pattern.content_type == "general"
        assert pattern.usage_count == 0
        assert pattern.created_at == ""
        assert pattern.metadata == {}

    def test_pattern_with_all_fields(self):
        """Should accept all fields."""
        pattern = CompressionPattern(
            id="pattern-003",
            compression_ratio=0.2,
            value_score=0.9,
            content_markers=["debate", "analysis"],
            content_type="debate",
            usage_count=5,
            created_at="2024-01-01T00:00:00Z",
            metadata={"source": "test"},
        )

        assert pattern.content_type == "debate"
        assert pattern.usage_count == 5
        assert pattern.created_at == "2024-01-01T00:00:00Z"
        assert pattern.metadata == {"source": "test"}


# =============================================================================
# ContentPriority Dataclass Tests
# =============================================================================


class TestContentPriority:
    """Tests for ContentPriority dataclass."""

    def test_create_priority(self):
        """Should create a content priority record."""
        priority = ContentPriority(
            content_id="content-001",
            access_count=10,
            last_accessed="2024-01-15T12:00:00Z",
            priority_score=0.8,
        )

        assert priority.content_id == "content-001"
        assert priority.access_count == 10
        assert priority.priority_score == 0.8

    def test_priority_with_defaults(self):
        """Should use default content type."""
        priority = ContentPriority(
            content_id="content-002",
            access_count=5,
            last_accessed="2024-01-15T12:00:00Z",
            priority_score=0.5,
        )

        assert priority.content_type == "general"


# =============================================================================
# RlmAdapter Initialization Tests
# =============================================================================


class TestRlmAdapterInit:
    """Tests for RlmAdapter initialization."""

    def test_default_init(self):
        """Should initialize with default values."""
        adapter = RlmAdapter()

        assert adapter._compressor is None
        assert adapter._enable_dual_write is False
        assert adapter.PATTERN_PREFIX == "cp_"
        assert adapter.PRIORITY_PREFIX == "pr_"
        assert adapter.MIN_VALUE_SCORE == 0.7

    def test_init_with_compressor(self):
        """Should accept compressor."""
        mock_compressor = MagicMock()
        adapter = RlmAdapter(compressor=mock_compressor)

        assert adapter._compressor is mock_compressor
        assert adapter.compressor is mock_compressor

    def test_init_with_dual_write(self):
        """Should accept dual write flag."""
        adapter = RlmAdapter(enable_dual_write=True)

        assert adapter._enable_dual_write is True

    def test_set_compressor(self):
        """Should set compressor after init."""
        adapter = RlmAdapter()
        mock_compressor = MagicMock()

        adapter.set_compressor(mock_compressor)

        assert adapter._compressor is mock_compressor


# =============================================================================
# Compression Pattern Storage Tests
# =============================================================================


class TestStoreCompressionPattern:
    """Tests for storing compression patterns."""

    def test_store_high_value_pattern(self):
        """Should store pattern with high value score."""
        adapter = RlmAdapter()

        result = adapter.store_compression_pattern(
            compression_ratio=0.3,
            value_score=0.85,
            content_markers=["security", "api"],
        )

        assert result is not None
        assert result.startswith("cp_")
        assert result in adapter._patterns

    def test_skip_low_value_pattern(self):
        """Should skip pattern with low value score."""
        adapter = RlmAdapter()

        result = adapter.store_compression_pattern(
            compression_ratio=0.3,
            value_score=0.5,  # Below MIN_VALUE_SCORE (0.7)
            content_markers=["test"],
        )

        assert result is None
        assert len(adapter._patterns) == 0

    def test_pattern_updates_usage_count(self):
        """Should update usage count for existing pattern."""
        adapter = RlmAdapter()

        # Store first pattern
        result1 = adapter.store_compression_pattern(
            compression_ratio=0.3,
            value_score=0.85,
            content_markers=["security"],
        )

        # Store same pattern again
        result2 = adapter.store_compression_pattern(
            compression_ratio=0.25,
            value_score=0.9,
            content_markers=["security"],
        )

        assert result1 == result2
        assert adapter._patterns[result1]["usage_count"] == 2

    def test_pattern_with_content_type(self):
        """Should store pattern with content type."""
        adapter = RlmAdapter()

        result = adapter.store_compression_pattern(
            compression_ratio=0.4,
            value_score=0.75,
            content_markers=["code", "python"],
            content_type="code",
        )

        assert adapter._patterns[result]["content_type"] == "code"

    def test_pattern_with_metadata(self):
        """Should store pattern with metadata."""
        adapter = RlmAdapter()

        result = adapter.store_compression_pattern(
            compression_ratio=0.3,
            value_score=0.8,
            content_markers=["test"],
            metadata={"source": "unit_test"},
        )

        assert adapter._patterns[result]["metadata"] == {"source": "unit_test"}

    def test_pattern_tracks_statistics(self):
        """Should track compression statistics."""
        adapter = RlmAdapter()

        # Store successful pattern
        adapter.store_compression_pattern(
            compression_ratio=0.3,
            value_score=0.85,
            content_markers=["test"],
        )

        # Store failed pattern (low value)
        adapter.store_compression_pattern(
            compression_ratio=0.3,
            value_score=0.5,
            content_markers=["low"],
        )

        assert adapter._total_compressions == 2
        assert adapter._successful_compressions == 1


# =============================================================================
# Pattern Lookup Tests
# =============================================================================


class TestGetPattern:
    """Tests for pattern lookup."""

    def test_get_pattern_by_id(self):
        """Should get pattern by ID."""
        adapter = RlmAdapter()

        pattern_id = adapter.store_compression_pattern(
            compression_ratio=0.3,
            value_score=0.85,
            content_markers=["lookup"],
        )

        result = adapter.get_pattern(pattern_id)

        assert result is not None
        assert result["id"] == pattern_id
        assert result["value_score"] == 0.85

    def test_get_pattern_without_prefix(self):
        """Should find pattern without prefix."""
        adapter = RlmAdapter()

        pattern_id = adapter.store_compression_pattern(
            compression_ratio=0.3,
            value_score=0.85,
            content_markers=["lookup"],
        )

        # Get without cp_ prefix
        raw_id = pattern_id[3:]  # Remove "cp_"
        result = adapter.get_pattern(raw_id)

        assert result is not None

    def test_get_pattern_not_found(self):
        """Should return None for missing pattern."""
        adapter = RlmAdapter()
        result = adapter.get_pattern("nonexistent")
        assert result is None


# =============================================================================
# Pattern Matching Tests
# =============================================================================


class TestGetPatternsForContent:
    """Tests for finding patterns matching content markers."""

    def test_match_by_markers(self):
        """Should find patterns matching content markers."""
        adapter = RlmAdapter()

        # Store patterns
        adapter.store_compression_pattern(
            compression_ratio=0.3,
            value_score=0.85,
            content_markers=["security", "api"],
        )
        adapter.store_compression_pattern(
            compression_ratio=0.4,
            value_score=0.8,
            content_markers=["code", "python"],
        )

        results = adapter.get_patterns_for_content(["security", "auth"])

        assert len(results) >= 1
        assert any("security" in p.get("content_markers", []) for p in results)

    def test_match_relevance_scoring(self):
        """Should include relevance score in results."""
        adapter = RlmAdapter()

        adapter.store_compression_pattern(
            compression_ratio=0.3,
            value_score=0.85,
            content_markers=["security", "api", "auth"],
        )

        results = adapter.get_patterns_for_content(["security", "api"])

        assert len(results) == 1
        assert "relevance" in results[0]
        assert results[0]["relevance"] > 0

    def test_no_match_returns_empty(self):
        """Should return empty for no matches."""
        adapter = RlmAdapter()

        adapter.store_compression_pattern(
            compression_ratio=0.3,
            value_score=0.85,
            content_markers=["security"],
        )

        results = adapter.get_patterns_for_content(["database", "sql"])

        assert len(results) == 0


# =============================================================================
# Access Pattern Tests
# =============================================================================


class TestUpdateAccessPattern:
    """Tests for recording content access patterns."""

    def test_track_new_content(self):
        """Should track access for new content."""
        adapter = RlmAdapter()

        adapter.update_access_pattern("content-001")

        assert "content-001" in adapter._priorities
        assert adapter._priorities["content-001"]["access_count"] == 1

    def test_increment_access_count(self):
        """Should increment access count."""
        adapter = RlmAdapter()

        adapter.update_access_pattern("content-001")
        adapter.update_access_pattern("content-001")
        adapter.update_access_pattern("content-001")

        assert adapter._priorities["content-001"]["access_count"] == 3

    def test_update_last_accessed(self):
        """Should update last accessed timestamp."""
        adapter = RlmAdapter()

        adapter.update_access_pattern("content-001")

        assert adapter._priorities["content-001"]["last_accessed"] != ""

    def test_calculate_priority_score(self):
        """Should calculate priority score based on access count."""
        adapter = RlmAdapter()

        for _ in range(5):
            adapter.update_access_pattern("content-001")

        # Priority = min(1.0, access_count / 10)
        assert adapter._priorities["content-001"]["priority_score"] == 0.5

    def test_track_content_type(self):
        """Should track content type."""
        adapter = RlmAdapter()

        adapter.update_access_pattern("content-001", content_type="code")

        assert adapter._priorities["content-001"]["content_type"] == "code"


# =============================================================================
# Priority Content Tests
# =============================================================================


class TestGetPriorityContent:
    """Tests for retrieving priority content."""

    def test_get_priority_content(self):
        """Should get high-priority content."""
        adapter = RlmAdapter()

        # Create content with enough accesses
        for i in range(5):
            adapter.update_access_pattern(f"content-{i:03d}")
            for _ in range(i + 3):  # Varying access counts
                adapter.update_access_pattern(f"content-{i:03d}")

        results = adapter.get_priority_content(limit=10)

        assert len(results) > 0
        # Should be sorted by priority score descending
        if len(results) > 1:
            assert results[0].priority_score >= results[1].priority_score

    def test_filter_by_min_access_count(self):
        """Should filter by minimum access count."""
        adapter = RlmAdapter()

        # Create content with low accesses
        adapter.update_access_pattern("low-access")
        adapter.update_access_pattern("low-access")

        # Create content with high accesses
        for _ in range(10):
            adapter.update_access_pattern("high-access")

        results = adapter.get_priority_content(min_access_count=5)

        assert all(r.access_count >= 5 for r in results)

    def test_filter_by_content_type(self):
        """Should filter by content type."""
        adapter = RlmAdapter()

        for _ in range(5):
            adapter.update_access_pattern("code-content", content_type="code")
            adapter.update_access_pattern("debate-content", content_type="debate")

        results = adapter.get_priority_content(content_type="code")

        assert all(r.content_type == "code" for r in results)


# =============================================================================
# Compression Hints Tests
# =============================================================================


class TestGetCompressionHints:
    """Tests for getting compression strategy hints."""

    def test_hints_with_no_patterns(self):
        """Should return default hints with no patterns."""
        adapter = RlmAdapter()

        hints = adapter.get_compression_hints(["security"])

        assert hints["recommended_ratio"] == 0.5
        assert hints["strategy"] == "default"
        assert hints["confidence"] == 0.0
        assert hints["based_on_patterns"] == 0

    def test_hints_from_patterns(self):
        """Should calculate hints from matching patterns."""
        adapter = RlmAdapter()

        adapter.store_compression_pattern(
            compression_ratio=0.2,
            value_score=0.9,
            content_markers=["security", "api"],
        )

        hints = adapter.get_compression_hints(["security", "api"])

        assert hints["recommended_ratio"] < 0.5
        assert hints["based_on_patterns"] == 1
        assert hints["top_pattern_id"] is not None

    def test_aggressive_strategy(self):
        """Should recommend aggressive for low ratio."""
        adapter = RlmAdapter()

        adapter.store_compression_pattern(
            compression_ratio=0.2,
            value_score=0.9,
            content_markers=["test"],
        )

        hints = adapter.get_compression_hints(["test"])

        assert hints["strategy"] == "aggressive"

    def test_conservative_strategy(self):
        """Should recommend conservative for high ratio."""
        adapter = RlmAdapter()

        adapter.store_compression_pattern(
            compression_ratio=0.7,
            value_score=0.8,
            content_markers=["test"],
        )

        hints = adapter.get_compression_hints(["test"])

        assert hints["strategy"] == "conservative"


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStats:
    """Tests for adapter statistics."""

    def test_get_stats_empty(self):
        """Should return stats for empty adapter."""
        adapter = RlmAdapter()

        stats = adapter.get_stats()

        assert stats["total_patterns"] == 0
        assert stats["total_priorities"] == 0
        assert stats["total_compressions"] == 0
        assert stats["successful_compressions"] == 0
        assert stats["success_rate"] == 0.0

    def test_get_stats_with_data(self):
        """Should return accurate stats."""
        adapter = RlmAdapter()

        # Add patterns
        adapter.store_compression_pattern(
            compression_ratio=0.3,
            value_score=0.85,
            content_markers=["test"],
            content_type="code",
        )

        # Add priorities
        for _ in range(5):
            adapter.update_access_pattern("content-001")

        stats = adapter.get_stats()

        assert stats["total_patterns"] == 1
        assert stats["total_priorities"] == 1
        assert stats["total_compressions"] == 1
        assert stats["successful_compressions"] == 1
        assert stats["success_rate"] == 1.0
        assert "code" in stats["pattern_types"]


# =============================================================================
# Sync Tests
# =============================================================================


class TestSync:
    """Tests for KM sync operations."""

    @pytest.mark.asyncio
    async def test_sync_to_mound(self):
        """Should sync patterns to KM."""
        adapter = RlmAdapter()

        # Store patterns with usage >= 2
        for _ in range(2):
            adapter.store_compression_pattern(
                compression_ratio=0.3,
                value_score=0.85,
                content_markers=["sync-test"],
            )

        mock_mound = AsyncMock()
        mock_mound.ingest = AsyncMock(return_value="item-001")

        result = await adapter.sync_to_mound(mock_mound, "ws-test")

        assert result["patterns_synced"] >= 1

    @pytest.mark.asyncio
    async def test_sync_skips_low_usage(self):
        """Should skip patterns with low usage count."""
        adapter = RlmAdapter()

        # Store pattern with usage = 1
        adapter.store_compression_pattern(
            compression_ratio=0.3,
            value_score=0.85,
            content_markers=["single-use"],
        )

        mock_mound = AsyncMock()
        mock_mound.ingest = AsyncMock(return_value="item-001")

        result = await adapter.sync_to_mound(mock_mound, "ws-test")

        assert result["patterns_synced"] == 0

    @pytest.mark.asyncio
    async def test_load_from_mound(self):
        """Should load patterns from KM."""
        adapter = RlmAdapter()

        mock_node = MagicMock()
        mock_node.metadata = {
            "type": "compression_pattern",
            "pattern_id": "cp_test123",
            "compression_ratio": 0.3,
            "value_score": 0.85,
            "content_type": "code",
            "content_markers": ["test"],
            "usage_count": 5,
        }
        mock_node.created_at = None
        mock_node.updated_at = None

        mock_mound = AsyncMock()
        mock_mound.query_nodes = AsyncMock(return_value=[mock_node])

        result = await adapter.load_from_mound(mock_mound, "ws-test")

        assert result["patterns_loaded"] == 1
        assert "cp_test123" in adapter._patterns
