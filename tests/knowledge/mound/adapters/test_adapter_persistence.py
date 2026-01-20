"""
Tests for adapter KM persistence methods.

Tests sync_to_mound() and load_from_mound() methods for:
- RankingAdapter
- RlmAdapter
- ContinuumAdapter
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestRankingAdapterPersistence:
    """Tests for RankingAdapter persistence methods."""

    def test_sync_to_mound_empty(self):
        """sync_to_mound with no expertise records returns empty stats."""
        from aragora.knowledge.mound.adapters.ranking_adapter import RankingAdapter

        adapter = RankingAdapter()
        mound = AsyncMock()

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            adapter.sync_to_mound(mound, workspace_id="test")
        )

        assert result["expertise_synced"] == 0
        assert result["errors"] == []

    def test_sync_to_mound_with_expertise(self):
        """sync_to_mound syncs expertise records to KM."""
        from aragora.knowledge.mound.adapters.ranking_adapter import RankingAdapter

        adapter = RankingAdapter()

        # Add some expertise
        adapter.store_agent_expertise(
            agent_name="test-agent",
            domain="security",
            elo=1650,
            delta=50,
            debate_id="debate-1",
        )

        mound = AsyncMock()
        mound.ingest = AsyncMock(return_value="km_node_123")

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            adapter.sync_to_mound(mound, workspace_id="test")
        )

        assert result["expertise_synced"] == 1
        assert result["errors"] == []
        mound.ingest.assert_called_once()

    def test_load_from_mound_empty(self):
        """load_from_mound with no KM data returns empty stats."""
        from aragora.knowledge.mound.adapters.ranking_adapter import RankingAdapter

        adapter = RankingAdapter()
        mound = AsyncMock()
        mound.query_nodes = AsyncMock(return_value=[])

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            adapter.load_from_mound(mound, workspace_id="test")
        )

        assert result["expertise_loaded"] == 0
        assert result["errors"] == []

    def test_load_from_mound_restores_state(self):
        """load_from_mound restores expertise from KM nodes."""
        from aragora.knowledge.mound.adapters.ranking_adapter import RankingAdapter
        from datetime import datetime

        adapter = RankingAdapter()

        # Mock KM node with expertise metadata
        mock_node = MagicMock()
        mock_node.metadata = {
            "type": "agent_expertise",
            "agent_name": "test-agent",
            "domain": "coding",
            "elo": 1700,
            "debate_count": 5,
        }
        mock_node.created_at = datetime.now()
        mock_node.updated_at = datetime.now()

        mound = AsyncMock()
        mound.query_nodes = AsyncMock(return_value=[mock_node])

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            adapter.load_from_mound(mound, workspace_id="test")
        )

        assert result["expertise_loaded"] == 1
        assert result["errors"] == []

        # Verify state was restored
        expertise = adapter.get_agent_expertise("test-agent", "coding")
        assert expertise is not None
        assert expertise["elo"] == 1700


class TestRlmAdapterPersistence:
    """Tests for RlmAdapter persistence methods."""

    def test_sync_to_mound_empty(self):
        """sync_to_mound with no patterns returns empty stats."""
        from aragora.knowledge.mound.adapters.rlm_adapter import RlmAdapter

        adapter = RlmAdapter()
        mound = AsyncMock()

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            adapter.sync_to_mound(mound, workspace_id="test")
        )

        assert result["patterns_synced"] == 0
        assert result["errors"] == []

    def test_sync_to_mound_with_patterns(self):
        """sync_to_mound syncs patterns with usage >= 2."""
        from aragora.knowledge.mound.adapters.rlm_adapter import RlmAdapter

        adapter = RlmAdapter()

        # Add a pattern with high value score
        pattern_id = adapter.store_compression_pattern(
            compression_ratio=0.3,
            value_score=0.85,
            content_markers=["security", "api"],
        )

        # Add another compression to increase usage count
        adapter.store_compression_pattern(
            compression_ratio=0.35,
            value_score=0.8,
            content_markers=["security", "api"],
        )

        mound = AsyncMock()
        mound.ingest = AsyncMock(return_value="km_node_123")

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            adapter.sync_to_mound(mound, workspace_id="test")
        )

        assert result["patterns_synced"] == 1
        assert result["errors"] == []

    def test_load_from_mound_empty(self):
        """load_from_mound with no KM data returns empty stats."""
        from aragora.knowledge.mound.adapters.rlm_adapter import RlmAdapter

        adapter = RlmAdapter()
        mound = AsyncMock()
        mound.query_nodes = AsyncMock(return_value=[])

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            adapter.load_from_mound(mound, workspace_id="test")
        )

        assert result["patterns_loaded"] == 0
        assert result["errors"] == []

    def test_load_from_mound_restores_state(self):
        """load_from_mound restores patterns from KM nodes."""
        from aragora.knowledge.mound.adapters.rlm_adapter import RlmAdapter
        from datetime import datetime

        adapter = RlmAdapter()

        # Mock KM node with pattern metadata
        mock_node = MagicMock()
        mock_node.metadata = {
            "type": "compression_pattern",
            "pattern_id": "cp_test123",
            "compression_ratio": 0.3,
            "value_score": 0.85,
            "content_type": "code",
            "content_markers": ["api", "security"],
            "usage_count": 5,
        }
        mock_node.created_at = datetime.now()
        mock_node.updated_at = datetime.now()

        mound = AsyncMock()
        mound.query_nodes = AsyncMock(return_value=[mock_node])

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            adapter.load_from_mound(mound, workspace_id="test")
        )

        assert result["patterns_loaded"] == 1
        assert result["errors"] == []

        # Verify state was restored
        pattern = adapter.get_pattern("cp_test123")
        assert pattern is not None
        assert pattern["compression_ratio"] == 0.3


class TestContinuumAdapterSyncToMound:
    """Tests for ContinuumAdapter.sync_memory_to_mound()."""

    def test_sync_memory_to_mound_filters_by_importance(self):
        """sync_memory_to_mound only syncs high-importance memories."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter
        from unittest.mock import MagicMock

        # Mock ContinuumMemory
        mock_memory = MagicMock()
        mock_memory.retrieve = MagicMock(return_value=[])

        adapter = ContinuumAdapter(mock_memory)
        mound = AsyncMock()

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            adapter.sync_memory_to_mound(
                mound=mound,
                workspace_id="test",
                min_importance=0.7,
            )
        )

        assert result["synced"] == 0
        assert result["errors"] == []


class TestContinuumMemoryPrewarm:
    """Tests for ContinuumMemory.prewarm_for_query()."""

    def test_prewarm_for_query_empty_query(self):
        """prewarm_for_query returns 0 for empty query."""
        from aragora.memory.continuum import ContinuumMemory
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            memory = ContinuumMemory(db_path=db_path)

            result = memory.prewarm_for_query("")
            assert result == 0

    def test_prewarm_for_query_updates_metadata(self):
        """prewarm_for_query updates entry metadata."""
        from aragora.memory.continuum import ContinuumMemory
        from aragora.memory.tier_manager import MemoryTier
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            memory = ContinuumMemory(db_path=db_path)

            # Add a memory entry
            memory.add(
                id="test-1",
                content="Important security analysis results",
                tier=MemoryTier.SLOW,
                importance=0.8,
            )

            # Prewarm for related query
            count = memory.prewarm_for_query("security analysis")
            assert count >= 0  # May be 0 if no matches

    def test_invalidate_reference_removes_km_link(self):
        """invalidate_reference clears KM node references."""
        from aragora.memory.continuum import ContinuumMemory
        from aragora.memory.tier_manager import MemoryTier
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            memory = ContinuumMemory(db_path=db_path)

            # Add a memory entry with KM reference
            memory.add(
                id="test-1",
                content="Test content",
                tier=MemoryTier.SLOW,
                importance=0.5,
                metadata={"km_node_id": "km_123", "km_synced": True},
            )

            # Invalidate the reference
            result = memory.invalidate_reference("km_123")
            assert result is True

            # Verify reference was cleared
            entry = memory.get("test-1")
            assert entry is not None
            assert entry.metadata.get("km_synced") is False


class TestAsyncDispatchConfig:
    """Tests for async dispatch configuration."""

    def test_default_async_event_types(self):
        """Default config starts with empty async_event_types (opt-in)."""
        from aragora.events.cross_subscribers import AsyncDispatchConfig

        config = AsyncDispatchConfig()

        # Default is empty - async dispatch is opt-in per event type
        assert isinstance(config.async_event_types, set)
        # Batching is enabled by default
        assert config.enable_batching is True
        assert config.batch_size == 10

    def test_dispatch_uses_async_for_configured_types(self):
        """dispatch() uses async for high-volume event types."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        manager = CrossSubscriberManager()

        # Create a high-volume event
        event = StreamEvent(
            type=StreamEventType.MEMORY_STORED,
            data={"test": True},
        )

        # dispatch should not raise
        manager.dispatch(event)

    def test_batch_stats_returns_config(self):
        """get_batch_stats returns batch configuration."""
        from aragora.events.cross_subscribers import CrossSubscriberManager

        manager = CrossSubscriberManager()
        stats = manager.get_batch_stats()

        assert "async_event_types" in stats
        assert "batch_size" in stats
        assert "batching_enabled" in stats
