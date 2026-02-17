"""Tests for MemoryFabric v2 integration bridges.

Tests cover:
- RLM Supermemory fallback (AragoraRLM bridge)
- Claude-Mem <-> KM sync
- CrossDebate -> Continuum bridge
- RLM <-> KM hierarchy bridge
- Coordinator dedup
- OpenClaw <-> Continuum bridge
- Coordinator RLM backend
"""

from __future__ import annotations

import asyncio
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# TestRLMSupermemoryBridge (~12 tests)
# ---------------------------------------------------------------------------


class TestRLMSupermemoryBridge:
    """Test supermemory_fallback on AragoraRLM."""

    def _make_rlm(self, supermemory_backend=None):
        """Create an AragoraRLM with mocked internals."""
        with patch("aragora.rlm.bridge.HAS_OFFICIAL_RLM", False):
            with patch("aragora.rlm.bridge.HierarchicalCompressor"):
                from aragora.rlm.bridge import AragoraRLM

                rlm = AragoraRLM(supermemory_backend=supermemory_backend)
        return rlm

    def test_init_stores_supermemory_backend(self):
        backend = MagicMock()
        rlm = self._make_rlm(supermemory_backend=backend)
        assert rlm._supermemory_backend is backend

    def test_init_no_supermemory_backend(self):
        rlm = self._make_rlm()
        assert rlm._supermemory_backend is None

    @pytest.mark.asyncio
    async def test_store_pre_compression_calls_store(self):
        backend = MagicMock()
        backend.store = MagicMock()
        rlm = self._make_rlm(supermemory_backend=backend)
        await rlm._store_pre_compression("test content", "text")
        backend.store.assert_called_once_with(
            content="test content",
            metadata={"source_type": "text", "role": "pre_compression_fallback"},
        )

    @pytest.mark.asyncio
    async def test_store_pre_compression_async_store(self):
        backend = MagicMock()
        backend.store = AsyncMock()
        rlm = self._make_rlm(supermemory_backend=backend)
        await rlm._store_pre_compression("test content", "debate")
        backend.store.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_store_pre_compression_no_backend(self):
        rlm = self._make_rlm()
        # Should not raise
        await rlm._store_pre_compression("test content", "text")

    @pytest.mark.asyncio
    async def test_store_pre_compression_error_handled(self):
        backend = MagicMock()
        backend.store = MagicMock(side_effect=RuntimeError("fail"))
        rlm = self._make_rlm(supermemory_backend=backend)
        # Should not raise
        await rlm._store_pre_compression("test content", "text")

    @pytest.mark.asyncio
    async def test_check_supermemory_cache_returns_content(self):
        backend = MagicMock()
        backend.search = MagicMock(return_value=[
            {"content": "cached result", "metadata": {"role": "pre_compression_fallback"}},
        ])
        rlm = self._make_rlm(supermemory_backend=backend)
        result = await rlm._check_supermemory_cache("test query")
        assert result == "cached result"

    @pytest.mark.asyncio
    async def test_check_supermemory_cache_async_search(self):
        backend = MagicMock()
        backend.search = AsyncMock(return_value=[
            {"content": "async cached", "metadata": {"role": "pre_compression_fallback"}},
        ])
        rlm = self._make_rlm(supermemory_backend=backend)
        result = await rlm._check_supermemory_cache("test query")
        assert result == "async cached"

    @pytest.mark.asyncio
    async def test_check_supermemory_cache_no_match(self):
        backend = MagicMock()
        backend.search = MagicMock(return_value=[
            {"content": "other", "metadata": {"role": "normal"}},
        ])
        rlm = self._make_rlm(supermemory_backend=backend)
        result = await rlm._check_supermemory_cache("test query")
        assert result is None

    @pytest.mark.asyncio
    async def test_check_supermemory_cache_no_backend(self):
        rlm = self._make_rlm()
        result = await rlm._check_supermemory_cache("test query")
        assert result is None

    @pytest.mark.asyncio
    async def test_check_supermemory_cache_empty_results(self):
        backend = MagicMock()
        backend.search = MagicMock(return_value=[])
        rlm = self._make_rlm(supermemory_backend=backend)
        result = await rlm._check_supermemory_cache("test query")
        assert result is None

    @pytest.mark.asyncio
    async def test_check_supermemory_cache_error_handled(self):
        backend = MagicMock()
        backend.search = MagicMock(side_effect=ValueError("fail"))
        rlm = self._make_rlm(supermemory_backend=backend)
        result = await rlm._check_supermemory_cache("test query")
        assert result is None


# ---------------------------------------------------------------------------
# TestClaudeMemKMSync (~12 tests)
# ---------------------------------------------------------------------------


class TestClaudeMemKMSync:
    """Test Claude-Mem <-> Knowledge Mound synchronization."""

    def _make_sync(self, claude_mem=None, km=None, min_surprise=0.0):
        from aragora.memory.claude_mem_km_sync import ClaudeMemKMSync

        return ClaudeMemKMSync(
            claude_mem_backend=claude_mem or MagicMock(),
            km_backend=km or MagicMock(),
            min_surprise=min_surprise,
        )

    @pytest.mark.asyncio
    async def test_sync_happy_path(self):
        cm = MagicMock()
        cm.search = MagicMock(return_value=[
            {"id": "1", "content": "important insight about architecture"},
        ])
        km = MagicMock()
        km.store_knowledge = MagicMock()
        sync = self._make_sync(claude_mem=cm, km=km, min_surprise=0.0)
        result = await sync.sync()
        assert result.synced_count == 1
        assert result.errors == 0
        km.store_knowledge.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_surprise_filtering(self):
        cm = MagicMock()
        cm.search = MagicMock(return_value=[
            {"id": "1", "content": "a"},  # very short, low surprise
        ])
        km = MagicMock()
        km.store_knowledge = MagicMock()
        sync = self._make_sync(claude_mem=cm, km=km, min_surprise=0.99)
        result = await sync.sync()
        assert result.synced_count == 0
        assert result.skipped_count == 1

    @pytest.mark.asyncio
    async def test_sync_dedup_by_id(self):
        cm = MagicMock()
        cm.search = MagicMock(return_value=[
            {"id": "1", "content": "unique insight about novel architecture"},
        ])
        km = MagicMock()
        km.store_knowledge = MagicMock()
        sync = self._make_sync(claude_mem=cm, km=km, min_surprise=0.0)
        # Sync once
        await sync.sync()
        # Sync again - should skip duplicate
        result = await sync.sync()
        assert result.skipped_count >= 1

    @pytest.mark.asyncio
    async def test_sync_empty_content_skipped(self):
        cm = MagicMock()
        cm.search = MagicMock(return_value=[
            {"id": "1", "content": ""},
        ])
        km = MagicMock()
        sync = self._make_sync(claude_mem=cm, km=km, min_surprise=0.0)
        result = await sync.sync()
        assert result.skipped_count == 1
        assert result.synced_count == 0

    @pytest.mark.asyncio
    async def test_sync_km_write_error(self):
        cm = MagicMock()
        cm.search = MagicMock(return_value=[
            {"id": "1", "content": "novel insight about architecture patterns"},
        ])
        km = MagicMock()
        km.store_knowledge = MagicMock(side_effect=RuntimeError("write failed"))
        sync = self._make_sync(claude_mem=cm, km=km, min_surprise=0.0)
        result = await sync.sync()
        assert result.errors == 1

    @pytest.mark.asyncio
    async def test_sync_claude_mem_error(self):
        cm = MagicMock()
        cm.search = MagicMock(side_effect=RuntimeError("search failed"))
        sync = self._make_sync(claude_mem=cm, min_surprise=0.0)
        result = await sync.sync()
        assert result.errors == 1
        assert result.synced_count == 0

    @pytest.mark.asyncio
    async def test_sync_no_search_method(self):
        cm = MagicMock(spec=[])  # No search method
        sync = self._make_sync(claude_mem=cm, min_surprise=0.0)
        result = await sync.sync()
        assert result.synced_count == 0

    @pytest.mark.asyncio
    async def test_sync_async_backends(self):
        cm = MagicMock()
        cm.search = AsyncMock(return_value=[
            {"id": "a1", "content": "asynchronous novel insight about patterns"},
        ])
        km = MagicMock()
        km.store_knowledge = AsyncMock()
        sync = self._make_sync(claude_mem=cm, km=km, min_surprise=0.0)
        result = await sync.sync()
        assert result.synced_count == 1
        km.store_knowledge.assert_awaited_once()

    def test_reset_sync_tracking(self):
        sync = self._make_sync()
        sync._synced_ids.add("test")
        sync.reset_sync_tracking()
        assert len(sync._synced_ids) == 0

    @pytest.mark.asyncio
    async def test_sync_none_results(self):
        cm = MagicMock()
        cm.search = MagicMock(return_value=None)
        sync = self._make_sync(claude_mem=cm, min_surprise=0.0)
        result = await sync.sync()
        assert result.synced_count == 0

    @pytest.mark.asyncio
    async def test_sync_multiple_items(self):
        cm = MagicMock()
        cm.search = MagicMock(return_value=[
            {"id": "1", "content": "first novel insight about architecture"},
            {"id": "2", "content": "second novel insight about design patterns"},
            {"id": "3", "content": "third novel insight about testing strategies"},
        ])
        km = MagicMock()
        km.store_knowledge = MagicMock()
        sync = self._make_sync(claude_mem=cm, km=km, min_surprise=0.0)
        result = await sync.sync()
        assert result.synced_count == 3


# ---------------------------------------------------------------------------
# TestCrossDebateContinuumBridge (~12 tests)
# ---------------------------------------------------------------------------


class TestCrossDebateContinuumBridge:
    """Test CrossDebateMemory -> ContinuumMemory bridge."""

    def _make_memory(self, continuum=None):
        from aragora.memory.cross_debate_rlm import CrossDebateMemory, CrossDebateConfig

        config = CrossDebateConfig(persist_to_disk=False, enable_rlm=False)
        return CrossDebateMemory(config=config, continuum_memory=continuum)

    @pytest.mark.asyncio
    async def test_add_debate_writes_to_continuum(self):
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="entry_1")
        mem = self._make_memory(continuum=continuum)

        result = MagicMock()
        result.debate_id = "d1"
        result.task = "Test task"
        result.domain = "general"
        result.participants = ["agent1"]
        result.consensus_reached = True
        result.final_answer = "The answer is 42"
        result.critiques = []
        result.messages = []

        await mem.add_debate(result)
        continuum.store_pattern.assert_called_once()
        call_kwargs = continuum.store_pattern.call_args[1]
        assert "Debate conclusion" in call_kwargs["content"]
        assert call_kwargs["importance"] == 0.6  # consensus_reached=True

    @pytest.mark.asyncio
    async def test_add_debate_no_consensus_lower_importance(self):
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="entry_1")
        mem = self._make_memory(continuum=continuum)

        result = MagicMock()
        result.debate_id = "d2"
        result.task = "Test task"
        result.domain = "general"
        result.participants = []
        result.consensus_reached = False
        result.final_answer = "No consensus"
        result.critiques = []
        result.messages = []

        await mem.add_debate(result)
        call_kwargs = continuum.store_pattern.call_args[1]
        assert call_kwargs["importance"] == 0.3

    @pytest.mark.asyncio
    async def test_add_debate_no_continuum_no_error(self):
        mem = self._make_memory(continuum=None)
        result = MagicMock()
        result.debate_id = "d3"
        result.task = "Test"
        result.domain = "general"
        result.participants = []
        result.consensus_reached = True
        result.final_answer = "Answer"
        result.critiques = []
        result.messages = []
        # Should not raise
        await mem.add_debate(result)

    @pytest.mark.asyncio
    async def test_add_debate_continuum_error_handled(self):
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(side_effect=RuntimeError("fail"))
        mem = self._make_memory(continuum=continuum)

        result = MagicMock()
        result.debate_id = "d4"
        result.task = "Test"
        result.domain = "general"
        result.participants = []
        result.consensus_reached = True
        result.final_answer = "Answer"
        result.critiques = []
        result.messages = []
        # Should not raise
        debate_id = await mem.add_debate(result)
        assert debate_id is not None

    @pytest.mark.asyncio
    async def test_store_debate_writes_to_continuum(self):
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="entry_2")
        mem = self._make_memory(continuum=continuum)

        await mem.store_debate(
            debate_id="sd1",
            topic="API Design",
            consensus="Use REST",
            domain="engineering",
        )
        continuum.store_pattern.assert_called_once()
        call_kwargs = continuum.store_pattern.call_args[1]
        assert "API Design" not in call_kwargs["content"] or "Use REST" in call_kwargs["content"]
        assert call_kwargs["metadata"]["domain"] == "engineering"

    @pytest.mark.asyncio
    async def test_store_debate_no_continuum_no_error(self):
        mem = self._make_memory(continuum=None)
        await mem.store_debate(debate_id="sd2", topic="Test", consensus="OK")

    @pytest.mark.asyncio
    async def test_store_debate_uses_add_fallback(self):
        continuum = MagicMock(spec=[])
        continuum.add = MagicMock(return_value=MagicMock(id="entry_3"))
        mem = self._make_memory(continuum=continuum)
        await mem.store_debate(debate_id="sd3", topic="Test", consensus="OK")
        continuum.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_debate_uses_add_fallback(self):
        continuum = MagicMock(spec=[])
        continuum.add = MagicMock(return_value=MagicMock(id="entry_4"))
        mem = self._make_memory(continuum=continuum)

        result = MagicMock()
        result.debate_id = "d5"
        result.task = "Test"
        result.domain = "general"
        result.participants = []
        result.consensus_reached = True
        result.final_answer = "Answer"
        result.critiques = []
        result.messages = []
        await mem.add_debate(result)
        continuum.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_debate_metadata_includes_source(self):
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="e1")
        mem = self._make_memory(continuum=continuum)
        await mem.store_debate(debate_id="sd4", topic="Test", consensus="OK", domain="legal")
        meta = continuum.store_pattern.call_args[1]["metadata"]
        assert meta["source"] == "cross_debate_bridge"
        assert meta["tier_hint"] == "slow"

    @pytest.mark.asyncio
    async def test_add_debate_metadata_includes_debate_id(self):
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="e2")
        mem = self._make_memory(continuum=continuum)

        result = MagicMock()
        result.debate_id = "d6"
        result.task = "Test"
        result.domain = "general"
        result.participants = []
        result.consensus_reached = True
        result.final_answer = "Answer"
        result.critiques = []
        result.messages = []
        await mem.add_debate(result)
        meta = continuum.store_pattern.call_args[1]["metadata"]
        assert "debate_id" in meta

    @pytest.mark.asyncio
    async def test_store_debate_continuum_error_handled(self):
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(side_effect=ValueError("write error"))
        mem = self._make_memory(continuum=continuum)
        # Should not raise
        debate_id = await mem.store_debate(debate_id="sd5", topic="Test", consensus="OK")
        assert debate_id == "sd5"

    @pytest.mark.asyncio
    async def test_store_debate_domain_in_content(self):
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="e3")
        mem = self._make_memory(continuum=continuum)
        await mem.store_debate(debate_id="sd6", topic="API", consensus="REST", domain="tech")
        content = continuum.store_pattern.call_args[1]["content"]
        assert "tech" in content


# ---------------------------------------------------------------------------
# TestRLMKMBridge (~12 tests)
# ---------------------------------------------------------------------------


class TestRLMKMBridge:
    """Test RLM Hierarchy <-> KM Graph bridge."""

    def _make_bridge(self, km=None):
        from aragora.memory.rlm_km_bridge import RLMKMBridge

        return RLMKMBridge(km_backend=km or MagicMock())

    @pytest.mark.asyncio
    async def test_map_hierarchy_creates_two_nodes(self):
        km = MagicMock()
        km.store_knowledge = MagicMock()
        km.add_relationship = MagicMock()
        bridge = self._make_bridge(km=km)

        result = await bridge.map_hierarchy("c1", "full text", "summary text")
        assert result.nodes_created == 2
        assert result.errors == 0

    @pytest.mark.asyncio
    async def test_map_hierarchy_creates_relationship(self):
        km = MagicMock()
        km.store_knowledge = MagicMock()
        km.add_relationship = MagicMock()
        bridge = self._make_bridge(km=km)

        result = await bridge.map_hierarchy("c2", "full", "summary")
        assert result.relationships_created == 1
        km.add_relationship.assert_called_once_with(
            source_id="rlm_summary_c2",
            target_id="rlm_full_c2",
            relationship_type="summarizes",
        )

    @pytest.mark.asyncio
    async def test_map_hierarchy_async_store(self):
        km = MagicMock()
        km.store_knowledge = AsyncMock()
        km.add_relationship = AsyncMock()
        bridge = self._make_bridge(km=km)

        result = await bridge.map_hierarchy("c3", "full", "summary")
        assert result.nodes_created == 2
        assert result.relationships_created == 1

    @pytest.mark.asyncio
    async def test_map_hierarchy_no_store_method(self):
        km = MagicMock(spec=[])  # No store_knowledge
        bridge = self._make_bridge(km=km)
        result = await bridge.map_hierarchy("c4", "full", "summary")
        assert result.nodes_created == 0

    @pytest.mark.asyncio
    async def test_map_hierarchy_no_relationship_method(self):
        km = MagicMock(spec=[])
        km.store_knowledge = MagicMock()
        bridge = self._make_bridge(km=km)
        result = await bridge.map_hierarchy("c5", "full", "summary")
        assert result.nodes_created == 2
        assert result.relationships_created == 0

    @pytest.mark.asyncio
    async def test_map_hierarchy_error_handled(self):
        km = MagicMock()
        km.store_knowledge = MagicMock(side_effect=RuntimeError("fail"))
        bridge = self._make_bridge(km=km)
        result = await bridge.map_hierarchy("c6", "full", "summary")
        assert result.errors == 1

    @pytest.mark.asyncio
    async def test_map_hierarchy_full_node_metadata(self):
        km = MagicMock()
        km.store_knowledge = MagicMock()
        bridge = self._make_bridge(km=km)
        await bridge.map_hierarchy("c7", "full text", "summary text")

        calls = km.store_knowledge.call_args_list
        full_call = calls[0]
        assert full_call[1]["source_id"] == "rlm_full_c7"
        assert full_call[1]["confidence"] == 1.0
        assert full_call[1]["metadata"]["level"] == "full"

    @pytest.mark.asyncio
    async def test_map_hierarchy_summary_node_metadata(self):
        km = MagicMock()
        km.store_knowledge = MagicMock()
        bridge = self._make_bridge(km=km)
        await bridge.map_hierarchy("c8", "full text", "summary text")

        calls = km.store_knowledge.call_args_list
        summary_call = calls[1]
        assert summary_call[1]["source_id"] == "rlm_summary_c8"
        assert summary_call[1]["confidence"] == 0.9
        assert summary_call[1]["metadata"]["level"] == "summary"
        assert summary_call[1]["metadata"]["summarizes"] == "rlm_full_c8"

    @pytest.mark.asyncio
    async def test_map_hierarchy_custom_source_type(self):
        km = MagicMock()
        km.store_knowledge = MagicMock()
        bridge = self._make_bridge(km=km)
        await bridge.map_hierarchy("c9", "full", "summary", source_type="debate_context")
        assert km.store_knowledge.call_args_list[0][1]["source"] == "debate_context"

    @pytest.mark.asyncio
    async def test_map_hierarchy_content_passed_correctly(self):
        km = MagicMock()
        km.store_knowledge = MagicMock()
        bridge = self._make_bridge(km=km)
        await bridge.map_hierarchy("c10", "the full content", "the summary")
        calls = km.store_knowledge.call_args_list
        assert calls[0][1]["content"] == "the full content"
        assert calls[1][1]["content"] == "the summary"

    @pytest.mark.asyncio
    async def test_map_hierarchy_content_id_in_metadata(self):
        km = MagicMock()
        km.store_knowledge = MagicMock()
        bridge = self._make_bridge(km=km)
        await bridge.map_hierarchy("c11", "full", "summary")
        calls = km.store_knowledge.call_args_list
        assert calls[0][1]["metadata"]["content_id"] == "c11"
        assert calls[1][1]["metadata"]["content_id"] == "c11"

    @pytest.mark.asyncio
    async def test_map_hierarchy_returns_bridge_result(self):
        from aragora.memory.rlm_km_bridge import BridgeResult

        km = MagicMock()
        km.store_knowledge = MagicMock()
        km.add_relationship = MagicMock()
        bridge = self._make_bridge(km=km)
        result = await bridge.map_hierarchy("c12", "full", "summary")
        assert isinstance(result, BridgeResult)


# ---------------------------------------------------------------------------
# TestCoordinatorDedup (~14 tests)
# ---------------------------------------------------------------------------


class TestCoordinatorDedup:
    """Test content hash dedup mechanism in MemoryCoordinator."""

    def _make_coordinator(self, **kwargs):
        from aragora.memory.coordinator import MemoryCoordinator

        return MemoryCoordinator(**kwargs)

    def test_check_dedup_first_write_not_duplicate(self):
        coord = self._make_coordinator()
        assert coord._check_dedup("hello world", "continuum") is False

    def test_check_dedup_after_record_is_duplicate(self):
        coord = self._make_coordinator()
        coord._record_dedup("hello world", "continuum")
        assert coord._check_dedup("hello world", "continuum") is True

    def test_check_dedup_without_record_not_duplicate(self):
        coord = self._make_coordinator()
        coord._check_dedup("hello world", "continuum")
        # Without _record_dedup, second check should NOT be a duplicate
        assert coord._check_dedup("hello world", "continuum") is False

    def test_check_dedup_different_content_not_duplicate(self):
        coord = self._make_coordinator()
        coord._record_dedup("hello world", "continuum")
        assert coord._check_dedup("different content", "continuum") is False

    def test_check_dedup_same_content_different_target_not_duplicate(self):
        coord = self._make_coordinator()
        coord._record_dedup("hello world", "continuum")
        assert coord._check_dedup("hello world", "consensus") is False

    def test_check_dedup_stats_checked(self):
        coord = self._make_coordinator()
        coord._check_dedup("a", "continuum")
        coord._check_dedup("b", "continuum")
        assert coord._dedup_stats["checked"] == 2

    def test_check_dedup_stats_skipped(self):
        coord = self._make_coordinator()
        coord._record_dedup("a", "continuum")
        coord._check_dedup("a", "continuum")
        assert coord._dedup_stats["skipped"] == 1

    def test_record_dedup_uses_sha256(self):
        coord = self._make_coordinator()
        content = "test content"
        expected_hash = hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()[:16]
        coord._record_dedup(content, "continuum")
        assert expected_hash in coord._content_hashes["continuum"]

    def test_record_dedup_multiple_targets(self):
        coord = self._make_coordinator()
        coord._record_dedup("content", "continuum")
        coord._record_dedup("content", "mound")
        assert len(coord._content_hashes) == 2

    def test_check_dedup_empty_content(self):
        coord = self._make_coordinator()
        assert coord._check_dedup("", "continuum") is False
        coord._record_dedup("", "continuum")
        assert coord._check_dedup("", "continuum") is True

    def test_check_dedup_unicode_content(self):
        coord = self._make_coordinator()
        assert coord._check_dedup("unicôde cöntent", "continuum") is False
        coord._record_dedup("unicôde cöntent", "continuum")
        assert coord._check_dedup("unicôde cöntent", "continuum") is True

    def test_content_hashes_initialized_empty(self):
        coord = self._make_coordinator()
        assert coord._content_hashes == {}

    def test_dedup_stats_initialized_zero(self):
        coord = self._make_coordinator()
        assert coord._dedup_stats == {"checked": 0, "skipped": 0}

    @pytest.mark.asyncio
    async def test_write_continuum_dedup_skips(self):
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="e1")
        coord = self._make_coordinator(continuum_memory=continuum)

        data = {
            "debate_id": "d1",
            "task": "test",
            "final_answer": "answer",
            "confidence": 0.8,
            "domain": "general",
            "consensus_reached": True,
        }
        # First write
        result1 = await coord._write_continuum(data)
        # Second write (same content) should be dedup'd
        result2 = await coord._write_continuum(data)
        assert result2.startswith("dedup_")
        # store_pattern should only be called once
        assert continuum.store_pattern.call_count == 1

    @pytest.mark.asyncio
    async def test_write_continuum_different_content_not_deduped(self):
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="e1")
        coord = self._make_coordinator(continuum_memory=continuum)

        data1 = {
            "debate_id": "d1",
            "task": "test1",
            "final_answer": "answer1",
            "confidence": 0.8,
            "domain": "general",
            "consensus_reached": True,
        }
        data2 = {
            "debate_id": "d2",
            "task": "test2",
            "final_answer": "answer2",
            "confidence": 0.9,
            "domain": "legal",
            "consensus_reached": False,
        }
        await coord._write_continuum(data1)
        await coord._write_continuum(data2)
        assert continuum.store_pattern.call_count == 2


# ---------------------------------------------------------------------------
# TestOpenClawContinuumBridge (~12 tests)
# ---------------------------------------------------------------------------


class TestOpenClawContinuumBridge:
    """Test OpenClaw <-> Continuum Memory bridge."""

    def _make_bridge(self, continuum=None):
        from aragora.memory.openclaw_bridge import OpenClawContinuumBridge

        return OpenClawContinuumBridge(continuum_memory=continuum or MagicMock())

    def _make_event(self, agent_id="agent-1", status="pass", details="verified"):
        from aragora.memory.openclaw_bridge import ValidationEvent

        return ValidationEvent(agent_id=agent_id, status=status, details=details)

    def test_record_pass_event(self):
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="e1")
        bridge = self._make_bridge(continuum=continuum)

        result = bridge.record_validation(self._make_event(status="pass"))
        assert result == "e1"
        call_kwargs = continuum.store_pattern.call_args[1]
        assert call_kwargs["importance"] == 0.4

    def test_record_fail_event(self):
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="e2")
        bridge = self._make_bridge(continuum=continuum)

        result = bridge.record_validation(self._make_event(status="fail"))
        assert result == "e2"
        call_kwargs = continuum.store_pattern.call_args[1]
        assert call_kwargs["importance"] == 0.7

    def test_record_revoke_event(self):
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="e3")
        bridge = self._make_bridge(continuum=continuum)

        result = bridge.record_validation(self._make_event(status="revoke"))
        call_kwargs = continuum.store_pattern.call_args[1]
        assert call_kwargs["importance"] == 0.9

    def test_record_unknown_status_default_importance(self):
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="e4")
        bridge = self._make_bridge(continuum=continuum)

        result = bridge.record_validation(self._make_event(status="unknown"))
        call_kwargs = continuum.store_pattern.call_args[1]
        assert call_kwargs["importance"] == 0.5

    def test_event_count_increments(self):
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="e1")
        bridge = self._make_bridge(continuum=continuum)

        bridge.record_validation(self._make_event())
        bridge.record_validation(self._make_event())
        assert bridge.get_event_count() == 2

    def test_event_count_starts_at_zero(self):
        bridge = self._make_bridge()
        assert bridge.get_event_count() == 0

    def test_record_error_returns_none(self):
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(side_effect=RuntimeError("fail"))
        bridge = self._make_bridge(continuum=continuum)

        result = bridge.record_validation(self._make_event())
        assert result is None

    def test_record_uses_add_fallback(self):
        continuum = MagicMock(spec=[])
        continuum.add = MagicMock(return_value=MagicMock(id="add_e1"))
        bridge = self._make_bridge(continuum=continuum)

        result = bridge.record_validation(self._make_event())
        assert result == "add_e1"
        continuum.add.assert_called_once()

    def test_record_content_format(self):
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="e1")
        bridge = self._make_bridge(continuum=continuum)

        bridge.record_validation(self._make_event(agent_id="a1", status="fail", details="bad cert"))
        content = continuum.store_pattern.call_args[1]["content"]
        assert "OpenClaw validation [fail]" in content
        assert "agent=a1" in content
        assert "bad cert" in content

    def test_record_metadata_includes_source(self):
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="e1")
        bridge = self._make_bridge(continuum=continuum)

        bridge.record_validation(self._make_event())
        meta = continuum.store_pattern.call_args[1]["metadata"]
        assert meta["source"] == "openclaw_bridge"
        assert meta["tier_hint"] == "medium"

    def test_record_metadata_includes_agent_id(self):
        continuum = MagicMock()
        continuum.store_pattern = MagicMock(return_value="e1")
        bridge = self._make_bridge(continuum=continuum)

        bridge.record_validation(self._make_event(agent_id="agent-xyz"))
        meta = continuum.store_pattern.call_args[1]["metadata"]
        assert meta["agent_id"] == "agent-xyz"

    def test_no_methods_returns_none(self):
        continuum = MagicMock(spec=[])
        bridge = self._make_bridge(continuum=continuum)
        result = bridge.record_validation(self._make_event())
        assert result is None


# ---------------------------------------------------------------------------
# TestCoordinatorRLMBackend (~14 tests)
# ---------------------------------------------------------------------------


class TestCoordinatorRLMBackend:
    """Test RLM backend integration in MemoryCoordinator."""

    def _make_coordinator(self, rlm_backend=None, **kwargs):
        from aragora.memory.coordinator import MemoryCoordinator, CoordinatorOptions

        opts = CoordinatorOptions(
            write_continuum=False,
            write_consensus=False,
            write_critique=False,
            write_mound=False,
            write_rlm=True,
        )
        return MemoryCoordinator(rlm_backend=rlm_backend, options=opts, **kwargs)

    def test_rlm_backend_stored(self):
        backend = MagicMock()
        coord = self._make_coordinator(rlm_backend=backend)
        assert coord.rlm_backend is backend

    def test_rlm_backend_none_by_default(self):
        from aragora.memory.coordinator import MemoryCoordinator

        coord = MemoryCoordinator()
        assert coord.rlm_backend is None

    @pytest.mark.asyncio
    async def test_write_rlm_calls_build_context(self):
        backend = AsyncMock()
        backend.build_context = AsyncMock(return_value=MagicMock())
        coord = self._make_coordinator(rlm_backend=backend)

        result = await coord._write_rlm({
            "debate_id": "d1",
            "task": "test task",
            "conclusion": "test conclusion",
        })
        assert result == "rlm_d1"
        backend.build_context.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_write_rlm_no_backend_raises(self):
        from aragora.memory.coordinator import MemoryCoordinator

        coord = MemoryCoordinator()
        with pytest.raises(ValueError, match="RLM backend not configured"):
            await coord._write_rlm({"debate_id": "d1", "task": "test"})

    @pytest.mark.asyncio
    async def test_write_rlm_no_build_context_returns_none(self):
        backend = MagicMock(spec=[])
        coord = self._make_coordinator(rlm_backend=backend)
        result = await coord._write_rlm({"debate_id": "d1", "task": "test"})
        assert result is None

    @pytest.mark.asyncio
    async def test_write_rlm_content_format(self):
        backend = AsyncMock()
        backend.build_context = AsyncMock(return_value=MagicMock())
        coord = self._make_coordinator(rlm_backend=backend)

        await coord._write_rlm({
            "debate_id": "d1",
            "task": "my task",
            "conclusion": "my conclusion",
        })
        call_args = backend.build_context.call_args
        content = call_args[0][0]
        assert "my task" in content
        assert "my conclusion" in content

    @pytest.mark.asyncio
    async def test_write_rlm_uses_final_answer_fallback(self):
        backend = AsyncMock()
        backend.build_context = AsyncMock(return_value=MagicMock())
        coord = self._make_coordinator(rlm_backend=backend)

        await coord._write_rlm({
            "debate_id": "d1",
            "task": "task",
            "final_answer": "the answer",
        })
        call_args = backend.build_context.call_args
        content = call_args[0][0]
        assert "the answer" in content

    @pytest.mark.asyncio
    async def test_write_rlm_source_type_coordinator(self):
        backend = AsyncMock()
        backend.build_context = AsyncMock(return_value=MagicMock())
        coord = self._make_coordinator(rlm_backend=backend)

        await coord._write_rlm({"debate_id": "d1", "task": "test"})
        call_args = backend.build_context.call_args
        assert call_args[1]["source_type"] == "coordinator"

    def test_write_rlm_option_default_false(self):
        from aragora.memory.coordinator import CoordinatorOptions

        opts = CoordinatorOptions()
        assert opts.write_rlm is False

    def test_write_rlm_option_can_be_set(self):
        from aragora.memory.coordinator import CoordinatorOptions

        opts = CoordinatorOptions(write_rlm=True)
        assert opts.write_rlm is True

    @pytest.mark.asyncio
    async def test_execute_operation_routes_to_write_rlm(self):
        from aragora.memory.coordinator import WriteOperation, WriteStatus, CoordinatorOptions

        backend = AsyncMock()
        backend.build_context = AsyncMock(return_value=MagicMock())
        coord = self._make_coordinator(rlm_backend=backend)

        op = WriteOperation(
            id="op1",
            target="rlm",
            data={"debate_id": "d1", "task": "test", "conclusion": "ok"},
        )
        opts = CoordinatorOptions(write_rlm=True)
        await coord._execute_operation(op, opts)
        assert op.status == WriteStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_execute_operation_rlm_failure(self):
        from aragora.memory.coordinator import WriteOperation, WriteStatus, CoordinatorOptions

        backend = MagicMock()
        backend.build_context = MagicMock(side_effect=RuntimeError("fail"))
        coord = self._make_coordinator(rlm_backend=backend)

        op = WriteOperation(
            id="op2",
            target="rlm",
            data={"debate_id": "d1", "task": "test"},
        )
        opts = CoordinatorOptions(write_rlm=True, max_retries=0)
        await coord._execute_operation(op, opts)
        assert op.status == WriteStatus.FAILED

    @pytest.mark.asyncio
    async def test_build_operations_includes_rlm(self):
        from aragora.memory.coordinator import CoordinatorOptions

        backend = MagicMock()
        backend.build_context = MagicMock()
        coord = self._make_coordinator(rlm_backend=backend)

        # Create minimal mocks for ctx and result
        ctx = MagicMock()
        ctx.debate_id = "d1"
        ctx.env.task = "test task"
        ctx.domain = "general"
        ctx.agents = []

        result = MagicMock()
        result.final_answer = "answer"
        result.confidence = 0.8
        result.consensus_reached = True
        result.winner = None
        result.rounds_used = 3
        result.key_claims = []

        opts = CoordinatorOptions(
            write_continuum=False,
            write_consensus=False,
            write_critique=False,
            write_mound=False,
            write_rlm=True,
        )
        operations, skipped = coord._build_operations(ctx, result, opts)
        targets = [op.target for op in operations]
        assert "rlm" in targets

    @pytest.mark.asyncio
    async def test_build_operations_excludes_rlm_when_disabled(self):
        from aragora.memory.coordinator import CoordinatorOptions

        backend = MagicMock()
        coord = self._make_coordinator(rlm_backend=backend)

        ctx = MagicMock()
        ctx.debate_id = "d1"
        ctx.env.task = "test"
        ctx.domain = "general"
        ctx.agents = []

        result = MagicMock()
        result.final_answer = "answer"
        result.confidence = 0.8
        result.consensus_reached = True
        result.winner = None
        result.rounds_used = 3
        result.key_claims = []

        opts = CoordinatorOptions(
            write_continuum=False,
            write_consensus=False,
            write_critique=False,
            write_mound=False,
            write_rlm=False,
        )
        operations, skipped = coord._build_operations(ctx, result, opts)
        targets = [op.target for op in operations]
        assert "rlm" not in targets
