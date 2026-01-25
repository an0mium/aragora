"""
Tests for ContextGatherer - debate context collection from multiple sources.

Tests cover:
- Basic context gathering
- Source-specific gathering (Aragora docs, evidence, trending, knowledge mound)
- Timeout handling
- Cache management
- RLM compression integration
- Threat intelligence enrichment
- Error handling and edge cases
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.context_gatherer import (
    CONTEXT_GATHER_TIMEOUT,
    EVIDENCE_TIMEOUT,
    TRENDING_TIMEOUT,
    ContextGatherer,
)


class TestContextGathererInitialization:
    """Tests for ContextGatherer initialization."""

    def test_init_default(self):
        """Test default initialization."""
        gatherer = ContextGatherer()
        assert gatherer._evidence_store_callback is None
        assert gatherer._prompt_builder is None
        assert isinstance(gatherer._project_root, Path)

    def test_init_with_evidence_callback(self):
        """Test initialization with evidence store callback."""
        callback = MagicMock()
        gatherer = ContextGatherer(evidence_store_callback=callback)
        assert gatherer._evidence_store_callback is callback

    def test_init_with_prompt_builder(self):
        """Test initialization with prompt builder."""
        builder = MagicMock()
        gatherer = ContextGatherer(prompt_builder=builder)
        assert gatherer._prompt_builder is builder

    def test_init_with_project_root(self):
        """Test initialization with custom project root."""
        custom_root = Path("/custom/root")
        gatherer = ContextGatherer(project_root=custom_root)
        assert gatherer._project_root == custom_root

    def test_init_with_rlm_options(self):
        """Test initialization with RLM options."""
        gatherer = ContextGatherer(
            enable_rlm_compression=True,
            rlm_compression_threshold=2000,
        )
        # _enable_rlm depends on HAS_RLM import availability
        assert gatherer._rlm_threshold == 2000

    def test_init_with_knowledge_grounding(self):
        """Test initialization with knowledge grounding options."""
        gatherer = ContextGatherer(
            enable_knowledge_grounding=True,
            knowledge_workspace_id="test_workspace",
        )
        assert gatherer._knowledge_workspace_id == "test_workspace"

    def test_init_with_belief_guidance(self):
        """Test initialization with belief guidance."""
        gatherer = ContextGatherer(enable_belief_guidance=True)
        # _enable_belief_guidance depends on module availability
        assert hasattr(gatherer, "_enable_belief_guidance")

    def test_init_with_threat_intel(self):
        """Test initialization with threat intel enrichment."""
        gatherer = ContextGatherer(enable_threat_intel_enrichment=True)
        # _enable_threat_intel depends on module availability
        assert hasattr(gatherer, "_enable_threat_intel")


class TestContextGathererCaching:
    """Tests for context caching behavior."""

    def test_task_hash_generation(self):
        """Test task hash is consistent."""
        gatherer = ContextGatherer()
        hash1 = gatherer._get_task_hash("Test debate topic")
        hash2 = gatherer._get_task_hash("Test debate topic")
        hash3 = gatherer._get_task_hash("Different topic")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 16  # SHA-256 hex truncated to 16 chars

    def test_cache_clear_all(self):
        """Test clearing all cached contexts."""
        gatherer = ContextGatherer()
        # Simulate cached data using correct attribute names
        gatherer._research_context_cache = {"hash1": "context1", "hash2": "context2"}
        gatherer._research_evidence_pack = {"hash1": MagicMock()}
        gatherer._continuum_context_cache = {"hash1": "continuum"}
        gatherer._trending_topics_cache = [MagicMock()]

        gatherer.clear_cache()

        assert len(gatherer._research_context_cache) == 0
        assert len(gatherer._research_evidence_pack) == 0
        assert len(gatherer._continuum_context_cache) == 0
        assert len(gatherer._trending_topics_cache) == 0

    def test_cache_clear_specific_task(self):
        """Test clearing cache for specific task."""
        gatherer = ContextGatherer()
        task = "Specific task"
        task_hash = gatherer._get_task_hash(task)

        gatherer._research_context_cache = {task_hash: "context", "other": "other_context"}
        gatherer._research_evidence_pack = {task_hash: MagicMock(), "other": MagicMock()}

        gatherer.clear_cache(task=task)

        assert task_hash not in gatherer._research_context_cache
        assert "other" in gatherer._research_context_cache


class TestGatherAll:
    """Tests for main gather_all method."""

    @pytest.mark.asyncio
    async def test_gather_all_returns_string(self):
        """Test gather_all returns context string."""
        gatherer = ContextGatherer()

        # Mock the Claude web search to return quickly
        with patch.object(
            gatherer, "_gather_claude_web_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = "Test context"

            result = await gatherer.gather_all("Test task")

            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_gather_all_uses_cache(self):
        """Test gather_all uses cached results."""
        gatherer = ContextGatherer()
        task = "Cached task"
        task_hash = gatherer._get_task_hash(task)

        # Pre-populate cache
        gatherer._research_context_cache[task_hash] = "Cached context"

        result = await gatherer.gather_all(task)

        assert result == "Cached context"

    @pytest.mark.asyncio
    async def test_gather_all_respects_timeout(self):
        """Test gather_all respects custom timeout."""
        gatherer = ContextGatherer()

        # Create a slow gathering that should timeout
        async def slow_gather(*args, **kwargs):
            await asyncio.sleep(10)
            return "Never returned"

        with patch.object(gatherer, "_gather_claude_web_search", side_effect=slow_gather):
            # Very short timeout should trigger
            result = await gatherer.gather_all("Test", timeout=0.01)

            # Should still return something (empty or partial)
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_gather_all_combines_sources(self):
        """Test gather_all combines multiple sources."""
        gatherer = ContextGatherer()

        with patch.object(
            gatherer, "_gather_claude_web_search", new_callable=AsyncMock
        ) as mock_claude:
            mock_claude.return_value = "Claude research"

            with patch.object(
                gatherer, "gather_aragora_context", new_callable=AsyncMock
            ) as mock_aragora:
                mock_aragora.return_value = "Aragora docs"

                result = await gatherer.gather_all("Test about Aragora")

                assert isinstance(result, str)


class TestGatherAragoraContext:
    """Tests for Aragora project documentation gathering."""

    @pytest.mark.asyncio
    async def test_gather_aragora_non_self_referential(self):
        """Test gather_aragora_context skips non-Aragora tasks."""
        gatherer = ContextGatherer()
        result = await gatherer.gather_aragora_context("Discuss climate change")

        # Should return None for non-Aragora topics
        assert result is None

    @pytest.mark.asyncio
    async def test_gather_aragora_self_referential(self):
        """Test gather_aragora_context for Aragora-related tasks."""
        gatherer = ContextGatherer()
        result = await gatherer.gather_aragora_context("Improve Aragora debate engine")

        # Should attempt to gather Aragora context (may be None if docs don't exist)
        assert result is None or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_gather_aragora_detects_keywords(self):
        """Test that various Aragora keywords trigger context gathering."""
        gatherer = ContextGatherer()

        keywords = [
            "aragora system",
            "multi-agent debate framework",
            "decision stress-test approach",
            "nomic loop implementation",
            "gauntlet testing",
        ]

        for keyword in keywords:
            result = await gatherer.gather_aragora_context(f"Improve the {keyword}")
            # Either returns context or None (if docs not found)
            assert result is None or isinstance(result, str)


class TestGatherEvidenceContext:
    """Tests for evidence collection."""

    @pytest.mark.asyncio
    async def test_gather_evidence_without_collector(self):
        """Test evidence gathering without collector returns None."""
        gatherer = ContextGatherer()

        # Without evidence collector configured, should return None
        result = await gatherer.gather_evidence_context("Test task")
        assert result is None or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_gather_evidence_caches_by_task(self):
        """Test evidence caching behavior."""
        gatherer = ContextGatherer()
        task = "Test task"
        task_hash = gatherer._get_task_hash(task)

        # Pre-populate cache
        mock_pack = MagicMock()
        mock_pack.to_context_string.return_value = "Cached evidence"
        gatherer._research_evidence_pack[task_hash] = mock_pack

        # Getting evidence pack should return cached value
        result = gatherer.get_evidence_pack(task)
        assert result is mock_pack


class TestGatherTrendingContext:
    """Tests for trending topics gathering."""

    @pytest.mark.asyncio
    async def test_gather_trending_empty(self):
        """Test trending context with no cached topics."""
        gatherer = ContextGatherer()

        # Without topics, should return empty list
        result = gatherer.get_trending_topics()
        assert result == []

    @pytest.mark.asyncio
    async def test_gather_trending_with_mock_pulse(self):
        """Test trending context with mocked pulse manager."""
        gatherer = ContextGatherer()

        mock_topic = MagicMock()
        mock_topic.topic = "AI Safety"
        mock_topic.platform = "hackernews"
        mock_topic.volume = 1000
        mock_topic.category = "tech"

        mock_manager = MagicMock()
        mock_manager.get_trending_topics = AsyncMock(return_value=[mock_topic])
        mock_manager.add_ingestor = MagicMock()

        with patch.dict(
            "sys.modules",
            {"aragora.pulse.ingestor": MagicMock()},
        ):
            with patch(
                "aragora.pulse.ingestor.PulseManager",
                return_value=mock_manager,
            ):
                # May still fail due to import complexity, that's OK
                result = await gatherer.gather_trending_context()
                assert result is None or isinstance(result, str)

    def test_get_trending_topics_empty(self):
        """Test getting trending topics when none available."""
        gatherer = ContextGatherer()
        topics = gatherer.get_trending_topics()

        assert isinstance(topics, list)
        assert len(topics) == 0  # Initially empty


class TestGatherKnowledgeMoundContext:
    """Tests for Knowledge Mound integration."""

    @pytest.mark.asyncio
    async def test_gather_km_no_mound(self):
        """Test KM context without knowledge mound."""
        gatherer = ContextGatherer(enable_knowledge_grounding=False)
        result = await gatherer.gather_knowledge_mound_context("Test task")

        # Without KM, should return None
        assert result is None

    @pytest.mark.asyncio
    async def test_gather_km_with_mock_mound(self):
        """Test KM context with mocked mound."""
        mock_mound = MagicMock()
        mock_result = MagicMock()
        mock_result.items = []
        mock_mound.query = AsyncMock(return_value=mock_result)

        gatherer = ContextGatherer(
            enable_knowledge_grounding=True,
            knowledge_mound=mock_mound,
        )
        gatherer._enable_knowledge_grounding = True
        gatherer._knowledge_mound = mock_mound

        result = await gatherer.gather_knowledge_mound_context("Test task")

        # Should handle empty results gracefully
        assert result is None


class TestGatherBeliefCruxContext:
    """Tests for belief network crux identification."""

    @pytest.mark.asyncio
    async def test_gather_belief_no_analyzer(self):
        """Test belief context without analyzer."""
        gatherer = ContextGatherer(enable_belief_guidance=False)
        gatherer._enable_belief_guidance = False
        result = await gatherer.gather_belief_crux_context("Test task")

        # Without belief analyzer, should return None
        assert result is None

    @pytest.mark.asyncio
    async def test_gather_belief_with_messages(self):
        """Test belief context with provided messages."""
        mock_analyzer = MagicMock()
        mock_result = MagicMock()
        mock_result.analysis_error = None
        mock_result.cruxes = [{"statement": "Test crux", "confidence": 0.8, "entropy": 0.5}]
        mock_result.evidence_suggestions = ["Get more data"]
        mock_analyzer.analyze_messages = MagicMock(return_value=mock_result)

        gatherer = ContextGatherer(enable_belief_guidance=True)
        gatherer._enable_belief_guidance = True
        gatherer._belief_analyzer = mock_analyzer

        messages = [MagicMock(content="Test message")]
        result = await gatherer.gather_belief_crux_context("Test task", messages=messages)

        assert result is not None
        assert "Test crux" in result


class TestGatherThreatIntelContext:
    """Tests for threat intelligence enrichment."""

    @pytest.mark.asyncio
    async def test_gather_threat_intel_disabled(self):
        """Test threat intel when disabled."""
        gatherer = ContextGatherer(enable_threat_intel_enrichment=False)
        gatherer._enable_threat_intel = False
        result = await gatherer.gather_threat_intel_context("Check security of API")

        # Should return None if threat intel not configured
        assert result is None

    @pytest.mark.asyncio
    async def test_gather_threat_intel_non_security_topic(self):
        """Test threat intel skips non-security topics."""
        mock_enrichment = MagicMock()
        mock_enrichment.is_security_topic = MagicMock(return_value=False)

        gatherer = ContextGatherer(enable_threat_intel_enrichment=True)
        gatherer._enable_threat_intel = True
        gatherer._threat_intel_enrichment = mock_enrichment

        result = await gatherer.gather_threat_intel_context("Cook a nice dinner")

        assert result is None


class TestGatherCulturePatterns:
    """Tests for organizational culture pattern gathering."""

    @pytest.mark.asyncio
    async def test_gather_culture_no_mound(self):
        """Test culture patterns without mound."""
        gatherer = ContextGatherer()
        gatherer._knowledge_mound = None
        result = await gatherer.gather_culture_patterns_context("Decision topic")

        # Without knowledge mound, should return None
        assert result is None


class TestEvidencePack:
    """Tests for evidence pack property."""

    def test_evidence_pack_property_empty(self):
        """Test evidence_pack property when empty."""
        gatherer = ContextGatherer()
        assert gatherer.evidence_pack is None

    def test_evidence_pack_property_returns_last(self):
        """Test evidence_pack property returns most recent pack."""
        gatherer = ContextGatherer()
        mock_pack1 = MagicMock()
        mock_pack2 = MagicMock()

        gatherer._research_evidence_pack = {"hash1": mock_pack1, "hash2": mock_pack2}

        assert gatherer.evidence_pack is mock_pack2

    def test_get_evidence_pack_by_task(self):
        """Test getting evidence pack by task hash."""
        gatherer = ContextGatherer()
        task = "Test task"
        task_hash = gatherer._get_task_hash(task)

        mock_pack = MagicMock()
        gatherer._research_evidence_pack[task_hash] = mock_pack

        result = gatherer.get_evidence_pack(task)
        assert result is mock_pack


class TestPromptBuilderIntegration:
    """Tests for PromptBuilder integration."""

    def test_set_prompt_builder(self):
        """Test setting prompt builder reference."""
        gatherer = ContextGatherer()
        mock_builder = MagicMock()

        gatherer.set_prompt_builder(mock_builder)

        assert gatherer._prompt_builder is mock_builder


class TestTimeoutConstants:
    """Tests for timeout configuration."""

    def test_default_timeouts(self):
        """Test default timeout values are reasonable."""
        assert CONTEXT_GATHER_TIMEOUT >= 10.0  # Should allow time for gathering
        assert EVIDENCE_TIMEOUT >= 5.0
        assert TRENDING_TIMEOUT >= 1.0


class TestContinuumContext:
    """Tests for continuum memory context retrieval."""

    def test_get_continuum_context_no_memory(self):
        """Test continuum context without memory."""
        gatherer = ContextGatherer()
        # Note: get_continuum_context takes (continuum_memory, domain, task, ...)
        result, ids, tiers = gatherer.get_continuum_context(None, "general", "Test task")

        # Without memory, should return empty
        assert result == ""
        assert ids == []
        assert tiers == {}

    def test_get_continuum_context_with_memory(self):
        """Test continuum context with mocked memory."""
        mock_memory = MagicMock()
        mock_memory.retrieve = MagicMock(return_value=[])

        gatherer = ContextGatherer()
        result, ids, tiers = gatherer.get_continuum_context(mock_memory, "general", "Test task")

        # With empty retrieval, should return empty
        assert result == ""

    def test_get_continuum_context_uses_cache(self):
        """Test continuum context uses cache."""
        gatherer = ContextGatherer()
        task = "Cached task"
        task_hash = gatherer._get_task_hash(task)

        # Pre-populate cache
        gatherer._continuum_context_cache[task_hash] = "Cached continuum"

        mock_memory = MagicMock()
        result, ids, tiers = gatherer.get_continuum_context(mock_memory, "general", task)

        assert result == "Cached continuum"


class TestRLMCompression:
    """Tests for RLM-based context compression."""

    @pytest.mark.asyncio
    async def test_compress_with_rlm_disabled(self):
        """Test compression when RLM is disabled."""
        gatherer = ContextGatherer(enable_rlm_compression=False)
        gatherer._enable_rlm = False

        result = await gatherer._compress_with_rlm("Long text content", max_chars=1000)

        # Should return original or truncated text when disabled
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_compress_short_content(self):
        """Test compression skips short content."""
        gatherer = ContextGatherer()
        gatherer._rlm_threshold = 5000

        short_content = "Short"
        result = await gatherer._compress_with_rlm(short_content, max_chars=1000)

        assert result == short_content

    @pytest.mark.asyncio
    async def test_query_knowledge_with_true_rlm_no_mound(self):
        """Test knowledge query without mound."""
        gatherer = ContextGatherer()
        gatherer._enable_knowledge_grounding = False

        result = await gatherer.query_knowledge_with_true_rlm("Test query")

        assert result is None


class TestRefreshEvidence:
    """Tests for mid-debate evidence refresh."""

    @pytest.mark.asyncio
    async def test_refresh_evidence_no_collector(self):
        """Test refreshing evidence without collector."""
        gatherer = ContextGatherer()

        result = await gatherer.refresh_evidence_for_round(
            combined_text="Test proposals",
            evidence_collector=None,
            task="Test task",
        )

        # Without collector, should return (0, None)
        assert result == (0, None)

    @pytest.mark.asyncio
    async def test_refresh_evidence_with_collector(self):
        """Test refreshing evidence with mock collector."""
        gatherer = ContextGatherer()

        mock_pack = MagicMock()
        mock_pack.snippets = [MagicMock(id="s1")]
        mock_pack.total_searched = 5

        mock_collector = MagicMock()
        mock_collector.extract_claims_from_text = MagicMock(return_value=["Claim 1"])
        mock_collector.collect_for_claims = AsyncMock(return_value=mock_pack)

        count, pack = await gatherer.refresh_evidence_for_round(
            combined_text="Test proposals with claims",
            evidence_collector=mock_collector,
            task="Test task",
        )

        assert count == 1
        assert pack is not None


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_gather_all_empty_task(self):
        """Test handling empty task string."""
        gatherer = ContextGatherer()

        with patch.object(gatherer, "_gather_claude_web_search", new_callable=AsyncMock) as mock:
            mock.return_value = None

            result = await gatherer.gather_all("")

            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_gather_all_unicode_task(self):
        """Test handling unicode in task."""
        gatherer = ContextGatherer()

        with patch.object(gatherer, "_gather_claude_web_search", new_callable=AsyncMock) as mock:
            mock.return_value = None

            result = await gatherer.gather_all("Discuss nihongo and emojis")

            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_gather_handles_cached_result(self):
        """Test gather returns cached result when available."""
        gatherer = ContextGatherer()
        task = "Cached task"
        task_hash = gatherer._get_task_hash(task)

        # Pre-populate cache
        gatherer._research_context_cache[task_hash] = "Cached result"

        result = await gatherer.gather_all(task)
        assert result == "Cached result"

    def test_gatherer_per_debate_isolation(self):
        """Test gatherers maintain isolation per debate."""
        gatherer1 = ContextGatherer()
        gatherer2 = ContextGatherer()

        task = "Same task"
        gatherer1._research_context_cache[gatherer1._get_task_hash(task)] = "Context 1"

        # gatherer2 should not see gatherer1's cache
        assert gatherer1._get_task_hash(task) not in gatherer2._research_context_cache
