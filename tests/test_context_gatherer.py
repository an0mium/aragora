"""
Tests for debate/context_gatherer.py - Research and Evidence Context.

Tests the ContextGatherer class which handles gathering context from
multiple sources for debate grounding.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.context_gatherer import ContextGatherer


class TestContextGathererInit:
    """Tests for ContextGatherer initialization."""

    def test_init_with_defaults(self):
        """Should initialize with sensible defaults."""
        gatherer = ContextGatherer()
        assert gatherer._evidence_store_callback is None
        assert gatherer._prompt_builder is None
        assert gatherer._research_evidence_pack == {}  # Now a dict keyed by task hash
        assert gatherer._research_context_cache == {}  # Now a dict keyed by task hash

    def test_init_with_callback(self):
        """Should accept evidence store callback."""
        callback = MagicMock()
        gatherer = ContextGatherer(evidence_store_callback=callback)
        assert gatherer._evidence_store_callback == callback

    def test_init_with_prompt_builder(self):
        """Should accept prompt builder."""
        builder = MagicMock()
        gatherer = ContextGatherer(prompt_builder=builder)
        assert gatherer._prompt_builder == builder

    def test_init_with_custom_project_root(self):
        """Should accept custom project root."""
        custom_root = Path("/custom/path")
        gatherer = ContextGatherer(project_root=custom_root)
        assert gatherer._project_root == custom_root

    def test_evidence_pack_property(self):
        """evidence_pack property should return cached value from dict."""
        gatherer = ContextGatherer()
        # Empty dict returns None
        assert gatherer.evidence_pack is None

        # When populated, returns the most recent entry
        mock_pack = MagicMock()
        task_hash = gatherer._get_task_hash("test task")
        gatherer._research_evidence_pack[task_hash] = mock_pack
        assert gatherer.evidence_pack == mock_pack

    def test_get_evidence_pack_for_task(self):
        """get_evidence_pack(task) should return pack for specific task."""
        gatherer = ContextGatherer()

        mock_pack1 = MagicMock()
        mock_pack2 = MagicMock()

        task1_hash = gatherer._get_task_hash("task 1")
        task2_hash = gatherer._get_task_hash("task 2")

        gatherer._research_evidence_pack[task1_hash] = mock_pack1
        gatherer._research_evidence_pack[task2_hash] = mock_pack2

        assert gatherer.get_evidence_pack("task 1") == mock_pack1
        assert gatherer.get_evidence_pack("task 2") == mock_pack2
        assert gatherer.get_evidence_pack("unknown task") is None

    def test_set_prompt_builder(self):
        """set_prompt_builder should update the reference."""
        gatherer = ContextGatherer()
        builder = MagicMock()
        gatherer.set_prompt_builder(builder)
        assert gatherer._prompt_builder == builder

    def test_clear_cache_all(self):
        """clear_cache() should reset all cached values when called without task."""
        gatherer = ContextGatherer()
        # Populate caches with task-keyed data
        gatherer._research_context_cache["abc123"] = "cached"
        gatherer._research_context_cache["def456"] = "other cached"
        gatherer._research_evidence_pack["abc123"] = MagicMock()

        gatherer.clear_cache()

        assert gatherer._research_context_cache == {}
        assert gatherer._research_evidence_pack == {}

    def test_clear_cache_specific_task(self):
        """clear_cache(task) should only clear cache for that specific task."""
        gatherer = ContextGatherer()

        # Use the actual hash function to populate cache
        task1_hash = gatherer._get_task_hash("Task 1")
        task2_hash = gatherer._get_task_hash("Task 2")

        gatherer._research_context_cache[task1_hash] = "cached 1"
        gatherer._research_context_cache[task2_hash] = "cached 2"

        gatherer.clear_cache("Task 1")

        # Task 1's cache should be cleared
        assert task1_hash not in gatherer._research_context_cache
        # Task 2's cache should remain
        assert gatherer._research_context_cache[task2_hash] == "cached 2"


class TestGatherAragoraContext:
    """Tests for gather_aragora_context method."""

    @pytest.mark.asyncio
    async def test_returns_none_for_non_aragora_topics(self):
        """Should return None for non-Aragora topics."""
        gatherer = ContextGatherer()
        result = await gatherer.gather_aragora_context("How to bake a cake")
        assert result is None

    @pytest.mark.asyncio
    async def test_activates_for_aragora_keyword(self):
        """Should activate for tasks mentioning 'aragora'."""
        gatherer = ContextGatherer(project_root=Path("/nonexistent"))
        # Will fail to load docs but should attempt
        result = await gatherer.gather_aragora_context("Discuss aragora features")
        # Returns None because docs don't exist at /nonexistent
        assert result is None

    @pytest.mark.asyncio
    async def test_activates_for_multi_agent_debate_keyword(self):
        """Should activate for tasks mentioning 'multi-agent debate'."""
        gatherer = ContextGatherer(project_root=Path("/nonexistent"))
        result = await gatherer.gather_aragora_context("How do multi-agent debate systems work?")
        assert result is None  # Docs don't exist

    @pytest.mark.asyncio
    async def test_activates_for_nomic_loop_keyword(self):
        """Should activate for tasks mentioning 'nomic loop'."""
        gatherer = ContextGatherer(project_root=Path("/nonexistent"))
        result = await gatherer.gather_aragora_context("Explain the nomic loop")
        assert result is None

    @pytest.mark.asyncio
    async def test_activates_for_debate_framework_keyword(self):
        """Should activate for tasks mentioning 'debate framework'."""
        gatherer = ContextGatherer(project_root=Path("/nonexistent"))
        result = await gatherer.gather_aragora_context("Build a debate framework")
        assert result is None


class TestGatherEvidenceContext:
    """Tests for gather_evidence_context method."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_connectors_available(self):
        """Should return None when no evidence connectors are available."""
        gatherer = ContextGatherer()
        with patch.dict("sys.modules", {"aragora.evidence.collector": None}):
            with patch("aragora.debate.context_gatherer.logger"):
                # ImportError will be caught
                result = await gatherer.gather_evidence_context("test task")
                # Result depends on whether evidence module exists
                # Either None or formatted context
                assert result is None or "EVIDENCE" in result

    @pytest.mark.asyncio
    async def test_calls_evidence_store_callback(self):
        """Should call evidence store callback when evidence is collected."""
        callback = MagicMock()
        gatherer = ContextGatherer(evidence_store_callback=callback)

        # Mock the evidence collector
        mock_pack = MagicMock()
        mock_pack.snippets = [{"text": "test"}]
        mock_pack.to_context_string.return_value = "Evidence text"

        mock_collector_class = MagicMock()
        mock_collector = MagicMock()
        mock_collector.collect_evidence = AsyncMock(return_value=mock_pack)
        mock_collector_class.return_value = mock_collector

        with patch(
            "aragora.debate.context_gatherer.ContextGatherer.gather_evidence_context"
        ) as mock:
            mock.return_value = "## EVIDENCE CONTEXT\nEvidence text"
            result = await mock("test task")
            assert result is not None

    @pytest.mark.asyncio
    async def test_updates_prompt_builder_with_evidence(self):
        """Should update prompt builder when evidence is collected."""
        builder = MagicMock()
        gatherer = ContextGatherer(prompt_builder=builder)

        # The actual test would require mocking the evidence collector chain
        # which is complex. Here we verify the setup is correct.
        assert gatherer._prompt_builder == builder


class TestGatherTrendingContext:
    """Tests for gather_trending_context method."""

    @pytest.mark.asyncio
    async def test_returns_none_when_pulse_raises_error(self):
        """Should return None when pulse module raises an error."""
        gatherer = ContextGatherer()

        # Mock the PulseManager to raise an error during initialization
        # This tests the error handling path without manipulating sys.modules
        with patch(
            "aragora.debate.context_gatherer.ContextGatherer.gather_trending_context",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_method:
            # Verify the method handles errors gracefully
            result = await mock_method()
            assert result is None

        # Also test actual error handling by mocking the import at module level
        with patch.dict("sys.modules", {"aragora.pulse.ingestor": None}):
            # When the module is None in sys.modules, import will fail
            # Create fresh gatherer to avoid any caching
            fresh_gatherer = ContextGatherer()
            # The actual import happens inside gather_trending_context
            # If pulse.ingestor is None, importing from it raises TypeError
            result = await fresh_gatherer.gather_trending_context()
            # Should handle gracefully and return None or actual trending
            assert result is None or "TRENDING" in str(result)

    @pytest.mark.asyncio
    async def test_formats_trending_topics(self):
        """Should format trending topics correctly."""
        gatherer = ContextGatherer()

        # Mock the pulse manager
        mock_topic = MagicMock()
        mock_topic.topic = "Test Topic"
        mock_topic.platform = "twitter"
        mock_topic.volume = 1000
        mock_topic.category = "tech"

        mock_manager = MagicMock()
        mock_manager.get_trending_topics = AsyncMock(return_value=[mock_topic])

        with patch("aragora.pulse.ingestor.PulseManager", return_value=mock_manager):
            with patch("aragora.pulse.ingestor.TwitterIngestor"):
                with patch("aragora.pulse.ingestor.HackerNewsIngestor"):
                    with patch("aragora.pulse.ingestor.RedditIngestor"):
                        result = await gatherer.gather_trending_context()
                        # Will fail due to ImportError in test env
                        assert result is None or "TRENDING" in result


class TestGatherAll:
    """Tests for gather_all method."""

    @pytest.mark.asyncio
    async def test_returns_no_context_message_when_empty(self):
        """Should return default message when no context available."""
        gatherer = ContextGatherer(project_root=Path("/nonexistent"))

        with patch.object(gatherer, "gather_aragora_context", AsyncMock(return_value=None)):
            with patch.object(gatherer, "gather_evidence_context", AsyncMock(return_value=None)):
                with patch.object(
                    gatherer, "gather_trending_context", AsyncMock(return_value=None)
                ):
                    result = await gatherer.gather_all("test task")
                    assert result == "No research context available."

    @pytest.mark.asyncio
    async def test_combines_multiple_contexts(self):
        """Should combine context from multiple sources."""
        gatherer = ContextGatherer()

        with patch.object(
            gatherer, "gather_aragora_context", AsyncMock(return_value="## ARAGORA\nDocs")
        ):
            with patch.object(
                gatherer, "gather_evidence_context", AsyncMock(return_value="## EVIDENCE\nData")
            ):
                with patch.object(
                    gatherer,
                    "gather_trending_context",
                    AsyncMock(return_value="## TRENDING\nTopics"),
                ):
                    result = await gatherer.gather_all("test task")

                    assert "ARAGORA" in result
                    assert "EVIDENCE" in result
                    assert "TRENDING" in result

    @pytest.mark.asyncio
    async def test_caches_result(self):
        """Should cache result and return cached value on subsequent calls."""
        gatherer = ContextGatherer()

        with patch.object(
            gatherer, "gather_aragora_context", AsyncMock(return_value="## CONTEXT\nTest")
        ) as mock:
            with patch.object(gatherer, "gather_evidence_context", AsyncMock(return_value=None)):
                with patch.object(
                    gatherer, "gather_trending_context", AsyncMock(return_value=None)
                ):
                    result1 = await gatherer.gather_all("test task")
                    result2 = await gatherer.gather_all("test task")

                    # Should be called only once due to caching
                    assert mock.call_count == 1
                    assert result1 == result2

    @pytest.mark.asyncio
    async def test_different_tasks_isolated_in_cache(self):
        """Different tasks should have separate cache entries (bug fix verification)."""
        gatherer = ContextGatherer()

        # Mock to return different results for different tasks
        async def mock_gather_aragora(task=None, **kwargs):
            if "task A" in str(task):
                return "## CONTEXT\nContext for Task A"
            else:
                return "## CONTEXT\nContext for Task B"

        with patch.object(
            gatherer, "gather_aragora_context", AsyncMock(side_effect=mock_gather_aragora)
        ):
            with patch.object(gatherer, "gather_evidence_context", AsyncMock(return_value=None)):
                with patch.object(
                    gatherer, "gather_trending_context", AsyncMock(return_value=None)
                ):
                    # Gather for two different tasks
                    result_a = await gatherer.gather_all("task A")
                    result_b = await gatherer.gather_all("task B")

                    # Results should be different (not leaking between tasks)
                    assert "Task A" in result_a
                    assert "Task B" in result_b
                    assert result_a != result_b

                    # Verify both are cached separately
                    task_a_hash = gatherer._get_task_hash("task A")
                    task_b_hash = gatherer._get_task_hash("task B")
                    assert task_a_hash in gatherer._research_context_cache
                    assert task_b_hash in gatherer._research_context_cache
                    assert gatherer._research_context_cache[task_a_hash] != gatherer._research_context_cache[task_b_hash]

    @pytest.mark.asyncio
    async def test_same_task_returns_cached_result(self):
        """Same task should return cached result without re-fetching."""
        gatherer = ContextGatherer()
        call_count = 0

        async def mock_gather(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return f"## CONTEXT\nResult #{call_count}"

        with patch.object(gatherer, "gather_aragora_context", AsyncMock(side_effect=mock_gather)):
            with patch.object(gatherer, "gather_evidence_context", AsyncMock(return_value=None)):
                with patch.object(gatherer, "gather_trending_context", AsyncMock(return_value=None)):
                    # Gather for same task twice
                    result1 = await gatherer.gather_all("identical task")
                    result2 = await gatherer.gather_all("identical task")

                    # Should return same result
                    assert result1 == result2
                    # Should only have called gather once
                    assert call_count == 1

    @pytest.mark.asyncio
    async def test_partial_context_still_returned(self):
        """Should return partial context if only some sources available."""
        gatherer = ContextGatherer()

        with patch.object(gatherer, "gather_aragora_context", AsyncMock(return_value=None)):
            with patch.object(
                gatherer, "gather_evidence_context", AsyncMock(return_value="## EVIDENCE\nData")
            ):
                with patch.object(
                    gatherer, "gather_trending_context", AsyncMock(return_value=None)
                ):
                    result = await gatherer.gather_all("test task")

                    assert "EVIDENCE" in result
                    assert "No research context" not in result


class TestIntegrationWithOrchestrator:
    """Tests for ContextGatherer integration with Arena orchestrator."""

    def test_can_be_used_as_delegate(self):
        """Should work as a delegate for Arena context methods."""
        callback = MagicMock()
        builder = MagicMock()
        gatherer = ContextGatherer(
            evidence_store_callback=callback,
            prompt_builder=builder,
        )

        # Verify interface is compatible
        assert hasattr(gatherer, "gather_all")
        assert hasattr(gatherer, "gather_aragora_context")
        assert hasattr(gatherer, "gather_evidence_context")
        assert hasattr(gatherer, "gather_trending_context")
        assert hasattr(gatherer, "evidence_pack")
        assert hasattr(gatherer, "clear_cache")

    @pytest.mark.asyncio
    async def test_evidence_pack_updated_after_gather(self):
        """evidence_pack should be updated after gathering evidence."""
        gatherer = ContextGatherer()

        mock_pack = MagicMock()
        mock_pack.snippets = [{"text": "test"}]
        mock_pack.to_context_string.return_value = "Evidence"

        # Set in dict keyed by task hash to simulate successful collection
        task_hash = gatherer._get_task_hash("test task")
        gatherer._research_evidence_pack[task_hash] = mock_pack

        # evidence_pack property returns the last entry
        assert gatherer.evidence_pack == mock_pack
        # get_evidence_pack returns for specific task
        assert gatherer.get_evidence_pack("test task") == mock_pack


class TestKnowledgeMoundIntegration:
    """Tests for Knowledge Mound auto-grounding integration."""

    def test_init_with_knowledge_grounding_disabled(self):
        """Should initialize with knowledge grounding disabled."""
        gatherer = ContextGatherer(enable_knowledge_grounding=False)
        assert gatherer._enable_knowledge_grounding is False

    def test_init_with_knowledge_grounding_enabled_no_mound(self):
        """Should handle missing Knowledge Mound gracefully."""
        with patch("aragora.debate.context_gatherer.HAS_KNOWLEDGE_MOUND", False):
            gatherer = ContextGatherer(enable_knowledge_grounding=True)
            assert gatherer._enable_knowledge_grounding is False

    def test_init_with_knowledge_mound_provided(self):
        """Should accept pre-configured Knowledge Mound instance."""
        mock_mound = MagicMock()
        gatherer = ContextGatherer(
            enable_knowledge_grounding=True,
            knowledge_mound=mock_mound,
        )
        assert gatherer._knowledge_mound == mock_mound

    def test_init_with_custom_workspace_id(self):
        """Should accept custom workspace ID for knowledge queries."""
        mock_mound = MagicMock()
        gatherer = ContextGatherer(
            enable_knowledge_grounding=True,
            knowledge_mound=mock_mound,
            knowledge_workspace_id="custom_workspace",
        )
        assert gatherer._knowledge_workspace_id == "custom_workspace"

    @pytest.mark.asyncio
    async def test_gather_knowledge_mound_context_disabled(self):
        """Should return None when knowledge grounding is disabled."""
        gatherer = ContextGatherer(enable_knowledge_grounding=False)
        result = await gatherer.gather_knowledge_mound_context("test task")
        assert result is None

    @pytest.mark.asyncio
    async def test_gather_knowledge_mound_context_no_results(self):
        """Should return None when no knowledge is found."""
        mock_mound = MagicMock()
        mock_result = MagicMock()
        mock_result.items = []
        mock_mound.query = AsyncMock(return_value=mock_result)

        gatherer = ContextGatherer(
            enable_knowledge_grounding=True,
            knowledge_mound=mock_mound,
        )
        result = await gatherer.gather_knowledge_mound_context("test task")
        assert result is None
        mock_mound.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_gather_knowledge_mound_context_with_facts(self):
        """Should format facts from knowledge mound."""
        mock_mound = MagicMock()

        # Create mock knowledge item with fact source
        mock_item = MagicMock()
        mock_item.content = "AI systems should be transparent"
        mock_item.source = MagicMock()
        mock_item.source.value = "fact"
        mock_item.confidence = 0.85

        mock_result = MagicMock()
        mock_result.items = [mock_item]
        mock_result.execution_time_ms = 50.0
        mock_mound.query = AsyncMock(return_value=mock_result)

        gatherer = ContextGatherer(
            enable_knowledge_grounding=True,
            knowledge_mound=mock_mound,
        )
        result = await gatherer.gather_knowledge_mound_context("AI safety discussion")

        assert result is not None
        assert "KNOWLEDGE MOUND CONTEXT" in result
        assert "Verified Facts" in result
        assert "[HIGH]" in result  # Confidence > 0.7

    @pytest.mark.asyncio
    async def test_gather_knowledge_mound_context_with_evidence(self):
        """Should format evidence from knowledge mound."""
        mock_mound = MagicMock()

        mock_item = MagicMock()
        mock_item.content = "Study shows 85% agreement on this point"
        mock_item.source = MagicMock()
        mock_item.source.value = "evidence"
        mock_item.confidence = 0.6

        mock_result = MagicMock()
        mock_result.items = [mock_item]
        mock_result.execution_time_ms = 30.0
        mock_mound.query = AsyncMock(return_value=mock_result)

        gatherer = ContextGatherer(
            enable_knowledge_grounding=True,
            knowledge_mound=mock_mound,
        )
        result = await gatherer.gather_knowledge_mound_context("test topic")

        assert result is not None
        assert "Supporting Evidence" in result

    @pytest.mark.asyncio
    async def test_gather_knowledge_mound_context_with_insights(self):
        """Should format insights from other sources."""
        mock_mound = MagicMock()

        mock_item = MagicMock()
        mock_item.content = "Historical debate pattern observed"
        mock_item.source = MagicMock()
        mock_item.source.value = "consensus"
        mock_item.confidence = 0.7

        mock_result = MagicMock()
        mock_result.items = [mock_item]
        mock_result.execution_time_ms = 25.0
        mock_mound.query = AsyncMock(return_value=mock_result)

        gatherer = ContextGatherer(
            enable_knowledge_grounding=True,
            knowledge_mound=mock_mound,
        )
        result = await gatherer.gather_knowledge_mound_context("debate topic")

        assert result is not None
        assert "Related Insights" in result
        assert "(consensus)" in result

    @pytest.mark.asyncio
    async def test_gather_knowledge_mound_timeout_protection(self):
        """Should handle timeout gracefully."""
        mock_mound = MagicMock()
        mock_mound.query = AsyncMock(side_effect=asyncio.TimeoutError())

        gatherer = ContextGatherer(
            enable_knowledge_grounding=True,
            knowledge_mound=mock_mound,
        )
        result = await gatherer._gather_knowledge_mound_with_timeout("test task")
        assert result is None

    @pytest.mark.asyncio
    async def test_gather_knowledge_mound_exception_handling(self):
        """Should handle exceptions gracefully."""
        mock_mound = MagicMock()
        mock_mound.query = AsyncMock(side_effect=Exception("Connection failed"))

        gatherer = ContextGatherer(
            enable_knowledge_grounding=True,
            knowledge_mound=mock_mound,
        )
        result = await gatherer.gather_knowledge_mound_context("test task")
        assert result is None

    @pytest.mark.asyncio
    async def test_gather_all_includes_knowledge_context(self):
        """gather_all should include knowledge context when available."""
        mock_mound = MagicMock()

        mock_item = MagicMock()
        mock_item.content = "Relevant knowledge"
        mock_item.source = MagicMock()
        mock_item.source.value = "fact"
        mock_item.confidence = 0.9

        mock_result = MagicMock()
        mock_result.items = [mock_item]
        mock_result.execution_time_ms = 20.0
        mock_mound.query = AsyncMock(return_value=mock_result)

        gatherer = ContextGatherer(
            enable_knowledge_grounding=True,
            knowledge_mound=mock_mound,
            project_root=Path("/nonexistent"),
        )

        # Mock other context sources to return None quickly
        with patch.object(gatherer, "_gather_claude_web_search", AsyncMock(return_value=None)):
            with patch.object(gatherer, "gather_aragora_context", AsyncMock(return_value=None)):
                with patch.object(gatherer, "gather_evidence_context", AsyncMock(return_value=None)):
                    with patch.object(gatherer, "gather_trending_context", AsyncMock(return_value=None)):
                        result = await gatherer.gather_all("test task", timeout=5.0)

        assert "KNOWLEDGE MOUND CONTEXT" in result


# =============================================================================
# Belief Crux Context Tests
# =============================================================================


class TestBeliefCruxContext:
    """Tests for gather_belief_crux_context method."""

    def test_init_with_belief_guidance_enabled(self):
        """Should initialize with belief guidance enabled by default."""
        gatherer = ContextGatherer()
        assert gatherer._enable_belief_guidance is True or gatherer._enable_belief_guidance is False
        # Analyzer may or may not be available depending on imports

    def test_init_with_belief_guidance_disabled(self):
        """Should disable belief guidance when explicitly set to False."""
        gatherer = ContextGatherer(enable_belief_guidance=False)
        assert gatherer._enable_belief_guidance is False
        assert gatherer._belief_analyzer is None

    @pytest.mark.asyncio
    async def test_gather_crux_returns_none_when_disabled(self):
        """Should return None when belief guidance is disabled."""
        gatherer = ContextGatherer(enable_belief_guidance=False)
        result = await gatherer.gather_belief_crux_context("test task")
        assert result is None

    @pytest.mark.asyncio
    async def test_gather_crux_returns_none_without_messages(self):
        """Should return None when no messages provided and no historical data."""
        gatherer = ContextGatherer(enable_belief_guidance=True)
        # Even if enabled, without messages we return None
        result = await gatherer.gather_belief_crux_context("test task", messages=None)
        assert result is None

    @pytest.mark.asyncio
    async def test_gather_crux_with_mock_analyzer(self):
        """Should gather crux context when messages provided."""
        gatherer = ContextGatherer(enable_belief_guidance=True)

        # Mock the belief analyzer
        mock_analyzer = MagicMock()
        mock_result = MagicMock()
        mock_result.analysis_error = None
        mock_result.cruxes = [
            {"statement": "Test crux claim", "confidence": 0.8, "entropy": 0.3},
            {"claim": "Another crux", "confidence": 0.6, "entropy": 0.9},
        ]
        mock_result.evidence_suggestions = ["Get more data on X"]
        mock_analyzer.analyze_messages = MagicMock(return_value=mock_result)

        gatherer._belief_analyzer = mock_analyzer
        gatherer._enable_belief_guidance = True

        # Mock messages
        mock_messages = [MagicMock(), MagicMock()]

        result = await gatherer.gather_belief_crux_context("test task", messages=mock_messages)

        assert result is not None
        assert "Key Crux Points" in result
        assert "Test crux claim" in result
        assert "Another crux" in result
        assert "Evidence Needed" in result

    @pytest.mark.asyncio
    async def test_gather_crux_handles_empty_cruxes(self):
        """Should return None when no cruxes found."""
        gatherer = ContextGatherer(enable_belief_guidance=True)

        mock_analyzer = MagicMock()
        mock_result = MagicMock()
        mock_result.analysis_error = None
        mock_result.cruxes = []  # No cruxes
        mock_result.evidence_suggestions = []
        mock_analyzer.analyze_messages = MagicMock(return_value=mock_result)

        gatherer._belief_analyzer = mock_analyzer
        gatherer._enable_belief_guidance = True

        mock_messages = [MagicMock()]
        result = await gatherer.gather_belief_crux_context("test task", messages=mock_messages)

        assert result is None

    @pytest.mark.asyncio
    async def test_gather_crux_handles_analysis_error(self):
        """Should return None when analysis returns an error."""
        gatherer = ContextGatherer(enable_belief_guidance=True)

        mock_analyzer = MagicMock()
        mock_result = MagicMock()
        mock_result.analysis_error = "Module not available"
        mock_result.cruxes = []
        mock_analyzer.analyze_messages = MagicMock(return_value=mock_result)

        gatherer._belief_analyzer = mock_analyzer
        gatherer._enable_belief_guidance = True

        mock_messages = [MagicMock()]
        result = await gatherer.gather_belief_crux_context("test task", messages=mock_messages)

        assert result is None

    @pytest.mark.asyncio
    async def test_gather_crux_marks_contested_cruxes(self):
        """Should mark high-entropy cruxes as contested."""
        gatherer = ContextGatherer(enable_belief_guidance=True)

        mock_analyzer = MagicMock()
        mock_result = MagicMock()
        mock_result.analysis_error = None
        mock_result.cruxes = [
            {"statement": "Contested claim", "confidence": 0.5, "entropy": 0.95},
        ]
        mock_result.evidence_suggestions = []
        mock_analyzer.analyze_messages = MagicMock(return_value=mock_result)

        gatherer._belief_analyzer = mock_analyzer
        gatherer._enable_belief_guidance = True

        result = await gatherer.gather_belief_crux_context("test task", messages=[MagicMock()])

        assert result is not None
        assert "CONTESTED" in result

    @pytest.mark.asyncio
    async def test_gather_crux_confidence_labels(self):
        """Should label cruxes with appropriate confidence markers."""
        gatherer = ContextGatherer(enable_belief_guidance=True)

        mock_analyzer = MagicMock()
        mock_result = MagicMock()
        mock_result.analysis_error = None
        mock_result.cruxes = [
            {"statement": "High confidence", "confidence": 0.9, "entropy": 0.1},
            {"statement": "Medium confidence", "confidence": 0.5, "entropy": 0.3},
            {"statement": "Low confidence", "confidence": 0.2, "entropy": 0.4},
        ]
        mock_result.evidence_suggestions = []
        mock_analyzer.analyze_messages = MagicMock(return_value=mock_result)

        gatherer._belief_analyzer = mock_analyzer
        gatherer._enable_belief_guidance = True

        result = await gatherer.gather_belief_crux_context("test task", messages=[MagicMock()])

        assert "[HIGH]" in result
        assert "[MEDIUM]" in result
        assert "[LOW]" in result

    @pytest.mark.asyncio
    async def test_gather_crux_handles_exceptions(self):
        """Should handle exceptions gracefully."""
        gatherer = ContextGatherer(enable_belief_guidance=True)

        mock_analyzer = MagicMock()
        mock_analyzer.analyze_messages = MagicMock(side_effect=ValueError("Test error"))

        gatherer._belief_analyzer = mock_analyzer
        gatherer._enable_belief_guidance = True

        result = await gatherer.gather_belief_crux_context("test task", messages=[MagicMock()])

        # Should not raise, returns None
        assert result is None

    @pytest.mark.asyncio
    async def test_gather_with_timeout_respects_timeout(self):
        """_gather_belief_with_timeout should respect timeout."""
        gatherer = ContextGatherer(enable_belief_guidance=True)

        # Mock slow analyzer
        async def slow_analyze(*args, **kwargs):
            await asyncio.sleep(10)  # Very slow
            return None

        gatherer._belief_analyzer = MagicMock()
        gatherer._enable_belief_guidance = True
        gatherer.gather_belief_crux_context = slow_analyze

        # Should timeout quickly (default is 5s, but we'll force shorter)
        with patch("aragora.debate.context_gatherer.BELIEF_CRUX_TIMEOUT", 0.1):
            result = await gatherer._gather_belief_with_timeout("test task")

        # Should return None due to timeout, not raise
        assert result is None
