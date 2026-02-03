"""Tests for the debate context gatherer orchestrator."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.context.gatherer import ContextGatherer


class TestContextGathererInit:
    """Test ContextGatherer initialization."""

    def test_default_init(self):
        gatherer = ContextGatherer(
            enable_knowledge_grounding=False,
            enable_belief_guidance=False,
            enable_threat_intel_enrichment=False,
            enable_trending_context=False,
        )
        assert gatherer._evidence_store_callback is None
        assert gatherer._prompt_builder is None
        assert gatherer._enable_trending_context is False

    def test_init_with_callback(self):
        callback = MagicMock()
        gatherer = ContextGatherer(
            evidence_store_callback=callback,
            enable_knowledge_grounding=False,
            enable_belief_guidance=False,
            enable_threat_intel_enrichment=False,
            enable_trending_context=False,
        )
        assert gatherer._evidence_store_callback is callback

    def test_init_with_project_root(self, tmp_path: Path):
        gatherer = ContextGatherer(
            project_root=tmp_path,
            enable_knowledge_grounding=False,
            enable_belief_guidance=False,
            enable_threat_intel_enrichment=False,
            enable_trending_context=False,
        )
        assert gatherer._project_root == tmp_path

    @patch.dict("os.environ", {"ARAGORA_DISABLE_TRENDING": "true"})
    def test_trending_disabled_by_env(self):
        gatherer = ContextGatherer(
            enable_knowledge_grounding=False,
            enable_belief_guidance=False,
            enable_threat_intel_enrichment=False,
            enable_trending_context=True,
        )
        assert gatherer._enable_trending_context is False

    def test_set_prompt_builder(self):
        gatherer = ContextGatherer(
            enable_knowledge_grounding=False,
            enable_belief_guidance=False,
            enable_threat_intel_enrichment=False,
            enable_trending_context=False,
        )
        builder = MagicMock()
        gatherer.set_prompt_builder(builder)
        assert gatherer._prompt_builder is builder


class TestContextGathererCache:
    """Test caching behavior."""

    def test_evidence_pack_none_initially(self):
        gatherer = ContextGatherer(
            enable_knowledge_grounding=False,
            enable_belief_guidance=False,
            enable_threat_intel_enrichment=False,
            enable_trending_context=False,
        )
        assert gatherer.evidence_pack is None

    def test_get_evidence_pack_for_task(self):
        gatherer = ContextGatherer(
            enable_knowledge_grounding=False,
            enable_belief_guidance=False,
            enable_threat_intel_enrichment=False,
            enable_trending_context=False,
        )
        assert gatherer.get_evidence_pack("some task") is None

    def test_task_hash_deterministic(self):
        gatherer = ContextGatherer(
            enable_knowledge_grounding=False,
            enable_belief_guidance=False,
            enable_threat_intel_enrichment=False,
            enable_trending_context=False,
        )
        h1 = gatherer._get_task_hash("test task")
        h2 = gatherer._get_task_hash("test task")
        assert h1 == h2

    def test_task_hash_differs_for_different_tasks(self):
        gatherer = ContextGatherer(
            enable_knowledge_grounding=False,
            enable_belief_guidance=False,
            enable_threat_intel_enrichment=False,
            enable_trending_context=False,
        )
        h1 = gatherer._get_task_hash("task one")
        h2 = gatherer._get_task_hash("task two")
        assert h1 != h2


class TestGatherAll:
    """Test the main gather_all async method."""

    @pytest.fixture
    def gatherer(self):
        return ContextGatherer(
            enable_knowledge_grounding=False,
            enable_belief_guidance=False,
            enable_threat_intel_enrichment=False,
            enable_trending_context=False,
        )

    async def test_returns_string(self, gatherer):
        # Mock the source fetcher with AsyncMock for all awaited methods
        fetcher = MagicMock()
        fetcher.gather_claude_web_search = AsyncMock(return_value="Web search context")
        fetcher.gather_knowledge_mound_with_timeout = AsyncMock(return_value=None)
        fetcher.gather_belief_with_timeout = AsyncMock(return_value=None)
        fetcher.gather_culture_with_timeout = AsyncMock(return_value=None)
        fetcher.gather_threat_intel_with_timeout = AsyncMock(return_value=None)
        gatherer._source_fetcher = fetcher
        # Mock gather_aragora_context on the gatherer itself
        gatherer.gather_aragora_context = AsyncMock(return_value=None)
        gatherer._cache = MagicMock()
        gatherer._cache.get_context = MagicMock(return_value=None)
        gatherer._cache.set_context = MagicMock()
        gatherer._cache.set_evidence_pack = MagicMock()

        result = await gatherer.gather_all("Test debate topic")
        assert isinstance(result, str)

    async def test_returns_cached_result(self, gatherer):
        gatherer._cache = MagicMock()
        gatherer._cache.get_context = MagicMock(return_value="Cached context")

        result = await gatherer.gather_all("Test topic")
        assert result == "Cached context"

    async def test_clear_cache(self, gatherer):
        gatherer._cache = MagicMock()
        gatherer.clear_cache()
        gatherer._cache.clear.assert_called_once()
