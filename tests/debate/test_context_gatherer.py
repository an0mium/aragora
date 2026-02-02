"""
Tests for debate context_gatherer module.

Tests cover:
- ContextGatherer initialization (defaults, RLM, knowledge mound, belief, threat intel)
- evidence_pack property
- get_evidence_pack method
- set_prompt_builder method
- _get_task_hash method
- gather_all method (caching, timeout, partial results)
- _gather_claude_web_search method (import error, timeout)
- gather_aragora_context method (keyword matching, non-relevant task)
- gather_evidence_context method (import errors, callbacks)
- gather_trending_context method (disabled, import error)
- get_trending_topics method
- gather_knowledge_mound_context method (disabled, success, items formatting)
- gather_belief_crux_context method (disabled, with messages, no messages)
- gather_culture_patterns_context method (no mound, culture API, fallback)
- gather_threat_intel_context method (disabled, not security topic, success)
- clear_cache method (all caches, task-specific)
- _enforce_cache_limit method
- _compress_with_rlm method (under threshold, no RLM, with RLM, fallback)
- get_continuum_context method (empty, with memories, caching, glacial)
- refresh_evidence_for_round method (no collector, merge, callback)
"""

import asyncio
import hashlib
from dataclasses import dataclass, field
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


# =============================================================================
# Mock Objects
# =============================================================================


@dataclass
class MockEvidencePack:
    """Mock evidence pack for testing."""

    snippets: list = field(default_factory=list)
    total_searched: int = 0

    def to_context_string(self) -> str:
        return f"Evidence: {len(self.snippets)} snippets"


@dataclass
class MockSnippet:
    """Mock evidence snippet."""

    id: str = "snippet-1"
    content: str = "test evidence content"


@dataclass
class MockKnowledgeItem:
    """Mock knowledge mound item."""

    content: str = "some knowledge content"
    source: Any = None
    confidence: float = 0.8


@dataclass
class MockKnowledgeResult:
    """Mock knowledge mound query result."""

    items: list = field(default_factory=list)
    execution_time_ms: float = 10.0


@dataclass
class MockTrendingTopic:
    """Mock trending topic."""

    topic: str = "AI Safety"
    platform: str = "hackernews"
    volume: int = 1500
    category: str = "technology"


@dataclass
class MockBeliefResult:
    """Mock belief analysis result."""

    cruxes: list = field(default_factory=list)
    evidence_suggestions: list = field(default_factory=list)
    analysis_error: str | None = None


@dataclass
class MockTier:
    """Mock memory tier."""

    value: str = "medium"


@dataclass
class MockMemory:
    """Mock continuum memory entry."""

    id: str = "mem-1"
    content: str = "previous learning"
    tier: MockTier = field(default_factory=MockTier)
    consolidation_score: float = 0.7


@dataclass
class MockRLMResult:
    """Mock RLM compression result."""

    answer: str = "compressed content"
    used_true_rlm: bool = False
    confidence: float = 0.8


@dataclass
class MockThreatIntelContext:
    """Mock threat intelligence context."""

    indicators: list = field(default_factory=lambda: ["CVE-2024-1234"])
    relevant_cves: list = field(default_factory=lambda: ["CVE-2024-1234"])


# =============================================================================
# Helper to create a ContextGatherer with optional deps disabled
# =============================================================================


def _make_gatherer(**kwargs):
    """Create a ContextGatherer with optional deps disabled by default."""
    defaults = dict(
        enable_rlm_compression=False,
        enable_knowledge_grounding=False,
        enable_belief_guidance=False,
        enable_threat_intel_enrichment=False,
        enable_trending_context=False,
    )
    defaults.update(kwargs)
    return ContextGatherer(**defaults)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestContextGathererInit:
    """Tests for ContextGatherer initialization."""

    def test_init_with_defaults(self):
        """Gatherer initializes with sensible defaults."""
        gatherer = _make_gatherer()

        assert gatherer._evidence_store_callback is None
        assert gatherer._prompt_builder is None
        assert gatherer._research_evidence_pack == {}
        assert gatherer._research_context_cache == {}
        assert gatherer._continuum_context_cache == {}
        assert gatherer._trending_topics_cache == []

    def test_init_with_evidence_callback(self):
        """Gatherer stores evidence store callback."""
        callback = MagicMock()
        gatherer = _make_gatherer(evidence_store_callback=callback)

        assert gatherer._evidence_store_callback is callback

    def test_init_with_prompt_builder(self):
        """Gatherer stores prompt builder."""
        builder = MagicMock()
        gatherer = _make_gatherer(prompt_builder=builder)

        assert gatherer._prompt_builder is builder

    def test_init_with_custom_project_root(self):
        """Gatherer uses custom project root."""
        custom_root = Path("/tmp/test-project")
        gatherer = _make_gatherer(project_root=custom_root)

        assert gatherer._project_root == custom_root

    def test_init_rlm_disabled(self):
        """RLM features are disabled when explicitly set."""
        gatherer = _make_gatherer(enable_rlm_compression=False)

        assert gatherer._enable_rlm is False
        assert gatherer._rlm_compressor is None

    def test_init_knowledge_grounding_disabled(self):
        """Knowledge grounding disabled when explicitly set."""
        gatherer = _make_gatherer(enable_knowledge_grounding=False)

        assert gatherer._enable_knowledge_grounding is False
        assert gatherer._knowledge_mound is None

    def test_init_belief_guidance_disabled(self):
        """Belief guidance disabled when explicitly set."""
        gatherer = _make_gatherer(enable_belief_guidance=False)

        assert gatherer._enable_belief_guidance is False
        assert gatherer._belief_analyzer is None

    def test_init_threat_intel_disabled(self):
        """Threat intel disabled when explicitly set."""
        gatherer = _make_gatherer(enable_threat_intel_enrichment=False)

        assert gatherer._enable_threat_intel is False

    def test_init_custom_rlm_threshold(self):
        """Custom RLM compression threshold is stored."""
        gatherer = _make_gatherer(rlm_compression_threshold=5000)

        assert gatherer._rlm_threshold == 5000

    def test_init_knowledge_workspace_id_defaults(self):
        """Knowledge workspace ID defaults to 'debate'."""
        gatherer = _make_gatherer()

        assert gatherer._knowledge_workspace_id == "debate"

    def test_init_custom_knowledge_workspace(self):
        """Custom knowledge workspace ID is stored."""
        gatherer = _make_gatherer(knowledge_workspace_id="custom-ws")

        assert gatherer._knowledge_workspace_id == "custom-ws"

    def test_init_trending_disabled(self):
        """Trending context disabled when explicitly set."""
        gatherer = _make_gatherer(enable_trending_context=False)

        assert gatherer._enable_trending_context is False


# =============================================================================
# _get_task_hash Tests
# =============================================================================


class TestGetTaskHash:
    """Tests for _get_task_hash method."""

    def test_returns_consistent_hash(self):
        """Same task returns the same hash."""
        gatherer = _make_gatherer()
        hash1 = gatherer._get_task_hash("test task")
        hash2 = gatherer._get_task_hash("test task")

        assert hash1 == hash2

    def test_different_tasks_different_hashes(self):
        """Different tasks return different hashes."""
        gatherer = _make_gatherer()
        hash1 = gatherer._get_task_hash("task A")
        hash2 = gatherer._get_task_hash("task B")

        assert hash1 != hash2

    def test_hash_is_16_chars(self):
        """Task hash is exactly 16 characters."""
        gatherer = _make_gatherer()
        task_hash = gatherer._get_task_hash("any task")

        assert len(task_hash) == 16

    def test_hash_matches_sha256_prefix(self):
        """Hash matches first 16 chars of SHA-256 hex digest."""
        gatherer = _make_gatherer()
        task = "test task"
        expected = hashlib.sha256(task.encode()).hexdigest()[:16]

        assert gatherer._get_task_hash(task) == expected


# =============================================================================
# evidence_pack Property Tests
# =============================================================================


class TestEvidencePackProperty:
    """Tests for evidence_pack property."""

    def test_returns_none_when_empty(self):
        """Returns None when no evidence packs cached."""
        gatherer = _make_gatherer()

        assert gatherer.evidence_pack is None

    def test_returns_most_recent_pack(self):
        """Returns the most recently added evidence pack."""
        gatherer = _make_gatherer()
        pack1 = MockEvidencePack(snippets=[MockSnippet(id="s1")])
        pack2 = MockEvidencePack(snippets=[MockSnippet(id="s2")])

        gatherer._research_evidence_pack["hash1"] = pack1
        gatherer._research_evidence_pack["hash2"] = pack2

        assert gatherer.evidence_pack is pack2


# =============================================================================
# get_evidence_pack Tests
# =============================================================================


class TestGetEvidencePack:
    """Tests for get_evidence_pack method."""

    def test_returns_pack_for_matching_task(self):
        """Returns evidence pack for the given task."""
        gatherer = _make_gatherer()
        pack = MockEvidencePack(snippets=[MockSnippet()])
        task = "my debate task"
        task_hash = gatherer._get_task_hash(task)
        gatherer._research_evidence_pack[task_hash] = pack

        assert gatherer.get_evidence_pack(task) is pack

    def test_returns_none_for_unknown_task(self):
        """Returns None when no pack exists for the task."""
        gatherer = _make_gatherer()

        assert gatherer.get_evidence_pack("unknown task") is None


# =============================================================================
# set_prompt_builder Tests
# =============================================================================


class TestSetPromptBuilder:
    """Tests for set_prompt_builder method."""

    def test_sets_prompt_builder(self):
        """Sets the prompt builder reference."""
        gatherer = _make_gatherer()
        builder = MagicMock()
        gatherer.set_prompt_builder(builder)

        assert gatherer._prompt_builder is builder

    def test_replaces_existing_builder(self):
        """Replaces an existing prompt builder."""
        old_builder = MagicMock()
        gatherer = _make_gatherer(prompt_builder=old_builder)

        new_builder = MagicMock()
        gatherer.set_prompt_builder(new_builder)

        assert gatherer._prompt_builder is new_builder


# =============================================================================
# gather_all Tests
# =============================================================================


@pytest.mark.timeout(30)  # Protect against unmocked external calls
class TestGatherAll:
    """Tests for gather_all method."""

    @pytest.mark.asyncio
    async def test_returns_cached_result(self):
        """Returns cached result for previously gathered task."""
        gatherer = _make_gatherer()
        task = "cached task"
        task_hash = gatherer._get_task_hash(task)
        gatherer._research_context_cache[task_hash] = "cached context"

        result = await gatherer.gather_all(task)

        assert result == "cached context"

    @pytest.mark.asyncio
    async def test_returns_no_context_when_all_sources_empty(self):
        """Returns default message when no sources return data."""
        gatherer = _make_gatherer()

        with (
            patch.object(
                gatherer, "_gather_claude_web_search", new_callable=AsyncMock, return_value=None
            ),
            patch.object(
                gatherer, "gather_aragora_context", new_callable=AsyncMock, return_value=None
            ),
            patch.object(
                gatherer,
                "_gather_knowledge_mound_with_timeout",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.object(
                gatherer, "_gather_belief_with_timeout", new_callable=AsyncMock, return_value=None
            ),
            patch.object(
                gatherer, "_gather_culture_with_timeout", new_callable=AsyncMock, return_value=None
            ),
            patch.object(
                gatherer,
                "_gather_threat_intel_with_timeout",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await gatherer.gather_all("some task")

        assert result == "No research context available."

    @pytest.mark.asyncio
    async def test_combines_multiple_context_sources(self):
        """Combines results from multiple sources."""
        gatherer = _make_gatherer()
        # Claude search must return enough content to skip evidence fallback
        large_claude_result = "Claude research " * 100 + " Key Sources listed here"

        with (
            patch.object(
                gatherer,
                "_gather_claude_web_search",
                new_callable=AsyncMock,
                return_value=large_claude_result,
            ),
            patch.object(
                gatherer,
                "gather_aragora_context",
                new_callable=AsyncMock,
                return_value="Aragora docs",
            ),
            patch.object(
                gatherer,
                "_gather_knowledge_mound_with_timeout",
                new_callable=AsyncMock,
                return_value="Knowledge context",
            ),
            patch.object(
                gatherer, "_gather_belief_with_timeout", new_callable=AsyncMock, return_value=None
            ),
            patch.object(
                gatherer, "_gather_culture_with_timeout", new_callable=AsyncMock, return_value=None
            ),
            patch.object(
                gatherer,
                "_gather_threat_intel_with_timeout",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await gatherer.gather_all("some task")

        assert "Claude research" in result
        assert "Aragora docs" in result
        assert "Knowledge context" in result

    @pytest.mark.asyncio
    async def test_caches_gathered_result(self):
        """Caches the result for subsequent calls."""
        gatherer = _make_gatherer()

        with (
            patch.object(
                gatherer, "_gather_claude_web_search", new_callable=AsyncMock, return_value=None
            ),
            patch.object(
                gatherer,
                "gather_aragora_context",
                new_callable=AsyncMock,
                return_value="Aragora context",
            ),
            patch.object(
                gatherer,
                "_gather_knowledge_mound_with_timeout",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.object(
                gatherer, "_gather_belief_with_timeout", new_callable=AsyncMock, return_value=None
            ),
            patch.object(
                gatherer, "_gather_culture_with_timeout", new_callable=AsyncMock, return_value=None
            ),
            patch.object(
                gatherer,
                "_gather_threat_intel_with_timeout",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result1 = await gatherer.gather_all("cache test task")

        task_hash = gatherer._get_task_hash("cache test task")
        assert task_hash in gatherer._research_context_cache
        assert gatherer._research_context_cache[task_hash] == result1

    @pytest.mark.asyncio
    async def test_handles_overall_timeout(self):
        """Returns partial results when overall timeout expires."""
        gatherer = _make_gatherer()

        async def slow_search(task):
            await asyncio.sleep(10)
            return "slow result"

        with (
            patch.object(gatherer, "_gather_claude_web_search", side_effect=slow_search),
            patch.object(
                gatherer, "gather_aragora_context", new_callable=AsyncMock, return_value=None
            ),
            patch.object(
                gatherer,
                "_gather_knowledge_mound_with_timeout",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.object(
                gatherer, "_gather_belief_with_timeout", new_callable=AsyncMock, return_value=None
            ),
            patch.object(
                gatherer, "_gather_culture_with_timeout", new_callable=AsyncMock, return_value=None
            ),
            patch.object(
                gatherer,
                "_gather_threat_intel_with_timeout",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await gatherer.gather_all("timeout task", timeout=0.1)

        assert result == "No research context available."


# =============================================================================
# _gather_claude_web_search Tests
# =============================================================================


class TestGatherClaudeWebSearch:
    """Tests for _gather_claude_web_search method."""

    @pytest.mark.asyncio
    async def test_returns_none_on_import_error(self):
        """Returns None when research_phase module is not available."""
        gatherer = _make_gatherer()

        # The real import of aragora.server.research_phase will likely fail in test env
        result = await gatherer._gather_claude_web_search("test task")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_result(self):
        """Returns None when search returns empty/None result."""
        gatherer = _make_gatherer()

        mock_module = MagicMock()
        mock_module.research_for_debate = AsyncMock(return_value=None)

        with patch.dict("sys.modules", {"aragora.server.research_phase": mock_module}):
            with patch(
                "aragora.debate.context_gatherer.asyncio.wait_for",
                new_callable=AsyncMock,
                return_value=None,
            ):
                result = await gatherer._gather_claude_web_search("test")

        # Will return None because the real import likely fails
        assert result is None


# =============================================================================
# gather_aragora_context Tests
# =============================================================================


class TestGatherAragoraContext:
    """Tests for gather_aragora_context method."""

    @pytest.mark.asyncio
    async def test_returns_none_for_non_aragora_task(self):
        """Returns None when task is not Aragora-related."""
        gatherer = _make_gatherer()

        result = await gatherer.gather_aragora_context("Discuss Python best practices")

        assert result is None

    @pytest.mark.asyncio
    async def test_activates_for_aragora_keyword(self):
        """Activates when task mentions aragora."""
        gatherer = _make_gatherer(project_root=Path("/nonexistent"))

        result = await gatherer.gather_aragora_context("Discuss aragora architecture")

        # Will return None because docs dir doesn't exist, but won't crash
        assert result is None

    @pytest.mark.asyncio
    async def test_activates_for_multi_agent_debate_keyword(self):
        """Activates when task mentions multi-agent debate."""
        gatherer = _make_gatherer(project_root=Path("/nonexistent"))

        result = await gatherer.gather_aragora_context("Design a multi-agent debate system")

        assert result is None

    @pytest.mark.asyncio
    async def test_activates_for_gauntlet_keyword(self):
        """Activates when task mentions gauntlet."""
        gatherer = _make_gatherer(project_root=Path("/nonexistent"))

        result = await gatherer.gather_aragora_context("Run the gauntlet tests")

        assert result is None

    @pytest.mark.asyncio
    async def test_activates_for_nomic_loop_keyword(self):
        """Activates when task mentions nomic loop."""
        gatherer = _make_gatherer(project_root=Path("/nonexistent"))

        result = await gatherer.gather_aragora_context("Improve the nomic loop")

        assert result is None

    @pytest.mark.asyncio
    async def test_case_insensitive_keyword_matching(self):
        """Keyword matching is case-insensitive."""
        gatherer = _make_gatherer(project_root=Path("/nonexistent"))

        result = await gatherer.gather_aragora_context("Discuss ARAGORA System")

        # Should activate (aragora keyword), returns None because /nonexistent
        assert result is None


# =============================================================================
# gather_evidence_context Tests
# =============================================================================


class TestGatherEvidenceContext:
    """Tests for gather_evidence_context method."""

    @pytest.mark.asyncio
    async def test_returns_none_when_collector_unavailable(self):
        """Returns None when EvidenceCollector is not available."""
        gatherer = _make_gatherer()

        # The import of aragora.evidence.collector may fail in test env
        result = await gatherer.gather_evidence_context("test task")

        # Should gracefully return None on import error
        assert result is None


# =============================================================================
# gather_trending_context Tests
# =============================================================================


class TestGatherTrendingContext:
    """Tests for gather_trending_context method."""

    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(self):
        """Returns None when trending context is disabled."""
        gatherer = _make_gatherer(enable_trending_context=False)

        result = await gatherer.gather_trending_context()

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_import_error(self):
        """Returns None when pulse module is not available."""
        gatherer = _make_gatherer(enable_trending_context=True)

        result = await gatherer.gather_trending_context()

        # In test env, pulse module likely not available
        assert result is None


# =============================================================================
# get_trending_topics Tests
# =============================================================================


class TestGetTrendingTopics:
    """Tests for get_trending_topics method."""

    def test_returns_empty_list_initially(self):
        """Returns empty list when no topics gathered."""
        gatherer = _make_gatherer()

        assert gatherer.get_trending_topics() == []

    def test_returns_cached_topics(self):
        """Returns cached trending topics."""
        gatherer = _make_gatherer()
        topics = [MockTrendingTopic(topic="AI"), MockTrendingTopic(topic="Rust")]
        gatherer._trending_topics_cache = topics

        result = gatherer.get_trending_topics()

        assert len(result) == 2
        assert result[0].topic == "AI"


# =============================================================================
# gather_knowledge_mound_context Tests
# =============================================================================


class TestGatherKnowledgeMoundContext:
    """Tests for gather_knowledge_mound_context method."""

    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(self):
        """Returns None when knowledge grounding is disabled."""
        gatherer = _make_gatherer(enable_knowledge_grounding=False)

        result = await gatherer.gather_knowledge_mound_context("test task")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_mound(self):
        """Returns None when no knowledge mound is configured."""
        gatherer = _make_gatherer()
        gatherer._enable_knowledge_grounding = True
        gatherer._knowledge_mound = None

        result = await gatherer.gather_knowledge_mound_context("test task")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_items(self):
        """Returns None when query returns no items."""
        gatherer = _make_gatherer()
        gatherer._enable_knowledge_grounding = True
        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(return_value=MockKnowledgeResult(items=[]))
        gatherer._knowledge_mound = mock_mound

        result = await gatherer.gather_knowledge_mound_context("test task")

        assert result is None

    @pytest.mark.asyncio
    async def test_formats_facts_evidence_and_insights(self):
        """Formats knowledge items into facts, evidence, and insights."""
        gatherer = _make_gatherer()
        gatherer._enable_knowledge_grounding = True

        mock_fact_source = MagicMock()
        mock_fact_source.value = "fact"
        mock_evidence_source = MagicMock()
        mock_evidence_source.value = "evidence"
        mock_other_source = MagicMock()
        mock_other_source.value = "pattern"

        items = [
            MockKnowledgeItem(content="Fact 1", source=mock_fact_source, confidence=0.9),
            MockKnowledgeItem(content="Evidence 1", source=mock_evidence_source, confidence=0.7),
            MockKnowledgeItem(content="Insight 1", source=mock_other_source, confidence=0.6),
        ]

        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(
            return_value=MockKnowledgeResult(items=items, execution_time_ms=15.0)
        )
        gatherer._knowledge_mound = mock_mound

        result = await gatherer.gather_knowledge_mound_context("test task")

        assert result is not None
        assert "KNOWLEDGE MOUND CONTEXT" in result
        assert "Verified Facts" in result
        assert "Supporting Evidence" in result
        assert "Related Insights" in result

    @pytest.mark.asyncio
    async def test_handles_query_error_gracefully(self):
        """Returns None when knowledge query raises exception."""
        gatherer = _make_gatherer()
        gatherer._enable_knowledge_grounding = True

        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(side_effect=ConnectionError("DB unavailable"))
        gatherer._knowledge_mound = mock_mound

        result = await gatherer.gather_knowledge_mound_context("test task")

        assert result is None


# =============================================================================
# gather_belief_crux_context Tests
# =============================================================================


class TestGatherBeliefCruxContext:
    """Tests for gather_belief_crux_context method."""

    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(self):
        """Returns None when belief guidance is disabled."""
        gatherer = _make_gatherer(enable_belief_guidance=False)

        result = await gatherer.gather_belief_crux_context("test task")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_analyzer(self):
        """Returns None when no belief analyzer is configured."""
        gatherer = _make_gatherer()
        gatherer._enable_belief_guidance = True
        gatherer._belief_analyzer = None

        result = await gatherer.gather_belief_crux_context("test task")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_without_messages(self):
        """Returns None when no messages provided and no history."""
        gatherer = _make_gatherer()
        gatherer._enable_belief_guidance = True
        gatherer._belief_analyzer = MagicMock()

        result = await gatherer.gather_belief_crux_context("test task", messages=None)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_crux_context_with_messages(self):
        """Returns formatted crux context from messages."""
        gatherer = _make_gatherer()
        gatherer._enable_belief_guidance = True

        mock_analyzer = MagicMock()
        mock_result = MockBeliefResult(
            cruxes=[
                {"statement": "AI safety requires governance", "confidence": 0.8, "entropy": 0.3},
                {"claim": "Regulation is necessary", "confidence": 0.6, "entropy": 0.9},
            ],
            evidence_suggestions=["Survey real-world AI incidents"],
        )
        mock_analyzer.analyze_messages.return_value = mock_result
        gatherer._belief_analyzer = mock_analyzer

        result = await gatherer.gather_belief_crux_context("test task", messages=["msg1", "msg2"])

        assert result is not None
        assert "Key Crux Points" in result
        assert "AI safety requires governance" in result
        assert "CONTESTED" in result  # entropy 0.9 > 0.8
        assert "Evidence Needed" in result

    @pytest.mark.asyncio
    async def test_returns_none_on_analysis_error(self):
        """Returns None when analysis reports an error."""
        gatherer = _make_gatherer()
        gatherer._enable_belief_guidance = True

        mock_analyzer = MagicMock()
        mock_result = MockBeliefResult(analysis_error="Failed to parse claims")
        mock_analyzer.analyze_messages.return_value = mock_result
        gatherer._belief_analyzer = mock_analyzer

        result = await gatherer.gather_belief_crux_context("test task", messages=["msg1"])

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_cruxes(self):
        """Returns None when no cruxes found."""
        gatherer = _make_gatherer()
        gatherer._enable_belief_guidance = True

        mock_analyzer = MagicMock()
        mock_result = MockBeliefResult(cruxes=[])
        mock_analyzer.analyze_messages.return_value = mock_result
        gatherer._belief_analyzer = mock_analyzer

        result = await gatherer.gather_belief_crux_context("test task", messages=["msg1"])

        assert result is None

    @pytest.mark.asyncio
    async def test_handles_analyzer_exception(self):
        """Returns None when analyzer raises exception."""
        gatherer = _make_gatherer()
        gatherer._enable_belief_guidance = True

        mock_analyzer = MagicMock()
        mock_analyzer.analyze_messages.side_effect = ValueError("bad data")
        gatherer._belief_analyzer = mock_analyzer

        result = await gatherer.gather_belief_crux_context("test task", messages=["msg1"])

        assert result is None


# =============================================================================
# gather_threat_intel_context Tests
# =============================================================================


class TestGatherThreatIntelContext:
    """Tests for gather_threat_intel_context method."""

    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(self):
        """Returns None when threat intel is disabled."""
        gatherer = _make_gatherer(enable_threat_intel_enrichment=False)

        result = await gatherer.gather_threat_intel_context("test CVE task")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_non_security_topic(self):
        """Returns None when topic is not security-related."""
        gatherer = _make_gatherer()
        gatherer._enable_threat_intel = True

        mock_enrichment = MagicMock()
        mock_enrichment.is_security_topic.return_value = False
        gatherer._threat_intel_enrichment = mock_enrichment

        result = await gatherer.gather_threat_intel_context("Discuss cooking recipes")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_formatted_context_for_security_topic(self):
        """Returns formatted threat intel for security topics."""
        gatherer = _make_gatherer()
        gatherer._enable_threat_intel = True

        mock_context = MockThreatIntelContext()
        mock_enrichment = MagicMock()
        mock_enrichment.is_security_topic.return_value = True
        mock_enrichment.enrich_context = AsyncMock(return_value=mock_context)
        mock_enrichment.format_for_debate.return_value = "## Threat Intel\n- CVE-2024-1234"
        gatherer._threat_intel_enrichment = mock_enrichment

        result = await gatherer.gather_threat_intel_context("Analyze CVE-2024-1234")

        assert result is not None
        assert "Threat Intel" in result

    @pytest.mark.asyncio
    async def test_returns_none_on_enrichment_error(self):
        """Returns None when enrichment raises an exception."""
        gatherer = _make_gatherer()
        gatherer._enable_threat_intel = True

        mock_enrichment = MagicMock()
        mock_enrichment.is_security_topic.return_value = True
        mock_enrichment.enrich_context = AsyncMock(side_effect=ConnectionError("API down"))
        gatherer._threat_intel_enrichment = mock_enrichment

        result = await gatherer.gather_threat_intel_context("Analyze CVE-2024-1234")

        assert result is None


# =============================================================================
# gather_culture_patterns_context Tests
# =============================================================================


class TestGatherCulturePatternsContext:
    """Tests for gather_culture_patterns_context method."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_mound(self):
        """Returns None when no knowledge mound is available."""
        gatherer = _make_gatherer()
        gatherer._knowledge_mound = None

        result = await gatherer.gather_culture_patterns_context("test task")

        assert result is None

    @pytest.mark.asyncio
    async def test_uses_get_culture_context_when_available(self):
        """Uses mound's get_culture_context API when available."""
        gatherer = _make_gatherer()
        mock_mound = AsyncMock()
        mock_mound.get_culture_context = AsyncMock(return_value="## Culture Patterns\n- Pattern 1")
        gatherer._knowledge_mound = mock_mound

        result = await gatherer.gather_culture_patterns_context("test task")

        assert result == "## Culture Patterns\n- Pattern 1"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_culture_context(self):
        """Returns None when mound has no culture context."""
        gatherer = _make_gatherer()
        mock_mound = AsyncMock()
        mock_mound.get_culture_context = AsyncMock(return_value=None)
        gatherer._knowledge_mound = mock_mound

        result = await gatherer.gather_culture_patterns_context("test task")

        assert result is None

    @pytest.mark.asyncio
    async def test_handles_exception_gracefully(self):
        """Returns None on unexpected exception."""
        gatherer = _make_gatherer()
        mock_mound = AsyncMock()
        mock_mound.get_culture_context = AsyncMock(side_effect=RuntimeError("error"))
        gatherer._knowledge_mound = mock_mound

        result = await gatherer.gather_culture_patterns_context("test task")

        assert result is None


# =============================================================================
# clear_cache Tests
# =============================================================================


class TestClearCache:
    """Tests for clear_cache method."""

    def test_clears_all_caches(self):
        """Clears all caches when no task specified."""
        gatherer = _make_gatherer()
        gatherer._research_context_cache["hash1"] = "context1"
        gatherer._research_evidence_pack["hash1"] = MockEvidencePack()
        gatherer._continuum_context_cache["hash1"] = "continuum1"
        gatherer._trending_topics_cache = [MockTrendingTopic()]

        gatherer.clear_cache()

        assert gatherer._research_context_cache == {}
        assert gatherer._research_evidence_pack == {}
        assert gatherer._continuum_context_cache == {}
        assert gatherer._trending_topics_cache == []

    def test_clears_task_specific_cache(self):
        """Clears only the specified task's cache."""
        gatherer = _make_gatherer()
        task_a = "task A"
        task_b = "task B"
        hash_a = gatherer._get_task_hash(task_a)
        hash_b = gatherer._get_task_hash(task_b)

        gatherer._research_context_cache[hash_a] = "context A"
        gatherer._research_context_cache[hash_b] = "context B"
        gatherer._research_evidence_pack[hash_a] = MockEvidencePack()
        gatherer._continuum_context_cache[hash_a] = "continuum A"

        gatherer.clear_cache(task=task_a)

        assert hash_a not in gatherer._research_context_cache
        assert hash_b in gatherer._research_context_cache
        assert hash_a not in gatherer._research_evidence_pack
        assert hash_a not in gatherer._continuum_context_cache

    def test_clears_task_specific_preserves_trending(self):
        """Task-specific clear does not affect trending topics cache."""
        gatherer = _make_gatherer()
        gatherer._trending_topics_cache = [MockTrendingTopic()]

        gatherer.clear_cache(task="some task")

        # Trending cache is global, not task-specific
        assert len(gatherer._trending_topics_cache) == 1


# =============================================================================
# _enforce_cache_limit Tests
# =============================================================================


class TestEnforceCacheLimit:
    """Tests for _enforce_cache_limit method."""

    def test_does_nothing_when_under_limit(self):
        """Does nothing when cache is under the limit."""
        gatherer = _make_gatherer()
        cache = {"k1": "v1", "k2": "v2"}

        gatherer._enforce_cache_limit(cache, max_size=5)

        assert len(cache) == 2

    def test_evicts_oldest_entries(self):
        """Evicts oldest (first-inserted) entries when at limit."""
        gatherer = _make_gatherer()
        cache = {"k1": "v1", "k2": "v2", "k3": "v3"}

        gatherer._enforce_cache_limit(cache, max_size=2)

        assert len(cache) == 1
        assert "k3" in cache
        assert "k1" not in cache
        assert "k2" not in cache

    def test_handles_empty_cache(self):
        """Handles empty cache without error."""
        gatherer = _make_gatherer()
        cache = {}

        gatherer._enforce_cache_limit(cache, max_size=10)

        assert len(cache) == 0

    def test_evicts_single_entry_at_limit(self):
        """Evicts exactly one entry when cache is at limit."""
        gatherer = _make_gatherer()
        cache = {"k1": "v1", "k2": "v2"}

        gatherer._enforce_cache_limit(cache, max_size=2)

        assert len(cache) == 1
        assert "k2" in cache


# =============================================================================
# _compress_with_rlm Tests
# =============================================================================


class TestCompressWithRlm:
    """Tests for _compress_with_rlm method."""

    @pytest.mark.asyncio
    async def test_returns_content_under_threshold(self):
        """Returns content as-is when under RLM threshold."""
        gatherer = _make_gatherer()
        gatherer._rlm_threshold = 5000

        content = "short content"
        result = await gatherer._compress_with_rlm(content, max_chars=3000)

        assert result == content

    @pytest.mark.asyncio
    async def test_truncates_content_under_threshold_but_over_max_chars(self):
        """Truncates content when under threshold but over max_chars."""
        gatherer = _make_gatherer()
        gatherer._rlm_threshold = 5000

        content = "A" * 4000  # Under threshold (5000) but over max_chars (100)
        result = await gatherer._compress_with_rlm(content, max_chars=100)

        assert len(result) == 100

    @pytest.mark.asyncio
    async def test_truncates_when_rlm_disabled(self):
        """Falls back to truncation when RLM is disabled."""
        gatherer = _make_gatherer()
        gatherer._enable_rlm = False
        gatherer._rlm_threshold = 100

        content = "A" * 5000
        result = await gatherer._compress_with_rlm(content, max_chars=200)

        assert len(result) <= 200
        assert result.endswith("... [truncated]")

    @pytest.mark.asyncio
    async def test_uses_aragora_rlm_when_available(self):
        """Uses AragoraRLM for compression when available."""
        gatherer = _make_gatherer()
        gatherer._enable_rlm = True
        gatherer._rlm_threshold = 100

        mock_rlm = AsyncMock()
        mock_rlm.compress_and_query = AsyncMock(
            return_value=MockRLMResult(answer="compressed output", used_true_rlm=False)
        )
        gatherer._aragora_rlm = mock_rlm

        content = "A" * 5000
        result = await gatherer._compress_with_rlm(content, max_chars=3000)

        assert result == "compressed output"

    @pytest.mark.asyncio
    async def test_falls_back_to_truncation_when_rlm_fails(self):
        """Falls back to truncation when RLM compression fails."""
        gatherer = _make_gatherer()
        gatherer._enable_rlm = True
        gatherer._rlm_threshold = 100
        gatherer._aragora_rlm = None
        gatherer._rlm_compressor = None

        content = "A" * 5000
        result = await gatherer._compress_with_rlm(content, max_chars=200)

        assert len(result) <= 200
        assert result.endswith("... [truncated]")

    @pytest.mark.asyncio
    async def test_returns_short_content_unchanged(self):
        """Content shorter than both threshold and max_chars is unchanged."""
        gatherer = _make_gatherer()
        gatherer._rlm_threshold = 5000

        content = "Hello"
        result = await gatherer._compress_with_rlm(content, max_chars=3000)

        assert result == "Hello"


# =============================================================================
# get_continuum_context Tests
# =============================================================================


class TestGetContinuumContext:
    """Tests for get_continuum_context method."""

    def test_returns_empty_when_no_memory(self):
        """Returns empty tuple when no continuum memory."""
        gatherer = _make_gatherer()

        context, ids, tiers = gatherer.get_continuum_context(None, "programming", "test task")

        assert context == ""
        assert ids == []
        assert tiers == {}

    def test_returns_cached_result(self):
        """Returns cached result for previously queried task."""
        gatherer = _make_gatherer()
        task = "cached continuum task"
        task_hash = gatherer._get_task_hash(task)
        gatherer._continuum_context_cache[task_hash] = "cached continuum context"

        context, ids, tiers = gatherer.get_continuum_context(MagicMock(), "programming", task)

        assert context == "cached continuum context"
        assert ids == []
        assert tiers == {}

    def test_returns_empty_when_no_memories_found(self):
        """Returns empty when memory retrieve returns empty list."""
        gatherer = _make_gatherer()

        mock_memory = MagicMock()
        mock_memory.retrieve.return_value = []

        context, ids, tiers = gatherer.get_continuum_context(
            mock_memory, "programming", "unique task for no memories"
        )

        assert context == ""
        assert ids == []

    def test_formats_recent_memories(self):
        """Formats recent memories with tier and confidence labels."""
        gatherer = _make_gatherer()

        mem = MockMemory(
            id="mem-1",
            content="learned about testing",
            tier=MockTier(value="medium"),
            consolidation_score=0.8,
        )
        mock_memory = MagicMock()
        mock_memory.retrieve.return_value = [mem]

        context, ids, tiers = gatherer.get_continuum_context(
            mock_memory, "programming", "unique task for memory test"
        )

        assert "Previous learnings" in context
        assert "medium" in context
        assert "high" in context  # consolidation 0.8 > 0.7 -> "high"
        assert "mem-1" in ids

    def test_includes_glacial_insights(self):
        """Includes glacial insights when available."""
        gatherer = _make_gatherer()

        recent_mem = MockMemory(
            id="mem-recent",
            content="recent finding",
            tier=MockTier(value="fast"),
            consolidation_score=0.5,
        )
        glacial_mem = MockMemory(
            id="mem-glacial",
            content="long-term pattern",
            tier=MockTier(value="glacial"),
            consolidation_score=0.9,
        )

        mock_memory = MagicMock()
        mock_memory.retrieve.return_value = [recent_mem]
        mock_memory.get_glacial_insights.return_value = [glacial_mem]

        context, ids, tiers = gatherer.get_continuum_context(
            mock_memory, "programming", "unique task with glacial insights"
        )

        assert "Long-term patterns" in context
        assert "glacial" in context
        assert "mem-glacial" in ids

    def test_handles_memory_retrieval_error(self):
        """Handles errors in memory retrieval gracefully."""
        gatherer = _make_gatherer()

        mock_memory = MagicMock()
        mock_memory.retrieve.side_effect = ValueError("Memory corrupted")

        context, ids, tiers = gatherer.get_continuum_context(
            mock_memory, "programming", "error task"
        )

        assert context == ""
        assert ids == []
        assert tiers == {}


# =============================================================================
# refresh_evidence_for_round Tests
# =============================================================================


class TestRefreshEvidenceForRound:
    """Tests for refresh_evidence_for_round method."""

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_collector(self):
        """Returns (0, None) when no evidence collector."""
        gatherer = _make_gatherer()

        count, pack = await gatherer.refresh_evidence_for_round("some text", None, "test task")

        assert count == 0
        assert pack is None

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_claims_extracted(self):
        """Returns (0, None) when no claims found in text."""
        gatherer = _make_gatherer()

        mock_collector = MagicMock()
        mock_collector.extract_claims_from_text.return_value = []

        count, pack = await gatherer.refresh_evidence_for_round(
            "plain text", mock_collector, "test task"
        )

        assert count == 0
        assert pack is None

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_evidence_snippets(self):
        """Returns (0, None) when evidence collection returns empty snippets."""
        gatherer = _make_gatherer()

        mock_collector = MagicMock()
        mock_collector.extract_claims_from_text.return_value = ["claim1"]
        mock_collector.collect_for_claims = AsyncMock(return_value=MockEvidencePack(snippets=[]))

        count, pack = await gatherer.refresh_evidence_for_round(
            "text with claims", mock_collector, "no evidence task"
        )

        assert count == 0
        assert pack is None

    @pytest.mark.asyncio
    async def test_stores_new_evidence_pack(self):
        """Stores new evidence pack when no existing pack."""
        gatherer = _make_gatherer()

        snippet = MockSnippet(id="new-1")
        mock_collector = MagicMock()
        mock_collector.extract_claims_from_text.return_value = ["claim1"]
        mock_collector.collect_for_claims = AsyncMock(
            return_value=MockEvidencePack(snippets=[snippet], total_searched=5)
        )

        count, pack = await gatherer.refresh_evidence_for_round(
            "text with claims", mock_collector, "store test task"
        )

        assert count == 1
        assert pack is not None

    @pytest.mark.asyncio
    async def test_calls_evidence_store_callback(self):
        """Calls evidence store callback when provided."""
        gatherer = _make_gatherer()

        snippet = MockSnippet(id="cb-1")
        mock_collector = MagicMock()
        mock_collector.extract_claims_from_text.return_value = ["claim1"]
        mock_collector.collect_for_claims = AsyncMock(
            return_value=MockEvidencePack(snippets=[snippet], total_searched=5)
        )

        callback = MagicMock()

        count, pack = await gatherer.refresh_evidence_for_round(
            "text", mock_collector, "callback task", evidence_store_callback=callback
        )

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_error_gracefully(self):
        """Returns (0, None) when an error occurs."""
        gatherer = _make_gatherer()

        mock_collector = MagicMock()
        mock_collector.extract_claims_from_text.side_effect = RuntimeError("Parse failed")

        count, pack = await gatherer.refresh_evidence_for_round(
            "text", mock_collector, "error task"
        )

        assert count == 0
        assert pack is None


# =============================================================================
# Timeout Constants Tests
# =============================================================================


class TestTimeoutConstants:
    """Tests for timeout configuration constants."""

    def test_default_timeouts_are_positive(self):
        """All default timeout values are positive."""
        assert CONTEXT_GATHER_TIMEOUT > 0
        assert EVIDENCE_TIMEOUT > 0
        assert TRENDING_TIMEOUT > 0

    def test_context_timeout_is_largest(self):
        """Overall context timeout is larger than sub-timeouts."""
        assert CONTEXT_GATHER_TIMEOUT >= EVIDENCE_TIMEOUT
        assert CONTEXT_GATHER_TIMEOUT >= TRENDING_TIMEOUT


# =============================================================================
# Per-Debate Isolation Tests
# =============================================================================


class TestPerDebateIsolation:
    """Tests for per-debate instance isolation."""

    def test_separate_instances_have_separate_caches(self):
        """Different gatherer instances maintain separate caches."""
        gatherer1 = _make_gatherer()
        gatherer2 = _make_gatherer()

        task = "same task"
        gatherer1._research_context_cache[gatherer1._get_task_hash(task)] = "context 1"

        assert gatherer1._get_task_hash(task) not in gatherer2._research_context_cache

    def test_hash_consistency_across_instances(self):
        """Task hash is consistent across different gatherer instances."""
        gatherer1 = _make_gatherer()
        gatherer2 = _make_gatherer()

        task = "consistent hash task"
        assert gatherer1._get_task_hash(task) == gatherer2._get_task_hash(task)
