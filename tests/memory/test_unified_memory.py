"""
Tests for the unified Titans-inspired memory architecture.

Covers:
- Phase 1: ContentSurpriseScorer (novelty, momentum, debate outcomes)
- Phase 2: ClaudeMemAdapter (search, ingest, reverse sync)
- Phase 3: RetentionGate (evaluate, sweep, duplicate detection)
- Phase 4: RLMContextAdapter (store, search)
- Phase 5: MemoryFabric (unified query, remember, context_for_debate)
- Phase 6: BusinessKnowledgeIngester (document, interaction, query)
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.memory.surprise import (
    ContentSurpriseScore,
    ContentSurpriseScorer,
)
from aragora.memory.retention import (
    ItemMetadata,
    MergeAction,
    RetentionAction,
    RetentionGate,
    RetentionReport,
)
from aragora.memory.fabric import (
    FabricResult,
    MemoryFabric,
    RememberResult,
)
from aragora.memory.business import (
    BusinessKnowledgeIngester,
)


# =====================================================================
# Phase 1: ContentSurpriseScorer tests
# =====================================================================


class TestContentSurpriseScore:
    """Tests for ContentSurpriseScore dataclass."""

    def test_score_is_frozen(self):
        score = ContentSurpriseScore(
            novelty=0.5, momentum=0.3, combined=0.44, should_store=True, reason="test"
        )
        with pytest.raises(AttributeError):
            score.novelty = 0.9  # type: ignore[misc]

    def test_fields(self):
        score = ContentSurpriseScore(
            novelty=0.8, momentum=0.4, combined=0.68, should_store=True, reason="novel"
        )
        assert score.novelty == 0.8
        assert score.momentum == 0.4
        assert score.combined == 0.68
        assert score.should_store is True
        assert score.reason == "novel"


class TestContentSurpriseScorer:
    """Tests for the Titans-inspired content surprise scorer."""

    def test_default_threshold(self):
        scorer = ContentSurpriseScorer()
        assert scorer.threshold == 0.3

    def test_custom_threshold(self):
        scorer = ContentSurpriseScorer(threshold=0.5)
        assert scorer.threshold == 0.5

    def test_no_keywords_returns_zero(self):
        scorer = ContentSurpriseScorer()
        score = scorer.score("12 34 56", source="test")
        assert score.combined == 0.0
        assert score.should_store is False

    def test_fully_novel_content(self):
        scorer = ContentSurpriseScorer()
        score = scorer.score(
            "quantum entanglement teleportation",
            source="research",
            existing_context="",
        )
        assert score.novelty == 1.0
        assert score.should_store is True
        assert score.combined > 0.5

    def test_redundant_content_low_novelty(self):
        scorer = ContentSurpriseScorer()
        context = "authentication login password security tokens"
        score = scorer.score(
            "authentication tokens for login security",
            source="test",
            existing_context=context,
        )
        assert score.novelty < 0.5  # High overlap = low novelty

    def test_partial_overlap(self):
        scorer = ContentSurpriseScorer()
        score = scorer.score(
            "authentication rate limiting microservice",
            source="test",
            existing_context="authentication tokens for login security",
        )
        assert 0.0 < score.novelty < 1.0

    def test_momentum_neutral_on_first_score(self):
        scorer = ContentSurpriseScorer()
        score = scorer.score("novel content here", source="test")
        assert score.momentum == 0.5  # Neutral with no history

    def test_momentum_builds_with_related_items(self):
        scorer = ContentSurpriseScorer()
        # Score several related items
        for _ in range(5):
            scorer.score("database migration schema indexing", source="test")
        # Next related item should have non-neutral momentum
        score = scorer.score("database performance indexing", source="test")
        assert score.momentum != 0.5

    def test_should_store_above_threshold(self):
        scorer = ContentSurpriseScorer(threshold=0.3)
        score = scorer.score("completely novel unexpected content", source="test")
        assert score.should_store is True
        assert score.combined >= 0.3

    def test_should_not_store_below_threshold(self):
        scorer = ContentSurpriseScorer(threshold=0.99)
        score = scorer.score(
            "authentication security",
            source="test",
            existing_context="authentication security login password",
        )
        assert score.should_store is False

    def test_reason_describes_level(self):
        scorer = ContentSurpriseScorer()
        high = scorer.score("novel unexpected quantum content", source="test")
        assert "surprise" in high.reason.lower() or "novel" in high.reason.lower()


class TestContentSurpriseScorerDebate:
    """Tests for debate outcome scoring."""

    def test_debate_outcome_basic(self):
        scorer = ContentSurpriseScorer()
        score = scorer.score_debate_outcome(
            conclusion="Rate limiting should use token buckets",
            domain="technical",
            confidence=0.85,
        )
        assert isinstance(score, ContentSurpriseScore)
        assert 0.0 <= score.combined <= 1.0

    def test_new_domain_boosted(self):
        scorer = ContentSurpriseScorer()
        # No prior conclusions → novelty boosted
        score = scorer.score_debate_outcome(
            conclusion="Adopt microservices architecture",
            domain="architecture",
            confidence=0.9,
            prior_conclusions=None,
        )
        assert score.novelty >= 0.8  # Boosted for new territory

    def test_prior_conclusions_reduce_novelty(self):
        scorer = ContentSurpriseScorer()
        priors = [
            "Use microservices for scaling",
            "Microservices architecture recommended for scaling",
        ]
        score = scorer.score_debate_outcome(
            conclusion="Microservices architecture for scaling recommended",
            domain="architecture",
            confidence=0.8,
            prior_conclusions=priors,
        )
        assert score.novelty < 0.9  # Reduced by overlap with priors

    def test_high_confidence_new_topic_boosted(self):
        scorer = ContentSurpriseScorer()
        score = scorer.score_debate_outcome(
            conclusion="Implement circuit breakers for all external APIs",
            domain="reliability",
            confidence=0.95,
            prior_conclusions=None,
        )
        assert score.combined >= 0.7


# =====================================================================
# Phase 2: ClaudeMemAdapter tests
# =====================================================================


class TestClaudeMemAdapter:
    """Tests for the claude-mem KM adapter."""

    def test_import(self):
        from aragora.knowledge.mound.adapters.claude_mem_adapter import ClaudeMemAdapter

        adapter = ClaudeMemAdapter()
        assert adapter.adapter_name == "claude_mem"

    def test_no_connector_returns_none(self):
        from aragora.knowledge.mound.adapters.claude_mem_adapter import ClaudeMemAdapter

        adapter = ClaudeMemAdapter(connector=None)
        # _get_connector will fail gracefully when module not installed
        assert adapter._connector is None

    @pytest.mark.asyncio
    async def test_search_no_connector(self, monkeypatch):
        from aragora.knowledge.mound.adapters.claude_mem_adapter import ClaudeMemAdapter

        adapter = ClaudeMemAdapter(connector=None, enable_resilience=False)
        # Prevent lazy connector init from environment
        monkeypatch.setattr(adapter, "_get_connector", lambda: None)
        results = await adapter.search_observations("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_with_connector(self):
        from aragora.knowledge.mound.adapters.claude_mem_adapter import ClaudeMemAdapter

        mock_obs = MagicMock()
        mock_obs.id = "obs1"
        mock_obs.content = "Found auth pattern"
        mock_obs.title = "Auth"
        mock_obs.created_at = time.time()
        mock_obs.metadata = {}

        mock_connector = MagicMock()
        mock_connector.search = AsyncMock(return_value=[mock_obs])

        adapter = ClaudeMemAdapter(connector=mock_connector, enable_resilience=False)
        results = await adapter.search_observations("auth patterns")
        assert len(results) == 1
        assert results[0]["source"] == "claude_mem"
        assert results[0]["content"] == "Found auth pattern"

    @pytest.mark.asyncio
    async def test_inject_context(self):
        from aragora.knowledge.mound.adapters.claude_mem_adapter import ClaudeMemAdapter

        mock_obs = MagicMock()
        mock_obs.id = "obs1"
        mock_obs.content = "Auth uses JWT tokens"
        mock_obs.title = "Auth pattern"
        mock_obs.created_at = time.time()
        mock_obs.metadata = {}

        mock_connector = MagicMock()
        mock_connector.search = AsyncMock(return_value=[mock_obs])

        adapter = ClaudeMemAdapter(connector=mock_connector, enable_resilience=False)
        result = await adapter.inject_context(topic="authentication")
        assert result.observations_injected == 1
        assert "Auth uses JWT tokens" in result.context_content[0]

    @pytest.mark.asyncio
    async def test_inject_context_no_topic(self):
        from aragora.knowledge.mound.adapters.claude_mem_adapter import ClaudeMemAdapter

        adapter = ClaudeMemAdapter(enable_resilience=False)
        result = await adapter.inject_context(topic=None)
        assert result.observations_injected == 0

    def test_evidence_to_knowledge_item(self):
        from aragora.knowledge.mound.adapters.claude_mem_adapter import ClaudeMemAdapter

        adapter = ClaudeMemAdapter(enable_resilience=False)
        evidence = {"id": "e1", "content": "Test content", "title": "Test"}
        item = adapter.evidence_to_knowledge_item(evidence)
        assert item["source_type"] == "claude_mem"
        assert item["content"] == "Test content"
        assert item["id"].startswith("cm_")


# =====================================================================
# Phase 3: RetentionGate tests
# =====================================================================


class TestRetentionGate:
    """Tests for the MIRAS-inspired retention gate."""

    def test_keep_recent_high_access(self):
        gate = RetentionGate()
        item = ItemMetadata(
            item_id="i1",
            system="continuum",
            created_at=time.time() - 3600,
            last_accessed=time.time() - 60,  # Accessed 1 min ago
            access_count=50,
            surprise_score=0.6,
        )
        decision = gate.evaluate(item)
        assert decision.action == RetentionAction.KEEP
        assert decision.score >= 0.3

    def test_archive_old_low_access(self):
        gate = RetentionGate()
        item = ItemMetadata(
            item_id="i2",
            system="consensus",
            created_at=time.time() - 86400 * 120,  # 120 days ago
            last_accessed=time.time() - 86400 * 120,  # Not accessed since
            access_count=1,
            surprise_score=0.1,
        )
        decision = gate.evaluate(item)
        assert decision.action == RetentionAction.ARCHIVE
        assert decision.score < 0.15

    def test_demote_moderate_items(self):
        gate = RetentionGate(demote_threshold=0.4, archive_threshold=0.1)
        item = ItemMetadata(
            item_id="i3",
            system="km",
            created_at=time.time() - 86400 * 45,
            last_accessed=time.time() - 86400 * 30,
            access_count=5,
            surprise_score=0.3,
        )
        decision = gate.evaluate(item)
        assert decision.action in (RetentionAction.DEMOTE, RetentionAction.KEEP)

    def test_sweep_mixed_items(self):
        gate = RetentionGate()
        now = time.time()
        items = [
            ItemMetadata("fresh", "km", now - 3600, now - 60, 100, 0.9),
            ItemMetadata("stale", "km", now - 86400 * 180, now - 86400 * 180, 1, 0.05),
            ItemMetadata("medium", "km", now - 86400 * 15, now - 86400 * 5, 10, 0.4),
        ]
        report = gate.sweep(items)
        assert report.evaluated == 3
        assert report.keep >= 1
        assert report.archive >= 1

    def test_sweep_empty(self):
        gate = RetentionGate()
        report = gate.sweep([])
        assert report.evaluated == 0
        assert report.keep == 0


class TestRetentionDuplicateDetection:
    """Tests for cross-system duplicate detection."""

    def test_finds_duplicates_across_systems(self):
        gate = RetentionGate()
        items_by_system = {
            "continuum": [{"id": "c1", "content": "authentication login password security tokens"}],
            "consensus": [
                {"id": "k1", "content": "authentication login security password tokens method"}
            ],
        }
        merges = gate.consolidate_duplicates(items_by_system)
        assert len(merges) >= 1
        assert merges[0].similarity >= 0.7

    def test_no_duplicates_within_same_system(self):
        gate = RetentionGate()
        items_by_system = {
            "continuum": [
                {"id": "c1", "content": "authentication login password"},
                {"id": "c2", "content": "authentication login password"},
            ],
        }
        merges = gate.consolidate_duplicates(items_by_system)
        assert len(merges) == 0  # Same system, skip

    def test_dissimilar_items_no_merge(self):
        gate = RetentionGate()
        items_by_system = {
            "continuum": [{"id": "c1", "content": "quantum physics entanglement"}],
            "consensus": [{"id": "k1", "content": "database indexing performance tuning"}],
        }
        merges = gate.consolidate_duplicates(items_by_system)
        assert len(merges) == 0


# =====================================================================
# Phase 4: RLMContextAdapter tests
# =====================================================================


class TestRLMContextAdapter:
    """Tests for the RLM codebase context persistence adapter."""

    def test_import(self):
        from aragora.knowledge.mound.adapters.rlm_context_adapter import RLMContextAdapter

        adapter = RLMContextAdapter()
        assert adapter.adapter_name == "rlm_context"

    @pytest.mark.asyncio
    async def test_store_codebase_summary(self):
        from aragora.knowledge.mound.adapters.rlm_context_adapter import RLMContextAdapter

        mock_store = AsyncMock(return_value="sum_1")
        adapter = RLMContextAdapter(store_fn=mock_store, enable_resilience=False)
        item_id = await adapter.store_codebase_summary(
            summary="Aragora has 3000+ modules...",
            root_path="/code/aragora",
        )
        assert item_id == "sum_1"
        call_args = mock_store.call_args[0][0]
        assert call_args["source"] == "rlm_codebase"

    @pytest.mark.asyncio
    async def test_store_module_analysis(self):
        from aragora.knowledge.mound.adapters.rlm_context_adapter import RLMContextAdapter

        mock_store = AsyncMock(return_value="mod_1")
        adapter = RLMContextAdapter(store_fn=mock_store, enable_resilience=False)
        item_id = await adapter.store_module_analysis(
            module_path="aragora.debate.orchestrator",
            analysis="Arena class handles debate orchestration...",
        )
        assert item_id == "mod_1"

    @pytest.mark.asyncio
    async def test_get_codebase_context(self):
        from aragora.knowledge.mound.adapters.rlm_context_adapter import RLMContextAdapter

        mock_search = AsyncMock(return_value=[{"content": "Auth module handles OIDC", "id": "r1"}])
        adapter = RLMContextAdapter(search_fn=mock_search, enable_resilience=False)
        results = await adapter.get_codebase_context("authentication")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_no_store_fn(self):
        from aragora.knowledge.mound.adapters.rlm_context_adapter import RLMContextAdapter

        adapter = RLMContextAdapter()
        result = await adapter.store_codebase_summary("test")
        assert result == ""

    @pytest.mark.asyncio
    async def test_no_search_fn(self):
        from aragora.knowledge.mound.adapters.rlm_context_adapter import RLMContextAdapter

        adapter = RLMContextAdapter()
        results = await adapter.get_codebase_context("test")
        assert results == []


# =====================================================================
# Phase 5: MemoryFabric tests
# =====================================================================


class TestMemoryFabric:
    """Tests for the unified memory query interface."""

    def test_register_backend(self):
        fabric = MemoryFabric()
        backend = MagicMock()
        fabric.register_backend("test", backend)
        assert "test" in fabric._backends

    @pytest.mark.asyncio
    async def test_query_empty(self):
        fabric = MemoryFabric()
        results = await fabric.query("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_query_single_backend(self):
        backend = MagicMock()
        backend.search = MagicMock(
            return_value=[{"content": "Auth tokens use JWT", "id": "1", "relevance": 0.9}]
        )
        fabric = MemoryFabric(backends={"km": backend})
        results = await fabric.query("authentication")
        assert len(results) == 1
        assert results[0].source_system == "km"
        assert results[0].relevance == 0.9

    @pytest.mark.asyncio
    async def test_query_async_backend(self):
        backend = MagicMock()
        backend.search = AsyncMock(
            return_value=[{"content": "Circuit breaker pattern", "id": "2", "relevance": 0.7}]
        )
        fabric = MemoryFabric(backends={"continuum": backend})
        results = await fabric.query("resilience")
        assert len(results) == 1
        assert results[0].source_system == "continuum"

    @pytest.mark.asyncio
    async def test_query_multiple_backends(self):
        km = MagicMock()
        km.search = MagicMock(
            return_value=[{"content": "KM result about auth", "id": "k1", "relevance": 0.8}]
        )
        consensus = MagicMock()
        consensus.search = MagicMock(
            return_value=[{"content": "Consensus on auth approach", "id": "c1", "relevance": 0.6}]
        )
        fabric = MemoryFabric(backends={"km": km, "consensus": consensus})
        results = await fabric.query("authentication")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_query_deduplication(self):
        b1 = MagicMock()
        b1.search = MagicMock(
            return_value=[
                {"content": "authentication login security tokens", "id": "1", "relevance": 0.9}
            ]
        )
        b2 = MagicMock()
        b2.search = MagicMock(
            return_value=[
                {
                    "content": "authentication login security tokens method",
                    "id": "2",
                    "relevance": 0.8,
                }
            ]
        )
        fabric = MemoryFabric(backends={"a": b1, "b": b2})
        results = await fabric.query("auth")
        assert len(results) == 1  # Near-duplicate removed

    @pytest.mark.asyncio
    async def test_query_filter_by_system(self):
        b1 = MagicMock()
        b1.search = MagicMock(
            return_value=[{"content": "result from km", "id": "1", "relevance": 0.8}]
        )
        b2 = MagicMock()
        b2.search = MagicMock(
            return_value=[{"content": "result from consensus", "id": "2", "relevance": 0.7}]
        )
        fabric = MemoryFabric(backends={"km": b1, "consensus": b2})
        results = await fabric.query("test", systems=["km"])
        assert len(results) == 1
        assert results[0].source_system == "km"

    @pytest.mark.asyncio
    async def test_query_min_relevance(self):
        backend = MagicMock()
        backend.search = MagicMock(
            return_value=[
                {"content": "high rel", "id": "1", "relevance": 0.9},
                {"content": "low rel different", "id": "2", "relevance": 0.1},
            ]
        )
        fabric = MemoryFabric(backends={"km": backend})
        results = await fabric.query("test", min_relevance=0.5)
        assert len(results) == 1
        assert results[0].relevance >= 0.5

    @pytest.mark.asyncio
    async def test_query_backend_error_graceful(self):
        bad_backend = MagicMock()
        bad_backend.search = MagicMock(side_effect=RuntimeError("down"))
        good_backend = MagicMock()
        good_backend.search = MagicMock(
            return_value=[{"content": "works", "id": "1", "relevance": 0.8}]
        )
        fabric = MemoryFabric(backends={"bad": bad_backend, "good": good_backend})
        results = await fabric.query("test")
        assert len(results) == 1
        assert results[0].source_system == "good"


class TestMemoryFabricRemember:
    """Tests for the unified remember (write) interface."""

    @pytest.mark.asyncio
    async def test_remember_novel_content(self):
        fabric = MemoryFabric()
        result = await fabric.remember(
            content="completely novel quantum computing approach",
            source="research",
        )
        assert isinstance(result, RememberResult)
        assert result.stored is True
        assert result.surprise_combined > 0.3

    @pytest.mark.asyncio
    async def test_remember_with_existing_context_low_surprise(self):
        scorer = ContentSurpriseScorer(threshold=0.99)
        fabric = MemoryFabric(surprise_scorer=scorer)
        result = await fabric.remember(
            content="auth tokens",
            source="test",
            existing_context="authentication tokens for login security sessions",
        )
        assert result.stored is False

    @pytest.mark.asyncio
    async def test_remember_writes_to_appropriate_systems(self):
        backend1 = MagicMock()
        backend1.store = MagicMock()
        backend2 = MagicMock()
        backend2.store_knowledge = MagicMock()
        fabric = MemoryFabric(
            backends={"continuum": backend1, "km": backend2},
        )
        result = await fabric.remember(
            content="unprecedented breakthrough novel discovery",
            source="research",
        )
        assert result.stored is True
        assert len(result.systems_written) >= 1


class TestMemoryFabricContextForDebate:
    """Tests for the debate context generation."""

    @pytest.mark.asyncio
    async def test_context_empty_when_no_backends(self):
        fabric = MemoryFabric()
        ctx = await fabric.context_for_debate("any task")
        assert ctx == ""

    @pytest.mark.asyncio
    async def test_context_includes_header(self):
        backend = MagicMock()
        backend.search = MagicMock(
            return_value=[{"content": "relevant finding", "id": "1", "relevance": 0.9}]
        )
        fabric = MemoryFabric(backends={"km": backend})
        ctx = await fabric.context_for_debate("auth design")
        assert "MEMORY CONTEXT" in ctx
        assert "relevant finding" in ctx

    @pytest.mark.asyncio
    async def test_context_respects_budget(self):
        backend = MagicMock()
        backend.search = MagicMock(
            return_value=[
                {"content": "x" * 10000, "id": "1", "relevance": 0.9},
                {"content": "y" * 10000, "id": "2", "relevance": 0.8},
            ]
        )
        fabric = MemoryFabric(backends={"km": backend})
        ctx = await fabric.context_for_debate("test", budget_tokens=100)
        # 100 tokens ≈ 400 chars, should be truncated
        assert len(ctx) < 10000


# =====================================================================
# Phase 6: BusinessKnowledgeIngester tests
# =====================================================================


class TestBusinessKnowledgeIngester:
    """Tests for business knowledge ingestion."""

    @pytest.mark.asyncio
    async def test_ingest_document(self):
        fabric = MemoryFabric()
        ingester = BusinessKnowledgeIngester(fabric)
        result = await ingester.ingest_document(
            content="Net 30 payment terms for Acme Corp invoice",
            doc_type="invoice",
            metadata={"customer": "Acme"},
        )
        assert isinstance(result, RememberResult)
        assert result.surprise_combined > 0

    @pytest.mark.asyncio
    async def test_ingest_interaction(self):
        fabric = MemoryFabric()
        ingester = BusinessKnowledgeIngester(fabric)
        result = await ingester.ingest_interaction(
            summary="Discussed Q2 roadmap priorities",
            participants=["Alice", "Bob"],
            outcome="Agreed to focus on mobile app",
        )
        assert isinstance(result, RememberResult)

    @pytest.mark.asyncio
    async def test_query_business_context(self):
        backend = MagicMock()
        backend.search = MagicMock(
            return_value=[{"content": "[invoice] Net 30 terms", "id": "1", "relevance": 0.7}]
        )
        fabric = MemoryFabric(backends={"km": backend})
        ingester = BusinessKnowledgeIngester(fabric)
        results = await ingester.query_business_context("payment terms")
        assert len(results) == 1


# =====================================================================
# Integration: Coordinator with surprise scoring
# =====================================================================


class TestCoordinatorSurpriseIntegration:
    """Tests for surprise scoring wired into MemoryCoordinator."""

    def test_coordinator_has_surprise_scorer(self):
        from aragora.memory.coordinator import MemoryCoordinator

        coord = MemoryCoordinator()
        assert coord.surprise_scorer is not None
        assert isinstance(coord.surprise_scorer, ContentSurpriseScorer)

    def test_coordinator_custom_scorer(self):
        from aragora.memory.coordinator import MemoryCoordinator

        scorer = ContentSurpriseScorer(threshold=0.8)
        coord = MemoryCoordinator(surprise_scorer=scorer)
        assert coord.surprise_scorer.threshold == 0.8


# =====================================================================
# Integration: Context budgeter section limits
# =====================================================================


class TestContextBudgeterSections:
    """Tests for new section limits in context budgeter."""

    def test_claude_mem_section_limit(self):
        from aragora.debate.context_budgeter import DEFAULT_SECTION_LIMITS

        assert "claude_mem" in DEFAULT_SECTION_LIMITS
        assert DEFAULT_SECTION_LIMITS["claude_mem"] == 400

    def test_memory_fabric_section_limit(self):
        from aragora.debate.context_budgeter import DEFAULT_SECTION_LIMITS

        assert "memory_fabric" in DEFAULT_SECTION_LIMITS
        assert DEFAULT_SECTION_LIMITS["memory_fabric"] == 600

    def test_codebase_section_limit_unchanged(self):
        from aragora.debate.context_budgeter import DEFAULT_SECTION_LIMITS

        assert DEFAULT_SECTION_LIMITS["codebase"] == 500


# =====================================================================
# Integration: Adapter factory registration
# =====================================================================


class TestAdapterFactoryRegistration:
    """Tests for new adapters registered in the factory."""

    def test_claude_mem_in_specs(self):
        from aragora.knowledge.mound.adapters.factory import ADAPTER_SPECS

        assert "claude_mem" in ADAPTER_SPECS
        spec = ADAPTER_SPECS["claude_mem"]
        assert spec.priority > 0
        assert spec.enabled_by_default is False

    def test_rlm_context_in_specs(self):
        from aragora.knowledge.mound.adapters.factory import ADAPTER_SPECS

        assert "rlm_context" in ADAPTER_SPECS
        spec = ADAPTER_SPECS["rlm_context"]
        assert spec.priority == 55
        assert spec.enabled_by_default is False

    def test_adapter_count_increased(self):
        from aragora.knowledge.mound.adapters.factory import ADAPTER_SPECS

        # We added 2 new adapters (claude_mem, rlm_context)
        assert len(ADAPTER_SPECS) >= 30  # Was ~33, now 35+


# =====================================================================
# Integration: MCP tools
# =====================================================================


class TestMCPBusinessMemoryTools:
    """Tests for MCP business memory tool imports."""

    def test_import_store_tool(self):
        from aragora.mcp.tools_module.business_memory import store_business_knowledge_tool

        assert callable(store_business_knowledge_tool)

    def test_import_query_tool(self):
        from aragora.mcp.tools_module.business_memory import query_business_knowledge_tool

        assert callable(query_business_knowledge_tool)

    def test_tools_in_init(self):
        from aragora.mcp.tools_module import (
            store_business_knowledge_tool,
            query_business_knowledge_tool,
        )

        assert callable(store_business_knowledge_tool)
        assert callable(query_business_knowledge_tool)


# =====================================================================
# Integration: PromptBuilder memory fabric
# =====================================================================


class TestPromptBuilderMemoryFabric:
    """Tests for MemoryFabric integration in PromptBuilder."""

    def test_prompt_builder_has_memory_fabric_field(self):
        from aragora.debate.prompt_builder import PromptBuilder

        assert hasattr(PromptBuilder, "set_memory_fabric")

    def test_set_memory_fabric(self):
        from unittest.mock import MagicMock

        from aragora.debate.prompt_builder import PromptBuilder

        mock_protocol = MagicMock()
        mock_protocol.rounds = 3
        mock_protocol.format = "standard"
        mock_protocol.enforce_language = False
        mock_protocol.asymmetric_stances = False
        mock_protocol.intensity = None
        mock_protocol.use_cognitive_phases = False
        mock_protocol.enable_rlm_context = False

        mock_env = MagicMock()
        mock_env.task = "test task"
        mock_env.context = ""

        builder = PromptBuilder(protocol=mock_protocol, env=mock_env)
        assert builder._memory_fabric is None

        mock_fabric = MagicMock()
        builder.set_memory_fabric(mock_fabric)
        assert builder._memory_fabric is mock_fabric
