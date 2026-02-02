"""Tests for CritiqueAdapter — bridges CritiqueStore to Knowledge Mound."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from aragora.knowledge.mound.adapters.critique_adapter import (
    CritiqueAdapter,
    CritiqueKMSyncResult,
    CritiqueSearchResult,
    KMPatternBoost,
    KMPatternValidation,
    KMReputationAdjustment,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_pattern(
    id: str = "p1",
    issue_text: str = "performance bottleneck",
    suggestion_text: str = "Add caching",
    issue_type: str = "performance",
    success_count: int = 5,
    failure_count: int = 1,
    avg_severity: float = 3.0,
    example_task: str = "Optimize API",
    created_at: str = "2026-01-01T00:00:00",
    updated_at: str = "2026-01-02T00:00:00",
):
    p = MagicMock()
    p.id = id
    p.issue_text = issue_text
    p.suggestion_text = suggestion_text
    p.issue_type = issue_type
    p.success_count = success_count
    p.failure_count = failure_count
    p.success_rate = (
        success_count / (success_count + failure_count)
        if (success_count + failure_count) > 0
        else 0
    )
    p.avg_severity = avg_severity
    p.example_task = example_task
    p.created_at = created_at
    p.updated_at = updated_at
    return p


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.retrieve_patterns = MagicMock(return_value=[])
    store.get_reputation = MagicMock(return_value=None)
    store.get_all_reputations = MagicMock(return_value=[])
    store.get_vote_weights_batch = MagicMock(return_value={})
    store.get_stats = MagicMock(return_value={"total_patterns": 42})
    store.get_archive_stats = MagicMock(return_value={"archived": 5})
    store.store_pattern = MagicMock()
    store.update_reputation = MagicMock()
    return store


@pytest.fixture
def adapter(mock_store):
    return CritiqueAdapter(store=mock_store)


# =============================================================================
# Initialization
# =============================================================================


class TestCritiqueAdapterInit:
    def test_init(self, mock_store):
        adapter = CritiqueAdapter(store=mock_store)
        assert adapter.store is mock_store
        assert adapter.adapter_name == "critique"

    def test_init_with_options(self, mock_store):
        cb = MagicMock()
        adapter = CritiqueAdapter(
            store=mock_store,
            enable_dual_write=True,
            event_callback=cb,
            enable_resilience=False,
        )
        assert adapter._enable_dual_write is True
        assert adapter._event_callback is cb


# =============================================================================
# search_patterns
# =============================================================================


class TestSearchPatterns:
    def test_search_by_known_type(self, adapter, mock_store):
        """Query matching a known issue type uses type filter."""
        patterns = [_make_pattern()]
        mock_store.retrieve_patterns.return_value = patterns

        result = adapter.search_patterns("performance", limit=5)

        mock_store.retrieve_patterns.assert_called_once_with(
            issue_type="performance", min_success=1, limit=5
        )
        assert result == patterns

    def test_search_by_explicit_type(self, adapter, mock_store):
        """Explicit issue_type param overrides query detection."""
        adapter.search_patterns("some query", issue_type="security", limit=10)

        mock_store.retrieve_patterns.assert_called_once_with(
            issue_type="security", min_success=1, limit=10
        )

    def test_search_with_type_in_query(self, adapter, mock_store):
        """Detects type keyword embedded in longer query."""
        adapter.search_patterns("fix security vulnerability", limit=5)

        mock_store.retrieve_patterns.assert_called_once_with(
            issue_type="security", min_success=1, limit=5
        )

    def test_search_keyword_filter(self, adapter, mock_store):
        """Unknown query type triggers keyword filtering."""
        p1 = _make_pattern(id="p1", issue_text="fix memory leak in API")
        p2 = _make_pattern(id="p2", issue_text="database connection pooling")
        mock_store.retrieve_patterns.return_value = [p1, p2]

        result = adapter.search_patterns("memory", limit=5)

        # Only p1 matches "memory"
        assert len(result) == 1
        assert result[0].id == "p1"

    def test_search_empty_results(self, adapter, mock_store):
        mock_store.retrieve_patterns.return_value = []
        result = adapter.search_patterns("anything")
        assert result == []


# =============================================================================
# get
# =============================================================================


class TestGet:
    def test_get_pattern_found(self, adapter, mock_store):
        p = _make_pattern(id="p1")
        mock_store.retrieve_patterns.return_value = [p]

        result = adapter.get("p1")
        assert result is p

    def test_get_pattern_not_found(self, adapter, mock_store):
        mock_store.retrieve_patterns.return_value = []
        assert adapter.get("nonexistent") is None

    def test_get_strips_prefix(self, adapter, mock_store):
        p = _make_pattern(id="p1")
        mock_store.retrieve_patterns.return_value = [p]

        result = adapter.get("cr_p1")
        assert result is p


# =============================================================================
# to_knowledge_item
# =============================================================================


class TestToKnowledgeItem:
    def test_high_confidence(self, adapter):
        p = _make_pattern(success_count=9, failure_count=1)  # 0.9 rate
        item = adapter.to_knowledge_item(p)

        assert item.id == "cr_p1"
        assert "performance bottleneck" in item.content
        assert "Suggestion: Add caching" in item.content
        assert item.metadata["issue_type"] == "performance"
        assert item.metadata["success_rate"] == pytest.approx(0.9)

    def test_medium_confidence(self, adapter):
        p = _make_pattern(success_count=7, failure_count=3)  # 0.7 rate
        item = adapter.to_knowledge_item(p)
        assert item.confidence.value == "medium" or "medium" in str(item.confidence).lower()

    def test_low_confidence(self, adapter):
        p = _make_pattern(success_count=4, failure_count=6)  # 0.4 rate
        item = adapter.to_knowledge_item(p)
        assert item.confidence.value == "low" or "low" in str(item.confidence).lower()

    def test_unverified_confidence(self, adapter):
        p = _make_pattern(success_count=1, failure_count=9)  # 0.1 rate
        item = adapter.to_knowledge_item(p)
        assert item.confidence.value == "unverified" or "unverified" in str(item.confidence).lower()

    def test_no_suggestion(self, adapter):
        p = _make_pattern(suggestion_text=None)
        item = adapter.to_knowledge_item(p)
        assert "Suggestion:" not in item.content


# =============================================================================
# Agent reputation methods
# =============================================================================


class TestAgentReputation:
    def test_get_agent_reputation(self, adapter, mock_store):
        rep = MagicMock()
        mock_store.get_reputation.return_value = rep
        assert adapter.get_agent_reputation("claude") is rep
        mock_store.get_reputation.assert_called_once_with("claude")

    def test_get_agent_reputation_not_found(self, adapter, mock_store):
        mock_store.get_reputation.return_value = None
        assert adapter.get_agent_reputation("unknown") is None

    def test_get_top_agents(self, adapter, mock_store):
        r1 = MagicMock(reputation_score=0.9)
        r2 = MagicMock(reputation_score=0.7)
        r3 = MagicMock(reputation_score=0.8)
        mock_store.get_all_reputations.return_value = [r1, r2, r3]

        result = adapter.get_top_agents(limit=3)
        assert result[0].reputation_score == 0.9
        assert result[1].reputation_score == 0.8
        assert result[2].reputation_score == 0.7

    def test_get_agent_vote_weights(self, adapter, mock_store):
        mock_store.get_vote_weights_batch.return_value = {"claude": 1.2, "gpt": 0.8}
        result = adapter.get_agent_vote_weights(["claude", "gpt"])
        assert result["claude"] == 1.2


# =============================================================================
# Stats
# =============================================================================


class TestStats:
    def test_get_stats(self, adapter, mock_store):
        assert adapter.get_stats() == {"total_patterns": 42}

    def test_get_archive_stats(self, adapter, mock_store):
        assert adapter.get_archive_stats() == {"archived": 5}


# =============================================================================
# Reverse flow: record_pattern_usage
# =============================================================================


class TestRecordPatternUsage:
    def test_records_usage(self, adapter):
        adapter.record_pattern_usage("p1", "debate-1", True, 0.9)
        adapter.record_pattern_usage("p1", "debate-2", False, 0.6)

        stats = adapter.get_reverse_flow_stats()
        assert stats["patterns_tracked"] == 1
        assert stats["total_usage_records"] == 2


# =============================================================================
# Reverse flow: validate_pattern_from_km
# =============================================================================


class TestValidatePatternFromKM:
    @pytest.mark.asyncio
    async def test_boost_recommendation(self, adapter, mock_store):
        """High success rate across 5+ outcomes → boost."""
        refs = [{"metadata": {"debate_id": f"d{i}", "outcome_success": True}} for i in range(6)]
        validation = await adapter.validate_pattern_from_km("p1", refs)

        assert validation.recommendation == "boost"
        assert validation.outcome_success_rate == 1.0
        assert validation.cross_debate_usage == 6
        assert validation.boost_amount > 0

    @pytest.mark.asyncio
    async def test_archive_recommendation(self, adapter, mock_store):
        """Low success rate across 5+ outcomes → archive."""
        refs = [{"metadata": {"debate_id": f"d{i}", "outcome_success": False}} for i in range(6)]
        validation = await adapter.validate_pattern_from_km("p1", refs)

        assert validation.recommendation == "archive"
        assert validation.boost_amount == 0

    @pytest.mark.asyncio
    async def test_keep_recommendation_insufficient_data(self, adapter, mock_store):
        """Fewer than 5 outcomes → keep."""
        refs = [
            {"metadata": {"debate_id": "d1", "outcome_success": True}},
        ]
        validation = await adapter.validate_pattern_from_km("p1", refs)

        assert validation.recommendation == "keep"

    @pytest.mark.asyncio
    async def test_includes_recorded_usage(self, adapter, mock_store):
        """Recorded usage contributes to validation metrics."""
        adapter.record_pattern_usage("p1", "local-1", True, 0.9)
        adapter.record_pattern_usage("p1", "local-2", True, 0.8)

        refs = [{"metadata": {"debate_id": f"d{i}", "outcome_success": True}} for i in range(4)]
        validation = await adapter.validate_pattern_from_km("p1", refs)

        # 4 from refs + 2 from usage = 6 total, all successful
        assert validation.outcome_success_rate == 1.0
        assert validation.cross_debate_usage == 6


# =============================================================================
# Reverse flow: apply_pattern_boost
# =============================================================================


class TestApplyPatternBoost:
    @pytest.mark.asyncio
    async def test_applies_boost(self, adapter, mock_store):
        p = _make_pattern(id="p1")
        mock_store.retrieve_patterns.return_value = [p]

        validation = KMPatternValidation(
            pattern_id="p1",
            recommendation="boost",
            boost_amount=2,
            km_confidence=0.9,
        )
        boost = await adapter.apply_pattern_boost(validation)

        assert boost.was_applied is True
        assert mock_store.store_pattern.call_count == 2

    @pytest.mark.asyncio
    async def test_skips_non_boost(self, adapter, mock_store):
        validation = KMPatternValidation(
            pattern_id="p1",
            recommendation="keep",
            boost_amount=0,
        )
        boost = await adapter.apply_pattern_boost(validation)
        assert boost.was_applied is False

    @pytest.mark.asyncio
    async def test_pattern_not_found(self, adapter, mock_store):
        mock_store.retrieve_patterns.return_value = []
        validation = KMPatternValidation(
            pattern_id="missing",
            recommendation="boost",
            boost_amount=1,
        )
        boost = await adapter.apply_pattern_boost(validation)
        assert boost.was_applied is False
        assert "error" in boost.metadata


# =============================================================================
# Reverse flow: compute_reputation_adjustment
# =============================================================================


class TestComputeReputationAdjustment:
    @pytest.mark.asyncio
    async def test_boost_recommendation(self, adapter):
        items = [{"metadata": {"agent_name": "claude", "outcome_success": True}} for _ in range(6)]
        adj = await adapter.compute_reputation_adjustment("claude", items)

        assert adj.recommendation == "boost"
        assert adj.adjustment > 0
        assert adj.pattern_contributions == 6

    @pytest.mark.asyncio
    async def test_penalize_recommendation(self, adapter):
        items = [{"metadata": {"agent_name": "badbot", "outcome_success": False}} for _ in range(6)]
        adj = await adapter.compute_reputation_adjustment("badbot", items)

        assert adj.recommendation == "penalize"
        assert adj.adjustment < 0

    @pytest.mark.asyncio
    async def test_keep_with_no_outcomes(self, adapter):
        items = [{"metadata": {"agent_name": "claude"}}]
        adj = await adapter.compute_reputation_adjustment("claude", items)

        assert adj.recommendation == "keep"
        assert adj.adjustment == 0.0

    @pytest.mark.asyncio
    async def test_agents_involved_matching(self, adapter):
        items = [
            {"metadata": {"agents_involved": ["claude", "gpt"], "outcome_success": True}}
            for _ in range(6)
        ]
        adj = await adapter.compute_reputation_adjustment("claude", items)
        assert adj.pattern_contributions == 6


# =============================================================================
# Reverse flow: apply_reputation_adjustment
# =============================================================================


class TestApplyReputationAdjustment:
    @pytest.mark.asyncio
    async def test_applies_positive(self, adapter, mock_store):
        rep = MagicMock()
        mock_store.get_reputation.return_value = rep

        adj = KMReputationAdjustment(
            agent_name="claude",
            adjustment=0.1,
            recommendation="boost",
        )
        result = await adapter.apply_reputation_adjustment(adj)

        assert result is True
        assert mock_store.update_reputation.call_count > 0

    @pytest.mark.asyncio
    async def test_skips_zero_adjustment(self, adapter, mock_store):
        adj = KMReputationAdjustment(agent_name="claude", adjustment=0.0)
        result = await adapter.apply_reputation_adjustment(adj)
        assert result is False

    @pytest.mark.asyncio
    async def test_agent_not_found(self, adapter, mock_store):
        mock_store.get_reputation.return_value = None
        adj = KMReputationAdjustment(agent_name="missing", adjustment=0.1)
        result = await adapter.apply_reputation_adjustment(adj)
        assert result is False


# =============================================================================
# Reverse flow: clear and stats
# =============================================================================


class TestReverseFlowState:
    def test_clear_reverse_flow_state(self, adapter):
        adapter.record_pattern_usage("p1", "d1", True)
        adapter.clear_reverse_flow_state()

        stats = adapter.get_reverse_flow_stats()
        assert stats["km_boosts_applied"] == 0
        assert stats["patterns_tracked"] == 0

    def test_get_reverse_flow_stats_initial(self, adapter):
        stats = adapter.get_reverse_flow_stats()
        assert stats["km_boosts_applied"] == 0
        assert stats["km_reputation_adjustments"] == 0
        assert stats["validations_stored"] == 0
        assert stats["patterns_tracked"] == 0
        assert stats["total_usage_records"] == 0


# =============================================================================
# Dataclass tests
# =============================================================================


class TestDataclasses:
    def test_km_pattern_boost_defaults(self):
        b = KMPatternBoost(pattern_id="p1")
        assert b.boost_amount == 0
        assert b.was_applied is False
        assert b.source_debates == []

    def test_km_reputation_adjustment_defaults(self):
        a = KMReputationAdjustment(agent_name="claude")
        assert a.adjustment == 0.0
        assert a.recommendation == "keep"

    def test_km_pattern_validation_defaults(self):
        v = KMPatternValidation(pattern_id="p1")
        assert v.km_confidence == 0.7
        assert v.recommendation == "keep"

    def test_critique_km_sync_result_defaults(self):
        r = CritiqueKMSyncResult()
        assert r.patterns_analyzed == 0
        assert r.errors == []

    def test_critique_search_result(self):
        p = MagicMock()
        r = CritiqueSearchResult(pattern=p, relevance_score=0.8, matched_category=True)
        assert r.pattern is p
        assert r.relevance_score == 0.8
