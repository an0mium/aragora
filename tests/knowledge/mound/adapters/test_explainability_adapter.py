"""Tests for ExplainabilityAdapter - bridges debate explanations to KM."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.knowledge.mound.adapters.explainability_adapter import (
    ExplainabilityAdapter,
    ExplainabilityEntry,
    ExplainabilitySearchResult,
    get_explainability_adapter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def adapter():
    """Create an ExplainabilityAdapter for testing."""
    return ExplainabilityAdapter(enable_resilience=False)


@pytest.fixture()
def sample_decision():
    """Create a mock Decision object for testing."""
    from aragora.explainability.decision import (
        BeliefChange,
        ConfidenceAttribution,
        Counterfactual,
        Decision,
        EvidenceLink,
        VotePivot,
    )

    return Decision(
        decision_id="dec-abc123",
        debate_id="debate-001",
        task="Design a rate limiter for the API gateway",
        domain="engineering",
        conclusion="Token bucket algorithm with 100 req/min default limit",
        consensus_reached=True,
        confidence=0.85,
        consensus_type="supermajority",
        rounds_used=3,
        agents_participated=["claude", "gpt4", "gemini"],
        evidence_chain=[
            EvidenceLink(
                id="ev1",
                content="Token bucket is industry standard",
                source="claude",
                relevance_score=0.9,
            ),
            EvidenceLink(
                id="ev2",
                content="Leaky bucket has less burst tolerance",
                source="gpt4",
                relevance_score=0.7,
            ),
        ],
        vote_pivots=[
            VotePivot(
                agent="gemini",
                choice="token_bucket",
                confidence=0.9,
                weight=1.2,
                reasoning_summary="Better burst handling",
                influence_score=0.45,
            ),
        ],
        belief_changes=[
            BeliefChange(
                agent="gpt4",
                round=2,
                topic="rate_limiting",
                prior_belief="leaky_bucket",
                posterior_belief="token_bucket",
                prior_confidence=0.6,
                posterior_confidence=0.85,
                trigger="evidence",
                trigger_source="claude",
            ),
        ],
        confidence_attribution=[
            ConfidenceAttribution(
                factor="consensus_strength",
                contribution=0.4,
                explanation="Strong agreement among agents",
            ),
            ConfidenceAttribution(
                factor="evidence_quality",
                contribution=0.35,
                explanation="High-quality technical evidence",
            ),
            ConfidenceAttribution(
                factor="agent_agreement",
                contribution=0.25,
                explanation="All agents converged",
            ),
        ],
        counterfactuals=[
            Counterfactual(
                condition="If gemini had not changed position",
                outcome_change="No consensus reached",
                likelihood=0.3,
                sensitivity=0.8,
                affected_agents=["gemini"],
            ),
            Counterfactual(
                condition="If evidence quality was lower",
                outcome_change="Lower confidence (0.6)",
                likelihood=0.5,
                sensitivity=0.6,
                affected_agents=["claude", "gpt4"],
            ),
        ],
        evidence_quality_score=0.82,
        agent_agreement_score=0.90,
        belief_stability_score=0.75,
    )


@pytest.fixture()
def sample_entry(sample_decision):
    """Create an ExplainabilityEntry from a Decision."""
    return ExplainabilityEntry.from_decision(
        sample_decision, task="Design a rate limiter for the API gateway"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExplainabilityEntry:
    """Tests for ExplainabilityEntry dataclass."""

    def test_from_decision_extracts_fields(self, sample_decision):
        """Test that from_decision correctly extracts all Decision fields."""
        entry = ExplainabilityEntry.from_decision(sample_decision)

        assert entry.decision_id == "dec-abc123"
        assert entry.debate_id == "debate-001"
        assert entry.task == "Design a rate limiter for the API gateway"
        assert entry.domain == "engineering"
        assert entry.confidence == 0.85
        assert entry.consensus_reached is True
        assert entry.consensus_type == "supermajority"
        assert entry.rounds_used == 3
        assert entry.agents_participated == ["claude", "gpt4", "gemini"]

    def test_from_decision_counts_components(self, sample_decision):
        """Test that from_decision counts evidence, pivots, beliefs, etc."""
        entry = ExplainabilityEntry.from_decision(sample_decision)

        assert entry.evidence_count == 2
        assert entry.vote_pivot_count == 1
        assert entry.belief_change_count == 1
        assert entry.factor_count == 3
        assert entry.counterfactual_count == 2

    def test_from_decision_extracts_top_factors(self, sample_decision):
        """Test that top factors are sorted by contribution."""
        entry = ExplainabilityEntry.from_decision(sample_decision)

        assert len(entry.top_factors) == 3
        # Sorted by contribution descending
        assert entry.top_factors[0]["factor"] == "consensus_strength"
        assert entry.top_factors[0]["contribution"] == 0.4
        assert entry.top_factors[1]["factor"] == "evidence_quality"

    def test_from_decision_extracts_top_counterfactuals(self, sample_decision):
        """Test that top counterfactuals are sorted by sensitivity."""
        entry = ExplainabilityEntry.from_decision(sample_decision)

        assert len(entry.top_counterfactuals) == 2
        # Sorted by sensitivity descending
        assert entry.top_counterfactuals[0]["sensitivity"] == 0.8
        assert entry.top_counterfactuals[1]["sensitivity"] == 0.6

    def test_from_decision_preserves_scores(self, sample_decision):
        """Test that quality scores are preserved."""
        entry = ExplainabilityEntry.from_decision(sample_decision)

        assert entry.evidence_quality_score == 0.82
        assert entry.agent_agreement_score == 0.90
        assert entry.belief_stability_score == 0.75

    def test_from_decision_stores_full_dict(self, sample_decision):
        """Test that full decision data is stored for deep retrieval."""
        entry = ExplainabilityEntry.from_decision(sample_decision)

        assert entry.decision_data["decision_id"] == "dec-abc123"
        assert len(entry.decision_data["evidence_chain"]) == 2
        assert len(entry.decision_data["counterfactuals"]) == 2

    def test_from_decision_with_task_override(self, sample_decision):
        """Test that task parameter overrides decision.task."""
        entry = ExplainabilityEntry.from_decision(
            sample_decision, task="Custom task override"
        )
        assert entry.task == "Custom task override"


class TestExplainabilityAdapter:
    """Tests for ExplainabilityAdapter class."""

    def test_store_explanation_with_decision(self, adapter, sample_decision):
        """Test storing a Decision object."""
        adapter.store_explanation(sample_decision, task="Rate limiter design")

        assert len(adapter._pending_entries) == 1
        entry = adapter._pending_entries[0]
        assert entry.decision_id == "dec-abc123"
        assert entry.metadata["km_sync_pending"] is True

    def test_store_explanation_with_entry(self, adapter, sample_entry):
        """Test storing an ExplainabilityEntry directly."""
        adapter.store_explanation(sample_entry)

        assert len(adapter._pending_entries) == 1
        assert adapter._pending_entries[0] is sample_entry

    def test_store_explanation_emits_event(self, adapter, sample_decision):
        """Test that storing emits an event."""
        events = []
        adapter.set_event_callback(lambda t, d: events.append((t, d)))

        adapter.store_explanation(sample_decision)

        assert len(events) == 1
        assert events[0][0] == "km_explainability_stored"
        assert events[0][1]["decision_id"] == "dec-abc123"
        assert events[0][1]["factor_count"] == 3

    def test_ingest_from_dict(self, adapter):
        """Test synchronous ingestion from a plain dict."""
        data = {
            "decision_id": "dec-xyz",
            "debate_id": "debate-002",
            "task": "Cache invalidation strategy",
            "domain": "engineering",
            "conclusion": "Use TTL-based expiry",
            "confidence": 0.75,
            "consensus_reached": True,
            "factor_count": 2,
            "counterfactual_count": 1,
            "top_factors": [{"factor": "evidence", "contribution": 0.6}],
            "top_counterfactuals": [
                {"condition": "No cache", "outcome_change": "Slower", "sensitivity": 0.7}
            ],
        }

        result = adapter.ingest(data)

        assert result is True
        assert len(adapter._pending_entries) == 1
        assert adapter._pending_entries[0].task == "Cache invalidation strategy"

    def test_ingest_handles_empty_dict(self, adapter):
        """Test that ingesting an empty dict succeeds with defaults."""
        result = adapter.ingest({})

        assert result is True
        assert len(adapter._pending_entries) == 1
        entry = adapter._pending_entries[0]
        assert entry.confidence == 0.0
        assert entry.task == ""

    def test_to_knowledge_item_content(self, adapter, sample_entry):
        """Test that to_knowledge_item builds proper content."""
        item = adapter.to_knowledge_item(sample_entry)

        assert item.id == "exp_dec-abc123"
        assert "Rate limiter" in item.content or "rate limiter" in item.content
        assert item.source_id == "debate-001"
        assert item.metadata["decision_id"] == "dec-abc123"
        assert item.metadata["factor_count"] == 3
        assert "explainability" in item.metadata["tags"]

    def test_to_knowledge_item_tags_counterfactuals(self, adapter, sample_entry):
        """Test that counterfactual entries are tagged."""
        item = adapter.to_knowledge_item(sample_entry)

        assert "has_counterfactuals" in item.metadata["tags"]
        assert "consensus_reached" in item.metadata["tags"]

    def test_to_knowledge_item_tags_high_evidence(self, adapter, sample_entry):
        """Test that high evidence quality entries are tagged."""
        # sample_entry has evidence_quality_score=0.82 >= 0.7
        item = adapter.to_knowledge_item(sample_entry)

        assert "high_evidence_quality" in item.metadata["tags"]

    def test_to_knowledge_item_no_counterfactual_tag_when_zero(self, adapter):
        """Test that entries without counterfactuals lack the tag."""
        entry = ExplainabilityEntry(
            decision_id="dec-no-cf",
            debate_id="debate-003",
            task="Simple task",
            domain="general",
            conclusion="Done",
            confidence=0.5,
            consensus_reached=False,
            counterfactual_count=0,
        )
        item = adapter.to_knowledge_item(entry)

        assert "has_counterfactuals" not in item.metadata["tags"]
        assert "consensus_reached" not in item.metadata["tags"]


class TestExplainabilityAdapterSync:
    """Tests for sync_to_km and search."""

    @pytest.mark.asyncio
    async def test_sync_to_km_stores_items(self, adapter, sample_entry):
        """Test that sync_to_km calls mound.store for pending entries."""
        adapter._pending_entries.append(sample_entry)

        # Use spec=[] so hasattr(mound, "store_item") returns False,
        # causing the adapter to fall through to mound.store.
        mound = MagicMock(spec=[])
        mound.store = AsyncMock()

        result = await adapter.sync_to_km(mound)

        assert result.records_synced == 1
        assert result.records_failed == 0
        assert len(adapter._pending_entries) == 0
        assert sample_entry.decision_id in adapter._synced_entries
        mound.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_to_km_skips_low_confidence(self, adapter):
        """Test that low-confidence entries are skipped."""
        entry = ExplainabilityEntry(
            decision_id="dec-low",
            debate_id="debate-low",
            task="Low confidence task",
            domain="general",
            conclusion="Maybe",
            confidence=0.1,
            consensus_reached=False,
        )
        adapter._pending_entries.append(entry)

        mound = AsyncMock()
        result = await adapter.sync_to_km(mound, min_confidence=0.3)

        assert result.records_skipped == 1
        assert result.records_synced == 0

    @pytest.mark.asyncio
    async def test_sync_to_km_handles_store_error(self, adapter, sample_entry):
        """Test that sync_to_km handles store errors gracefully."""
        adapter._pending_entries.append(sample_entry)

        mound = MagicMock(spec=[])
        mound.store = AsyncMock(side_effect=RuntimeError("Store failed"))

        result = await adapter.sync_to_km(mound)

        assert result.records_failed == 1
        assert len(result.errors) == 1
        assert "Store failed" in result.errors[0]

    @pytest.mark.asyncio
    async def test_search_by_topic_finds_matching(self, adapter, sample_entry):
        """Test that search_by_topic finds entries by topic text."""
        adapter._synced_entries[sample_entry.decision_id] = sample_entry

        results = await adapter.search_by_topic("rate limiter")

        assert len(results) == 1
        assert results[0].decision_id == "dec-abc123"
        assert results[0].similarity == 0.8

    @pytest.mark.asyncio
    async def test_search_by_topic_partial_match(self, adapter, sample_entry):
        """Test that search_by_topic finds partial word matches."""
        adapter._synced_entries[sample_entry.decision_id] = sample_entry

        # "gateway" appears as a word in the task but the full query
        # "gateway optimization" does not appear as a substring, triggering
        # the word-level fallback path (similarity=0.5).
        results = await adapter.search_by_topic("gateway optimization")

        assert len(results) >= 1
        assert results[0].similarity == 0.5  # Partial word match

    @pytest.mark.asyncio
    async def test_search_by_topic_respects_confidence(self, adapter):
        """Test that search respects min_confidence filter."""
        low_entry = ExplainabilityEntry(
            decision_id="dec-low2",
            debate_id="debate-low2",
            task="Design a cache",
            domain="engineering",
            conclusion="Use Redis",
            confidence=0.2,
            consensus_reached=False,
        )
        adapter._synced_entries["dec-low2"] = low_entry

        results = await adapter.search_by_topic("cache", min_confidence=0.5)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_by_topic_returns_empty_for_no_match(self, adapter, sample_entry):
        """Test that search returns empty for non-matching queries."""
        adapter._synced_entries[sample_entry.decision_id] = sample_entry

        results = await adapter.search_by_topic("quantum_computing")

        assert len(results) == 0


class TestExplainabilityAdapterStats:
    """Tests for get_stats."""

    def test_get_stats_empty(self, adapter):
        """Test stats with no data."""
        stats = adapter.get_stats()

        assert stats["total_synced"] == 0
        assert stats["pending_sync"] == 0
        assert stats["avg_confidence"] == 0.0
        assert stats["avg_factor_count"] == 0.0

    def test_get_stats_with_data(self, adapter, sample_entry):
        """Test stats with synced entries."""
        adapter._synced_entries["dec-abc123"] = sample_entry
        adapter._pending_entries.append(
            ExplainabilityEntry(
                decision_id="dec-pending",
                debate_id="debate-pending",
                task="Pending task",
                domain="general",
                conclusion="TBD",
                confidence=0.5,
                consensus_reached=False,
            )
        )

        stats = adapter.get_stats()

        assert stats["total_synced"] == 1
        assert stats["pending_sync"] == 1
        assert stats["avg_confidence"] == 0.85
        assert stats["consensus_rate"] == 1.0  # 1/1 synced has consensus


class TestGetExplainabilityAdapter:
    """Tests for module-level singleton."""

    def test_get_returns_adapter(self):
        """Test that get_explainability_adapter returns an adapter."""
        import aragora.knowledge.mound.adapters.explainability_adapter as mod

        # Reset singleton
        mod._explainability_adapter_singleton = None
        adapter = get_explainability_adapter()

        assert isinstance(adapter, ExplainabilityAdapter)

    def test_get_returns_same_instance(self):
        """Test that get_explainability_adapter returns singleton."""
        import aragora.knowledge.mound.adapters.explainability_adapter as mod

        mod._explainability_adapter_singleton = None
        a1 = get_explainability_adapter()
        a2 = get_explainability_adapter()

        assert a1 is a2


class TestFactoryRegistration:
    """Tests for adapter registration in factory."""

    def test_adapter_spec_registered(self):
        """Test that explainability adapter spec is in ADAPTER_SPECS."""
        from aragora.knowledge.mound.adapters.factory import ADAPTER_SPECS

        assert "explainability" in ADAPTER_SPECS
        spec = ADAPTER_SPECS["explainability"]
        assert spec.adapter_class is ExplainabilityAdapter
        assert spec.forward_method == "sync_to_km"
        assert spec.reverse_method == "search_by_topic"
        assert spec.priority == 66

    def test_factory_creates_adapter(self):
        """Test that AdapterFactory can create the adapter."""
        from aragora.knowledge.mound.adapters.factory import AdapterFactory

        factory = AdapterFactory()
        adapters = factory.create_from_subsystems()

        # explainability has no required_deps, so it should be auto-created
        assert "explainability" in adapters
        assert isinstance(adapters["explainability"].adapter, ExplainabilityAdapter)
