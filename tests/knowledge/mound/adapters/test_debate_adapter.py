"""Tests for the DebateAdapter Knowledge Mound adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.knowledge.mound.adapters.debate_adapter import (
    DebateAdapter,
    DebateOutcome,
    DebateSearchResult,
)


@pytest.fixture
def adapter() -> DebateAdapter:
    return DebateAdapter()


@pytest.fixture
def sample_outcome() -> DebateOutcome:
    return DebateOutcome(
        debate_id="debate-001",
        task="Design a rate limiter for API requests",
        final_answer="Use token bucket algorithm with per-user quotas",
        confidence=0.85,
        consensus_reached=True,
        status="completed",
        rounds_used=3,
        duration_seconds=45.2,
        participants=["claude", "gpt4", "gemini"],
        winner="claude",
        consensus_strength="strong",
        total_cost_usd=0.15,
        total_tokens=5000,
        per_agent_cost={"claude": 0.05, "gpt4": 0.06, "gemini": 0.04},
        dissenting_views=["Redis-based approach might be simpler"],
        debate_cruxes=[{"crux": "Distributed vs single-node", "agents_split": 2}],
        convergence_similarity=0.92,
    )


class TestDebateAdapterInit:
    def test_init_defaults(self, adapter: DebateAdapter) -> None:
        assert adapter.adapter_name == "debate"
        assert adapter.source_type == "debate"
        assert adapter._pending_outcomes == []
        assert adapter._synced_outcomes == {}

    def test_init_with_callback(self) -> None:
        callback = MagicMock()
        adapter = DebateAdapter(event_callback=callback)
        assert adapter._event_callback == callback


class TestDebateOutcome:
    def test_from_debate_result(self) -> None:
        mock_result = MagicMock()
        mock_result.debate_id = "d-123"
        mock_result.task = "Test task"
        mock_result.final_answer = "Test answer"
        mock_result.confidence = 0.9
        mock_result.consensus_reached = True
        mock_result.status = "completed"
        mock_result.rounds_used = 2
        mock_result.duration_seconds = 30.0
        mock_result.participants = ["agent1"]
        mock_result.winner = "agent1"
        mock_result.consensus_strength = "strong"
        mock_result.total_cost_usd = 0.1
        mock_result.total_tokens = 1000
        mock_result.per_agent_cost = {}
        mock_result.dissenting_views = []
        mock_result.debate_cruxes = []
        mock_result.evidence_suggestions = []
        mock_result.convergence_similarity = 0.95
        mock_result.per_agent_similarity = {}
        mock_result.metadata = {}

        outcome = DebateOutcome.from_debate_result(mock_result)

        assert outcome.debate_id == "d-123"
        assert outcome.task == "Test task"
        assert outcome.confidence == 0.9
        assert outcome.consensus_reached is True


class TestStoreOutcome:
    def test_store_outcome_adds_to_pending(
        self, adapter: DebateAdapter, sample_outcome: DebateOutcome
    ) -> None:
        adapter.store_outcome(sample_outcome)

        assert len(adapter._pending_outcomes) == 1
        assert adapter._pending_outcomes[0].debate_id == "debate-001"
        assert adapter._pending_outcomes[0].metadata["km_sync_pending"] is True

    def test_store_outcome_emits_event(self, sample_outcome: DebateOutcome) -> None:
        callback = MagicMock()
        adapter = DebateAdapter(event_callback=callback)
        adapter.store_outcome(sample_outcome)

        callback.assert_called_once()
        event_type, event_data = callback.call_args[0]
        assert event_type == "km_adapter_forward_sync"
        assert event_data["debate_id"] == "debate-001"

    def test_store_from_debate_result(self, adapter: DebateAdapter) -> None:
        mock_result = MagicMock()
        mock_result.debate_id = "d-auto"
        mock_result.task = "Auto task"
        mock_result.final_answer = "Auto answer"
        mock_result.confidence = 0.7
        mock_result.consensus_reached = False
        mock_result.status = "completed"
        mock_result.rounds_used = 1
        mock_result.duration_seconds = 10.0
        mock_result.participants = []
        mock_result.winner = None
        mock_result.consensus_strength = ""
        mock_result.total_cost_usd = 0.0
        mock_result.total_tokens = 0
        mock_result.per_agent_cost = {}
        mock_result.dissenting_views = []
        mock_result.debate_cruxes = []
        mock_result.evidence_suggestions = []
        mock_result.convergence_similarity = 0.0
        mock_result.per_agent_similarity = {}
        mock_result.metadata = {}

        adapter.store_outcome(mock_result)
        assert len(adapter._pending_outcomes) == 1
        assert adapter._pending_outcomes[0].debate_id == "d-auto"


class TestGet:
    def test_get_returns_none_when_empty(self, adapter: DebateAdapter) -> None:
        assert adapter.get("nonexistent") is None

    def test_get_with_prefix(self, adapter: DebateAdapter, sample_outcome: DebateOutcome) -> None:
        adapter._synced_outcomes["debate-001"] = sample_outcome
        result = adapter.get("db_debate-001")
        assert result is not None
        assert result.debate_id == "debate-001"

    def test_get_without_prefix(
        self, adapter: DebateAdapter, sample_outcome: DebateOutcome
    ) -> None:
        adapter._synced_outcomes["debate-001"] = sample_outcome
        result = adapter.get("debate-001")
        assert result is not None


class TestSearchByTopic:
    @pytest.mark.asyncio
    async def test_search_finds_matching(
        self, adapter: DebateAdapter, sample_outcome: DebateOutcome
    ) -> None:
        adapter._synced_outcomes["debate-001"] = sample_outcome
        results = await adapter.search_by_topic("rate limiter")

        assert len(results) == 1
        assert results[0].debate_id == "debate-001"
        assert results[0].similarity > 0

    @pytest.mark.asyncio
    async def test_search_filters_by_confidence(
        self, adapter: DebateAdapter, sample_outcome: DebateOutcome
    ) -> None:
        adapter._synced_outcomes["debate-001"] = sample_outcome
        results = await adapter.search_by_topic("rate limiter", min_confidence=0.9)
        assert len(results) == 0  # confidence is 0.85

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, adapter: DebateAdapter) -> None:
        for i in range(5):
            outcome = DebateOutcome(
                debate_id=f"d-{i}",
                task=f"Design a rate limiter version {i}",
                final_answer="Answer",
                confidence=0.8,
                consensus_reached=True,
            )
            adapter._synced_outcomes[f"d-{i}"] = outcome

        results = await adapter.search_by_topic("rate limiter", limit=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_returns_empty_no_match(self, adapter: DebateAdapter) -> None:
        results = await adapter.search_by_topic("quantum computing")
        assert len(results) == 0


class TestToKnowledgeItem:
    def test_converts_outcome(self, adapter: DebateAdapter, sample_outcome: DebateOutcome) -> None:
        item = adapter.to_knowledge_item(sample_outcome)

        assert item.id == "db_debate-001"
        assert item.source_id == "debate-001"
        assert item.confidence == 0.85
        assert "rate limiter" in item.content.lower()
        assert item.metadata["consensus_reached"] is True
        assert item.metadata["rounds_used"] == 3
        assert item.metadata["participants"] == ["claude", "gpt4", "gemini"]


class TestSyncToKM:
    @pytest.mark.asyncio
    async def test_sync_calls_mound_store(
        self, adapter: DebateAdapter, sample_outcome: DebateOutcome
    ) -> None:
        mound = MagicMock()
        mound.store_item = AsyncMock()

        adapter.store_outcome(sample_outcome)
        result = await adapter.sync_to_km(mound)

        assert result.records_synced == 1
        assert result.records_failed == 0
        mound.store_item.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_skips_low_confidence(self, adapter: DebateAdapter) -> None:
        outcome = DebateOutcome(
            debate_id="d-low",
            task="Low confidence debate",
            final_answer="Maybe",
            confidence=0.1,
            consensus_reached=False,
        )
        mound = MagicMock()
        mound.store_item = AsyncMock()

        adapter.store_outcome(outcome)
        result = await adapter.sync_to_km(mound, min_confidence=0.5)

        assert result.records_synced == 0
        assert result.records_skipped == 1

    @pytest.mark.asyncio
    async def test_sync_handles_errors(
        self, adapter: DebateAdapter, sample_outcome: DebateOutcome
    ) -> None:
        mound = MagicMock()
        mound.store_item = AsyncMock(side_effect=Exception("Store failed"))

        adapter.store_outcome(sample_outcome)
        result = await adapter.sync_to_km(mound)

        assert result.records_failed == 1
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_sync_moves_to_synced(
        self, adapter: DebateAdapter, sample_outcome: DebateOutcome
    ) -> None:
        mound = MagicMock()
        mound.store_item = AsyncMock()

        adapter.store_outcome(sample_outcome)
        assert len(adapter._pending_outcomes) == 1

        await adapter.sync_to_km(mound)

        assert len(adapter._pending_outcomes) == 0
        assert "debate-001" in adapter._synced_outcomes


class TestGetStats:
    def test_stats_empty(self, adapter: DebateAdapter) -> None:
        stats = adapter.get_stats()
        assert stats["total_synced"] == 0
        assert stats["pending_sync"] == 0

    def test_stats_with_data(self, adapter: DebateAdapter, sample_outcome: DebateOutcome) -> None:
        adapter._synced_outcomes["debate-001"] = sample_outcome
        stats = adapter.get_stats()

        assert stats["total_synced"] == 1
        assert stats["avg_confidence"] == 0.85
        assert stats["consensus_rate"] == 1.0
        assert stats["total_cost_usd"] == 0.15


class TestMixinMethods:
    def test_get_record_by_id(self, adapter: DebateAdapter, sample_outcome: DebateOutcome) -> None:
        adapter._synced_outcomes["debate-001"] = sample_outcome
        record = adapter._get_record_by_id("debate-001")
        assert record is not None

    def test_record_to_dict(self, adapter: DebateAdapter, sample_outcome: DebateOutcome) -> None:
        d = adapter._record_to_dict(sample_outcome, similarity=0.9)
        assert d["id"] == "debate-001"
        assert d["similarity"] == 0.9
        assert d["confidence"] == 0.85

    def test_extract_source_id(self, adapter: DebateAdapter) -> None:
        assert adapter._extract_source_id({"source_id": "db_test"}) == "test"
        assert adapter._extract_source_id({"source_id": "plain"}) == "plain"
        assert adapter._extract_source_id({}) is None

    def test_get_fusion_sources(self, adapter: DebateAdapter) -> None:
        sources = adapter._get_fusion_sources()
        assert "consensus" in sources
        assert "evidence" in sources
