"""
Tests for OutcomeAdapter - Decision Outcome to Knowledge Mound bridge.

Tests cover:
- Outcome ingestion to Knowledge Mound
- KPI delta computation
- Lessons learned persistence
- Semantic search for similar outcomes
- Timeline queries
- Error handling and edge cases
- Singleton management
- Statistics tracking
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.knowledge.mound.adapters.outcome_adapter import (
    OutcomeAdapter,
    OutcomeAdapterError,
    OutcomeNotFoundError,
    OutcomeIngestionResult,
    get_outcome_adapter,
)
from aragora.knowledge.unified.types import ConfidenceLevel, KnowledgeSource


def _make_outcome(**overrides):
    """Create a sample outcome data dict."""
    base = {
        "outcome_id": "out_test_001",
        "decision_id": "dec_test_001",
        "debate_id": "dbt_test_001",
        "outcome_type": "success",
        "outcome_description": "Vendor delivered project on time and under budget",
        "impact_score": 0.85,
        "kpis_before": {"cost": 100000, "timeline_days": 90},
        "kpis_after": {"cost": 92000, "timeline_days": 85},
        "lessons_learned": "Early vendor screening improved outcomes",
        "tags": ["vendor", "procurement"],
    }
    base.update(overrides)
    return base


class TestOutcomeIngestionResult:
    """Tests for OutcomeIngestionResult dataclass."""

    def test_success_when_items_ingested(self):
        result = OutcomeIngestionResult(
            outcome_id="out_1",
            items_ingested=1,
            knowledge_item_ids=["item_1"],
            errors=[],
        )
        assert result.success is True

    def test_failure_when_errors(self):
        result = OutcomeIngestionResult(
            outcome_id="out_1",
            items_ingested=0,
            knowledge_item_ids=[],
            errors=["Something went wrong"],
        )
        assert result.success is False

    def test_failure_when_no_items(self):
        result = OutcomeIngestionResult(
            outcome_id="out_1",
            items_ingested=0,
            knowledge_item_ids=[],
            errors=[],
        )
        assert result.success is False

    def test_to_dict(self):
        result = OutcomeIngestionResult(
            outcome_id="out_1",
            items_ingested=2,
            knowledge_item_ids=["a", "b"],
            errors=[],
        )
        d = result.to_dict()
        assert d["outcome_id"] == "out_1"
        assert d["items_ingested"] == 2
        assert d["success"] is True
        assert len(d["knowledge_item_ids"]) == 2


class TestOutcomeAdapterInit:
    """Tests for OutcomeAdapter initialization."""

    def test_init_defaults(self):
        adapter = OutcomeAdapter()
        assert adapter._mound is None
        assert adapter._ingested_outcomes == {}
        assert adapter.adapter_name == "outcome"

    def test_init_with_mound(self):
        mound = MagicMock()
        adapter = OutcomeAdapter(mound=mound)
        assert adapter._mound is mound

    def test_set_mound(self):
        adapter = OutcomeAdapter()
        mound = MagicMock()
        adapter.set_mound(mound)
        assert adapter._mound is mound


class TestOutcomeIngest:
    """Tests for synchronous outcome ingestion."""

    def test_ingest_success(self):
        mound = MagicMock()
        mound.store_sync = MagicMock()
        adapter = OutcomeAdapter(mound=mound)

        result = adapter.ingest(_make_outcome())
        assert result is True
        mound.store_sync.assert_called_once()

    def test_ingest_stores_knowledge_item(self):
        mound = MagicMock()
        mound.store_sync = MagicMock()
        adapter = OutcomeAdapter(mound=mound)

        adapter.ingest(_make_outcome())

        item = mound.store_sync.call_args[0][0]
        assert "Outcome:success" in item.content
        assert "Vendor delivered" in item.content
        assert item.metadata["outcome_type"] == "success"
        assert item.metadata["impact_score"] == 0.85

    def test_ingest_computes_kpi_deltas(self):
        mound = MagicMock()
        mound.store_sync = MagicMock()
        adapter = OutcomeAdapter(mound=mound)

        adapter.ingest(_make_outcome())

        item = mound.store_sync.call_args[0][0]
        deltas = item.metadata["kpi_deltas"]
        assert deltas["cost"] == -8000  # 92000 - 100000
        assert deltas["timeline_days"] == -5  # 85 - 90

    def test_ingest_high_impact_maps_high_confidence(self):
        mound = MagicMock()
        mound.store_sync = MagicMock()
        adapter = OutcomeAdapter(mound=mound)

        adapter.ingest(_make_outcome(impact_score=0.9))

        item = mound.store_sync.call_args[0][0]
        assert item.confidence == ConfidenceLevel.HIGH

    def test_ingest_medium_impact_maps_medium_confidence(self):
        mound = MagicMock()
        mound.store_sync = MagicMock()
        adapter = OutcomeAdapter(mound=mound)

        adapter.ingest(_make_outcome(impact_score=0.5))

        item = mound.store_sync.call_args[0][0]
        assert item.confidence == ConfidenceLevel.MEDIUM

    def test_ingest_low_impact_maps_low_confidence(self):
        mound = MagicMock()
        mound.store_sync = MagicMock()
        adapter = OutcomeAdapter(mound=mound)

        adapter.ingest(_make_outcome(impact_score=0.2))

        item = mound.store_sync.call_args[0][0]
        assert item.confidence == ConfidenceLevel.LOW

    def test_ingest_includes_lessons_in_content(self):
        mound = MagicMock()
        mound.store_sync = MagicMock()
        adapter = OutcomeAdapter(mound=mound)

        adapter.ingest(_make_outcome(lessons_learned="Always check references"))

        item = mound.store_sync.call_args[0][0]
        assert "Always check references" in item.content

    def test_ingest_without_mound_succeeds(self):
        adapter = OutcomeAdapter(mound=None)
        result = adapter.ingest(_make_outcome())
        assert result is True

    def test_ingest_tracks_result(self):
        adapter = OutcomeAdapter()
        adapter.ingest(_make_outcome(outcome_id="out_abc"))

        result = adapter.get_ingestion_result("out_abc")
        assert result is not None
        assert result.items_ingested == 1
        assert result.success is True

    def test_ingest_emits_event(self):
        callback = MagicMock()
        adapter = OutcomeAdapter(event_callback=callback)

        adapter.ingest(_make_outcome())

        callback.assert_called_once()
        event_type, data = callback.call_args[0]
        assert event_type == "outcome_ingested"
        assert data["outcome_type"] == "success"

    def test_ingest_handles_missing_fields(self):
        adapter = OutcomeAdapter()
        result = adapter.ingest({"outcome_id": "minimal"})
        assert result is True

    def test_ingest_tags_include_outcome_type(self):
        mound = MagicMock()
        mound.store_sync = MagicMock()
        adapter = OutcomeAdapter(mound=mound)

        adapter.ingest(_make_outcome(outcome_type="failure"))

        item = mound.store_sync.call_args[0][0]
        assert "decision_outcome" in item.metadata["tags"]
        assert "type:failure" in item.metadata["tags"]

    def test_ingest_handles_non_numeric_kpis(self):
        """Non-numeric KPI values should be ignored in delta computation."""
        mound = MagicMock()
        mound.store_sync = MagicMock()
        adapter = OutcomeAdapter(mound=mound)

        adapter.ingest(_make_outcome(
            kpis_before={"quality": "high", "score": 80},
            kpis_after={"quality": "medium", "score": 85},
        ))

        item = mound.store_sync.call_args[0][0]
        deltas = item.metadata["kpi_deltas"]
        assert "quality" not in deltas  # string, not computed
        assert deltas["score"] == 5


class TestOutcomeSearch:
    """Tests for async search and timeline operations."""

    @pytest.mark.asyncio
    async def test_find_similar_outcomes_no_mound(self):
        adapter = OutcomeAdapter(mound=None)
        results = await adapter.find_similar_outcomes("vendor selection")
        assert results == []

    @pytest.mark.asyncio
    async def test_find_similar_outcomes_with_mound(self):
        mock_item = MagicMock()
        mock_results = MagicMock()
        mock_results.items = [mock_item]

        mound = MagicMock()
        mound.query = AsyncMock(return_value=mock_results)

        adapter = OutcomeAdapter(mound=mound)
        results = await adapter.find_similar_outcomes("vendor selection")

        assert len(results) == 1
        mound.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_find_similar_outcomes_with_type_filter(self):
        mock_results = MagicMock()
        mock_results.items = []
        mound = MagicMock()
        mound.query = AsyncMock(return_value=mock_results)

        adapter = OutcomeAdapter(mound=mound)
        await adapter.find_similar_outcomes("test", outcome_type="success")

        call_kwargs = mound.query.call_args[1]
        assert "type:success" in call_kwargs["tags"]

    @pytest.mark.asyncio
    async def test_get_outcome_timeline_no_mound(self):
        adapter = OutcomeAdapter(mound=None)
        results = await adapter.get_outcome_timeline("dec_123")
        assert results == []

    @pytest.mark.asyncio
    async def test_get_outcome_timeline_with_results(self):
        mock_item = MagicMock()
        mock_results = MagicMock()
        mock_results.items = [mock_item, mock_item]

        mound = MagicMock()
        mound.query = AsyncMock(return_value=mock_results)

        adapter = OutcomeAdapter(mound=mound)
        results = await adapter.get_outcome_timeline("dec_123")
        assert len(results) == 2


class TestOutcomeStats:
    """Tests for adapter statistics."""

    def test_stats_empty(self):
        adapter = OutcomeAdapter()
        stats = adapter.get_stats()
        assert stats["outcomes_processed"] == 0
        assert stats["total_items_ingested"] == 0
        assert stats["mound_connected"] is False

    def test_stats_after_ingestion(self):
        adapter = OutcomeAdapter()
        adapter.ingest(_make_outcome(outcome_id="a"))
        adapter.ingest(_make_outcome(outcome_id="b"))

        stats = adapter.get_stats()
        assert stats["outcomes_processed"] == 2
        assert stats["total_items_ingested"] == 2

    def test_stats_mound_connected(self):
        adapter = OutcomeAdapter(mound=MagicMock())
        stats = adapter.get_stats()
        assert stats["mound_connected"] is True


class TestOutcomeSingleton:
    """Tests for singleton management."""

    def test_get_outcome_adapter_returns_instance(self):
        # Reset singleton for isolation
        import aragora.knowledge.mound.adapters.outcome_adapter as mod

        mod._outcome_adapter_singleton = None
        adapter = get_outcome_adapter()
        assert isinstance(adapter, OutcomeAdapter)

    def test_get_outcome_adapter_returns_same_instance(self):
        import aragora.knowledge.mound.adapters.outcome_adapter as mod

        mod._outcome_adapter_singleton = None
        a1 = get_outcome_adapter()
        a2 = get_outcome_adapter()
        assert a1 is a2
        # Cleanup
        mod._outcome_adapter_singleton = None
