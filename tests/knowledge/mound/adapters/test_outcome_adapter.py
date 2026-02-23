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

import json
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
from aragora.storage.governance.models import OutcomeRecord


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

        adapter.ingest(
            _make_outcome(
                kpis_before={"quality": "high", "score": 80},
                kpis_after={"quality": "medium", "score": 85},
            )
        )

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


def _make_outcome_record(**overrides):
    """Create a sample OutcomeRecord dataclass for testing."""
    defaults = {
        "outcome_id": "out_rec_001",
        "decision_id": "dec_rec_001",
        "debate_id": "dbt_rec_001",
        "outcome_type": "success",
        "outcome_description": "Migration completed ahead of schedule",
        "impact_score": 0.75,
        "measured_at": datetime(2026, 2, 22, 12, 0, 0, tzinfo=timezone.utc),
        "kpis_before_json": json.dumps({"latency_ms": 200, "uptime": 0.99}),
        "kpis_after_json": json.dumps({"latency_ms": 120, "uptime": 0.999}),
        "lessons_learned": "Phased rollout reduced risk significantly",
        "tags_json": json.dumps(["infrastructure", "migration"]),
    }
    defaults.update(overrides)
    return OutcomeRecord(**defaults)


class TestOutcomeRecordBridge:
    """Tests for OutcomeRecord -> adapter bridge methods."""

    def test_record_to_dict_converts_fields(self):
        record = _make_outcome_record()
        d = OutcomeAdapter.record_to_dict(record)

        assert d["outcome_id"] == "out_rec_001"
        assert d["decision_id"] == "dec_rec_001"
        assert d["debate_id"] == "dbt_rec_001"
        assert d["outcome_type"] == "success"
        assert d["impact_score"] == 0.75
        assert d["kpis_before"]["latency_ms"] == 200
        assert d["kpis_after"]["uptime"] == 0.999
        assert d["lessons_learned"] == "Phased rollout reduced risk significantly"
        assert "infrastructure" in d["tags"]

    def test_record_to_dict_handles_empty_kpis(self):
        record = _make_outcome_record(
            kpis_before_json="{}",
            kpis_after_json="{}",
        )
        d = OutcomeAdapter.record_to_dict(record)
        assert d["kpis_before"] == {}
        assert d["kpis_after"] == {}

    def test_record_to_dict_handles_empty_tags(self):
        record = _make_outcome_record(tags_json="[]")
        d = OutcomeAdapter.record_to_dict(record)
        assert d["tags"] == []

    def test_ingest_record_delegates_to_ingest(self):
        mound = MagicMock()
        mound.store_sync = MagicMock()
        adapter = OutcomeAdapter(mound=mound)
        record = _make_outcome_record()

        result = adapter.ingest_record(record)

        assert result is True
        mound.store_sync.assert_called_once()
        item = mound.store_sync.call_args[0][0]
        assert "Outcome:success" in item.content
        assert item.metadata["decision_id"] == "dec_rec_001"

    def test_ingest_record_computes_kpi_deltas(self):
        mound = MagicMock()
        mound.store_sync = MagicMock()
        adapter = OutcomeAdapter(mound=mound)
        record = _make_outcome_record()

        adapter.ingest_record(record)

        item = mound.store_sync.call_args[0][0]
        deltas = item.metadata["kpi_deltas"]
        assert deltas["latency_ms"] == -80  # 120 - 200
        assert abs(deltas["uptime"] - 0.009) < 0.001  # 0.999 - 0.99

    def test_ingest_record_without_mound(self):
        adapter = OutcomeAdapter(mound=None)
        record = _make_outcome_record()
        result = adapter.ingest_record(record)
        assert result is True

    def test_ingest_record_tracks_ingestion_result(self):
        adapter = OutcomeAdapter()
        record = _make_outcome_record(outcome_id="out_tracked")
        adapter.ingest_record(record)

        ingestion = adapter.get_ingestion_result("out_tracked")
        assert ingestion is not None
        assert ingestion.success is True


class TestOutcomeEventHandling:
    """Tests for event emission edge cases."""

    def test_event_callback_error_does_not_break_ingest(self):
        """A failing event callback should not prevent ingestion."""

        def bad_callback(event_type, data):
            raise RuntimeError("callback crashed")

        adapter = OutcomeAdapter(event_callback=bad_callback)
        result = adapter.ingest(_make_outcome())
        assert result is True

    def test_event_callback_type_error_is_caught(self):
        """TypeError in callback should be handled gracefully."""

        def bad_callback(event_type, data):
            raise TypeError("wrong type")

        adapter = OutcomeAdapter(event_callback=bad_callback)
        result = adapter.ingest(_make_outcome())
        assert result is True

    def test_no_event_emitted_without_callback(self):
        """No crash when event_callback is None."""
        adapter = OutcomeAdapter(event_callback=None)
        result = adapter.ingest(_make_outcome())
        assert result is True


class TestOutcomeEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_ingest_empty_description(self):
        adapter = OutcomeAdapter()
        result = adapter.ingest(_make_outcome(outcome_description=""))
        assert result is True

    def test_ingest_zero_impact_score(self):
        mound = MagicMock()
        mound.store_sync = MagicMock()
        adapter = OutcomeAdapter(mound=mound)

        adapter.ingest(_make_outcome(impact_score=0.0))

        item = mound.store_sync.call_args[0][0]
        assert item.confidence == ConfidenceLevel.LOW

    def test_ingest_boundary_impact_0_7(self):
        """Impact score exactly at 0.7 should map to HIGH confidence."""
        mound = MagicMock()
        mound.store_sync = MagicMock()
        adapter = OutcomeAdapter(mound=mound)

        adapter.ingest(_make_outcome(impact_score=0.7))

        item = mound.store_sync.call_args[0][0]
        assert item.confidence == ConfidenceLevel.HIGH

    def test_ingest_boundary_impact_0_4(self):
        """Impact score exactly at 0.4 should map to MEDIUM confidence."""
        mound = MagicMock()
        mound.store_sync = MagicMock()
        adapter = OutcomeAdapter(mound=mound)

        adapter.ingest(_make_outcome(impact_score=0.4))

        item = mound.store_sync.call_args[0][0]
        assert item.confidence == ConfidenceLevel.MEDIUM

    def test_ingest_kpis_only_in_before(self):
        """KPI key only in before dict should not produce a delta."""
        mound = MagicMock()
        mound.store_sync = MagicMock()
        adapter = OutcomeAdapter(mound=mound)

        adapter.ingest(
            _make_outcome(
                kpis_before={"old_metric": 100},
                kpis_after={},
            )
        )

        item = mound.store_sync.call_args[0][0]
        assert "old_metric" not in item.metadata["kpi_deltas"]

    def test_ingest_kpis_only_in_after(self):
        """KPI key only in after dict should not produce a delta."""
        mound = MagicMock()
        mound.store_sync = MagicMock()
        adapter = OutcomeAdapter(mound=mound)

        adapter.ingest(
            _make_outcome(
                kpis_before={},
                kpis_after={"new_metric": 50},
            )
        )

        item = mound.store_sync.call_args[0][0]
        assert "new_metric" not in item.metadata["kpi_deltas"]

    def test_ingest_with_mound_async_only(self):
        """Mound with only async 'store' method (no store_sync) should not crash."""
        mound = MagicMock(spec=["store"])  # has store but not store_sync
        adapter = OutcomeAdapter(mound=mound)

        result = adapter.ingest(_make_outcome())
        assert result is True

    @pytest.mark.asyncio
    async def test_find_similar_outcomes_handles_query_error(self):
        """Mound query raising an error should return empty list."""
        mound = MagicMock()
        mound.query = AsyncMock(side_effect=RuntimeError("search broke"))

        adapter = OutcomeAdapter(mound=mound)
        results = await adapter.find_similar_outcomes("test")
        assert results == []

    @pytest.mark.asyncio
    async def test_get_outcome_timeline_handles_query_error(self):
        """Mound query raising an error should return empty list."""
        mound = MagicMock()
        mound.query = AsyncMock(side_effect=ValueError("bad query"))

        adapter = OutcomeAdapter(mound=mound)
        results = await adapter.get_outcome_timeline("dec_123")
        assert results == []

    def test_get_ingestion_result_not_found(self):
        adapter = OutcomeAdapter()
        result = adapter.get_ingestion_result("nonexistent")
        assert result is None

    def test_multiple_ingestions_different_outcomes(self):
        adapter = OutcomeAdapter()
        adapter.ingest(_make_outcome(outcome_id="a"))
        adapter.ingest(_make_outcome(outcome_id="b"))
        adapter.ingest(_make_outcome(outcome_id="c"))

        stats = adapter.get_stats()
        assert stats["outcomes_processed"] == 3
        assert stats["total_items_ingested"] == 3
        assert stats["total_errors"] == 0
