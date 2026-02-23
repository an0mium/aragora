"""Tests for the KnowledgeMound Pipeline Adapter.

Tests the PipelineAdapter which bridges Idea-to-Execution Pipeline
results to the Knowledge Mound for cross-pipeline learning.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.knowledge.mound.adapters.pipeline_adapter import (
    PipelineAdapter,
    PipelineIngestionResult,
    PipelineStatus,
    SimilarPipeline,
    get_pipeline_adapter,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound with store and search."""
    mound = MagicMock()
    store_result = MagicMock()
    store_result.item_id = "km-item-001"
    store_result.deduplicated = False
    mound.store = AsyncMock(return_value=store_result)
    mound.search = AsyncMock(return_value=[])
    return mound


@pytest.fixture
def adapter(mock_mound):
    """Create adapter with mock mound."""
    return PipelineAdapter(mound=mock_mound)


@pytest.fixture
def sample_pipeline_data():
    """Sample PipelineResult.to_dict() output."""
    return {
        "pipeline_id": "pipe-abc12345",
        "stage_status": {
            "ideas": "complete",
            "goals": "complete",
            "actions": "complete",
            "orchestration": "complete",
        },
        "transitions": [
            {
                "id": "trans-ideas-goals-001",
                "from_stage": "ideas",
                "to_stage": "goals",
                "confidence": 0.72,
                "ai_rationale": "Extracted 3 goals from 4 ideas",
            },
            {
                "id": "trans-goals-actions-002",
                "from_stage": "goals",
                "to_stage": "actions",
                "confidence": 0.7,
                "ai_rationale": "Decomposed 3 goals into 8 steps",
            },
        ],
        "goals": {
            "id": "goals-demo",
            "goals": [
                {
                    "id": "goal-1",
                    "title": "Achieve: API reliability under load",
                    "type": "goal",
                    "priority": "critical",
                    "description": "Ensure API < 200ms P99 latency",
                    "confidence": 0.85,
                },
                {
                    "id": "goal-2",
                    "title": "Implement: Developer experience",
                    "type": "strategy",
                    "priority": "high",
                    "description": "Interactive API docs",
                    "confidence": 0.7,
                },
            ],
        },
        "orchestration_result": {
            "status": "executed",
            "tasks_completed": 3,
            "tasks_total": 4,
            "results": [
                {"task_id": "t1", "name": "Research rate limiter", "status": "completed"},
                {"task_id": "t2", "name": "Implement cache", "status": "completed"},
                {"task_id": "t3", "name": "Review design", "status": "awaiting_approval"},
                {"task_id": "t4", "name": "Deploy monitoring", "status": "completed"},
            ],
        },
        "provenance_count": 6,
        "integrity_hash": "abc123def456",
        "duration": 42.5,
    }


# =============================================================================
# PipelineIngestionResult tests
# =============================================================================


class TestIngestionResult:
    """Test PipelineIngestionResult dataclass."""

    def test_success_when_items_ingested(self):
        result = PipelineIngestionResult(
            pipeline_id="pipe-1",
            items_ingested=3,
            transitions_recorded=2,
            provenance_links_recorded=5,
            knowledge_item_ids=["a", "b", "c"],
            errors=[],
        )
        assert result.success is True

    def test_not_success_when_errors(self):
        result = PipelineIngestionResult(
            pipeline_id="pipe-1",
            items_ingested=1,
            transitions_recorded=0,
            provenance_links_recorded=0,
            knowledge_item_ids=["a"],
            errors=["boom"],
        )
        assert result.success is False

    def test_not_success_when_no_items(self):
        result = PipelineIngestionResult(
            pipeline_id="pipe-1",
            items_ingested=0,
            transitions_recorded=0,
            provenance_links_recorded=0,
            knowledge_item_ids=[],
            errors=[],
        )
        assert result.success is False

    def test_to_dict(self):
        result = PipelineIngestionResult(
            pipeline_id="pipe-1",
            items_ingested=2,
            transitions_recorded=1,
            provenance_links_recorded=4,
            knowledge_item_ids=["x"],
            errors=[],
        )
        d = result.to_dict()
        assert d["pipeline_id"] == "pipe-1"
        assert d["items_ingested"] == 2
        assert d["success"] is True


# =============================================================================
# SimilarPipeline tests
# =============================================================================


class TestSimilarPipeline:
    """Test SimilarPipeline dataclass."""

    def test_to_dict(self):
        sp = SimilarPipeline(
            pipeline_id="pipe-old",
            description="API refactor",
            similarity=0.8,
            status="complete",
            stages_completed=4,
            goals_extracted=3,
            tasks_executed=5,
            what_worked=["caching"],
        )
        d = sp.to_dict()
        assert d["pipeline_id"] == "pipe-old"
        assert d["similarity"] == 0.8
        assert d["what_worked"] == ["caching"]


# =============================================================================
# Pipeline ingestion tests
# =============================================================================


class TestPipelineIngestion:
    """Test ingesting pipeline results into KM."""

    @pytest.mark.asyncio
    async def test_ingest_stores_summary(self, adapter, mock_mound, sample_pipeline_data):
        result = await adapter.ingest_pipeline_result(sample_pipeline_data)
        assert result.items_ingested >= 1
        assert mock_mound.store.call_count >= 1
        # First call is the summary
        first_call = mock_mound.store.call_args_list[0]
        req = first_call[0][0]
        assert "PIPELINE:" in req.content

    @pytest.mark.asyncio
    async def test_ingest_records_pipeline_id(self, adapter, sample_pipeline_data):
        result = await adapter.ingest_pipeline_result(sample_pipeline_data)
        assert result.pipeline_id == "pipe-abc12345"

    @pytest.mark.asyncio
    async def test_ingest_records_transitions(self, adapter, mock_mound, sample_pipeline_data):
        result = await adapter.ingest_pipeline_result(sample_pipeline_data)
        assert result.transitions_recorded == 2

    @pytest.mark.asyncio
    async def test_ingest_records_provenance_count(self, adapter, sample_pipeline_data):
        result = await adapter.ingest_pipeline_result(sample_pipeline_data)
        assert result.provenance_links_recorded == 6

    @pytest.mark.asyncio
    async def test_ingest_stores_goals(self, adapter, mock_mound, sample_pipeline_data):
        result = await adapter.ingest_pipeline_result(sample_pipeline_data)
        # Summary (1) + transitions (2) + goals (2) + task outcomes (4) = 9
        assert mock_mound.store.call_count >= 5

    @pytest.mark.asyncio
    async def test_ingest_stores_task_outcomes(self, adapter, mock_mound, sample_pipeline_data):
        await adapter.ingest_pipeline_result(sample_pipeline_data)
        # Find calls with "PIPELINE TASK OUTCOME" content
        task_calls = [
            c for c in mock_mound.store.call_args_list if "PIPELINE TASK OUTCOME" in c[0][0].content
        ]
        assert len(task_calls) == 4

    @pytest.mark.asyncio
    async def test_ingest_returns_knowledge_item_ids(self, adapter, sample_pipeline_data):
        result = await adapter.ingest_pipeline_result(sample_pipeline_data)
        assert len(result.knowledge_item_ids) > 0
        assert all(isinstance(kid, str) for kid in result.knowledge_item_ids)

    @pytest.mark.asyncio
    async def test_ingest_success(self, adapter, sample_pipeline_data):
        result = await adapter.ingest_pipeline_result(sample_pipeline_data)
        assert result.success is True
        assert len(result.errors) == 0


# =============================================================================
# Mound unavailable tests
# =============================================================================


class TestMoundUnavailable:
    """Test graceful fallback when KM not available."""

    @pytest.mark.asyncio
    async def test_ingest_without_mound(self):
        adapter = PipelineAdapter(mound=None)
        # Patch get_knowledge_mound to raise ImportError
        with patch(
            "aragora.knowledge.mound.adapters.pipeline_adapter.PipelineAdapter.mound",
            new_callable=lambda: property(lambda self: None),
        ):
            result = await adapter.ingest_pipeline_result({"pipeline_id": "pipe-x"})
        assert result.success is False
        assert "not available" in result.errors[0]

    @pytest.mark.asyncio
    async def test_find_similar_without_mound(self):
        adapter = PipelineAdapter(mound=None)
        with patch(
            "aragora.knowledge.mound.adapters.pipeline_adapter.PipelineAdapter.mound",
            new_callable=lambda: property(lambda self: None),
        ):
            result = await adapter.find_similar_pipelines("test query")
        assert result == []

    @pytest.mark.asyncio
    async def test_high_roi_without_mound(self):
        adapter = PipelineAdapter(mound=None)
        with patch(
            "aragora.knowledge.mound.adapters.pipeline_adapter.PipelineAdapter.mound",
            new_callable=lambda: property(lambda self: None),
        ):
            result = await adapter.get_high_roi_patterns()
        assert result == []


# =============================================================================
# Similar pipeline search tests
# =============================================================================


class TestSimilarPipelineSearch:
    """Test finding similar pipelines."""

    @pytest.mark.asyncio
    async def test_find_similar_returns_list(self, adapter, mock_mound):
        mock_result = MagicMock()
        mock_result.metadata = {
            "type": "pipeline_result",
            "pipeline_id": "pipe-old-1",
            "stages_completed": 4,
        }
        mock_result.score = 0.85
        mock_result.content = "PIPELINE: pipe-old-1\nStages Completed: 4/4"
        mock_mound.search = AsyncMock(return_value=[mock_result])

        results = await adapter.find_similar_pipelines("rate limiter API")
        assert len(results) == 1
        assert results[0].pipeline_id == "pipe-old-1"
        assert results[0].similarity == 0.85

    @pytest.mark.asyncio
    async def test_find_similar_filters_low_score(self, adapter, mock_mound):
        mock_result = MagicMock()
        mock_result.metadata = {"type": "pipeline_result", "pipeline_id": "pipe-x"}
        mock_result.score = 0.1
        mock_result.content = "low relevance"
        mock_mound.search = AsyncMock(return_value=[mock_result])

        results = await adapter.find_similar_pipelines("test", min_similarity=0.3)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_find_similar_sorted_by_score(self, adapter, mock_mound):
        r1 = MagicMock()
        r1.metadata = {"type": "pipeline_result", "pipeline_id": "p1", "stages_completed": 2}
        r1.score = 0.6
        r1.content = "first"

        r2 = MagicMock()
        r2.metadata = {"type": "pipeline_result", "pipeline_id": "p2", "stages_completed": 4}
        r2.score = 0.9
        r2.content = "second"

        mock_mound.search = AsyncMock(return_value=[r1, r2])
        results = await adapter.find_similar_pipelines("query")
        assert results[0].pipeline_id == "p2"  # highest score first

    @pytest.mark.asyncio
    async def test_find_similar_respects_limit(self, adapter, mock_mound):
        mock_results = []
        for i in range(10):
            r = MagicMock()
            r.metadata = {"type": "pipeline_result", "pipeline_id": f"p{i}", "stages_completed": 3}
            r.score = 0.5 + i * 0.04
            r.content = f"pipeline {i}"
            mock_results.append(r)
        mock_mound.search = AsyncMock(return_value=mock_results)

        results = await adapter.find_similar_pipelines("query", limit=3)
        assert len(results) == 3


# =============================================================================
# High-ROI patterns tests
# =============================================================================


class TestHighROIPatterns:
    """Test extracting high-ROI goal patterns."""

    @pytest.mark.asyncio
    async def test_returns_patterns(self, adapter, mock_mound):
        r1 = MagicMock()
        r1.metadata = {"type": "pipeline_goal", "goal_type": "goal", "priority": "high"}
        r1.content = "PIPELINE GOAL: Build rate limiter"
        r2 = MagicMock()
        r2.metadata = {"type": "pipeline_goal", "goal_type": "goal", "priority": "high"}
        r2.content = "PIPELINE GOAL: Build cache"
        mock_mound.search = AsyncMock(return_value=[r1, r2])

        patterns = await adapter.get_high_roi_patterns()
        assert len(patterns) >= 1
        assert patterns[0]["goal_type"] == "goal"
        assert patterns[0]["count"] == 2

    @pytest.mark.asyncio
    async def test_empty_when_no_results(self, adapter, mock_mound):
        mock_mound.search = AsyncMock(return_value=[])
        patterns = await adapter.get_high_roi_patterns()
        assert patterns == []

    @pytest.mark.asyncio
    async def test_patterns_limited(self, adapter, mock_mound):
        results = []
        for i in range(20):
            r = MagicMock()
            r.metadata = {"type": "pipeline_goal", "goal_type": f"type{i}", "priority": "medium"}
            r.content = f"Goal {i}"
            results.append(r)
        mock_mound.search = AsyncMock(return_value=results)

        patterns = await adapter.get_high_roi_patterns(limit=3)
        assert len(patterns) <= 3


# =============================================================================
# Convenience function tests
# =============================================================================


class TestConvenienceFunction:
    """Test module-level convenience functions."""

    def test_get_pipeline_adapter_returns_adapter(self):
        adapter = get_pipeline_adapter()
        assert isinstance(adapter, PipelineAdapter)

    def test_pipeline_status_constants(self):
        assert PipelineStatus.SUCCESS == "success"
        assert PipelineStatus.PARTIAL == "partial"
        assert PipelineStatus.FAILED == "failed"
        assert PipelineStatus.PLANNED == "planned"


# =============================================================================
# Event emission tests
# =============================================================================


class TestEventEmission:
    """Test event callback emission."""

    @pytest.mark.asyncio
    async def test_emits_pipeline_ingested(self, mock_mound, sample_pipeline_data):
        events = []
        adapter = PipelineAdapter(mound=mock_mound, on_event=lambda e, d: events.append((e, d)))
        await adapter.ingest_pipeline_result(sample_pipeline_data)

        event_types = [e[0] for e in events]
        assert "pipeline_ingested" in event_types

    @pytest.mark.asyncio
    async def test_no_error_when_callback_raises(self, mock_mound, sample_pipeline_data):
        def bad_callback(e, d):
            raise RuntimeError("callback failed")

        adapter = PipelineAdapter(mound=mock_mound, on_event=bad_callback)
        # Should not raise
        result = await adapter.ingest_pipeline_result(sample_pipeline_data)
        assert result.success is True
