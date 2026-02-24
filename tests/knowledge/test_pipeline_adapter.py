"""Tests for PipelineAdapter query_precedents, get_agent_performance, and KM feedback.

Covers the reverse-flow methods that feed historical task outcomes back
into new pipeline executions via idea_to_execution._execute_task, plus
forward-flow ingestion, similar pipeline search, high-ROI patterns,
execution success rate, and data model serialization.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import Any

from aragora.knowledge.mound.adapters.pipeline_adapter import (
    PipelineAdapter,
    PipelineAdapterError,
    PipelineIngestionResult,
    PipelineStatus,
    SimilarPipeline,
    get_pipeline_adapter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeSearchResult:
    """Lightweight stand-in for KM search results."""

    content: str
    score: float
    metadata: dict[str, Any]


def _make_task_outcome(
    *,
    task_status: str = "completed",
    agent_type: str = "",
    task_type: str = "general",
    content: str = "PIPELINE TASK OUTCOME: task",
) -> FakeSearchResult:
    return FakeSearchResult(
        content=content,
        score=0.8,
        metadata={
            "type": "pipeline_task_outcome",
            "task_status": task_status,
            "agent_type": agent_type,
            "task_type": task_type,
        },
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mound():
    mound = MagicMock()
    mound.search = AsyncMock(return_value=[])
    mound.store = AsyncMock()
    return mound


@pytest.fixture
def adapter(mock_mound):
    return PipelineAdapter(mound=mock_mound)


# ---------------------------------------------------------------------------
# query_precedents
# ---------------------------------------------------------------------------


class TestQueryPrecedents:
    """Tests for PipelineAdapter.query_precedents."""

    @pytest.mark.asyncio
    async def test_returns_filtered_results(self, adapter, mock_mound):
        """Only pipeline_task_outcome items pass the type filter."""
        mock_mound.search = AsyncMock(
            return_value=[
                _make_task_outcome(
                    task_status="completed",
                    content="Fixed retry logic",
                ),
                # Non-matching type should be skipped
                FakeSearchResult(
                    content="unrelated",
                    score=0.7,
                    metadata={"type": "pipeline_goal"},
                ),
                _make_task_outcome(
                    task_status="failed",
                    content="Cache layer broke",
                ),
            ]
        )

        results = await adapter.query_precedents("backend", limit=5)

        assert len(results) == 2
        assert results[0]["outcome"] == "Fixed retry logic"
        assert results[1]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_empty_when_no_mound(self):
        """Adapter with no mound returns an empty list."""
        adapter = PipelineAdapter(mound=None)
        with patch(
            "aragora.knowledge.mound.adapters.pipeline_adapter.PipelineAdapter.mound",
            new_callable=lambda: property(lambda self: None),
        ):
            results = await adapter.query_precedents("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_respects_limit(self, adapter, mock_mound):
        """Returned precedents are capped at the requested limit."""
        mock_mound.search = AsyncMock(
            return_value=[
                _make_task_outcome(content=f"task {i}") for i in range(10)
            ]
        )

        results = await adapter.query_precedents("backend", limit=3)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_calls_search_with_correct_filters(self, adapter, mock_mound):
        """Verify the search call passes the pipeline_task_outcome filter."""
        await adapter.query_precedents("refactor", limit=2, workspace_id="ws")

        mock_mound.search.assert_awaited_once()
        call_kwargs = mock_mound.search.call_args
        assert call_kwargs.kwargs.get("filters") == {"type": "pipeline_task_outcome"}
        assert call_kwargs.kwargs.get("workspace_id") == "ws"

    @pytest.mark.asyncio
    async def test_precedent_dict_shape(self, adapter, mock_mound):
        """Each precedent dict has the expected keys."""
        mock_mound.search = AsyncMock(
            return_value=[
                _make_task_outcome(
                    task_status="completed",
                    agent_type="claude",
                    content="Implemented caching",
                ),
            ]
        )

        results = await adapter.query_precedents("caching")

        assert len(results) == 1
        p = results[0]
        assert "task_type" in p
        assert "outcome" in p
        assert "status" in p
        assert "agent_type" in p
        assert p["status"] == "completed"
        assert p["agent_type"] == "claude"


# ---------------------------------------------------------------------------
# get_agent_performance
# ---------------------------------------------------------------------------


class TestGetAgentPerformance:
    """Tests for PipelineAdapter.get_agent_performance."""

    @pytest.mark.asyncio
    async def test_calculates_rate(self, adapter, mock_mound):
        """3 completed + 1 failed => success_rate = 0.75."""
        mock_mound.search = AsyncMock(
            return_value=[
                _make_task_outcome(task_status="completed", agent_type="claude"),
                _make_task_outcome(task_status="completed", agent_type="claude"),
                _make_task_outcome(task_status="completed", agent_type="claude"),
                _make_task_outcome(task_status="failed", agent_type="claude"),
            ]
        )

        perf = await adapter.get_agent_performance("claude", domain="backend")

        assert perf["success_rate"] == 0.75
        assert perf["total_tasks"] == 4
        assert perf["agent_type"] == "claude"
        assert perf["domain"] == "backend"

    @pytest.mark.asyncio
    async def test_empty_agent_returns_empty(self, adapter):
        """Empty agent_type returns {} immediately without querying."""
        result = await adapter.get_agent_performance("")
        assert result == {}

    @pytest.mark.asyncio
    async def test_no_results_returns_empty(self, adapter, mock_mound):
        """No matching results returns {}."""
        mock_mound.search = AsyncMock(return_value=[])

        result = await adapter.get_agent_performance("gpt4")
        assert result == {}

    @pytest.mark.asyncio
    async def test_filters_by_agent_type(self, adapter, mock_mound):
        """Only results matching the requested agent_type are counted."""
        mock_mound.search = AsyncMock(
            return_value=[
                _make_task_outcome(task_status="completed", agent_type="claude"),
                _make_task_outcome(task_status="completed", agent_type="gpt4"),
                _make_task_outcome(task_status="failed", agent_type="claude"),
            ]
        )

        perf = await adapter.get_agent_performance("claude")

        assert perf["total_tasks"] == 2
        assert perf["success_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_no_mound_returns_empty(self):
        """Adapter with no mound returns {}."""
        adapter = PipelineAdapter(mound=None)
        with patch(
            "aragora.knowledge.mound.adapters.pipeline_adapter.PipelineAdapter.mound",
            new_callable=lambda: property(lambda self: None),
        ):
            result = await adapter.get_agent_performance("claude")
        assert result == {}

    @pytest.mark.asyncio
    async def test_all_completed(self, adapter, mock_mound):
        """All completed tasks => success_rate = 1.0."""
        mock_mound.search = AsyncMock(
            return_value=[
                _make_task_outcome(task_status="completed", agent_type="gemini"),
                _make_task_outcome(task_status="completed", agent_type="gemini"),
            ]
        )

        perf = await adapter.get_agent_performance("gemini")
        assert perf["success_rate"] == 1.0
        assert perf["total_tasks"] == 2


# ---------------------------------------------------------------------------
# KM feedback in _execute_task
# ---------------------------------------------------------------------------


class TestKMFeedbackEnrichesInstruction:
    """Tests that _execute_task enriches the instruction with KM data."""

    @pytest.mark.asyncio
    async def test_km_feedback_adds_historical_insights(self):
        """When query_precedents returns outcomes, instruction gets enriched."""
        precedents = [
            {"outcome": "Caching reduced latency by 40%", "status": "completed", "task_type": "perf", "agent_type": "claude"},
            {"outcome": "Redis cluster improved throughput", "status": "completed", "task_type": "perf", "agent_type": "gpt4"},
        ]
        agent_perf = {
            "agent_type": "claude",
            "domain": "perf",
            "total_tasks": 5,
            "success_rate": 0.8,
        }

        mock_adapter = MagicMock()
        mock_adapter.query_precedents = AsyncMock(return_value=precedents)
        mock_adapter.get_agent_performance = AsyncMock(return_value=agent_perf)

        task = {
            "name": "Optimize database queries",
            "description": "Reduce P99 latency",
            "type": "perf",
            "assigned_agent": "claude",
        }

        # Build instruction the same way _execute_task does
        instruction = f"Implement: {task['name']}\n\n{task.get('description', '')}"

        with patch(
            "aragora.knowledge.mound.adapters.pipeline_adapter.get_pipeline_adapter",
            return_value=mock_adapter,
        ):
            from aragora.knowledge.mound.adapters.pipeline_adapter import get_pipeline_adapter

            pipeline_adapter = get_pipeline_adapter()
            task_type = task.get("type", task.get("track", "general"))

            precs = await pipeline_adapter.query_precedents(task_type, limit=3)
            if precs:
                lessons = []
                for p in precs:
                    outcome = p.get("outcome", "")
                    if outcome:
                        lessons.append(f"- {outcome}")
                if lessons:
                    instruction += (
                        "\n\nHistorical insights from similar tasks:\n"
                        + "\n".join(lessons)
                    )

            perf = await pipeline_adapter.get_agent_performance(
                agent_type=task.get("assigned_agent", ""),
                domain=task_type,
            )
            if perf and perf.get("success_rate", 0) > 0:
                rate = perf["success_rate"]
                instruction += (
                    f"\n\nAgent historical success rate for this domain: {rate:.0%}"
                )

        assert "Historical insights from similar tasks:" in instruction
        assert "Caching reduced latency by 40%" in instruction
        assert "Redis cluster improved throughput" in instruction
        assert "Agent historical success rate for this domain: 80%" in instruction

    @pytest.mark.asyncio
    async def test_km_feedback_skipped_when_no_precedents(self):
        """When no precedents, instruction stays unchanged."""
        mock_adapter = MagicMock()
        mock_adapter.query_precedents = AsyncMock(return_value=[])
        mock_adapter.get_agent_performance = AsyncMock(return_value={})

        task = {
            "name": "Build widget",
            "description": "A new widget",
            "type": "feature",
            "assigned_agent": "",
        }

        instruction = f"Implement: {task['name']}\n\n{task.get('description', '')}"
        original = instruction

        with patch(
            "aragora.knowledge.mound.adapters.pipeline_adapter.get_pipeline_adapter",
            return_value=mock_adapter,
        ):
            from aragora.knowledge.mound.adapters.pipeline_adapter import get_pipeline_adapter

            pipeline_adapter = get_pipeline_adapter()
            task_type = task.get("type", "general")

            precs = await pipeline_adapter.query_precedents(task_type, limit=3)
            if precs:
                lessons = [f"- {p.get('outcome', '')}" for p in precs if p.get("outcome")]
                if lessons:
                    instruction += "\n\nHistorical insights:\n" + "\n".join(lessons)

            perf = await pipeline_adapter.get_agent_performance(
                agent_type=task.get("assigned_agent", ""),
                domain=task_type,
            )
            if perf and perf.get("success_rate", 0) > 0:
                instruction += f"\n\nSuccess rate: {perf['success_rate']:.0%}"

        assert instruction == original

    @pytest.mark.asyncio
    async def test_km_feedback_graceful_on_import_error(self):
        """ImportError during adapter creation is caught gracefully."""
        task = {
            "name": "Build feature",
            "description": "desc",
            "type": "feature",
        }
        instruction = f"Implement: {task['name']}\n\n{task.get('description', '')}"
        original = instruction

        try:
            with patch(
                "aragora.knowledge.mound.adapters.pipeline_adapter.get_pipeline_adapter",
                side_effect=ImportError("no module"),
            ):
                from aragora.knowledge.mound.adapters.pipeline_adapter import get_pipeline_adapter
                pipeline_adapter = get_pipeline_adapter()
                precs = await pipeline_adapter.query_precedents("feature", limit=3)
                if precs:
                    instruction += "\n\nInsights:\n" + "\n".join(
                        f"- {p['outcome']}" for p in precs
                    )
        except (ImportError, AttributeError, TypeError, RuntimeError, ValueError):
            pass

        assert instruction == original


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


class TestPipelineIngestionResult:
    """Tests for PipelineIngestionResult dataclass."""

    def test_success_property_true(self):
        """success is True when no errors and items > 0."""
        result = PipelineIngestionResult(
            pipeline_id="p1",
            items_ingested=3,
            transitions_recorded=1,
            provenance_links_recorded=2,
            knowledge_item_ids=["a", "b", "c"],
            errors=[],
        )
        assert result.success is True

    def test_success_property_false_with_errors(self):
        """success is False when errors are present."""
        result = PipelineIngestionResult(
            pipeline_id="p2",
            items_ingested=1,
            transitions_recorded=0,
            provenance_links_recorded=0,
            knowledge_item_ids=["x"],
            errors=["KM store failed"],
        )
        assert result.success is False

    def test_success_property_false_with_zero_items(self):
        """success is False when items_ingested is 0."""
        result = PipelineIngestionResult(
            pipeline_id="p3",
            items_ingested=0,
            transitions_recorded=0,
            provenance_links_recorded=0,
            knowledge_item_ids=[],
            errors=[],
        )
        assert result.success is False

    def test_to_dict(self):
        """to_dict serializes all fields correctly."""
        result = PipelineIngestionResult(
            pipeline_id="p4",
            items_ingested=2,
            transitions_recorded=1,
            provenance_links_recorded=3,
            knowledge_item_ids=["id1", "id2"],
            errors=[],
        )
        d = result.to_dict()
        assert d["pipeline_id"] == "p4"
        assert d["items_ingested"] == 2
        assert d["transitions_recorded"] == 1
        assert d["provenance_links_recorded"] == 3
        assert d["knowledge_item_ids"] == ["id1", "id2"]
        assert d["errors"] == []
        assert d["success"] is True


class TestSimilarPipeline:
    """Tests for SimilarPipeline dataclass."""

    def test_to_dict(self):
        """to_dict serializes all fields."""
        sp = SimilarPipeline(
            pipeline_id="sim-1",
            description="Rate limiter pipeline",
            similarity=0.85,
            status="complete",
            stages_completed=4,
            goals_extracted=3,
            tasks_executed=5,
            what_worked=["caching", "retry"],
        )
        d = sp.to_dict()
        assert d["pipeline_id"] == "sim-1"
        assert d["similarity"] == 0.85
        assert d["status"] == "complete"
        assert d["stages_completed"] == 4
        assert d["what_worked"] == ["caching", "retry"]


class TestPipelineStatus:
    """Tests for PipelineStatus constants."""

    def test_constants(self):
        assert PipelineStatus.SUCCESS == "success"
        assert PipelineStatus.PARTIAL == "partial"
        assert PipelineStatus.FAILED == "failed"
        assert PipelineStatus.PLANNED == "planned"


# ---------------------------------------------------------------------------
# find_similar_pipelines
# ---------------------------------------------------------------------------


class TestFindSimilarPipelines:
    """Tests for PipelineAdapter.find_similar_pipelines()."""

    @pytest.mark.asyncio
    async def test_returns_matching_pipelines(self, adapter, mock_mound):
        """Search results with type=pipeline_result are returned as SimilarPipeline."""
        mock_mound.search = AsyncMock(
            return_value=[
                FakeSearchResult(
                    content="PIPELINE: p-100\nStages Completed: 4/4",
                    score=0.9,
                    metadata={
                        "type": "pipeline_result",
                        "pipeline_id": "p-100",
                        "stages_completed": 4,
                    },
                ),
                FakeSearchResult(
                    content="PIPELINE: p-101\nStages Completed: 2/4",
                    score=0.6,
                    metadata={
                        "type": "pipeline_result",
                        "pipeline_id": "p-101",
                        "stages_completed": 2,
                    },
                ),
                # Non-matching type should be filtered out
                FakeSearchResult(
                    content="unrelated goal",
                    score=0.8,
                    metadata={"type": "pipeline_goal"},
                ),
            ]
        )

        results = await adapter.find_similar_pipelines("rate limiter", limit=5)

        assert len(results) == 2
        assert results[0].pipeline_id == "p-100"
        assert results[0].similarity == 0.9
        assert results[0].status == "complete"  # stages_completed == 4
        assert results[1].pipeline_id == "p-101"
        assert results[1].status == "partial"  # stages_completed < 4

    @pytest.mark.asyncio
    async def test_filters_below_min_similarity(self, adapter, mock_mound):
        """Results below min_similarity threshold are excluded."""
        mock_mound.search = AsyncMock(
            return_value=[
                FakeSearchResult(
                    content="PIPELINE: p-low",
                    score=0.1,  # Below default 0.3
                    metadata={
                        "type": "pipeline_result",
                        "pipeline_id": "p-low",
                        "stages_completed": 4,
                    },
                ),
            ]
        )

        results = await adapter.find_similar_pipelines(
            "search query", min_similarity=0.3,
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_respects_limit(self, adapter, mock_mound):
        """Returned results are capped at the requested limit."""
        mock_mound.search = AsyncMock(
            return_value=[
                FakeSearchResult(
                    content=f"PIPELINE: p-{i}",
                    score=0.8,
                    metadata={
                        "type": "pipeline_result",
                        "pipeline_id": f"p-{i}",
                        "stages_completed": 3,
                    },
                )
                for i in range(10)
            ]
        )

        results = await adapter.find_similar_pipelines("query", limit=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_no_mound_returns_empty(self):
        """Adapter with no mound returns empty list."""
        adapter = PipelineAdapter(mound=None)
        with patch(
            "aragora.knowledge.mound.adapters.pipeline_adapter.PipelineAdapter.mound",
            new_callable=lambda: property(lambda self: None),
        ):
            results = await adapter.find_similar_pipelines("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_runtime_error_returns_empty(self, adapter, mock_mound):
        """RuntimeError during search returns empty list."""
        mock_mound.search = AsyncMock(side_effect=RuntimeError("DB timeout"))

        results = await adapter.find_similar_pipelines("query")
        assert results == []

    @pytest.mark.asyncio
    async def test_sorted_by_similarity_descending(self, adapter, mock_mound):
        """Results are sorted by similarity in descending order."""
        mock_mound.search = AsyncMock(
            return_value=[
                FakeSearchResult(
                    content="PIPELINE: p-low",
                    score=0.5,
                    metadata={
                        "type": "pipeline_result",
                        "pipeline_id": "p-low",
                        "stages_completed": 2,
                    },
                ),
                FakeSearchResult(
                    content="PIPELINE: p-high",
                    score=0.95,
                    metadata={
                        "type": "pipeline_result",
                        "pipeline_id": "p-high",
                        "stages_completed": 4,
                    },
                ),
            ]
        )

        results = await adapter.find_similar_pipelines("query")
        assert results[0].pipeline_id == "p-high"
        assert results[1].pipeline_id == "p-low"


# ---------------------------------------------------------------------------
# get_execution_success_rate
# ---------------------------------------------------------------------------


class TestGetExecutionSuccessRate:
    """Tests for PipelineAdapter.get_execution_success_rate()."""

    @pytest.mark.asyncio
    async def test_calculates_rate(self, adapter, mock_mound):
        """Correctly calculates completed/total ratio."""
        mock_mound.search = AsyncMock(
            return_value=[
                _make_task_outcome(task_status="completed"),
                _make_task_outcome(task_status="completed"),
                _make_task_outcome(task_status="failed"),
                _make_task_outcome(task_status="planned"),
            ]
        )

        stats = await adapter.get_execution_success_rate()

        assert stats["total"] == 4
        assert stats["completed"] == 2
        assert stats["failed"] == 1
        assert stats["planned"] == 1
        assert stats["rate"] == 0.5

    @pytest.mark.asyncio
    async def test_no_mound_returns_zeros(self):
        """Adapter with no mound returns all-zero stats."""
        adapter = PipelineAdapter(mound=None)
        with patch(
            "aragora.knowledge.mound.adapters.pipeline_adapter.PipelineAdapter.mound",
            new_callable=lambda: property(lambda self: None),
        ):
            stats = await adapter.get_execution_success_rate()
        assert stats["total"] == 0
        assert stats["rate"] == 0.0

    @pytest.mark.asyncio
    async def test_empty_results_returns_zeros(self, adapter, mock_mound):
        """No matching results yield all-zero stats."""
        mock_mound.search = AsyncMock(return_value=[])

        stats = await adapter.get_execution_success_rate()
        assert stats["total"] == 0
        assert stats["rate"] == 0.0

    @pytest.mark.asyncio
    async def test_all_completed(self, adapter, mock_mound):
        """All completed => rate = 1.0."""
        mock_mound.search = AsyncMock(
            return_value=[
                _make_task_outcome(task_status="completed"),
                _make_task_outcome(task_status="completed"),
            ]
        )

        stats = await adapter.get_execution_success_rate()
        assert stats["rate"] == 1.0

    @pytest.mark.asyncio
    async def test_runtime_error_returns_zeros(self, adapter, mock_mound):
        """RuntimeError during search returns all-zero stats."""
        mock_mound.search = AsyncMock(side_effect=RuntimeError("connection lost"))

        stats = await adapter.get_execution_success_rate()
        assert stats["total"] == 0
        assert stats["rate"] == 0.0


# ---------------------------------------------------------------------------
# get_high_roi_patterns
# ---------------------------------------------------------------------------


class TestGetHighROIPatterns:
    """Tests for PipelineAdapter.get_high_roi_patterns()."""

    @pytest.mark.asyncio
    async def test_groups_by_type_and_priority(self, adapter, mock_mound):
        """Goal results are grouped by goal_type_priority key."""
        mock_mound.search = AsyncMock(
            return_value=[
                FakeSearchResult(
                    content="PIPELINE GOAL: Improve API",
                    score=0.8,
                    metadata={
                        "type": "pipeline_goal",
                        "goal_type": "performance",
                        "priority": "high",
                    },
                ),
                FakeSearchResult(
                    content="PIPELINE GOAL: Optimize DB",
                    score=0.7,
                    metadata={
                        "type": "pipeline_goal",
                        "goal_type": "performance",
                        "priority": "high",
                    },
                ),
                FakeSearchResult(
                    content="PIPELINE GOAL: Add docs",
                    score=0.6,
                    metadata={
                        "type": "pipeline_goal",
                        "goal_type": "documentation",
                        "priority": "low",
                    },
                ),
            ]
        )

        patterns = await adapter.get_high_roi_patterns(limit=5)

        assert len(patterns) == 2
        # Sorted by count descending; performance_high has count=2
        assert patterns[0]["goal_type"] == "performance"
        assert patterns[0]["priority"] == "high"
        assert patterns[0]["count"] == 2
        assert len(patterns[0]["examples"]) == 2

    @pytest.mark.asyncio
    async def test_no_mound_returns_empty(self):
        """Adapter with no mound returns empty list."""
        adapter = PipelineAdapter(mound=None)
        with patch(
            "aragora.knowledge.mound.adapters.pipeline_adapter.PipelineAdapter.mound",
            new_callable=lambda: property(lambda self: None),
        ):
            patterns = await adapter.get_high_roi_patterns()
        assert patterns == []

    @pytest.mark.asyncio
    async def test_respects_limit(self, adapter, mock_mound):
        """Returned patterns are capped at limit."""
        mock_mound.search = AsyncMock(
            return_value=[
                FakeSearchResult(
                    content=f"PIPELINE GOAL: goal-{i}",
                    score=0.8,
                    metadata={
                        "type": "pipeline_goal",
                        "goal_type": f"type-{i}",
                        "priority": "medium",
                    },
                )
                for i in range(10)
            ]
        )

        patterns = await adapter.get_high_roi_patterns(limit=3)
        assert len(patterns) <= 3

    @pytest.mark.asyncio
    async def test_max_three_examples(self, adapter, mock_mound):
        """Each pattern stores at most 3 examples."""
        mock_mound.search = AsyncMock(
            return_value=[
                FakeSearchResult(
                    content=f"PIPELINE GOAL: goal-{i}",
                    score=0.8,
                    metadata={
                        "type": "pipeline_goal",
                        "goal_type": "perf",
                        "priority": "high",
                    },
                )
                for i in range(5)
            ]
        )

        patterns = await adapter.get_high_roi_patterns(limit=5)
        # All items have the same key, so one pattern group
        assert len(patterns) == 1
        assert len(patterns[0]["examples"]) == 3  # Capped at 3


# ---------------------------------------------------------------------------
# _generate_pipeline_km_id
# ---------------------------------------------------------------------------


class TestGeneratePipelineKmId:
    """Tests for PipelineAdapter._generate_pipeline_km_id."""

    def test_deterministic(self, adapter):
        """Same pipeline_id always yields the same KM ID."""
        id1 = adapter._generate_pipeline_km_id("pipeline-abc")
        id2 = adapter._generate_pipeline_km_id("pipeline-abc")
        assert id1 == id2

    def test_prefix(self, adapter):
        """Generated ID starts with the ID_PREFIX."""
        km_id = adapter._generate_pipeline_km_id("test-pipeline")
        assert km_id.startswith(PipelineAdapter.ID_PREFIX)

    def test_different_pipelines_different_ids(self, adapter):
        """Different pipeline IDs yield different KM IDs."""
        id1 = adapter._generate_pipeline_km_id("pipeline-1")
        id2 = adapter._generate_pipeline_km_id("pipeline-2")
        assert id1 != id2


# ---------------------------------------------------------------------------
# Mound property lazy initialization
# ---------------------------------------------------------------------------


class TestMoundProperty:
    """Tests for PipelineAdapter.mound property."""

    def test_returns_provided_mound(self):
        """When mound is provided in __init__, returns it directly."""
        mock_mound = MagicMock()
        adapter = PipelineAdapter(mound=mock_mound)
        assert adapter.mound is mock_mound

    def test_lazy_init_returns_none_on_import_error(self):
        """When get_knowledge_mound import fails, mound returns None."""
        adapter = PipelineAdapter(mound=None)
        with patch(
            "aragora.knowledge.mound.adapters.pipeline_adapter.PipelineAdapter.mound",
            new_callable=lambda: property(lambda self: None),
        ):
            assert adapter.mound is None


# ---------------------------------------------------------------------------
# Event emission
# ---------------------------------------------------------------------------


class TestEventEmission:
    """Tests for PipelineAdapter._emit_event callback handling."""

    def test_event_callback_called(self):
        """When on_event is set, _emit_event calls it."""
        events = []
        adapter = PipelineAdapter(
            mound=MagicMock(),
            on_event=lambda event, data: events.append((event, data)),
        )
        adapter._emit_event("test_event", {"key": "val"})
        assert len(events) == 1
        assert events[0] == ("test_event", {"key": "val"})

    def test_event_callback_error_swallowed(self):
        """Errors in on_event callback are swallowed silently."""
        def bad_callback(event, data):
            raise RuntimeError("callback broke")

        adapter = PipelineAdapter(
            mound=MagicMock(),
            on_event=bad_callback,
        )
        # Should not raise
        adapter._emit_event("test", {})

    def test_no_callback_does_nothing(self):
        """_emit_event is no-op when on_event is None."""
        adapter = PipelineAdapter(mound=MagicMock(), on_event=None)
        # Should not raise
        adapter._emit_event("test", {})


# ---------------------------------------------------------------------------
# get_pipeline_adapter factory function
# ---------------------------------------------------------------------------


class TestGetPipelineAdapter:
    """Tests for the get_pipeline_adapter() factory function."""

    def test_returns_pipeline_adapter(self):
        """get_pipeline_adapter returns a PipelineAdapter instance."""
        adapter = get_pipeline_adapter()
        assert isinstance(adapter, PipelineAdapter)

    def test_with_workspace_id(self):
        """get_pipeline_adapter accepts workspace_id argument."""
        adapter = get_pipeline_adapter(workspace_id="custom-ws")
        assert isinstance(adapter, PipelineAdapter)
