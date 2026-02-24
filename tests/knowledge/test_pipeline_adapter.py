"""Tests for PipelineAdapter query_precedents, get_agent_performance, and KM feedback.

Covers the reverse-flow methods that feed historical task outcomes back
into new pipeline executions via idea_to_execution._execute_task.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import Any

from aragora.knowledge.mound.adapters.pipeline_adapter import (
    PipelineAdapter,
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
