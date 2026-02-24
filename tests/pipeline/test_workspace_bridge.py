"""Tests for the WorkspacePipelineBridge.

Tests workspace-to-pipeline integration:
- Related beads found by keyword matching
- Empty results when no matches
- Completed goals are deduplicated
- Stale beads (>30 days) excluded
- Bridge handles missing workspace store gracefully
- Goal graph annotation via mark_completed_goals
- Pipeline uses workspace context when enabled
"""

import time

import pytest
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.pipeline.workspace_bridge import (
    DEFAULT_STALENESS_DAYS,
    BeadSummary,
    WorkspaceContext,
    WorkspacePipelineBridge,
    _extract_keywords,
    _keyword_overlap,
)


# =============================================================================
# Helpers
# =============================================================================


@dataclass
class FakeBead:
    """Minimal bead for testing."""

    bead_id: str
    title: str
    description: str = ""
    status: str = "done"
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    result: dict[str, Any] | None = None


@dataclass
class FakeGoalNode:
    """Minimal goal node for testing."""

    id: str
    title: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeGoalGraph:
    """Minimal goal graph for testing."""

    id: str = "test-graph"
    goals: list[FakeGoalNode] = field(default_factory=list)


def _make_bead(
    bead_id: str,
    title: str,
    status: str = "done",
    description: str = "",
    days_ago: float = 1.0,
) -> FakeBead:
    """Create a bead completed *days_ago* days in the past."""
    now = time.time()
    completed = now - (days_ago * 86400) if status == "done" else None
    created = now - (days_ago * 86400)
    return FakeBead(
        bead_id=bead_id,
        title=title,
        description=description,
        status=status,
        created_at=created,
        completed_at=completed,
    )


# =============================================================================
# Unit tests: keyword utilities
# =============================================================================


class TestKeywordExtraction:
    def test_basic_extraction(self):
        kw = _extract_keywords("Implement rate limiter for API")
        assert "implement" in kw
        assert "rate" in kw
        assert "limiter" in kw
        assert "api" in kw
        # Stop-words excluded
        assert "for" not in kw

    def test_empty_string(self):
        assert _extract_keywords("") == set()

    def test_stop_words_filtered(self):
        kw = _extract_keywords("the and or but is not")
        assert kw == set()

    def test_single_char_filtered(self):
        kw = _extract_keywords("a b c implement")
        assert "implement" in kw
        assert "b" not in kw


class TestKeywordOverlap:
    def test_full_overlap(self):
        assert _keyword_overlap({"rate", "limiter"}, {"rate", "limiter", "api"}) == 1.0

    def test_partial_overlap(self):
        result = _keyword_overlap({"rate", "limiter", "api"}, {"rate", "api"})
        assert abs(result - 2 / 3) < 0.01

    def test_no_overlap(self):
        assert _keyword_overlap({"rate", "limiter"}, {"database", "schema"}) == 0.0

    def test_empty_query(self):
        assert _keyword_overlap(set(), {"rate", "limiter"}) == 0.0


# =============================================================================
# WorkspacePipelineBridge tests
# =============================================================================


class TestWorkspacePipelineBridge:
    """Tests for WorkspacePipelineBridge.query_context."""

    @pytest.mark.asyncio
    async def test_related_beads_found_by_keyword(self):
        """Beads with matching keywords are returned in context."""
        mock_mgr = AsyncMock()
        mock_mgr.list_beads = AsyncMock(
            return_value=[
                _make_bead("bd-001", "Implement rate limiter", days_ago=2),
                _make_bead("bd-002", "Fix database schema", days_ago=3),
            ]
        )

        bridge = WorkspacePipelineBridge(bead_manager=mock_mgr)
        ctx = await bridge.query_context("rate limiter implementation")

        assert ctx.has_context
        assert len(ctx.related_beads) == 1
        assert ctx.related_beads[0].bead_id == "bd-001"

    @pytest.mark.asyncio
    async def test_no_matches_returns_empty_context(self):
        """No keyword matches yields an empty context."""
        mock_mgr = AsyncMock()
        mock_mgr.list_beads = AsyncMock(
            return_value=[
                _make_bead("bd-001", "Fix database schema", days_ago=1),
            ]
        )

        bridge = WorkspacePipelineBridge(bead_manager=mock_mgr)
        ctx = await bridge.query_context("implement rate limiter")

        assert not ctx.has_context
        assert ctx.related_beads == []
        assert ctx.completed_goals == set()

    @pytest.mark.asyncio
    async def test_completed_goals_tracked(self):
        """Completed beads are added to the completed_goals set."""
        mock_mgr = AsyncMock()
        mock_mgr.list_beads = AsyncMock(
            return_value=[
                _make_bead("bd-001", "Implement rate limiter", status="done", days_ago=2),
            ]
        )

        bridge = WorkspacePipelineBridge(bead_manager=mock_mgr)
        ctx = await bridge.query_context("rate limiter implementation")

        assert "implement rate limiter" in ctx.completed_goals
        assert len(ctx.execution_history) == 1
        assert "success" in ctx.execution_history[0]

    @pytest.mark.asyncio
    async def test_failed_beads_in_execution_history(self):
        """Failed beads appear in execution_history with 'failed' result."""
        mock_mgr = AsyncMock()
        mock_mgr.list_beads = AsyncMock(
            return_value=[
                _make_bead("bd-001", "Implement rate limiter", status="failed", days_ago=1),
            ]
        )

        bridge = WorkspacePipelineBridge(bead_manager=mock_mgr)
        ctx = await bridge.query_context("rate limiter implementation")

        assert ctx.has_context
        assert "implement rate limiter" not in ctx.completed_goals
        assert len(ctx.execution_history) == 1
        assert "failed" in ctx.execution_history[0]

    @pytest.mark.asyncio
    async def test_stale_beads_excluded(self):
        """Beads older than staleness threshold are excluded."""
        mock_mgr = AsyncMock()
        mock_mgr.list_beads = AsyncMock(
            return_value=[
                _make_bead("bd-old", "Implement rate limiter", days_ago=60),
                _make_bead("bd-new", "Build rate limiter module", days_ago=5),
            ]
        )

        bridge = WorkspacePipelineBridge(bead_manager=mock_mgr, staleness_days=30)
        ctx = await bridge.query_context("rate limiter implementation")

        assert len(ctx.related_beads) == 1
        assert ctx.related_beads[0].bead_id == "bd-new"

    @pytest.mark.asyncio
    async def test_custom_staleness_days(self):
        """Custom staleness_days parameter is respected."""
        mock_mgr = AsyncMock()
        mock_mgr.list_beads = AsyncMock(
            return_value=[
                _make_bead("bd-001", "Implement rate limiter", days_ago=10),
            ]
        )

        bridge = WorkspacePipelineBridge(bead_manager=mock_mgr)
        # Query with 5-day staleness window should exclude 10-day-old bead
        ctx = await bridge.query_context("rate limiter", staleness_days=5)
        assert not ctx.has_context

        # Query with 15-day window should include it
        ctx = await bridge.query_context("rate limiter", staleness_days=15)
        assert ctx.has_context

    @pytest.mark.asyncio
    async def test_duplicate_titles_deduplicated(self):
        """Beads with identical normalised titles are deduplicated."""
        mock_mgr = AsyncMock()
        mock_mgr.list_beads = AsyncMock(
            return_value=[
                _make_bead("bd-001", "Implement Rate Limiter", days_ago=2),
                _make_bead("bd-002", "implement rate limiter", days_ago=3),
            ]
        )

        bridge = WorkspacePipelineBridge(bead_manager=mock_mgr)
        ctx = await bridge.query_context("rate limiter implementation")

        assert len(ctx.related_beads) == 1

    @pytest.mark.asyncio
    async def test_missing_workspace_store_graceful(self):
        """Bridge returns empty context when workspace store fails to init."""
        with patch(
            "aragora.pipeline.workspace_bridge.WorkspacePipelineBridge._ensure_bead_manager",
            new_callable=AsyncMock,
            return_value=False,
        ):
            bridge = WorkspacePipelineBridge()
            ctx = await bridge.query_context("anything")

            assert not ctx.has_context
            assert ctx.related_beads == []

    @pytest.mark.asyncio
    async def test_list_beads_exception_handled(self):
        """Exception in list_beads returns empty context gracefully."""
        mock_mgr = AsyncMock()
        mock_mgr.list_beads = AsyncMock(side_effect=RuntimeError("store unavailable"))

        bridge = WorkspacePipelineBridge(bead_manager=mock_mgr)
        ctx = await bridge.query_context("rate limiter")

        assert not ctx.has_context

    @pytest.mark.asyncio
    async def test_suggest_skip_stages_when_goals_completed(self):
        """Suggests skipping ideation when completed goals exist."""
        mock_mgr = AsyncMock()
        mock_mgr.list_beads = AsyncMock(
            return_value=[
                _make_bead("bd-001", "Implement rate limiter", status="done", days_ago=2),
            ]
        )

        bridge = WorkspacePipelineBridge(bead_manager=mock_mgr)
        ctx = await bridge.query_context("rate limiter implementation")

        assert "ideation" in ctx.suggested_skip_stages

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self):
        """A query of all stop-words returns empty context."""
        mock_mgr = AsyncMock()
        mock_mgr.list_beads = AsyncMock(return_value=[])

        bridge = WorkspacePipelineBridge(bead_manager=mock_mgr)
        ctx = await bridge.query_context("the and or")

        assert not ctx.has_context


# =============================================================================
# mark_completed_goals tests
# =============================================================================


class TestMarkCompletedGoals:
    """Tests for WorkspacePipelineBridge.mark_completed_goals."""

    @pytest.mark.asyncio
    async def test_goals_annotated_with_workspace_status(self):
        """Matching goals get workspace_status metadata."""
        bridge = WorkspacePipelineBridge(bead_manager=AsyncMock())

        goal_graph = FakeGoalGraph(
            goals=[
                FakeGoalNode(id="g1", title="Implement rate limiter"),
                FakeGoalNode(id="g2", title="Design API schema"),
            ]
        )

        ctx = WorkspaceContext(
            related_beads=[
                BeadSummary(
                    bead_id="bd-001",
                    title="Implement rate limiter",
                    description="",
                    status="done",
                    days_ago=2.0,
                ),
            ],
            completed_goals={"implement rate limiter"},
        )

        marked = await bridge.mark_completed_goals(goal_graph, ctx)

        assert marked == 1
        assert goal_graph.goals[0].metadata["workspace_status"] == "already_done"
        assert goal_graph.goals[0].metadata["workspace_bead_id"] == "bd-001"
        assert "workspace_status" not in goal_graph.goals[1].metadata

    @pytest.mark.asyncio
    async def test_no_completed_goals_marks_nothing(self):
        """Empty completed_goals set results in zero marks."""
        bridge = WorkspacePipelineBridge(bead_manager=AsyncMock())

        goal_graph = FakeGoalGraph(
            goals=[FakeGoalNode(id="g1", title="Implement rate limiter")]
        )
        ctx = WorkspaceContext()

        marked = await bridge.mark_completed_goals(goal_graph, ctx)
        assert marked == 0


# =============================================================================
# WorkspaceContext tests
# =============================================================================


class TestWorkspaceContext:
    def test_has_context_true_with_beads(self):
        ctx = WorkspaceContext(
            related_beads=[
                BeadSummary(
                    bead_id="bd-001",
                    title="test",
                    description="",
                    status="done",
                )
            ]
        )
        assert ctx.has_context

    def test_has_context_false_when_empty(self):
        ctx = WorkspaceContext()
        assert not ctx.has_context

    def test_to_dict_serialization(self):
        ctx = WorkspaceContext(
            related_beads=[
                BeadSummary(
                    bead_id="bd-001",
                    title="test bead",
                    description="desc",
                    status="done",
                    days_ago=1.5,
                )
            ],
            completed_goals={"test bead"},
            suggested_skip_stages=["ideation"],
            execution_history=["Last attempt at 'test bead' was 1.5 days ago, result: success"],
        )
        d = ctx.to_dict()
        assert d["has_context"] is True
        assert len(d["related_beads"]) == 1
        assert d["completed_goals"] == ["test bead"]
        assert d["suggested_skip_stages"] == ["ideation"]


# =============================================================================
# Pipeline integration test
# =============================================================================


class TestPipelineIntegration:
    """Verify that the pipeline's enable_workspace_context flag is wired."""

    def test_pipeline_config_has_workspace_flag(self):
        """PipelineConfig includes enable_workspace_context."""
        from aragora.pipeline.idea_to_execution import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.enable_workspace_context is True

    def test_pipeline_config_workspace_flag_disabled(self):
        """enable_workspace_context can be set to False."""
        from aragora.pipeline.idea_to_execution import PipelineConfig

        cfg = PipelineConfig(enable_workspace_context=False)
        assert cfg.enable_workspace_context is False

    def test_pipeline_result_has_workspace_context_field(self):
        """PipelineResult includes workspace_context field."""
        from aragora.pipeline.idea_to_execution import PipelineResult

        pr = PipelineResult(pipeline_id="test")
        assert pr.workspace_context is None

        ctx = WorkspaceContext()
        pr.workspace_context = ctx
        assert pr.workspace_context is ctx
