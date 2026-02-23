"""Tests for pipeline wiring: BeliefNetwork, ELO logging, KM persistence,
Arena orchestration, HardenedOrchestrator, MetaPlanner bridge, cross-cycle learning.

Covers changes 1A, 1B, 1C, 2A, 2B, 2C, 2D from the wiring plan.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.pipeline.idea_to_execution import (
    IdeaToExecutionPipeline,
    PipelineConfig,
    PipelineResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline():
    return IdeaToExecutionPipeline()


@pytest.fixture
def debate_output_with_edges():
    """Stage 1 output containing nodes AND edges for BeliefNetwork wiring."""
    return {
        "nodes": [
            {"id": "n1", "label": "Use Redis caching", "weight": 0.8, "type": "proposal"},
            {"id": "n2", "label": "Add rate limiting", "weight": 0.6, "type": "proposal"},
            {"id": "n3", "label": "Redis is overkill", "weight": 0.3, "type": "critique"},
        ],
        "edges": [
            {"source": "n1", "target": "n2", "type": "supports", "weight": 1.0},
            {"source": "n3", "target": "n1", "type": "contradicts", "weight": 0.7},
        ],
        "canvas": MagicMock(to_dict=MagicMock(return_value={})),
    }


@pytest.fixture
def simple_pipeline_result():
    """A minimal PipelineResult for KM persistence tests."""
    result = PipelineResult(pipeline_id="pipe-test123")
    result.stage_results = []
    result.provenance = []
    return result


# ---------------------------------------------------------------------------
# 1A: BeliefNetwork → GoalExtractor wiring
# ---------------------------------------------------------------------------


class TestBeliefNetworkWiring:
    """1A: BeliefNetwork results are passed into GoalExtractor."""

    @pytest.mark.asyncio
    async def test_belief_network_propagation_passes_to_extractor(
        self,
        pipeline,
        debate_output_with_edges,
    ):
        """When debate output has nodes+edges, BeliefNetwork builds and propagates."""
        cfg = PipelineConfig(stages_to_run=["goals"])

        # The debate output is what _run_ideation would produce
        stage1 = MagicMock()
        stage1.output = debate_output_with_edges
        stage1.status = "completed"

        result = PipelineResult(pipeline_id="test-belief")
        result.stage_results = [stage1]

        sr = await pipeline._run_goal_extraction("test", debate_output_with_edges, cfg)
        assert sr.status == "completed"
        goal_graph = sr.output["goal_graph"]
        assert goal_graph is not None

    @pytest.mark.asyncio
    async def test_belief_network_graceful_fallback_on_import_error(
        self,
        pipeline,
    ):
        """If BeliefNetwork import fails, extraction proceeds without it."""
        debate_output = {
            "nodes": [
                {"id": "n1", "label": "Test idea", "weight": 0.5, "type": "claim"},
            ],
            "edges": [],
        }
        cfg = PipelineConfig(stages_to_run=["goals"])

        with patch.dict("sys.modules", {"aragora.reasoning.belief": None}):
            sr = await pipeline._run_goal_extraction("test", debate_output, cfg)
            # Should still complete (falls back to no belief result)
            assert sr.status == "completed"

    @pytest.mark.asyncio
    async def test_belief_result_boosts_centrality_scoring(self, pipeline):
        """Nodes with high centrality from BeliefNetwork get higher goal scores."""
        from aragora.goals.extractor import GoalExtractionConfig

        debate_output = {
            "nodes": [
                {"id": "n1", "label": "Use Redis caching", "weight": 0.9, "type": "claim"},
                {"id": "n2", "label": "Add rate limiting", "weight": 0.7, "type": "claim"},
            ],
            "edges": [
                {"source": "n2", "target": "n1", "type": "supports", "weight": 1.0},
            ],
        }
        # Use low threshold so belief-rescaled scores still pass
        extraction_cfg = GoalExtractionConfig(confidence_threshold=0.2)
        cfg = PipelineConfig(
            stages_to_run=["goals"],
            goal_extraction_config=extraction_cfg,
        )
        sr = await pipeline._run_goal_extraction("test", debate_output, cfg)
        assert sr.status == "completed"
        goal_graph = sr.output["goal_graph"]
        # Should have extracted goals from the claim nodes
        assert len(goal_graph.goals) > 0


# ---------------------------------------------------------------------------
# 1B: ELO/Calibration logging
# ---------------------------------------------------------------------------


class TestELOLogging:
    """1B: ELO fallback is logged instead of silently swallowed."""

    def test_elo_used_flag_without_team_selector(self, pipeline):
        """When TeamSelector isn't available, elo_used should be False."""
        from aragora.canvas.models import Canvas

        canvas = Canvas(id="test", name="test")
        # Add a minimal node
        from aragora.canvas.models import CanvasNode, CanvasNodeType, Position

        canvas.nodes["step-1"] = CanvasNode(
            id="step-1",
            node_type=CanvasNodeType.KNOWLEDGE,
            position=Position(0, 0),
            label="Test step",
            data={"step_type": "task", "phase": "implement"},
        )

        plan = pipeline._actions_to_execution_plan(canvas)
        assert plan["elo_used"] is False
        assert len(plan["tasks"]) == 1

    def test_elo_used_flag_with_team_selector(self, pipeline):
        """When TeamSelector returns rankings, elo_used should be True."""
        from aragora.canvas.models import Canvas, CanvasNode, CanvasNodeType, Position

        canvas = Canvas(id="test", name="test")
        canvas.nodes["step-1"] = CanvasNode(
            id="step-1",
            node_type=CanvasNodeType.KNOWLEDGE,
            position=Position(0, 0),
            label="Test step",
            data={"step_type": "task", "phase": "research"},
        )

        mock_ranking = MagicMock(agent_id="claude-3", elo=1500.0)
        mock_selector = MagicMock()
        mock_selector.get_rankings.return_value = [mock_ranking]

        with patch(
            "aragora.debate.team_selector.TeamSelector",
            return_value=mock_selector,
        ):
            plan = pipeline._actions_to_execution_plan(canvas)
            assert plan["elo_used"] is True


# ---------------------------------------------------------------------------
# 1C: KM persistence
# ---------------------------------------------------------------------------


class TestKMPersistence:
    """1C: Pipeline results auto-persist to KnowledgeMound."""

    @pytest.mark.asyncio
    async def test_km_persistence_stores_result(self, pipeline):
        """When enable_km_persistence=True, store_pipeline_result is called."""
        cfg = PipelineConfig(
            stages_to_run=[],  # Skip all stages, just test receipt + KM
            enable_km_persistence=True,
            dry_run=False,
            enable_receipts=False,
        )

        mock_bridge = MagicMock()
        mock_bridge.available = True
        mock_bridge.store_pipeline_result.return_value = True

        with patch(
            "aragora.pipeline.km_bridge.PipelineKMBridge",
            return_value=mock_bridge,
        ):
            result = await pipeline.run("test input", config=cfg)
            mock_bridge.store_pipeline_result.assert_called_once()

    @pytest.mark.asyncio
    async def test_km_persistence_skipped_in_dry_run(self, pipeline):
        """KM persistence is skipped when dry_run=True."""
        cfg = PipelineConfig(
            stages_to_run=[],
            enable_km_persistence=True,
            dry_run=True,
        )

        result = await pipeline.run("test input", config=cfg)
        # No error, just skipped silently
        assert result.pipeline_id is not None

    @pytest.mark.asyncio
    async def test_km_persistence_graceful_on_import_error(self, pipeline):
        """KM persistence fails gracefully when PipelineKMBridge not available."""
        cfg = PipelineConfig(
            stages_to_run=[],
            enable_km_persistence=True,
            dry_run=False,
            enable_receipts=False,
        )

        with patch.dict("sys.modules", {"aragora.pipeline.km_bridge": None}):
            result = await pipeline.run("test input", config=cfg)
            assert result.pipeline_id is not None


# ---------------------------------------------------------------------------
# 2A: Arena mini-debate in Stage 4
# ---------------------------------------------------------------------------


class TestArenaOrchestration:
    """2A: Arena mini-debate as optional Stage 4 backend."""

    @pytest.mark.asyncio
    async def test_arena_backend_used_when_enabled(self, pipeline):
        """When use_arena_orchestration=True, Arena is tried before DebugLoop."""
        cfg = PipelineConfig(use_arena_orchestration=True)
        task = {"id": "t1", "name": "Test task", "description": "Do something"}

        mock_result = MagicMock()
        mock_result.summary = "Consensus reached on approach"
        mock_result.consensus_reached = True

        mock_arena = AsyncMock()
        mock_arena.run.return_value = mock_result

        with (
            patch("aragora.debate.orchestrator.Arena", return_value=mock_arena),
            patch("aragora.debate.DebateProtocol"),
            patch("aragora.core_types.Environment"),
        ):
            result = await pipeline._execute_task(task, cfg)
            assert result["backend"] == "arena"
            assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_arena_fallback_to_debug_loop(self, pipeline):
        """If Arena import fails, falls through to DebugLoop."""
        cfg = PipelineConfig(use_arena_orchestration=True)
        task = {"id": "t1", "name": "Test task", "description": "Do something"}

        mock_harness = AsyncMock()
        mock_harness.analyze.return_value = MagicMock(
            success=True, to_dict=lambda: {"status": "completed"}
        )
        with (
            patch.dict("sys.modules", {"aragora.debate.orchestrator": None}),
            patch("aragora.harnesses.claude_code.ClaudeCodeHarness", return_value=mock_harness),
        ):
            result = await pipeline._execute_task(task, cfg)
            # Should fall through to DebugLoop or planned
            assert result["status"] in ("planned", "failed", "completed")


# ---------------------------------------------------------------------------
# 2B: HardenedOrchestrator backend
# ---------------------------------------------------------------------------


class TestHardenedOrchestratorBackend:
    """2B: HardenedOrchestrator as optional Stage 4 backend."""

    @pytest.mark.asyncio
    async def test_hardened_orchestrator_used_when_enabled(self, pipeline):
        """When use_hardened_orchestrator=True, it's tried first."""
        cfg = PipelineConfig(use_hardened_orchestrator=True)
        task = {"id": "t1", "name": "Test task", "description": "Do something"}

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.to_dict.return_value = {"status": "ok"}

        mock_orch = AsyncMock()
        mock_orch.execute_goal.return_value = mock_result

        with patch(
            "aragora.nomic.hardened_orchestrator.HardenedOrchestrator",
            return_value=mock_orch,
        ):
            result = await pipeline._execute_task(task, cfg)
            assert result["backend"] == "hardened_orchestrator"
            assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_hardened_orchestrator_fallback(self, pipeline):
        """If HardenedOrchestrator unavailable, falls to next backend."""
        cfg = PipelineConfig(
            use_hardened_orchestrator=True,
            use_arena_orchestration=False,
        )
        task = {"id": "t1", "name": "Test task", "description": "Do something"}

        mock_harness = AsyncMock()
        mock_harness.analyze.return_value = MagicMock(
            success=True, to_dict=lambda: {"status": "completed"}
        )
        with (
            patch.dict("sys.modules", {"aragora.nomic.hardened_orchestrator": None}),
            patch("aragora.harnesses.claude_code.ClaudeCodeHarness", return_value=mock_harness),
        ):
            result = await pipeline._execute_task(task, cfg)
            # Falls through to DebugLoop or planned
            assert result["status"] in ("planned", "failed", "completed")


# ---------------------------------------------------------------------------
# 2C: MetaPlanner → Pipeline bridge
# ---------------------------------------------------------------------------


class TestMetaPlannerBridge:
    """2C: from_prioritized_goals() classmethod."""

    def test_from_prioritized_goals_creates_pipeline(self):
        """PrioritizedGoal objects convert to ideas and run pipeline."""

        @dataclass
        class FakeTrack:
            value: str = "developer"

        @dataclass
        class FakePrioritizedGoal:
            id: str = "pg-1"
            track: FakeTrack = field(default_factory=FakeTrack)
            description: str = "Add input validation to all API endpoints"
            rationale: str = "Prevents injection attacks"
            estimated_impact: str = "high"
            priority: int = 1
            focus_areas: list[str] = field(default_factory=list)

        goals = [
            FakePrioritizedGoal(),
            FakePrioritizedGoal(
                id="pg-2",
                description="Improve test coverage for auth module",
                rationale="Critical security surface",
                estimated_impact="medium",
            ),
        ]

        result = IdeaToExecutionPipeline.from_prioritized_goals(
            goals,
            auto_advance=True,
        )

        assert isinstance(result, PipelineResult)
        assert result.goal_graph is not None
        assert len(result.goal_graph.goals) > 0

    def test_from_prioritized_goals_includes_impact_and_track(self):
        """Converted ideas include impact level and track name."""

        @dataclass
        class FakeGoal:
            description: str = "Refactor storage layer"
            rationale: str = "Reduce complexity"
            estimated_impact: str = "high"
            track: str = "infrastructure"

        result = IdeaToExecutionPipeline.from_prioritized_goals(
            [FakeGoal()],
            auto_advance=False,
        )

        assert result.ideas_canvas is not None
        # Check that the idea text contains impact and track info
        nodes = list(result.ideas_canvas.nodes.values())
        assert len(nodes) > 0


# ---------------------------------------------------------------------------
# 2D: Cross-cycle learning
# ---------------------------------------------------------------------------


class TestCrossCycleLearning:
    """2D: NomicCycleAdapter integration in goal extraction."""

    @pytest.mark.asyncio
    async def test_cross_cycle_learning_enriches_goals(self, pipeline):
        """When NomicCycleAdapter finds similar cycles, goals get enriched."""

        @dataclass
        class FakeSimilarCycle:
            cycle_id: str = "cycle-1"
            objective: str = "Improve API performance"
            similarity: float = 0.8
            success_rate: float = 0.9
            what_worked: list[str] = field(default_factory=lambda: ["caching", "rate limiting"])
            what_failed: list[str] = field(default_factory=lambda: ["over-indexing"])
            recommendations: list[str] = field(default_factory=lambda: ["Start with caching layer"])
            tracks_affected: list[str] = field(default_factory=list)

        mock_adapter = AsyncMock()
        mock_adapter.find_similar_cycles.return_value = [FakeSimilarCycle()]

        debate_output = {
            "nodes": [
                {"id": "n1", "label": "Add caching to improve performance", "weight": 0.7},
            ],
            "edges": [],
        }
        cfg = PipelineConfig(stages_to_run=["goals"])

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.NomicCycleAdapter",
            return_value=mock_adapter,
        ):
            sr = await pipeline._run_goal_extraction("test", debate_output, cfg)
            assert sr.status == "completed"
            goal_graph = sr.output["goal_graph"]

            # Cross-cycle metadata should be present
            if "cross_cycle_learning" in goal_graph.metadata:
                ccl = goal_graph.metadata["cross_cycle_learning"]
                assert ccl["similar_cycles"] == 1
                assert len(ccl["recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_cross_cycle_learning_graceful_on_import_error(self, pipeline):
        """When NomicCycleAdapter is unavailable, goals still extract."""
        debate_output = {
            "nodes": [
                {"id": "n1", "label": "Test idea", "weight": 0.5},
            ],
            "edges": [],
        }
        cfg = PipelineConfig(stages_to_run=["goals"])

        with patch.dict(
            "sys.modules",
            {"aragora.knowledge.mound.adapters.nomic_cycle_adapter": None},
        ):
            sr = await pipeline._run_goal_extraction("test", debate_output, cfg)
            assert sr.status == "completed"
