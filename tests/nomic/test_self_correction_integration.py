"""Integration tests for SelfCorrectionEngine wiring into the Nomic Loop.

Tests that:
1. SelfCorrectionEngine exports from aragora.nomic work
2. AutonomousOrchestrator._apply_self_correction stores data on result.after_metrics
3. _apply_self_correction is a no-op when no completed/failed assignments exist
4. _apply_self_correction is a no-op when self_correction is None
5. FeedbackLoop.apply_strategy_recommendations stores recommendations
6. FeedbackLoop.get_recommendation_for_track returns highest confidence rec
7. MetaPlanner._apply_self_correction_adjustments re-ranks goals
8. MetaPlanner._apply_self_correction_adjustments is a no-op with no past outcomes
9. Full pipeline: execute_goal triggers self-correction analysis
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.autonomous_orchestrator import (
    AgentAssignment,
    AutonomousOrchestrator,
    FeedbackLoop,
    OrchestrationResult,
    Track,
)
from aragora.nomic.meta_planner import MetaPlanner, MetaPlannerConfig, PrioritizedGoal
from aragora.nomic.self_correction import (
    CorrectionReport,
    SelfCorrectionConfig,
    SelfCorrectionEngine,
    StrategyRecommendation,
)
from aragora.nomic.task_decomposer import SubTask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_subtask(
    id: str = "st_1",
    title: str = "Sample subtask",
    description: str = "A sample subtask",
    file_scope: list[str] | None = None,
) -> SubTask:
    return SubTask(id=id, title=title, description=description, file_scope=file_scope or [])


def _make_assignment(
    status: str = "completed",
    track: Track = Track.QA,
    agent_type: str = "claude",
    subtask: SubTask | None = None,
) -> AgentAssignment:
    return AgentAssignment(
        subtask=subtask or _make_subtask(),
        track=track,
        agent_type=agent_type,
        status=status,
        completed_at=datetime.now(timezone.utc) if status in ("completed", "failed") else None,
    )


def _make_result(
    assignments: list[AgentAssignment] | None = None,
) -> OrchestrationResult:
    assignments = assignments or []
    completed = sum(1 for a in assignments if a.status == "completed")
    failed = sum(1 for a in assignments if a.status == "failed")
    return OrchestrationResult(
        goal="test goal",
        total_subtasks=len(assignments),
        completed_subtasks=completed,
        failed_subtasks=failed,
        skipped_subtasks=0,
        assignments=assignments,
        duration_seconds=1.0,
        success=failed == 0,
    )


# ---------------------------------------------------------------------------
# 1. Package-level exports
# ---------------------------------------------------------------------------


class TestSelfCorrectionExports:
    """Verify self-correction symbols are importable from aragora.nomic."""

    def test_engine_importable_from_package(self):
        from aragora.nomic import SelfCorrectionEngine as Cls

        assert Cls is SelfCorrectionEngine

    def test_config_importable_from_package(self):
        from aragora.nomic import SelfCorrectionConfig as Cls

        assert Cls is SelfCorrectionConfig

    def test_report_importable_from_package(self):
        from aragora.nomic import CorrectionReport as Cls

        assert Cls is CorrectionReport

    def test_recommendation_importable_from_package(self):
        from aragora.nomic import StrategyRecommendation as Cls

        assert Cls is StrategyRecommendation


# ---------------------------------------------------------------------------
# 2-4. _apply_self_correction on AutonomousOrchestrator
# ---------------------------------------------------------------------------


class TestApplySelfCorrection:
    """Tests for AutonomousOrchestrator._apply_self_correction."""

    def _build_orchestrator(self, self_correction=None, skip_init_sc: bool = False):
        """Construct an AutonomousOrchestrator with mocked internals."""
        with patch(
            "aragora.nomic.self_correction.SelfCorrectionEngine"
        ) as MockEngine:
            if skip_init_sc:
                # Simulate ImportError during __init__ so _self_correction stays None
                MockEngine.side_effect = ImportError("no engine")
            orch = AutonomousOrchestrator(
                require_human_approval=False,
                enable_curriculum=False,
            )
        # Override the engine after construction so we control its behaviour
        if not skip_init_sc:
            orch._self_correction = self_correction
        return orch

    def test_stores_adjustments_on_after_metrics(self):
        """_apply_self_correction stores adjustments dict on result.after_metrics."""
        mock_engine = MagicMock(spec=SelfCorrectionEngine)
        mock_engine.analyze_patterns.return_value = CorrectionReport(
            total_cycles=5,
            overall_success_rate=0.8,
            track_success_rates={"qa": 0.9},
            track_streaks={"qa": 3},
            agent_correlations={"claude": 0.9},
            failing_patterns=[],
        )
        mock_engine.compute_priority_adjustments.return_value = {"qa": 1.3}
        mock_engine.recommend_strategy_change.return_value = []

        orch = self._build_orchestrator(self_correction=mock_engine)
        assignments = [_make_assignment(status="completed", track=Track.QA)]
        result = _make_result(assignments)

        orch._apply_self_correction(assignments, result)

        assert result.after_metrics is not None
        assert "self_correction_adjustments" in result.after_metrics
        assert result.after_metrics["self_correction_adjustments"] == {"qa": 1.3}

    def test_stores_recommendations_on_after_metrics(self):
        """_apply_self_correction stores recommendations list on result.after_metrics."""
        rec = StrategyRecommendation(
            track="qa",
            recommendation="Decrease scope",
            reason="3 consecutive failures",
            confidence=0.8,
            action_type="decrease_scope",
        )
        mock_engine = MagicMock(spec=SelfCorrectionEngine)
        mock_engine.analyze_patterns.return_value = CorrectionReport(
            total_cycles=5,
            overall_success_rate=0.4,
            track_success_rates={"qa": 0.3},
            track_streaks={"qa": -3},
            agent_correlations={},
            failing_patterns=["Track 'qa' has 3 consecutive failures."],
        )
        mock_engine.compute_priority_adjustments.return_value = {"qa": 0.55}
        mock_engine.recommend_strategy_change.return_value = [rec]

        orch = self._build_orchestrator(self_correction=mock_engine)
        assignments = [_make_assignment(status="failed", track=Track.QA)]
        result = _make_result(assignments)

        orch._apply_self_correction(assignments, result)

        assert result.after_metrics is not None
        recs = result.after_metrics.get("self_correction_recommendations", [])
        assert len(recs) == 1
        assert recs[0]["track"] == "qa"
        assert recs[0]["action"] == "decrease_scope"
        assert recs[0]["confidence"] == 0.8

    def test_feeds_recommendations_to_feedback_loop(self):
        """_apply_self_correction passes recommendations to feedback_loop."""
        rec = StrategyRecommendation(
            track="developer",
            recommendation="Rotate agent",
            reason="Low correlation",
            confidence=0.7,
            action_type="rotate_agent",
        )
        mock_engine = MagicMock(spec=SelfCorrectionEngine)
        mock_engine.analyze_patterns.return_value = CorrectionReport(
            total_cycles=5,
            overall_success_rate=0.5,
            track_success_rates={"developer": 0.4},
            track_streaks={"developer": -2},
            agent_correlations={"codex": 0.2},
            failing_patterns=[],
        )
        mock_engine.compute_priority_adjustments.return_value = {"developer": 0.85}
        mock_engine.recommend_strategy_change.return_value = [rec]

        orch = self._build_orchestrator(self_correction=mock_engine)
        assignments = [
            _make_assignment(status="failed", track=Track.DEVELOPER, agent_type="codex"),
        ]
        result = _make_result(assignments)

        orch._apply_self_correction(assignments, result)

        # The feedback loop should have received the recommendation
        got = orch.feedback_loop.get_recommendation_for_track("developer")
        assert got is not None
        assert got["action"] == "rotate_agent"

    def test_no_op_when_no_completed_or_failed(self):
        """_apply_self_correction is a no-op when all assignments are pending."""
        mock_engine = MagicMock(spec=SelfCorrectionEngine)
        orch = self._build_orchestrator(self_correction=mock_engine)
        assignments = [_make_assignment(status="pending")]
        result = _make_result(assignments)

        orch._apply_self_correction(assignments, result)

        # Engine should not have been called because no outcome dicts are built
        mock_engine.analyze_patterns.assert_not_called()
        assert result.after_metrics is None

    def test_no_op_when_self_correction_is_none(self):
        """_apply_self_correction is a no-op when _self_correction is None."""
        orch = self._build_orchestrator(skip_init_sc=True)
        assert orch._self_correction is None

        assignments = [_make_assignment(status="completed")]
        result = _make_result(assignments)

        # Should return immediately without error
        orch._apply_self_correction(assignments, result)
        assert result.after_metrics is None


# ---------------------------------------------------------------------------
# 5-6. FeedbackLoop strategy recommendation storage
# ---------------------------------------------------------------------------


class TestFeedbackLoopRecommendations:
    """Tests for FeedbackLoop.apply_strategy_recommendations / get_recommendation_for_track."""

    def test_apply_stores_recommendations(self):
        """apply_strategy_recommendations stores the provided list."""
        fl = FeedbackLoop(max_iterations=3)
        recs = [
            StrategyRecommendation(
                track="qa", recommendation="Smaller PRs", reason="x",
                confidence=0.7, action_type="decrease_scope",
            ),
            StrategyRecommendation(
                track="sme", recommendation="Rotate agent", reason="y",
                confidence=0.5, action_type="rotate_agent",
            ),
        ]
        fl.apply_strategy_recommendations(recs)
        assert len(fl._strategy_recommendations) == 2

    def test_apply_replaces_previous(self):
        """Calling apply_strategy_recommendations again replaces the old list."""
        fl = FeedbackLoop()
        fl.apply_strategy_recommendations([
            StrategyRecommendation(
                track="qa", recommendation="A", reason="r",
                confidence=0.6, action_type="decrease_scope",
            ),
        ])
        assert len(fl._strategy_recommendations) == 1

        fl.apply_strategy_recommendations([
            StrategyRecommendation(
                track="sme", recommendation="B", reason="r2",
                confidence=0.8, action_type="rotate_agent",
            ),
            StrategyRecommendation(
                track="core", recommendation="C", reason="r3",
                confidence=0.9, action_type="deprioritize",
            ),
        ])
        assert len(fl._strategy_recommendations) == 2

    def test_get_recommendation_returns_highest_confidence(self):
        """get_recommendation_for_track returns the recommendation with highest confidence."""
        fl = FeedbackLoop()
        fl.apply_strategy_recommendations([
            StrategyRecommendation(
                track="qa", recommendation="Lower", reason="r1",
                confidence=0.4, action_type="decrease_scope",
            ),
            StrategyRecommendation(
                track="qa", recommendation="Higher", reason="r2",
                confidence=0.9, action_type="rotate_agent",
            ),
            StrategyRecommendation(
                track="sme", recommendation="Other", reason="r3",
                confidence=0.95, action_type="deprioritize",
            ),
        ])
        got = fl.get_recommendation_for_track("qa")
        assert got is not None
        assert got["recommendation"] == "Higher"
        assert got["action"] == "rotate_agent"
        assert got["confidence"] == "0.9"

    def test_get_recommendation_returns_none_for_missing_track(self):
        """get_recommendation_for_track returns None when no rec matches the track."""
        fl = FeedbackLoop()
        fl.apply_strategy_recommendations([
            StrategyRecommendation(
                track="qa", recommendation="A", reason="r",
                confidence=0.7, action_type="decrease_scope",
            ),
        ])
        assert fl.get_recommendation_for_track("developer") is None

    def test_get_recommendation_returns_none_when_empty(self):
        """get_recommendation_for_track returns None when no recommendations exist."""
        fl = FeedbackLoop()
        assert fl.get_recommendation_for_track("qa") is None


# ---------------------------------------------------------------------------
# 7-8. MetaPlanner._apply_self_correction_adjustments
# ---------------------------------------------------------------------------


class TestMetaPlannerSelfCorrectionAdjustments:
    """Tests for MetaPlanner._apply_self_correction_adjustments."""

    def _make_goals(self) -> list[PrioritizedGoal]:
        return [
            PrioritizedGoal(
                id="goal_0", track=Track.SME,
                description="Improve SME dashboard",
                rationale="Direct SME value",
                estimated_impact="high", priority=1,
            ),
            PrioritizedGoal(
                id="goal_1", track=Track.QA,
                description="Add E2E tests",
                rationale="Reliability",
                estimated_impact="medium", priority=2,
            ),
            PrioritizedGoal(
                id="goal_2", track=Track.DEVELOPER,
                description="Improve SDK docs",
                rationale="Developer experience",
                estimated_impact="medium", priority=3,
            ),
        ]

    @patch("aragora.nomic.self_correction.SelfCorrectionEngine")
    def test_reranks_goals_with_adjustments(self, MockEngine):
        """Goals get re-ranked when adjustments differ across tracks."""
        mock_engine = MockEngine.return_value

        # SME is penalized (0.5), QA is neutral (1.0), DEVELOPER gets a big boost (2.0)
        # Initial priorities: SME=1, QA=2, DEVELOPER=3
        # After adjustment:
        #   SME: max(1, round(1 / 0.5)) = 2
        #   QA:  max(1, round(2 / 1.0)) = 2
        #   DEV: max(1, round(3 / 2.0)) = 2  -> all become 2, tied by stable sort
        # So we use values that create clear separation:
        #   SME adj=0.3 -> max(1, round(1/0.3)) = 3
        #   QA  adj=1.0 -> max(1, round(2/1.0)) = 2
        #   DEV adj=2.0 -> max(1, round(3/2.0)) = 2 (tie with QA, but QA was originally 2nd)
        # Better: use priorities that separate cleanly
        report = CorrectionReport(
            total_cycles=10,
            overall_success_rate=0.6,
            track_success_rates={"qa": 0.9, "sme": 0.2, "developer": 0.7},
            track_streaks={"qa": 4, "sme": -3, "developer": 2},
            agent_correlations={},
            failing_patterns=[],
        )
        mock_engine.analyze_patterns.return_value = report
        # DEVELOPER has highest boost -> priority decreases (better)
        # SME is penalized -> priority increases (worse)
        mock_engine.compute_priority_adjustments.return_value = {
            "sme": 0.3,       # SME: round(1/0.3) = 3
            "qa": 1.0,        # QA:  round(2/1.0) = 2
            "developer": 2.0, # DEV: round(3/2.0) = 2 (tie, but stable-sort keeps QA first)
        }

        planner = MetaPlanner(config=MetaPlannerConfig(quick_mode=True))
        goals = self._make_goals()

        # Patch _get_past_outcomes to return some data
        with patch.object(planner, "_get_past_outcomes", return_value=[
            {"track": "qa", "success": True, "agent": "claude"},
            {"track": "qa", "success": True, "agent": "claude"},
            {"track": "developer", "success": True, "agent": "codex"},
            {"track": "sme", "success": False, "agent": "claude"},
        ]):
            adjusted = planner._apply_self_correction_adjustments(goals)

        # SME should be last (penalized), QA or DEVELOPER first
        assert adjusted[-1].track == Track.SME
        # First two are QA and DEVELOPER (tied at round-value 2, stable sort preserved)
        assert adjusted[0].track in (Track.QA, Track.DEVELOPER)
        assert adjusted[1].track in (Track.QA, Track.DEVELOPER)
        # Sequential priority re-assignment
        assert [g.priority for g in adjusted] == [1, 2, 3]

    def test_no_op_when_no_past_outcomes(self):
        """_apply_self_correction_adjustments is a no-op when _get_past_outcomes is empty."""
        planner = MetaPlanner(config=MetaPlannerConfig(quick_mode=True))
        goals = self._make_goals()
        original_order = [g.track for g in goals]

        with patch.object(planner, "_get_past_outcomes", return_value=[]):
            result = planner._apply_self_correction_adjustments(goals)

        # Order should be unchanged
        assert [g.track for g in result] == original_order

    @patch("aragora.nomic.self_correction.SelfCorrectionEngine")
    def test_handles_engine_exception_gracefully(self, MockEngine):
        """Engine exceptions are caught and goals are returned unchanged."""
        mock_engine = MockEngine.return_value
        mock_engine.analyze_patterns.side_effect = RuntimeError("boom")

        planner = MetaPlanner(config=MetaPlannerConfig(quick_mode=True))
        goals = self._make_goals()
        original_order = [g.track for g in goals]

        with patch.object(planner, "_get_past_outcomes", return_value=[
            {"track": "qa", "success": True},
        ]):
            result = planner._apply_self_correction_adjustments(goals)

        assert [g.track for g in result] == original_order


# ---------------------------------------------------------------------------
# 9. Full pipeline: execute_goal triggers self-correction
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end test: execute_goal runs self-correction after task execution."""

    @pytest.mark.asyncio
    async def test_execute_goal_calls_self_correction(self, tmp_path):
        """execute_goal triggers _apply_self_correction after computing the result."""
        subtask = _make_subtask(
            id="st_pipe_1",
            title="Add test coverage for QA",
            description="Improve test coverage for critical modules",
            file_scope=["tests/nomic/test_example.py"],
        )

        mock_decomposer = MagicMock()
        mock_decomposition = MagicMock()
        mock_decomposition.subtasks = [subtask]
        mock_decomposer.analyze.return_value = mock_decomposition
        mock_decomposer.analyze_with_debate = AsyncMock(return_value=mock_decomposition)

        # Mock the SelfCorrectionEngine at the import location
        with patch(
            "aragora.nomic.self_correction.SelfCorrectionEngine"
        ) as MockEngineClass:
            mock_engine = MockEngineClass.return_value
            mock_engine.analyze_patterns.return_value = CorrectionReport(
                total_cycles=5,
                overall_success_rate=0.8,
                track_success_rates={"qa": 0.9},
                track_streaks={"qa": 2},
                agent_correlations={"claude": 0.85},
                failing_patterns=[],
            )
            mock_engine.compute_priority_adjustments.return_value = {"qa": 1.2}
            mock_engine.recommend_strategy_change.return_value = [
                StrategyRecommendation(
                    track="qa",
                    recommendation="Keep the momentum",
                    reason="Consecutive successes",
                    confidence=0.75,
                    action_type="increase_scope",
                )
            ]

            orch = AutonomousOrchestrator(
                aragora_path=tmp_path,
                task_decomposer=mock_decomposer,
                require_human_approval=False,
                enable_curriculum=False,
            )

        # Patch _execute_assignments to simulate a successful run
        async def fake_execute_all(assignments, max_cycles):
            for a in assignments:
                a.status = "completed"
                a.completed_at = datetime.now(timezone.utc)
                a.result = {"output": "done"}

        with (
            patch.object(orch, "_execute_assignments", side_effect=fake_execute_all),
            patch.object(orch, "_store_priority_adjustments"),
            patch.object(orch, "_checkpoint"),
            patch.object(orch, "_emit_improvement_event"),
        ):
            result = await orch.execute_goal("Improve test coverage", tracks=["qa"])

        # Verify self-correction was invoked
        mock_engine.analyze_patterns.assert_called_once()
        mock_engine.compute_priority_adjustments.assert_called_once()
        mock_engine.recommend_strategy_change.assert_called_once()

        # Verify results contain self-correction data
        assert result.after_metrics is not None
        assert "self_correction_adjustments" in result.after_metrics
        assert result.after_metrics["self_correction_adjustments"] == {"qa": 1.2}
        assert "self_correction_recommendations" in result.after_metrics
        assert len(result.after_metrics["self_correction_recommendations"]) == 1
        assert result.after_metrics["self_correction_recommendations"][0]["action"] == "increase_scope"

    @pytest.mark.asyncio
    async def test_execute_goal_succeeds_when_self_correction_disabled(self, tmp_path):
        """execute_goal still succeeds when SelfCorrectionEngine import fails."""
        subtask = _make_subtask(
            id="st_no_sc",
            title="Add test coverage for QA",
            description="Improve test coverage for critical modules",
            file_scope=["tests/nomic/test_example.py"],
        )

        mock_decomposer = MagicMock()
        mock_decomposition = MagicMock()
        mock_decomposition.subtasks = [subtask]
        mock_decomposer.analyze.return_value = mock_decomposition
        mock_decomposer.analyze_with_debate = AsyncMock(return_value=mock_decomposition)

        with patch(
            "aragora.nomic.self_correction.SelfCorrectionEngine",
            side_effect=ImportError("unavailable"),
        ):
            orch = AutonomousOrchestrator(
                aragora_path=tmp_path,
                task_decomposer=mock_decomposer,
                require_human_approval=False,
                enable_curriculum=False,
            )

        assert orch._self_correction is None

        async def fake_execute_all(assignments, max_cycles):
            for a in assignments:
                a.status = "completed"
                a.completed_at = datetime.now(timezone.utc)
                a.result = {"output": "done"}

        with (
            patch.object(orch, "_execute_assignments", side_effect=fake_execute_all),
            patch.object(orch, "_store_priority_adjustments"),
            patch.object(orch, "_checkpoint"),
            patch.object(orch, "_emit_improvement_event"),
        ):
            result = await orch.execute_goal("Test goal", tracks=["qa"])

        assert result.success
        # No self-correction data should be stored
        assert result.after_metrics is None
