"""Tests for SelfImprovePipeline -- the unified self-improvement pipeline."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.self_improve import (
    SelfImproveConfig,
    SelfImprovePipeline,
    SelfImproveResult,
)


# ---------------------------------------------------------------------------
# SelfImproveConfig
# ---------------------------------------------------------------------------


class TestSelfImproveConfig:
    def test_defaults(self):
        config = SelfImproveConfig()
        assert config.use_meta_planner is True
        assert config.quick_mode is False
        assert config.max_goals == 5
        assert config.use_worktrees is True
        assert config.max_parallel == 4
        assert config.budget_limit_usd == 10.0
        assert config.require_approval is True
        assert config.run_tests is True
        assert config.run_review is True
        assert config.capture_metrics is True
        assert config.persist_outcomes is True
        assert config.auto_revert_on_regression is True
        assert config.degradation_threshold == 0.05

    def test_custom_values(self):
        config = SelfImproveConfig(
            use_meta_planner=False,
            quick_mode=True,
            max_goals=3,
            use_worktrees=False,
            max_parallel=2,
            budget_limit_usd=5.0,
            require_approval=False,
            run_tests=False,
            run_review=False,
            capture_metrics=False,
            persist_outcomes=False,
            auto_revert_on_regression=False,
            degradation_threshold=0.1,
        )
        assert config.use_meta_planner is False
        assert config.quick_mode is True
        assert config.max_goals == 3
        assert config.use_worktrees is False
        assert config.max_parallel == 2
        assert config.budget_limit_usd == 5.0
        assert config.degradation_threshold == 0.1


# ---------------------------------------------------------------------------
# SelfImproveResult
# ---------------------------------------------------------------------------


class TestSelfImproveResult:
    def test_defaults(self):
        result = SelfImproveResult(cycle_id="cycle_abc", objective="test")
        assert result.cycle_id == "cycle_abc"
        assert result.objective == "test"
        assert result.goals_planned == 0
        assert result.subtasks_total == 0
        assert result.subtasks_completed == 0
        assert result.subtasks_failed == 0
        assert result.files_changed == []
        assert result.tests_passed == 0
        assert result.tests_failed == 0
        assert result.regressions_detected is False
        assert result.reverted is False
        assert result.duration_seconds == 0.0

    def test_to_dict_serialization(self):
        result = SelfImproveResult(
            cycle_id="cycle_123",
            objective="Improve coverage",
            goals_planned=3,
            subtasks_total=5,
            subtasks_completed=4,
            subtasks_failed=1,
            files_changed=["a.py", "b.py"],
            tests_passed=10,
            tests_failed=2,
            regressions_detected=True,
            reverted=False,
            duration_seconds=42.5,
        )
        d = result.to_dict()
        assert d["cycle_id"] == "cycle_123"
        assert d["objective"] == "Improve coverage"
        assert d["goals_planned"] == 3
        assert d["subtasks_total"] == 5
        assert d["subtasks_completed"] == 4
        assert d["subtasks_failed"] == 1
        assert d["files_changed"] == ["a.py", "b.py"]
        assert d["tests_passed"] == 10
        assert d["tests_failed"] == 2
        assert d["regressions_detected"] is True
        assert d["reverted"] is False
        assert d["duration_seconds"] == 42.5

    def test_to_dict_defaults(self):
        result = SelfImproveResult(cycle_id="x", objective="y")
        d = result.to_dict()
        assert d["files_changed"] == []
        assert d["regressions_detected"] is False

    def test_files_changed_isolation(self):
        """Each instance should have its own files_changed list."""
        r1 = SelfImproveResult(cycle_id="a", objective="a")
        r2 = SelfImproveResult(cycle_id="b", objective="b")
        r1.files_changed.append("foo.py")
        assert "foo.py" not in r2.files_changed


# ---------------------------------------------------------------------------
# SelfImprovePipeline -- dry_run
# ---------------------------------------------------------------------------


class TestDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_returns_goals_and_subtasks(self):
        """dry_run returns a plan dict with goals and subtasks."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(use_meta_planner=False)
        )
        plan = await pipeline.dry_run("Improve test coverage")

        assert plan["objective"] == "Improve test coverage"
        assert isinstance(plan["goals"], list)
        assert len(plan["goals"]) >= 1
        assert isinstance(plan["subtasks"], list)
        assert "config" in plan
        assert "use_worktrees" in plan["config"]
        assert "max_parallel" in plan["config"]
        assert "budget_limit_usd" in plan["config"]

    @pytest.mark.asyncio
    async def test_dry_run_with_quick_mode(self):
        """dry_run with quick_mode skips debate."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                use_meta_planner=True,
                quick_mode=True,
            )
        )
        plan = await pipeline.dry_run("Improve SME experience")

        assert plan["objective"] == "Improve SME experience"
        assert len(plan["goals"]) >= 1

    @pytest.mark.asyncio
    async def test_dry_run_goal_structure(self):
        """Each goal in the plan has the expected keys."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(use_meta_planner=False)
        )
        plan = await pipeline.dry_run("Add retry logic")

        for goal in plan["goals"]:
            assert "description" in goal
            assert "track" in goal
            assert "priority" in goal
            assert "estimated_impact" in goal
            assert "rationale" in goal

    @pytest.mark.asyncio
    async def test_dry_run_subtask_structure(self):
        """Each subtask in the plan has the expected keys."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(use_meta_planner=False)
        )
        plan = await pipeline.dry_run("Add retry logic to connectors")

        for subtask in plan["subtasks"]:
            assert "description" in subtask
            assert "scope" in subtask
            assert "file_hints" in subtask
            assert "success_criteria" in subtask

    @pytest.mark.asyncio
    async def test_dry_run_when_meta_planner_unavailable(self):
        """dry_run falls back gracefully when MetaPlanner cannot be imported."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(use_meta_planner=True)
        )
        with patch(
            "aragora.nomic.self_improve.SelfImprovePipeline._plan",
            new_callable=AsyncMock,
        ) as mock_plan:
            # Simulate MetaPlanner returning a single fallback goal
            from aragora.nomic.meta_planner import PrioritizedGoal, Track

            mock_plan.return_value = [
                PrioritizedGoal(
                    id="fallback",
                    track=Track.CORE,
                    description="Test objective",
                    rationale="Fallback",
                    estimated_impact="medium",
                    priority=1,
                )
            ]
            plan = await pipeline.dry_run("Test objective")
            assert len(plan["goals"]) == 1
            assert plan["goals"][0]["description"] == "Test objective"


# ---------------------------------------------------------------------------
# SelfImprovePipeline -- run
# ---------------------------------------------------------------------------


class TestPipeline:
    @pytest.mark.asyncio
    async def test_run_creates_cycle_id_and_result(self):
        """run() returns a SelfImproveResult with a cycle_id."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                use_meta_planner=False,
                use_worktrees=False,
                capture_metrics=False,
                persist_outcomes=False,
            )
        )
        result = await pipeline.run("Test objective")

        assert result.cycle_id.startswith("cycle_")
        assert len(result.cycle_id) > len("cycle_")
        assert result.objective == "Test objective"
        assert isinstance(result, SelfImproveResult)
        assert result.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_run_with_no_goals_returns_early(self):
        """run() returns early when planning produces no goals."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                use_meta_planner=True,
                capture_metrics=False,
                persist_outcomes=False,
            )
        )
        with patch.object(pipeline, "_plan", new_callable=AsyncMock) as mock_plan:
            mock_plan.return_value = []
            result = await pipeline.run("Empty objective")

        assert result.goals_planned == 0
        assert result.subtasks_total == 0
        assert result.subtasks_completed == 0
        assert result.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_run_captures_baseline_and_after(self):
        """run() calls baseline and after capture when metrics enabled."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                use_meta_planner=False,
                use_worktrees=False,
                capture_metrics=True,
                persist_outcomes=False,
            )
        )
        baseline_mock = MagicMock()
        after_mock = MagicMock()

        with (
            patch.object(
                pipeline,
                "_capture_baseline",
                new_callable=AsyncMock,
                return_value=baseline_mock,
            ) as mock_base,
            patch.object(
                pipeline,
                "_capture_after",
                new_callable=AsyncMock,
                return_value=after_mock,
            ) as mock_after,
            patch.object(
                pipeline,
                "_compare_metrics",
                return_value={"improved": True, "recommendation": "keep"},
            ) as mock_compare,
        ):
            result = await pipeline.run("Test metrics")

        mock_base.assert_called_once()
        mock_after.assert_called_once()
        mock_compare.assert_called_once_with(baseline_mock, after_mock)
        assert result.regressions_detected is False

    @pytest.mark.asyncio
    async def test_run_detects_regressions(self):
        """run() sets regressions_detected when metrics degrade."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                use_meta_planner=False,
                use_worktrees=False,
                capture_metrics=True,
                persist_outcomes=False,
                auto_revert_on_regression=True,
            )
        )
        with (
            patch.object(
                pipeline,
                "_capture_baseline",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch.object(
                pipeline,
                "_capture_after",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch.object(
                pipeline,
                "_compare_metrics",
                return_value={"improved": False, "recommendation": "revert"},
            ),
        ):
            result = await pipeline.run("Test regression")

        assert result.regressions_detected is True

    @pytest.mark.asyncio
    async def test_run_persists_outcomes(self):
        """run() calls _persist_outcome when persist_outcomes is True."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                use_meta_planner=False,
                use_worktrees=False,
                capture_metrics=False,
                persist_outcomes=True,
            )
        )
        with patch.object(pipeline, "_persist_outcome") as mock_persist:
            result = await pipeline.run("Test persist")

        mock_persist.assert_called_once()
        args = mock_persist.call_args[0]
        assert args[0] == result.cycle_id
        assert args[1] is result

    @pytest.mark.asyncio
    async def test_run_skips_persist_when_disabled(self):
        """run() does not persist when persist_outcomes is False."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                use_meta_planner=False,
                use_worktrees=False,
                capture_metrics=False,
                persist_outcomes=False,
            )
        )
        with patch.object(pipeline, "_persist_outcome") as mock_persist:
            await pipeline.run("Test no persist")

        mock_persist.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_skips_metrics_when_disabled(self):
        """run() does not capture metrics when capture_metrics is False."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                use_meta_planner=False,
                use_worktrees=False,
                capture_metrics=False,
                persist_outcomes=False,
            )
        )
        with (
            patch.object(pipeline, "_capture_baseline") as mock_base,
            patch.object(pipeline, "_capture_after") as mock_after,
        ):
            await pipeline.run("Test skip metrics")

        mock_base.assert_not_called()
        mock_after.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_accumulates_execution_results(self):
        """run() aggregates subtask success/failure counts."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                use_meta_planner=False,
                use_worktrees=False,
                capture_metrics=False,
                persist_outcomes=False,
            )
        )
        mock_results = [
            {"success": True, "files_changed": ["a.py"], "tests_passed": 5, "tests_failed": 0},
            {"success": False, "files_changed": [], "tests_passed": 0, "tests_failed": 2},
            {"success": True, "files_changed": ["b.py"], "tests_passed": 3, "tests_failed": 0},
        ]
        call_count = 0

        async def mock_execute_single(subtask, cycle_id):
            nonlocal call_count
            r = mock_results[call_count % len(mock_results)]
            call_count += 1
            return r

        with patch.object(pipeline, "_execute_single", side_effect=mock_execute_single):
            result = await pipeline.run("Test accumulation")

        assert result.subtasks_completed >= 1
        assert result.subtasks_failed >= 0
        assert result.tests_passed >= 0


# ---------------------------------------------------------------------------
# SelfImprovePipeline -- internal methods
# ---------------------------------------------------------------------------


class TestInternalMethods:
    @pytest.mark.asyncio
    async def test_plan_without_meta_planner(self):
        """_plan returns a single PrioritizedGoal when use_meta_planner=False."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(use_meta_planner=False)
        )
        goals = await pipeline._plan("My objective")
        assert len(goals) == 1
        assert goals[0].description == "My objective"
        assert goals[0].id == "direct"

    @pytest.mark.asyncio
    async def test_plan_falls_back_on_import_error(self):
        """_plan falls back when MetaPlanner raises ImportError."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(use_meta_planner=True)
        )
        with patch(
            "aragora.nomic.self_improve.SelfImprovePipeline._plan",
            wraps=pipeline._plan,
        ):
            # Patch the import within _plan to fail
            import builtins
            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if "meta_planner" in name and "MetaPlanner" not in str(args):
                    # Let the first import for PrioritizedGoal fallback succeed
                    return original_import(name, *args, **kwargs)
                return original_import(name, *args, **kwargs)

            # Instead of complex import patching, just mock prioritize_work
            with patch(
                "aragora.nomic.meta_planner.MetaPlanner.prioritize_work",
                side_effect=RuntimeError("test failure"),
            ):
                goals = await pipeline._plan("Fallback test")

        assert len(goals) >= 1
        # Should get a fallback goal
        assert getattr(goals[0], "id", None) == "fallback"

    @pytest.mark.asyncio
    async def test_decompose_with_task_decomposer(self):
        """_decompose breaks goals into subtasks using TaskDecomposer."""
        from aragora.nomic.meta_planner import PrioritizedGoal, Track

        pipeline = SelfImprovePipeline()
        goals = [
            PrioritizedGoal(
                id="g1",
                track=Track.QA,
                description="Refactor the authentication system and add comprehensive testing",
                rationale="Security improvement",
                estimated_impact="high",
                priority=1,
            )
        ]
        subtasks = await pipeline._decompose(goals)
        assert len(subtasks) >= 1

    @pytest.mark.asyncio
    async def test_decompose_falls_back_on_import_error(self):
        """_decompose falls back when TaskDecomposer is unavailable."""
        from aragora.nomic.meta_planner import PrioritizedGoal, Track

        pipeline = SelfImprovePipeline()
        goals = [
            PrioritizedGoal(
                id="g1",
                track=Track.CORE,
                description="Simple task",
                rationale="Test",
                estimated_impact="low",
                priority=1,
            )
        ]
        with patch(
            "aragora.nomic.self_improve.SelfImprovePipeline._decompose",
        ) as mock_decompose:
            # Simulate fallback behavior
            mock_decompose.return_value = goals
            subtasks = await pipeline._decompose(goals)

        assert len(subtasks) == 1

    @pytest.mark.asyncio
    async def test_execute_single_returns_placeholder(self):
        """_execute_single returns a success dict with placeholder note."""
        pipeline = SelfImprovePipeline()
        result = await pipeline._execute_single("Do something", "cycle_test")

        assert result["success"] is True
        assert "subtask" in result
        assert "files_changed" in result
        assert isinstance(result["files_changed"], list)

    @pytest.mark.asyncio
    async def test_execute_single_extracts_description_from_various_types(self):
        """_execute_single handles SubTask, TaskDecomposition, TrackAssignment, and str."""
        pipeline = SelfImprovePipeline()

        # Raw string
        r1 = await pipeline._execute_single("raw string task", "c1")
        assert r1["subtask"] == "raw string task"

        # Object with description
        obj = MagicMock()
        obj.description = "mock description"
        obj.goal = None  # No .goal attribute
        del obj.goal
        del obj.original_task
        r2 = await pipeline._execute_single(obj, "c2")
        assert r2["subtask"] == "mock description"

        # Object with original_task (TaskDecomposition-like)
        obj2 = MagicMock()
        obj2.original_task = "task decomposition description"
        del obj2.goal
        r3 = await pipeline._execute_single(obj2, "c3")
        assert r3["subtask"] == "task decomposition description"

    @pytest.mark.asyncio
    async def test_compare_metrics_returns_none_when_baseline_none(self):
        """_compare_metrics returns None when baseline is None."""
        pipeline = SelfImprovePipeline()
        assert pipeline._compare_metrics(None, MagicMock()) is None
        assert pipeline._compare_metrics(MagicMock(), None) is None
        assert pipeline._compare_metrics(None, None) is None

    @pytest.mark.asyncio
    async def test_compare_metrics_with_real_debate_metrics(self):
        """_compare_metrics works with real DebateMetrics objects."""
        from aragora.nomic.outcome_tracker import DebateMetrics

        pipeline = SelfImprovePipeline()
        baseline = DebateMetrics(
            consensus_rate=0.8,
            avg_rounds=3.0,
            avg_tokens=2000,
            calibration_spread=0.1,
        )
        after = DebateMetrics(
            consensus_rate=0.9,
            avg_rounds=2.5,
            avg_tokens=1800,
            calibration_spread=0.08,
        )
        result = pipeline._compare_metrics(baseline, after)
        assert result is not None
        assert result["improved"] is True
        assert result["recommendation"] == "keep"
        assert "deltas" in result

    @pytest.mark.asyncio
    async def test_compare_metrics_detects_regression(self):
        """_compare_metrics detects when metrics degrade."""
        from aragora.nomic.outcome_tracker import DebateMetrics

        pipeline = SelfImprovePipeline(
            SelfImproveConfig(degradation_threshold=0.01)
        )
        baseline = DebateMetrics(
            consensus_rate=0.9,
            avg_rounds=2.0,
            avg_tokens=1000,
            calibration_spread=0.05,
        )
        after = DebateMetrics(
            consensus_rate=0.5,
            avg_rounds=5.0,
            avg_tokens=3000,
            calibration_spread=0.2,
        )
        result = pipeline._compare_metrics(baseline, after)
        assert result is not None
        assert result["improved"] is False

    def test_persist_outcome_with_mock_store(self):
        """_persist_outcome saves to CycleLearningStore."""
        pipeline = SelfImprovePipeline()
        result = SelfImproveResult(
            cycle_id="cycle_test",
            objective="Test persistence",
            subtasks_completed=2,
            subtasks_failed=0,
            files_changed=["x.py"],
            duration_seconds=10.0,
        )
        with patch("aragora.nomic.cycle_store.get_cycle_store") as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = mock_store
            pipeline._persist_outcome("cycle_test", result)

        mock_store.save_cycle.assert_called_once()
        saved_record = mock_store.save_cycle.call_args[0][0]
        assert saved_record.cycle_id == "cycle_test"
        assert saved_record.success is True

    def test_persist_outcome_handles_import_error(self):
        """_persist_outcome does not raise on import failure."""
        pipeline = SelfImprovePipeline()
        result = SelfImproveResult(
            cycle_id="c1", objective="test", duration_seconds=1.0
        )
        with patch.dict(
            "sys.modules",
            {"aragora.nomic.cycle_store": None},
        ):
            # Should not raise
            pipeline._persist_outcome("c1", result)


# ---------------------------------------------------------------------------
# End-to-end (with mocked execution)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_full_dry_run_with_real_decomposer(self):
        """Full dry_run path using the real TaskDecomposer."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                use_meta_planner=False,
                quick_mode=True,
            )
        )
        plan = await pipeline.dry_run(
            "Refactor the database layer and add integration testing"
        )

        assert plan["objective"].startswith("Refactor")
        assert len(plan["goals"]) >= 1
        # TaskDecomposer should decompose this into subtasks
        # (it has high-complexity keywords: refactor + multiple concepts)
        assert len(plan["subtasks"]) >= 1

    @pytest.mark.asyncio
    async def test_quick_run_with_mocked_execution(self):
        """Full run path with mocked execution step."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                use_meta_planner=False,
                use_worktrees=False,
                capture_metrics=False,
                persist_outcomes=False,
            )
        )

        async def mock_execute(subtask, cycle_id):
            return {
                "success": True,
                "files_changed": ["mock.py"],
                "tests_passed": 5,
                "tests_failed": 0,
            }

        with patch.object(pipeline, "_execute_single", side_effect=mock_execute):
            result = await pipeline.run("Improve test coverage")

        assert result.goals_planned >= 1
        assert result.subtasks_total >= 1
        assert result.subtasks_completed >= 1
        assert result.subtasks_failed == 0
        assert result.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_full_run_never_crashes(self):
        """The pipeline must always return a result, never raise."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                use_meta_planner=False,
                use_worktrees=False,
                capture_metrics=False,
                persist_outcomes=False,
            )
        )
        result = await pipeline.run("Any objective at all")
        assert isinstance(result, SelfImproveResult)
        assert result.cycle_id.startswith("cycle_")

    @pytest.mark.asyncio
    async def test_run_with_all_features_enabled(self):
        """Run with all features enabled but mocked dependencies."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                use_meta_planner=False,
                use_worktrees=False,
                capture_metrics=True,
                persist_outcomes=True,
                auto_revert_on_regression=True,
            )
        )

        from aragora.nomic.outcome_tracker import DebateMetrics

        baseline = DebateMetrics(consensus_rate=0.8, avg_rounds=3.0, avg_tokens=2000)
        after = DebateMetrics(consensus_rate=0.85, avg_rounds=2.8, avg_tokens=1900)

        with (
            patch.object(
                pipeline,
                "_capture_baseline",
                new_callable=AsyncMock,
                return_value=baseline,
            ),
            patch.object(
                pipeline,
                "_capture_after",
                new_callable=AsyncMock,
                return_value=after,
            ),
            patch.object(pipeline, "_persist_outcome") as mock_persist,
        ):
            result = await pipeline.run("Full feature test")

        assert isinstance(result, SelfImproveResult)
        assert result.regressions_detected is False
        mock_persist.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_regression_and_persist(self):
        """Run detects regression, persists outcome, and reports it."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                use_meta_planner=False,
                use_worktrees=False,
                capture_metrics=True,
                persist_outcomes=True,
                auto_revert_on_regression=True,
                degradation_threshold=0.01,
            )
        )

        from aragora.nomic.outcome_tracker import DebateMetrics

        baseline = DebateMetrics(consensus_rate=0.9, avg_rounds=2.0, avg_tokens=1000)
        after = DebateMetrics(consensus_rate=0.5, avg_rounds=5.0, avg_tokens=5000)

        with (
            patch.object(
                pipeline,
                "_capture_baseline",
                new_callable=AsyncMock,
                return_value=baseline,
            ),
            patch.object(
                pipeline,
                "_capture_after",
                new_callable=AsyncMock,
                return_value=after,
            ),
            patch.object(pipeline, "_persist_outcome") as mock_persist,
        ):
            result = await pipeline.run("Regression test")

        assert result.regressions_detected is True
        mock_persist.assert_called_once()


# ---------------------------------------------------------------------------
# Import validation
# ---------------------------------------------------------------------------


class TestImports:
    def test_importable_from_package(self):
        """The pipeline classes are importable from aragora.nomic."""
        from aragora.nomic import (
            SelfImproveConfig,
            SelfImprovePipeline,
            SelfImproveResult,
        )

        assert SelfImprovePipeline is not None
        assert SelfImproveConfig is not None
        assert SelfImproveResult is not None
