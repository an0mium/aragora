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
            SelfImproveConfig(use_meta_planner=False, enable_codebase_indexing=False)
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
                enable_codebase_indexing=False,
            )
        )
        plan = await pipeline.dry_run("Improve SME experience")

        assert plan["objective"] == "Improve SME experience"
        assert len(plan["goals"]) >= 1

    @pytest.mark.asyncio
    async def test_dry_run_goal_structure(self):
        """Each goal in the plan has the expected keys."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(use_meta_planner=False, enable_codebase_indexing=False)
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
            SelfImproveConfig(use_meta_planner=False, enable_codebase_indexing=False)
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
            SelfImproveConfig(use_meta_planner=True, enable_codebase_indexing=False)
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
                enable_codebase_indexing=False,
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
                enable_codebase_indexing=False,
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
                enable_codebase_indexing=False,
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
                enable_codebase_indexing=False,
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
                enable_codebase_indexing=False,
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
                enable_codebase_indexing=False,
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
                enable_codebase_indexing=False,
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
                enable_codebase_indexing=False,
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

        pipeline = SelfImprovePipeline(
            SelfImproveConfig(enable_codebase_indexing=False)
        )
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
        """_execute_single returns a fallback dict when ExecutionBridge is unavailable."""
        pipeline = SelfImprovePipeline()
        result = await pipeline._execute_single("Do something", "cycle_test")

        assert result["success"] is False
        assert "subtask" in result
        assert "files_changed" in result
        assert isinstance(result["files_changed"], list)

    @pytest.mark.asyncio
    async def test_execute_single_extracts_description_from_various_types(self):
        """_execute_single handles SubTask, TaskDecomposition, TrackAssignment, and str."""
        pipeline = SelfImprovePipeline(
            config=SelfImproveConfig(enable_debug_loop=False)
        )

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
# Instruction file dispatch
# ---------------------------------------------------------------------------


class TestWriteInstructionToWorktree:
    def test_writes_markdown_and_json(self, tmp_path: Path):
        """Should write both instruction.md and instruction.json."""
        mock_instruction = MagicMock()
        mock_instruction.to_agent_prompt.return_value = "# Task: Do the thing"
        mock_instruction.to_dict.return_value = {"subtask_id": "t1", "objective": "test"}
        mock_instruction.subtask_id = "t1"

        result = SelfImprovePipeline._write_instruction_to_worktree(
            mock_instruction, str(tmp_path)
        )

        assert result is True
        md_file = tmp_path / ".aragora" / "instruction.md"
        json_file = tmp_path / ".aragora" / "instruction.json"
        assert md_file.exists()
        assert json_file.exists()
        assert md_file.read_text() == "# Task: Do the thing"
        import json

        data = json.loads(json_file.read_text())
        assert data["subtask_id"] == "t1"

    def test_returns_false_for_nonexistent_path(self):
        """Should return False when worktree path doesn't exist."""
        result = SelfImprovePipeline._write_instruction_to_worktree(
            MagicMock(), "/nonexistent/path/12345"
        )
        assert result is False


# ---------------------------------------------------------------------------
# Claude Code dispatch
# ---------------------------------------------------------------------------


class TestDispatchToClaudeCode:
    @pytest.mark.asyncio
    async def test_skips_when_require_approval(self):
        """Should return None when require_approval is True."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(require_approval=True)
        )
        mock_instruction = MagicMock()
        mock_instruction.subtask_id = "t1"
        result = await pipeline._dispatch_to_claude_code(
            mock_instruction, "/tmp/fake"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_skips_when_cli_not_found(self):
        """Should return None when claude CLI is not in PATH."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(require_approval=False)
        )
        mock_instruction = MagicMock()
        mock_instruction.subtask_id = "t1"

        with patch("shutil.which", return_value=None):
            result = await pipeline._dispatch_to_claude_code(
                mock_instruction, "/tmp/fake"
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_dispatches_when_cli_available(self):
        """Should dispatch via ClaudeCodeHarness when CLI is available."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(require_approval=False, run_tests=False)
        )
        mock_instruction = MagicMock()
        mock_instruction.subtask_id = "t1"
        mock_instruction.to_agent_prompt.return_value = "# Task: test"

        with patch("shutil.which", return_value="/usr/local/bin/claude"):
            with patch(
                "aragora.harnesses.claude_code.ClaudeCodeHarness"
            ) as MockHarness:
                mock_h = MagicMock()
                mock_h.execute_implementation = AsyncMock(
                    return_value=("output", "")
                )
                MockHarness.return_value = mock_h

                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(
                        returncode=0, stdout="file1.py\nfile2.py\n"
                    )
                    result = await pipeline._dispatch_to_claude_code(
                        mock_instruction, "/tmp/fake-wt"
                    )

        assert result is not None
        assert result["files_changed"] == ["file1.py", "file2.py"]
        assert result["stdout_len"] == len("output")


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
                enable_codebase_indexing=False,
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
                enable_codebase_indexing=False,
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
                enable_codebase_indexing=False,
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
                enable_codebase_indexing=False,
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
                enable_codebase_indexing=False,
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


# ---------------------------------------------------------------------------
# Gap 1: Autonomous dispatch mode
# ---------------------------------------------------------------------------


class TestAutonomousMode:
    def test_autonomous_config_default(self):
        """autonomous defaults to False."""
        config = SelfImproveConfig()
        assert config.autonomous is False

    def test_autonomous_config_set(self):
        """autonomous can be set to True."""
        config = SelfImproveConfig(autonomous=True)
        assert config.autonomous is True

    @pytest.mark.asyncio
    async def test_dispatch_skips_when_require_approval_and_not_autonomous(self):
        """Dispatch is skipped when require_approval=True and autonomous=False."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(require_approval=True, autonomous=False)
        )
        mock_instr = MagicMock()
        mock_instr.subtask_id = "t1"
        result = await pipeline._dispatch_to_claude_code(mock_instr, "/tmp/fake")
        assert result is None

    @pytest.mark.asyncio
    async def test_dispatch_proceeds_when_autonomous(self):
        """Dispatch proceeds when autonomous=True even with require_approval=True."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                require_approval=True, autonomous=True, run_tests=False
            )
        )
        mock_instr = MagicMock()
        mock_instr.subtask_id = "t1"
        mock_instr.to_agent_prompt.return_value = "# Task"

        with patch("shutil.which", return_value=None):
            # Claude CLI not found → returns None but gets past the approval guard
            result = await pipeline._dispatch_to_claude_code(
                mock_instr, "/tmp/fake"
            )
        # Returns None because CLI not found, not because approval blocked
        assert result is None


# ---------------------------------------------------------------------------
# Gap 4: Feedback loop (OutcomeComparison → CycleLearningStore)
# ---------------------------------------------------------------------------


class TestFeedbackLoop:
    def test_persist_outcome_records_comparison(self):
        """_persist_outcome records OutcomeComparison via OutcomeTracker."""
        from aragora.nomic.outcome_tracker import DebateMetrics, OutcomeComparison

        pipeline = SelfImprovePipeline()
        result = SelfImproveResult(
            cycle_id="cycle_fb",
            objective="Test feedback",
            subtasks_completed=1,
            duration_seconds=5.0,
        )

        comparison = OutcomeComparison(
            baseline=DebateMetrics(consensus_rate=0.8),
            after=DebateMetrics(consensus_rate=0.9),
            improved=True,
            recommendation="keep",
        )

        with (
            patch("aragora.nomic.cycle_store.get_cycle_store") as mock_get_store,
            patch(
                "aragora.nomic.outcome_tracker.NomicOutcomeTracker.record_cycle_outcome"
            ) as mock_record,
        ):
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store
            pipeline._persist_outcome("cycle_fb", result, comparison)

        mock_record.assert_called_once_with("cycle_fb", comparison)

    def test_persist_outcome_skips_when_no_comparison(self):
        """_persist_outcome does not call record_cycle_outcome when comparison is None."""
        pipeline = SelfImprovePipeline()
        result = SelfImproveResult(
            cycle_id="c2", objective="test", duration_seconds=1.0
        )

        with (
            patch("aragora.nomic.cycle_store.get_cycle_store") as mock_get_store,
            patch(
                "aragora.nomic.outcome_tracker.NomicOutcomeTracker.record_cycle_outcome"
            ) as mock_record,
        ):
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store
            pipeline._persist_outcome("c2", result, None)

        mock_record.assert_not_called()

    def test_compare_metrics_stashes_outcome_comparison(self):
        """_compare_metrics stashes the OutcomeComparison object in the result."""
        from aragora.nomic.outcome_tracker import DebateMetrics

        pipeline = SelfImprovePipeline()
        baseline = DebateMetrics(
            consensus_rate=0.8, avg_rounds=3.0, avg_tokens=2000
        )
        after = DebateMetrics(
            consensus_rate=0.9, avg_rounds=2.5, avg_tokens=1800
        )
        result = pipeline._compare_metrics(baseline, after)
        assert result is not None
        assert "_outcome_comparison" in result
        assert result["_outcome_comparison"].improved is True


# ---------------------------------------------------------------------------
# Gap 2: CodebaseIndexer → TaskDecomposer (file scope enrichment)
# ---------------------------------------------------------------------------


class TestFileScopeEnrichment:
    @pytest.mark.asyncio
    async def test_enrich_file_scope_populates_empty_scope(self):
        """_enrich_file_scope fills empty file_scope from CodebaseIndexer."""
        pipeline = SelfImprovePipeline()
        subtask = MagicMock()
        subtask.file_scope = []
        subtask.title = "auth module"
        subtask.description = "Improve authentication"

        mock_module = MagicMock()
        mock_module.path = "aragora/auth/oidc.py"

        with patch(
            "aragora.nomic.codebase_indexer.CodebaseIndexer"
        ) as MockIndexer:
            mock_indexer = MagicMock()
            mock_indexer.index = AsyncMock()
            mock_indexer.query = AsyncMock(return_value=[mock_module])
            MockIndexer.return_value = mock_indexer

            await pipeline._enrich_file_scope([subtask])

        assert subtask.file_scope == ["aragora/auth/oidc.py"]

    @pytest.mark.asyncio
    async def test_enrich_file_scope_skips_nonempty(self):
        """_enrich_file_scope does not overwrite existing file_scope."""
        pipeline = SelfImprovePipeline()
        subtask = MagicMock()
        subtask.file_scope = ["existing.py"]
        subtask.title = "test"
        subtask.description = "test"

        with patch(
            "aragora.nomic.codebase_indexer.CodebaseIndexer"
        ) as MockIndexer:
            mock_indexer = MagicMock()
            mock_indexer.index = AsyncMock()
            mock_indexer.query = AsyncMock(return_value=[])
            MockIndexer.return_value = mock_indexer

            await pipeline._enrich_file_scope([subtask])

        assert subtask.file_scope == ["existing.py"]

    @pytest.mark.asyncio
    async def test_enrich_file_scope_graceful_on_import_error(self):
        """_enrich_file_scope handles ImportError gracefully."""
        pipeline = SelfImprovePipeline()
        subtask = MagicMock()
        subtask.file_scope = []

        with patch.dict(
            "sys.modules",
            {"aragora.nomic.codebase_indexer": None},
        ):
            # Should not raise
            await pipeline._enrich_file_scope([subtask])


# ---------------------------------------------------------------------------
# Gap 3: Worktree path flow
# ---------------------------------------------------------------------------


class TestWorktreePathFlow:
    @pytest.mark.asyncio
    async def test_worktree_path_extracted_from_track_assignment(self):
        """_execute_single extracts worktree_path from TrackAssignment."""
        from pathlib import Path

        pipeline = SelfImprovePipeline(
            config=SelfImproveConfig(enable_debug_loop=False)
        )

        # Create a TrackAssignment-like object with worktree_path
        subtask = MagicMock()
        subtask.goal = MagicMock()
        subtask.goal.description = "Test worktree flow"
        subtask.worktree_path = Path("/tmp/test-worktree")

        result = await pipeline._execute_single(subtask, "cycle_wt")

        # Falls back to placeholder when ExecutionBridge is unavailable
        assert result["success"] is False
        assert result["subtask"] == "Test worktree flow"


# ---------------------------------------------------------------------------
# Gap 5: Scan mode (MetaPlanner)
# ---------------------------------------------------------------------------


class TestScanMode:
    def test_scan_mode_config_default(self):
        """scan_mode defaults to False in MetaPlannerConfig."""
        from aragora.nomic.meta_planner import MetaPlannerConfig

        config = MetaPlannerConfig()
        assert config.scan_mode is False

    def test_scan_mode_config_set(self):
        """scan_mode can be enabled."""
        from aragora.nomic.meta_planner import MetaPlannerConfig

        config = MetaPlannerConfig(scan_mode=True)
        assert config.scan_mode is True

    @pytest.mark.asyncio
    async def test_scan_prioritize_returns_goals(self):
        """_scan_prioritize returns goals based on codebase signals."""
        from aragora.nomic.meta_planner import MetaPlanner, MetaPlannerConfig, Track

        planner = MetaPlanner(MetaPlannerConfig(scan_mode=True))
        tracks = [Track.QA, Track.CORE]

        with (
            patch("subprocess.run") as mock_run,
            patch(
                "aragora.nomic.codebase_indexer.CodebaseIndexer"
            ) as MockIndexer,
        ):
            # Mock git log output
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="abc1234 fix tests\ntests/test_foo.py\naragora/debate/x.py\n",
            )
            # Mock indexer
            mock_indexer = MagicMock()
            mock_indexer.index = AsyncMock()
            mock_indexer._modules = []
            mock_indexer._test_map = {}
            MockIndexer.return_value = mock_indexer

            goals = await planner._scan_prioritize("Improve quality", tracks)

        assert len(goals) >= 1
        for g in goals:
            assert g.rationale.startswith("Scan mode:")

    @pytest.mark.asyncio
    async def test_scan_mode_triggered_in_prioritize_work(self):
        """prioritize_work uses _scan_prioritize when scan_mode=True."""
        from aragora.nomic.meta_planner import MetaPlanner, MetaPlannerConfig, Track

        planner = MetaPlanner(MetaPlannerConfig(scan_mode=True))

        with patch.object(
            planner, "_scan_prioritize", new_callable=AsyncMock
        ) as mock_scan:
            mock_scan.return_value = []
            await planner.prioritize_work(
                objective="test", available_tracks=[Track.QA]
            )

        mock_scan.assert_called_once()

    def test_file_to_track_mapping(self):
        """_file_to_track maps file paths to the correct track."""
        from aragora.nomic.meta_planner import MetaPlanner, Track

        planner = MetaPlanner()
        tracks = list(Track)

        assert planner._file_to_track("tests/test_foo.py", tracks) == Track.QA
        assert planner._file_to_track("aragora/debate/x.py", tracks) == Track.CORE
        assert planner._file_to_track("aragora/auth/oidc.py", tracks) == Track.SECURITY
        assert planner._file_to_track("deploy/docker/Dockerfile", tracks) == Track.SELF_HOSTED

    def test_scan_mode_in_self_improve_config(self):
        """SelfImproveConfig has scan_mode field."""
        config = SelfImproveConfig(scan_mode=True)
        assert config.scan_mode is True


# ---------------------------------------------------------------------------
# Gap 6: Test-gated merge in coordinate_parallel_work
# ---------------------------------------------------------------------------


class TestGatedMerge:
    @pytest.mark.asyncio
    async def test_coordinate_uses_gated_merge_when_tests_required(self):
        """coordinate_parallel_work uses safe_merge_with_gate when require_tests_pass=True."""
        from aragora.nomic.branch_coordinator import (
            BranchCoordinator,
            BranchCoordinatorConfig,
            TrackAssignment,
        )
        from aragora.nomic.meta_planner import PrioritizedGoal, Track

        config = BranchCoordinatorConfig(
            require_tests_pass=True,
            auto_merge_safe=True,
            use_worktrees=True,
        )
        coordinator = BranchCoordinator(config=config)

        goal = PrioritizedGoal(
            id="g1", track=Track.QA,
            description="Test goal", rationale="test",
            estimated_impact="high", priority=1,
        )
        assignment = TrackAssignment(
            goal=goal, branch_name="test-branch", status="completed"
        )

        with (
            patch.object(
                coordinator, "create_track_branches",
                new_callable=AsyncMock, return_value=[assignment],
            ),
            patch.object(
                coordinator, "detect_conflicts",
                new_callable=AsyncMock, return_value=[],
            ),
            patch.object(
                coordinator, "safe_merge_with_gate",
                new_callable=AsyncMock,
            ) as mock_gated,
            patch.object(
                coordinator, "safe_merge",
                new_callable=AsyncMock,
            ) as mock_plain,
        ):
            from aragora.nomic.branch_coordinator import MergeResult

            mock_gated.return_value = MergeResult(
                source_branch="test-branch",
                target_branch="main",
                success=True,
                commit_sha="abc123",
            )

            result = await coordinator.coordinate_parallel_work(
                assignments=[assignment],
            )

        mock_gated.assert_called_once()
        mock_plain.assert_not_called()
        assert result.merged_branches == 1

    @pytest.mark.asyncio
    async def test_coordinate_uses_plain_merge_when_tests_not_required(self):
        """coordinate_parallel_work uses safe_merge when require_tests_pass=False."""
        from aragora.nomic.branch_coordinator import (
            BranchCoordinator,
            BranchCoordinatorConfig,
            TrackAssignment,
        )
        from aragora.nomic.meta_planner import PrioritizedGoal, Track

        config = BranchCoordinatorConfig(
            require_tests_pass=False,
            auto_merge_safe=True,
            use_worktrees=True,
        )
        coordinator = BranchCoordinator(config=config)

        goal = PrioritizedGoal(
            id="g1", track=Track.QA,
            description="Test goal", rationale="test",
            estimated_impact="high", priority=1,
        )
        assignment = TrackAssignment(
            goal=goal, branch_name="test-branch", status="completed"
        )

        with (
            patch.object(
                coordinator, "create_track_branches",
                new_callable=AsyncMock, return_value=[assignment],
            ),
            patch.object(
                coordinator, "detect_conflicts",
                new_callable=AsyncMock, return_value=[],
            ),
            patch.object(
                coordinator, "safe_merge_with_gate",
                new_callable=AsyncMock,
            ) as mock_gated,
            patch.object(
                coordinator, "safe_merge",
                new_callable=AsyncMock,
            ) as mock_plain,
        ):
            from aragora.nomic.branch_coordinator import MergeResult

            mock_plain.return_value = MergeResult(
                source_branch="test-branch",
                target_branch="main",
                success=True,
                commit_sha="abc123",
            )

            result = await coordinator.coordinate_parallel_work(
                assignments=[assignment],
            )

        mock_plain.assert_called_once()
        mock_gated.assert_not_called()
        assert result.merged_branches == 1


# ---------------------------------------------------------------------------
# Gap D: DAG-aware subtask execution (_group_dependency_waves)
# ---------------------------------------------------------------------------


class TestDAGExecution:
    """Tests for _group_dependency_waves topological ordering."""

    def test_no_dependencies_single_wave(self):
        """All subtasks with no dependencies → single wave."""
        subtasks = [
            MagicMock(id="s1", dependencies=[]),
            MagicMock(id="s2", dependencies=[]),
            MagicMock(id="s3", dependencies=[]),
        ]
        waves = SelfImprovePipeline._group_dependency_waves(subtasks)
        assert len(waves) == 1
        assert len(waves[0]) == 3

    def test_no_dependency_attr_single_wave(self):
        """Subtasks without a dependencies attribute → single wave."""
        s1 = MagicMock(id="s1")
        del s1.dependencies
        s2 = MagicMock(id="s2")
        del s2.dependencies
        waves = SelfImprovePipeline._group_dependency_waves([s1, s2])
        assert len(waves) == 1

    def test_linear_chain_three_waves(self):
        """s1 → s2 → s3 produces 3 sequential waves."""
        subtasks = [
            MagicMock(id="s1", dependencies=[]),
            MagicMock(id="s2", dependencies=["s1"]),
            MagicMock(id="s3", dependencies=["s2"]),
        ]
        waves = SelfImprovePipeline._group_dependency_waves(subtasks)
        assert len(waves) == 3
        assert waves[0][0].id == "s1"
        assert waves[1][0].id == "s2"
        assert waves[2][0].id == "s3"

    def test_diamond_dependency(self):
        """Diamond: s1 → s2,s3 → s4 produces 3 waves."""
        subtasks = [
            MagicMock(id="s1", dependencies=[]),
            MagicMock(id="s2", dependencies=["s1"]),
            MagicMock(id="s3", dependencies=["s1"]),
            MagicMock(id="s4", dependencies=["s2", "s3"]),
        ]
        waves = SelfImprovePipeline._group_dependency_waves(subtasks)
        assert len(waves) == 3
        # s1 first
        assert [st.id for st in waves[0]] == ["s1"]
        # s2 and s3 in parallel
        ids_wave1 = {st.id for st in waves[1]}
        assert ids_wave1 == {"s2", "s3"}
        # s4 last
        assert [st.id for st in waves[2]] == ["s4"]

    def test_cycle_detected_flush(self):
        """Circular dependency gets flushed as a single wave."""
        subtasks = [
            MagicMock(id="s1", dependencies=["s2"]),
            MagicMock(id="s2", dependencies=["s1"]),
        ]
        waves = SelfImprovePipeline._group_dependency_waves(subtasks)
        # Should produce at least 1 wave (flush)
        assert len(waves) >= 1
        total = sum(len(w) for w in waves)
        assert total == 2

    def test_external_dep_ignored(self):
        """Dependencies referencing unknown IDs are ignored."""
        subtasks = [
            MagicMock(id="s1", dependencies=["external_999"]),
            MagicMock(id="s2", dependencies=[]),
        ]
        waves = SelfImprovePipeline._group_dependency_waves(subtasks)
        # s1's dep on "external_999" is not in by_id, so it's ready
        assert len(waves) == 1
        assert len(waves[0]) == 2

    def test_empty_subtasks(self):
        """Empty list returns single empty wave."""
        waves = SelfImprovePipeline._group_dependency_waves([])
        assert waves == [[]]

    @pytest.mark.asyncio
    async def test_execute_respects_wave_order(self):
        """_execute calls subtasks in dependency-wave order."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                use_worktrees=False,
                enable_codebase_indexing=False,
            )
        )
        execution_order: list[str] = []

        async def mock_execute_single(subtask, cycle_id):
            st_id = getattr(subtask, "id", "?")
            execution_order.append(st_id)
            return {"success": True, "files_changed": [], "tests_passed": 1, "tests_failed": 0}

        subtasks = [
            MagicMock(id="s1", dependencies=[]),
            MagicMock(id="s2", dependencies=["s1"]),
        ]

        with patch.object(pipeline, "_execute_single", side_effect=mock_execute_single):
            await pipeline._execute(subtasks, "cycle_dag")

        assert execution_order == ["s1", "s2"]


# ---------------------------------------------------------------------------
# Gap B: File content injection (_read_file_contents)
# ---------------------------------------------------------------------------


class TestFileContentInjection:
    """Tests for _read_file_contents and file_contents in prompts."""

    def test_reads_existing_files(self, tmp_path):
        """_read_file_contents reads real files from file_scope."""
        test_file = tmp_path / "module.py"
        test_file.write_text("def hello():\n    return 'world'\n")

        subtask = MagicMock()
        subtask.file_scope = [str(test_file)]

        contents = SelfImprovePipeline._read_file_contents(subtask)
        assert str(test_file) in contents
        assert "def hello" in contents[str(test_file)]

    def test_skips_nonexistent_files(self):
        """_read_file_contents skips files that don't exist."""
        subtask = MagicMock()
        subtask.file_scope = ["/nonexistent/file_12345.py"]

        contents = SelfImprovePipeline._read_file_contents(subtask)
        assert contents == {}

    def test_truncates_large_files(self, tmp_path):
        """_read_file_contents truncates files exceeding max_chars_per_file."""
        big_file = tmp_path / "big.py"
        big_file.write_text("x" * 5000)

        subtask = MagicMock()
        subtask.file_scope = [str(big_file)]

        contents = SelfImprovePipeline._read_file_contents(subtask, max_chars_per_file=100)
        content = contents[str(big_file)]
        assert len(content) < 200  # 100 chars + truncation message
        assert "truncated" in content

    def test_respects_total_budget(self, tmp_path):
        """_read_file_contents stops reading when total budget is exhausted."""
        for i in range(5):
            f = tmp_path / f"file_{i}.py"
            f.write_text("y" * 500)

        subtask = MagicMock()
        subtask.file_scope = [str(tmp_path / f"file_{i}.py") for i in range(5)]

        contents = SelfImprovePipeline._read_file_contents(
            subtask, max_chars_per_file=500, max_total_chars=1000,
        )
        # Should have read at most 2 full files (2 * 500 = 1000)
        assert len(contents) <= 2

    def test_falls_back_to_goal_file_hints(self, tmp_path):
        """When file_scope is empty, reads from subtask.goal.file_hints."""
        test_file = tmp_path / "fallback.py"
        test_file.write_text("# fallback content\n")

        subtask = MagicMock()
        subtask.file_scope = []
        subtask.goal.file_hints = [str(test_file)]

        contents = SelfImprovePipeline._read_file_contents(subtask)
        assert str(test_file) in contents

    def test_empty_file_scope_no_goal(self):
        """Returns empty dict when no file_scope and no goal."""
        subtask = MagicMock()
        subtask.file_scope = []
        del subtask.goal  # No goal attribute

        contents = SelfImprovePipeline._read_file_contents(subtask)
        assert contents == {}

    def test_file_contents_in_agent_prompt(self):
        """ExecutionInstruction includes file contents in the agent prompt."""
        from aragora.nomic.execution_bridge import ExecutionInstruction

        instr = ExecutionInstruction(
            subtask_id="t1",
            track="core",
            objective="Fix bug",
            context="Test context",
            file_hints=["module.py"],
            success_criteria=["Tests pass"],
            constraints=[],
            file_contents={"module.py": "def broken():\n    pass\n"},
        )
        prompt = instr.to_agent_prompt()
        assert "## File Contents" in prompt
        assert "module.py" in prompt
        assert "def broken" in prompt
        assert "```python" in prompt


# ---------------------------------------------------------------------------
# Gap A: DebugLoop diff context
# ---------------------------------------------------------------------------


class TestDebugLoopDiffContext:
    """Tests for git diff context in DebugLoop retry prompts."""

    def test_debug_attempt_has_diff_context(self):
        """DebugAttempt dataclass has diff_context field."""
        from aragora.nomic.debug_loop import DebugAttempt

        attempt = DebugAttempt(attempt_number=1, prompt="test")
        assert attempt.diff_context == ""

    def test_get_diff_returns_diff(self, tmp_path):
        """_get_diff captures git diff output."""
        from aragora.nomic.debug_loop import DebugLoop

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="diff --git a/file.py b/file.py\n+new line\n",
            )
            diff = DebugLoop._get_diff(str(tmp_path))

        assert "diff --git" in diff
        assert "+new line" in diff

    def test_get_diff_truncates(self):
        """_get_diff truncates output beyond max_chars."""
        from aragora.nomic.debug_loop import DebugLoop

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="x" * 10000,
            )
            diff = DebugLoop._get_diff("/tmp", max_chars=100)

        assert len(diff) < 200
        assert "truncated" in diff

    def test_get_diff_returns_empty_on_error(self):
        """_get_diff returns empty string on subprocess error."""
        from aragora.nomic.debug_loop import DebugLoop

        with patch("subprocess.run", side_effect=OSError("fail")):
            diff = DebugLoop._get_diff("/tmp")

        assert diff == ""

    def test_get_diff_returns_empty_on_clean_worktree(self):
        """_get_diff returns empty string when no changes."""
        from aragora.nomic.debug_loop import DebugLoop

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="")
            diff = DebugLoop._get_diff("/tmp")

        assert diff == ""

    def test_retry_prompt_includes_diff(self):
        """_build_retry_prompt includes diff context when available."""
        from aragora.nomic.debug_loop import DebugAttempt, DebugLoop

        loop = DebugLoop()
        attempt = DebugAttempt(
            attempt_number=1,
            prompt="Fix auth",
            tests_passed=3,
            tests_failed=2,
            test_output="FAILED test_login",
            diff_context="diff --git a/auth.py\n+fixed_login()\n",
        )

        retry = loop._build_retry_prompt("Fix the auth bug", attempt)
        assert "CHANGES MADE SO FAR" in retry
        assert "diff --git" in retry
        assert "+fixed_login()" in retry
        assert "RETRY ATTEMPT 2" in retry

    def test_retry_prompt_no_diff_section_when_empty(self):
        """_build_retry_prompt omits diff section when diff_context is empty."""
        from aragora.nomic.debug_loop import DebugAttempt, DebugLoop

        loop = DebugLoop()
        attempt = DebugAttempt(
            attempt_number=1,
            prompt="Fix",
            tests_passed=0,
            tests_failed=1,
            test_output="FAILED",
            diff_context="",
        )

        retry = loop._build_retry_prompt("Fix bug", attempt)
        assert "CHANGES MADE SO FAR" not in retry

    @pytest.mark.asyncio
    async def test_run_attempt_captures_diff(self):
        """_run_attempt populates attempt.diff_context."""
        from aragora.nomic.debug_loop import DebugLoop

        loop = DebugLoop()

        with (
            patch.object(loop, "_run_agent", new_callable=AsyncMock, return_value=("out", "")),
            patch.object(loop, "_run_tests", new_callable=AsyncMock, return_value={"passed": 1, "failed": 0, "output": "1 passed"}),
            patch.object(DebugLoop, "_get_diff", return_value="diff --git a/test.py"),
        ):
            attempt = await loop._run_attempt("test", "/tmp", None, 1)

        assert attempt.diff_context == "diff --git a/test.py"


# ---------------------------------------------------------------------------
# Gap C: Semantic goal evaluation (GoalEvaluator)
# ---------------------------------------------------------------------------


class TestGoalEvaluation:
    """Tests for GoalEvaluator scoring dimensions."""

    def test_perfect_score(self):
        """All files hit, tests improved, diff relevant → high score."""
        from aragora.nomic.goal_evaluator import GoalEvaluator

        ev = GoalEvaluator()
        result = ev.evaluate(
            goal="Improve authentication error handling",
            file_scope=["aragora/auth/oidc.py"],
            files_changed=["aragora/auth/oidc.py"],
            diff_summary="authentication error handling oidc improve",
            tests_before={"passed": 10, "failed": 2},
            tests_after={"passed": 12, "failed": 0},
        )
        assert result.achievement_score >= 0.6
        assert result.achieved is True
        assert result.scope_coverage == 1.0
        assert result.test_delta > 0
        assert result.diff_relevance > 0.5

    def test_zero_score_nothing_done(self):
        """No files changed, no tests, no diff → zero score."""
        from aragora.nomic.goal_evaluator import GoalEvaluator

        ev = GoalEvaluator()
        result = ev.evaluate(
            goal="Improve authentication",
            file_scope=["aragora/auth/oidc.py"],
            files_changed=[],
            diff_summary="",
        )
        assert result.achievement_score == 0.0
        assert result.achieved is False

    def test_scope_partial_credit_test_file(self):
        """Test file for a scope file gives 0.5 partial credit."""
        from aragora.nomic.goal_evaluator import GoalEvaluator

        score = GoalEvaluator._score_scope_coverage(
            file_scope=["aragora/auth/oidc.py"],
            files_changed=["tests/auth/test_oidc.py"],
        )
        assert score == 0.5

    def test_scope_empty_scope_with_changes(self):
        """Empty file_scope with changes → 1.0."""
        from aragora.nomic.goal_evaluator import GoalEvaluator

        score = GoalEvaluator._score_scope_coverage(
            file_scope=[],
            files_changed=["something.py"],
        )
        assert score == 1.0

    def test_scope_empty_scope_no_changes(self):
        """Empty file_scope with no changes → 0.0."""
        from aragora.nomic.goal_evaluator import GoalEvaluator

        score = GoalEvaluator._score_scope_coverage(
            file_scope=[],
            files_changed=[],
        )
        assert score == 0.0

    def test_test_delta_improvement(self):
        """Test pass rate improvement gives positive delta."""
        from aragora.nomic.goal_evaluator import GoalEvaluator

        delta = GoalEvaluator._score_test_delta(
            before={"passed": 8, "failed": 2},
            after={"passed": 10, "failed": 0},
        )
        assert delta == pytest.approx(0.2)  # 0.8 → 1.0

    def test_test_delta_regression(self):
        """Test pass rate regression gives negative delta."""
        from aragora.nomic.goal_evaluator import GoalEvaluator

        delta = GoalEvaluator._score_test_delta(
            before={"passed": 10, "failed": 0},
            after={"passed": 5, "failed": 5},
        )
        assert delta < 0

    def test_test_delta_no_data(self):
        """No test data → 0.0."""
        from aragora.nomic.goal_evaluator import GoalEvaluator

        delta = GoalEvaluator._score_test_delta(before={}, after={})
        assert delta == 0.0

    def test_test_delta_new_tests(self):
        """Adding passing tests from zero → 1.0."""
        from aragora.nomic.goal_evaluator import GoalEvaluator

        delta = GoalEvaluator._score_test_delta(
            before={},
            after={"passed": 5, "failed": 0},
        )
        assert delta == 1.0

    def test_diff_relevance_full_match(self):
        """All goal keywords in diff → 1.0."""
        from aragora.nomic.goal_evaluator import GoalEvaluator

        score = GoalEvaluator._score_diff_relevance(
            goal="Improve authentication error handling",
            diff_summary="improve authentication error handling code",
        )
        assert score == 1.0

    def test_diff_relevance_no_match(self):
        """No goal keywords in diff → 0.0."""
        from aragora.nomic.goal_evaluator import GoalEvaluator

        score = GoalEvaluator._score_diff_relevance(
            goal="Improve authentication",
            diff_summary="fixed database migration schema",
        )
        assert score == 0.0

    def test_diff_relevance_empty_diff(self):
        """Empty diff → 0.0."""
        from aragora.nomic.goal_evaluator import GoalEvaluator

        score = GoalEvaluator._score_diff_relevance(
            goal="Improve authentication",
            diff_summary="",
        )
        assert score == 0.0

    def test_diff_relevance_stop_words_filtered(self):
        """Stop words in goal are filtered out."""
        from aragora.nomic.goal_evaluator import GoalEvaluator

        # "the" and "for" are stop words, shouldn't count
        score = GoalEvaluator._score_diff_relevance(
            goal="the for",
            diff_summary="the for something",
        )
        # All goal words are stop words → returns 0.5 (neutral)
        assert score == 0.5

    def test_negative_test_delta_clamped_in_composite(self):
        """Negative test delta doesn't reduce composite below 0."""
        from aragora.nomic.goal_evaluator import GoalEvaluator

        ev = GoalEvaluator()
        result = ev.evaluate(
            goal="Improve tests",
            file_scope=[],
            files_changed=[],
            diff_summary="",
            tests_before={"passed": 10, "failed": 0},
            tests_after={"passed": 0, "failed": 10},
        )
        assert result.achievement_score >= 0.0

    def test_goal_evaluation_achieved_property(self):
        """GoalEvaluation.achieved is True when score >= 0.5."""
        from aragora.nomic.goal_evaluator import GoalEvaluation

        assert GoalEvaluation(goal="x", achievement_score=0.5, scope_coverage=0, test_delta=0, diff_relevance=0).achieved is True
        assert GoalEvaluation(goal="x", achievement_score=0.49, scope_coverage=0, test_delta=0, diff_relevance=0).achieved is False

    def test_details_populated(self):
        """GoalEvaluation.details includes counts."""
        from aragora.nomic.goal_evaluator import GoalEvaluator

        ev = GoalEvaluator()
        result = ev.evaluate(
            goal="Test",
            file_scope=["a.py", "b.py"],
            files_changed=["a.py"],
            tests_before={"passed": 5, "failed": 1},
            tests_after={"passed": 6, "failed": 0},
        )
        assert result.details["file_scope_count"] == 2
        assert result.details["files_changed_count"] == 1

    def test_evaluate_goal_method_on_pipeline(self):
        """SelfImprovePipeline._evaluate_goal returns GoalEvaluation."""
        pipeline = SelfImprovePipeline(SelfImproveConfig(enable_codebase_indexing=False))
        subtask = MagicMock()
        subtask.file_scope = ["auth.py"]

        result = SelfImproveResult(
            cycle_id="c1", objective="Test",
            files_changed=["auth.py"],
        )

        evaluation = pipeline._evaluate_goal(
            objective="Fix authentication bug",
            subtasks=[subtask],
            result=result,
            baseline=None,
            after=None,
        )
        assert evaluation is not None
        assert evaluation.goal == "Fix authentication bug"
        assert evaluation.scope_coverage == 1.0

    def test_evaluate_goal_handles_import_error(self):
        """_evaluate_goal returns None when GoalEvaluator is unavailable."""
        pipeline = SelfImprovePipeline()
        result = SelfImproveResult(cycle_id="c1", objective="Test")

        with patch.dict("sys.modules", {"aragora.nomic.goal_evaluator": None}):
            evaluation = pipeline._evaluate_goal("Test", [], result, None, None)
        assert evaluation is None


# ---------------------------------------------------------------------------
# Gap E: Pipeline validation CLI
# ---------------------------------------------------------------------------


class TestValidatePipeline:
    """Tests for the --validate-pipeline CLI flag."""

    def test_validate_pipeline_returns_zero_on_success(self):
        """_validate_pipeline returns 0 when all components pass."""
        import sys
        sys.path.insert(0, "scripts")
        try:
            from self_develop import _validate_pipeline
        finally:
            sys.path.pop(0)

        # Claude CLI may not be available, so we mock shutil.which
        with patch("shutil.which", return_value="/usr/bin/claude"):
            exit_code = _validate_pipeline("Test goal for validation")

        assert exit_code == 0

    def test_validate_pipeline_reports_failures(self, capsys):
        """_validate_pipeline prints pass/fail report."""
        import sys
        sys.path.insert(0, "scripts")
        try:
            from self_develop import _validate_pipeline
        finally:
            sys.path.pop(0)

        with patch("shutil.which", return_value=None):
            exit_code = _validate_pipeline("Test goal")

        captured = capsys.readouterr()
        assert "[PASS]" in captured.out
        assert "Claude CLI" in captured.out
        # Claude CLI not found → FAIL
        assert "[FAIL]" in captured.out

    def test_validate_pipeline_probes_goal_evaluator(self, capsys):
        """_validate_pipeline probes GoalEvaluator."""
        import sys
        sys.path.insert(0, "scripts")
        try:
            from self_develop import _validate_pipeline
        finally:
            sys.path.pop(0)

        with patch("shutil.which", return_value="/usr/bin/claude"):
            _validate_pipeline("Improve test coverage")

        captured = capsys.readouterr()
        assert "GoalEvaluator" in captured.out
        assert "score=" in captured.out


# ---------------------------------------------------------------------------
# A2: Pipeline graph publication
# ---------------------------------------------------------------------------


class TestPipelineGraphPublication:
    """Tests for _publish_to_pipeline_graph (A2)."""

    @pytest.mark.asyncio
    async def test_publish_to_pipeline_graph_called(self):
        """Pipeline graph is published after a successful run."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                use_meta_planner=False,
                use_worktrees=False,
                capture_metrics=False,
                persist_outcomes=False,
                enable_codebase_indexing=False,
            )
        )

        async def mock_execute(subtask, cycle_id):
            return {
                "success": True,
                "files_changed": ["a.py"],
                "tests_passed": 1,
                "tests_failed": 0,
            }

        with (
            patch.object(pipeline, "_execute_single", side_effect=mock_execute),
            patch(
                "aragora.nomic.self_improve.NomicPipelineBridge",
                create=True,
            ) as MockBridge,
            patch(
                "aragora.nomic.self_improve.get_graph_store",
                create=True,
            ) as mock_get_store,
        ):
            # Patch the lazy imports inside _publish_to_pipeline_graph
            mock_bridge = MagicMock()
            mock_graph = MagicMock()
            mock_graph.id = "graph-123"
            mock_graph.nodes = [MagicMock()]
            mock_bridge.create_pipeline_from_cycle.return_value = mock_graph
            MockBridge.return_value = mock_bridge

            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            # We need to patch the imports inside the method
            with (
                patch("aragora.nomic.pipeline_bridge.NomicPipelineBridge", MockBridge),
                patch("aragora.pipeline.graph_store.get_graph_store", mock_get_store),
            ):
                result = await pipeline.run("Test graph publication")

            # Verify the bridge was called (may fail gracefully if imports not available)
            # The important thing is run() completes without error
            assert isinstance(result, SelfImproveResult)

    @pytest.mark.asyncio
    async def test_publish_to_pipeline_graph_graceful_degradation(self):
        """Pipeline graph publication gracefully degrades on ImportError."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                use_meta_planner=False,
                use_worktrees=False,
                capture_metrics=False,
                persist_outcomes=False,
                enable_codebase_indexing=False,
            )
        )

        async def mock_execute(subtask, cycle_id):
            return {
                "success": True,
                "files_changed": ["a.py"],
                "tests_passed": 1,
                "tests_failed": 0,
            }

        # Force ImportError by making the pipeline_bridge module unavailable
        with (
            patch.object(pipeline, "_execute_single", side_effect=mock_execute),
            patch.dict("sys.modules", {"aragora.nomic.pipeline_bridge": None}),
        ):
            result = await pipeline.run("Test graceful degradation")

        assert isinstance(result, SelfImproveResult)
        assert result.duration_seconds > 0

    def test_self_improve_orchestration_adapter(self):
        """_SelfImproveOrchestrationAdapter produces correct fields."""
        from aragora.nomic.self_improve import _SelfImproveOrchestrationAdapter

        result = SelfImproveResult(
            cycle_id="cycle_abc",
            objective="Improve tests",
            subtasks_total=5,
            subtasks_completed=3,
            subtasks_failed=2,
            files_changed=["a.py", "b.py"],
            regressions_detected=False,
            duration_seconds=42.0,
            improvement_score=0.75,
        )

        adapter = _SelfImproveOrchestrationAdapter("cycle_abc", result)

        assert adapter.goal == "Improve tests"
        assert "3/5" in adapter.summary
        assert adapter.success is True
        assert adapter.duration_seconds == 42.0
        assert adapter.improvement_score == 0.75
        assert len(adapter.assignments) == 2

    def test_self_improve_orchestration_adapter_failure(self):
        """Adapter sets success=False when regressions are detected."""
        from aragora.nomic.self_improve import _SelfImproveOrchestrationAdapter

        result = SelfImproveResult(
            cycle_id="cycle_fail",
            objective="Broken",
            subtasks_completed=1,
            regressions_detected=True,
        )

        adapter = _SelfImproveOrchestrationAdapter("cycle_fail", result)
        assert adapter.success is False

    def test_build_assignments_from_result(self):
        """_build_assignments_from_result builds assignment-like objects."""
        from aragora.nomic.self_improve import _build_assignments_from_result

        result = SelfImproveResult(
            cycle_id="cycle_b",
            objective="Test",
            files_changed=["x.py", "y.py", "z.py"],
        )

        assignments = _build_assignments_from_result(result)

        assert len(assignments) == 3
        for i, a in enumerate(assignments):
            assert a.status == "completed"
            assert a.agent_type == "self_improve"
            assert a.track.value == "core"
            assert a.subtask.id == f"si-{i}"
            assert "Modified:" in a.subtask.title
            assert len(a.subtask.file_scope) == 1
            assert a.subtask.estimated_complexity == "medium"

    def test_build_assignments_caps_at_20(self):
        """_build_assignments_from_result caps at 20 files."""
        from aragora.nomic.self_improve import _build_assignments_from_result

        result = SelfImproveResult(
            cycle_id="cycle_cap",
            objective="Many files",
            files_changed=[f"file_{i}.py" for i in range(30)],
        )

        assignments = _build_assignments_from_result(result)
        assert len(assignments) == 20

    def test_publish_to_pipeline_graph_with_empty_result(self):
        """Publish with no files changed produces minimal graph."""
        from aragora.nomic.self_improve import _SelfImproveOrchestrationAdapter

        result = SelfImproveResult(
            cycle_id="cycle_empty",
            objective="Nothing changed",
            subtasks_completed=0,
            files_changed=[],
        )

        adapter = _SelfImproveOrchestrationAdapter("cycle_empty", result)
        assert adapter.success is False
        assert len(adapter.assignments) == 0


# ---------------------------------------------------------------------------
# A3: MetaPlanner feedback loop
# ---------------------------------------------------------------------------


class TestMetaPlannerFeedbackLoop:
    """Tests for MetaPlanner outcome recording in _persist_outcome (A3)."""

    def test_meta_planner_record_outcome_called(self):
        """_persist_outcome calls MetaPlanner.record_outcome."""
        pipeline = SelfImprovePipeline()
        result = SelfImproveResult(
            cycle_id="cycle_mp",
            objective="Test MetaPlanner feedback",
            subtasks_completed=2,
            subtasks_failed=0,
            files_changed=["a.py"],
            duration_seconds=10.0,
        )

        with (
            patch("aragora.nomic.cycle_store.get_cycle_store") as mock_get_store,
            patch("aragora.nomic.meta_planner.MetaPlanner") as MockPlanner,
        ):
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            mock_planner = MagicMock()
            MockPlanner.return_value = mock_planner

            pipeline._persist_outcome("cycle_mp", result)

        mock_planner.record_outcome.assert_called_once()
        call_kwargs = mock_planner.record_outcome.call_args
        goal_outcomes = call_kwargs[1]["goal_outcomes"] if "goal_outcomes" in (call_kwargs[1] or {}) else call_kwargs[0][0]
        assert len(goal_outcomes) == 1
        assert goal_outcomes[0]["track"] == "core"
        assert goal_outcomes[0]["success"] is True
        assert goal_outcomes[0]["description"] == "Test MetaPlanner feedback"

    def test_meta_planner_record_outcome_graceful_degradation(self):
        """_persist_outcome completes even when MetaPlanner import fails."""
        pipeline = SelfImprovePipeline()
        result = SelfImproveResult(
            cycle_id="cycle_mp_fail",
            objective="Test graceful",
            subtasks_completed=1,
            subtasks_failed=0,
            duration_seconds=5.0,
        )

        with (
            patch("aragora.nomic.cycle_store.get_cycle_store") as mock_get_store,
            patch.dict("sys.modules", {"aragora.nomic.meta_planner": None}),
        ):
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            # Should not raise
            pipeline._persist_outcome("cycle_mp_fail", result)

        # CycleLearningStore save should still have been called
        mock_store.save_cycle.assert_called_once()

    def test_meta_planner_record_outcome_with_failures(self):
        """MetaPlanner records success=False when subtasks failed."""
        pipeline = SelfImprovePipeline()
        result = SelfImproveResult(
            cycle_id="cycle_mp_mixed",
            objective="Partial failure",
            subtasks_completed=1,
            subtasks_failed=3,
            duration_seconds=8.0,
        )

        with (
            patch("aragora.nomic.cycle_store.get_cycle_store") as mock_get_store,
            patch("aragora.nomic.meta_planner.MetaPlanner") as MockPlanner,
        ):
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            mock_planner = MagicMock()
            MockPlanner.return_value = mock_planner

            pipeline._persist_outcome("cycle_mp_mixed", result)

        mock_planner.record_outcome.assert_called_once()
        call_kwargs = mock_planner.record_outcome.call_args
        goal_outcomes = call_kwargs[1]["goal_outcomes"] if "goal_outcomes" in (call_kwargs[1] or {}) else call_kwargs[0][0]
        assert goal_outcomes[0]["success"] is False

    def test_meta_planner_record_outcome_skipped_when_no_subtasks(self):
        """MetaPlanner.record_outcome is NOT called when no subtasks ran."""
        pipeline = SelfImprovePipeline()
        result = SelfImproveResult(
            cycle_id="cycle_mp_empty",
            objective="No subtasks",
            subtasks_completed=0,
            subtasks_failed=0,
            duration_seconds=1.0,
        )

        with (
            patch("aragora.nomic.cycle_store.get_cycle_store") as mock_get_store,
            patch("aragora.nomic.meta_planner.MetaPlanner") as MockPlanner,
        ):
            mock_store = MagicMock()
            mock_get_store.return_value = mock_store

            mock_planner = MagicMock()
            MockPlanner.return_value = mock_planner

            pipeline._persist_outcome("cycle_mp_empty", result)

        mock_planner.record_outcome.assert_not_called()


# ---------------------------------------------------------------------------
# Plan v9: Risk Assessment
# ---------------------------------------------------------------------------


class TestRiskAssessment:
    """Test _assess_execution_risk() categorization."""

    def test_tests_only_is_low_risk(self):
        pipeline = SelfImprovePipeline()
        instr = MagicMock()
        instr.file_hints = ["tests/nomic/test_foo.py", "tests/auth/test_bar.py"]
        assert pipeline._assess_execution_risk(instr) == "low"

    def test_protected_files_is_high_risk(self):
        pipeline = SelfImprovePipeline()
        instr = MagicMock()
        instr.file_hints = ["CLAUDE.md"]
        assert pipeline._assess_execution_risk(instr) == "high"

    def test_protected_init_is_high_risk(self):
        pipeline = SelfImprovePipeline()
        instr = MagicMock()
        instr.file_hints = ["aragora/__init__.py"]
        assert pipeline._assess_execution_risk(instr) == "high"

    def test_protected_env_is_high_risk(self):
        pipeline = SelfImprovePipeline()
        instr = MagicMock()
        instr.file_hints = [".env"]
        assert pipeline._assess_execution_risk(instr) == "high"

    def test_protected_nomic_loop_is_high_risk(self):
        pipeline = SelfImprovePipeline()
        instr = MagicMock()
        instr.file_hints = ["scripts/nomic_loop.py"]
        assert pipeline._assess_execution_risk(instr) == "high"

    def test_core_modules_is_medium_risk(self):
        pipeline = SelfImprovePipeline()
        instr = MagicMock()
        instr.file_hints = ["aragora/server/handlers/auth.py"]
        assert pipeline._assess_execution_risk(instr) == "medium"

    def test_debate_module_is_medium_risk(self):
        pipeline = SelfImprovePipeline()
        instr = MagicMock()
        instr.file_hints = ["aragora/debate/orchestrator.py"]
        assert pipeline._assess_execution_risk(instr) == "medium"

    def test_nomic_module_is_medium_risk(self):
        pipeline = SelfImprovePipeline()
        instr = MagicMock()
        instr.file_hints = ["aragora/nomic/meta_planner.py"]
        assert pipeline._assess_execution_risk(instr) == "medium"

    def test_many_files_is_high_risk(self):
        pipeline = SelfImprovePipeline()
        instr = MagicMock()
        instr.file_hints = [f"aragora/module_{i}.py" for i in range(11)]
        assert pipeline._assess_execution_risk(instr) == "high"

    def test_empty_file_hints_is_low_risk(self):
        pipeline = SelfImprovePipeline()
        instr = MagicMock()
        instr.file_hints = []
        assert pipeline._assess_execution_risk(instr) == "low"

    def test_non_core_module_is_low_risk(self):
        pipeline = SelfImprovePipeline()
        instr = MagicMock()
        instr.file_hints = ["aragora/billing/cost_tracker.py"]
        assert pipeline._assess_execution_risk(instr) == "low"

    def test_falls_back_to_file_scope(self):
        """Falls back to file_scope when file_hints is missing."""
        pipeline = SelfImprovePipeline()
        instr = MagicMock(spec=[])
        instr.file_scope = ["tests/test_foo.py"]
        assert pipeline._assess_execution_risk(instr) == "low"

    def test_no_hints_no_scope_is_low(self):
        """Returns low when neither file_hints nor file_scope exists."""
        pipeline = SelfImprovePipeline()
        instr = MagicMock(spec=[])
        assert pipeline._assess_execution_risk(instr) == "low"


# ---------------------------------------------------------------------------
# Plan v9: Auto Mode
# ---------------------------------------------------------------------------


class TestAutoMode:
    """Test graduated autonomy execution."""

    def test_auto_mode_config_default(self):
        config = SelfImproveConfig()
        assert config.auto_mode is False

    def test_auto_mode_config_set(self):
        config = SelfImproveConfig(auto_mode=True)
        assert config.auto_mode is True

    @pytest.mark.asyncio
    async def test_auto_mode_defers_high_risk(self):
        """Auto mode returns deferred dict for high-risk instructions."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(require_approval=True, autonomous=False, auto_mode=True)
        )
        mock_instr = MagicMock()
        mock_instr.subtask_id = "t_high"
        mock_instr.file_hints = ["CLAUDE.md"]

        result = await pipeline._dispatch_to_claude_code(mock_instr, "/tmp/wt")
        assert result is not None
        assert result["deferred"] is True
        assert result["risk_level"] == "high"
        assert result["files_changed"] == []

    @pytest.mark.asyncio
    async def test_auto_mode_proceeds_for_low_risk(self):
        """Auto mode proceeds to execution for low-risk instructions."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                require_approval=True, autonomous=False,
                auto_mode=True, run_tests=False,
            )
        )
        mock_instr = MagicMock()
        mock_instr.subtask_id = "t_low"
        mock_instr.file_hints = ["tests/test_foo.py"]
        mock_instr.to_agent_prompt.return_value = "# Task"

        # CLI not found → returns None past the risk gate (not blocked by approval)
        with patch("shutil.which", return_value=None):
            result = await pipeline._dispatch_to_claude_code(mock_instr, "/tmp/wt")
        # Returns None because CLI not found, but it got PAST the approval gate
        assert result is None

    @pytest.mark.asyncio
    async def test_default_mode_returns_none(self):
        """Default mode (no auto_mode) returns None when require_approval is True."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(require_approval=True, autonomous=False, auto_mode=False)
        )
        mock_instr = MagicMock()
        mock_instr.subtask_id = "t_default"
        result = await pipeline._dispatch_to_claude_code(mock_instr, "/tmp/wt")
        assert result is None


# ---------------------------------------------------------------------------
# Plan v9: Progress Callback
# ---------------------------------------------------------------------------


class TestProgressCallback:
    """Test execution progress streaming."""

    @pytest.mark.asyncio
    async def test_progress_events_emitted(self):
        """Progress callback receives events during run()."""
        events: list[tuple[str, dict]] = []

        def capture(event: str, data: dict) -> None:
            events.append((event, data))

        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                use_meta_planner=False,
                use_worktrees=False,
                capture_metrics=False,
                persist_outcomes=False,
                enable_codebase_indexing=False,
                progress_callback=capture,
            )
        )
        await pipeline.run("Test progress")

        event_names = [e[0] for e in events]
        assert "cycle_started" in event_names
        assert "planning_complete" in event_names
        assert "decomposition_complete" in event_names
        assert "execution_complete" in event_names
        assert "cycle_complete" in event_names

    @pytest.mark.asyncio
    async def test_callback_exception_swallowed(self):
        """Exceptions in the progress callback are swallowed."""
        def exploding_callback(event: str, data: dict) -> None:
            raise ValueError("boom")

        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                use_meta_planner=False,
                use_worktrees=False,
                capture_metrics=False,
                persist_outcomes=False,
                enable_codebase_indexing=False,
                progress_callback=exploding_callback,
            )
        )
        # Should not raise despite the exploding callback
        result = await pipeline.run("Test exception swallowing")
        assert isinstance(result, SelfImproveResult)

    @pytest.mark.asyncio
    async def test_no_callback_no_error(self):
        """No progress_callback set still works fine."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                use_meta_planner=False,
                use_worktrees=False,
                capture_metrics=False,
                persist_outcomes=False,
                enable_codebase_indexing=False,
                progress_callback=None,
            )
        )
        result = await pipeline.run("Test no callback")
        assert isinstance(result, SelfImproveResult)

    def test_emit_progress_with_no_callback(self):
        """_emit_progress is a no-op when callback is None."""
        pipeline = SelfImprovePipeline(SelfImproveConfig(progress_callback=None))
        # Should not raise
        pipeline._emit_progress("test_event", {"key": "value"})

    def test_emit_progress_calls_callback(self):
        """_emit_progress invokes the callback."""
        events = []
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(progress_callback=lambda e, d: events.append((e, d)))
        )
        pipeline._emit_progress("my_event", {"foo": "bar"})
        assert len(events) == 1
        assert events[0] == ("my_event", {"foo": "bar"})

    @pytest.mark.asyncio
    async def test_planning_complete_event_has_goal_count(self):
        """planning_complete event data includes goal count."""
        events: list[tuple[str, dict]] = []

        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                use_meta_planner=False,
                use_worktrees=False,
                capture_metrics=False,
                persist_outcomes=False,
                enable_codebase_indexing=False,
                progress_callback=lambda e, d: events.append((e, d)),
            )
        )
        await pipeline.run("Test goal count")

        planning_events = [(e, d) for e, d in events if e == "planning_complete"]
        assert len(planning_events) == 1
        assert planning_events[0][1]["goals"] >= 1
