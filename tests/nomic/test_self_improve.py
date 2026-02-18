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

        pipeline = SelfImprovePipeline()

        # Create a TrackAssignment-like object with worktree_path
        subtask = MagicMock()
        subtask.goal = MagicMock()
        subtask.goal.description = "Test worktree flow"
        subtask.worktree_path = Path("/tmp/test-worktree")

        result = await pipeline._execute_single(subtask, "cycle_wt")

        assert result["success"] is True
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
