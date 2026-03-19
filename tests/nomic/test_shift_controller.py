"""Tests for aragora.nomic.shift_controller — ShiftController (lightweight)."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.shift_controller import (
    ShiftCheckpoint,
    ShiftConfig,
    ShiftController,
    ShiftResult,
)


# ---------------------------------------------------------------------------
# TestShiftConfig
# ---------------------------------------------------------------------------


class TestShiftConfig:
    """Verify ShiftConfig defaults and overrides."""

    def test_defaults(self):
        cfg = ShiftConfig()
        assert cfg.max_duration_seconds == 28800.0
        assert cfg.max_items_per_shift == 10
        assert cfg.checkpoint_interval_seconds == 900.0
        assert cfg.budget_limit_usd == 50.0
        assert cfg.require_assessment_before_start is True
        assert cfg.require_assessment_after_shift is True
        assert cfg.stop_on_regression is True
        assert cfg.stop_on_dirty_worktree is True
        assert cfg.stop_on_budget_exceeded is True

    def test_custom_values(self):
        cfg = ShiftConfig(
            max_duration_seconds=3600.0,
            max_items_per_shift=5,
            budget_limit_usd=10.0,
            stop_on_regression=False,
        )
        assert cfg.max_duration_seconds == 3600.0
        assert cfg.max_items_per_shift == 5
        assert cfg.budget_limit_usd == 10.0
        assert cfg.stop_on_regression is False


# ---------------------------------------------------------------------------
# TestShiftCheckpoint
# ---------------------------------------------------------------------------


class TestShiftCheckpoint:
    """Verify ShiftCheckpoint auto-generates ID and timestamp."""

    def test_auto_id_and_timestamp(self):
        cp = ShiftCheckpoint()
        assert cp.checkpoint_id.startswith("chk-")
        assert len(cp.checkpoint_id) == 12  # "chk-" + 8 hex chars
        assert cp.timestamp > 0

    def test_preserves_explicit_values(self):
        cp = ShiftCheckpoint(checkpoint_id="chk-custom", timestamp=1234.0)
        assert cp.checkpoint_id == "chk-custom"
        assert cp.timestamp == 1234.0

    def test_default_fields(self):
        cp = ShiftCheckpoint()
        assert cp.shift_id == ""
        assert cp.phase == ""
        assert cp.items_attempted == 0
        assert cp.items_landed == 0
        assert cp.items_failed == 0
        assert cp.items_reverted == 0
        assert cp.wall_clock_seconds == 0.0
        assert cp.budget_spent_usd == 0.0
        assert cp.health_score == 0.0
        assert cp.stop_reason is None
        assert cp.details == {}


# ---------------------------------------------------------------------------
# Helpers for mocking
# ---------------------------------------------------------------------------


def _mock_assessment(
    health_score: float = 0.85,
    candidates: list | None = None,
    assessment_id: str = "ca-test123",
):
    """Return a mock CanonicalRepoAssessment."""
    mock = MagicMock()
    mock.health_report = {"health_score": health_score}
    mock.improvement_candidates = candidates or []
    mock.assessment_id = assessment_id
    return mock


def _mock_orch_result(
    success: bool = True,
    failed_subtasks: int = 0,
    summary: str = "done",
    total_cost_usd: float = 1.0,
    reverted: bool = False,
):
    """Return a mock OrchestrationResult."""
    mock = MagicMock()
    mock.success = success
    mock.failed_subtasks = failed_subtasks
    mock.summary = summary
    mock.total_cost_usd = total_cost_usd
    mock.reverted = reverted
    return mock


# ---------------------------------------------------------------------------
# TestShiftController
# ---------------------------------------------------------------------------


class TestShiftController:
    """Tests for the lightweight ShiftController."""

    @pytest.mark.asyncio
    async def test_run_shift_no_work(self):
        """Assessment returns empty candidates -> stops with 'no_work'."""
        controller = ShiftController(
            repo_path=Path("/tmp/test-repo"),
            config=ShiftConfig(stop_on_dirty_worktree=False),
        )

        mock_compiled = _mock_assessment(candidates=[])
        mock_compiler = MagicMock()
        mock_compiler.compile = AsyncMock(return_value=mock_compiled)

        with patch(
            "aragora.nomic.shift_controller.CanonicalAssessmentCompiler",
            return_value=mock_compiler,
        ):
            result = await controller.run_shift()

        assert result.stop_reason == "no_work"
        assert result.items_attempted == 0
        assert result.items_landed == 0
        assert isinstance(result.shift_id, str)
        assert result.shift_id.startswith("shift-")

    @pytest.mark.asyncio
    async def test_run_shift_budget_exceeded(self):
        """Budget tracking stops shift when limit reached."""
        config = ShiftConfig(
            budget_limit_usd=2.0,
            max_items_per_shift=5,
            stop_on_dirty_worktree=False,
            require_assessment_after_shift=False,
        )
        controller = ShiftController(repo_path=Path("/tmp/test-repo"), config=config)

        candidates = [{"description": f"Goal {i}"} for i in range(5)]
        mock_compiled = _mock_assessment(candidates=candidates)
        mock_compiler = MagicMock()
        mock_compiler.compile = AsyncMock(return_value=mock_compiled)

        # First goal costs 2.5, which exceeds budget of 2.0
        mock_result = _mock_orch_result(success=True, total_cost_usd=2.5)

        with (
            patch(
                "aragora.nomic.shift_controller.CanonicalAssessmentCompiler",
                return_value=mock_compiler,
            ),
            patch("aragora.nomic.hardened_orchestrator.HardenedOrchestrator") as mock_orch_cls,
        ):
            mock_orch_cls.return_value.execute_goal_coordinated = AsyncMock(
                return_value=mock_result
            )
            result = await controller.run_shift()

        # After first goal, budget_spent=2.5 >= 2.0, so second goal triggers stop
        assert result.items_attempted >= 1
        assert result.budget_spent_usd >= 2.0
        assert result.stop_reason == "budget_exceeded"

    @pytest.mark.asyncio
    async def test_run_shift_time_exceeded(self):
        """Time limit stops shift."""
        config = ShiftConfig(
            max_duration_seconds=0.0,  # Immediately expired
            max_items_per_shift=5,
            stop_on_dirty_worktree=False,
            require_assessment_after_shift=False,
        )
        controller = ShiftController(repo_path=Path("/tmp/test-repo"), config=config)

        candidates = [{"description": "Goal 1"}]
        mock_compiled = _mock_assessment(candidates=candidates)
        mock_compiler = MagicMock()
        mock_compiler.compile = AsyncMock(return_value=mock_compiled)

        with patch(
            "aragora.nomic.shift_controller.CanonicalAssessmentCompiler",
            return_value=mock_compiler,
        ):
            result = await controller.run_shift()

        assert result.stop_reason == "time_exceeded"
        assert result.items_attempted == 0

    @pytest.mark.asyncio
    async def test_run_shift_successful_execution(self):
        """Goals execute and land successfully."""
        config = ShiftConfig(
            max_items_per_shift=3,
            stop_on_dirty_worktree=False,
            require_assessment_after_shift=False,
        )
        controller = ShiftController(repo_path=Path("/tmp/test-repo"), config=config)

        candidates = [
            {"description": "Fix bug A"},
            {"description": "Fix bug B"},
        ]
        mock_compiled = _mock_assessment(candidates=candidates)
        mock_compiler = MagicMock()
        mock_compiler.compile = AsyncMock(return_value=mock_compiled)

        mock_result = _mock_orch_result(success=True, total_cost_usd=0.5)

        with (
            patch(
                "aragora.nomic.shift_controller.CanonicalAssessmentCompiler",
                return_value=mock_compiler,
            ),
            patch("aragora.nomic.hardened_orchestrator.HardenedOrchestrator") as mock_orch_cls,
        ):
            mock_orch_cls.return_value.execute_goal_coordinated = AsyncMock(
                return_value=mock_result
            )
            result = await controller.run_shift()

        assert result.items_attempted == 2
        assert result.items_landed == 2
        assert result.items_failed == 0
        assert result.budget_spent_usd == 1.0  # 2 * 0.5
        assert result.goals_executed == ["Fix bug A", "Fix bug B"]
        assert result.stop_reason == "completed"

    @pytest.mark.asyncio
    async def test_run_shift_regression_stops(self):
        """Regression detection stops shift."""
        config = ShiftConfig(
            stop_on_regression=True,
            stop_on_dirty_worktree=False,
            require_assessment_after_shift=False,
        )
        controller = ShiftController(repo_path=Path("/tmp/test-repo"), config=config)

        candidates = [
            {"description": "Goal 1"},
            {"description": "Goal 2"},
        ]
        mock_compiled = _mock_assessment(candidates=candidates)
        mock_compiler = MagicMock()
        mock_compiler.compile = AsyncMock(return_value=mock_compiled)

        # First goal succeeds, second has regression
        mock_ok = _mock_orch_result(success=True)
        mock_regression = _mock_orch_result(success=False, failed_subtasks=1)

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_ok if call_count == 1 else mock_regression

        with (
            patch(
                "aragora.nomic.shift_controller.CanonicalAssessmentCompiler",
                return_value=mock_compiler,
            ),
            patch("aragora.nomic.hardened_orchestrator.HardenedOrchestrator") as mock_orch_cls,
        ):
            mock_orch_cls.return_value.execute_goal_coordinated = AsyncMock(side_effect=side_effect)
            result = await controller.run_shift()

        assert result.stop_reason == "regression"
        assert result.items_attempted == 2
        assert result.items_landed == 1
        assert result.items_failed == 1

    @pytest.mark.asyncio
    async def test_run_shift_assessment_failure_graceful(self):
        """Assessment fails -> proceeds with empty -> no_work."""
        config = ShiftConfig(stop_on_dirty_worktree=False)
        controller = ShiftController(repo_path=Path("/tmp/test-repo"), config=config)

        mock_compiler = MagicMock()
        mock_compiler.compile = AsyncMock(side_effect=RuntimeError("compile boom"))

        with patch(
            "aragora.nomic.shift_controller.CanonicalAssessmentCompiler",
            return_value=mock_compiler,
        ):
            result = await controller.run_shift()

        # Assessment fails gracefully, returns empty candidates -> no_work
        assert result.stop_reason == "no_work"
        assert result.health_score_before == 0.0

    @pytest.mark.asyncio
    async def test_manual_stop(self):
        """Calling stop() during shift causes graceful termination."""
        config = ShiftConfig(
            max_items_per_shift=10,
            stop_on_dirty_worktree=False,
            require_assessment_after_shift=False,
        )
        controller = ShiftController(repo_path=Path("/tmp/test-repo"), config=config)

        candidates = [{"description": f"Goal {i}"} for i in range(10)]
        mock_compiled = _mock_assessment(candidates=candidates)
        mock_compiler = MagicMock()
        mock_compiler.compile = AsyncMock(return_value=mock_compiled)

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                controller.stop()
            return _mock_orch_result(success=True, total_cost_usd=0.1)

        with (
            patch(
                "aragora.nomic.shift_controller.CanonicalAssessmentCompiler",
                return_value=mock_compiler,
            ),
            patch("aragora.nomic.hardened_orchestrator.HardenedOrchestrator") as mock_orch_cls,
        ):
            mock_orch_cls.return_value.execute_goal_coordinated = AsyncMock(side_effect=side_effect)
            result = await controller.run_shift()

        assert result.stop_reason == "manual_stop"
        assert not controller.is_running

    @pytest.mark.asyncio
    async def test_checkpoints_emitted(self):
        """Verify checkpoint sequence: assessment -> planning -> executing -> completed."""
        config = ShiftConfig(
            max_items_per_shift=1,
            stop_on_dirty_worktree=False,
        )
        controller = ShiftController(repo_path=Path("/tmp/test-repo"), config=config)

        candidates = [{"description": "Single goal"}]
        mock_compiled = _mock_assessment(health_score=0.9, candidates=candidates)
        mock_compiler = MagicMock()
        mock_compiler.compile = AsyncMock(return_value=mock_compiled)

        mock_result = _mock_orch_result(success=True, total_cost_usd=0.5)

        with (
            patch(
                "aragora.nomic.shift_controller.CanonicalAssessmentCompiler",
                return_value=mock_compiler,
            ),
            patch("aragora.nomic.hardened_orchestrator.HardenedOrchestrator") as mock_orch_cls,
        ):
            mock_orch_cls.return_value.execute_goal_coordinated = AsyncMock(
                return_value=mock_result
            )
            result = await controller.run_shift()

        phases = [cp.phase for cp in result.checkpoints]
        # assessment -> planning -> executing -> completed (post-assessment)
        assert phases[0] == "assessment"
        assert phases[1] == "planning"
        assert "executing" in phases
        assert phases[-1] == "completed"

        # All checkpoints reference the same shift
        for cp in result.checkpoints:
            assert cp.shift_id == result.shift_id

    @pytest.mark.asyncio
    async def test_shift_result_structure(self):
        """Verify all ShiftResult fields are populated."""
        config = ShiftConfig(
            max_items_per_shift=1,
            stop_on_dirty_worktree=False,
            require_assessment_after_shift=False,
        )
        controller = ShiftController(repo_path=Path("/tmp/test-repo"), config=config)

        candidates = [{"description": "Test goal"}]
        mock_compiled = _mock_assessment(health_score=0.8, candidates=candidates)
        mock_compiler = MagicMock()
        mock_compiler.compile = AsyncMock(return_value=mock_compiled)

        mock_result = _mock_orch_result(success=True, total_cost_usd=1.5)

        with (
            patch(
                "aragora.nomic.shift_controller.CanonicalAssessmentCompiler",
                return_value=mock_compiler,
            ),
            patch("aragora.nomic.hardened_orchestrator.HardenedOrchestrator") as mock_orch_cls,
        ):
            mock_orch_cls.return_value.execute_goal_coordinated = AsyncMock(
                return_value=mock_result
            )
            result = await controller.run_shift()

        assert isinstance(result, ShiftResult)
        assert result.shift_id.startswith("shift-")
        assert result.started_at > 0
        assert result.completed_at >= result.started_at
        assert result.duration_seconds >= 0
        assert result.items_attempted == 1
        assert result.items_landed == 1
        assert result.items_failed == 0
        assert result.items_reverted == 0
        assert result.budget_spent_usd == 1.5
        assert result.health_score_before == 0.8
        assert isinstance(result.checkpoints, list)
        assert len(result.checkpoints) > 0
        assert result.goals_executed == ["Test goal"]
        assert result.error is None

    @pytest.mark.asyncio
    async def test_execute_goal_failure_counted(self):
        """Failed goals are tracked in items_failed."""
        config = ShiftConfig(
            max_items_per_shift=3,
            stop_on_regression=False,
            stop_on_dirty_worktree=False,
            require_assessment_after_shift=False,
        )
        controller = ShiftController(repo_path=Path("/tmp/test-repo"), config=config)

        candidates = [
            {"description": "Goal A"},
            {"description": "Goal B"},
            {"description": "Goal C"},
        ]
        mock_compiled = _mock_assessment(candidates=candidates)
        mock_compiler = MagicMock()
        mock_compiler.compile = AsyncMock(return_value=mock_compiled)

        # All goals fail (success=False, no regression)
        mock_fail = _mock_orch_result(success=False, failed_subtasks=0)

        with (
            patch(
                "aragora.nomic.shift_controller.CanonicalAssessmentCompiler",
                return_value=mock_compiler,
            ),
            patch("aragora.nomic.hardened_orchestrator.HardenedOrchestrator") as mock_orch_cls,
        ):
            mock_orch_cls.return_value.execute_goal_coordinated = AsyncMock(return_value=mock_fail)
            result = await controller.run_shift()

        assert result.items_attempted == 3
        assert result.items_failed == 3
        assert result.items_landed == 0
        assert result.goals_executed == []

    @pytest.mark.asyncio
    async def test_dirty_worktree_detection(self):
        """Mocked git status with output triggers dirty_worktree stop."""
        config = ShiftConfig(
            stop_on_dirty_worktree=True,
            require_assessment_after_shift=False,
        )
        controller = ShiftController(repo_path=Path("/tmp/test-repo"), config=config)

        candidates = [{"description": "Goal 1"}]
        mock_compiled = _mock_assessment(candidates=candidates)
        mock_compiler = MagicMock()
        mock_compiler.compile = AsyncMock(return_value=mock_compiled)

        mock_proc = MagicMock()
        mock_proc.stdout = " M some_file.py\n"
        mock_proc.returncode = 0

        with (
            patch(
                "aragora.nomic.shift_controller.CanonicalAssessmentCompiler",
                return_value=mock_compiler,
            ),
            patch(
                "aragora.nomic.shift_controller.subprocess.run",
                return_value=mock_proc,
            ),
        ):
            result = await controller.run_shift()

        assert result.stop_reason == "dirty_worktree"
        assert result.items_attempted == 0

    @pytest.mark.asyncio
    async def test_run_shift_error_handling(self):
        """Exception during execution is caught and reported."""
        config = ShiftConfig(stop_on_dirty_worktree=False)
        controller = ShiftController(repo_path=Path("/tmp/test-repo"), config=config)

        # Make CanonicalAssessmentCompiler constructor raise
        with patch(
            "aragora.nomic.shift_controller.CanonicalAssessmentCompiler",
            side_effect=RuntimeError("total failure"),
        ):
            result = await controller.run_shift()

        # The assessment failure is caught gracefully inside _run_assessment,
        # which returns empty candidates -> no_work
        assert result.stop_reason == "no_work"

    @pytest.mark.asyncio
    async def test_shift_without_pre_assessment(self):
        """Skipping pre-assessment still plans from None -> no_work."""
        config = ShiftConfig(
            require_assessment_before_start=False,
            stop_on_dirty_worktree=False,
        )
        controller = ShiftController(repo_path=Path("/tmp/test-repo"), config=config)

        result = await controller.run_shift()

        assert result.stop_reason == "no_work"
        assert result.health_score_before == 0.0

    @pytest.mark.asyncio
    async def test_reverted_items_tracked(self):
        """Reverted goals are counted separately."""
        config = ShiftConfig(
            max_items_per_shift=2,
            stop_on_regression=False,
            stop_on_dirty_worktree=False,
            require_assessment_after_shift=False,
        )
        controller = ShiftController(repo_path=Path("/tmp/test-repo"), config=config)

        candidates = [{"description": "Goal 1"}, {"description": "Goal 2"}]
        mock_compiled = _mock_assessment(candidates=candidates)
        mock_compiler = MagicMock()
        mock_compiler.compile = AsyncMock(return_value=mock_compiled)

        mock_reverted = _mock_orch_result(success=False, reverted=True)

        with (
            patch(
                "aragora.nomic.shift_controller.CanonicalAssessmentCompiler",
                return_value=mock_compiler,
            ),
            patch("aragora.nomic.hardened_orchestrator.HardenedOrchestrator") as mock_orch_cls,
        ):
            mock_orch_cls.return_value.execute_goal_coordinated = AsyncMock(
                return_value=mock_reverted
            )
            result = await controller.run_shift()

        assert result.items_reverted == 2
        assert result.items_failed == 0
        assert result.items_landed == 0
