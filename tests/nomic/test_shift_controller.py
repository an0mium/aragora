"""Tests for the ShiftController bounded improvement shift manager."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


@pytest.fixture()
def shift_dir(tmp_path: Path) -> str:
    """Return a temporary directory for shift checkpoints."""
    d = tmp_path / "shifts"
    d.mkdir()
    return str(d)


@pytest.fixture()
def config(shift_dir: str):
    """Return a ShiftConfig with short intervals for testing."""
    from aragora.nomic.shift_controller import ShiftConfig

    return ShiftConfig(
        max_duration_hours=1.0,
        refresh_interval_minutes=30.0,
        budget_limit_usd=5.0,
        max_cycles=10,
        require_fresh_assessment=True,
        checkpoint_dir=shift_dir,
    )


@pytest.fixture()
def config_no_assessment(shift_dir: str):
    """Return a ShiftConfig that does NOT require an assessment."""
    from aragora.nomic.shift_controller import ShiftConfig

    return ShiftConfig(
        max_duration_hours=1.0,
        refresh_interval_minutes=30.0,
        budget_limit_usd=5.0,
        max_cycles=10,
        require_fresh_assessment=False,
        checkpoint_dir=shift_dir,
    )


# ---------------------------------------------------------------------------
# start_shift
# ---------------------------------------------------------------------------


class TestStartShift:
    """Tests for ShiftController.start_shift()."""

    def test_start_shift_creates_state(self, config):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config)
        state = _run(ctrl.start_shift(assessment_id="assess-1"))

        assert state.shift_id
        assert state.status == "running"
        assert state.assessment_id == "assess-1"
        assert state.current_cycle == 0
        assert state.started_at > 0

    def test_start_shift_requires_assessment(self, config):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config)
        with pytest.raises(ValueError, match="assessment_id is required"):
            _run(ctrl.start_shift())

    def test_start_shift_without_assessment_when_not_required(self, config_no_assessment):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config_no_assessment)
        state = _run(ctrl.start_shift())

        assert state.status == "running"
        assert state.assessment_id is None

    def test_start_shift_rejects_duplicate(self, config):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config)
        _run(ctrl.start_shift(assessment_id="a-1"))
        with pytest.raises(ValueError, match="already running"):
            _run(ctrl.start_shift(assessment_id="a-2"))


# ---------------------------------------------------------------------------
# check_refresh_due
# ---------------------------------------------------------------------------


class TestCheckRefreshDue:
    """Tests for refresh interval checking."""

    def test_check_refresh_due_returns_true_after_interval(self, config):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config)
        _run(ctrl.start_shift(assessment_id="a-1"))

        # Simulate time passing beyond the refresh interval (30 min)
        with patch("aragora.nomic.shift_controller.time") as mock_time:
            mock_time.time.return_value = ctrl.state.started_at + 31 * 60
            assert ctrl.check_refresh_due() is True

    def test_check_refresh_not_due_before_interval(self, config):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config)
        _run(ctrl.start_shift(assessment_id="a-1"))

        # Still within the 30 min window
        with patch("aragora.nomic.shift_controller.time") as mock_time:
            mock_time.time.return_value = ctrl.state.started_at + 10 * 60
            assert ctrl.check_refresh_due() is False


# ---------------------------------------------------------------------------
# check_should_stop
# ---------------------------------------------------------------------------


class TestCheckShouldStop:
    """Tests for stop-condition evaluation."""

    def test_check_should_stop_time_limit(self, config):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config)
        _run(ctrl.start_shift(assessment_id="a-1"))

        # Exceed 1h max_duration
        with patch("aragora.nomic.shift_controller.time") as mock_time:
            mock_time.time.return_value = ctrl.state.started_at + 2 * 3600
            should_stop, reason = ctrl.check_should_stop()
            assert should_stop is True
            assert "TimeLimit" in reason

    def test_check_should_stop_budget(self, config):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config)
        _run(ctrl.start_shift(assessment_id="a-1"))

        # Exhaust budget
        ctrl.state.budget_spent_usd = 6.0

        # Keep time within bounds
        with patch("aragora.nomic.shift_controller.time") as mock_time:
            mock_time.time.return_value = ctrl.state.started_at + 60
            should_stop, reason = ctrl.check_should_stop()
            assert should_stop is True
            assert "BudgetExhausted" in reason

    def test_check_should_stop_cycle_limit(self, config):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config)
        _run(ctrl.start_shift(assessment_id="a-1"))

        # Hit cycle limit
        ctrl.state.current_cycle = 10

        with patch("aragora.nomic.shift_controller.time") as mock_time:
            mock_time.time.return_value = ctrl.state.started_at + 60
            should_stop, reason = ctrl.check_should_stop()
            assert should_stop is True
            assert "CycleLimit" in reason

    def test_check_should_stop_refresh_due(self, config):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config)
        _run(ctrl.start_shift(assessment_id="a-1"))

        # Past refresh interval (30 min) but within time limit (1h)
        with patch("aragora.nomic.shift_controller.time") as mock_time:
            mock_time.time.return_value = ctrl.state.started_at + 31 * 60
            should_stop, reason = ctrl.check_should_stop()
            assert should_stop is True
            assert "RefreshDue" in reason

    def test_check_should_stop_dirty_worktree_when_enabled(self, config):
        from aragora.nomic.shift_controller import ShiftController

        config.enforce_clean_repo = True
        ctrl = ShiftController(config)
        _run(ctrl.start_shift(assessment_id="a-1"))

        with patch.object(ctrl, "_is_repo_dirty", return_value=True):
            should_stop, reason = ctrl.check_should_stop()

        assert should_stop is True
        assert "DirtyWorktree" in reason

    def test_check_should_stop_colliding_worktrees_when_enabled(self, config):
        from aragora.nomic.shift_controller import ShiftController

        config.enforce_worktree_collision_check = True
        ctrl = ShiftController(config)
        _run(ctrl.start_shift(assessment_id="a-1"))

        with patch.object(ctrl, "_has_colliding_worktrees", return_value=True):
            should_stop, reason = ctrl.check_should_stop()

        assert should_stop is True
        assert "CollidingWorktrees" in reason

    def test_no_stop_when_within_bounds(self, config):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config)
        _run(ctrl.start_shift(assessment_id="a-1"))

        with patch("aragora.nomic.shift_controller.time") as mock_time:
            mock_time.time.return_value = ctrl.state.started_at + 5 * 60
            should_stop, reason = ctrl.check_should_stop()
            assert should_stop is False
            assert reason == ""


# ---------------------------------------------------------------------------
# pause / resume
# ---------------------------------------------------------------------------


class TestPauseResume:
    """Tests for pause_for_refresh and resume_after_refresh."""

    def test_pause_for_refresh(self, config):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config)
        _run(ctrl.start_shift(assessment_id="a-1"))
        state = _run(ctrl.pause_for_refresh())

        assert state.status == "paused_for_refresh"

    def test_resume_after_refresh(self, config):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config)
        _run(ctrl.start_shift(assessment_id="a-1"))
        _run(ctrl.pause_for_refresh())
        state = _run(ctrl.resume_after_refresh(new_assessment_id="a-2"))

        assert state.status == "running"
        assert state.assessment_id == "a-2"
        assert state.refresh_count == 1

    def test_resume_requires_assessment(self, config):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config)
        _run(ctrl.start_shift(assessment_id="a-1"))
        _run(ctrl.pause_for_refresh())

        with pytest.raises(ValueError, match="assessment_id is required"):
            _run(ctrl.resume_after_refresh(new_assessment_id=""))

    def test_resume_requires_new_assessment_id(self, config):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config)
        _run(ctrl.start_shift(assessment_id="a-1"))
        _run(ctrl.pause_for_refresh())

        with pytest.raises(ValueError, match="must differ from the previous assessment"):
            _run(ctrl.resume_after_refresh(new_assessment_id="a-1"))

    def test_resume_fails_if_not_paused(self, config):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config)
        _run(ctrl.start_shift(assessment_id="a-1"))

        with pytest.raises(RuntimeError, match="not paused"):
            _run(ctrl.resume_after_refresh(new_assessment_id="a-2"))


# ---------------------------------------------------------------------------
# complete_shift
# ---------------------------------------------------------------------------


class TestCompleteShift:
    """Tests for shift completion."""

    def test_complete_shift(self, config):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config)
        _run(ctrl.start_shift(assessment_id="a-1"))
        state = ctrl.complete_shift("all objectives done")

        assert state.status == "completed"
        assert state.stop_reason == "all objectives done"

    def test_complete_shift_no_active(self, config):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config)
        with pytest.raises(RuntimeError, match="No active shift"):
            ctrl.complete_shift()


# ---------------------------------------------------------------------------
# run_cycle
# ---------------------------------------------------------------------------


class TestRunCycle:
    """Tests for run_cycle execution tracking."""

    def test_run_cycle_increments_count(self, config):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config)
        _run(ctrl.start_shift(assessment_id="a-1"))

        with patch("aragora.nomic.shift_controller.time") as mock_time:
            mock_time.time.return_value = ctrl.state.started_at + 60
            result = _run(ctrl.run_cycle("improve coverage", cost_usd=1.25, receipt_id="rcpt-1"))

        assert result["completed"] is True
        assert result["cycle"] == 1
        assert ctrl.state.current_cycle == 1
        assert "improve coverage" in ctrl.state.objectives_completed
        assert ctrl.state.budget_spent_usd == pytest.approx(1.25)
        assert ctrl.state.last_receipt_id == "rcpt-1"

    def test_run_cycle_stops_on_limit(self, config):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config)
        _run(ctrl.start_shift(assessment_id="a-1"))
        ctrl.state.current_cycle = 10  # Already at limit

        with patch("aragora.nomic.shift_controller.time") as mock_time:
            mock_time.time.return_value = ctrl.state.started_at + 60
            result = _run(ctrl.run_cycle("another objective"))

        assert result["completed"] is False
        assert "CycleLimit" in result["stop_reason"]

    def test_run_cycle_no_active_shift(self, config):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config)
        with pytest.raises(RuntimeError, match="No active shift"):
            _run(ctrl.run_cycle("objective"))

    def test_run_cycle_stops_on_missing_receipt_when_required(self, config):
        from aragora.nomic.shift_controller import ShiftController

        config.require_receipts = True
        ctrl = ShiftController(config)
        _run(ctrl.start_shift(assessment_id="a-1"))

        with patch("aragora.nomic.shift_controller.time") as mock_time:
            mock_time.time.return_value = ctrl.state.started_at + 60
            result = _run(ctrl.run_cycle("objective without receipt"))

        assert result["completed"] is False
        assert "MissingReceipt" in result["stop_reason"]
        assert "objective without receipt" in ctrl.state.objectives_failed

    def test_run_cycle_stops_on_quality_gate_failure(self, config):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config)
        _run(ctrl.start_shift(assessment_id="a-1"))

        with patch("aragora.nomic.shift_controller.time") as mock_time:
            mock_time.time.return_value = ctrl.state.started_at + 60
            result = _run(ctrl.run_cycle("objective", quality_gate_passed=False))

        assert result["completed"] is False
        assert "QualityGateFailed" in result["stop_reason"]
        assert "objective" in ctrl.state.objectives_failed

    def test_run_cycle_stops_on_ownership_ambiguity(self, config):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config)
        _run(ctrl.start_shift(assessment_id="a-1"))

        with patch("aragora.nomic.shift_controller.time") as mock_time:
            mock_time.time.return_value = ctrl.state.started_at + 60
            result = _run(ctrl.run_cycle("objective", ownership_clear=False))

        assert result["completed"] is False
        assert "OwnershipAmbiguous" in result["stop_reason"]
        assert "objective" in ctrl.state.objectives_failed


# ---------------------------------------------------------------------------
# progress_summary
# ---------------------------------------------------------------------------


class TestProgressSummary:
    """Tests for get_progress_summary."""

    def test_progress_summary(self, config):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config)
        _run(ctrl.start_shift(assessment_id="a-1"))

        with patch("aragora.nomic.shift_controller.time") as mock_time:
            mock_time.time.return_value = ctrl.state.started_at + 15 * 60
            summary = ctrl.get_progress_summary()

        assert summary["active"] is True
        assert summary["elapsed_hours"] == 0.25
        assert summary["remaining_hours"] == 0.75
        assert summary["cycles_completed"] == 0
        assert summary["max_cycles"] == 10

    def test_progress_summary_includes_assessment_and_blockers(self, config):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config)
        _run(ctrl.start_shift(assessment_id="a-1"))
        ctrl.state.current_objective = "improve coverage"
        ctrl.state.last_receipt_id = "rcpt-1"
        ctrl.state.blocked_reasons = ["RefreshDue"]
        ctrl.state.next_refresh_actions = ["Run `aragora assess --save --diff` before resuming."]
        ctrl.state.stop_reason = "RefreshDue"

        with patch("aragora.nomic.shift_controller.time") as mock_time:
            mock_time.time.return_value = ctrl.state.started_at + 15 * 60
            summary = ctrl.get_progress_summary()

        assert summary["assessment_id"] == "a-1"
        assert summary["current_objective"] == "improve coverage"
        assert summary["last_receipt_id"] == "rcpt-1"
        assert summary["blocked_reasons"] == ["RefreshDue"]
        assert summary["stop_reason"] == "RefreshDue"

    def test_progress_summary_no_shift(self, config):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config)
        summary = ctrl.get_progress_summary()
        assert summary == {"active": False}


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


class TestCheckpointRoundTrip:
    """Tests for checkpoint persistence and resume."""

    def test_checkpoint_round_trip(self, config, shift_dir):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config)
        _run(ctrl.start_shift(assessment_id="a-1"))

        # Run a cycle so state has content
        with patch("aragora.nomic.shift_controller.time") as mock_time:
            mock_time.time.return_value = ctrl.state.started_at + 60
            _run(ctrl.run_cycle("first objective"))

        # Pause for refresh (saves checkpoint)
        _run(ctrl.pause_for_refresh())
        original_id = ctrl.state.shift_id

        # Resume from checkpoint in a new controller
        ctrl2 = ShiftController.resume_from_checkpoint(shift_dir)
        assert ctrl2 is not None
        assert ctrl2.state.shift_id == original_id
        assert ctrl2.state.status == "paused_for_refresh"
        assert ctrl2.state.current_cycle == 1
        assert "first objective" in ctrl2.state.objectives_completed

    def test_resume_returns_none_when_empty(self, shift_dir):
        from aragora.nomic.shift_controller import ShiftController

        result = ShiftController.resume_from_checkpoint(shift_dir)
        assert result is None

    def test_load_shift_history(self, config, shift_dir):
        from aragora.nomic.shift_controller import ShiftController

        ctrl = ShiftController(config)
        _run(ctrl.start_shift(assessment_id="a-1"))
        ctrl.complete_shift("done")

        history = ShiftController.load_shift_history(shift_dir)
        assert len(history) >= 1
        assert history[0]["status"] == "completed"
        assert history[0]["stop_reason"] == "done"


# ---------------------------------------------------------------------------
# ShiftState serialization
# ---------------------------------------------------------------------------


class TestShiftStateSerialization:
    """Tests for ShiftState.to_dict / from_dict round-trip."""

    def test_shift_state_serialization(self):
        from aragora.nomic.shift_controller import ShiftState

        state = ShiftState(
            shift_id="abc123",
            started_at=1000.0,
            config={"max_duration_hours": 4.0},
            status="running",
            current_cycle=5,
            elapsed_seconds=300.0,
            budget_spent_usd=1.23,
            refresh_count=1,
            last_refresh_at=1200.0,
            last_checkpoint_at=1250.0,
            assessment_id="assess-x",
            current_objective="obj-running",
            last_receipt_id="rcpt-1",
            objectives_completed=["obj1", "obj2"],
            objectives_failed=["obj3"],
            blocked_reasons=["RefreshDue"],
            next_refresh_actions=["Run `aragora assess --save --diff` before resuming."],
            stop_reason="",
        )

        data = state.to_dict()
        restored = ShiftState.from_dict(data)

        assert restored.shift_id == state.shift_id
        assert restored.started_at == state.started_at
        assert restored.status == state.status
        assert restored.current_cycle == state.current_cycle
        assert restored.elapsed_seconds == state.elapsed_seconds
        assert restored.budget_spent_usd == state.budget_spent_usd
        assert restored.refresh_count == state.refresh_count
        assert restored.last_refresh_at == state.last_refresh_at
        assert restored.assessment_id == state.assessment_id
        assert restored.current_objective == "obj-running"
        assert restored.last_receipt_id == "rcpt-1"
        assert restored.objectives_completed == ["obj1", "obj2"]
        assert restored.objectives_failed == ["obj3"]
        assert restored.blocked_reasons == ["RefreshDue"]

    def test_to_dict_is_json_serializable(self):
        from aragora.nomic.shift_controller import ShiftState

        state = ShiftState(
            shift_id="test",
            started_at=1000.0,
            config={"key": "value"},
        )
        # Must not raise
        json.dumps(state.to_dict())


# ---------------------------------------------------------------------------
# ShiftRefreshDue stopping rule
# ---------------------------------------------------------------------------


class TestShiftRefreshDueStoppingRule:
    """Tests for the ShiftRefreshDue rule in StoppingRuleEngine."""

    def test_refresh_due_triggers(self):
        from aragora.nomic.stopping_rules import StoppingConfig, StoppingRuleEngine

        engine = StoppingRuleEngine()
        start = 1000.0
        config = StoppingConfig(refresh_interval_minutes=30.0, last_refresh_at=0.0)

        # 31 minutes after start
        with patch("aragora.nomic.stopping_rules.time") as mock_time:
            mock_time.time.return_value = start + 31 * 60
            stop, reason = engine._check_shift_refresh_due(start, config)
            assert stop is True
            assert "ShiftRefreshDue" in reason

    def test_refresh_not_due_yet(self):
        from aragora.nomic.stopping_rules import StoppingConfig, StoppingRuleEngine

        engine = StoppingRuleEngine()
        start = 1000.0
        config = StoppingConfig(refresh_interval_minutes=30.0, last_refresh_at=0.0)

        # 10 minutes after start
        with patch("aragora.nomic.stopping_rules.time") as mock_time:
            mock_time.time.return_value = start + 10 * 60
            stop, reason = engine._check_shift_refresh_due(start, config)
            assert stop is False

    def test_refresh_due_uses_last_refresh_at(self):
        from aragora.nomic.stopping_rules import StoppingConfig, StoppingRuleEngine

        engine = StoppingRuleEngine()
        start = 1000.0
        # Last refresh was at t=2000
        config = StoppingConfig(
            refresh_interval_minutes=30.0,
            last_refresh_at=2000.0,
        )

        # 10 minutes after last refresh — not due
        with patch("aragora.nomic.stopping_rules.time") as mock_time:
            mock_time.time.return_value = 2000.0 + 10 * 60
            stop, _ = engine._check_shift_refresh_due(start, config)
            assert stop is False

        # 31 minutes after last refresh — due
        with patch("aragora.nomic.stopping_rules.time") as mock_time:
            mock_time.time.return_value = 2000.0 + 31 * 60
            stop, reason = engine._check_shift_refresh_due(start, config)
            assert stop is True
            assert "ShiftRefreshDue" in reason
