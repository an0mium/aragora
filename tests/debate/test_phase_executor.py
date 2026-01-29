"""
Tests for PhaseExecutor module.

Covers PhaseStatus, PhaseResult, ExecutionResult, PhaseConfig,
and PhaseExecutor execution logic including timeouts, failures,
optional/critical phases, early termination, and phase management.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from aragora.debate.phase_executor import (
    CRITICAL_PHASES,
    OPTIONAL_PHASES,
    STANDARD_PHASE_ORDER,
    ExecutionResult,
    PhaseConfig,
    PhaseExecutor,
    PhaseResult,
    PhaseStatus,
)


# ===========================================================================
# Helpers
# ===========================================================================


class SimplePhase:
    """A simple phase for testing."""

    def __init__(self, name: str, output: str = "ok", delay: float = 0.0, fail: bool = False):
        self._name = name
        self._output = output
        self._delay = delay
        self._fail = fail

    @property
    def name(self) -> str:
        return self._name

    async def execute(self, context):
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._fail:
            raise RuntimeError(f"Phase {self._name} failed")
        return self._output


# ===========================================================================
# PhaseStatus
# ===========================================================================


class TestPhaseStatus:
    """Tests for PhaseStatus enum."""

    def test_values(self):
        assert PhaseStatus.PENDING.value == "pending"
        assert PhaseStatus.RUNNING.value == "running"
        assert PhaseStatus.COMPLETED.value == "completed"
        assert PhaseStatus.SKIPPED.value == "skipped"
        assert PhaseStatus.FAILED.value == "failed"


# ===========================================================================
# PhaseResult
# ===========================================================================


class TestPhaseResult:
    """Tests for PhaseResult dataclass."""

    def test_success_when_completed(self):
        r = PhaseResult(phase_name="test", status=PhaseStatus.COMPLETED)
        assert r.success is True

    def test_success_when_skipped(self):
        r = PhaseResult(phase_name="test", status=PhaseStatus.SKIPPED)
        assert r.success is True

    def test_not_success_when_failed(self):
        r = PhaseResult(phase_name="test", status=PhaseStatus.FAILED)
        assert r.success is False

    def test_not_success_when_pending(self):
        r = PhaseResult(phase_name="test", status=PhaseStatus.PENDING)
        assert r.success is False

    def test_defaults(self):
        r = PhaseResult(phase_name="test", status=PhaseStatus.PENDING)
        assert r.started_at is None
        assert r.completed_at is None
        assert r.duration_ms == 0.0
        assert r.output is None
        assert r.error is None
        assert r.metrics == {}


# ===========================================================================
# ExecutionResult
# ===========================================================================


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_get_phase_result_found(self):
        phases = [
            PhaseResult(phase_name="proposal", status=PhaseStatus.COMPLETED),
            PhaseResult(phase_name="consensus", status=PhaseStatus.COMPLETED),
        ]
        result = ExecutionResult(
            debate_id="d1", success=True, phases=phases, total_duration_ms=100.0
        )
        assert result.get_phase_result("consensus") is not None
        assert result.get_phase_result("consensus").status == PhaseStatus.COMPLETED

    def test_get_phase_result_not_found(self):
        result = ExecutionResult(debate_id="d1", success=True, phases=[], total_duration_ms=0.0)
        assert result.get_phase_result("missing") is None


# ===========================================================================
# Constants
# ===========================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_standard_phase_order(self):
        assert "proposal" in STANDARD_PHASE_ORDER
        assert "consensus" in STANDARD_PHASE_ORDER
        assert STANDARD_PHASE_ORDER.index("proposal") < STANDARD_PHASE_ORDER.index("consensus")

    def test_optional_phases(self):
        assert "analytics" in OPTIONAL_PHASES
        assert "feedback" in OPTIONAL_PHASES
        assert "consensus" not in OPTIONAL_PHASES

    def test_critical_phases(self):
        assert "consensus" in CRITICAL_PHASES


# ===========================================================================
# PhaseExecutor — execution
# ===========================================================================


class TestPhaseExecutorExecution:
    """Tests for PhaseExecutor.execute."""

    @pytest.mark.asyncio
    async def test_executes_all_phases(self):
        phases = {
            "proposal": SimplePhase("proposal", output="proposals"),
            "consensus": SimplePhase("consensus", output="agreed"),
        }
        executor = PhaseExecutor(phases, PhaseConfig(enable_tracing=False))
        result = await executor.execute(
            context={},
            debate_id="test-1",
            phase_order=["proposal", "consensus"],
        )
        assert result.success is True
        assert len(result.phases) == 2
        assert result.phases[0].phase_name == "proposal"
        assert result.phases[1].phase_name == "consensus"

    @pytest.mark.asyncio
    async def test_captures_consensus_output(self):
        phases = {
            "proposal": SimplePhase("proposal"),
            "consensus": SimplePhase("consensus", output="final_answer"),
        }
        executor = PhaseExecutor(phases, PhaseConfig(enable_tracing=False))
        result = await executor.execute(
            context={}, debate_id="test-1", phase_order=["proposal", "consensus"]
        )
        assert result.final_output == "final_answer"

    @pytest.mark.asyncio
    async def test_filters_unknown_phases(self):
        phases = {"proposal": SimplePhase("proposal")}
        executor = PhaseExecutor(phases, PhaseConfig(enable_tracing=False))
        result = await executor.execute(
            context={},
            debate_id="test-1",
            phase_order=["proposal", "nonexistent"],
        )
        assert len(result.phases) == 1

    @pytest.mark.asyncio
    async def test_uses_standard_order_by_default(self):
        phases = {
            "proposal": SimplePhase("proposal"),
            "consensus": SimplePhase("consensus"),
        }
        executor = PhaseExecutor(phases, PhaseConfig(enable_tracing=False))
        result = await executor.execute(context={}, debate_id="test-1")
        # Should execute only phases that exist in STANDARD_PHASE_ORDER
        names = [r.phase_name for r in result.phases]
        assert "proposal" in names
        assert "consensus" in names

    @pytest.mark.asyncio
    async def test_overall_timeout(self):
        phases = {
            "slow": SimplePhase("slow", delay=5.0),
        }
        config = PhaseConfig(total_timeout_seconds=0.1, enable_tracing=False)
        executor = PhaseExecutor(phases, config)
        result = await executor.execute(context={}, debate_id="test-1", phase_order=["slow"])
        assert result.success is False
        assert "timed out" in (result.error or "").lower()


# ===========================================================================
# PhaseExecutor — failure handling
# ===========================================================================


class TestPhaseExecutorFailure:
    """Tests for failure handling."""

    @pytest.mark.asyncio
    async def test_required_phase_failure_stops_execution(self):
        phases = {
            "proposal": SimplePhase("proposal", fail=True),
            "feedback": SimplePhase("feedback"),
        }
        config = PhaseConfig(stop_on_failure=True, enable_tracing=False)
        executor = PhaseExecutor(phases, config)
        result = await executor.execute(
            context={}, debate_id="test-1", phase_order=["proposal", "feedback"]
        )
        assert result.success is False
        # Feedback should not have run
        assert len(result.phases) == 1

    @pytest.mark.asyncio
    async def test_optional_phase_failure_continues(self):
        phases = {
            "analytics": SimplePhase("analytics", fail=True),
            "feedback": SimplePhase("feedback"),
        }
        config = PhaseConfig(stop_on_failure=True, enable_tracing=False)
        executor = PhaseExecutor(phases, config)
        result = await executor.execute(
            context={}, debate_id="test-1", phase_order=["analytics", "feedback"]
        )
        # analytics is optional, so feedback should still run
        assert len(result.phases) == 2

    @pytest.mark.asyncio
    async def test_critical_phase_runs_after_earlier_failure(self):
        phases = {
            "proposal": SimplePhase("proposal", fail=True),
            "consensus": SimplePhase("consensus", output="answer"),
        }
        config = PhaseConfig(stop_on_failure=True, enable_tracing=False)
        executor = PhaseExecutor(phases, config)
        result = await executor.execute(
            context={}, debate_id="test-1", phase_order=["proposal", "consensus"]
        )
        # consensus is critical, should still run after proposal failure
        names = [r.phase_name for r in result.phases]
        assert "consensus" in names

    @pytest.mark.asyncio
    async def test_phase_timeout_optional_skipped(self):
        phases = {
            "analytics": SimplePhase("analytics", delay=5.0),
        }
        config = PhaseConfig(
            phase_timeout_seconds=0.05,
            skip_optional_on_timeout=True,
            enable_tracing=False,
        )
        executor = PhaseExecutor(phases, config)
        result = await executor.execute(context={}, debate_id="test-1", phase_order=["analytics"])
        assert result.phases[0].status == PhaseStatus.SKIPPED


# ===========================================================================
# PhaseExecutor — termination
# ===========================================================================


class TestPhaseExecutorTermination:
    """Tests for early termination."""

    @pytest.mark.asyncio
    async def test_request_termination_mid_execution(self):
        """Termination requested after first phase stops before second."""

        class TerminatingPhase:
            def __init__(self, executor_ref):
                self._executor = executor_ref

            @property
            def name(self):
                return "terminator"

            async def execute(self, context):
                self._executor.request_termination("Test reason")
                return "done"

        executor = PhaseExecutor({}, PhaseConfig(enable_tracing=False))
        executor.add_phase("first", TerminatingPhase(executor))
        executor.add_phase("second", SimplePhase("second"))

        result = await executor.execute(
            context={}, debate_id="test-1", phase_order=["first", "second"]
        )
        # Only first phase should have executed; second skipped due to termination
        assert len(result.phases) == 1
        assert result.phases[0].phase_name == "first"

    def test_check_termination_default(self):
        executor = PhaseExecutor({}, PhaseConfig())
        should, reason = executor.check_termination()
        assert should is False
        assert reason is None

    def test_check_termination_after_request(self):
        executor = PhaseExecutor({}, PhaseConfig())
        executor.request_termination("done")
        should, reason = executor.check_termination()
        assert should is True
        assert reason == "done"


# ===========================================================================
# PhaseExecutor — phase management
# ===========================================================================


class TestPhaseManagement:
    """Tests for add/remove/get phase methods."""

    def test_add_phase(self):
        executor = PhaseExecutor({})
        phase = SimplePhase("new_phase")
        executor.add_phase("new_phase", phase)
        assert executor.get_phase("new_phase") is phase

    def test_remove_phase(self):
        phases = {"existing": SimplePhase("existing")}
        executor = PhaseExecutor(phases)
        assert executor.remove_phase("existing") is True
        assert executor.get_phase("existing") is None

    def test_remove_nonexistent_phase(self):
        executor = PhaseExecutor({})
        assert executor.remove_phase("missing") is False

    def test_phase_names(self):
        phases = {
            "proposal": SimplePhase("proposal"),
            "consensus": SimplePhase("consensus"),
        }
        executor = PhaseExecutor(phases)
        assert set(executor.phase_names) == {"proposal", "consensus"}

    def test_current_phase_initially_none(self):
        executor = PhaseExecutor({})
        assert executor.current_phase is None


# ===========================================================================
# PhaseExecutor — metrics
# ===========================================================================


class TestPhaseMetrics:
    """Tests for metrics collection."""

    @pytest.mark.asyncio
    async def test_get_metrics(self):
        phases = {
            "proposal": SimplePhase("proposal"),
            "consensus": SimplePhase("consensus"),
        }
        executor = PhaseExecutor(phases, PhaseConfig(enable_tracing=False))
        await executor.execute(
            context={}, debate_id="test-1", phase_order=["proposal", "consensus"]
        )
        metrics = executor.get_metrics()
        assert metrics["total_phases"] == 2
        assert metrics["completed_phases"] == 2
        assert metrics["failed_phases"] == 0
        assert metrics["skipped_phases"] == 0
        assert "proposal" in metrics["phase_durations"]
        assert "consensus" in metrics["phase_durations"]

    @pytest.mark.asyncio
    async def test_get_results_returns_copy(self):
        phases = {"proposal": SimplePhase("proposal")}
        executor = PhaseExecutor(phases, PhaseConfig(enable_tracing=False))
        await executor.execute(context={}, debate_id="test-1", phase_order=["proposal"])
        results = executor.get_results()
        assert len(results) == 1
        # Modifying the copy should not affect the original
        results.clear()
        assert len(executor.get_results()) == 1

    @pytest.mark.asyncio
    async def test_metrics_callback_called(self):
        called = {}

        def on_metric(name, value):
            called[name] = value

        phases = {"proposal": SimplePhase("proposal")}
        config = PhaseConfig(enable_tracing=False, metrics_callback=on_metric)
        executor = PhaseExecutor(phases, config)
        await executor.execute(context={}, debate_id="test-1", phase_order=["proposal"])
        assert "phase_proposal_duration_ms" in called
