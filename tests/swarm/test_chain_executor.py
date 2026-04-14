"""Tests for chain executor wave-based dispatch planning."""

from __future__ import annotations

import pytest

from aragora.swarm.chain_executor import (
    ChainExecutionPlan,
    ChainExecutor,
    StepDispatch,
    StepOutcome,
    WaveDispatchPlan,
)
from aragora.swarm.work_chain import (
    ChainStatus,
    ChainStep,
    StepStatus,
    WorkChain,
)


def _linear_chain() -> WorkChain:
    return WorkChain(
        chain_id="linear",
        title="Linear chain",
        steps=[
            ChainStep(step_id="a", work_order_id="wo-a", title="Step A"),
            ChainStep(step_id="b", work_order_id="wo-b", title="Step B", depends_on=["a"]),
            ChainStep(step_id="c", work_order_id="wo-c", title="Step C", depends_on=["b"]),
        ],
    )


def _diamond_chain() -> WorkChain:
    return WorkChain(
        chain_id="diamond",
        title="Diamond chain",
        steps=[
            ChainStep(step_id="root", work_order_id="wo-root", title="Root"),
            ChainStep(step_id="left", work_order_id="wo-left", title="Left", depends_on=["root"]),
            ChainStep(
                step_id="right", work_order_id="wo-right", title="Right", depends_on=["root"]
            ),
            ChainStep(
                step_id="join",
                work_order_id="wo-join",
                title="Join",
                depends_on=["left", "right"],
            ),
        ],
    )


def _parallel_chain() -> WorkChain:
    return WorkChain(
        chain_id="parallel",
        title="Parallel chain",
        steps=[
            ChainStep(step_id="a", work_order_id="wo-a", title="A"),
            ChainStep(step_id="b", work_order_id="wo-b", title="B"),
            ChainStep(step_id="c", work_order_id="wo-c", title="C"),
        ],
    )


class TestChainExecutorPlan:
    def test_plan_linear_chain(self) -> None:
        executor = ChainExecutor(_linear_chain())
        plan = executor.plan()
        assert plan.total_steps == 3
        assert plan.max_parallelism == 1
        assert len(plan.waves) == 3

    def test_plan_diamond_chain(self) -> None:
        executor = ChainExecutor(_diamond_chain())
        plan = executor.plan()
        assert plan.total_steps == 4
        assert plan.max_parallelism == 2
        assert len(plan.waves) == 3

    def test_plan_parallel_chain(self) -> None:
        executor = ChainExecutor(_parallel_chain())
        plan = executor.plan()
        assert plan.total_steps == 3
        assert plan.max_parallelism == 3
        assert len(plan.waves) == 1

    def test_plan_to_dict(self) -> None:
        executor = ChainExecutor(_linear_chain())
        plan = executor.plan()
        d = plan.to_dict()
        assert d["chain_id"] == "linear"
        assert len(d["waves"]) == 3

    def test_empty_chain_raises(self) -> None:
        chain = WorkChain(chain_id="empty", title="Empty")
        with pytest.raises(ValueError, match="empty chain"):
            ChainExecutor(chain)


class TestNextWave:
    def test_first_wave_dispatches_roots(self) -> None:
        executor = ChainExecutor(_diamond_chain())
        wave = executor.next_wave()
        assert wave is not None
        assert len(wave.dispatches) == 1
        assert wave.dispatches[0].step_id == "root"

    def test_completing_root_unlocks_second_wave(self) -> None:
        executor = ChainExecutor(_diamond_chain())
        wave = executor.next_wave()
        assert wave is not None

        executor.record_outcome(
            StepOutcome(step_id="root", status=StepStatus.COMPLETED, branch="feat/root")
        )
        wave2 = executor.next_wave()
        assert wave2 is not None
        assert len(wave2.dispatches) == 2
        dispatch_ids = sorted(d.step_id for d in wave2.dispatches)
        assert dispatch_ids == ["left", "right"]

    def test_predecessor_outputs_are_passed(self) -> None:
        executor = ChainExecutor(_linear_chain())
        wave1 = executor.next_wave()
        assert wave1 is not None

        executor.record_outcome(
            StepOutcome(
                step_id="a",
                status=StepStatus.COMPLETED,
                branch="feat/a",
                changed_files=["mod.py"],
            )
        )
        wave2 = executor.next_wave()
        assert wave2 is not None
        dispatch = wave2.dispatches[0]
        assert "a" in dispatch.predecessor_outputs
        assert dispatch.predecessor_outputs["a"].branch == "feat/a"

    def test_no_more_waves_when_complete(self) -> None:
        executor = ChainExecutor(_parallel_chain())
        wave = executor.next_wave()
        assert wave is not None

        for d in wave.dispatches:
            executor.record_outcome(StepOutcome(step_id=d.step_id, status=StepStatus.COMPLETED))

        assert executor.next_wave() is None
        assert executor.is_complete()


class TestFailurePropagation:
    def test_failed_step_skips_dependents(self) -> None:
        executor = ChainExecutor(_linear_chain())
        wave = executor.next_wave()
        assert wave is not None

        executor.record_outcome(
            StepOutcome(step_id="a", status=StepStatus.FAILED, error="compile error")
        )

        # Next wave should return None because b and c are skipped
        assert executor.next_wave() is None
        assert executor.is_complete()
        assert executor.chain.status == ChainStatus.FAILED

    def test_diamond_partial_failure(self) -> None:
        executor = ChainExecutor(_diamond_chain())

        # Execute root
        wave = executor.next_wave()
        assert wave is not None
        executor.record_outcome(StepOutcome(step_id="root", status=StepStatus.COMPLETED))

        # Execute left+right: left succeeds, right fails
        wave2 = executor.next_wave()
        assert wave2 is not None
        executor.record_outcome(StepOutcome(step_id="left", status=StepStatus.COMPLETED))
        executor.record_outcome(StepOutcome(step_id="right", status=StepStatus.FAILED))

        # Join should be skipped because right failed
        assert executor.next_wave() is None
        step_join = executor.chain.step_by_id("join")
        assert step_join is not None
        assert step_join.status == StepStatus.SKIPPED


class TestSummary:
    def test_summary_after_full_execution(self) -> None:
        executor = ChainExecutor(_parallel_chain())
        wave = executor.next_wave()
        assert wave is not None

        executor.record_outcome(StepOutcome(step_id="a", status=StepStatus.COMPLETED))
        executor.record_outcome(StepOutcome(step_id="b", status=StepStatus.FAILED, error="oops"))
        executor.record_outcome(StepOutcome(step_id="c", status=StepStatus.COMPLETED))

        summary = executor.summary()
        assert summary["chain_status"] == "failed"
        assert summary["completed"] == 2
        assert summary["failed"] == 1
        assert summary["skipped"] == 0
        assert summary["remaining"] == 0

    def test_summary_mid_execution(self) -> None:
        executor = ChainExecutor(_linear_chain())
        wave = executor.next_wave()
        assert wave is not None
        executor.record_outcome(StepOutcome(step_id="a", status=StepStatus.COMPLETED))

        summary = executor.summary()
        assert summary["completed"] == 1
        assert summary["remaining"] == 2


class TestSingleStepChain:
    """Edge case: chain with only one step."""

    def test_single_step_plan(self) -> None:
        chain = WorkChain(
            chain_id="single",
            title="Single step",
            steps=[ChainStep(step_id="only", work_order_id="wo-only", title="Only")],
        )
        executor = ChainExecutor(chain)
        plan = executor.plan()
        assert plan.total_steps == 1
        assert plan.max_parallelism == 1
        assert len(plan.waves) == 1

    def test_single_step_execution(self) -> None:
        chain = WorkChain(
            chain_id="single",
            title="Single step",
            steps=[ChainStep(step_id="only", work_order_id="wo-only", title="Only")],
        )
        executor = ChainExecutor(chain)
        wave = executor.next_wave()
        assert wave is not None
        assert len(wave.dispatches) == 1
        assert wave.dispatches[0].step_id == "only"

        executor.record_outcome(StepOutcome(step_id="only", status=StepStatus.COMPLETED))
        assert executor.next_wave() is None
        assert executor.is_complete()
        assert executor.chain.status == ChainStatus.COMPLETED


class TestAllStepsFail:
    """Edge case: every step in the chain fails."""

    def test_all_parallel_steps_fail(self) -> None:
        executor = ChainExecutor(_parallel_chain())
        wave = executor.next_wave()
        assert wave is not None

        for d in wave.dispatches:
            executor.record_outcome(
                StepOutcome(step_id=d.step_id, status=StepStatus.FAILED, error="boom")
            )

        assert executor.next_wave() is None
        assert executor.is_complete()
        assert executor.chain.status == ChainStatus.FAILED

        summary = executor.summary()
        assert summary["completed"] == 0
        assert summary["failed"] == 3
        assert summary["skipped"] == 0

    def test_first_linear_step_fails_skips_rest(self) -> None:
        executor = ChainExecutor(_linear_chain())
        wave = executor.next_wave()
        assert wave is not None

        executor.record_outcome(StepOutcome(step_id="a", status=StepStatus.FAILED, error="fail"))

        assert executor.next_wave() is None
        assert executor.chain.status == ChainStatus.FAILED

        summary = executor.summary()
        assert summary["failed"] == 1
        assert summary["skipped"] == 2
        assert summary["completed"] == 0


class TestUnknownStepOutcome:
    """Edge case: recording outcome for a step_id that doesn't exist."""

    def test_record_outcome_unknown_step_raises(self) -> None:
        executor = ChainExecutor(_parallel_chain())
        with pytest.raises(KeyError, match="Unknown step"):
            executor.record_outcome(StepOutcome(step_id="nonexistent", status=StepStatus.COMPLETED))


class TestPlanSerializationRoundtrip:
    """Edge case: plan to_dict produces consistent, complete output."""

    def test_plan_to_dict_roundtrip_fields(self) -> None:
        executor = ChainExecutor(_diamond_chain())
        plan = executor.plan()
        d = plan.to_dict()

        assert d["chain_id"] == "diamond"
        assert d["total_steps"] == 4
        assert d["max_parallelism"] == 2
        assert len(d["waves"]) == 3

        # Verify each wave has correct structure
        for i, wave_dict in enumerate(d["waves"]):
            assert wave_dict["wave_index"] == i
            assert "dispatches" in wave_dict
            for dispatch_dict in wave_dict["dispatches"]:
                assert "step_id" in dispatch_dict
                assert "work_order_id" in dispatch_dict
                assert "title" in dispatch_dict
                assert "wave_index" in dispatch_dict
                assert "predecessor_outputs" in dispatch_dict

    def test_plan_to_dict_idempotent(self) -> None:
        """Calling to_dict twice produces identical output."""
        executor = ChainExecutor(_linear_chain())
        plan = executor.plan()
        assert plan.to_dict() == plan.to_dict()

    def test_plan_to_dict_all_steps_present(self) -> None:
        """Every step in the chain appears in exactly one wave dispatch."""
        executor = ChainExecutor(_diamond_chain())
        d = executor.plan().to_dict()

        all_step_ids = []
        for wave_dict in d["waves"]:
            for dispatch_dict in wave_dict["dispatches"]:
                all_step_ids.append(dispatch_dict["step_id"])

        assert sorted(all_step_ids) == ["join", "left", "right", "root"]


class TestStepDispatchSerialization:
    def test_dispatch_to_dict(self) -> None:
        dispatch = StepDispatch(
            step_id="a",
            work_order_id="wo-a",
            title="Step A",
            wave_index=0,
        )
        d = dispatch.to_dict()
        assert d["step_id"] == "a"
        assert d["wave_index"] == 0
        assert d["predecessor_outputs"] == {}

    def test_outcome_to_dict(self) -> None:
        outcome = StepOutcome(
            step_id="a",
            status=StepStatus.COMPLETED,
            branch="feat/a",
            changed_files=["mod.py"],
            elapsed_seconds=45.2,
        )
        d = outcome.to_dict()
        assert d["status"] == "completed"
        assert d["branch"] == "feat/a"
        assert d["elapsed_seconds"] == 45.2
