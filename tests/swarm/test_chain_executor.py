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
