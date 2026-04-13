"""Chain executor: wave-based execution of work chain DAGs.

Consumes a validated WorkChain and drives execution wave by wave. Each wave
dispatches its ready steps, waits for completion, then advances to the next
wave. Failed steps cause dependent steps to be skipped automatically.

The executor does not launch workers directly — it produces dispatch
instructions that the supervisor can consume. This keeps the boundary clean:
chain_executor plans, supervisor executes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timezone
from typing import Any

from aragora.swarm.work_chain import (
    ChainStatus,
    ExecutionWave,
    StepStatus,
    WorkChain,
)

logger = logging.getLogger(__name__)
UTC = timezone.utc


@dataclass(slots=True)
class StepDispatch:
    """Instruction to dispatch one chain step to a worker."""

    step_id: str
    work_order_id: str
    title: str
    wave_index: int
    predecessor_outputs: dict[str, StepOutcome] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "work_order_id": self.work_order_id,
            "title": self.title,
            "wave_index": self.wave_index,
            "predecessor_outputs": {k: v.to_dict() for k, v in self.predecessor_outputs.items()},
        }


@dataclass(slots=True)
class StepOutcome:
    """Result of one chain step execution."""

    step_id: str
    status: StepStatus
    branch: str = ""
    changed_files: list[str] = field(default_factory=list)
    commit_shas: list[str] = field(default_factory=list)
    error: str = ""
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "status": self.status.value,
            "branch": self.branch,
            "changed_files": list(self.changed_files),
            "commit_shas": list(self.commit_shas),
            "error": self.error,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass
class ChainExecutionPlan:
    """Complete execution plan for a work chain."""

    chain_id: str
    waves: list[WaveDispatchPlan]
    total_steps: int
    max_parallelism: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "waves": [wave.to_dict() for wave in self.waves],
            "total_steps": self.total_steps,
            "max_parallelism": self.max_parallelism,
        }


@dataclass
class WaveDispatchPlan:
    """Dispatch plan for one execution wave."""

    wave_index: int
    dispatches: list[StepDispatch]

    def to_dict(self) -> dict[str, Any]:
        return {
            "wave_index": self.wave_index,
            "dispatches": [d.to_dict() for d in self.dispatches],
        }


class ChainExecutor:
    """Drives wave-by-wave execution of a WorkChain.

    Usage::

        executor = ChainExecutor(chain)
        plan = executor.plan()

        for wave_plan in plan.waves:
            for dispatch in wave_plan.dispatches:
                # Submit dispatch.work_order_id to supervisor
                ...
            # Collect results
            for dispatch in wave_plan.dispatches:
                executor.record_outcome(StepOutcome(...))

        summary = executor.summary()
    """

    def __init__(self, chain: WorkChain) -> None:
        if not chain.steps:
            raise ValueError("Cannot execute an empty chain")
        self._chain = chain
        self._outcomes: dict[str, StepOutcome] = {}

    @property
    def chain(self) -> WorkChain:
        return self._chain

    def plan(self) -> ChainExecutionPlan:
        """Generate the full execution plan without executing anything."""
        wave_plans: list[WaveDispatchPlan] = []
        max_parallelism = 0

        for wave in self._chain.waves:
            dispatches = self._dispatches_for_wave(wave)
            wave_plans.append(WaveDispatchPlan(wave_index=wave.wave_index, dispatches=dispatches))
            max_parallelism = max(max_parallelism, len(dispatches))

        return ChainExecutionPlan(
            chain_id=self._chain.chain_id,
            waves=wave_plans,
            total_steps=len(self._chain.steps),
            max_parallelism=max_parallelism,
        )

    def next_wave(self) -> WaveDispatchPlan | None:
        """Return the next wave of dispatches, or None if chain is done.

        Skips steps whose predecessors have failed.
        """
        self._propagate_failures()

        ready = self._chain.ready_steps()
        if not ready:
            return None

        # Determine wave index from the chain's wave structure
        ready_ids = {step.step_id for step in ready}
        wave_index = 0
        for wave in self._chain.waves:
            if any(sid in ready_ids for sid in wave.step_ids):
                wave_index = wave.wave_index
                break

        dispatches: list[StepDispatch] = []
        for step in ready:
            predecessor_outputs: dict[str, StepOutcome] = {}
            for dep_id in step.depends_on:
                if dep_id in self._outcomes:
                    predecessor_outputs[dep_id] = self._outcomes[dep_id]

            dispatches.append(
                StepDispatch(
                    step_id=step.step_id,
                    work_order_id=step.work_order_id,
                    title=step.title,
                    wave_index=wave_index,
                    predecessor_outputs=predecessor_outputs,
                )
            )
            self._chain.mark_step(step.step_id, StepStatus.RUNNING)

        return WaveDispatchPlan(wave_index=wave_index, dispatches=dispatches)

    def record_outcome(self, outcome: StepOutcome) -> None:
        """Record the result of a step execution."""
        self._outcomes[outcome.step_id] = outcome
        self._chain.mark_step(outcome.step_id, outcome.status)
        logger.info("Chain step %s finished: %s", outcome.step_id, outcome.status.value)

    def _propagate_failures(self) -> None:
        """Skip steps whose predecessors have failed or been skipped.

        Runs transitively: if A fails, B (depends on A) is skipped, then
        C (depends on B) is also skipped.
        """
        changed = True
        while changed:
            changed = False
            blocked_ids = {
                step.step_id
                for step in self._chain.steps
                if step.status in (StepStatus.FAILED, StepStatus.SKIPPED)
            }
            if not blocked_ids:
                return

            for step in self._chain.steps:
                if step.status != StepStatus.PENDING:
                    continue
                blocker = next((dep for dep in step.depends_on if dep in blocked_ids), None)
                if blocker is not None:
                    self._chain.mark_step(step.step_id, StepStatus.SKIPPED)
                    logger.info(
                        "Skipping step %s: predecessor %s blocked",
                        step.step_id,
                        blocker,
                    )
                    changed = True

    def _dispatches_for_wave(self, wave: ExecutionWave) -> list[StepDispatch]:
        """Build dispatch instructions for a wave (used by plan())."""
        dispatches: list[StepDispatch] = []
        for step_id in wave.step_ids:
            step = self._chain.step_by_id(step_id)
            if step is None:
                continue
            dispatches.append(
                StepDispatch(
                    step_id=step.step_id,
                    work_order_id=step.work_order_id,
                    title=step.title,
                    wave_index=wave.wave_index,
                )
            )
        return dispatches

    def is_complete(self) -> bool:
        return self._chain.status in (ChainStatus.COMPLETED, ChainStatus.FAILED)

    def summary(self) -> dict[str, Any]:
        """Return a summary of the chain execution."""
        completed = sum(1 for s in self._chain.steps if s.status == StepStatus.COMPLETED)
        failed = sum(1 for s in self._chain.steps if s.status == StepStatus.FAILED)
        skipped = sum(1 for s in self._chain.steps if s.status == StepStatus.SKIPPED)
        return {
            "chain_id": self._chain.chain_id,
            "chain_status": self._chain.status.value,
            "total_steps": len(self._chain.steps),
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "remaining": len(self._chain.steps) - completed - failed - skipped,
            "outcomes": {k: v.to_dict() for k, v in self._outcomes.items()},
        }


__all__ = [
    "ChainExecutionPlan",
    "ChainExecutor",
    "StepDispatch",
    "StepOutcome",
    "WaveDispatchPlan",
]
