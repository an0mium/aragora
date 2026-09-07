"""End-to-end lane cycle: conductor dispatch followed by supervisor launch.

``lane_conductor`` intentionally stays focused on assigning and dispatching
work orders. This module closes the one-shot operational gap: a dry-run previews
the orders that would be launched, while execute mode claims lanes, writes work
orders, and drains exactly those newly dispatched orders through
``lane_supervisor``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from aragora.swarm.lane_conductor import (
    DEFAULT_TARGET_AGENT,
    ConductorPass,
    WorkOrderSpec,
    default_claim,
    default_dispatch,
    run_pass,
)
from aragora.swarm.lane_dispatcher import DEFAULT_MAX_WORKERS, default_session_id
from aragora.swarm.lane_supervisor import (
    DrainResult,
    LaunchFn,
    drain_once,
)


@dataclass
class LaneCycleResult:
    conductor: ConductorPass
    supervisor: DrainResult
    executed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "conductor": self.conductor.to_dict(),
            "supervisor": self.supervisor.to_dict(),
            "executed": self.executed,
            "reason": self.reason,
        }


def _preview_supervisor_launches(
    work_orders: Sequence[WorkOrderSpec], *, max_launches: int
) -> DrainResult:
    cap = max(0, int(max_launches))
    result = DrainResult()
    for work_order in work_orders:
        if len(result.launched) < cap:
            result.launched.append(work_order.work_order_id)
        else:
            result.deferred.append(work_order.work_order_id)
    result.reason = (
        f"[dry-run] {len(result.launched)} of {len(work_orders)} planned order(s) "
        f"would launch after conductor dispatch; {len(result.deferred)} deferred "
        f"(max_launches={cap})"
    )
    return result


def _dispatched_work_order_ids(paths: Sequence[str]) -> set[str]:
    return {Path(path).stem for path in paths if str(path).strip()}


def run_cycle(
    *,
    repo: str,
    root: Path,
    fetch_candidates: Callable[[str], Sequence[dict[str, Any]]],
    fetch_live_claims: Callable[[str, Sequence[dict[str, Any]]], dict[int, str]],
    launch_fn: LaunchFn | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_launches: int | None = None,
    target_agent: str = DEFAULT_TARGET_AGENT,
    execute: bool = False,
    claim_fn: Callable[[WorkOrderSpec], bool | None] | None = None,
    dispatch_fn: Callable[[WorkOrderSpec], str] | None = None,
    session_id_for: Callable[[int], str] = default_session_id,
    now: Callable[[], str] | None = None,
) -> LaneCycleResult:
    """Run one conductor->supervisor cycle.

    Dry-run mode is side-effect free: it does not claim, dispatch, drain, or
    launch. Execute mode first runs the conductor side effects, then drains only
    the work-order ids that this cycle dispatched. That filter is deliberate:
    old pending backlog must not be launched as a surprise side effect of asking
    the conductor to advance the current live queue.
    """
    launch_cap = max_workers if max_launches is None else max_launches
    checked_launch_fn = launch_fn
    if execute and checked_launch_fn is None:
        raise ValueError("launch_fn is required when execute=True")

    conductor = run_pass(
        repo=repo,
        fetch_candidates=fetch_candidates,
        fetch_live_claims=fetch_live_claims,
        max_workers=max_workers,
        target_agent=target_agent,
        execute=execute,
        claim_fn=claim_fn or (lambda wo: default_claim(wo, repo_root=root)),
        dispatch_fn=dispatch_fn or (lambda wo: default_dispatch(wo, repo_root=root)),
        session_id_for=session_id_for,
        now=now,
    )

    if not execute:
        supervisor = _preview_supervisor_launches(conductor.work_orders, max_launches=launch_cap)
        return LaneCycleResult(
            conductor=conductor,
            supervisor=supervisor,
            executed=False,
            reason=(
                "dry-run: no claims written, no work orders dispatched, "
                "no supervisor drain, no workers launched"
            ),
        )

    dispatched_ids = _dispatched_work_order_ids(conductor.dispatched)
    if not dispatched_ids:
        supervisor = DrainResult(reason="no newly dispatched work orders to launch")
    else:
        execute_launch_fn = cast(LaunchFn, checked_launch_fn)
        supervisor = drain_once(
            root=root,
            launch_fn=execute_launch_fn,
            max_launches=launch_cap,
            work_order_ids=dispatched_ids,
            now=now,
        )
    return LaneCycleResult(
        conductor=conductor,
        supervisor=supervisor,
        executed=True,
        reason="execute: conductor dispatched work orders and supervisor drained new orders",
    )


__all__ = ["LaneCycleResult", "run_cycle"]
