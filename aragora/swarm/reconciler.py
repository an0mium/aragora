"""Periodic reconciler for supervisor-backed swarm runs.

Keeps a supervised Codex/Claude swarm moving by topping up leases,
dispatching ready workers, collecting finished results, and syncing
pending coordination artifacts into the global work queue.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from aragora.swarm.supervisor import SupervisorRun, SwarmSupervisor

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SwarmReconcilerConfig:
    """Configuration for periodic swarm reconciliation."""

    interval_seconds: float = 5.0
    max_ticks: int | None = None
    sync_pending_queue: bool = True


class SwarmReconciler:
    """Periodic reconciler for supervised swarm runs."""

    def __init__(
        self,
        repo_root: Path | None = None,
        *,
        supervisor: SwarmSupervisor | None = None,
        config: SwarmReconcilerConfig | None = None,
    ) -> None:
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.supervisor = supervisor or SwarmSupervisor(repo_root=self.repo_root)
        self.config = config or SwarmReconcilerConfig()

    async def tick_run(self, run_id: str) -> SupervisorRun:
        """Advance one run by one reconciliation tick."""
        self.supervisor.refresh_run(run_id)
        await self.supervisor.dispatch_workers(run_id)
        completed = await self.supervisor.collect_finished_results(run_id)
        if completed:
            self.supervisor.refresh_run(run_id)
            await self.supervisor.dispatch_workers(run_id)

        if self.config.sync_pending_queue:
            await self.supervisor.store.sync_pending_work_queue()

        record = self.supervisor.store.get_supervisor_run(run_id)
        if record is None:
            raise KeyError(f"Unknown supervisor run: {run_id}")
        return SupervisorRun.from_record(record)

    async def tick_open_runs(self, *, limit: int = 20) -> list[SupervisorRun]:
        """Advance all open runs by one tick."""
        runs = self.supervisor.status_summary(limit=limit, refresh_scaling=False).get("runs", [])
        refreshed: list[SupervisorRun] = []
        for item in runs:
            if not isinstance(item, dict):
                continue
            run_id = str(item.get("run_id", "")).strip()
            if not run_id:
                continue
            refreshed.append(await self.tick_run(run_id))
        return refreshed

    async def watch_run(
        self,
        run_id: str,
        *,
        interval_seconds: float | None = None,
        max_ticks: int | None = None,
        force_collect_on_max_ticks: bool = False,
    ) -> SupervisorRun:
        """Reconcile a run until it reaches a stable stop condition.

        When ``force_collect_on_max_ticks`` is enabled, exhausting
        ``max_ticks`` becomes a hard stop for bounded runs: any remaining
        dispatched workers are force-collected once so supervisor salvage can
        preserve concrete work instead of abandoning dirty worktrees.
        """
        interval = (
            self.config.interval_seconds if interval_seconds is None else max(0.1, interval_seconds)
        )
        tick_limit = self.config.max_ticks if max_ticks is None else max_ticks
        # Count completed sleep/reconcile intervals after the initial tick.
        # This preserves the immediate first reconciliation and still allows
        # the final boundary tick before max_ticks exhaustion.
        ticks = 0
        run = await self.tick_run(run_id)
        while not self._should_stop(run):
            if tick_limit is not None and ticks >= tick_limit:
                if force_collect_on_max_ticks:
                    run = await self._force_collect_dispatched(run_id)
                break
            await asyncio.sleep(interval)
            run = await self.tick_run(run_id)
            ticks += 1

        return run

    async def _force_collect_dispatched(self, run_id: str) -> SupervisorRun:
        """Force-collect dispatched work orders after bounded max_ticks exhaustion.

        This is intentionally opt-in from ``watch_run()`` so generic
        long-running supervisor sessions do not terminate active work simply
        because a caller stopped waiting. Bounded dispatch paths can enable it
        to treat ``max_ticks`` as a hard time cap and run one final salvage
        collection pass before returning.
        """
        record = self.supervisor.store.get_supervisor_run(run_id)
        if record is None:
            raise KeyError(f"Unknown supervisor run: {run_id}")

        work_orders = [dict(item) for item in record.get("work_orders", [])]
        dispatched = [item for item in work_orders if str(item.get("status", "")) == "dispatched"]
        if not dispatched:
            return SupervisorRun.from_record(record)

        logger.warning(
            "max_ticks exhausted for run %s with %d dispatched work orders — "
            "force-collecting with salvage",
            run_id,
            len(dispatched),
        )

        # Kill workers and collect — this invokes the supervisor's existing
        # no-progress timeout path which handles salvage auto-commit.
        for item in dispatched:
            await self.supervisor._kill_worker(item)

        # One final collect pass picks up salvaged commits from killed workers.
        await self.supervisor.collect_finished_results(run_id)
        self.supervisor.refresh_run(run_id)

        final_record = self.supervisor.store.get_supervisor_run(run_id)
        if final_record is None:
            raise KeyError(f"Unknown supervisor run: {run_id}")
        return SupervisorRun.from_record(final_record)

    @staticmethod
    def _should_stop(run: SupervisorRun) -> bool:
        statuses = {str(item.get("status", "")).strip() for item in run.work_orders}
        if run.status == "completed":
            return True
        if run.status == "needs_human":
            return True
        # Only statuses that represent real forward progress keep the
        # reconciler alive.  waiting_conflict and waiting_resource are
        # dead-end blockers — _derive_status escalates them to needs_human
        # when no forward-progress path remains.
        forward_progress_statuses = {"queued", "leased", "dispatched"}
        return not (statuses & forward_progress_statuses)
