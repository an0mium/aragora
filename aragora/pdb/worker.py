"""In-process brief-generation worker.

A :class:`BriefGenerationWorker` singleton owns the Protocol B pipeline
invocations that the HTTP layer schedules on-demand. Key invariants:

- **Bounded concurrency.** ``asyncio.Semaphore(max_concurrent)``
  caps how many briefs can run at once.
- **Dedupe.** For a given ``(repo, pr_number, head_sha)`` tuple, only
  one task is accepted at a time. A concurrent ``POST`` for the same
  tuple returns the existing task without enqueuing a second one.
- **Cancel-safe.** :meth:`BriefGenerationWorker.cancel` triggers
  ``Task.cancel()`` which interrupts awaitable boundaries. The
  underlying ``run_protocol_b`` call runs in a thread executor; we
  treat cancellation as best-effort while that call is in flight and
  record ``failed`` with ``failed_phase="cancelled"``.
- **Self-owned event loop.** The worker starts a dedicated background
  thread with its own asyncio loop so the sync HTTP handler can hand
  work off without blocking. Shutdown joins the thread.
- **Storage is truth.** On every transition the worker updates the
  :mod:`aragora.pdb.storage` lifecycle record. If the process dies
  mid-run the disk state still reflects where things stopped.

The worker is intentionally minimal — no retry, no scheduling. PR 4 /
successor PRs layer retry and batch semantics on top.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Callable

from aragora.pdb import storage
from aragora.pdb.brief_state import BriefLifecycleState
from aragora.pdb.protocol import (
    PDBExecutionInput,
    PDBExecutionResult,
    PDBExecutionStatus,
    ProviderInvoker,
    run_protocol_b,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AlreadyRunningError",
    "BriefGenerationWorker",
    "JobKey",
    "JobRequest",
    "get_worker",
    "reset_worker",
    "set_worker",
]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JobKey:
    """Dedupe key for a single brief-generation job."""

    repo: str
    pr_number: int
    head_sha: str


@dataclass(frozen=True, slots=True)
class JobRequest:
    """Everything needed to run one Protocol B execution.

    ``invoker_factory`` is a zero-arg callable that returns a
    :class:`ProviderInvoker`. The factory pattern lets the worker run
    on its own thread without sharing mutable state with the HTTP
    layer.
    """

    key: JobKey
    input: PDBExecutionInput
    invoker_factory: Callable[[], ProviderInvoker]
    panel_models: tuple[str, ...] = field(default_factory=tuple)


class AlreadyRunningError(RuntimeError):
    """Raised by :meth:`BriefGenerationWorker.submit` for duplicate keys.

    Carries the current :class:`BriefLifecycleState` so the HTTP layer
    can emit a 409 with useful context.
    """

    def __init__(self, key: JobKey, state: BriefLifecycleState) -> None:
        self.key = key
        self.state = state
        super().__init__(
            f"generation already active for repo={key.repo!r} pr={key.pr_number} "
            f"head_sha={key.head_sha[:12]!r} (state={state.value})"
        )


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class BriefGenerationWorker:
    """Bounded-concurrency worker for on-demand Protocol B runs."""

    def __init__(self, *, max_concurrent: int = 5) -> None:
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive")
        self._max_concurrent = max_concurrent
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._lock = threading.Lock()
        # Dedupe map: JobKey → asyncio.Task (live on the worker loop)
        self._tasks: dict[JobKey, asyncio.Task[PDBExecutionResult]] = {}
        self._semaphore: asyncio.Semaphore | None = None
        self._shutting_down = False

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Start the background event-loop thread if not already running."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._ready.clear()
            self._shutting_down = False
            self._thread = threading.Thread(
                target=self._run_loop,
                name="pdb-brief-worker",
                daemon=True,
            )
            self._thread.start()
        self._ready.wait(timeout=5)
        if not self._ready.is_set():
            raise RuntimeError("BriefGenerationWorker failed to start its loop")

    def stop(self, *, timeout: float = 5.0) -> None:
        """Stop the worker loop, cancelling in-flight tasks."""
        with self._lock:
            loop = self._loop
            thread = self._thread
            self._shutting_down = True
        if loop is None or thread is None:
            return

        def _cancel_all() -> None:
            for task in list(self._tasks.values()):
                task.cancel()

        try:
            loop.call_soon_threadsafe(_cancel_all)
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            # Loop may have already stopped.
            pass
        thread.join(timeout=timeout)
        with self._lock:
            self._loop = None
            self._thread = None
            self._tasks.clear()
            self._semaphore = None

    # -- submission -------------------------------------------------------

    def submit(self, request: JobRequest) -> "Future[PDBExecutionResult]":
        """Accept a new job. Returns a :class:`concurrent.futures.Future`.

        The future resolves when the generation task terminates (success
        or failure). Callers generally don't block on it — the storage
        layer is the authoritative source for state — but the future is
        useful for tests.

        Raises
        ------
        AlreadyRunningError:
            If an in-flight task already owns the same
            ``(repo, pr_number, head_sha)`` key.
        """
        self.start()
        loop = self._loop
        if loop is None:
            raise RuntimeError("worker loop is not running")

        def _schedule() -> asyncio.Task[PDBExecutionResult]:
            if request.key in self._tasks:
                current = storage.get_state(request.key.pr_number, request.key.head_sha)
                raise AlreadyRunningError(request.key, current)
            task = loop.create_task(self._run_job(request))
            self._tasks[request.key] = task
            task.add_done_callback(lambda _t: self._tasks.pop(request.key, None))
            return task

        # Dispatch scheduling onto the worker loop and wait for the
        # coroutine-safe result.
        sched_future: Future[asyncio.Task[PDBExecutionResult]] = Future()

        def _runner() -> None:
            try:
                sched_future.set_result(_schedule())
            except BaseException as exc:  # noqa: BLE001
                sched_future.set_exception(exc)

        loop.call_soon_threadsafe(_runner)
        task = sched_future.result(timeout=5)
        return asyncio.run_coroutine_threadsafe(
            _await_task(task),
            loop,
        )

    def cancel(self, key: JobKey) -> bool:
        """Cancel the job for ``key`` if present.

        Returns ``True`` if a task was cancelled, ``False`` otherwise.
        The cancellation is best-effort — if the job is inside
        ``run_protocol_b`` on the executor thread the actual work may
        continue until it returns; the storage layer records the
        cancel via :func:`_finalize_cancel`.
        """
        with self._lock:
            loop = self._loop
        if loop is None:
            return False
        cancel_done: Future[bool] = Future()

        def _do_cancel() -> None:
            task = self._tasks.get(key)
            if task is None:
                cancel_done.set_result(False)
                return
            task.cancel()
            cancel_done.set_result(True)

        loop.call_soon_threadsafe(_do_cancel)
        try:
            return cancel_done.result(timeout=2)
        except TimeoutError:
            return False

    def is_active(self, key: JobKey) -> bool:
        """Return ``True`` if a live task exists for ``key``."""
        return key in self._tasks

    def active_count(self) -> int:
        return len(self._tasks)

    # -- loop thread ------------------------------------------------------

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            finally:
                loop.close()

    # -- task body --------------------------------------------------------

    async def _run_job(self, request: JobRequest) -> PDBExecutionResult:
        sem = self._semaphore
        if sem is None:
            raise RuntimeError("semaphore must be created on loop startup")
        key = request.key
        # Track whether mark_running has run so cancel/exception paths
        # know the on-disk state they need to transition from.
        marked_running = False
        try:
            async with sem:
                state = await asyncio.to_thread(storage.get_state, key.pr_number, key.head_sha)
                if state != BriefLifecycleState.QUEUED:
                    logger.warning(
                        "pdb.worker: job %s launched without queued state (saw %s)",
                        key,
                        state.value,
                    )
                    # Only mark_failed when the source state permits the
                    # queued|running → failed transition.
                    if state in (
                        BriefLifecycleState.QUEUED,
                        BriefLifecycleState.RUNNING,
                    ):
                        await asyncio.to_thread(
                            storage.mark_failed,
                            key.pr_number,
                            key.head_sha,
                            f"worker found state={state.value}, expected queued",
                            "pre_run",
                            0.0,
                        )
                    return _empty_failed_result()

                await asyncio.to_thread(
                    storage.mark_running,
                    key.pr_number,
                    key.head_sha,
                    "findings",
                )
                marked_running = True

                invoker = request.invoker_factory()

                # run_protocol_b is synchronous — hand it to the default
                # thread executor so Task.cancel can at least unblock
                # the coroutine even if the sync work keeps going.
                result: PDBExecutionResult = await asyncio.to_thread(
                    run_protocol_b,
                    request.input,
                    invoker=invoker,
                )

                await _finalize_result(request.key, result)
                return result
        except asyncio.CancelledError:
            # Best-effort mark_failed — only valid from running/queued.
            # Swallow any downstream storage errors so cancel always
            # propagates to the submitter.
            try:
                if marked_running:
                    current = await asyncio.to_thread(
                        storage.get_state, key.pr_number, key.head_sha
                    )
                    if current == BriefLifecycleState.RUNNING:
                        await asyncio.to_thread(
                            storage.mark_failed,
                            key.pr_number,
                            key.head_sha,
                            "generation cancelled",
                            "cancelled",
                            0.0,
                        )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "pdb.worker: failed to record cancel for pr=%s sha=%s",
                    key.pr_number,
                    key.head_sha,
                )
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "pdb.worker: run_protocol_b failed for pr=%s sha=%s",
                key.pr_number,
                key.head_sha,
            )
            try:
                current = await asyncio.to_thread(storage.get_state, key.pr_number, key.head_sha)
                if current in (BriefLifecycleState.QUEUED, BriefLifecycleState.RUNNING):
                    await asyncio.to_thread(
                        storage.mark_failed,
                        key.pr_number,
                        key.head_sha,
                        str(exc),
                        "run_protocol_b_exception",
                        0.0,
                    )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "pdb.worker: failed to record exception state for pr=%s sha=%s",
                    key.pr_number,
                    key.head_sha,
                )
            return _empty_failed_result()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _await_task(task: asyncio.Task[PDBExecutionResult]) -> PDBExecutionResult:
    return await task


async def _finalize_result(key: JobKey, result: PDBExecutionResult) -> None:
    """Persist the terminal lifecycle state for a completed job."""

    cost = float(result.actual_cost_usd)

    if result.status == PDBExecutionStatus.SUCCESS:
        if result.brief is None:
            await asyncio.to_thread(
                storage.mark_failed,
                key.pr_number,
                key.head_sha,
                "protocol_b success without brief payload",
                "run_protocol_b_missing_brief",
                cost,
            )
            return
        await asyncio.to_thread(
            storage.mark_ready,
            key.pr_number,
            key.head_sha,
            result.brief.to_dict(),
            None,
        )
        return

    if result.status == PDBExecutionStatus.DEGRADED:
        # A degraded outcome still has a brief — record it as ready so
        # the founder can consult it. Degrade context survives inside
        # the brief payload's ``cost_estimate`` / packet metadata.
        if result.brief is None:
            await asyncio.to_thread(
                storage.mark_failed,
                key.pr_number,
                key.head_sha,
                "protocol_b degraded without brief payload",
                "run_protocol_b_missing_brief",
                cost,
            )
            return
        await asyncio.to_thread(
            storage.mark_ready,
            key.pr_number,
            key.head_sha,
            result.brief.to_dict(),
            None,
        )
        return

    if result.status == PDBExecutionStatus.BUDGET_EXCEEDED:
        await asyncio.to_thread(
            storage.mark_failed,
            key.pr_number,
            key.head_sha,
            result.failure_reason or "budget exhausted",
            "budget_exhausted",
            cost,
        )
        return

    # FAILED_CLOSED
    await asyncio.to_thread(
        storage.mark_failed,
        key.pr_number,
        key.head_sha,
        result.failure_reason or "protocol_b failed closed",
        "fail_closed",
        cost,
    )


def _empty_failed_result() -> PDBExecutionResult:
    """Return a minimal failed :class:`PDBExecutionResult` for worker errors."""
    from aragora.pdb.budget import PDBBudgetDecision, PDBBudgetStatus
    from aragora.review.provider_slots import ProviderSlotAvailabilitySummary
    from aragora.swarm.pr_review_protocol import PRReviewBinding, PRReviewProtocolPacket

    binding = PRReviewBinding(repo="", pr_number=0, base_sha="", head_sha="")
    return PDBExecutionResult(
        status=PDBExecutionStatus.FAILED_CLOSED,
        packet=PRReviewProtocolPacket(
            protocol_version="pr_review_protocol.v1",
            status="failed_closed",
            binding=binding,
            review_roles=[],
            provider_slots=[],
            availability_summary=ProviderSlotAvailabilitySummary(total_slots=0, resolved_slots=0),
            recommendation_class="needs_human_attention",
            recommendation_reason="worker-level failure",
            confidence=0.0,
            confidence_basis="failed_closed",
            dissent_summary="",
            dissenting_views=[],
            validation_summary={},
            top_findings=[],
            cost_estimate={},
        ),
        brief=None,
        budget_decision=PDBBudgetDecision(
            status=PDBBudgetStatus.ALLOWED,
            total_estimated_usd=0.0,
            per_brief_cap_usd=0.0,
            per_day_cap_usd=0.0,
            per_day_spent_before_usd=0.0,
            per_day_remaining_before_usd=0.0,
            funded_slots=(),
            dropped_slots=(),
            slot_estimates_usd={},
            synthesis_estimate_usd=0.0,
            reason="worker-level failure",
            binding_cap=None,
        ),
        active_roster=(),
        missing_slots=(),
        degrade_reasons=(),
        failure_reason="worker-level failure",
        findings_by_slot={},
        critiques_by_slot={},
        synthesis=None,
        resolutions=(),
        availability_summary=ProviderSlotAvailabilitySummary(total_slots=0, resolved_slots=0),
        actual_cost_usd=0.0,
    )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


_worker_lock = threading.Lock()
_worker_singleton: BriefGenerationWorker | None = None


def get_worker(*, max_concurrent: int | None = None) -> BriefGenerationWorker:
    """Return the process-wide worker singleton, creating it if needed."""
    global _worker_singleton
    with _worker_lock:
        if _worker_singleton is None:
            import os

            mc = max_concurrent
            if mc is None:
                env = os.environ.get("ARAGORA_PDB_MAX_CONCURRENT_BRIEFS")
                if env:
                    try:
                        mc = int(env)
                    except ValueError:
                        mc = 5
                else:
                    mc = 5
            _worker_singleton = BriefGenerationWorker(max_concurrent=mc)
        return _worker_singleton


def set_worker(worker: BriefGenerationWorker | None) -> None:
    """Replace (or clear) the process-wide worker singleton.

    Intended for tests that want to inject a worker with specific
    concurrency settings or stop the singleton between runs.
    """
    global _worker_singleton
    with _worker_lock:
        if _worker_singleton is not None and _worker_singleton is not worker:
            _worker_singleton.stop()
        _worker_singleton = worker


def reset_worker() -> None:
    """Stop and clear the process-wide worker singleton.

    Tests call this in a fixture so each test gets a fresh loop.
    """
    set_worker(None)
