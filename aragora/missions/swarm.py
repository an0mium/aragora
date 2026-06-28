"""Swarm worker loop — the pheromone wired to the gate.

One worker repeatedly: atomic-claims a unit (``select_for``), runs the merge-gate
``dispatch``, and writes the *outcome* back to the shared environment — ``done`` on
success, a **park** constraint after repeated blocks. The merge-quorum gate stays
the only thing that says "yes" (propose = swarm, accept = gate, the
FunSearch/AlphaEvolve split); the ledger remembers failures **as data**, so the
whole swarm escapes a treadmill without any prompt self-editing.

``run_worker`` is process-agnostic: run one per thread or per process against the
same ``state_path``/``ledger_path`` and they self-partition. All cross-worker
mutation goes through the file-locked ledger, so workers never write
``MissionState`` — it stays a static backlog that only ``reconcile_from_ledger``
folds the swarm's results back into, from a single writer.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .ledger import DEFAULT_LEASE_TTL, Ledger, select_for
from .orchestrator import Dispatch, Handoff
from .state import MissionState, Status, mission_owner_lock

logger = logging.getLogger(__name__)


class _LeaseHeartbeat:
    """Keep a *live* worker's lease fresh while a long dispatch runs.

    The lease TTL exists so a *dead* worker's claim evaporates and the unit becomes
    reclaimable. But a live worker running a legitimately long dispatch (e.g.
    collecting heterogeneous-model quorum evidence, minutes) would let its own lease
    expire, and another worker could then claim and double-dispatch the same unit
    (claude's [P2]). A background thread re-claims (refreshes ``claimed_at``) every
    ``ttl/3`` until the dispatch returns, so a live worker's lease never lapses; if
    the worker process dies, the thread dies with it and the TTL fallback still frees
    the unit. The heartbeat itself never raises into the worker, but it records
    lost ownership so the caller can fail closed after dispatch returns instead
    of treating a stale result as support.
    """

    def __init__(self, ledger: Ledger, unit: str, worker_id: str, ttl: float) -> None:
        self._ledger = ledger
        self._unit = unit
        self._worker_id = worker_id
        self._ttl = ttl
        self._interval = ttl / 3 if ttl > 0 else DEFAULT_LEASE_TTL / 3
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lost_reason: str | None = None

    @property
    def lost_reason(self) -> str | None:
        return self._lost_reason

    def __enter__(self) -> _LeaseHeartbeat:
        self._thread = threading.Thread(target=self._beat, daemon=True)
        self._thread.start()
        return self

    def _beat(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                if not self._ledger.claim_actionable(
                    self._unit,
                    self._worker_id,
                    constraint_key=f"feature:{self._unit}",
                    ttl=self._ttl,
                ):
                    self._lost_reason = (
                        f"lease heartbeat for {self._unit} lost ownership or actionability"
                    )
                    logger.warning(self._lost_reason)
                    self._stop.set()
                    return
            except Exception:  # noqa: BLE001 - a heartbeat must never crash the worker
                logger.warning("lease heartbeat for %s failed; will retry next beat", self._unit)

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


@dataclass
class SwarmResult:
    """What one worker did this run."""

    worker_id: str
    done: list[str] = field(default_factory=list)
    parked: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)  # blocked attempts (incl. pre-park)
    lost_leases: list[str] = field(default_factory=list)


def run_worker(
    state_path: str | Path,
    ledger_path: str | Path,
    worker_id: str,
    dispatch: Dispatch,
    *,
    park_threshold: int = 2,
    max_units: int | None = None,
) -> SwarmResult:
    """Drain available units for ``worker_id`` until the queue is dry to it.

    A unit that blocks but hasn't hit ``park_threshold`` stays available, so a
    later attempt (by this or any worker) retries it; once the *shared* attempt
    count reaches the threshold it is parked and the swarm moves on. Convergence
    is guaranteed: attempts accumulate in the ledger, so a persistent blocker is
    parked after at most ``park_threshold`` total attempts across the swarm.

    Holds the *shared* side of :func:`mission_owner_lock` for its whole run: many
    workers coexist (shared), but an orchestrator (exclusive) cannot run against the
    same mission concurrently — so orchestrator-mode and swarm-mode never
    double-dispatch a feature. A long dispatch keeps its lease alive via
    :class:`_LeaseHeartbeat`.
    """
    with mission_owner_lock(state_path, exclusive=False):
        return _run_worker_fenced(
            state_path,
            ledger_path,
            worker_id,
            dispatch,
            park_threshold=park_threshold,
            max_units=max_units,
        )


def _run_worker_fenced(
    state_path: str | Path,
    ledger_path: str | Path,
    worker_id: str,
    dispatch: Dispatch,
    *,
    park_threshold: int,
    max_units: int | None,
) -> SwarmResult:
    state = MissionState.load(state_path)
    ledger = Ledger(ledger_path)
    res = SwarmResult(worker_id=worker_id)

    def abandon_lost_lease(unit: str) -> None:
        ledger.rollback_attempt(f"feature:{unit}")
        res.lost_leases.append(unit)

    n = 0
    while max_units is None or n < max_units:
        unit = select_for(state, ledger, worker_id)
        if unit is None:
            break
        n += 1

        # Count the attempt BEFORE dispatch so a *raising* dispatch is bounded too.
        attempts = ledger.bump_attempt(f"feature:{unit}")
        try:
            # Heartbeat keeps the lease fresh so a long dispatch isn't reclaimed.
            heartbeat = _LeaseHeartbeat(ledger, unit, worker_id, DEFAULT_LEASE_TTL)
            with heartbeat:
                handoff = dispatch(state.get(unit))
        except (
            Exception
        ) as exc:  # dispatch is an external callback — may raise anything  # noqa: BLE001
            handoff = Handoff(success=False, blocked_reason=f"dispatch raised: {exc!r}")

        if heartbeat.lost_reason:
            logger.warning(
                "worker %s abandoned stale result for %s after losing the lease: %s",
                worker_id,
                unit,
                heartbeat.lost_reason,
            )
            abandon_lost_lease(unit)
            continue

        # Discovered work is *advisory* in swarm mode (propose/accept boundary): the
        # swarm records what it found — discovered notes and proposed follow-ups —
        # but only the orchestrator+gate turn a note into executable work, so ledger
        # JSON can never inject a Feature. Recorded on *every* path, success or not.
        notes = list(handoff.discovered)
        notes += [f"follow-up proposed: {f.id} — {f.description}" for f in handoff.follow_ups]

        if handoff.success:
            # Atomic: done + notes + lease-release under ONE lock — no released-but-
            # not-done window for a concurrent claim_actionable to re-grab (the [P1]).
            if ledger.complete(unit, worker_id, discoveries=notes):
                res.done.append(unit)
            else:
                logger.warning(
                    "worker %s discarded success for %s after losing the lease",
                    worker_id,
                    unit,
                )
                abandon_lost_lease(unit)
                continue
            continue

        # Failure: record discoveries, optional park, and lease release as one owned
        # transaction. Terminal blocks (operator-gated / re-derive) park immediately.
        constraint_key = None
        constraint_reason = None
        parked = False
        if handoff.terminal or attempts >= park_threshold:
            kind = "terminal" if handoff.terminal else f"{attempts} blocks"
            constraint_key = f"feature:{unit}"
            constraint_reason = f"parked ({kind}): {handoff.blocked_reason}"
            parked = True
        if ledger.fail(
            unit,
            worker_id,
            discoveries=notes,
            constraint_key=constraint_key,
            constraint_reason=constraint_reason,
        ):
            res.blocked.append(unit)
            if parked:
                res.parked.append(unit)
                logger.info("worker %s parked %s (%s)", worker_id, unit, kind)
        else:
            logger.warning(
                "worker %s discarded failure for %s after losing the lease",
                worker_id,
                unit,
            )
            abandon_lost_lease(unit)
            continue

    return res


def reconcile_from_ledger(state_path: str | Path, ledger_path: str | Path) -> int:
    """Fold the swarm's ledger-recorded completions back into ``MissionState``.

    In swarm mode the ledger is the source of truth (so no locked MissionState is
    needed across workers); call this once afterward — from a single writer — to
    make ``MissionState`` consistent with what the swarm did: ledger ``done`` →
    COMPLETED, active **parks** → BLOCKED for any not-completed feature (so a
    parked IN_PROGRESS checkpoint is not reclaimed by the orchestrator path,
    preserving the anti-treadmill guarantee), and worker-recorded **discovered
    notes** folded into the matching feature's notes (so swarm mode never silently
    drops what it found). Discovered work stays advisory — reconcile never
    *creates* a feature from ledger data, so there is no path to inject
    gate-bypassing work. Returns the number of features whose status or notes
    changed.

    Holds the exclusive side of :func:`mission_owner_lock`, so it cannot run while
    an orchestrator or live swarm worker is driving the same mission. Workers touch
    the ledger, not ``MissionState``, but they still hold the shared side of the
    owner fence for the duration of ``run_worker``.
    """
    with mission_owner_lock(state_path):
        return _reconcile_locked(state_path, ledger_path)


def _reconcile_locked(state_path: str | Path, ledger_path: str | Path) -> int:
    state = MissionState.load(state_path)
    ledger = Ledger(ledger_path)
    done = ledger.done_units()
    n = 0
    for feat in state.features:
        if feat.id in done and feat.status != Status.COMPLETED:
            if _has_stale_validation_done(feat):
                note = "ledger done ignored because validation reopened this feature"
                if note not in feat.notes:
                    feat.notes = (feat.notes + "\n" if feat.notes else "") + note
                    n += 1
                continue
            feat.status = Status.COMPLETED
            _clear_validation_reopen_metadata(feat)
            n += 1
        elif feat.status != Status.COMPLETED and ledger.is_excluded(f"feature:{feat.id}"):
            # Never downgrade COMPLETED on a stale park, but do fold active parks over
            # PENDING or IN_PROGRESS so the orchestrator cannot reclaim a parked unit.
            reason = ledger.constraint_reason(f"feature:{feat.id}")
            if feat.status != Status.BLOCKED:
                feat.status = Status.BLOCKED
                n += 1
            if reason and reason not in feat.notes:  # keep operator context for handoff/debugging
                feat.notes = (feat.notes + "\n" if feat.notes else "") + f"BLOCKED (park): {reason}"
                n += 1

    # Fold discovered notes (advisory) into the matching feature. Never insert a
    # feature from ledger data — that stays the orchestrator+gate's job.
    for unit, notes in ledger.discoveries().items():
        try:
            feat = state.get(unit)
        except KeyError:
            continue
        for note in notes:
            stamp = f"discovered: {note}"
            if stamp not in feat.notes:
                feat.notes = (feat.notes + "\n" if feat.notes else "") + stamp
                n += 1
    if n:
        state.save(state_path)
    return n


def _has_stale_validation_done(feat) -> bool:
    return bool(
        feat.metadata.get("validation_reopened_by")
        and not feat.metadata.get("validation_reopened_ledger_done_invalidated")
    )


def _clear_validation_reopen_metadata(feat) -> None:
    for key in (
        "validation_reopened_by",
        "validation_reopened_reason",
        "validation_reopened_ledger_done_invalidated",
    ):
        feat.metadata.pop(key, None)
