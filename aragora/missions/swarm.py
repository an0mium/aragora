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
from dataclasses import dataclass, field
from pathlib import Path

from .ledger import Ledger, select_for
from .orchestrator import Dispatch, Handoff
from .state import MissionState, Status, mission_owner_lock

logger = logging.getLogger(__name__)


@dataclass
class SwarmResult:
    """What one worker did this run."""

    worker_id: str
    done: list[str] = field(default_factory=list)
    parked: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)  # blocked attempts (incl. pre-park)


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
    """
    state = MissionState.load(state_path)
    ledger = Ledger(ledger_path)
    res = SwarmResult(worker_id=worker_id)

    n = 0
    while max_units is None or n < max_units:
        unit = select_for(state, ledger, worker_id)
        if unit is None:
            break
        n += 1

        # Count the attempt BEFORE dispatch so a *raising* dispatch is bounded too.
        attempts = ledger.bump_attempt(f"feature:{unit}")
        try:
            handoff = dispatch(state.get(unit))
        except (
            Exception
        ) as exc:  # dispatch is an external callback — may raise anything  # noqa: BLE001
            handoff = Handoff(success=False, blocked_reason=f"dispatch raised: {exc!r}")

        # Discovered work is *advisory* in swarm mode (propose/accept boundary): the
        # swarm records what it found — discovered notes and proposed follow-ups —
        # but only the orchestrator+gate turn a note into executable work, so ledger
        # JSON can never inject a Feature. Recorded on *every* path, success or not.
        notes = list(handoff.discovered)
        notes += [f"follow-up proposed: {f.id} — {f.description}" for f in handoff.follow_ups]

        if handoff.success:
            # Atomic: done + notes + lease-release under ONE lock — no released-but-
            # not-done window for a concurrent claim_actionable to re-grab (the [P1]).
            ledger.complete(unit, worker_id, discoveries=notes)
            res.done.append(unit)
            continue

        # Failure: record the discoveries, free the lease so a retry can re-claim,
        # then park if the shared attempt budget is spent. Terminal blocks
        # (operator-gated / re-derive) park immediately.
        for note in notes:
            ledger.record_discovery(unit, note)
        ledger.release(unit, worker_id)
        res.blocked.append(unit)
        if handoff.terminal or attempts >= park_threshold:
            kind = "terminal" if handoff.terminal else f"{attempts} blocks"
            ledger.record_constraint(
                f"feature:{unit}", f"parked ({kind}): {handoff.blocked_reason}"
            )
            res.parked.append(unit)
            logger.info("worker %s parked %s (%s)", worker_id, unit, kind)

    return res


def reconcile_from_ledger(state_path: str | Path, ledger_path: str | Path) -> int:
    """Fold the swarm's ledger-recorded completions back into ``MissionState``.

    In swarm mode the ledger is the source of truth (so no locked MissionState is
    needed across workers); call this once afterward — from a single writer — to
    make ``MissionState`` consistent with what the swarm did: ledger ``done`` →
    COMPLETED, active **parks** → BLOCKED (so a parked feature is not later
    re-dispatched by the orchestrator path, preserving the anti-treadmill
    guarantee), and worker-recorded **discovered notes** folded into the matching
    feature's notes (so swarm mode never silently drops what it found). Discovered
    work stays advisory — reconcile never *creates* a feature from ledger data, so
    there is no path to inject gate-bypassing work. Returns the number of features
    whose status or notes changed.

    Holds the single-writer :func:`mission_owner_lock`, so it cannot run while an
    orchestrator is driving the same mission. It is still the caller's contract to
    invoke this only *after* the swarm's workers have stopped (workers touch the
    ledger, not ``MissionState``, so the fence does not cover them).
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
            feat.status = Status.COMPLETED
            n += 1
        elif feat.status == Status.PENDING and ledger.is_excluded(f"feature:{feat.id}"):
            # Only PENDING → BLOCKED: never downgrade a COMPLETED/IN_PROGRESS feature
            # on a stale park (the COMPLETED→BLOCKED revert claude flagged).
            feat.status = Status.BLOCKED
            reason = ledger.constraint_reason(f"feature:{feat.id}")
            if reason:  # keep the operator context for handoff/debugging
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
