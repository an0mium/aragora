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
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .ledger import Ledger, select_for
from .orchestrator import Dispatch, Handoff
from .state import Feature, MissionState, Status

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

        # Count the attempt BEFORE dispatch so a *raising* dispatch is bounded too,
        # and always release the lease (a raise must not leak the claim for a TTL).
        attempts = ledger.bump_attempt(f"feature:{unit}")
        try:
            handoff = dispatch(state.get(unit))
        except (
            Exception
        ) as exc:  # dispatch is an external callback — may raise anything  # noqa: BLE001
            handoff = Handoff(success=False, blocked_reason=f"dispatch raised: {exc!r}")
        finally:
            ledger.release(unit, worker_id)

        if handoff.success:
            ledger.record_done(unit)
            res.done.append(unit)
            # Swarm mode treats MissionState as a static backlog, so it cannot
            # insert follow-ups directly. Record them to the *locked* ledger so the
            # work is never lost; reconcile_from_ledger folds them in (single writer).
            for follow in handoff.follow_ups:
                ledger.record_follow_up(asdict(follow))
                logger.info(
                    "swarm recorded follow-up %s from %s (folds at reconcile)", follow.id, unit
                )
            for note in handoff.discovered:
                ledger.record_discovery(unit, note)
            continue

        # Terminal blocks (operator-gated / re-derive) park immediately; transient
        # ones retry until the shared attempt count hits the threshold.
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
    guarantee), and worker-recorded **follow-ups/discoveries** folded into the
    backlog (so swarm mode never silently drops discovered work). Returns the
    number of features whose status changed *or* were inserted.
    """
    state = MissionState.load(state_path)
    ledger = Ledger(ledger_path)
    done = ledger.done_units()
    n = 0
    for feat in state.features:
        if feat.id in done and feat.status != Status.COMPLETED:
            feat.status = Status.COMPLETED
            n += 1
        elif feat.status != Status.BLOCKED and ledger.is_excluded(f"feature:{feat.id}"):
            feat.status = Status.BLOCKED
            reason = ledger.constraint_reason(f"feature:{feat.id}")
            if reason:  # keep the operator context for handoff/debugging
                feat.notes = (feat.notes + "\n" if feat.notes else "") + f"BLOCKED (park): {reason}"
            n += 1

    # Fold worker-discovered work the static backlog couldn't hold (grok [P2]):
    # follow-up features the swarm found, and discovered notes on existing ones.
    existing = {f.id for f in state.features}
    for follow in ledger.pending_follow_ups():
        if follow["id"] not in existing:
            state.insert_feature(Feature(**follow))
            existing.add(follow["id"])
            n += 1
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
