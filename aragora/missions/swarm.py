"""Swarm worker loop — the pheromone wired to the gate.

One worker repeatedly: atomic-claims a unit (``select_for``), runs the merge-gate
``dispatch``, and writes the *outcome* back to the shared environment — ``done`` on
success, a **park** constraint after repeated blocks. The merge-quorum gate stays
the only thing that says "yes" (propose = swarm, accept = gate, the
FunSearch/AlphaEvolve split); the ledger remembers failures **as data**, so the
whole swarm escapes a treadmill without any prompt self-editing.

``run_worker`` is process-agnostic: run one per thread or per process against the
same ``state_path``/``ledger_path`` and they self-partition. All cross-worker
mutation goes through the file-locked ledger, so ``MissionState`` stays a static
backlog and never needs its own lock.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .ledger import Ledger, select_for
from .orchestrator import Dispatch
from .state import MissionState

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
        handoff = dispatch(state.get(unit))

        if handoff.success:
            ledger.record_done(unit)
            res.done.append(unit)
        else:
            attempts = ledger.bump_attempt(f"feature:{unit}")
            res.blocked.append(unit)
            if attempts >= park_threshold:
                ledger.record_constraint(
                    f"feature:{unit}",
                    f"parked after {attempts} blocks: {handoff.blocked_reason}",
                )
                res.parked.append(unit)
                logger.info("worker %s parked %s after %d blocks", worker_id, unit, attempts)

        ledger.release(unit, worker_id)

    return res
