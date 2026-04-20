"""Brief lifecycle state machine.

One authoritative state per ``(pr_number, head_sha)`` pair. This module is
pure types — no filesystem I/O, no logging, no external imports beyond
the standard library. The storage layer (:mod:`aragora.pdb.storage`) is
the only place that materializes state on disk.

States
------

``absent``   — no brief is known for this ``(pr, sha)`` pair.
``queued``   — a generation request was accepted; worker will pick it up.
``running``  — a worker is actively generating the brief.
``ready``    — the final signed brief JSON is on disk.
``failed``   — generation terminated abnormally; error record written.
``stale``    — a ``ready`` brief whose ``head_sha`` no longer matches the
               current GitHub PR head. Stale briefs are moved to
               ``invalidated/`` and the tuple ``(pr, new_sha)`` begins a
               fresh ``absent → queued → …`` cycle.

Legal transitions
-----------------

::

    absent  → queued
    queued  → running
    queued  → failed    (e.g., cancelled before pickup)
    queued  → absent    (cancel before pickup; no artifact left behind)
    running → ready
    running → failed
    running → absent    (cancel mid-run; partial cost preserved in failed
                         record only if caller explicitly asks for it —
                         see storage.cancel_generation)
    ready   → stale     (head_sha advanced; artifact moved to
                         invalidated/)
    failed  → queued    (retry; storage layer enforces that the failed
                         record is cleared atomically before the queued
                         record lands)
    stale   → queued    (regenerate for the same (pr, new_sha) — the new
                         SHA is effectively a new lifecycle; this
                         transition is expressed in state-machine terms
                         for symmetry but should not occur in practice
                         because ``get_state`` keys on ``(pr, sha)``)

Any transition not in the table above raises :class:`StateTransitionError`.
Transitioning into the same state is illegal (idempotency is the
caller's responsibility, not the state machine's).
"""

from __future__ import annotations

import enum
from typing import Mapping

__all__ = [
    "BriefLifecycleState",
    "LEGAL_TRANSITIONS",
    "StateTransitionError",
    "validate_transition",
]


class BriefLifecycleState(str, enum.Enum):
    """Canonical lifecycle states for a PR brief.

    Inherits from ``str`` so the enum values serialize cleanly into JSON
    and comparisons with raw strings (``state == "ready"``) work as
    expected at the API boundary.
    """

    ABSENT = "absent"
    QUEUED = "queued"
    RUNNING = "running"
    READY = "ready"
    FAILED = "failed"
    STALE = "stale"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


# Full transition table. Keyed by the source state; value is the set of
# legal destination states. Re-entering the same state is NOT legal.
LEGAL_TRANSITIONS: Mapping[BriefLifecycleState, frozenset[BriefLifecycleState]] = {
    BriefLifecycleState.ABSENT: frozenset(
        {
            BriefLifecycleState.QUEUED,
        }
    ),
    BriefLifecycleState.QUEUED: frozenset(
        {
            BriefLifecycleState.RUNNING,
            BriefLifecycleState.FAILED,
            BriefLifecycleState.ABSENT,
        }
    ),
    BriefLifecycleState.RUNNING: frozenset(
        {
            BriefLifecycleState.READY,
            BriefLifecycleState.FAILED,
            BriefLifecycleState.ABSENT,
        }
    ),
    BriefLifecycleState.READY: frozenset(
        {
            BriefLifecycleState.STALE,
        }
    ),
    BriefLifecycleState.FAILED: frozenset(
        {
            BriefLifecycleState.QUEUED,
        }
    ),
    BriefLifecycleState.STALE: frozenset(
        {
            BriefLifecycleState.QUEUED,
        }
    ),
}


class StateTransitionError(ValueError):
    """Raised when an illegal lifecycle transition is requested.

    Inherits from :class:`ValueError` so callers that want to swallow
    "bad input" errors without depending on this package directly still
    catch it naturally.
    """

    def __init__(
        self,
        source: BriefLifecycleState,
        destination: BriefLifecycleState,
        *,
        reason: str | None = None,
    ) -> None:
        self.source = source
        self.destination = destination
        detail = f"illegal brief lifecycle transition: {source.value} → {destination.value}"
        if reason:
            detail = f"{detail} ({reason})"
        super().__init__(detail)


def validate_transition(source: BriefLifecycleState, destination: BriefLifecycleState) -> None:
    """Assert that ``source → destination`` is a legal transition.

    Raises :class:`StateTransitionError` otherwise. Self-transitions
    (same source and destination) are always rejected — callers must
    express idempotent writes at a higher level.
    """

    if source == destination:
        raise StateTransitionError(
            source,
            destination,
            reason="self-transition; idempotency is the caller's responsibility",
        )
    allowed = LEGAL_TRANSITIONS.get(source, frozenset())
    if destination not in allowed:
        raise StateTransitionError(source, destination)
