"""MissionOrchestrator — the stateless tick loop.

The single property that makes a mission survive days: **the orchestrator holds
no state across ticks.** Every ``tick()`` reloads ``MissionState`` from disk,
advances exactly one feature, and persists before returning. Kill the process at
any point and a fresh ``run()`` resumes from the last persisted feature with no
lost or double-done work.

Dispatch is pluggable. Phase A ships a stub; the real engine wires
``swarm/boss_loop.py``'s tick + ``quorum_evidence``/``review_queue`` merge-gate in
as the ``dispatch`` callable (see the spec, Phase A2).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .state import Feature, MissionState, Status

logger = logging.getLogger(__name__)


@dataclass
class Handoff:
    """What a worker returns. The orchestrator must dispose of every field.

    ``success`` advances the feature; ``follow_ups`` extend the queue (handoff
    triage); ``blocked_reason`` parks it for operator escalation.
    """

    success: bool = True
    blocked_reason: str | None = None
    follow_ups: list[Feature] = field(default_factory=list)
    discovered: list[str] = field(default_factory=list)  # tracked into notes/library
    session_id: str | None = None


# A dispatch takes the feature to work and returns a Handoff. It must be
# idempotent: a feature retried after a crash must converge to the same result.
Dispatch = Callable[[Feature], Handoff]


class MissionOrchestrator:
    """Drives a mission to completion, one survivable tick at a time."""

    def __init__(self, state_path: str | Path) -> None:
        self.state_path = Path(state_path)

    # ---- single tick ---------------------------------------------------------

    def tick(self, dispatch: Dispatch) -> bool:
        """Advance one feature. Returns True if work was done, False if drained.

        Reload → reclaim orphans → pick next → mark in_progress (persist) →
        dispatch → triage handoff → persist. The persist *before* dispatch is
        what makes a mid-dispatch crash recoverable.
        """
        state = MissionState.load(self.state_path)

        reclaimed = state.reclaim_in_progress()
        if reclaimed:
            logger.info("reclaimed orphaned features from a prior crash: %s", reclaimed)
            state.save(self.state_path)

        feature = state.next_pending()
        if feature is None:
            return False

        # Checkpoint BEFORE dispatch — if the worker dies, resume reclaims this.
        state.mark_in_progress(feature.id)
        state.save(self.state_path)
        logger.info("dispatch feature %s (%s)", feature.id, feature.milestone)

        handoff = dispatch(feature)

        # Reload before triage: dispatch may have run in another process and the
        # on-disk truth may have advanced. Never trust the in-memory copy.
        state = MissionState.load(self.state_path)
        self._triage(state, feature.id, handoff)
        state.save(self.state_path)
        return True

    # ---- handoff triage ------------------------------------------------------

    def _triage(self, state: MissionState, feature_id: str, handoff: Handoff) -> None:
        feat = state.get(feature_id)
        if handoff.session_id and handoff.session_id not in feat.worker_session_ids:
            feat.worker_session_ids.append(handoff.session_id)

        for note in handoff.discovered:
            feat.notes = (feat.notes + "\n" if feat.notes else "") + f"discovered: {note}"

        for follow in handoff.follow_ups:
            if not any(f.id == follow.id for f in state.features):
                state.insert_feature(follow)
                logger.info("handoff inserted follow-up feature %s", follow.id)

        if handoff.success:
            state.mark_completed(feature_id)
        elif handoff.blocked_reason:
            state.mark_blocked(feature_id, handoff.blocked_reason)
        else:
            # No success, no explicit block → reset for retry.
            feat.status = Status.PENDING

    # ---- run loop ------------------------------------------------------------

    def run(self, dispatch: Dispatch, max_ticks: int = 10_000) -> tuple[int, int]:
        """Tick until the queue drains (or a tick cap). Resumable across crashes."""
        for _ in range(max_ticks):
            if not self.tick(dispatch):
                break
        done, total = MissionState.load(self.state_path).progress()
        logger.info("mission run paused/complete: %d/%d features", done, total)
        return done, total
