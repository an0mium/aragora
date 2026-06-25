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

from .state import Feature, MissionState, Status, mission_owner_lock

logger = logging.getLogger(__name__)


@dataclass
class Handoff:
    """What a worker returns. The orchestrator must dispose of every field.

    ``success`` advances the feature; ``follow_ups`` extend the queue (handoff
    triage); ``blocked_reason`` records why it failed. ``terminal`` distinguishes a
    block that **cannot self-heal by retrying** (operator-gated, Tier-3, a
    contaminated branch needing re-derive) from a transient one — a structured flag
    instead of sniffing the reason string, so the orchestrator and swarm agree.
    """

    success: bool = True
    terminal: bool = False  # True = do not retry; park/block immediately
    blocked_reason: str | None = None
    follow_ups: list[Feature] = field(default_factory=list)
    discovered: list[str] = field(default_factory=list)  # tracked into notes/library
    session_id: str | None = None


# A dispatch takes the feature to work and returns a Handoff. It must be
# idempotent: a feature retried after a crash must converge to the same result.
Dispatch = Callable[[Feature], Handoff]


class MissionOrchestrator:
    """Drives a mission to completion, one survivable tick at a time.

    ``max_retries`` bounds the retry loop: a feature that keeps failing — whether
    by *returning* a failure or by *raising* (kill-9 / 402 / poison) — is marked
    BLOCKED after this many attempts instead of being re-picked forever (the same
    park-after-N protection the swarm path has). A ``Handoff(terminal=True)`` is
    blocked immediately — no point retrying a fork only a human can resolve.

    Single-writer contract: one orchestrator per ``state_path``. ``MissionState.save``
    now takes an exclusive lock so a contract-violating second writer serializes
    rather than interleaving, but the contract is still the real guarantee — do not
    run a second orchestrator, or a swarm, against the same state file concurrently.
    To hand off between the orchestrator and swarm modes, reconcile first via
    :func:`aragora.missions.swarm.reconcile_from_ledger`.
    """

    def __init__(self, state_path: str | Path, *, max_retries: int = 3) -> None:
        self.state_path = Path(state_path)
        self.max_retries = max_retries

    # ---- single tick ---------------------------------------------------------

    def tick(self, dispatch: Dispatch) -> bool:
        """Advance one feature. Returns True if work was done, False if drained.

        Reload → reclaim orphans → pick next → mark in_progress (persist) →
        dispatch → triage handoff → persist. The persist *before* dispatch is
        what makes a mid-dispatch crash recoverable.
        """
        state = MissionState.load(self.state_path)

        # Reclaim orphans, honoring the retry cap — a *raising* dispatch (kill-9 /
        # recurring 402 / poison feature) leaves the feature IN_PROGRESS, and
        # without a cap here it would be reset to PENDING and re-picked forever.
        if self._reclaim_with_cap(state):
            state.save(self.state_path)

        feature = state.next_pending()
        if feature is None:
            return False

        # Provisionally count a *crash* BEFORE dispatch, so a raise leaves the count
        # on disk for reclaim to cap. _triage resets it the moment dispatch returns,
        # so a dispatch that succeeded-but-crashed-before-save is re-dispatched
        # (idempotent → completes) rather than false-BLOCKed.
        feature.crash_count += 1
        state.mark_in_progress(feature.id)
        state.save(self.state_path)
        logger.info(
            "dispatch feature %s (%s), crash_count=%d retry_count=%d",
            feature.id,
            feature.milestone,
            feature.crash_count,
            feature.retry_count,
        )

        try:
            handoff = dispatch(feature)
        except Exception:  # noqa: BLE001 - dispatch is an external callback (402/poison may raise anything)
            # In-process bound: a raising dispatch must NOT abort the whole mission
            # ("set a goal and walk away" needs the loop to outlive one bad feature).
            # crash_count was persisted pre-dispatch; leave the feature IN_PROGRESS so
            # the next tick's reclaim caps it (PENDING under the cap, BLOCKED at it).
            logger.exception(
                "dispatch raised for feature %s (crash_count=%d); next tick reclaims/caps it",
                feature.id,
                feature.crash_count,
            )
            return True

        # Reload before triage: dispatch may have run in another process and the
        # on-disk truth may have advanced. Never trust the in-memory copy.
        state = MissionState.load(self.state_path)
        self._triage(state, feature.id, handoff)
        state.save(self.state_path)
        return True

    def _reclaim_with_cap(self, state: MissionState) -> bool:
        """Reset orphaned IN_PROGRESS features to PENDING, but BLOCK any that have
        crashed ``max_retries`` times in a row (the raising-dispatch bound). A
        non-crash-looping feature is always retried — so a crash *after* a
        successful dispatch is re-dispatched and confirmed, not false-blocked.
        Returns True if anything changed."""
        changed = False
        for feat in state.features:
            if feat.status != Status.IN_PROGRESS:
                continue
            changed = True
            if feat.crash_count >= self.max_retries:
                feat.status = Status.BLOCKED
                feat.notes = (feat.notes + "\n" if feat.notes else "") + (
                    f"blocked: crashed {feat.crash_count}x in a row (poison/crash-looping dispatch)"
                )
            else:
                feat.status = Status.PENDING
        return changed

    # ---- handoff triage ------------------------------------------------------

    def _triage(self, state: MissionState, feature_id: str, handoff: Handoff) -> None:
        feat = state.get(feature_id)
        feat.crash_count = 0  # dispatch returned — it did not crash this round
        if handoff.session_id and handoff.session_id not in feat.worker_session_ids:
            feat.worker_session_ids.append(handoff.session_id)

        for note in handoff.discovered:
            stamp = f"discovered: {note}"
            if stamp not in feat.notes:  # dedupe across retries
                feat.notes = (feat.notes + "\n" if feat.notes else "") + stamp

        for follow in handoff.follow_ups:
            if not any(f.id == follow.id for f in state.features):
                state.insert_feature(follow)
                logger.info("handoff inserted follow-up feature %s", follow.id)

        if handoff.success:
            state.mark_completed(feature_id)
            return

        # Returned failure: bound by retry_count. Park a terminal block immediately
        # (never loop on operator-gated forks); retry a transient one until the cap.
        feat.retry_count += 1
        reason = handoff.blocked_reason or f"failed {feat.retry_count}x with no reason"
        if handoff.terminal or feat.retry_count >= self.max_retries:
            state.mark_blocked(feature_id, reason)
        else:
            feat.status = Status.PENDING  # bounded retry

    # ---- run loop ------------------------------------------------------------

    def run(self, dispatch: Dispatch, max_ticks: int = 10_000) -> tuple[int, int]:
        """Tick until the queue drains (or a tick cap). Resumable across crashes.

        Holds the single-writer :func:`mission_owner_lock` for the whole run, so a
        second concurrent orchestrator fails fast with ``MissionOwnershipError``
        instead of racing ``next_pending`` and double-dispatching a feature. Hitting
        ``max_ticks`` is logged as a *cap*, not silently treated as completion.
        """
        hit_cap = True
        with mission_owner_lock(self.state_path):
            for _ in range(max_ticks):
                if not self.tick(dispatch):
                    hit_cap = False
                    break
        done, total = MissionState.load(self.state_path).progress()
        if hit_cap and done < total:
            logger.warning(
                "mission run hit max_ticks=%d with %d/%d done — NOT complete, re-run to continue",
                max_ticks,
                done,
                total,
            )
        else:
            logger.info("mission run paused/complete: %d/%d features", done, total)
        return done, total
