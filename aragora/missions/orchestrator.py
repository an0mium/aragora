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

from .ledger import LedgerCorruptError
from .state import Feature, MissionState, Status, mission_owner_lock

logger = logging.getLogger(__name__)


@dataclass
class Handoff:
    """What a worker returns. The orchestrator must dispose of every field.

    ``success`` advances the feature; ``follow_ups`` are worker-proposed queue
    extensions; ``blocked_reason`` records why it failed. Proposed follow-ups are
    advisory by default. A dispatch must set ``accept_follow_ups=True`` to make them
    executable, so a buggy worker cannot silently widen mission scope. ``terminal``
    distinguishes a block that **cannot self-heal by retrying** (operator-gated,
    Tier-3, a contaminated branch needing re-derive) from a transient one — a
    structured flag instead of sniffing the reason string, so the orchestrator and
    swarm agree.
    """

    success: bool = False
    terminal: bool = False  # True = do not retry; park/block immediately
    blocked_reason: str | None = None
    follow_ups: list[Feature] = field(default_factory=list)
    accept_follow_ups: bool = False
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

    Concurrency contract — orchestrator-mode and swarm-mode are **mutually
    exclusive**, enforced (not just documented) by the shared/exclusive
    :func:`mission_owner_lock`: ``run``/``tick`` take the exclusive side, swarm
    workers take the shared side, so a second orchestrator *or* a live swarm on the
    same ``state_path`` fails fast with :class:`MissionOwnershipError`. If a
    ``ledger_path`` is given (or a sibling ``ledger.json`` already exists), ``run``
    reconciles ledger results into state before ticking, so switching
    swarm→orchestrator never re-dispatches already-done work.
    Per-save persistence stays atomic via ``os.replace`` (no torn reads).
    """

    def __init__(
        self,
        state_path: str | Path,
        *,
        max_retries: int = 3,
        ledger_path: str | Path | None = None,
    ) -> None:
        self.state_path = Path(state_path)
        self.max_retries = max_retries
        self.ledger_path = Path(ledger_path) if ledger_path is not None else None

    # ---- single tick ---------------------------------------------------------

    def tick(self, dispatch: Dispatch) -> bool:
        """Advance one feature under the exclusive fence. Returns True if work was
        done, False if drained.

        Public single-tick entry point: it acquires the exclusive
        :func:`mission_owner_lock` for this one tick, including dispatch and
        triage. A hand-rolled ``while orch.tick(...)`` loop is therefore safe for
        each tick, but only :meth:`run` holds the fence continuously across the
        whole session. Inside :meth:`run` the loop calls :meth:`_tick` directly,
        holding the fence once for the whole run.
        """
        with mission_owner_lock(self.state_path, exclusive=True):
            if self._ledger_path_for_reconcile() is not None:
                self._reconcile_ledger()
            return self._tick(dispatch)

    def _tick(self, dispatch: Dispatch) -> bool:
        """One tick, assuming the exclusive fence is already held.

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
            if self._block_unrunnable_pending(
                state,
                "unmet or unsupported preconditions left no runnable mission work",
            ):
                state.save(self.state_path)
            return False

        # Provisionally count a *crash* BEFORE dispatch, so a raise leaves the count
        # on disk for reclaim to cap. _triage resets it the moment dispatch returns.
        # Reclaim allows one final idempotent confirmation when crash_count reaches
        # the cap, so a dispatch that succeeded externally but died before triage is
        # re-dispatched once to observe already-done/success instead of false-BLOCKed.
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
            if feat.crash_count > self.max_retries:
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
            proposal = f"follow-up proposed: {follow.id} — {follow.description}"
            stamp = f"discovered: {proposal}"
            if stamp not in feat.notes:
                feat.notes = (feat.notes + "\n" if feat.notes else "") + stamp
            if handoff.accept_follow_ups and not any(f.id == follow.id for f in state.features):
                if "paths" not in follow.metadata and "paths" in feat.metadata:
                    follow.metadata["paths"] = list(feat.metadata["paths"])
                state.insert_feature(follow)
                logger.info("handoff inserted accepted follow-up feature %s", follow.id)

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

    def _block_unrunnable_pending(self, state: MissionState, reason: str) -> bool:
        changed = False
        for feat in state.features:
            if feat.status != Status.PENDING:
                continue
            details = (
                ", ".join(feat.preconditions) if feat.preconditions else "no runnable dispatch"
            )
            state.mark_blocked(feat.id, f"{reason}: {details}")
            changed = True
        return changed

    # ---- run loop ------------------------------------------------------------

    def run(self, dispatch: Dispatch, max_ticks: int = 10_000) -> tuple[int, int]:
        """Tick until the queue drains (or a tick cap). Resumable across crashes.

        Holds the exclusive :func:`mission_owner_lock` for the whole run, so a second
        orchestrator *or* a live swarm fails fast with ``MissionOwnershipError``
        instead of racing ``next_pending``. If a ``ledger_path`` was given (or a
        sibling ``ledger.json`` already exists), folds the swarm's ledger results
        into state first (so swarm→orchestrator never re-dispatches done/parked
        work). The final state is reported precisely:
        complete, blocked-remaining, or hit the tick cap.
        """
        hit_cap = True
        with mission_owner_lock(self.state_path, exclusive=True):
            if self._ledger_path_for_reconcile() is not None:
                self._reconcile_ledger()
            for _ in range(max_ticks):
                if not self._tick(dispatch):
                    hit_cap = False
                    break
        state = MissionState.load(self.state_path)
        done, total = state.progress()
        if hit_cap and done < total:
            logger.warning(
                "mission run hit max_ticks=%d with %d/%d done — NOT complete, re-run to continue",
                max_ticks,
                done,
                total,
            )
        elif done < total:
            blocked = sum(1 for f in state.features if f.status == Status.BLOCKED)
            logger.info(
                "mission run drained: %d/%d done, %d blocked (queue has no runnable work)",
                done,
                total,
                blocked,
            )
        else:
            logger.info("mission run complete: %d/%d features", done, total)
        return done, total

    def _reconcile_ledger(self) -> None:
        """Fold ledger results into state before driving (swarm→orchestrator switch).

        Imported lazily to avoid a circular import (swarm imports from this module).
        Called while the exclusive fence is already held, so it does not re-acquire.
        """
        from .swarm import _reconcile_locked

        ledger_path = self._ledger_path_for_reconcile()
        if ledger_path is not None:
            try:
                _reconcile_locked(self.state_path, ledger_path)
            except LedgerCorruptError as exc:
                self._block_all_open_work(f"ledger reconcile failed closed: {exc}")

    def _block_all_open_work(self, reason: str) -> None:
        state = MissionState.load(self.state_path)
        changed = False
        for feat in state.features:
            if feat.status in {Status.PENDING, Status.IN_PROGRESS}:
                state.mark_blocked(feat.id, reason)
                changed = True
        if changed:
            state.save(self.state_path)

    def _ledger_path_for_reconcile(self) -> Path | None:
        if self.ledger_path is not None:
            return self.ledger_path
        default_path = self.state_path.with_name("ledger.json")
        return default_path if default_path.exists() else None
