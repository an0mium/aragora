"""Drain-mode decision core for the boss loop.

When the open-PR queue is over its backpressure cap, the boss loop should DRAIN
the queue instead of idling or generating more issues. This is the pure,
deterministic heart of that behavior: given one open PR's observed state, decide
whether to MERGE it, REPAIR it (dispatch a worker to fix a useful-but-red PR),
CLOSE it (only if TRULY superseded), or LEAVE it alone. No I/O — the caller
gathers PR state (gh/merge-packet) and performs the chosen action; the boss_loop
wiring lands in a separate PR (boss_loop.py is large and drives the live loop).

Encodes the operator's rule verbatim:
- "Repair and merge useful work; close ONLY truly-entirely-superseded PRs."
  -> CLOSE_SUPERSEDED requires has_changes is False OR an explicit superseded
     flag. A merely-red PR is REPAIR, never CLOSE.
- "Don't interfere with Factory." -> off_limits or owned_by_other_agent always
  LEAVE — the drainer never touches a PR another fleet owns or one pinned
  off-limits via the steering mailbox.
- The merge-quorum gate stays the sole settlement authority: MERGE requires
  green required checks + a supportive 2-family quorum + mergeable + a tier
  within the autonomous-settle bound; anything above that bound LEAVEs for the
  operator (never auto-merged).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class DrainAction(Enum):
    MERGE = "merge"  # green + quorum + mergeable + in-bound tier -> land via the gate
    REPAIR = "repair"  # useful but red/near-green -> dispatch a repair worker
    CLOSE_SUPERSEDED = "close_superseded"  # TRULY superseded (empty/dup/landed) -> close
    LEAVE = "leave"  # owned/off-limits/over-tier/in-flight -> do not touch


@dataclass(frozen=True)
class DrainPolicy:
    """Bounds for autonomous draining."""

    auto_settle_max_tier: int = 2  # tiers 0..N may auto-merge; above -> LEAVE for human

    def __post_init__(self) -> None:
        if not 0 <= self.auto_settle_max_tier <= 4:
            raise ValueError("auto_settle_max_tier must be in [0, 4]")


@dataclass(frozen=True)
class DrainCandidate:
    """Observed state of one open PR at a drain decision point."""

    pr: int
    has_changes: bool = True  # changed_files > 0 (False => empty => superseded)
    superseded: bool = False  # explicitly superseded (exact-dup / fully landed elsewhere)
    off_limits: bool = False  # pinned off-limits (e.g. Factory's branch) -> never touch
    owned_by_other_agent: bool = False  # a live owner is driving it -> never touch
    required_checks_green: bool = False
    quorum_satisfied: bool = False  # 2-family supportive, no unresolved dissent
    mergeable: bool = False  # no conflicts; head matches
    tier: int = 0  # merge tier (0-4)


@dataclass(frozen=True)
class DrainDecision:
    action: DrainAction
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action.value, "reason": self.reason}


def decide_drain_action(policy: DrainPolicy, cand: DrainCandidate) -> DrainDecision:
    """Decide what to do with one open PR while draining. Pure; precedence matters.

    Order: (1) ownership/off-limits -> LEAVE (never collide with another fleet);
    (2) truly-superseded -> CLOSE_SUPERSEDED; (3) green+quorum+mergeable+in-bound
    tier -> MERGE; (4) over the autonomous-settle tier -> LEAVE for the operator;
    (5) otherwise it is useful but not landable -> REPAIR.
    """
    # (1) Never touch another fleet's / off-limits PR — the anti-collision guarantee.
    if cand.off_limits:
        return DrainDecision(DrainAction.LEAVE, "pinned off-limits (e.g. Factory) — not touched")
    if cand.owned_by_other_agent:
        return DrainDecision(DrainAction.LEAVE, "a live owner is driving it — not touched")

    # (2) CLOSE only when TRULY superseded (empty or explicitly superseded). Never close
    #     a merely-red PR — that is repairable, useful work.
    if not cand.has_changes:
        return DrainDecision(DrainAction.CLOSE_SUPERSEDED, "empty (no changes) — truly superseded")
    if cand.superseded:
        return DrainDecision(DrainAction.CLOSE_SUPERSEDED, "explicitly superseded (dup/landed)")

    # (3) Land it only if fully gated AND within the autonomous-settle tier.
    if (
        cand.required_checks_green
        and cand.quorum_satisfied
        and cand.mergeable
        and cand.tier <= policy.auto_settle_max_tier
    ):
        return DrainDecision(DrainAction.MERGE, "green + quorum + mergeable + in-bound tier")

    # (4) Gated work above the autonomous tier is the operator's call — never auto-merge.
    if cand.tier > policy.auto_settle_max_tier:
        return DrainDecision(
            DrainAction.LEAVE, f"tier {cand.tier} > auto_settle_max_tier — parks for operator"
        )

    # (5) Useful work that isn't landable yet -> repair it (don't discard it).
    return DrainDecision(DrainAction.REPAIR, "useful but not green/mergeable — dispatch repair")
