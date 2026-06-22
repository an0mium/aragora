"""Bounded drain-pass orchestrator for the boss loop.

When the boss loop is over the open-PR cap (backpressure on), it should DRAIN the
existing queue instead of idling or generating new work. This module is the pure
orchestration core of that pass: given a batch of observed open-PR candidates, it
applies :func:`aragora.swarm.drain_policy.decide_drain_action` to each, then
executes the actionable decisions **under per-type caps and a priority order**,
via an injected ``execute_fn`` (so the core stays pure and unit-testable; the
boss_loop hook supplies the real gh merge/close/dispatch).

Two properties matter for safety and for not making things worse:
- **REPAIR is tightly bounded** (``max_repairs_per_pass``, default 2): a REPAIR
  dispatches a worker, so 263 red PRs must NOT trigger 263 workers. Excess
  actionable items are deferred to the next pass, so the queue drains gradually.
- **LEAVE is never executed.** Off-limits (e.g. Factory's ``structex/*``),
  other-agent-owned, and over-tier PRs are preserved untouched — the
  anti-collision guarantee is enforced at the candidate level (``off_limits`` /
  ``owned_by_other_agent``) and honored here by simply never acting on LEAVE.

Priority when capped: MERGE first (lands work + drains), then CLOSE_SUPERSEDED
(drains cheaply), then REPAIR (creates load — most bounded).
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from aragora.swarm.drain_policy import DrainAction
from aragora.swarm.drain_policy import DrainCandidate
from aragora.swarm.drain_policy import DrainPolicy
from aragora.swarm.drain_policy import decide_drain_action

# execute_fn(pr_number, action) -> True on success. Injected by the boss_loop hook.
ExecuteFn = Callable[[int, DrainAction], bool]

# Priority order when caps force a choice (earlier = preferred).
_PRIORITY = (DrainAction.MERGE, DrainAction.CLOSE_SUPERSEDED, DrainAction.REPAIR)


@dataclass(frozen=True)
class DrainPassPolicy:
    """Per-pass bounds. Wraps the per-candidate ``DrainPolicy``."""

    drain: DrainPolicy = field(default_factory=DrainPolicy)
    max_merges_per_pass: int = 5
    max_closes_per_pass: int = 10
    max_repairs_per_pass: int = 2  # worker dispatch — keep small to avoid a repair storm

    def __post_init__(self) -> None:
        for name in ("max_merges_per_pass", "max_closes_per_pass", "max_repairs_per_pass"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")

    def cap_for(self, action: DrainAction) -> int:
        return {
            DrainAction.MERGE: self.max_merges_per_pass,
            DrainAction.CLOSE_SUPERSEDED: self.max_closes_per_pass,
            DrainAction.REPAIR: self.max_repairs_per_pass,
        }.get(action, 0)


@dataclass(frozen=True)
class PlannedAction:
    pr: int
    action: DrainAction
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"pr": self.pr, "action": self.action.value, "reason": self.reason}


@dataclass(frozen=True)
class DrainPassResult:
    planned: tuple[PlannedAction, ...]  # selected for action this pass (within caps)
    deferred: tuple[PlannedAction, ...]  # actionable but cap-exceeded -> next pass
    left: tuple[PlannedAction, ...]  # LEAVE decisions (off-limits/owned/over-tier)
    executed: tuple[int, ...]
    failed: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "planned": [p.to_dict() for p in self.planned],
            "deferred": [p.to_dict() for p in self.deferred],
            "left": [p.to_dict() for p in self.left],
            "executed": list(self.executed),
            "failed": list(self.failed),
            "counts": {
                "planned": len(self.planned),
                "deferred": len(self.deferred),
                "left": len(self.left),
                "executed": len(self.executed),
                "failed": len(self.failed),
            },
        }


def plan_drain_pass(
    policy: DrainPassPolicy, candidates: Iterable[DrainCandidate]
) -> tuple[list[PlannedAction], list[PlannedAction], list[PlannedAction]]:
    """Decide + bound, without executing. Returns ``(planned, deferred, left)``.

    Deterministic: candidate order is preserved within each action bucket, and
    per-type caps are applied in priority order. ``planned`` are the actions to
    take this pass; ``deferred`` are actionable items that exceeded a cap (try
    next pass); ``left`` are LEAVE decisions (never executed).
    """
    buckets: dict[DrainAction, list[PlannedAction]] = {a: [] for a in DrainAction}
    for cand in candidates:
        decision = decide_drain_action(policy.drain, cand)
        buckets[decision.action].append(
            PlannedAction(pr=cand.pr, action=decision.action, reason=decision.reason)
        )

    planned: list[PlannedAction] = []
    deferred: list[PlannedAction] = []
    for action in _PRIORITY:
        cap = policy.cap_for(action)
        items = buckets[action]
        planned.extend(items[:cap])
        deferred.extend(items[cap:])
    left = list(buckets[DrainAction.LEAVE])
    return planned, deferred, left


def run_drain_pass(
    policy: DrainPassPolicy,
    candidates: Iterable[DrainCandidate],
    execute_fn: ExecuteFn,
) -> DrainPassResult:
    """Plan then execute the planned actions via ``execute_fn``. LEAVE is never executed.

    ``execute_fn`` returning falsy (or raising) marks that PR failed; one failure
    never aborts the pass (the queue keeps draining).
    """
    planned, deferred, left = plan_drain_pass(policy, candidates)
    executed: list[int] = []
    failed: list[int] = []
    for item in planned:
        try:
            ok = execute_fn(item.pr, item.action)
        except Exception:  # noqa: BLE001 - injected callback; one bad PR must not abort the drain
            ok = False
        (executed if ok else failed).append(item.pr)
    return DrainPassResult(
        planned=tuple(planned),
        deferred=tuple(deferred),
        left=tuple(left),
        executed=tuple(executed),
        failed=tuple(failed),
    )
