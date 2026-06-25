"""Dispatch adapters — how a mission feature becomes a merged, gated change.

The orchestrator (``orchestrator.py``) is gate-agnostic: it hands a ``Feature`` to
a ``Dispatch`` callable and triages the returned ``Handoff``. This module supplies
the real one: :class:`BossLoopDispatch`, which drives Aragora's merge-quorum gate
with every rule we learned the hard way operating Factory for ~13 days —

  * **idempotent**: an already-merged branch is *success*, not an error (a retry
    after a crash must converge, never double-merge);
  * **foreign-commit guard** (the #8616 lesson): if any non-mission commit landed
    on the branch, do NOT collect evidence — return blocked for re-derive;
  * **head-bound merge, never ``--admin``**;
  * **Tier-3 surfaces escalate to the operator** rather than auto-settling.

The live wiring (``swarm/boss_loop`` worker spawn, ``aragora/worktree`` isolation,
``swarm/quorum_evidence`` + ``cli/commands/review_queue`` gate) plugs in behind the
small :class:`FleetGate` protocol, so this logic is fully testable without touching
live ``main``. ``LiveBossLoopGate`` (the thin real binding) is the only remaining
seam — deliberately a separate, reviewable step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from .orchestrator import Handoff
from .state import Feature

logger = logging.getLogger(__name__)


@dataclass
class GateVerdict:
    """Outcome of collecting heterogeneous-model quorum evidence on a head."""

    satisfied: bool
    tier: int = 0
    dissent: list[str] = field(default_factory=list)


class FleetGate(Protocol):
    """The merge-gate surface a dispatch needs. Live impl wraps review_queue."""

    def branch_for(self, feature: Feature) -> str: ...

    def already_merged(self, branch: str) -> bool: ...

    def head_of(self, branch: str) -> str: ...

    def foreign_commits(
        self, branch: str, base: str, allowed_prefixes: tuple[str, ...]
    ) -> list[str]: ...

    def tier_of(self, feature: Feature) -> int: ...  # cheap classification, before evidence

    def collect_evidence(self, branch: str, head: str) -> GateVerdict: ...

    def merge_head_bound(self, branch: str, head: str) -> bool: ...


class BossLoopDispatch:
    """Turn a feature into a gated, head-bound merge — or a precise handoff.

    Operator-settlement boundary (see the Tier-3 bright line, 2026-06-25): pure
    structural moves auto-settle on a clean quorum; Tier >= ``operator_tier``
    surfaces (server/persistence/security) are escalated, never auto-settled.
    """

    def __init__(
        self,
        gate: FleetGate,
        *,
        base: str = "main",
        allowed_prefixes: tuple[str, ...] = ("structex/", "mission/"),
        operator_tier: int = 3,
    ) -> None:
        self.gate = gate
        self.base = base
        self.allowed_prefixes = allowed_prefixes
        self.operator_tier = operator_tier

    def __call__(self, feature: Feature) -> Handoff:
        branch = self.gate.branch_for(feature)

        # Idempotency: a crash-retried feature whose PR already merged is done.
        if self.gate.already_merged(branch):
            logger.info("feature %s already merged (idempotent success)", feature.id)
            return Handoff(success=True, discovered=["branch already merged on a prior attempt"])

        head = self.gate.head_of(branch)

        # Foreign-commit guard (#8616): never collect evidence on a contaminated
        # head — park for re-derive instead of merging someone else's work.
        foreign = self.gate.foreign_commits(branch, self.base, self.allowed_prefixes)
        if foreign:
            return Handoff(
                success=False,
                blocked_reason=f"contaminated by foreign commits {foreign}; re-derive clean off {self.base} before evidence",
                discovered=[f"foreign-commit guard tripped on {branch}"],
            )

        # Tier-3+ surfaces are an operator fork — classify first and escalate
        # before spending an (expensive) quorum on something that can't auto-settle.
        tier = self.gate.tier_of(feature)
        if tier >= self.operator_tier:
            return Handoff(
                success=False,
                blocked_reason=f"tier-{tier} surface requires operator settlement (head {head})",
            )

        verdict = self.gate.collect_evidence(branch, head)
        if not verdict.satisfied:
            return Handoff(
                success=False,
                blocked_reason=f"quorum not satisfied: {verdict.dissent or 'incomplete'}",
            )

        if self.gate.merge_head_bound(branch, head):
            logger.info("feature %s merged head-bound at %s", feature.id, head)
            return Handoff(success=True, session_id=head)

        return Handoff(
            success=False, blocked_reason=f"head-bound merge of {head} did not land (head moved?)"
        )
