"""Decision-stakes model routing — route by decision tier, record *why*.

Per-prompt routing (OpenRouter auto-router, Martian, Not Diamond, native lab
routers) is commoditizing. The defensible layer, and the one that belongs in a
decision receipt, is **decision-stakes routing**: choose models by the *stakes
of the decision* (its merge-gate tier) and make "why was this model trusted
with this decision" an auditable artifact.

Policy (issue #8233): low-stakes bounded lane / evidence work (Tier 0-2) routes
to cost-efficient models; strategic synthesis and Tier 3-4 settlement juries
escalate to frontier models. Selection within a tier is delegated to the
existing Pareto cost/quality optimizer (``cost_quality_optimizer``), driven by
the live ELO/calibration data already collected per provider.

This module is the routing *policy + rationale* core. The :class:`RoutingRationale`
it emits is the shape destined for the ODR receipt's ``routing`` block once that
block is un-reserved (spec §4.8 reserves it in v0.1); wiring it into the
emitter, and the cost-per-settled-PR instrument, are separate follow-ups.
Related to the ODR epic (#8223): the routing decision is decision-semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from aragora.routing.cost_quality_optimizer import SelectionStrategy

if TYPE_CHECKING:
    from aragora.routing.cost_quality_optimizer import CostQualityOptimizer

__all__ = [
    "TierClass",
    "RoutingPolicy",
    "RoutingRationale",
    "DecisionStakesRouter",
    "TIER_POLICY",
    "policy_for_tier",
    "ROUTING_RATIONALE_SCHEMA",
]

COST_EFFICIENT = "cost_efficient"
FRONTIER = "frontier"
TierClass = str  # COST_EFFICIENT | FRONTIER

# Canonical routing-rationale record family, shared with the evidence-collection
# variant written by ``scripts/auto_evidence_cycle.py``. That variant records a
# *static-configuration* family selection with ``pareto_optimizer_consulted:
# False`` ("the Pareto optimizer is not wired into this path yet"). This router
# IS that wiring for execution routing, so its record reuses the same schema and
# flips the flag to ``True``.
ROUTING_RATIONALE_SCHEMA = "aragora.routing_rationale/v1"


@dataclass(frozen=True)
class RoutingPolicy:
    """The routing policy applied at one decision tier."""

    tier_class: TierClass
    strategy: SelectionStrategy
    min_quality: float


# Tier 0-2: cost-efficient (bounded lane / evidence work, damped by exact-head
# gates + quorum). Tier 3-4: frontier (strategic synthesis / human-risk
# settlement juries). min_quality rises with stakes.
TIER_POLICY: dict[int, RoutingPolicy] = {
    0: RoutingPolicy(COST_EFFICIENT, SelectionStrategy.COST_OPTIMIZED, 0.0),
    1: RoutingPolicy(COST_EFFICIENT, SelectionStrategy.COST_OPTIMIZED, 0.3),
    2: RoutingPolicy(COST_EFFICIENT, SelectionStrategy.BALANCED, 0.5),
    3: RoutingPolicy(FRONTIER, SelectionStrategy.QUALITY_OPTIMIZED, 0.7),
    4: RoutingPolicy(FRONTIER, SelectionStrategy.QUALITY_OPTIMIZED, 0.8),
}

_MAX_TIER = max(TIER_POLICY)


def policy_for_tier(decision_tier: int) -> RoutingPolicy:
    """Return the :class:`RoutingPolicy` for ``decision_tier`` (clamped to 0-4)."""
    if decision_tier < 0:
        decision_tier = 0
    elif decision_tier > _MAX_TIER:
        decision_tier = _MAX_TIER
    return TIER_POLICY[decision_tier]


@dataclass
class RoutingRationale:
    """The auditable record of a decision-stakes routing choice.

    This is the artifact no generic per-prompt router produces: it binds the
    selected model to the *stakes* of the decision and the cost/quality inputs
    that justified it. ``to_dict()`` is the payload shape for the ODR receipt
    ``routing`` block.
    """

    decision_tier: int
    tier_class: TierClass
    strategy: str
    min_quality: float
    selected_provider: str | None
    selection_reason: str
    models_considered: list[dict[str, Any]] = field(default_factory=list)
    budget_remaining: float | None = None
    escalated_to_frontier: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Emit the ``aragora.routing_rationale/v1`` record for this choice.

        Same schema family as the evidence-collection record
        (``scripts/auto_evidence_cycle.py``); ``selector`` and
        ``pareto_optimizer_consulted`` mark this as the execution-routing
        variant where the Pareto optimizer *is* consulted. Cost stays
        ``recorded: false`` — per-provider ``avg_cost_per_debate`` (in
        ``models_considered``) is an expectation from metrics, not observed
        spend, and is never reported as recorded cost.
        """
        return {
            "record_type": "routing_rationale",
            "schema": ROUTING_RATIONALE_SCHEMA,
            "status": "present",
            "selector": "decision_stakes_pareto",
            "pareto_optimizer_consulted": True,
            **asdict(self),
            "cost": {
                "recorded": False,
                "total_usd": None,
                "absent_reason": (
                    "per-provider avg_cost_per_debate is an expectation from "
                    "metrics, not observed spend; recorded as absent"
                ),
            },
        }


class DecisionStakesRouter:
    """Routes by decision tier and records the rationale.

    Wraps a :class:`~aragora.routing.cost_quality_optimizer.CostQualityOptimizer`
    (which carries the live per-provider cost/quality/calibration metrics) and
    applies the tier policy on top.
    """

    def __init__(self, optimizer: CostQualityOptimizer) -> None:
        self._optimizer = optimizer

    def route(
        self,
        decision_tier: int,
        *,
        budget_remaining: float | None = None,
        exclude_providers: set[str] | None = None,
    ) -> RoutingRationale:
        """Select a provider for a decision of ``decision_tier`` and record why."""
        policy = policy_for_tier(decision_tier)

        # Pareto frontier is the set of non-dominated cost/quality options the
        # choice was made among — the "models considered" the audit record needs.
        try:
            frontier = self._optimizer.get_pareto_frontier()
        except Exception:  # noqa: BLE001 - never let metric gaps break routing
            frontier = []
        models_considered = [
            {
                "provider": m.provider_name,
                "avg_cost_per_debate": m.avg_cost_per_debate,
                "avg_quality_score": m.avg_quality_score,
                "failure_rate": m.failure_rate,
            }
            for m in frontier
        ]

        selected = self._optimizer.select_provider(
            strategy=policy.strategy,
            budget_remaining=budget_remaining,
            min_quality=policy.min_quality,
            exclude_providers=exclude_providers,
        )

        if selected is None:
            reason = (
                f"no provider met tier-{decision_tier} {policy.tier_class} policy "
                f"(strategy={policy.strategy.value}, min_quality={policy.min_quality}"
                + (f", budget={budget_remaining}" if budget_remaining is not None else "")
                + ")"
            )
        else:
            reason = (
                f"tier {decision_tier} ({policy.tier_class}) selects '{selected}' via "
                f"{policy.strategy.value} at min_quality={policy.min_quality}"
            )

        return RoutingRationale(
            decision_tier=decision_tier,
            tier_class=policy.tier_class,
            strategy=policy.strategy.value,
            min_quality=policy.min_quality,
            selected_provider=selected,
            selection_reason=reason,
            models_considered=models_considered,
            budget_remaining=budget_remaining,
            escalated_to_frontier=policy.tier_class == FRONTIER,
        )
