"""Tests for decision-stakes routing (aragora.routing.decision_stakes_router)."""

from __future__ import annotations

from typing import Any

from aragora.routing.cost_quality_optimizer import CostQualityOptimizer, SelectionStrategy
from aragora.routing.decision_stakes_router import (
    COST_EFFICIENT,
    FRONTIER,
    DecisionStakesRouter,
    RoutingRationale,
    policy_for_tier,
)
from aragora.routing.provider_metrics import ProviderMetrics


class _Store:
    """Minimal ProviderMetricsStore stand-in: the optimizer only reads metrics."""

    def __init__(self, metrics: list[ProviderMetrics]) -> None:
        self._m = {m.provider_name: m for m in metrics}

    def get_all_metrics(self) -> dict[str, ProviderMetrics]:
        return dict(self._m)


def _optimizer() -> CostQualityOptimizer:
    return CostQualityOptimizer(
        _Store(
            [
                ProviderMetrics(
                    provider_name="cheap",
                    avg_cost_per_debate=0.01,
                    avg_quality_score=0.40,
                    failure_rate=0.0,
                ),
                ProviderMetrics(
                    provider_name="mid",
                    avg_cost_per_debate=0.10,
                    avg_quality_score=0.75,
                    failure_rate=0.0,
                ),
                ProviderMetrics(
                    provider_name="frontier",
                    avg_cost_per_debate=1.00,
                    avg_quality_score=0.95,
                    failure_rate=0.0,
                ),
            ]
        )
    )


def test_policy_for_tier_classes() -> None:
    assert policy_for_tier(0).tier_class == COST_EFFICIENT
    assert policy_for_tier(2).tier_class == COST_EFFICIENT
    assert policy_for_tier(3).tier_class == FRONTIER
    assert policy_for_tier(4).tier_class == FRONTIER


def test_policy_for_tier_clamps_out_of_range() -> None:
    assert policy_for_tier(-5).tier_class == policy_for_tier(0).tier_class
    assert policy_for_tier(99) == policy_for_tier(4)


def test_min_quality_rises_with_stakes() -> None:
    qualities = [policy_for_tier(t).min_quality for t in range(5)]
    assert qualities == sorted(qualities)
    assert qualities[0] <= qualities[4]
    assert qualities != [qualities[0]] * 5  # actually rises


def test_low_tier_routes_to_cost_efficient_model() -> None:
    rationale = DecisionStakesRouter(_optimizer()).route(0)
    assert rationale.selected_provider == "cheap"
    assert rationale.tier_class == COST_EFFICIENT
    assert rationale.escalated_to_frontier is False
    assert rationale.strategy == SelectionStrategy.COST_OPTIMIZED.value


def test_high_tier_escalates_to_frontier_model() -> None:
    rationale = DecisionStakesRouter(_optimizer()).route(4)
    assert rationale.selected_provider == "frontier"
    assert rationale.tier_class == FRONTIER
    assert rationale.escalated_to_frontier is True
    assert rationale.strategy == SelectionStrategy.QUALITY_OPTIMIZED.value


def test_min_quality_floor_excludes_low_quality_at_high_tier() -> None:
    # Tier 4 requires min_quality 0.8; only "frontier" (0.95) qualifies.
    rationale = DecisionStakesRouter(_optimizer()).route(4, exclude_providers={"frontier"})
    assert rationale.selected_provider is None
    assert "no provider met" in rationale.selection_reason


def test_models_considered_records_the_pareto_frontier() -> None:
    rationale = DecisionStakesRouter(_optimizer()).route(2)
    providers = {m["provider"] for m in rationale.models_considered}
    assert providers == {"cheap", "mid", "frontier"}
    for entry in rationale.models_considered:
        assert {"provider", "avg_cost_per_debate", "avg_quality_score", "failure_rate"} <= set(
            entry
        )


def test_budget_constraint_is_recorded_and_applied() -> None:
    # Budget 0.05 excludes mid (0.10) and frontier (1.00); only "cheap" remains.
    rationale = DecisionStakesRouter(_optimizer()).route(0, budget_remaining=0.05)
    assert rationale.budget_remaining == 0.05
    assert rationale.selected_provider == "cheap"


def test_to_dict_is_receipt_routing_block_shaped() -> None:
    import json

    rationale = DecisionStakesRouter(_optimizer()).route(3)
    payload = rationale.to_dict()
    # Canonical routing-rationale schema family (shared with auto_evidence_cycle).
    assert payload["record_type"] == "routing_rationale"
    assert payload["schema"] == "aragora.routing_rationale/v1"
    assert payload["status"] == "present"
    assert payload["selector"] == "decision_stakes_pareto"
    assert payload["pareto_optimizer_consulted"] is True
    assert payload["decision_tier"] == 3
    assert payload["tier_class"] == FRONTIER
    assert "selection_reason" in payload
    # Cost stays honestly absent (metrics expectation, not observed spend).
    assert payload["cost"]["recorded"] is False
    assert payload["cost"]["total_usd"] is None
    json.dumps(payload)  # must be JSON-serializable for the receipt


def test_empty_metrics_store_yields_no_selection_not_crash() -> None:
    router = DecisionStakesRouter(CostQualityOptimizer(_Store([])))
    rationale = router.route(1)
    assert isinstance(rationale, RoutingRationale)
    assert rationale.selected_provider is None
    assert rationale.models_considered == []
