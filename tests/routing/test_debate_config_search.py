"""Tests for the syftr-pattern debate-config Pareto search."""

from __future__ import annotations

from aragora.routing.debate_config_search import (
    ConfigEvaluation,
    DebateConfig,
    DebateSearchSpace,
    pareto_optimal,
    search_pareto_configs,
)


def _ev(cost: float, quality: float, latency: float, label: str = "x") -> ConfigEvaluation:
    return ConfigEvaluation(
        config=DebateConfig(rounds=1, consensus=label, families=("claude",)),
        cost_usd=cost,
        quality=quality,
        latency_s=latency,
    )


def test_domination_three_objectives() -> None:
    cheap_good_fast = _ev(0.1, 0.9, 1.0)
    pricey_worse = _ev(0.5, 0.8, 2.0)
    assert cheap_good_fast.dominates(pricey_worse)
    assert not pricey_worse.dominates(cheap_good_fast)


def test_no_domination_on_tradeoff() -> None:
    cheap_lowq = _ev(0.1, 0.7, 1.0)
    pricey_highq = _ev(0.5, 0.95, 1.0)
    # Neither dominates: one wins cost, the other wins quality.
    assert not cheap_lowq.dominates(pricey_highq)
    assert not pricey_highq.dominates(cheap_lowq)


def test_pareto_optimal_drops_dominated_keeps_tradeoffs() -> None:
    cheap = _ev(0.1, 0.7, 1.0, "cheap")
    balanced = _ev(0.3, 0.85, 1.0, "balanced")
    premium = _ev(0.5, 0.95, 1.0, "premium")
    dominated = _ev(0.6, 0.6, 3.0, "dominated")  # worse than all on every axis
    frontier = pareto_optimal([cheap, balanced, premium, dominated])
    labels = {e.config.consensus for e in frontier}
    assert labels == {"cheap", "balanced", "premium"}
    assert frontier[0].config.consensus == "cheap"  # sorted by ascending cost


def test_search_space_enumeration_is_deterministic_and_cheap_first() -> None:
    space = DebateSearchSpace()
    configs = space.enumerate()
    assert len(configs) == 3 * 3 * 3
    # cheapest-leaning family set (claude+deepseek) is enumerated first
    assert configs[0].families == ("claude", "deepseek")


def test_search_bounds_trials_and_returns_frontier() -> None:
    space = DebateSearchSpace()
    seen: list[str] = []

    def evaluator(config: DebateConfig) -> ConfigEvaluation:
        seen.append(config.label())
        # cheaper as rounds drop; quality rises with rounds — a real tradeoff.
        return ConfigEvaluation(
            config=config,
            cost_usd=0.05 * config.rounds * len(config.families),
            quality=0.6 + 0.1 * config.rounds,
            latency_s=float(config.rounds),
        )

    result = search_pareto_configs(space, evaluator, max_trials=4)
    assert len(seen) == 4  # bounded
    assert result.frontier  # non-empty
    assert all(isinstance(e, ConfigEvaluation) for e in result.frontier)


def test_recommend_honors_constraints_then_falls_back() -> None:
    result = search_pareto_configs(
        DebateSearchSpace(),
        lambda c: ConfigEvaluation(
            c, cost_usd=0.1 * c.rounds, quality=0.6 + 0.1 * c.rounds, latency_s=1.0
        ),
        max_trials=3,
    )
    # With a cost cap, pick cheapest eligible.
    cheap = result.recommend(max_cost_usd=0.15)
    assert cheap is not None and cheap.cost_usd <= 0.15
    # Impossible quality floor → fall back to highest-quality frontier point.
    best = result.recommend(min_quality=99.0)
    assert best is not None
    assert best.quality == max(e.quality for e in result.frontier)
