"""Pareto-optimal debate-workflow search (syftr-pattern).

Where :mod:`aragora.routing.cost_quality_optimizer` picks the cost/quality-optimal
*provider* from observed metrics, this module searches over debate **workflow
configurations** — number of rounds, consensus mode, and the reviewing model
family set — to surface the configs on the cost / quality / latency Pareto
frontier.

Inspired by DataRobot's syftr (Pareto-optimized agentic workflows) but deliberately
dependency-free: instead of pulling in Optuna/MOTPE we use a *bounded* enumeration
of the (small, discrete) debate config space plus multi-objective non-domination.
The objective is pluggable — in production an evaluator runs a real debate and
reads cost from ``billing.cost_tracker``, quality from ``evaluation.llm_judge``, and
wall-clock latency; in tests it is a pure function. This keeps the expensive,
credential-bound part (actually running debates) out of the searchable core so the
search logic is fully unit-testable offline.

Cost note: a search runs one debate per trial, so it spends real money. Trials are
bounded by ``max_trials`` (default small) and the whole point is to *find* the cheap
configs — run a search once, then reuse the recommended config.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "DebateConfig",
    "ConfigEvaluation",
    "DebateSearchSpace",
    "SearchResult",
    "pareto_optimal",
    "search_pareto_configs",
]


@dataclass(frozen=True)
class DebateConfig:
    """One point in the debate-workflow search space."""

    rounds: int
    consensus: str
    families: tuple[str, ...]

    def label(self) -> str:
        return f"r{self.rounds}/{self.consensus}/{'+'.join(self.families)}"


@dataclass(frozen=True)
class ConfigEvaluation:
    """A config scored on the three objectives: cost ↓, quality ↑, latency ↓."""

    config: DebateConfig
    cost_usd: float
    quality: float  # normalized 0..1, higher is better
    latency_s: float

    def dominates(self, other: "ConfigEvaluation") -> bool:
        """True if this evaluation Pareto-dominates ``other``.

        Dominates ≡ no worse on every objective AND strictly better on ≥1.
        """
        no_worse = (
            self.cost_usd <= other.cost_usd
            and self.quality >= other.quality
            and self.latency_s <= other.latency_s
        )
        strictly_better = (
            self.cost_usd < other.cost_usd
            or self.quality > other.quality
            or self.latency_s < other.latency_s
        )
        return no_worse and strictly_better


@dataclass(frozen=True)
class DebateSearchSpace:
    """The discrete grid of debate configs to search.

    Defaults stay small and cheap-leaning (DeepSeek pairing first) so an
    unconfigured search doesn't fan out into an expensive combinatorial sweep.
    """

    rounds: tuple[int, ...] = (1, 2, 3)
    consensus_modes: tuple[str, ...] = ("majority", "unanimous", "prover_estimator")
    family_sets: tuple[tuple[str, ...], ...] = (
        ("claude", "deepseek"),
        ("claude", "openai"),
        ("claude", "openai", "grok"),
    )

    def enumerate(self) -> list[DebateConfig]:
        """All configs in the grid, deterministically ordered (cheapest-leaning first)."""
        configs: list[DebateConfig] = []
        for families in self.family_sets:
            for rounds in self.rounds:
                for consensus in self.consensus_modes:
                    configs.append(
                        DebateConfig(rounds=rounds, consensus=consensus, families=families)
                    )
        return configs


def pareto_optimal(evaluations: Sequence[ConfigEvaluation]) -> list[ConfigEvaluation]:
    """Return the non-dominated set, sorted by ascending cost then descending quality."""
    frontier: list[ConfigEvaluation] = []
    for candidate in evaluations:
        if not any(other.dominates(candidate) for other in evaluations if other is not candidate):
            frontier.append(candidate)
    frontier.sort(key=lambda e: (e.cost_usd, -e.quality, e.latency_s))
    return frontier


@dataclass
class SearchResult:
    frontier: list[ConfigEvaluation]
    all_evaluations: list[ConfigEvaluation] = field(default_factory=list)

    def recommend(
        self,
        *,
        max_cost_usd: float | None = None,
        min_quality: float | None = None,
    ) -> Optional[ConfigEvaluation]:
        """Pick a frontier config under the given constraints.

        Returns the cheapest frontier point meeting ``max_cost_usd``/``min_quality``;
        if none qualifies, returns the highest-quality frontier point (so a caller
        always gets a usable answer), or None if the frontier is empty.
        """
        if not self.frontier:
            return None
        eligible = [
            e
            for e in self.frontier
            if (max_cost_usd is None or e.cost_usd <= max_cost_usd)
            and (min_quality is None or e.quality >= min_quality)
        ]
        if eligible:
            return min(eligible, key=lambda e: e.cost_usd)
        return max(self.frontier, key=lambda e: e.quality)


def search_pareto_configs(
    space: DebateSearchSpace,
    evaluator: Callable[[DebateConfig], ConfigEvaluation],
    *,
    max_trials: int = 6,
) -> SearchResult:
    """Evaluate up to ``max_trials`` configs and return the Pareto frontier.

    ``evaluator`` runs/scores a single config — kept injectable so the costly,
    credential-bound debate execution lives outside this searchable core. Trials
    are bounded (each is a real debate = real spend); configs are taken in the
    space's deterministic, cheapest-leaning order.
    """
    if max_trials <= 0:
        raise ValueError("max_trials must be positive")
    configs = space.enumerate()[:max_trials]
    evaluations = [evaluator(config) for config in configs]
    return SearchResult(frontier=pareto_optimal(evaluations), all_evaluations=evaluations)
