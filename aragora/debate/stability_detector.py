"""Adaptive stability detection for debate early stopping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class StabilityResult:
    """Stability score output."""

    stability: float
    agreement_score: float
    successes: int
    total: int


class BetaBinomialStabilityDetector:
    """Beta-Binomial stability detector over agreement scores."""

    def __init__(
        self,
        agreement_threshold: float = 0.75,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
    ) -> None:
        self.agreement_threshold = agreement_threshold
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

    def calculate_stability(self, agreement_scores: Iterable[float]) -> StabilityResult:
        scores = list(agreement_scores)
        if not scores:
            return StabilityResult(stability=0.0, agreement_score=0.0, successes=0, total=0)

        successes = sum(1 for score in scores if score >= self.agreement_threshold)
        total = len(scores)
        posterior_mean = (self.alpha_prior + successes) / (
            self.alpha_prior + self.beta_prior + total
        )

        return StabilityResult(
            stability=posterior_mean,
            agreement_score=scores[-1],
            successes=successes,
            total=total,
        )
