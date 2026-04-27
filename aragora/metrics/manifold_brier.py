"""Manifold Brier Scorer — rolling calibration against Manifold Markets outcomes.

Skeleton for AGT-03: Manifold Markets integration with rolling Brier scoring.
See ``docs/plans/AGENT_CIVILIZATION_SUBSTRATE.md`` §3 and issue #6064.

Brier score: BS = (p - o)²  where p ∈ [0,1] and o ∈ {0,1}.
Lower is better; a perfectly calibrated constant predictor at 0.5 scores 0.25.

This module is **flag-gated** and disabled by default.  Enable via:

    ARAGORA_MANIFOLD_BRIER_ENABLED=1

No live Manifold API calls are made in this module.  The scorer is a pure
in-memory accumulator; connector wiring that resolves questions against the
Manifold API is deferred to the next slice (AGT-03 sub-deliverable 2).
"""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Optional

__all__ = [
    "manifold_brier_enabled",
    "brier_score",
    "ManifoldPrediction",
    "ManifoldBrierScorer",
    "BrierWindowSummary",
]

# ---------------------------------------------------------------------------
# Feature gate
# ---------------------------------------------------------------------------

_TRUTHY = {"1", "true", "yes", "on"}


def manifold_brier_enabled() -> bool:
    """Return True when ARAGORA_MANIFOLD_BRIER_ENABLED is set to a truthy value."""
    return os.environ.get("ARAGORA_MANIFOLD_BRIER_ENABLED", "").lower() in _TRUTHY


def _require_enabled() -> None:
    if not manifold_brier_enabled():
        raise RuntimeError(
            "ManifoldBrierScorer is disabled. Set ARAGORA_MANIFOLD_BRIER_ENABLED=1 to enable."
        )


# ---------------------------------------------------------------------------
# Pure scoring function
# ---------------------------------------------------------------------------


def brier_score(predicted_probability: float, outcome: int) -> float:
    """Return the Brier score for a single prediction.

    Args:
        predicted_probability: Agent's stated probability in [0, 1].
        outcome: Resolved binary outcome — 1 (YES) or 0 (NO).

    Returns:
        (predicted_probability - outcome) ** 2

    Raises:
        ValueError: If ``predicted_probability`` is outside [0, 1] or
            ``outcome`` is not 0 or 1.
    """
    if not (0.0 <= predicted_probability <= 1.0):
        raise ValueError(f"predicted_probability must be in [0, 1]; got {predicted_probability}")
    if outcome not in (0, 1):
        raise ValueError(f"outcome must be 0 or 1; got {outcome}")
    return (predicted_probability - outcome) ** 2


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifoldPrediction:
    """A single agent prediction paired with its Manifold resolution.

    Attributes:
        question_id: Manifold question identifier (e.g. ``"will-aragora-merge-50-prs"``).
        predicted_probability: Probability in [0, 1] stated at prediction time.
        outcome: Resolved binary outcome — 1 (YES) or 0 (NO).
        predicted_at: UTC timestamp of when the prediction was recorded.
        resolved_at: UTC timestamp of Manifold resolution; may be None for
            pending questions.
        question_slug: Human-readable question slug for logging/debugging.
        agent_id: Optional agent identifier that made this prediction.
    """

    question_id: str
    predicted_probability: float
    outcome: int
    predicted_at: datetime
    resolved_at: Optional[datetime] = None
    question_slug: str = ""
    agent_id: str = ""

    def __post_init__(self) -> None:
        if not (0.0 <= self.predicted_probability <= 1.0):
            raise ValueError(
                f"predicted_probability must be in [0, 1]; got {self.predicted_probability}"
            )
        if self.outcome not in (0, 1):
            raise ValueError(f"outcome must be 0 or 1; got {self.outcome}")

    @property
    def score(self) -> float:
        """Brier score for this prediction."""
        return brier_score(self.predicted_probability, self.outcome)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "predicted_probability": self.predicted_probability,
            "outcome": self.outcome,
            "predicted_at": self.predicted_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "question_slug": self.question_slug,
            "agent_id": self.agent_id,
            "brier_score": self.score,
        }


@dataclass(frozen=True)
class BrierWindowSummary:
    """Rolling-window Brier summary returned by :meth:`ManifoldBrierScorer.rolling_score`.

    Attributes:
        window_days: Width of the rolling window in calendar days.
        n_predictions: Number of resolved predictions in the window.
        mean_brier: Mean Brier score across predictions; None when n == 0.
        median_brier: Median Brier score; None when n == 0.
        window_start: Inclusive start of the window (UTC).
        window_end: Inclusive end of the window (UTC).
    """

    window_days: int
    n_predictions: int
    mean_brier: Optional[float]
    median_brier: Optional[float]
    window_start: datetime
    window_end: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_days": self.window_days,
            "n_predictions": self.n_predictions,
            "mean_brier": self.mean_brier,
            "median_brier": self.median_brier,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
        }


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class ManifoldBrierScorer:
    """In-memory accumulator and rolling Brier scorer for Manifold predictions.

    All public methods raise ``RuntimeError`` unless the feature flag
    ``ARAGORA_MANIFOLD_BRIER_ENABLED=1`` is set.  This keeps the scorer
    invisible on the live path until the gate is explicitly opened.

    Usage::

        scorer = ManifoldBrierScorer()
        scorer.add(ManifoldPrediction(...))
        summary = scorer.rolling_score(window_days=30)
    """

    def __init__(self) -> None:
        self._predictions: list[ManifoldPrediction] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, prediction: ManifoldPrediction) -> None:
        """Append a resolved prediction.  Requires the feature flag."""
        _require_enabled()
        self._predictions.append(prediction)

    def clear(self) -> None:
        """Remove all stored predictions.  Requires the feature flag."""
        _require_enabled()
        self._predictions.clear()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def all_predictions(self) -> list[ManifoldPrediction]:
        """Return a snapshot of all stored predictions."""
        _require_enabled()
        return list(self._predictions)

    def rolling_score(
        self,
        *,
        window_days: int = 30,
        reference_time: Optional[datetime] = None,
    ) -> BrierWindowSummary:
        """Compute mean and median Brier score over a rolling calendar window.

        Predictions are selected by their ``predicted_at`` timestamp falling
        within ``[reference_time - window_days, reference_time]``.

        Args:
            window_days: Width of the window in calendar days (must be >= 1).
            reference_time: Upper bound of the window; defaults to ``datetime.now(UTC)``.

        Returns:
            :class:`BrierWindowSummary` with counts and score statistics.
        """
        _require_enabled()
        if window_days < 1:
            raise ValueError(f"window_days must be >= 1; got {window_days}")

        now = reference_time if reference_time is not None else datetime.now(UTC)
        cutoff = now - timedelta(days=window_days)

        in_window = [p for p in self._predictions if cutoff <= p.predicted_at <= now]
        scores = [p.score for p in in_window]

        return BrierWindowSummary(
            window_days=window_days,
            n_predictions=len(scores),
            mean_brier=statistics.mean(scores) if scores else None,
            median_brier=statistics.median(scores) if scores else None,
            window_start=cutoff,
            window_end=now,
        )
