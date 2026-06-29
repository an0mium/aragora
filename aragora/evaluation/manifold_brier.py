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
from decimal import Decimal, ROUND_FLOOR
from typing import Any, Optional

__all__ = [
    "manifold_brier_enabled",
    "brier_score",
    "ManifoldPrediction",
    "ManifoldBrierScorer",
    "BrierWindowSummary",
    "CalibrationBin",
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


@dataclass(frozen=True)
class CalibrationBin:
    """One probability bracket in a calibration curve.

    A calibration curve divides ``[0, 1]`` into equal-width bins and, for
    each bin, reports how often predictions in that bracket resolved YES.
    A well-calibrated agent has ``fraction_yes ≈ mean_predicted`` in every
    bin.

    Attributes:
        low: Inclusive lower bound of the bracket.
        high: Exclusive upper bound (last bin is [low, 1.0] inclusive).
        count: Number of resolved predictions in this bracket.
        fraction_yes: Empirical frequency of ``outcome == 1``; None when
            ``count == 0``.
        mean_predicted: Mean ``predicted_probability`` in this bracket; None
            when ``count == 0``.
    """

    low: float
    high: float
    count: int
    fraction_yes: Optional[float]
    mean_predicted: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "low": self.low,
            "high": self.high,
            "count": self.count,
            "fraction_yes": self.fraction_yes,
            "mean_predicted": self.mean_predicted,
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

    def agents(self) -> list[str]:
        """Sorted list of distinct non-empty agent_ids with stored predictions."""
        _require_enabled()
        return sorted({p.agent_id for p in self._predictions if p.agent_id})

    def rolling_score_for_agent(
        self,
        agent_id: str,
        *,
        window_days: int = 90,
        reference_time: Optional[datetime] = None,
    ) -> BrierWindowSummary:
        """Per-agent rolling Brier score (AGT-03 sub-deliverable 3).

        Like :meth:`rolling_score` but filtered to ``agent_id``.  Default
        *window_days* is 90 per the AGT-03 plan.
        """
        _require_enabled()
        if window_days < 1:
            raise ValueError(f"window_days must be >= 1; got {window_days}")
        if not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        now = reference_time if reference_time is not None else datetime.now(UTC)
        cutoff = now - timedelta(days=window_days)
        in_window = [
            p
            for p in self._predictions
            if p.agent_id == agent_id and cutoff <= p.predicted_at <= now
        ]
        scores = [p.score for p in in_window]
        return BrierWindowSummary(
            window_days=window_days,
            n_predictions=len(scores),
            mean_brier=statistics.mean(scores) if scores else None,
            median_brier=statistics.median(scores) if scores else None,
            window_start=cutoff,
            window_end=now,
        )

    def calibration_curve_for_agent(
        self,
        agent_id: str,
        *,
        n_bins: int = 10,
    ) -> list[CalibrationBin]:
        """Calibration curve filtered to one agent (AGT-03 sub-deliverable 4).

        Like :meth:`calibration_curve` but restricted to ``agent_id``.
        """
        _require_enabled()
        if not (2 <= n_bins <= 100):
            raise ValueError(f"n_bins must be in [2, 100]; got {n_bins}")
        if not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        step = 1.0 / n_bins
        counts_yes: list[int] = [0] * n_bins
        counts_total: list[int] = [0] * n_bins
        sum_predicted: list[float] = [0.0] * n_bins
        for pred in self._predictions:
            if pred.agent_id != agent_id:
                continue
            decimal_probability = Decimal(str(pred.predicted_probability))
            idx = int((decimal_probability * n_bins).to_integral_value(rounding=ROUND_FLOOR))
            idx = min(idx, n_bins - 1)
            counts_total[idx] += 1
            sum_predicted[idx] += pred.predicted_probability
            if pred.outcome == 1:
                counts_yes[idx] += 1
        bins: list[CalibrationBin] = []
        for i in range(n_bins):
            n = counts_total[i]
            bins.append(
                CalibrationBin(
                    low=round(i * step, 10),
                    high=round((i + 1) * step, 10),
                    count=n,
                    fraction_yes=(counts_yes[i] / n) if n > 0 else None,
                    mean_predicted=(sum_predicted[i] / n) if n > 0 else None,
                )
            )
        return bins

    def calibration_curve(self, *, n_bins: int = 10) -> list[CalibrationBin]:
        """Compute a reliability diagram calibration curve over all stored predictions.

        Divides ``[0, 1]`` into ``n_bins`` equal-width brackets and, for each
        bracket, reports the empirical YES frequency and the mean predicted
        probability.  A well-calibrated agent has ``fraction_yes ≈
        mean_predicted`` in every non-empty bin.

        Predictions with ``predicted_probability == 1.0`` fall into the last
        bin (``[1 - step, 1.0]`` inclusive upper bound).

        Args:
            n_bins: Number of equal-width bins (must be between 2 and 100).

        Returns:
            A list of :class:`CalibrationBin` of length ``n_bins``, ordered
            from low to high probability.  Empty bins have ``count == 0`` and
            ``fraction_yes == mean_predicted == None``.

        Raises:
            RuntimeError: If the feature flag is not enabled.
            ValueError: If ``n_bins`` is outside ``[2, 100]``.
        """
        _require_enabled()
        if not (2 <= n_bins <= 100):
            raise ValueError(f"n_bins must be in [2, 100]; got {n_bins}")

        step = 1.0 / n_bins
        # Each bin: counts_yes, counts_total, sum_predicted
        counts_yes: list[int] = [0] * n_bins
        counts_total: list[int] = [0] * n_bins
        sum_predicted: list[float] = [0.0] * n_bins

        for pred in self._predictions:
            # Interpret the shortest decimal rendering of the probability so
            # operator-entered boundaries like 0.3 are binned as exact decimals
            # without widening the high-exclusive side of each bracket.
            decimal_probability = Decimal(str(pred.predicted_probability))
            idx = int((decimal_probability * n_bins).to_integral_value(rounding=ROUND_FLOOR))
            idx = min(idx, n_bins - 1)
            counts_total[idx] += 1
            sum_predicted[idx] += pred.predicted_probability
            if pred.outcome == 1:
                counts_yes[idx] += 1

        bins: list[CalibrationBin] = []
        for i in range(n_bins):
            n = counts_total[i]
            bins.append(
                CalibrationBin(
                    low=round(i * step, 10),
                    high=round((i + 1) * step, 10),
                    count=n,
                    fraction_yes=(counts_yes[i] / n) if n > 0 else None,
                    mean_predicted=(sum_predicted[i] / n) if n > 0 else None,
                )
            )
        return bins
