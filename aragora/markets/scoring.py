"""Brier scoring for resolved synthetic-market positions.

Brier score is the proper scoring rule used across the AGT-* track. Lower
is better; floor 0, ceiling 1. The aggregate is computed per agent, per
domain, optionally weighted by stake size, with optional time decay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Iterable

from aragora.markets.types import (
    MarketPosition,
    ResolutionEvent,
    ResolutionOutcome,
)


def binary_outcome_value(outcome: ResolutionOutcome) -> float | None:
    """Map a ResolutionOutcome to its binary value for Brier scoring.

    Returns None for ``inconclusive`` outcomes because they do not
    contribute to calibration: an unresolvable question cannot tell us
    whether an agent was wrong.
    """
    if outcome == "yes":
        return 1.0
    if outcome == "no":
        return 0.0
    return None


def brier_score(predicted_probability: float, outcome: ResolutionOutcome) -> float | None:
    """Brier score for a single prediction against a resolved outcome.

    Returns None when the outcome is ``inconclusive`` so callers can
    exclude unscored positions from the aggregate.
    """
    realized = binary_outcome_value(outcome)
    if realized is None:
        return None
    return (predicted_probability - realized) ** 2


@dataclass(frozen=True)
class BrierBreakdown:
    """Aggregate Brier statistics for an agent over a set of positions."""

    agent_id: str
    scored_positions: int
    inconclusive_positions: int
    mean_brier: float | None
    stake_weighted_brier: float | None
    decayed_brier: float | None
    total_stake: int


def aggregate_brier(
    *,
    agent_id: str,
    positions: Iterable[MarketPosition],
    resolutions: dict[str, ResolutionEvent],
    now: datetime | None = None,
    half_life_days: float | None = 30.0,
) -> BrierBreakdown:
    """Compute mean, stake-weighted, and time-decayed Brier scores for one agent.

    - ``positions``: all positions taken by ``agent_id``; non-resolved
      positions are skipped.
    - ``resolutions``: market_id → ResolutionEvent for resolved markets.
    - ``half_life_days``: if non-None, applies exponential time decay so
      older positions weigh less; default 30 days matches the AGT-05
      90-day rolling window with mid-window centroid.

    Returns a :class:`BrierBreakdown`. ``mean_brier``,
    ``stake_weighted_brier``, and ``decayed_brier`` are None when there
    are no scored positions.
    """
    reference = now or datetime.now(tz=UTC)
    scored = 0
    inconclusive = 0
    total_stake = 0
    sum_brier = 0.0
    sum_stake_weighted = 0.0
    sum_stake_for_score = 0
    sum_decay_weighted = 0.0
    sum_decay_weight = 0.0

    for position in positions:
        if position.agent_id != agent_id:
            continue
        resolution = resolutions.get(position.market_id)
        if resolution is None:
            continue
        score = brier_score(position.probability, resolution.outcome)
        if score is None:
            inconclusive += 1
            continue
        scored += 1
        total_stake += position.stake
        sum_brier += score
        sum_stake_weighted += score * position.stake
        sum_stake_for_score += position.stake

        if half_life_days is not None and half_life_days > 0:
            try:
                resolved_at = datetime.fromisoformat(
                    resolution.resolved_at.replace("Z", "+00:00")
                ).astimezone(UTC)
            except ValueError:
                resolved_at = reference
            age_days = max((reference - resolved_at).total_seconds() / 86400.0, 0.0)
            decay_weight = math.exp(-math.log(2.0) * age_days / float(half_life_days))
        else:
            decay_weight = 1.0
        sum_decay_weighted += score * decay_weight * position.stake
        sum_decay_weight += decay_weight * position.stake

    mean_brier = (sum_brier / scored) if scored else None
    stake_weighted = (sum_stake_weighted / sum_stake_for_score) if sum_stake_for_score else None
    decayed = (sum_decay_weighted / sum_decay_weight) if sum_decay_weight else None

    return BrierBreakdown(
        agent_id=agent_id,
        scored_positions=scored,
        inconclusive_positions=inconclusive,
        mean_brier=mean_brier,
        stake_weighted_brier=stake_weighted,
        decayed_brier=decayed,
        total_stake=total_stake,
    )


def calibration_curve(
    *,
    positions: Iterable[MarketPosition],
    resolutions: dict[str, ResolutionEvent],
    bucket_count: int = 10,
) -> list[dict[str, float]]:
    """Compute a calibration curve: predicted bucket → realized frequency.

    Returns a list of bucket dictionaries with keys ``bucket_low``,
    ``bucket_high``, ``predicted_mean``, ``realized_mean``, and
    ``count``. Empty buckets are omitted.
    """
    if bucket_count < 2:
        raise ValueError("bucket_count must be >= 2")
    width = 1.0 / bucket_count
    buckets: dict[int, list[tuple[float, float]]] = {}

    for position in positions:
        resolution = resolutions.get(position.market_id)
        if resolution is None:
            continue
        realized = binary_outcome_value(resolution.outcome)
        if realized is None:
            continue
        # Use multiplication instead of division to avoid float precision
        # surprises like int(0.7 / 0.1) == 6 (should be 7).
        idx = min(int(position.probability * bucket_count), bucket_count - 1)
        buckets.setdefault(idx, []).append((position.probability, realized))

    out: list[dict[str, float]] = []
    for idx in sorted(buckets):
        entries = buckets[idx]
        predicted_mean = sum(p for p, _ in entries) / len(entries)
        realized_mean = sum(r for _, r in entries) / len(entries)
        out.append(
            {
                "bucket_low": idx * width,
                "bucket_high": (idx + 1) * width,
                "predicted_mean": predicted_mean,
                "realized_mean": realized_mean,
                "count": float(len(entries)),
            }
        )
    return out


def evaluate_position_payout(
    *,
    position: MarketPosition,
    resolution: ResolutionEvent,
) -> int:
    """Compute the per-position credit refund/forfeit under proper Brier.

    Returns a signed integer in ``[-stake, +stake]`` rounded toward zero.
    Convention: a perfectly calibrated prediction (Brier = 0) returns
    the full stake; a maximally wrong prediction (Brier = 1) forfeits
    the full stake; in between, the agent receives ``stake * (1 - 2 *
    brier)`` credits.

    Inconclusive resolutions return 0 (full refund of stake handled at
    the store layer, not encoded as a payout).
    """
    score = brier_score(position.probability, resolution.outcome)
    if score is None:
        return 0
    payout_fraction = 1.0 - 2.0 * score
    payout = int(position.stake * payout_fraction)
    return max(-position.stake, min(position.stake, payout))


__all__ = [
    "BrierBreakdown",
    "aggregate_brier",
    "binary_outcome_value",
    "brier_score",
    "calibration_curve",
    "evaluate_position_payout",
]
