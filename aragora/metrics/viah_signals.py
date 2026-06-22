"""AGT-06 sidecar signal bridge: ReputationStore → VIAH counters (AGT-05 → AGT-06).

Supplies the two integer sidecar parameters that :func:`aragora.metrics.viah.compute_viah`
accepts as defaults-to-zero until AGT-05 settlement is live:

- :func:`count_crux_resolutions_correct` — SD-2: cruxes correctly resolved (positive delta)
- :func:`count_predictions_above_brier_threshold` — SD-3: prediction-market predictions
  at or below the Brier quality threshold (lower Brier = better calibration)

Both return 0 when ``ARAGORA_VIAH_TREND_ENABLED`` is unset.
Advances issue #6067 (AGT-06 SD-2, SD-3).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from aragora.reputation.types import DOMAIN_CRUX_RESOLUTION, DOMAIN_PREDICTION_MARKET

if TYPE_CHECKING:
    from aragora.reputation.store import ReputationStore

_VIAH_FLAG = "ARAGORA_VIAH_TREND_ENABLED"
_TRUTHY = frozenset({"1", "true", "yes", "on"})
DEFAULT_BRIER_THRESHOLD = 0.25


def _signals_enabled() -> bool:
    return os.environ.get(_VIAH_FLAG, "").strip().lower() in _TRUTHY


def _dt(iso: str) -> datetime | None:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(UTC)
    except (ValueError, TypeError):
        return None


def count_crux_resolutions_correct(
    store: "ReputationStore",
    *,
    window_hours: float = 168.0,
    now: datetime | None = None,
) -> int:
    """Count ``DOMAIN_CRUX_RESOLUTION`` deltas with positive payout within *window_hours*.

    A positive delta means the agent's crux position matched the eventual resolution
    outcome (AGT-06 SD-2). Returns 0 when ``ARAGORA_VIAH_TREND_ENABLED`` is not set.
    """
    if not _signals_enabled():
        return 0
    ref = (now or datetime.now(UTC)).astimezone(UTC)
    cutoff = ref - timedelta(hours=window_hours)
    count = 0
    for agent_id in store.agent_ids():
        for d in store.deltas_for(agent_id):
            if d.domain != DOMAIN_CRUX_RESOLUTION or d.delta <= 0:
                continue
            dt = _dt(d.applied_at)
            if dt is not None and dt >= cutoff:
                count += 1
    return count


def count_predictions_above_brier_threshold(
    store: "ReputationStore",
    *,
    window_hours: float = 168.0,
    brier_threshold: float = DEFAULT_BRIER_THRESHOLD,
    now: datetime | None = None,
) -> int:
    """Count ``DOMAIN_PREDICTION_MARKET`` predictions with Brier ≤ *brier_threshold* in window.

    Only ``brier_proper``-scored deltas store ``reason["brier"]``; deltas without that
    key are skipped to avoid double-counting binary-scored positions.
    Returns 0 when ``ARAGORA_VIAH_TREND_ENABLED`` is not set.
    """
    if not _signals_enabled():
        return 0
    if not (0.0 <= brier_threshold <= 1.0):
        raise ValueError(f"brier_threshold must be in [0, 1]; got {brier_threshold!r}")
    ref = (now or datetime.now(UTC)).astimezone(UTC)
    cutoff = ref - timedelta(hours=window_hours)
    count = 0
    for agent_id in store.agent_ids():
        for d in store.deltas_for(agent_id):
            if d.domain != DOMAIN_PREDICTION_MARKET:
                continue
            brier = d.reason.get("brier")
            if brier is None or brier > brier_threshold:
                continue
            dt = _dt(d.applied_at)
            if dt is not None and dt >= cutoff:
                count += 1
    return count


__all__ = [
    "DEFAULT_BRIER_THRESHOLD",
    "count_crux_resolutions_correct",
    "count_predictions_above_brier_threshold",
]
