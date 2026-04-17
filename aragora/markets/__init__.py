"""Synthetic GitHub prediction markets — AGT-04 vision-layer planning track.

This module is a planning-truth implementation of the synthetic prediction
market substrate described in
``docs/plans/2026-04-17-prediction-market-validation.md``. It supplies the
``AGT-05`` skin-in-the-game reputation flow with high-volume internal
calibration data without requiring an external venue dependency.

The module is gated behind ``ARAGORA_SYNTHETIC_MARKETS_ENABLED``. Importing
it has no live effect; explicit construction of a :class:`MarketStore`
or :func:`enable_synthetic_markets` is required.

Question shapes supported by the GitHub resolver:

- ``pr_merge``: will a given PR merge within ``resolution_window_days``?
- ``issue_close``: will a given issue close within ``resolution_window_days``?
- ``ci_pass``: will a given branch's first scheduled CI run pass all required checks?

Resolution is deterministic: the GitHub state at expiry decides the outcome.
"""

from __future__ import annotations

import os

from aragora.markets.resolver import GitHubMarketResolver, ResolutionError, resolve_market
from aragora.markets.scoring import (
    BrierBreakdown,
    aggregate_brier,
    binary_outcome_value,
    brier_score,
)
from aragora.markets.store import MarketStore
from aragora.markets.types import (
    Market,
    MarketPosition,
    QuestionKind,
    ResolutionEvent,
    ResolutionOutcome,
)

__all__ = [
    "BrierBreakdown",
    "GitHubMarketResolver",
    "Market",
    "MarketPosition",
    "MarketStore",
    "QuestionKind",
    "ResolutionError",
    "ResolutionEvent",
    "ResolutionOutcome",
    "aggregate_brier",
    "binary_outcome_value",
    "brier_score",
    "enable_synthetic_markets",
    "resolve_market",
    "synthetic_markets_enabled",
]


def synthetic_markets_enabled() -> bool:
    """Return True if the AGT-04 synthetic-markets surface is enabled.

    Reads ``ARAGORA_SYNTHETIC_MARKETS_ENABLED`` from the process environment.
    Default is False; the markets module is dormant until explicitly enabled.
    """
    raw = str(os.environ.get("ARAGORA_SYNTHETIC_MARKETS_ENABLED") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def enable_synthetic_markets() -> None:
    """Enable the synthetic-markets surface for the current process.

    Sets ``ARAGORA_SYNTHETIC_MARKETS_ENABLED=1`` so subsequent calls to
    :func:`synthetic_markets_enabled` return True. This does not start any
    background loops or write to disk; resolution and prediction submission
    remain explicit calls.
    """
    os.environ["ARAGORA_SYNTHETIC_MARKETS_ENABLED"] = "1"
