"""External prediction-market venue adapters.

Currently exposes the AGT-03 Manifold Markets read-only adapter. See
``docs/plans/2026-04-17-prediction-market-validation.md`` for the
venue stack rationale and graduation gates.
"""

from __future__ import annotations

from aragora.connectors.prediction_markets.manifold import (
    MANIFOLD_API_BASE,
    ManifoldAdapter,
    ManifoldError,
    ManifoldMarket,
    ManifoldResolution,
    manifold_to_market_resolution,
)

__all__ = [
    "MANIFOLD_API_BASE",
    "ManifoldAdapter",
    "ManifoldError",
    "ManifoldMarket",
    "ManifoldResolution",
    "manifold_to_market_resolution",
]
