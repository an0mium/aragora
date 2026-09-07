"""Deprecated re-export shim for the Manifold Brier scorer.

The originals now live in :mod:`aragora.evaluation.manifold_brier`. This
module re-exports them so the legacy ``aragora.metrics.manifold_brier``
import path keeps working for one release.
"""

import warnings

from aragora.evaluation.manifold_brier import (
    BrierWindowSummary,
    CalibrationBin,
    ManifoldBrierScorer,
    ManifoldPrediction,
    brier_score,
    manifold_brier_enabled,
)

warnings.warn(
    "aragora.metrics.manifold_brier is deprecated; import from "
    "aragora.evaluation.manifold_brier instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "BrierWindowSummary",
    "CalibrationBin",
    "ManifoldBrierScorer",
    "ManifoldPrediction",
    "brier_score",
    "manifold_brier_enabled",
]
