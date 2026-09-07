"""Deprecated re-export shim for the Manifold Brier resolution bridge.

The originals now live in :mod:`aragora.evaluation.manifold_brier_bridge`.
This module re-exports them so the legacy
``aragora.metrics.manifold_brier_bridge`` import path keeps working for one
release.
"""

import warnings

from aragora.evaluation.manifold_brier_bridge import (
    PendingPrediction,
    ResolutionEventProtocol,
    batch_record_resolutions,
    record_resolution,
    resolution_to_binary_outcome,
)

warnings.warn(
    "aragora.metrics.manifold_brier_bridge is deprecated; import from "
    "aragora.evaluation.manifold_brier_bridge instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "PendingPrediction",
    "ResolutionEventProtocol",
    "batch_record_resolutions",
    "record_resolution",
    "resolution_to_binary_outcome",
]
