"""Deprecated re-export shim for VIAH signal helpers.

The originals now live in :mod:`aragora.evaluation.viah_signals`. This
module re-exports them so the legacy ``aragora.metrics.viah_signals``
import path keeps working for one release.
"""

import warnings

from aragora.evaluation.viah_signals import (
    DEFAULT_BRIER_THRESHOLD,
    count_crux_resolutions_correct,
    count_predictions_above_brier_threshold,
)

warnings.warn(
    "aragora.metrics.viah_signals is deprecated; import from "
    "aragora.evaluation.viah_signals instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "DEFAULT_BRIER_THRESHOLD",
    "count_crux_resolutions_correct",
    "count_predictions_above_brier_threshold",
]
