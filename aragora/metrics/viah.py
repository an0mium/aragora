"""Deprecated re-export shim for VIAH.

The originals now live in :mod:`aragora.evaluation.viah`. This module
re-exports them so the legacy ``aragora.metrics.viah`` import path keeps
working for one release.
"""

import warnings

from aragora.evaluation.viah import (
    DEFAULT_BRIER_THRESHOLD,
    DEFAULT_CRUX_WEIGHT,
    DEFAULT_FAILED_CLAIM_WEIGHT,
    DEFAULT_PREDICTION_WEIGHT,
    DEFAULT_PR_WEIGHT,
    DEFAULT_RESCUE_WEIGHT,
    VIAH_SNAPSHOT_ENTRY_TYPE,
    VIAH_TREND_FLAG,
    ViahCoefficients,
    ViahReport,
    ViahTrend,
    ViahTrendPoint,
    compute_viah,
    persist_viah_snapshot,
    read_viah_snapshots,
    rolling_viah_trend,
    viah_score,
    viah_trend_enabled,
)

warnings.warn(
    "aragora.metrics.viah is deprecated; import from aragora.evaluation.viah instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "DEFAULT_BRIER_THRESHOLD",
    "DEFAULT_CRUX_WEIGHT",
    "DEFAULT_FAILED_CLAIM_WEIGHT",
    "DEFAULT_PREDICTION_WEIGHT",
    "DEFAULT_PR_WEIGHT",
    "DEFAULT_RESCUE_WEIGHT",
    "VIAH_SNAPSHOT_ENTRY_TYPE",
    "VIAH_TREND_FLAG",
    "ViahCoefficients",
    "ViahReport",
    "ViahTrend",
    "ViahTrendPoint",
    "compute_viah",
    "persist_viah_snapshot",
    "read_viah_snapshots",
    "rolling_viah_trend",
    "viah_score",
    "viah_trend_enabled",
]
