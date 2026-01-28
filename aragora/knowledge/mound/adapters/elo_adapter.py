"""
EloAdapter - DEPRECATED, use PerformanceAdapter instead.

This module re-exports from performance_adapter for backward compatibility.
All functionality has been merged into PerformanceAdapter.

Migration:
    # Before
    from aragora.knowledge.mound.adapters.elo_adapter import EloAdapter

    # After
    from aragora.knowledge.mound.adapters import PerformanceAdapter
    # or
    from aragora.knowledge.mound.adapters import EloAdapter  # alias for PerformanceAdapter
"""

import warnings

# Re-export everything from performance_adapter for backward compatibility
from .performance_adapter import (
    PerformanceAdapter,
    EloAdapter,
    RatingSearchResult,
    KMEloPattern,
    EloAdjustmentRecommendation,
    EloSyncResult,
)

# Issue deprecation warning on direct import
warnings.warn(
    "elo_adapter module is deprecated. Import EloAdapter from "
    "aragora.knowledge.mound.adapters or use PerformanceAdapter directly.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "EloAdapter",
    "PerformanceAdapter",
    "RatingSearchResult",
    "KMEloPattern",
    "EloAdjustmentRecommendation",
    "EloSyncResult",
]
