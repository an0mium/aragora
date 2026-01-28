"""
RankingAdapter - DEPRECATED, use PerformanceAdapter instead.

This module re-exports from performance_adapter for backward compatibility.
All functionality has been merged into PerformanceAdapter.

Migration:
    # Before
    from aragora.knowledge.mound.adapters.ranking_adapter import RankingAdapter

    # After
    from aragora.knowledge.mound.adapters import PerformanceAdapter
    # or
    from aragora.knowledge.mound.adapters import RankingAdapter  # alias for PerformanceAdapter
"""

import warnings

# Re-export everything from performance_adapter for backward compatibility
from .performance_adapter import (
    PerformanceAdapter,
    RankingAdapter,
    AgentExpertise,
    ExpertiseSearchResult,
)

# Issue deprecation warning on direct import
warnings.warn(
    "ranking_adapter module is deprecated. Import RankingAdapter from "
    "aragora.knowledge.mound.adapters or use PerformanceAdapter directly.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "RankingAdapter",
    "PerformanceAdapter",
    "AgentExpertise",
    "ExpertiseSearchResult",
]
