"""
Performance adapter subpackage for agent performance metrics.

This package provides the PerformanceAdapter (unified ELO + Ranking adapter)
split into focused submodules:

- models: Data classes (KMEloPattern, AgentExpertise, etc.)
- elo_storage: ELO rating, match, calibration storage and retrieval
- expertise: Domain expertise tracking with caching
- reverse_flow: KM-to-ELO pattern analysis and adjustments
- km_persistence: Knowledge Mound sync/load
- search: Search, fusion, and semantic search implementations
- adapter: Main PerformanceAdapter class composing all submodules

All public APIs are re-exported here for convenient access:

    from aragora.knowledge.mound.adapters.performance import PerformanceAdapter
"""

from aragora.knowledge.mound.adapters.performance.models import (
    AgentExpertise,
    EloAdjustmentRecommendation,
    EloSyncResult,
    ExpertiseSearchResult,
    KMEloPattern,
    RatingSearchResult,
)
from aragora.knowledge.mound.adapters.performance.adapter import (
    EloAdapter,
    EventCallback,
    PerformanceAdapter,
    RankingAdapter,
)

__all__ = [
    # Main adapter
    "PerformanceAdapter",
    # Backwards compatibility
    "EloAdapter",
    "RankingAdapter",
    # Type alias
    "EventCallback",
    # Dataclasses from EloAdapter
    "KMEloPattern",
    "EloAdjustmentRecommendation",
    "EloSyncResult",
    "RatingSearchResult",
    # Dataclasses from RankingAdapter
    "AgentExpertise",
    "ExpertiseSearchResult",
]
